r"""Ahead-of-time compilation warmup and persistent compilation caching.

XLA compilation of the refmod kernels can be expensive on GPU — pathologically
so when the device memory is nearly full, because XLA's autotuner allocates
real scratch buffers while timing kernel candidates. Two mitigations:

1. :func:`enable_compilation_cache` — persist compiled kernels to disk so any
   compile is paid at most once per machine.
2. :func:`warmup` — trigger the compiles up front (ideally right after import,
   while the GPU is still empty), so they never interleave with real work.

The Levenberg-Marquardt inversion kernel is specialized on the *pixel count*
and ``max_steps`` only; all Hapke parameters (roughness, opposition effects,
phase-function coefficients) enter through runtime arrays. A single warmup per
chunk shape therefore covers every parameter combination. Chunk sizes produced
by :func:`refmod.hapke.prepare_amsa_inversion` are powers of two, so warming
the expected chunk size (or a small ladder of sizes) is enough.

Usage::

    import refmod.warmup
    refmod.warmup.enable_compilation_cache()   # before any jit executes
    refmod.warmup.warmup(n_pixels=(1 << 20,))  # while the GPU is still empty

or from the command line::

    python -m refmod.warmup --pixels 1048576 --max-steps 40
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from pathlib import Path

import jax
import jax.numpy as jnp

__all__ = ["enable_compilation_cache", "warmup"]


def _default_cache_dir() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "refmod" / "jax"


def enable_compilation_cache(
    cache_dir: str | os.PathLike | None = None,
    min_compile_time_secs: float = 0.0,
) -> Path:
    r"""Enable JAX's persistent compilation cache (if not already enabled).

    If a cache directory is already configured — e.g. via the
    ``JAX_COMPILATION_CACHE_DIR`` environment variable, as this project's
    devenv does — and no explicit *cache_dir* is passed, the existing
    configuration is kept and returned. Passing an explicit *cache_dir*
    overrides it (resetting an already-initialized cache).

    Parameters
    ----------
    cache_dir : path-like or None, optional
        Cache location. Defaults to the already-configured directory, or
        ``$XDG_CACHE_HOME/refmod/jax`` (``~/.cache/refmod/jax``) when none
        is configured.
    min_compile_time_secs : float, optional
        Only compiles slower than this are persisted. Default 0.0 (persist
        everything), so even the ~2 s CPU compiles are amortized.

    Returns
    -------
    pathlib.Path
        The cache directory in use.
    """
    configured = getattr(jax.config, "jax_compilation_cache_dir", None)
    if cache_dir is None and configured:
        return Path(configured)

    path = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    path.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(path))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", min_compile_time_secs)
    if configured and str(path) != configured:
        # The cache backend binds its directory on first use; reset so the
        # new location actually takes effect.
        try:
            from jax.experimental.compilation_cache import compilation_cache

            compilation_cache.reset_cache()
        except Exception:
            pass
    return path


def warmup(
    n_pixels: Iterable[int] = (1 << 20,),
    max_steps: int = 40,
    cache: bool = True,
    verbose: bool = True,
) -> None:
    r"""Compile the AMSA forward/gradient/inversion kernels ahead of time.

    Run this right after importing, before loading data onto the device:
    compiling while device memory is nearly full can slow XLA's autotuner
    down by orders of magnitude.

    Parameters
    ----------
    n_pixels : iterable of int, optional
        Pixel counts (chunk shapes) to compile for. Use the chunk size your
        inversions will run at; :func:`refmod.hapke.prepare_amsa_inversion`
        quantizes adaptive chunk sizes to powers of two. Default: ``2**20``.
    max_steps : int, optional
        ``max_steps`` value the inversion will be called with (part of the
        compiled kernel). Default 40, matching :func:`refmod.hapke.invert_amsa`.
    cache : bool, optional
        Also enable the persistent compilation cache (default True), so the
        warmup outlives this process.
    verbose : bool, optional
        Print per-kernel compile times.
    """
    from refmod.hapke import dhg_legendre_coefficients
    from refmod.hapke.amsa import (
        _fast_refl_amsa_and_grad_batched,
        _fast_refl_amsa_batched,
        precompute_amsa,
    )
    from refmod.hapke.inverse import _invert_chunk_jit

    if cache:
        cache_path = enable_compilation_cache()
        if verbose:
            print(f"[refmod.warmup] compilation cache: {cache_path}")

    b_n = dhg_legendre_coefficients(0.21, 0.7, 15)

    for count in n_pixels:
        count = int(count)
        up = jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0]), (count, 3))
        oblique = jnp.broadcast_to(
            jnp.asarray([0.0, 0.5, 0.8660254037844387]), (count, 3)
        )
        w = jnp.full((count,), 1.0 / 3.0)

        t0 = time.perf_counter()
        # Non-trivial parameters so every model branch is materialized; the
        # compiled kernels are reused for any parameter values.
        pre = precompute_amsa(b_n, oblique, up, up, 0.1, 0.05, 0.5, 0.05, 0.5)
        refl, _ = _fast_refl_amsa_and_grad_batched(w, pre)
        _fast_refl_amsa_batched(w, pre)
        sol = _invert_chunk_jit(refl, pre, w, max_steps)
        jax.block_until_ready(sol)
        if verbose:
            print(
                f"[refmod.warmup] n_pixels={count:>9,} max_steps={max_steps}: "
                f"{time.perf_counter() - t0:6.1f} s"
            )


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Warm up refmod's compiled kernels and persist them to the JAX compilation cache."
    )
    parser.add_argument(
        "--pixels",
        type=int,
        nargs="+",
        default=[1 << 20],
        help="chunk sizes (pixel counts) to compile for (default: 1048576)",
    )
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--cache-dir", default=None, help="compilation cache directory")
    parser.add_argument("--no-cache", action="store_true", help="skip the persistent cache")
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", True)
    if not args.no_cache:
        print(f"[refmod.warmup] compilation cache: {enable_compilation_cache(args.cache_dir)}")
    warmup(n_pixels=args.pixels, max_steps=args.max_steps, cache=False)


if __name__ == "__main__":
    _main()
