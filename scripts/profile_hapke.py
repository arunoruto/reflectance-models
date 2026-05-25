"""Profile the Hapke AMSA forward and inverse passes using JAX's built-in profiler.

Generates a trace file viewable in Perfetto (https://ui.perfetto.dev).

Usage:
    uv run python scripts/profile_hapke.py
    # -> prints trace file path -> open in https://ui.perfetto.dev
"""

import glob
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from astropy.io import fits

from refmod.dtm_helper import dtm2grad
from refmod.hapke import amsa, dhg_legendre_coefficients
from refmod.hapke.inverse import invert_amsa


def load_hopper_data(pixel_subset: int | None = None):
    f = fits.open("test/data/hopper_amsa.fits")

    result = f["result"].data.astype(np.float64)
    i_deg = f["result"].header["i"]
    e_deg = f["result"].header["e"]
    b = f["result"].header["b"]
    c = f["result"].header["c"]
    hs = f["result"].header["hs"]
    bs0 = f["result"].header["bs0"]
    tb = f["result"].header["tb"]
    hc = f["result"].header["hc"]
    bc0 = f["result"].header["bc0"]
    albedo = f["albedo"].data.astype(np.float64)
    dtm = f["dtm"].data.astype(np.float64)
    res = f["dtm"].header["res"]
    n = dtm2grad(dtm, res, normalize=False)

    u, v = result.shape
    i_rad = np.deg2rad(i_deg)
    e_rad = np.deg2rad(e_deg)

    if pixel_subset is not None:
        r = pixel_subset // 2
        uc = u // 2 + np.arange(-r, r)
        vc = v // 2 + np.arange(-r, r)
        albedo = albedo[uc, :][:, vc]
        n = n[uc, :, :][:, vc, :]

    b_n = dhg_legendre_coefficients(b, c)

    w_flat = albedo.reshape(-1)
    n_flat = n.reshape(-1, 3)

    w_jax = jnp.asarray(w_flat)
    i_jax = jnp.tile(
        jnp.array([np.sin(i_rad), 0.0, np.cos(i_rad)]), (w_flat.shape[0], 1)
    )
    e_jax = jnp.tile(
        jnp.array([np.sin(e_rad), 0.0, np.cos(e_rad)]), (w_flat.shape[0], 1)
    )
    n_jax = jnp.asarray(n_flat)

    params = dict(
        roughness=tb,
        h_sh=hs,
        b0_sh=bs0,
        h_cb=hc,
        b0_cb=bc0,
    )

    f.close()
    return w_jax, b_n, i_jax, e_jax, n_jax, params


def main():
    print("Loading data...")
    print(f"JAX cache: {os.environ.get('JAX_COMPILATION_CACHE_DIR')}")
    w, b_n, i, e, n, params = load_hopper_data(pixel_subset=None)
    n_pixels = w.shape[0]
    print(f"  {n_pixels} pixels")

    # Warmup to compile everything before profiling
    print("Warmup (compiling)...")
    refl = amsa(w, b_n, i, e, n, **params).block_until_ready()
    _ = invert_amsa(refl, b_n, i, e, n, **params).block_until_ready()
    print("  done")

    # Profiled run
    print("Profiling...")
    trace_dir = "/tmp/jax-profile-hapke"

    options = jax.profiler.ProfileOptions()
    options.host_tracer_level = 2
    options.device_tracer_level = 1
    options.python_tracer_level = 0

    with jax.profiler.trace(trace_dir, profiler_options=options):
        t0 = time.perf_counter()

        # Forward
        t_fwd = time.perf_counter()
        refl = amsa(w, b_n, i, e, n, **params).block_until_ready()
        fwd_s = time.perf_counter() - t_fwd

        # Inverse
        t_inv = time.perf_counter()
        w_rec = invert_amsa(refl, b_n, i, e, n, **params).block_until_ready()
        inv_s = time.perf_counter() - t_inv

        total_s = time.perf_counter() - t0

    err = float(jnp.max(jnp.abs(w_rec - w)))
    print(f"  Forward:  {fwd_s:.3f}s")
    print(f"  Inverse:  {inv_s:.3f}s")
    print(f"  Total:    {total_s:.3f}s")
    print(f"  Max err:  {err:.2e}")

    traces = glob.glob(f"{trace_dir}/plugins/profile/*/*.trace.json.gz")
    if traces:
        print(f"\nTrace: {traces[-1]}")
        print("Open https://ui.perfetto.dev → drag-drop this file")
    else:
        print(f"\nNo trace found in {trace_dir}/ — check JAX profiler setup.")


if __name__ == "__main__":
    main()
