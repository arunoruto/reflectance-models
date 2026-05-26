from dataclasses import dataclass

import jax
import jax.numpy as jnp
import psutil

from .amsa import (
    _fast_refl_amsa_and_grad_batched,
    _fast_refl_amsa_batched,
    precompute_amsa,
)

_BYTES_PER_PIXEL = 500


@dataclass(frozen=True)
class AmsaInversionState:
    r"""Reusable geometry-dependent state for AMSA albedo inversion.

    Instances are created with :func:`prepare_amsa_inversion` and consumed by
    :func:`invert_amsa_precomputed`. The state stores one or more precomputed
    chunks so repeated inversions with fixed geometry can skip the expensive
    geometry-only setup.
    """

    chunks: tuple[tuple[int, int, dict], ...]
    n_pixels: int
    chunk_size: int


def _adaptive_chunk_size(n_pixels: int) -> int:
    r"""Determine a safe per-chunk pixel count based on available memory.

    When a GPU is available, 90 % of the VRAM capacity is used to estimate
    the maximum chunk size.  On CPU-only systems, 75 % of the free RAM
    reported by ``psutil`` is used instead.  The returned chunk size is
    capped at *n_pixels*.

    Parameters
    ----------
    n_pixels : int
        Total number of pixels to be inverted.

    Returns
    -------
    int
        Maximum number of pixels that can be inverted in a single chunk
        without exhausting memory.
    """
    if n_pixels <= 0:
        return 0

    def _safe_chunk_count(usable_bytes: float) -> int:
        return min(max(1, int(usable_bytes / _BYTES_PER_PIXEL)), n_pixels)

    try:
        device = jax.local_devices()[0]
        if device.platform == "gpu":
            memory_stats = getattr(device, "memory_stats", None)
            mem = memory_stats() if memory_stats is not None else None
            bytes_limit = mem.get("bytes_limit") if mem else None
            if bytes_limit:
                return _safe_chunk_count(bytes_limit * 0.9)
    except Exception:
        pass

    usable = psutil.virtual_memory().available * 0.75
    return _safe_chunk_count(usable)


def _tanh_to_w(x: jax.Array) -> jax.Array:
    return 0.5 * (jnp.tanh(x) + 1.0)


def _w_to_tanh(w: jax.Array) -> jax.Array:
    return jnp.arctanh(jnp.clip(2.0 * w - 1.0, -0.999999, 0.999999))


def _d_w_d_x(x: jax.Array) -> jax.Array:
    return 0.5 * (1.0 - jnp.tanh(x) ** 2)


def _invert_chunk(
    refl_obs: jax.Array,
    pre: dict,
    w0: jax.Array,
    max_steps: int = 40,
) -> jax.Array:
    r"""Levenberg-Marquardt inversion for a single pixel chunk.

    Iteratively minimises the squared residual between the observed
    reflectance and the AMSA forward model.  The single scattering albedo
    *w* is transformed via :math:`x \mapsto \tanh` to keep it in
    :math:`(0, 1)` during the optimisation.

    Uses precomputed *w*-independent state (radiance-transfer quantities,
    shadowing corrections, etc.) to accelerate each iteration.  Only pixels
    flagged as active (finite observed reflectance) are updated; converged
    pixels are automatically masked out.

    Parameters
    ----------
    refl_obs : jax.Array
        Observed reflectance.  Shape ``(n_pixels,)``.
    pre : dict
        Precomputed geometry-dependent terms from
        :func:`~.amsa.precompute_amsa`.
    w0 : jax.Array
        Initial guess for the single scattering albedo.  Shape
        ``(n_pixels,)``.
    max_steps : int, optional
        Maximum number of Levenberg-Marquardt iterations.  Default is 40.

    Returns
    -------
    jax.Array
        Recovered single scattering albedo.  Shape ``(n_pixels,)``.
    """
    n_pixels = w0.shape[0]
    x = jnp.where(
        jnp.isfinite(refl_obs),
        _w_to_tanh(w0),
        0.0,
    )
    active = jnp.isfinite(refl_obs)
    lam = jnp.full((n_pixels,), 1e-3)

    def _step(carry):
        x, lam, active, step_count = carry
        w = _tanh_to_w(x)

        refl_model, j_w = _fast_refl_amsa_and_grad_batched(w, pre)
        r = refl_model - refl_obs
        r = jnp.where(active, r, 0.0)

        j_x = j_w * _d_w_d_x(x)
        delta = j_x * r / (j_x**2 + lam)
        x_new = x - delta
        w_new = _tanh_to_w(x_new)

        refl_new = _fast_refl_amsa_batched(w_new, pre)
        r_new = refl_new - refl_obs
        r_new = jnp.where(active, r_new, 0.0)

        e_old = r**2
        e_new = r_new**2

        gain = e_old - e_new
        predicted = delta * (lam * delta + j_x * r)
        gain_ratio = jnp.where(
            jnp.abs(predicted) > 1e-30,
            gain / predicted,
            0.0,
        )

        improved = active & (gain_ratio > 0.0) & (e_new < e_old)

        x = jnp.where(improved, x_new, x)
        lam = jnp.where(
            improved,
            lam * jnp.maximum(1.0 / 3.0, 1.0 - (2.0 * gain_ratio - 1.0) ** 3),
            lam * 2.0,
        )
        lam = jnp.clip(lam, 1e-7, 1e7)

        grad_max = jnp.max(jnp.where(active, jnp.abs(j_x * r), 0.0))
        active = active & (jnp.abs(j_x * r) > 1e-10)

        return x, lam, active, step_count + 1

    def _cond(carry):
        _x, _lam, active, step_count = carry
        return jnp.any(active) & (step_count < max_steps)

    x, _, _, _ = jax.lax.while_loop(
        _cond,
        _step,
        (x, lam, active, 0),
    )

    return _tanh_to_w(x)


_invert_chunk_jit = jax.jit(_invert_chunk, static_argnames=("max_steps",))


def prepare_amsa_inversion(
    b_n: jax.Array,
    i: jax.Array,
    e: jax.Array,
    n: jax.Array,
    roughness: float = 0.0,
    h_sh: float = 0.0,
    b0_sh: float = 0.0,
    h_cb: float = 0.0,
    b0_cb: float = 0.0,
    chunk_size: int | None = None,
) -> AmsaInversionState:
    r"""Prepare reusable geometry-dependent state for AMSA inversion.

    Use this when the geometry and Hapke phase/opposition parameters stay
    fixed while multiple observed reflectance arrays are inverted.

    Parameters
    ----------
    b_n : jax.Array
        Legendre polynomial coefficients for the single-scattering phase
        function. Shape ``(n_coeffs,)``.
    i : jax.Array
        Incidence direction vectors. Shape ``(n_pixels, 3)``.
    e : jax.Array
        Emission direction vectors. Shape ``(n_pixels, 3)``.
    n : jax.Array
        Surface normal vectors. Shape ``(n_pixels, 3)``.
    roughness, h_sh, b0_sh, h_cb, b0_cb : float, optional
        AMSA geometry/opposition parameters matching :func:`invert_amsa`.
    chunk_size : int or None, optional
        Number of pixels to precompute per chunk. When ``None``, determined
        automatically from available memory.

    Returns
    -------
    AmsaInversionState
        Reusable precomputed state for :func:`invert_amsa_precomputed`.
    """
    n_pixels = i.shape[0]

    if chunk_size is None:
        chunk_size = _adaptive_chunk_size(n_pixels)

    if n_pixels <= chunk_size:
        pre = precompute_amsa(b_n, i, e, n, roughness, h_sh, b0_sh, h_cb, b0_cb)
        return AmsaInversionState(((0, n_pixels, pre),), n_pixels, chunk_size)

    chunks = []
    for start in range(0, n_pixels, chunk_size):
        end = min(start + chunk_size, n_pixels)
        sl = slice(start, end)
        pre = precompute_amsa(
            b_n,
            i[sl],
            e[sl],
            n[sl],
            roughness,
            h_sh,
            b0_sh,
            h_cb,
            b0_cb,
        )
        chunks.append((start, end, pre))

    return AmsaInversionState(tuple(chunks), n_pixels, chunk_size)


def invert_amsa_precomputed(
    refl_obs: jax.Array,
    state: AmsaInversionState,
    w0: jax.Array | None = None,
    max_steps: int = 40,
) -> jax.Array:
    r"""Invert AMSA reflectance using precomputed geometry state.

    This is the high-throughput path for repeated inversions with fixed
    geometry. For one-off calls, use :func:`invert_amsa`.

    Parameters
    ----------
    refl_obs : jax.Array
        Observed reflectance. Shape ``(n_pixels,)``.
    state : AmsaInversionState
        Prepared state from :func:`prepare_amsa_inversion`.
    w0 : jax.Array or None, optional
        Initial guess for the single scattering albedo. Shape
        ``(n_pixels,)``. When ``None``, defaults to 1/3 everywhere.
    max_steps : int, optional
        Maximum number of Levenberg-Marquardt iterations per chunk.

    Returns
    -------
    jax.Array
        Recovered single scattering albedo. Shape ``(n_pixels,)``.
    """
    if refl_obs.shape[0] != state.n_pixels:
        raise ValueError(
            "reflectance length does not match prepared inversion state: "
            f"{refl_obs.shape[0]} != {state.n_pixels}"
        )

    if w0 is None:
        w0_local = jnp.full_like(refl_obs, 1.0 / 3.0)
    else:
        if w0.shape[0] != state.n_pixels:
            raise ValueError(
                "initial guess length does not match prepared inversion state: "
                f"{w0.shape[0]} != {state.n_pixels}"
            )
        w0_local = w0

    if len(state.chunks) == 1:
        start, end, pre = state.chunks[0]
        if start == 0 and end == state.n_pixels:
            return _invert_chunk_jit(refl_obs, pre, w0_local, max_steps)

    w_sol = jnp.empty_like(refl_obs)
    for start, end, pre in state.chunks:
        sl = slice(start, end)
        w_sol = w_sol.at[sl].set(
            _invert_chunk_jit(refl_obs[sl], pre, w0_local[sl], max_steps)
        )

    return w_sol


def invert_amsa(
    refl_obs: jax.Array,
    b_n: jax.Array,
    i: jax.Array,
    e: jax.Array,
    n: jax.Array,
    roughness: float = 0.0,
    h_sh: float = 0.0,
    b0_sh: float = 0.0,
    h_cb: float = 0.0,
    b0_cb: float = 0.0,
    w0: jax.Array | None = None,
    max_steps: int = 40,
    chunk_size: int | None = None,
) -> jax.Array:
    r"""Invert the AMSA model to recover single scattering albedo from
    observed reflectance.

    Solves the inverse problem for the Hapke AMSA forward model using a
    chunked, batched Levenberg-Marquardt algorithm.  The single scattering
    albedo :math:`w` is constrained to :math:`(0, 1)` via a
    :math:`\tanh`-based variable transformation.  Convergence is tracked
    per pixel through an active-set mask, and large datasets are
    automatically split into memory-safe chunks.

    Parameters
    ----------
    refl_obs : jax.Array
        Observed reflectance.  Shape ``(n_pixels,)``.
    b_n : jax.Array
        Legendre polynomial coefficients for the single-scattering phase
        function.  Shape ``(n_coeffs,)``.
    i : jax.Array
        Incidence direction vectors.  Shape ``(n_pixels, 3)``.
    e : jax.Array
        Emission direction vectors.  Shape ``(n_pixels, 3)``.
    n : jax.Array
        Surface normal vectors.  Shape ``(n_pixels, 3)``.
    roughness : float, optional
        Macroscopic roughness angle in radians, :math:`\bar{\theta}`.
        Default is 0.0.
    h_sh : float, optional
        SHOE angular width parameter.  Default is 0.0.
    b0_sh : float, optional
        SHOE amplitude.  Default is 0.0.
    h_cb : float, optional
        CBOE angular width parameter.  Default is 0.0.
    b0_cb : float, optional
        CBOE amplitude.  Default is 0.0.
    w0 : jax.Array or None, optional
        Initial guess for the single scattering albedo.  Shape
        ``(n_pixels,)``.  When ``None``, defaults to 1/3 everywhere.
    max_steps : int, optional
        Maximum number of Levenberg-Marquardt iterations per chunk.
        Default is 40.
    chunk_size : int or None, optional
        Number of pixels to process per chunk.  When ``None``, determined
        automatically by :func:`_adaptive_chunk_size`.

    Returns
    -------
    jax.Array
        Recovered single scattering albedo.  Shape ``(n_pixels,)``.
    """
    state = prepare_amsa_inversion(
        b_n,
        i,
        e,
        n,
        roughness,
        h_sh,
        b0_sh,
        h_cb,
        b0_cb,
        chunk_size,
    )
    return invert_amsa_precomputed(refl_obs, state, w0, max_steps)
