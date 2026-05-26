import jax
import jax.numpy as jnp

from ._core import (
    cos_angle,
    h_function,
    legendre_eval,
    normalize,
    roughness_correction,
)


def _refl_imsa_scalar(
    w: jax.Array,
    b_n: jax.Array,
    i: jax.Array,
    e: jax.Array,
    n: jax.Array,
    roughness: float,
) -> jax.Array:
    r"""IMSA reflectance for a single pixel.

    Simplest Hapke model: isotropic multiple scattering with a Legendre
    polynomial phase function for single scattering.  No opposition effects
    (SHOE, CBOE) are included.

    Parameters
    ----------
    w : jax.Array
        Single scattering albedo (scalar).
    b_n : jax.Array
        Legendre polynomial coefficients for the single-scattering phase
        function.  Shape ``(n_coeffs,)``.
    i : jax.Array
        Incidence (illumination) direction vector.  Shape ``(3,)``.
    e : jax.Array
        Emission (viewing) direction vector.  Shape ``(3,)``.
    n : jax.Array
        Surface normal vector.  Shape ``(3,)``.
    roughness : float
        Macroscopic roughness angle in radians, :math:`\bar{\theta}`.

    Returns
    -------
    jax.Array
        Reflectance (scalar).  Pixels with the source or detector behind the
        local horizon are set to NaN.

    References
    ----------
    .. bibliography::
       :filter: False

       Hapke-2012
    """
    i = normalize(i)
    e = normalize(e)
    n = normalize(n)
    s, mu0, mu = roughness_correction(roughness, i, e, n)
    cos_alpha = cos_angle(i, e)
    p = legendre_eval(cos_alpha, b_n)
    h_mu0 = h_function(mu0, w)
    h_mu = h_function(mu, w)
    m = h_mu0 * h_mu - 1.0
    albedo_indep = mu0 / (mu0 + mu) * s / (4.0 * jnp.pi)
    refl = albedo_indep * w * (p + m)
    return jnp.where((mu <= 0.0) | (mu0 <= 0.0), jnp.nan, refl)


_imsa_batched = jax.vmap(
    _refl_imsa_scalar,
    in_axes=(0, None, 0, 0, 0, None),
)


def imsa(
    w: jax.Array,
    b_n: jax.Array,
    i: jax.Array,
    e: jax.Array,
    n: jax.Array,
    roughness: float = 0.0,
) -> jax.Array:
    r"""Batched IMSA reflectance.

    Vectorised wrapper around :func:`_refl_imsa_scalar` that evaluates the
    isotropic multiple-scattering Hapke model for an arbitrary number of
    pixels sharing the same Legendre coefficients and roughness.

    Parameters
    ----------
    w : jax.Array
        Single scattering albedo per pixel.  Shape ``(n_pixels,)``.
    b_n : jax.Array
        Legendre polynomial coefficients.  Shape ``(n_coeffs,)``.
    i : jax.Array
        Incidence direction vectors.  Shape ``(n_pixels, 3)``.
    e : jax.Array
        Emission direction vectors.  Shape ``(n_pixels, 3)``.
    n : jax.Array
        Surface normal vectors.  Shape ``(n_pixels, 3)``.
    roughness : float, optional
        Macroscopic roughness angle in radians, :math:`\bar{\theta}`.
        Default is 0.0.

    Returns
    -------
    jax.Array
        Reflectance per pixel.  Shape ``(n_pixels,)``.  Pixels where the
        source or detector is behind the local horizon are returned as NaN.

    References
    ----------
    .. bibliography::
       :filter: False

       Hapke-2012
    """
    return _imsa_batched(w, b_n, i, e, n, roughness)
