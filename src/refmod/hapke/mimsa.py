import jax
import jax.numpy as jnp

from ._core import (
    coef_a,
    cos_angle,
    function_p,
    h_function,
    legendre_eval,
    normalize,
    roughness_correction,
    value_p,
)
def _refl_mimsa_scalar(
    w: jax.Array,
    b_n: jax.Array,
    i: jax.Array,
    e: jax.Array,
    n: jax.Array,
    roughness: float,
    a_n: jax.Array | None = None,
) -> jax.Array:
    r"""MIMSA reflectance for a single pixel.

    Modified IMSA model that replaces the isotropic multiple-scattering
    approximation with the full Legendre *P*-term formulation.  No opposition
    effects are included.

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
    a_n : jax.Array or None, optional
        Legendre expansion coefficients for the multiple-scattering term.
        When ``None`` (default), they are computed from the order of *b_n*
        using :func:`~._core.coef_a`.

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
       Hapke-2002
    """
    i = normalize(i)
    e = normalize(e)
    n = normalize(n)
    s, mu0, mu = roughness_correction(roughness, i, e, n)
    cos_alpha = cos_angle(i, e)
    p = legendre_eval(cos_alpha, b_n)
    h_mu0 = h_function(mu0, w)
    h_mu = h_function(mu, w)
    if a_n is None:
        a_n = coef_a(b_n.shape[0] - 1)
    p_mu0 = function_p(mu0, b_n, a_n)
    p_mu = function_p(mu, b_n, a_n)
    P = value_p(b_n, a_n)
    m = p_mu0 * (h_mu - 1.0) + p_mu * (h_mu0 - 1.0) + P * (h_mu0 - 1.0) * (h_mu - 1.0)
    albedo_indep = mu0 / (mu0 + mu) * s / (4.0 * jnp.pi)
    refl = albedo_indep * w * (p + m)
    return jnp.where((mu <= 0.0) | (mu0 <= 0.0), jnp.nan, refl)


_mimsa_batched = jax.vmap(
    _refl_mimsa_scalar,
    in_axes=(0, None, 0, 0, 0, None, None),
)


def mimsa(
    w: jax.Array,
    b_n: jax.Array,
    i: jax.Array,
    e: jax.Array,
    n: jax.Array,
    roughness: float = 0.0,
    a_n: jax.Array | None = None,
) -> jax.Array:
    r"""Batched MIMSA reflectance.

    Vectorised wrapper around :func:`_refl_mimsa_scalar` that evaluates the
    modified isotropic multiple-scattering Hapke model for an arbitrary
    number of pixels sharing the same Legendre coefficients and roughness.

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
    a_n : jax.Array or None, optional
        Legendre expansion coefficients for the multiple-scattering term.
        When ``None`` (default), they are computed from the order of *b_n*
        using :func:`~._core.coef_a`.

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
       Hapke-2002
    """
    return _mimsa_batched(w, b_n, i, e, n, roughness, a_n)
