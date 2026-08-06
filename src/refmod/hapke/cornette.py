import jax
import jax.numpy as jnp

from ._core import (
    coef_a,
    coherent_backscatter,
    cornette_shanks,
    cos_angle,
    h_function,
    normalize,
    roughness_correction,
    shadow_hiding,
)
from .imsa_modified import _modified_h


def cornette_legendre_coefficients(xi: float, n: int = 11) -> jax.Array:
    r"""Cornette-Shanks Legendre expansion coefficients.

    Implemented from the reference ``hapke_amsa_cornette.m``. These have not
    yet been fixture-validated against the reference output; prefer the
    Double Henyey-Greenstein models for validated results.
    """
    coeffs = [jnp.array(1.0)]
    denom = 2.0 + xi**2
    if n >= 1:
        coeffs.append(1.5 / denom * (24.0 / 5.0 * xi + 6.0 / 5.0 * xi**3))
    for order in range(2, n + 1):
        order_f = float(order)
        c1 = order_f * (order_f - 1.0) / (2.0 * order_f - 1.0)
        c2 = (5.0 * order_f**2 - 1.0) / (2.0 * order_f - 1.0) + (order_f + 1.0) ** 2 / (
            2.0 * order_f + 3.0
        )
        c3 = (order_f + 1.0) * (order_f + 2.0) / (2.0 * order_f + 3.0)
        coeffs.append(1.5 / denom * (c1 * xi ** (order - 2) + c2 * xi**order + c3 * xi ** (order + 2)))
    return jnp.asarray(coeffs)


def _cornette_multiple_scattering_terms(
    mu0: jax.Array,
    mu: jax.Array,
    xi: float,
    n_order: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    b_n = cornette_legendre_coefficients(xi, n_order)
    a_n = coef_a(n_order)
    p_mu0 = 1.0
    p_mu = 1.0
    rho = 1.0
    p0_prev2 = 1.0
    p0_prev1 = mu0
    p_prev2 = 1.0
    p_prev1 = mu
    for order in range(1, n_order + 1):
        if order == 1:
            p0_curr = p0_prev1
            p_curr = p_prev1
        else:
            order_f = float(order)
            p0_curr = (2.0 * order_f - 1.0) / order_f * mu0 * p0_prev1 - (order_f - 1.0) / order_f * p0_prev2
            p_curr = (2.0 * order_f - 1.0) / order_f * mu * p_prev1 - (order_f - 1.0) / order_f * p_prev2
            p0_prev2, p0_prev1 = p0_prev1, p0_curr
            p_prev2, p_prev1 = p_prev1, p_curr
        if order % 2 == 1:
            p_mu0 = p_mu0 + a_n[order] * b_n[order] * p0_curr
            p_mu = p_mu + a_n[order] * b_n[order] * p_curr
            rho = rho - a_n[order] ** 2 * b_n[order]
    return p_mu0, p_mu, rho


def _refl_amsa_cornette_scalar(
    w: jax.Array,
    xi: float,
    i: jax.Array,
    e: jax.Array,
    n: jax.Array,
    roughness: float,
    h_sh: float,
    b0_sh: float,
    h_cb: float,
    b0_cb: float,
    n_order: int = 11,
) -> jax.Array:
    i = normalize(i)
    e = normalize(e)
    n = normalize(n)
    # Raw-cosine mask: the roughness correction is undefined for
    # back-facing geometry and can yield spurious positive cosines.
    invalid = (cos_angle(i, n) <= 0.0) | (cos_angle(e, n) <= 0.0)
    s, mu0, mu = roughness_correction(roughness, i, e, n)
    invalid = invalid | (mu <= 0.0) | (mu0 <= 0.0)
    cos_alpha = cos_angle(i, e)
    sin_alpha = jnp.sqrt(jnp.maximum(1.0 - cos_alpha**2, 0.0))
    tan_alpha_2 = sin_alpha / (1.0 + cos_alpha)
    phase = cornette_shanks(cos_alpha, xi)
    p_mu0, p_mu, rho = _cornette_multiple_scattering_terms(mu0, mu, xi, n_order)
    h_mu0 = h_function(mu0, w)
    h_mu = h_function(mu, w)
    hm0_1 = h_mu0 - 1.0
    hm_1 = h_mu - 1.0
    m_term = p_mu0 * hm_1 + p_mu * hm0_1 + rho * hm0_1 * hm_1
    b_sh = shadow_hiding(tan_alpha_2, h_sh, b0_sh)
    b_cb = coherent_backscatter(tan_alpha_2, h_cb, b0_cb)
    prefactor = mu0 / (mu0 + mu) * b_cb * s / (4.0 * jnp.pi)
    refl = w * (phase * b_sh + m_term) * prefactor
    return jnp.where(invalid, jnp.nan, refl)


def _refl_imsa_cornette_scalar(
    w: jax.Array,
    xi: float,
    i: jax.Array,
    e: jax.Array,
    n: jax.Array,
    roughness: float,
    h_sh: float,
    b0_sh: float,
) -> jax.Array:
    i = normalize(i)
    e = normalize(e)
    n = normalize(n)
    # Raw-cosine mask: the roughness correction is undefined for
    # back-facing geometry and can yield spurious positive cosines.
    invalid = (cos_angle(i, n) <= 0.0) | (cos_angle(e, n) <= 0.0)
    s, mu0, mu = roughness_correction(roughness, i, e, n)
    invalid = invalid | (mu <= 0.0) | (mu0 <= 0.0)
    cos_alpha = cos_angle(i, e)
    sin_alpha = jnp.sqrt(jnp.maximum(1.0 - cos_alpha**2, 0.0))
    tan_alpha_2 = sin_alpha / (1.0 + cos_alpha)
    phase = cornette_shanks(cos_alpha, xi)
    b_sh = shadow_hiding(tan_alpha_2, h_sh, b0_sh)
    h_mu0 = _modified_h(mu0, w)
    h_mu = _modified_h(mu, w)
    prefactor = mu0 / (mu0 + mu) * s / (4.0 * jnp.pi)
    refl = w * (b_sh * phase + h_mu0 * h_mu - 1.0) * prefactor
    return jnp.where(invalid, jnp.nan, refl)


_amsa_cornette_batched = jax.vmap(
    _refl_amsa_cornette_scalar,
    in_axes=(0, None, 0, 0, 0, None, None, None, None, None, None),
)
_imsa_cornette_batched = jax.vmap(
    _refl_imsa_cornette_scalar,
    in_axes=(0, None, 0, 0, 0, None, None, None),
)


def amsa_cornette(
    w: jax.Array,
    xi: float,
    i: jax.Array,
    e: jax.Array,
    n: jax.Array,
    roughness: float = 0.0,
    h_sh: float = 0.0,
    b0_sh: float = 0.0,
    h_cb: float = 0.0,
    b0_cb: float = 0.0,
    n_order: int = 11,
) -> jax.Array:
    return _amsa_cornette_batched(w, xi, i, e, n, roughness, h_sh, b0_sh, h_cb, b0_cb, n_order)


def imsa_cornette(
    w: jax.Array,
    xi: float,
    i: jax.Array,
    e: jax.Array,
    n: jax.Array,
    roughness: float = 0.0,
    h_sh: float = 0.0,
    b0_sh: float = 0.0,
) -> jax.Array:
    return _imsa_cornette_batched(w, xi, i, e, n, roughness, h_sh, b0_sh)
