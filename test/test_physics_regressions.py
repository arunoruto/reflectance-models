"""Regression tests for physics bugs found in the 2026-07 review.

1. Roughness correction must keep the psi/pi * E1 denominator term for both
   effective cosines (Hapke 1984, Eqs. 47-50). The hopper fixtures cannot
   detect this because e = 0 deg makes the term vanish; here we compare
   against a direct transcription of the MATLAB reference at doubly-oblique
   geometry.
2. The coherent-backscatter factor must be continuous at exact opposition:
   B_CB(0) = 1 + B0.
3. Back-facing geometry (raw cos_i or cos_e <= 0) must be masked as NaN even
   when roughness > 0, where the Hapke correction would otherwise fabricate
   positive effective cosines.
"""

import jax.numpy as jnp
import numpy as np

from refmod.hapke import amsa, dhg_legendre_coefficients
from refmod.hapke._core import coherent_backscatter, roughness_correction


def _matlab_roughness(tb, s, v, n):
    """Single-pixel transcription of the MATLAB roughness correction.

    Transcribed from the block that was inline in ``hapke_amsa.m``. As of the
    toolbox's 2.0.0 release it lives in ``src/private/hapke_roughness.m``,
    shared by five models. It was extracted bit-exactly for ``hapke_amsa``, so
    this transcription still matches the reference it was taken from.
    """
    s = s / np.linalg.norm(s)
    v = v / np.linalg.norm(v)
    n = n / np.linalg.norm(n)
    mu0 = np.clip(np.dot(s, n), -1, 1)
    mu = np.clip(np.dot(v, n), -1, 1)
    si = np.sqrt(1 - mu0**2)
    se = np.sqrt(1 - mu**2)
    coti = mu0 / si
    cote = mu / se

    projs = s - mu0 * n
    projv = v - mu * n
    cpsi = np.clip(
        np.dot(projs, projv) / np.linalg.norm(projs) / np.linalg.norm(projv), -1, 1
    )
    spsi2_sq = (1 - cpsi) / 2
    spsi = np.sqrt(1 - cpsi**2)
    psi = np.arccos(cpsi)

    tantb = np.tan(tb)
    cottb = 1 / tantb
    factor = 1 / np.sqrt(1 + np.pi * tantb * tantb)

    def fe(x):
        return np.exp(-2 / np.pi * cottb * x)

    def fe2(x):
        return np.exp(-cottb * cottb * x * x / np.pi)

    fpsi = np.exp(-2 * (spsi / (1 + cpsi))) if cpsi != -1 else 0.0

    mu0s0 = factor * (mu0 + si * tantb * fe2(coti) / (2 - fe(coti)))
    mus0 = factor * (mu + se * tantb * fe2(cote) / (2 - fe(cote)))

    if mu0 >= mu:  # i <= e
        den = 2 - fe(cote) - psi / np.pi * fe(coti)
        mu0s = factor * (
            mu0 + si * tantb * (cpsi * fe2(cote) + spsi2_sq * fe2(coti)) / den
        )
        mus = factor * (mu + se * tantb * (fe2(cote) - spsi2_sq * fe2(coti)) / den)
        s_corr = factor * (mus / mus0) * (mu0 / mu0s0)
        s_corr = s_corr / (1 - fpsi + fpsi * (mu0 / mu0s0) * factor)
    else:  # i > e
        den = 2 - fe(coti) - psi / np.pi * fe(cote)
        mu0s = factor * (mu0 + si * tantb * (fe2(coti) - spsi2_sq * fe2(cote)) / den)
        mus = factor * (
            mu + se * tantb * (cpsi * fe2(coti) + spsi2_sq * fe2(cote)) / den
        )
        s_corr = factor * (mus / mus0) * (mu0 / mu0s0)
        s_corr = s_corr / (1 - fpsi + fpsi * (mu / mus0) * factor)
    return s_corr, mu0s, mus


def _random_upward(rng, floor=0.3):
    v = rng.normal(size=3)
    v[2] = abs(v[2]) + floor
    return v / np.linalg.norm(v)


def test_roughness_correction_matches_matlab_at_oblique_geometry():
    rng = np.random.default_rng(0)
    tb = 0.1396  # ~8 deg, as in the hopper fixture

    checked = 0
    for _ in range(500):
        s = _random_upward(rng)
        v = _random_upward(rng)
        n = _random_upward(rng)
        if np.dot(s, n) <= 0.05 or np.dot(v, n) <= 0.05:
            continue
        s_ref, mu0_ref, mu_ref = _matlab_roughness(tb, s, v, n)
        s_out, mu0_out, mu_out = roughness_correction(
            tb, jnp.asarray(s), jnp.asarray(v), jnp.asarray(n)
        )
        np.testing.assert_allclose(float(mu0_out), mu0_ref, rtol=1e-12)
        np.testing.assert_allclose(float(mu_out), mu_ref, rtol=1e-12)
        np.testing.assert_allclose(float(s_out), s_ref, rtol=1e-12)
        checked += 1

    assert checked > 100  # geometry filter must leave a meaningful sample


def test_coherent_backscatter_continuous_at_opposition():
    h, b0 = 0.1, 2.0
    at_zero = float(coherent_backscatter(jnp.asarray(0.0), h, b0))
    near_zero = float(coherent_backscatter(jnp.asarray(1e-9), h, b0))

    np.testing.assert_allclose(at_zero, 1.0 + b0, rtol=1e-12)
    np.testing.assert_allclose(near_zero, at_zero, rtol=1e-6)


def test_backfacing_geometry_is_nan_with_roughness():
    b_n = dhg_legendre_coefficients(0.21, 0.7, 15)
    n = jnp.asarray([[0.0, 0.0, 1.0]] * 2)
    e = jnp.asarray([[0.0, 0.0, 1.0]] * 2)
    # first pixel: sun behind the surface; second: valid oblique sun
    s = jnp.asarray([[0.6, 0.6, -0.2], [0.6, 0.6, 0.2]])
    s = s / jnp.linalg.norm(s, axis=1, keepdims=True)
    w = jnp.asarray([0.4, 0.4])

    refl = amsa(w, b_n, s, e, n, roughness=0.1396, h_sh=0.09, b0_sh=2.7)

    assert np.isnan(np.asarray(refl[0]))
    assert np.isfinite(np.asarray(refl[1]))


def test_dhg_truncation_warning():
    import warnings

    from refmod.hapke import dhg_truncation_error, recommended_dhg_order

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dhg_legendre_coefficients(0.18, 1.1, 15)  # accurate: must not warn

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        dhg_legendre_coefficients(0.6, 0.7, 15)
    assert len(rec) == 1
    assert "Consider n=" in str(rec[0].message)

    from refmod.hapke._core import DHG_TRUNCATION_WARN_THRESHOLD

    n_rec = recommended_dhg_order(0.6, 0.7)
    assert dhg_truncation_error(0.6, 0.7, n_rec) < DHG_TRUNCATION_WARN_THRESHOLD
