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


def test_lunar_lambert_table_survives_late_x64():
    """Importing refmod before enabling x64 must not freeze its tables at float32.

    Runs in subprocesses without JAX_ENABLE_X64, which the dev shell sets and
    which would otherwise hide the bug: x64 has to be off at import time.
    """
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}

    evaluate = (
        "import jax.numpy as jnp; from refmod.lunar_lambert import lunar_lambert; "
        "print(repr(float(lunar_lambert(1.0, 0.5, 0.3, jnp.asarray(0.7345)))))"
    )
    enable = "import jax; jax.config.update('jax_enable_x64', True); "
    early = subprocess.run(
        [sys.executable, "-c", enable + evaluate],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    late = subprocess.run(
        [sys.executable, "-c", "import refmod.lunar_lambert; " + enable + evaluate],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert late.stdout == early.stdout


def test_roughness_correction_is_continuous_at_zero_angles():
    """S and the effective cosines approach their e = 0 and i = 0 values smoothly.

    Both the MATLAB reference and refmod used to return S = 1 with the
    uncorrected cosines exactly at cos == 1, which is not the limit: mu_e
    jumped from 0.9455 to 1 for a normal within ~1e-8 rad of the viewer.
    """
    rough = 0.19198621771937624
    n = jnp.array([0.0, 0.0, 1.0])

    def vec(angle, azimuth):
        return jnp.array(
            [
                np.sin(angle) * np.cos(azimuth),
                np.sin(angle) * np.sin(azimuth),
                np.cos(angle),
            ]
        )

    for make in (
        lambda d: (vec(0.9, 0.0), vec(d, 1.0)),  # e -> 0
        lambda d: (vec(d, 0.0), vec(0.7, 1.0)),  # i -> 0
        lambda d: (vec(d, 0.0), vec(d, 1.0)),  # both
    ):
        exact = np.array(roughness_correction(rough, *make(0.0), n))
        near = np.array(roughness_correction(rough, *make(1e-6), n))
        np.testing.assert_allclose(exact, near, rtol=1e-8)


def test_amsa_gradient_is_finite_with_normal_in_sun_view_plane():
    """psi = 0 must not give NaN derivatives.

    Here the normal lies in the plane of sun and view, and rounding puts the
    cos(psi) quotient just above 1. The clip then fed arccos'(1) = -inf into
    the chain rule. Geometry taken from a shape-from-shading run where it hit.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    from refmod.api import HapkeAmsaParams, reflectance_pq_and_grad_jax

    params = HapkeAmsaParams(
        b=0.21, c=0.7, hs=0.11, Bs0=3.1, tb=0.19198621771937624, hc=None, Bc0=None
    )

    def one(x):
        return jnp.asarray(x, dtype=jnp.float64).reshape(1, 1, *np.shape(x))

    sun = one([0.75082469, -0.45385596, 0.47987193])
    view = one([0.0, 0.0, 1.0])
    p, q, h = 0.027414135636370695, -0.016571206599273285, 1e-6
    _, d_p, d_q = reflectance_pq_and_grad_jax(
        "amsa", one(0.13), sun, view, one(p), one(q), params
    )
    r_plus = reflectance_pq_and_grad_jax(
        "amsa", one(0.13), sun, view, one(p + h), one(q), params
    )[0]
    r_minus = reflectance_pq_and_grad_jax(
        "amsa", one(0.13), sun, view, one(p - h), one(q), params
    )[0]
    assert np.isfinite(float(d_p[0, 0])) and np.isfinite(float(d_q[0, 0]))
    np.testing.assert_allclose(
        float(d_p[0, 0]), float((r_plus - r_minus)[0, 0]) / (2 * h), rtol=1e-6
    )
