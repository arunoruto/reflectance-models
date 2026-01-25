import numpy as np
import pytest

from refmod.utils import fit_linear_spectrum_combination


def test_fit_linear_combination_recovers_weights_relative_error():
    rng = np.random.default_rng(0)

    wl = np.linspace(1.0, 2.0, 200)
    s1 = 0.4 + 0.05 * (wl - wl.min())
    s2 = 0.9 - 0.15 * (wl - wl.min()) + 0.03 * np.sin(20 * wl)
    basis = np.column_stack([s1, s2])

    w_true = np.array([0.2, 0.8])
    target = basis @ w_true
    target = target + 1e-4 * rng.standard_normal(target.shape)

    res = fit_linear_spectrum_combination(
        basis,
        target,
        bounds=(0.0, 1.0),
        relative_error=True,
        sum_to_one=True,
        sum_to_one_weight=10.0,
    )

    np.testing.assert_allclose(res.weights.sum(), 1.0, atol=1e-5)
    np.testing.assert_allclose(res.weights, w_true, atol=2e-2)


def test_fit_mask_is_respected():
    wl = np.linspace(1.0, 2.0, 100)
    s1 = np.ones_like(wl)
    s2 = wl
    basis = np.column_stack([s1, s2])

    w_true = np.array([1.0, 0.0])
    target = basis @ w_true

    # Corrupt the first half; fit only the second half.
    target_corrupt = target.copy()
    target_corrupt[:50] = 1000.0
    mask = np.zeros_like(wl, dtype=bool)
    mask[50:] = True

    res = fit_linear_spectrum_combination(
        basis,
        target_corrupt,
        bounds=(-np.inf, np.inf),
        fit_mask=mask,
        relative_error=False,
    )

    np.testing.assert_allclose(res.weights, w_true, atol=1e-8)


def test_continuum_alpha_is_clipped_with_warning():
    wl = np.linspace(1.0, 2.0, 50)
    s1 = 0.3 + 0.1 * (wl - wl.min())
    s2 = 0.7 - 0.05 * (wl - wl.min())
    basis = np.column_stack([s1, s2])
    target = basis @ np.array([0.4, 0.6])

    with pytest.warns(RuntimeWarning):
        res = fit_linear_spectrum_combination(
            basis,
            target,
            wavelength=wl,
            continuum_alpha=2.0,
            bounds=(0.0, 1.0),
            sum_to_one=True,
            sum_to_one_weight=10.0,
        )
    assert 0.0 <= res.alpha <= 1.0


def test_warn_if_continuum_smooth_false_with_alpha_lt_one():
    wl = np.linspace(1.0, 2.0, 50)
    s1 = 0.3 + 0.1 * (wl - wl.min())
    s2 = 0.7 - 0.05 * (wl - wl.min())
    basis = np.column_stack([s1, s2])
    target = basis @ np.array([0.4, 0.6])

    with pytest.warns(RuntimeWarning):
        fit_linear_spectrum_combination(
            basis,
            target,
            wavelength=wl,
            continuum_alpha=0.5,
            continuum_smooth=False,
            bounds=(0.0, 1.0),
            sum_to_one=True,
            sum_to_one_weight=10.0,
        )
