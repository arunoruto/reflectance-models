import numpy as np

from refmod.utils.spectral_continuum import (
    continuum_remove_upper_hull,
    smooth_spectrum_m3,
    upper_hull_continuum,
)


def test_upper_hull_constant_spectrum():
    wl = np.array([1.0, 2.0, 3.0, 4.0])
    r = np.full_like(wl, 0.5)

    cont, idx = upper_hull_continuum(wl, r)
    np.testing.assert_allclose(cont, r)
    assert idx.size >= 2

    cr, cont2, idx2 = continuum_remove_upper_hull(wl, r, smooth=False)
    np.testing.assert_allclose(cont2, r)
    np.testing.assert_allclose(cr, np.ones_like(r))
    np.testing.assert_array_equal(idx2, idx)


def test_upper_hull_linear_spectrum():
    wl = np.linspace(1.0, 2.0, 21)
    r = 0.2 + 0.3 * (wl - wl.min())

    cont, _ = upper_hull_continuum(wl, r)
    np.testing.assert_allclose(cont, r, rtol=0, atol=1e-12)

    cr, _, _ = continuum_remove_upper_hull(wl, r, smooth=False)
    np.testing.assert_allclose(cr, np.ones_like(r), rtol=0, atol=1e-12)


def test_continuum_removed_ratio_leq_one_for_absorption():
    wl = np.linspace(1.0, 2.0, 200)
    baseline = 1.0 + 0.05 * (wl - wl.min())
    dip = 0.25 * np.exp(-0.5 * ((wl - 1.5) / 0.03) ** 2)
    r = baseline - dip

    cr, cont, _ = continuum_remove_upper_hull(wl, r, smooth=False)
    assert np.nanmin(cr) < 0.95
    assert np.nanmax(cr) <= 1.0 + 1e-9
    assert np.all(cont + 1e-12 >= r)


def test_smooth_spectrum_m3_trivial_sizes_and_nans():
    wl = np.array([1.0, 2.0, np.nan])
    r = np.array([0.1, 0.2, 0.3])
    out = smooth_spectrum_m3(wl, r)
    np.testing.assert_allclose(out[:2], r[:2])
    assert np.isnan(out[2])
