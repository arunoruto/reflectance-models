import jax.numpy as jnp
import numpy as np

from refmod.hapke import amsa, dhg_legendre_coefficients, imsa, mimsa
from refmod.hapke._core import h_function, normalize
from refmod.lambert import lambert


def test_lambert_is_linear_in_albedo():
    w = jnp.array([0.2, 0.4, 0.7])
    k = 1.5
    i = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.4, 0.91651514], [0.0, 0.7, 0.71414284]])
    e1 = jnp.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.70710678], [0.0, 1.0, 0.0]])
    e2 = jnp.array([[0.0, 1.0, 0.0], [0.2, 0.0, 0.9797959], [1.0, 0.0, 0.0]])
    n = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (3, 1))

    r = lambert(w, i, e1, n)
    r_scaled = lambert(k * w, i, e1, n)
    r_alt_e = lambert(w, i, e2, n)

    np.testing.assert_allclose(np.array(r_scaled), np.array(k * r), rtol=1e-7)
    np.testing.assert_allclose(np.array(r_alt_e), np.array(r), rtol=1e-7)
    assert np.all(np.array(r) >= 0.0)


def test_hapke_family_returns_nan_for_hidden_geometry():
    w = jnp.array([0.4])
    b_n = dhg_legendre_coefficients(0.2, 0.4, 10)
    i = jnp.array([[0.0, 0.0, -1.0]])
    e = jnp.array([[0.0, 0.0, 1.0]])
    n = jnp.array([[0.0, 0.0, 1.0]])

    r_imsa = imsa(w, b_n, i, e, n)
    r_mimsa = mimsa(w, b_n, i, e, n)
    r_amsa = amsa(w, b_n, i, e, n)

    assert np.isnan(np.array(r_imsa)[0])
    assert np.isnan(np.array(r_mimsa)[0])
    assert np.isnan(np.array(r_amsa)[0])


def test_hapke_family_zero_albedo_yields_zero_reflectance():
    w = jnp.array([0.0, 0.0])
    b_n = dhg_legendre_coefficients(0.25, 0.5, 10)
    i = jnp.array([[0.0, 0.3, 0.9539392], [0.0, 0.1, 0.9949874]])
    e = jnp.array([[0.0, 0.2, 0.9797959], [0.0, 0.4, 0.9165151]])
    n = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (2, 1))

    np.testing.assert_allclose(np.array(imsa(w, b_n, i, e, n)), np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(
        np.array(mimsa(w, b_n, i, e, n)), np.zeros(2), atol=1e-12
    )
    np.testing.assert_allclose(np.array(amsa(w, b_n, i, e, n)), np.zeros(2), atol=1e-12)


def test_h_function_increases_with_albedo_for_positive_mu():
    mu = jnp.array([0.1, 0.4, 0.8])
    h_low = h_function(mu, 0.2)
    h_high = h_function(mu, 0.8)
    assert np.all(np.array(h_high) > np.array(h_low))


def test_normalize_operates_per_vector_for_batched_input():
    vectors = jnp.array([[3.0, 0.0, 4.0], [0.0, 5.0, 12.0]])
    normalized = normalize(vectors)
    norms = jnp.sqrt(jnp.sum(normalized**2, axis=-1))
    np.testing.assert_allclose(np.array(norms), np.ones(2), rtol=1e-12)
