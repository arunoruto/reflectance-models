import jax.numpy as jnp
import numpy as np

from refmod.shkuratov import shkuratov


def test_shkuratov_zero_params():
    a_n = jnp.array([0.5])
    i = jnp.array([[0.0, 0.0, 1.0]])
    e = jnp.array([[0.0, 0.0, 1.0]])
    n = jnp.array([[0.0, 0.0, 1.0]])

    r = shkuratov(a_n, mu1=0.0, eta=0.0, i=i, e=e, n=n)
    np.testing.assert_allclose(np.array(r), [0.5])


def test_shkuratov_dark_side():
    a_n = jnp.array([0.5])
    i = jnp.array([[0.0, -1.0, 0.0]])
    n = jnp.array([[0.0, 0.0, 1.0]])
    e = jnp.array([[0.0, 0.0, 1.0]])

    r = shkuratov(a_n, mu1=0.0, eta=0.0, i=i, e=e, n=n)
    np.testing.assert_allclose(np.array(r), [0.0])


def test_shkuratov_roughness_reduces():
    a_n = jnp.array([0.5])
    i = jnp.array([[0.0, 0.5, 0.8660254]])
    n = jnp.array([[0.0, 0.0, 1.0]])
    e = jnp.array([[0.0, 0.0, 1.0]])

    r_flat = shkuratov(a_n, mu1=0.0, eta=0.0, i=i, e=e, n=n)
    r_rough = shkuratov(a_n, mu1=0.5, eta=0.0, i=i, e=e, n=n)
    assert float(r_rough[0]) < float(r_flat[0])


def test_shkuratov_opposition_surge():
    a_n = jnp.array([0.5])
    i = jnp.array([[0.0, 0.01745, 0.99985]])  # ~1°
    n = jnp.array([[0.0, 0.0, 1.0]])
    e = jnp.array([[0.0, 0.0, 1.0]])

    r_no = shkuratov(a_n, mu1=0.5, i=i, e=e, n=n)
    r_opp = shkuratov(a_n, mu1=0.5, m0=0.5, mu2=0.05, i=i, e=e, n=n)
    assert float(r_opp[0]) > float(r_no[0])


def test_shkuratov_batch():
    a_n = jnp.array([0.3, 0.5, 0.7])
    i = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (3, 1))
    e = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (3, 1))
    n = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (3, 1))

    r = shkuratov(a_n, mu1=0.0, eta=0.0, i=i, e=e, n=n)
    np.testing.assert_allclose(np.array(r), np.array([0.3, 0.5, 0.7]))
