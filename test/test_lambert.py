import jax.numpy as jnp
import numpy as np

from refmod.lambert import lambert


def test_lambert_single():
    w = jnp.array([0.5])
    i = jnp.array([[0.0, 0.5, 0.8660254]])
    n = jnp.array([[0.0, 0.0, 1.0]])
    e = jnp.array([[1.0, 0.0, 0.0]])

    r = lambert(w, i, e, n)
    expected = 0.5 * 0.8660254 / jnp.pi
    np.testing.assert_allclose(np.array(r), np.array(expected))


def test_lambert_dark_side():
    w = jnp.array([0.5])
    i = jnp.array([[0.0, -1.0, 0.0]])
    n = jnp.array([[0.0, 0.0, 1.0]])
    e = jnp.array([[0.0, 0.0, 1.0]])

    r = lambert(w, i, e, n)
    np.testing.assert_allclose(np.array(r), [0.0])


def test_lambert_batch():
    w = jnp.array([0.3, 0.5, 0.7])
    i = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.5, 0.8660254]])
    n = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (3, 1))
    e = jnp.zeros_like(i)

    r = lambert(w, i, e, n)
    expected = jnp.array([0.3, 0.0, 0.7 * 0.8660254]) / jnp.pi
    np.testing.assert_allclose(np.array(r), np.array(expected))
