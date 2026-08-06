import numpy as np
import jax
import jax.numpy as jnp

from refmod.hapke._core import (
    coef_a,
    dhg_legendre_coefficients,
    double_henyey_greenstein,
    function_p,
    legendre_eval,
    value_p,
)


def test_a():
    n = 15
    hapke_table = np.array(
        [-0.5, 0.1250, -0.0625, 0.0391, -0.0273, 0.0205, -0.0161, 0.0131]
    )
    a_n = np.array(coef_a(n=n))
    a_n = np.round(a_n, 4)
    np.testing.assert_allclose(a_n[1::2], hapke_table)
    np.testing.assert_allclose(a_n[0::2], np.zeros((n + 1) // 2))


def test_legendre_eval_scalar():
    b = 0.21
    c_value = 0.7
    b_n = dhg_legendre_coefficients(b, c_value, n=15)

    g = np.linspace(0, np.pi, 100)
    cos_g = np.cos(g)

    p_scalar = np.array([float(legendre_eval(jnp.array(x), b_n)) for x in cos_g])

    legval = np.polynomial.legendre.legval(cos_g, np.array(b_n))

    np.testing.assert_allclose(p_scalar, legval, rtol=1e-5, atol=1e-5)


def test_legendre_eval_vmapped():
    b = 0.21
    c_value = 0.7
    b_n = dhg_legendre_coefficients(b, c_value, n=15)

    cos_g = jnp.cos(jnp.linspace(0, jnp.pi, 100))

    p_vmapped = jax.vmap(legendre_eval, in_axes=(0, None))(cos_g, b_n)

    legval = np.polynomial.legendre.legval(np.array(cos_g), np.array(b_n))

    np.testing.assert_allclose(np.array(p_vmapped), legval, rtol=1e-5, atol=1e-5)


def test_consistency_phase_legendre():
    b = 0.42
    limit = (1 + 3 * b**2) / b / (3 + b**2)
    c_value = 0.3 * limit

    g = np.deg2rad(np.linspace(0, 180, 30))
    cos_g = np.cos(g)

    b_n = dhg_legendre_coefficients(b, c_value, n=200)
    p_legendre = np.array([float(legendre_eval(jnp.array(x), b_n)) for x in cos_g])

    p_explicit = np.array(
        [float(double_henyey_greenstein(jnp.array(x), b, c_value)) for x in cos_g]
    )

    np.testing.assert_allclose(
        p_legendre,
        p_explicit,
        rtol=1e-5,
        atol=1e-8,
        err_msg="Legendre coefficients do not match Phase Function definition",
    )


def test_dhg_external_c_convention_is_direct():
    b = 0.21
    c_value = 0.7
    cos_g = jnp.array([-0.5, 0.0, 0.5])
    b_n = dhg_legendre_coefficients(b, c_value, n=200)

    p_legendre = jax.vmap(legendre_eval, in_axes=(0, None))(cos_g, b_n)
    p_explicit = double_henyey_greenstein(cos_g, b, c_value)

    np.testing.assert_allclose(np.array(p_legendre), np.array(p_explicit), rtol=1e-5, atol=1e-8)


def test_function_p_compatible():
    N = 15
    b = 0.21
    c_value = 0.7
    a_n = coef_a(n=N)
    range_n = np.arange(N + 1)
    b_n_old = c_value * (2 * range_n + 1) * np.power(b, range_n)
    b_n_new = dhg_legendre_coefficients(b, c_value, n=N)

    x = jnp.linspace(-1, 1, 10)

    f_old = np.array(function_p(x, jnp.array(b_n_old), a_n))
    f_new = np.array(function_p(x, b_n_new, a_n))

    np.testing.assert_allclose(f_old, f_new)


def test_value_p_compatible():
    import warnings

    N = 5
    b = 0.21
    c_value = 0.7
    a_n = coef_a(n=N)
    range_n = np.arange(N + 1)
    b_n_old = c_value * (2 * range_n + 1) * np.power(b, range_n)
    with warnings.catch_warnings():
        # The deliberately short N=5 expansion triggers the truncation
        # warning; this test only checks coefficient-convention parity.
        warnings.simplefilter("ignore", UserWarning)
        b_n_new = dhg_legendre_coefficients(b, c_value, n=N)

    v_old = np.array(value_p(jnp.array(b_n_old), a_n))
    v_new = np.array(value_p(b_n_new, a_n))

    np.testing.assert_allclose(v_old, v_new)
