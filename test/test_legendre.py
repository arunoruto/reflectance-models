import numpy as np
import numpy.testing as npt

from refmod.hapke.functions.legendre import (
    coef_a,
    dhg_legendre_coefficients,
    function_p,
    legendre_eval,
    value_p,
)
from refmod.hapke.functions.phase import double_henyey_greenstein

L = 15


def test_a():
    n = 15
    hapke_table = np.array(
        [-0.5, 0.1250, -0.0625, 0.0391, -0.0273, 0.0205, -0.0161, 0.0131]
    )
    a_n = coef_a(n=n)
    a_n = a_n.squeeze()
    a_n = np.round(a_n, 4)
    npt.assert_allclose(a_n[1::2], hapke_table)
    npt.assert_allclose(a_n[0::2], np.zeros((n + 1) // 2))


def test_legendre_eval():
    shape = (2, 2)
    b = np.random.rand(*shape)
    limit = (1 + 3 * b**2) / b / (3 + b**2)
    c = (2 * np.random.rand(*shape) - 1) * limit
    n = np.expand_dims(np.arange(L + 1), axis=tuple(range(1, len(shape) + 1)))
    b_n = (2 * n + 1) * np.power(b[np.newaxis, ...], n)
    b_n[1::2] *= c[np.newaxis, ...]

    g = np.linspace(0, np.pi, 180).reshape(12, 15)
    cos_g = np.cos(g)

    eval = legendre_eval(cos_g, b_n)
    legval = np.polynomial.legendre.legval(cos_g, b_n)
    for _ in range(len(cos_g.shape)):
        legval = np.moveaxis(legval, -1, 0)

    npt.assert_allclose(legval, eval, rtol=1e-5, atol=1e-5)


def test_consistency_phase_legendre():
    # cap it between 0 and 0.99
    # being close to b=1 makes it less accurate!
    b = np.random.rand() * 0.99
    limit = (1 + 3 * b**2) / b / (3 + b**2)
    c = (2 * np.random.rand() - 1) * limit

    g = np.deg2rad(np.linspace(0, 180, 100))
    cos_g = np.cos(g)

    p_explicit = double_henyey_greenstein(cos_g, b, c)

    b_n = dhg_legendre_coefficients(b, c, n=10_000)
    p_legendre = legendre_eval(cos_g, b_n.squeeze())

    np.testing.assert_allclose(
        p_legendre,
        p_explicit,
        err_msg="Legendre coefficients do not match Phase Function definition",
    )


def test_do_even_terms_matter_in_function_p():
    N = 15
    b = 0.21
    c = 0.7
    a_n = coef_a(n=N)
    range_n = np.arange(N + 1)
    b_n_old = c * (2 * range_n + 1) * np.power(b, range_n)
    b_n_new = dhg_legendre_coefficients(b, c, n=N)

    x = np.linspace(-1, 1, 10)

    f_old = function_p(x, b_n_old, a_n)
    f_new = function_p(x, b_n_new, a_n)

    npt.assert_allclose(f_old, f_new)


def test_do_even_terms_matter_in_value_p():
    N = 5
    b = 0.21
    c = 0.7
    a_n = coef_a(n=N)
    range_n = np.arange(N + 1)
    b_n_old = c * (2 * range_n + 1) * np.power(b, range_n)
    # b_n_old = np.reshape(c * (2 * range_n + 1) * np.power(b, range_n), (-1, 1, 1))
    b_n_new = dhg_legendre_coefficients(b, c, n=N)

    v_old = value_p(b_n_old, a_n)
    v_new = value_p(b_n_new, a_n)

    npt.assert_allclose(v_old, v_new)
