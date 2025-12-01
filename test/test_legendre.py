import logging

import numpy as np
import numpy.testing as npt
from refmod.hapke.functions.legendre import legendre_eval

L = 15

logger = logging.getLogger(__name__)


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
    from refmod.hapke.functions.legendre import coef_b, legendre_eval
    from refmod.hapke.functions.phase import double_henyey_greenstein

    b = 0.21
    c = 0.7
    # Test angles: 0 to 180 degrees
    g = np.deg2rad(np.linspace(0, 180, 100))
    cos_g = np.cos(g)

    # 1. Calculate via explicit Phase Function
    p_explicit = double_henyey_greenstein(cos_g, b, c)

    # 2. Calculate via Legendre Expansion
    # High degree to ensure convergence
    b_n = coef_b(b, c, n=100)
    p_legendre = legendre_eval(cos_g, b_n)

    # Compare
    # They should be very close if definitions match
    np.testing.assert_allclose(
        p_legendre,
        p_explicit,
        rtol=1e-4,
        atol=1e-4,
        err_msg="Legendre coefficients do not match Phase Function definition",
    )

