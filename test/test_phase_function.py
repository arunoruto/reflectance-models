import logging

import numpy as np
from refmod.hapke.functions.phase import double_henyey_greenstein

N = 180 * 4 + 1

logger = logging.getLogger(__name__)


def test_dhg():
    b = np.random.rand()
    limit = (1 + 3 * b**2) / b / (3 + b**2)
    c = (2 * np.random.rand() - 1) * limit
    g = np.linspace(0, np.pi, N)
    d_g = np.mean(g[1:] - g[:-1])
    cos_g = np.cos(g)
    sin_g = np.sin(g)

    p_g = double_henyey_greenstein(cos_g, b, c)
    integral = float(0.5 * np.sum(p_g * sin_g) * d_g)
    logger.info(f"{b=} {c=} {integral=}")
    np.testing.assert_allclose(integral, 1, rtol=5e-3)
