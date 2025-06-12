"""Reference for Hapke's Legendre polynomial coefficients and functions.

This module implements coefficients and functions related to Legendre polynomial
expansions as described by Hapke. These are primarily used for modeling
anisotropic scattering and phase functions.

??? info "References"

    Hapke (2002)
"""

import numpy as np
import numpy.typing as npt
from numba import jit
from refmod.config import cache


@jit(nogil=True, fastmath=True, cache=cache)
def coef_a(n: int = 15) -> npt.NDArray:
    """Calculates coefficients 'a_n' for Legendre polynomial series.

    These coefficients are used in Hapke's photometric model.

    Parameters
    ----------

    n : int, optional
        The number of coefficients to calculate (degree of Legendre polynomial),
        by default 15. The resulting array will have `n + 1` elements.

    Returns
    -------
    npt.NDArray
        Array of 'a_n' coefficients, shape (n + 1,).

    References
    ----------
    Hapke (2002, Eq. 27).
    """
    a_n = np.zeros(n + 1)
    a_n[1] = -0.5
    for i in range(3, n + 1, 2):
        a_n[i] = (2 - i) / (i + 1) * a_n[i - 2]
    return a_n


@jit(nogil=True, fastmath=True, cache=cache)
def coef_b(b: float = 0.21, c: float = 0.7, n: int = 15) -> npt.NDArray:
    """Calculates coefficients 'b_n' for Legendre polynomial expansion.

    These coefficients are used in Hapke's photometric model, specifically
    for the phase function representation.

    Parameters
    ----------

    b : float, optional
        Asymmetry parameter for the Henyey-Greenstein phase function component,
        by default 0.21.
    c : float, optional
        Parameter determining the mixture of Henyey-Greenstein functions or
        a single function if NaN, by default 0.7.
        If `c` is `np.nan`, a single Henyey-Greenstein function is assumed.
    n : int, optional
        The number of coefficients to calculate (degree of Legendre polynomial),
        by default 15. The resulting array will have `n + 1` elements.

    Returns
    -------
    npt.NDArray
        Array of 'b_n' coefficients, shape (n + 1,).

    Notes
    -----
    The calculation method depends on whether `c` is NaN.
    The first element `b_n[0]` is set to 1 if `c` is not NaN, which differs
    from the direct formula application for that term.

    References
    ----------
    Hapke (2002, p. 530).
    """
    if np.isnan(c):
        range_n = np.arange(n + 1) + 1  # Corrected variable name
        b_n = (2 * range_n + 1) * np.power(-b, range_n)
    else:
        range_n = np.arange(n + 1)  # Corrected variable name
        b_n = c * (2 * range_n + 1) * np.power(b, range_n)
        # TODO: why is the first element one and not c?
        # b_n[0] = 1
    return b_n


@jit(nogil=True, fastmath=True, cache=cache)
def function_p(
    x: npt.NDArray,
    b_n: npt.NDArray | None = None,
    a_n: npt.NDArray | None = None,
) -> npt.NDArray:
    """Calculates the P function from Hapke's model.

    This function relates to the integrated phase function and accounts for
    anisotropic scattering.

    Parameters
    ----------

    x : npt.NDArray
        Input array, typically cosine of angles (e.g., mu, mu0).
    b_n : npt.NDArray
        Array of 'b_n' coefficients.
    a_n : npt.NDArray, optional
        Array of 'a_n' coefficients. If not provided or `None`, they are
        calculated using `coef_a(b_n.size)`, by default `None`.

    Returns
    -------
    npt.NDArray
        Calculated P function values. The shape will match `x` after broadcasting.

    References
    ----------
    Hapke (2002, Eqs. 23, 24).
    """
    if b_n is None:
        return x * 0 + 1  # P = 1 if no b_n coefficients are provided
    if a_n is None:
        a_n = coef_a(b_n.size - 1)  # Corrected size for coef_a

    p_n_2 = np.zeros_like(x) * x + 1
    p_n_1 = np.ones_like(x) * x
    p_n = np.empty_like(x)
    res = a_n[0] * b_n[0] + x * a_n[1] * b_n[1]
    for i in range(2, b_n.shape[0]):
        p_n = (2 - 1 / i) * x * p_n_1 - (1 - 1 / i) * p_n_2
        res += p_n * a_n[i] * b_n[i]
        p_n_2 = p_n_1
        p_n_1 = p_n
    res += 1
    return res


@jit(nogil=True, fastmath=True, cache=cache)
def value_p(
    b_n: npt.NDArray | None,
    a_n: npt.NDArray | None = None,
) -> float | np.floating:
    """Calculates the scalar value P from Hapke's model.

    This value is used in the expression for single particle phase function.

    Parameters
    ----------

    b_n : npt.NDArray
        Array of 'b_n' coefficients.
    a_n : npt.NDArray, optional
        Array of 'a_n' coefficients. If not provided or `None`, they are
        calculated using `coef_a(b_n.size)`, by default `None`.

    Returns
    -------
    float
        The calculated scalar value P.

    References
    ----------
    Hapke (2002, Eq. 25).
    """
    if b_n is None:
        return 1.0
    if a_n is None:
        a_n = coef_a(b_n.size - 1)  # Corrected size for coef_a
    return 1.0 + np.sum(a_n**2 * b_n)
