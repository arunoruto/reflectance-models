"""
??? info "References"

    1. Hapke, B. (2002). Bidirectional Reflectance Spectroscopy: 5.
    The Coherent Backscatter Opposition Effect and Anisotropic Scattering.
    Icarus, 157(2), 523–534. <https://doi.org/10.1006/icar.2002.6853>
"""

from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.special import eval_legendre


def coef_a(n: int = 15):
    """
    Calculates the coefficients 'a_n' for the Legendre polynomial series.

    Args:
        n (int): The number of coefficients to calculate. Default is 15.

    Returns:
        (numpy.ndarray): An array of coefficients 'a_n' for the Legendre polynomial series.

    Note:
        Equation 27 in Hapke (2002).
    """
    a_n = np.zeros(n + 1)
    range = np.arange(n)
    a_n[1:] = -1 * eval_legendre(range, 0) / (range + 2)
    return a_n


def coef_b(b: float = 0.21, c: float = 0.7, n: int = 15):
    """
    Calculates the coefficients for the Hapke reflectance model Legendre polynomial expansion.

    Args:
        b (float, optional): The single scattering albedo. Defaults to 0.21.
        c (float, optional): The asymmetry factor. Defaults to 0.7.
        n (int, optional): The number of coefficients to calculate. Defaults to 15.

    Returns:
        (numpy.ndarray): The calculated coefficients for the Legendre polynomial expansion.

    Note:
        Equation on page 530 in Hapke (2002).
    """
    if np.isnan(c):
        range = np.arange(n + 1) + 1
        b_n = (2 * range + 1) * np.power(-b, range)
    else:
        range = np.arange(n + 1)
        b_n = c * (2 * range + 1) * np.power(b, range)
        # TODO: why is the first element one and not c?
        b_n[0] = 1
    return b_n


def function_p(
    x: npt.NDArray, b_n: npt.NDArray, a_n: npt.NDArray = np.empty(1) * np.nan
):
    """
    Calculates the P function using the Hapke reflectance model.

    Args:
        x (numpy.ndarray): The input array.
        b_n (numpy.ndarray): The B_n coefficients.
        a_n (npt.NDArray, optional): The A_n coefficients. Defaults to np.empty(1) * np.nan.

    Returns:
        (numpy.ndarray): The calculated P function.

    Note:
        Equations 23 and 24 in Hapke (2002).
    """
    n = np.arange(b_n.size)
    if np.any(np.isnan(a_n)):
        a_n = coef_a(b_n.size)
    # match x.ndim:
    #     case 1:
    #         x = np.expand_dims(x, axis=1)
    #     case 2:
    #         x = np.expand_dims(x, axis=2)
    x = np.expand_dims(x, axis=-1)
    return 1 + np.sum(a_n * b_n * eval_legendre(n, x), axis=-1)
    # return 1 + np.sum(a_n * b_n * eval_legendre(n, x), axis=2)


def value_p(b_n: npt.NDArray, a_n: npt.NDArray = np.empty(1) * np.nan):
    """
    Calculates the value of the P function.

    Args:
        b_n (numpy.ndarray): Array of coefficients.
        a_n (npt.NDArray, optional): Array of coefficients. Defaults to np.empty(1) * np.nan.

    Returns:
        (float): The calculated value of the P function.

    Note:
        Equations 25 in Hapke (2002).
    """
    if np.any(np.isnan(a_n)):
        a_n = coef_a(b_n.size)
    return 1 + np.sum(a_n**2 * b_n)
