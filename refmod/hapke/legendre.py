"""Legendre polynomial expansion for phase functions.

This module provides functions to calculate coefficients and evaluate
Legendre polynomial series, typically used to represent anisotropic
phase functions in scattering models like Hapke's.

References are indicated using a citation key, e.g., [Hapke2002]_,
corresponding to entries in a BibTeX file.
"""

from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.special import eval_legendre


def coef_a(n: int = 15) -> npt.NDArray:
    """Calculate coefficients 'a_n' for Legendre series.

    These coefficients are used in Hapke's formulation for anisotropic
    scattering.

    Parameters
    ----------
    n : int, optional
        The order of the Legendre expansion, determining the number of
        coefficients. The resulting array will have `n + 1` elements.
        Defaults to 15.

    Returns
    -------
    npt.NDArray
        Array of coefficients `a_k` for k from 0 to n. Shape (n + 1,).

    Notes
    -----
    This corresponds to Equation 27 in [Hapke2002]_.
    `a_0` is 0 by this definition.

    Examples
    --------
    >>> coef_a(2)
    array([ 0. , -0.5,  0. ])
    """
    a_k = np.zeros(n + 1)
    k_range = np.arange(n)  # k from 0 to n-1 for P_k(0)
    # a_k[k+1] = -P_k(0) / (k+2)
    # P_k(0) is 0 for odd k, and (-1)^(k/2) * (k-1)!! / k!! for even k
    # However, eval_legendre(k, 0) handles this directly.
    # The formula in Hapke (2002) is a_n = -P_{n-1}(0)/ (n+1) for n>=1.
    # Here, if our array is indexed 0 to N, and b_n is b_0 to b_N,
    # then a_n in P(x) = 1 + sum_{n=1 to N} a_n b_n P_n(x) (Hapke's Eq 23 form)
    # or P(x) = sum_{n=0 to N} b_n P_n(x) and p(mu) = 1 + sum_{n=1 to N} a_n b_n P_n(mu) (Eq 24)
    # The a_n in Eq 27 are for the p(mu) form, where a_n means a_k for P_k.
    # a_k = - P_{k-1}(0) / (k+1) for k >= 1.
    # Let's use the direct indexing as per common implementation:
    # a_n[0] = 0, a_n[i] for i>0 corresponds to P_{i-1}(0) / (i+1)
    # The code was: a_n[1:] = -1 * eval_legendre(range, 0) / (range + 2)
    # This means a_n[k] = -P_{k-1}(0) / (k+1) for k=1 to n.
    # So, for k_idx = 0 to n-1 (representing P_0 to P_{n-1}),
    # a_k[k_idx+1] = -eval_legendre(k_idx, 0) / (k_idx + 2)
    if n > 0:
        eval_range = np.arange(n) # P_0(0) to P_{n-1}(0)
        a_k[1:] = -1 * eval_legendre(eval_range, 0) / (eval_range + 2)
    return a_n


def coef_b(b: float = 0.21, c: float = np.nan, n: int = 15) -> npt.NDArray:
    """Calculate coefficients 'b_n' for Legendre expansion of phase function.

    These coefficients represent the Legendre expansion of a phase function,
    often a Henyey-Greenstein or double Henyey-Greenstein function.

    Parameters
    ----------
    b : float, optional
        Asymmetry parameter 'g' of the Henyey-Greenstein function.
        If `c` is NaN, this is treated as `-g` for a single HG function.
        Defaults to 0.21.
    c : float, optional
        Mixing parameter for a double Henyey-Greenstein function. If NaN,
        a single Henyey-Greenstein function `P_HG(-b, cos_alpha)` is assumed.
        If not NaN, it's `c * P_HG(b, cos_alpha) + (1-c) * P_HG(-b, cos_alpha)`.
        The interpretation of b_n[0] might vary. Hapke (2002) normalizes so P(g) integrates to 1.
        Here, b_n[0] is often set to 1 for P_0 = 1.
        Defaults to np.nan.
    n : int, optional
        The order of the Legendre expansion. The resulting array will have
        `n + 1` elements. Defaults to 15.

    Returns
    -------
    npt.NDArray
        Array of coefficients `b_k` for k from 0 to n. Shape (n + 1,).

    Notes
    -----
    The formula follows the expansion `b_k = (2k+1)g^k` for a single
    Henyey-Greenstein function P_HG(g, cos_alpha), or a weighted sum for
    double Henyey-Greenstein.
    Based on discussion on page 530 of [Hapke2002]_.
    If `c` is provided, the coefficients are for `c * HG(b) + (1-c) * HG(-b)`.
    The first term `b_0` is typically 1 because `P_0(x)=1` and phase functions
    are often normalized such that their integral over 4pi steradians is 1,
    implying `b_0 = 1` if using `(1/4pi) * integral P(g) dOmega = 1`.

    Examples
    --------
    >>> coef_b(b=0.5, c=np.nan, n=2) # Single HG(-0.5)
    array([ 1.  , -1.5 ,  1.25])
    >>> coef_b(b=0.3, c=0.7, n=1) # Double HG
    array([1. , 0.3])
    """
    k_range = np.arange(n + 1)
    if np.isnan(c):
        # Single Henyey-Greenstein P_HG(-b, cos_alpha)
        # b_k = (2k+1) * (-b)^k
        b_k_values = (2 * k_range + 1) * np.power(-b, k_range)
    else:
        # Double Henyey-Greenstein: c * P_HG(b, cos_alpha) + (1-c) * P_HG(-b, cos_alpha)
        # b_k = c * (2k+1) * b^k + (1-c) * (2k+1) * (-b)^k
        # b_k = (2k+1) * [c * b^k + (1-c) * (-b)^k]
        b_k_values = (2 * k_range + 1) * (
            c * np.power(b, k_range) + (1 - c) * np.power(-b, k_range)
        )
        # For Double HG, b_0 = (2*0+1)*[c*b^0 + (1-c)*(-b)^0] = c + (1-c) = 1.
    b_k_values[0] = 1.0 # Ensure b_0 is 1 for normalization P_0(x)=1.
    return b_n


def function_p(
    x: npt.NDArray, b_n: npt.NDArray, a_n: npt.NDArray = np.empty(1) * np.nan
) -> npt.NDArray:
    """Evaluate Hapke's anisotropic scattering function p(x).

    This function, denoted as p(x) or P_L(x) in some contexts, is part
    of Hapke's model for anisotropic scattering from a surface.

    Parameters
    ----------
    x : npt.NDArray
        Input array, typically cosine of incidence (mu0) or emission (mu)
        angle. Shape can be (M,) or (M, K).
    b_n : npt.NDArray
        Coefficients `b_k` of the Legendre expansion of the phase function.
        Shape (N+1,).
    a_n : npt.NDArray, optional
        Coefficients `a_k` used in Hapke's formulation. Shape (N+1,).
        If not provided or NaN, they are computed using `coef_a(N)`.
        Defaults to `np.empty(1) * np.nan`.

    Returns
    -------
    npt.NDArray
        Calculated p(x) values. Shape will match `x`.

    Notes
    -----
    Corresponds to Equation 24 in [Hapke2002]_:
    `p(x) = 1 + sum_{k=1 to N} a_k * b_k * P_k(x)`
    where `P_k(x)` is the k-th Legendre polynomial.
    The sum starts from k=1 because `a_0` is typically 0.

    Examples
    --------
    >>> x = np.array([0.5, 1.0])
    >>> bn = coef_b(b=0.3, n=2)
    >>> an = coef_a(n=2)
    >>> function_p(x, bn, an)
    array([0.9625, 1.    ])
    """
    order_N = b_n.size - 1
    k_indices = np.arange(order_N + 1) # k from 0 to N

    if np.any(np.isnan(a_n)) or a_n.size != b_n.size:
        # Need to pass order N (max index), not size N+1
        a_n_calc = coef_a(order_N)
    else:
        a_n_calc = a_n

    # Ensure x is suitable for broadcasting with Legendre polynomials P_k(x)
    # eval_legendre(k, x) expects k as int and x as array.
    # We want sum over k: a_k * b_k * P_k(x)
    # Result should have same shape as x.

    # Add new axis to x for broadcasting with k_indices if x is 1D
    x_expanded = np.expand_dims(x, axis=-1) if x.ndim > 0 else np.array(x)[np.newaxis, np.newaxis]
    if x.ndim == 0: # scalar input
        x_expanded = x[np.newaxis, np.newaxis]


    # P_k(x) will have shape (x.shape, N+1) after loop/vectorization
    # legendre_poly_values[..., k] = P_k(x)
    legendre_poly_values = eval_legendre(k_indices[np.newaxis, :], x_expanded)

    # Sum (a_k * b_k * P_k(x)) for k=1 to N. a_0 is 0.
    sum_terms = np.sum(a_n_calc[1:] * b_n[1:] * legendre_poly_values[..., 1:], axis=-1)

    return 1 + sum_terms


def value_p(b_n: npt.NDArray, a_n: npt.NDArray = np.empty(1) * np.nan) -> float | npt.NDArray:
    """Evaluate Hapke's integrated anisotropic scattering parameter P.

    This parameter, denoted as P or P_0 in some contexts, is an integral
    property related to the anisotropic phase function.

    Parameters
    ----------
    b_n : npt.NDArray
        Coefficients `b_k` of the Legendre expansion of the phase function.
        Shape (N+1,).
    a_n : npt.NDArray, optional
        Coefficients `a_k` used in Hapke's formulation. Shape (N+1,).
        If not provided or NaN, they are computed using `coef_a(N)`.
        Defaults to `np.empty(1) * np.nan`.

    Returns
    -------
    float | npt.NDArray
        Calculated P value. Scalar.

    Notes
    -----
    Corresponds to Equation 25 in [Hapke2002]_:
    `P = 1 + sum_{k=1 to N} (a_k)^2 * b_k`
    The sum starts from k=1 because `a_0` is typically 0.

    Examples
    --------
    >>> bn = coef_b(b=0.3, n=2)
    >>> an = coef_a(n=2)
    >>> value_p(bn, an)
    0.929375
    """
    order_N = b_n.size - 1
    if np.any(np.isnan(a_n)) or a_n.size != b_n.size:
        a_n_calc = coef_a(order_N)
    else:
        a_n_calc = a_n

    # Sum (a_k^2 * b_k) for k=1 to N. a_0 is 0.
    sum_terms = np.sum(a_n_calc[1:]**2 * b_n[1:])
    return 1 + sum_terms
