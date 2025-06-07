"""
??? info "References"

    1. Cornette and Shanks (1992). Bidirectional reflectance
    of flat, optically thick particulate systems. Applied Optics, 31(15),
    3152-3160. <https://doi.org/10.1364/AO.31.003152>
"""

from typing import Literal

import numpy as np
import numpy.typing as npt

PhaseFunctionType = Literal[
    "dhg",
    "double_henyey_greenstein",
    "cs",
    "cornette",
    "cornette_shanks",
]


def h_function_1(x: npt.NDArray, w: npt.NDArray) -> npt.NDArray:
    """Calculates the H-function (level 1).

    Parameters
    ----------

    x : npt.NDArray
        Input parameter.
    w : npt.NDArray
        Single scattering albedo.

    Returns
    -------
    npt.NDArray
        H-function values.

    References
    ----------
    Hapke (1993, p. 121, Eq. 8.31a).
    """
    gamma = np.sqrt(1 - w)
    return (1 + 2 * x) / (1 + 2 * x * gamma)


def h_function_2(x: npt.NDArray, w: npt.NDArray) -> npt.NDArray:
    """Calculates the H-function (level 2).

    Parameters
    ----------

    x : npt.NDArray
        Input parameter, often mu or mu_0 (cosine of angles).
    w : npt.NDArray
        Single scattering albedo.

    Returns
    -------
    npt.NDArray
        H-function values.

    References
    ----------
    Cornette and Shanks (1992)
    """
    gamma = np.sqrt(1 - w)
    r0 = (1 - gamma) / (1 + gamma)
    h_inv = 1 - w * x * (r0 + (1 - 2 * r0 * x) / 2 * np.log(1 + 1 / x))
    return 1 / h_inv


def h_function_2_derivative(x: npt.NDArray, w: npt.NDArray) -> npt.NDArray:
    """Calculates the derivative of the H-function (level 2) with respect to w.

    Parameters
    ----------

    x : npt.NDArray
        Input parameter, often mu or mu_0 (cosine of angles).
    w : npt.NDArray
        Single scattering albedo.

    Returns
    -------
    npt.NDArray
        Derivative of the H-function (level 2) with respect to w.
    """
    gamma = np.sqrt(1 - w)
    r0 = (1 - gamma) / (1 + gamma)
    x_log_term = np.log(1 + 1 / x)

    dr0_dw = 1 / (gamma * (1 + gamma) ** 2)
    h = h_function_2(x, w)
    return (
        h**2
        * x
        * (r0 + (1 - 2 * r0 * x) / 2 * x_log_term + w * dr0_dw * (1 - x * x_log_term))
    )


def h_function(x: npt.NDArray, w: npt.NDArray, level: int = 1) -> npt.NDArray:
    """Calculates the Hapke H-function.

    This function can compute two different versions (levels) of the H-function.

    Parameters
    ----------

    x : npt.NDArray
        Input parameter, often mu or mu_0 (cosine of angles).
    w : npt.NDArray
        Single scattering albedo.
    level : int, optional
        Level of the H-function to calculate (1 or 2), by default 1.
        Level 1 refers to `h_function_1`.
        Level 2 refers to `h_function_2`.

    Returns
    -------
    npt.NDArray
        Calculated H-function values.

    Raises
    ------
    Exception
        If an invalid level (not 1 or 2) is provided.
    """

    match level:
        case 1:
            h = h_function_1(x, w)
        case 2:
            h = h_function_2(x, w)
        case _:
            raise Exception("Please provide a level between 1 and 2!")

    return h


def h_function_derivative(
    x: npt.NDArray, w: npt.NDArray, level: int = 1
) -> npt.NDArray:
    """Calculates the derivative of the Hapke H-function with respect to w.

    This function can compute the derivative for two different versions (levels)
    of the H-function.

    Parameters
    ----------

    x : npt.NDArray
        Input parameter, often mu or mu_0 (cosine of angles).
    w : npt.NDArray
        Single scattering albedo.
    level : int, optional
        Level of the H-function derivative to calculate (1 or 2), by default 1.
        Level 1 derivative is not implemented.
        Level 2 refers to `h_function_2_derivative`.

    Returns
    -------
    npt.NDArray
        Calculated H-function derivative values.

    Raises
    ------
    NotImplementedError
        If level 1 is selected, as its derivative is not implemented.
    Exception
        If an invalid level (not 1 or 2) is provided.
    """

    match level:
        case 1:
            dh_dw = np.zeros_like(x)
            raise NotImplementedError(
                "The derivative for level 1 is not implemented. Please use level 2!"
            )
        case 2:
            dh_dw = h_function_2_derivative(x, w)
        case _:
            raise Exception("Please provide a level between 1 and 2!")

    return dh_dw


def double_henyey_greenstein(
    cos_g: npt.NDArray, b: float = 0.21, c: float = 0.7
) -> npt.NDArray:
    """Calculates the Double Henyey-Greenstein phase function.

    Parameters
    ----------

    cos_g : npt.NDArray
        Cosine of the scattering angle (g).
    b : float, optional
        Asymmetry parameter, by default 0.21.
    c : float, optional
        Backscatter fraction, by default 0.7.

    Returns
    -------
    npt.NDArray
        Phase function values.
    """
    return (
        (1 + c) / 2 * (1 - b**2) / np.power(1 - 2 * b * cos_g + b**2, 1.5)
        +  # NOTE: just for formatting
        (1 - c) / 2 * (1 - b**2) / np.power(1 + 2 * b * cos_g + b**2, 1.5)
    )


def cornette_shanks(cos_g: npt.NDArray, xi: float) -> npt.NDArray:
    """Calculates the Cornette-Shanks phase function.

    Parameters
    ----------

    cos_g : npt.NDArray
        Cosine of the scattering angle (g).
    xi : float
        Asymmetry parameter, related to the average scattering angle.
        Note: This `xi` is different from the single scattering albedo `w`.

    Returns
    -------
    npt.NDArray
        Phase function values.

    References
    ----------
    Cornette and Shanks (1992, Eq. 8).
    """
    return (
        1.5
        * (1 - xi**2)
        / (2 + xi**2)
        * (1 + cos_g**2)
        / np.power(1 + xi**2 - 2 * xi * cos_g, 1.5)
    )


def phase_function(
    cos_g: npt.NDArray,
    type: PhaseFunctionType,
    args: tuple,
) -> npt.NDArray:
    """Selects and evaluates a phase function.

    Parameters
    ----------

    cos_g : npt.NDArray
        Cosine of the scattering angle (g).
    type : PhaseFunctionType
        Type of phase function to use.
        Valid options are:
        - "dhg" or "double_henyey_greenstein": Double Henyey-Greenstein
        - "cs" or "cornette" or "cornette_shanks": Cornette-Shanks
    args : tuple
        Arguments for the selected phase function.
        - For "dhg": (b, c) where b is asymmetry and c is backscatter fraction.
        - For "cs": (xi,) where xi is the Cornette-Shanks asymmetry parameter.

    Returns
    -------
    npt.NDArray
        Calculated phase function values.

    Raises
    ------
    Exception
        If an unsupported `type` is provided.
    """
    match type:
        case "dhg" | "double_henyey_greenstein":
            return double_henyey_greenstein(cos_g, args[0], args[1])
        case "cs" | "cornette" | "cornette_shanks":
            return cornette_shanks(cos_g, args[0])
        case _:
            raise Exception("Unsupported phase function")


def normalize(x: npt.NDArray, axis: int = -1) -> npt.NDArray:
    """Normalizes a vector or a batch of vectors.

    Calculates the L2 norm (Euclidean norm) of the input array along the
    specified axis.

    Parameters
    ----------

    x : npt.NDArray
        Input array representing a vector or a batch of vectors.
    axis : int, optional
        Axis along which to compute the norm, by default -1.

    Returns
    -------
    npt.NDArray
        The L2 norm of the input array. If `x` is a batch of vectors,
        the output will be an array of norms.
    """
    temp = np.sum(x**2, axis=axis)
    if isinstance(temp, float):
        temp = np.array([temp])
    return np.sqrt(temp)


def normalize_keepdims(x: npt.NDArray, axis: int = -1) -> npt.NDArray:
    """Normalizes a vector or batch of vectors, keeping dimensions.

    Calculates the L2 norm of the input array along the specified axis,
    then expands the dimensions of the output to match the input array's
    dimension along the normalization axis. This is useful for broadcasting
    the norm for division.

    Parameters
    ----------

    x : npt.NDArray
        Input array representing a vector or a batch of vectors.
    axis : int, optional
        Axis along which to compute the norm, by default -1.

    Returns
    -------
    npt.NDArray
        The L2 norm of the input array, with dimensions kept for broadcasting.
    """
    temp = np.sqrt(np.sum(x**2, axis=axis))
    if isinstance(temp, float):
        temp = np.array(temp)
        # temp = np.array([temp])
    return np.expand_dims(temp, axis=axis)


def angle_processing_base(
    vec_a: npt.NDArray, vec_b: npt.NDArray, axis: int = -1
) -> tuple[npt.NDArray, npt.NDArray]:
    """Computes cosine and sine of the angle between two vectors.

    Parameters
    ----------

    vec_a : npt.NDArray
        First vector or batch of vectors.
    vec_b : npt.NDArray
        Second vector or batch of vectors. Must have the same shape as vec_a.
    axis : int, optional
        Axis along which the dot product is performed, by default -1.

    Returns
    -------
    tuple[npt.NDArray, npt.NDArray]
        A tuple containing:
            - cos_phi : npt.NDArray
                Cosine of the angle(s) between vec_a and vec_b.
            - sin_phi : npt.NDArray
                Sine of the angle(s) between vec_a and vec_b.
    """
    cos_phi = np.sum(vec_a * vec_b, axis=axis)
    cos_phi = np.array([cos_phi]) if isinstance(cos_phi, float) else cos_phi
    cos_phi = np.clip(cos_phi, -1, 1)
    sin_phi = np.sqrt(1 - cos_phi**2)
    return cos_phi, sin_phi


def angle_processing(
    vec_a: npt.NDArray, vec_b: npt.NDArray, axis: int = -1
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Computes various trigonometric quantities related to the angle between two vectors.

    Parameters
    ----------

    vec_a : npt.NDArray
        First vector or batch of vectors.
    vec_b : npt.NDArray
        Second vector or batch of vectors. Must have the same shape as vec_a.
    axis : int, optional
        Axis along which the dot product is performed, by default -1.

    Returns
    -------
    tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]
        A tuple containing:
            - cos_phi : npt.NDArray
                Cosine of the angle(s) between vec_a and vec_b.
            - sin_phi : npt.NDArray
                Sine of the angle(s) between vec_a and vec_b.
            - cot_phi : npt.NDArray
                Cotangent of the angle(s) between vec_a and vec_b.
                (Returns np.inf where sin_phi is 0).
            - i : npt.NDArray
                The angle(s) in radians between vec_a and vec_b (i.e., arccos(cos_phi)).
    """
    cos_phi, sin_phi = angle_processing_base(vec_a, vec_b, axis)
    cot_phi = np.where(sin_phi == 0, np.inf, cos_phi / sin_phi)
    i = np.arccos(cos_phi)
    return cos_phi, sin_phi, cot_phi, i
