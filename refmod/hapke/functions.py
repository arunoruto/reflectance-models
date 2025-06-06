"""
??? info "References"

    1. Cornette, J. J., & Shanks, R. E. (1992). Bidirectional reflectance
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
    gamma = np.sqrt(1 - w)
    return (1 + 2 * x) / (1 + 2 * x * gamma)


def h_function_2(x: npt.NDArray, w: npt.NDArray) -> npt.NDArray:
    gamma = np.sqrt(1 - w)
    r0 = (1 - gamma) / (1 + gamma)
    h_inv = 1 - w * x * (r0 + (1 - 2 * r0 * x) / 2 * np.log(1 + 1 / x))
    return 1 / h_inv


def h_function_2_derivative(x: npt.NDArray, w: npt.NDArray) -> npt.NDArray:
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
    """
    Calculates the Hapke function for a given set of parameters.

    Args:
        x (float): The input parameter.
        w (numpy.ndarray): The weight array.
        level (int, optional): The level of the Hapke function to calculate. Defaults to 1.

    Returns:
        (float): The calculated Hapke function value.

    Raises:
        Exception: If an invalid level is provided.
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
    """
    Calculates the Hapke function for a given set of parameters.

    Args:
        x (float): The input parameter.
        w (numpy.ndarray): The weight array.
        level (int, optional): The level of the Hapke function to calculate. Defaults to 1.

    Returns:
        (float): The calculated Hapke function value.

    Raises:
        Exception: If an invalid level is provided.
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


def double_henyey_greenstein(cos_g, b: float = 0.21, c: float = 0.7):
    """
    Calculates the phase function for the double Henyey-Greenstein model.

    Args:
        cos_g (float): The cosine of the scattering angle.
        b (float, optional): The asymmetry parameter. Defaults to 0.21.
        c (float, optional): The backscatter fraction. Defaults to 0.7.

    Returns:
        (float): The phase function value.

    """
    return (
        (1 + c) / 2 * (1 - b**2) / np.power(1 - 2 * b * cos_g + b**2, 1.5)
        +  # NOTE: just for formatting
        (1 - c) / 2 * (1 - b**2) / np.power(1 + 2 * b * cos_g + b**2, 1.5)
    )


def cornette_shanks(cos_g, xi: float):
    """
    Calculates the Cornette-Shanks function.

    Args:
        cos_g (float): The cosine of the incidence angle.
        xi (float): The single scattering albedo.

    Returns:
        (float): The value of the Cornette-Shanks function.

    Note:
        Equation 8 from Cornette and Shanks (1992).

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
):
    match type:
        case "dhg" | "double_henyey_greenstein":
            return double_henyey_greenstein(cos_g, args[0], args[1])
        case "cs" | "cornette" | "cornette_shanks":
            return cornette_shanks(cos_g, args[0])
        case _:
            raise Exception("Unsupported phase function")


def normalize(x, axis: int = -1):
    temp = np.sum(x**2, axis=axis)
    if isinstance(temp, float):
        temp = np.array([temp])
    return np.sqrt(temp)


def normalize_keepdims(x, axis: int = -1):
    temp = np.sqrt(np.sum(x**2, axis=axis))
    if isinstance(temp, float):
        temp = np.array(temp)
        # temp = np.array([temp])
    return np.expand_dims(temp, axis=axis)


def angle_processing_base(vec_a: npt.NDArray, vec_b: npt.NDArray, axis: int = -1):
    cos_phi = np.sum(vec_a * vec_b, axis=axis)
    cos_phi = np.array([cos_phi]) if isinstance(cos_phi, float) else cos_phi
    cos_phi = np.clip(cos_phi, -1, 1)
    sin_phi = np.sqrt(1 - cos_phi**2)
    return cos_phi, sin_phi


def angle_processing(vec_a: npt.NDArray, vec_b: npt.NDArray, axis: int = -1):
    cos_phi, sin_phi = angle_processing_base(vec_a, vec_b, axis)
    cot_phi = np.where(sin_phi == 0, np.inf, cos_phi / sin_phi)
    i = np.arccos(cos_phi)
    return cos_phi, sin_phi, cot_phi, i
