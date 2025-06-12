import numpy as np
import numpy.typing as npt
from numba import float64, jit, vectorize
from refmod.config import cache


@vectorize([float64(float64, float64)], target="cpu", cache=cache)
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


@vectorize([float64(float64, float64)], target="cpu", cache=cache)
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


@vectorize([float64(float64, float64)], target="cpu", cache=cache)
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


@jit(nogil=True, fastmath=True, cache=cache)
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

    # if w.ndim == x.ndim + 1:
    #     # x = np.broadcast_to(x, w.shape)
    #     x = np.expand_dims(x, axis=-1)

    match level:
        case 1:
            h = h_function_1(x, w)
        case 2:
            h = h_function_2(x, w)
        case _:
            raise Exception("Please provide a level between 1 and 2!")

    return h


@jit(nogil=True, fastmath=True, cache=cache)
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
