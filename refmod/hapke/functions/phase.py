from typing import Literal

import numpy as np
import numpy.typing as npt
from numba import float64, jit, vectorize
from refmod.config import cache

PhaseFunctionType = Literal[
    "dhg",
    "double_henyey_greenstein",
    "cs",
    "cornette",
    "cornette_shanks",
]


@vectorize([float64(float64, float64, float64)], target="cpu", cache=cache)
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


@vectorize([float64(float64, float64)], target="cpu", cache=cache)
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


@jit(nogil=True, fastmath=True, cache=cache)
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
