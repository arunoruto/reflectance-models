import numpy as np
import numpy.typing as npt
from numba import float64, vectorize
from refmod.config import cache


@vectorize([float64(float64, float64, float64)], target="cpu", cache=cache)
def shadow_hiding(
    x: npt.NDArray,
    h: float = 0.0,
    b0: float = 0.0,
) -> npt.NDArray:
    """Calculates the shadow hiding opposition effect term B_sh(g).

    Parameters
    ----------
    x : npt.NDArray
        Input parameter, typically related to phase angle, e.g., tan(alpha/2).
    h : float, optional
        Width of the opposition surge, by default 0.0.
    b0 : float, optional
        Amplitude of the opposition surge, by default 0.0.

    Returns
    -------
    npt.NDArray
        Shadow hiding term values.
    """
    # b_sh = np.ones_like(x)
    b_sh = 0.0 * x + 1.0
    if (b0 > 0.0) and (h > 0.0):
        b_sh += b0 / (1 + x / h)
    return b_sh


@vectorize([float64(float64, float64, float64)], target="cpu", cache=cache)
def coherant_backscattering(
    x: npt.NDArray,
    h: float = 0.0,
    b0: float = 0.0,
) -> npt.NDArray:
    """Calculates the coherent backscattering opposition effect term B_cb(g).

    Parameters
    ----------
    x : npt.NDArray
        Input parameter, typically related to phase angle, e.g., tan(alpha/2).
    h : float, optional
        Width of the coherent backscattering peak, by default 0.0.
    b0 : float, optional
        Amplitude of the coherent backscattering peak, by default 0.0.

    Returns
    -------
    npt.NDArray
        Coherent backscattering term values.
    """
    # b_cb = np.ones_like(x)
    b_cb = 0.0 * x + 1.0
    if (b0 != 0) and (h != 0):
        hc_2 = x / h
        bc = 0.5 * (1 + (1 - np.exp(-hc_2)) / hc_2) / (1 + hc_2) ** 2
        b_cb += b0 * bc
    return b_cb
