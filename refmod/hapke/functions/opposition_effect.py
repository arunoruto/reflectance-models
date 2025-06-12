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
    # b_cb = np.ones_like(x)
    b_cb = 0.0 * x + 1.0
    if (b0 != 0) and (h != 0):
        hc_2 = x / h
        bc = 0.5 * (1 + (1 - np.exp(-hc_2)) / hc_2) / (1 + hc_2) ** 2
        b_cb += b0 * bc
    return b_cb
