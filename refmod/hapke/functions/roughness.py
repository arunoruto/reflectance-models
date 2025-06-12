import numpy as np
import numpy.typing as npt
from numba import float64, jit, vectorize
from refmod.config import cache

from .vectors import angle_processing, normalize_keepdims


@vectorize([float64(float64, float64)], target="cpu", cache=cache)
def __f_exp(x: npt.NDArray, y: float) -> npt.NDArray:
    """Helper function for microscopic roughness calculation.

    Calculates `exp(-2 * y * x / pi)`.

    Parameters
    ----------

    x : npt.NDArray
        Input array.
    y : float
        Factor, typically related to cot(roughness).

    Returns
    -------
    npt.NDArray
        Result of the exponential function.
    """
    return np.exp(-2 / np.pi * y * x)


@vectorize([float64(float64, float64)], target="cpu", cache=cache)
def __f_exp_2(x: npt.NDArray, y: float) -> npt.NDArray:
    """Helper function for microscopic roughness calculation.

    Calculates `exp(-(y^2 * x^2) / pi)`.

    Parameters
    ----------

    x : npt.NDArray
        Input array.
    y : float
        Factor, typically related to cot(roughness), which is squared.

    Returns
    -------
    npt.NDArray
        Result of the exponential function.
    """
    return np.exp(-(y**2) * x**2 / np.pi)


@vectorize(
    [
        float64(
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
        )
    ],
    target="cpu",
    cache=cache,
)
def __prime_term(
    cos_x: npt.NDArray,
    sin_x: npt.NDArray,
    cot_r: float,
    cos_psi: npt.NDArray,
    sin_psi_div_2_sq: npt.NDArray,
    psi: npt.NDArray,
    cot_a: npt.NDArray,
    cot_b: npt.NDArray,
    cot_c: npt.NDArray,
    cot_d: npt.NDArray,
    index: npt.NDArray,
):
    temp = cos_x + sin_x / cot_r * (
        cos_psi * __f_exp_2(cot_a, cot_r) + sin_psi_div_2_sq * __f_exp_2(cot_b, cot_r)
    ) / (2 - __f_exp(cot_c, cot_r) - psi / np.pi * __f_exp(cot_d, cot_r))
    return temp * index


# @vectorize([float64(float64, float64, float64, float64)], target="cpu", cache=cache)
@jit(nogil=True, fastmath=True, cache=cache)
def microscopic_roughness(
    roughness: float,
    incidence_direction: npt.NDArray,
    emission_direction: npt.NDArray,
    surface_orientation: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    r"""Calculates the microscopic roughness factor for Hapke's model.

    This correction accounts for the effects of sub-resolution roughness on
    the observed reflectance.

    Parameters
    ----------

    roughness : float
        The mean slope angle of surface facets, in radians.
        A value of 0 means a smooth surface.
    incidence_direction : npt.NDArray
        Incidence direction vector(s), shape (..., 3). Assumed to be normalized.
    emission_direction : npt.NDArray
        Emission direction vector(s), shape (..., 3). Assumed to be normalized.
    surface_orientation : npt.NDArray
        Surface normal vector(s), shape (..., 3). Assumed to be normalized.

    Returns
    -------
    s : npt.NDArray
        The microscopic roughness factor, shape (...).
    mu_0_prime : npt.NDArray
        The modified cosine of the incidence angle ($\mu_0^{\prime}$), accounting
        for roughness, shape (...).
    mu_prime : npt.NDArray
        The modified cosine of the emission angle ($\mu^{\prime}$), accounting
        for roughness, shape (...).

    Notes
    -----
    The calculations are based on Hapke (1984).

    - The terms $\mu_0^{\prime}$ (mu_0_s0, mu_0_s) and $\mu^{\prime}$ (mu_s0, mu_s)
      are calculated based on different conditions for incidence angle `i`
      and emission angle `e`:

      - For prime-zero terms ($\mu_0^{\prime(0)}$, $\mu^{\prime(0)}$ used in `mu_0_s0`, `mu_s0`):
        See Hapke (1984, Eqs. 48, 49).
      - For $\mu_0^{\prime}$ and $\mu^{\prime}$ when $i < e$:
        See Hapke (1984, Eqs. 46, 47).
      - For $\mu_0^{\prime}$ and $\mu^{\prime}$ when $i \ge e$:
        See Hapke (1984, Eqs. 50, 51).

    - Input vectors (`incidence_direction`, `emission_direction`, `surface_orientation`)
      are normalized internally.
    - If `roughness` is 0, `s` is 1, `mu_0_prime` is `cos(i)`, and `mu_prime` is `cos(e)`.

    References
    ----------
    Hapke (1984)

    """
    original_shape = surface_orientation.shape[1:]
    incidence_direction = np.ascontiguousarray(incidence_direction).reshape(3, -1)
    emission_direction = np.ascontiguousarray(emission_direction).reshape(3, -1)
    surface_orientation = np.ascontiguousarray(surface_orientation).reshape(3, -1)
    # original_shape = surface_orientation.shape[:-1]
    # incidence_direction = np.ascontiguousarray(incidence_direction).reshape(-1, 3)
    # emission_direction = np.ascontiguousarray(emission_direction).reshape(-1, 3)
    # surface_orientation = np.ascontiguousarray(surface_orientation).reshape(-1, 3)

    # Angles
    # incidence_direction /= normalize_keepdims(incidence_direction)
    # emission_direction /= normalize_keepdims(emission_direction)
    # surface_orientation /= normalize_keepdims(surface_orientation)

    # Incidence angle
    cos_i, sin_i, cot_i, i = angle_processing(
        incidence_direction, surface_orientation, 0
    )
    # Emission angle
    cos_e, sin_e, cot_e, e = angle_processing(
        emission_direction, surface_orientation, 0
    )

    if roughness == 0:
        print("Roughness is zero, returning default values")
        return (
            np.ones(original_shape),
            cos_i.reshape(original_shape),
            cos_e.reshape(original_shape),
        )
        # return np.ones_like(cos_e), cos_i, cos_e

    # Projections
    projection_incidence = (
        incidence_direction - np.expand_dims(cos_i, axis=0) * surface_orientation
    )
    projection_emission = (
        emission_direction - np.expand_dims(cos_e, axis=0) * surface_orientation
    )
    # projection_incidence_norm = np.linalg.norm(projection_incidence, axis=-1)
    # projection_emission_norm = np.linalg.norm(projection_emission, axis=-1)
    # projection_incidence_norm = normalize(incidence_direction)
    # projection_emission_norm = normalize(projection_emission)
    projection_incidence /= normalize_keepdims(projection_incidence)
    projection_emission /= normalize_keepdims(projection_emission)

    # Azicos_eth angle
    cos_psi, sin_psi, _, psi = angle_processing(
        projection_incidence, projection_emission, 0
    )
    sin_psi_div_2_sq = np.abs(0.5 - cos_psi / 2)
    # psi = np.arccos(cos_psi)

    # Macroscopic Roughness
    cot_rough = 1 / np.tan(roughness)
    # Check for cases
    ile = i < e
    ige = i >= e
    # Check for singularities
    index_cos_e0 = 1 == cos_i
    index_cos_e = 1 == cos_e

    factor = 1 / np.sqrt(1 + np.pi / cot_rough**2)
    f_psi = np.exp(
        -2 * sin_psi / (1 + cos_psi)
        # * np.divide(
        #     sin_psi,
        #     1 + cos_psi,
        #     out=np.ones_like(sin_psi) * -1,
        #     where=cos_psi != -1,
        # )
    )

    cos_i_s0 = factor * (
        cos_i
        + sin_i
        / cot_rough
        * __f_exp_2(cot_i, cot_rough)
        / (2.0 - __f_exp(cot_i, cot_rough))
    )
    cos_e_s0 = factor * (
        cos_e
        + sin_e
        / cot_rough
        * __f_exp_2(cot_e, cot_rough)
        / (2.0 - __f_exp(cot_e, cot_rough))
    )

    cos_i_s = np.zeros_like(cos_i)
    cos_i_s += factor * __prime_term(
        cos_i,
        sin_i,
        cot_rough,
        cos_psi,
        sin_psi_div_2_sq,
        psi,
        cot_e,
        cot_i,
        cot_e,
        cot_i,
        ile,
    )
    cos_i_s += factor * __prime_term(
        cos_i,
        sin_i,
        cot_rough,
        np.ones_like(cos_psi),
        -sin_psi_div_2_sq,
        psi,
        cot_i,
        cot_e,
        cot_i,
        cot_e,
        ige,
    )
    cos_i_s[index_cos_e0 | index_cos_e] = cos_i[index_cos_e0 | index_cos_e]

    cos_e_s = np.zeros_like(cos_e)
    cos_e_s += factor * __prime_term(
        cos_e,
        sin_e,
        cot_rough,
        np.ones_like(cos_psi),
        -sin_psi_div_2_sq,
        psi,
        cot_e,
        cot_i,
        cot_e,
        cot_i,
        ile,
    )
    cos_e_s += factor * __prime_term(
        cos_e,
        sin_e,
        cot_rough,
        cos_psi,
        sin_psi_div_2_sq,
        psi,
        cot_i,
        cot_e,
        cot_i,
        cot_e,
        ige,
    )
    cos_e_s[index_cos_e0 | index_cos_e] = cos_e[index_cos_e0 | index_cos_e]

    s = factor * (cos_e_s / cos_e_s0) * (cos_i / cos_i_s0)
    s[ile] /= 1 + f_psi[ile] * (factor * (cos_i[ile] / cos_i_s0[ile]) - 1)
    s[ige] /= 1 + f_psi[ige] * (factor * (cos_e[ige] / cos_e_s0[ige]) - 1)
    s[index_cos_e0 | index_cos_e] = 1

    return (
        s.reshape(original_shape),
        cos_i_s.reshape(original_shape),
        cos_e_s.reshape(original_shape),
    )
