import numpy as np
import numpy.typing as npt
from numba import float64, guvectorize, jit, vectorize
from refmod.config import EPS, cache

from .vectors import angle_processing, dot0, normalize_keepdims


@vectorize([float64(float64, float64)], target="cpu", cache=cache)
def __f_exp(x: npt.NDArray, y: float) -> npt.NDArray | float:
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
    if np.isinf(x):
        return 0.0
    return np.exp(-2 / np.pi * y * x)


@vectorize([float64(float64, float64)], target="cpu", cache=cache)
def __f_exp_2(x: npt.NDArray, y: float) -> npt.NDArray | float:
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
    if np.isinf(x):
        return 0.0
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
        )
    ],
    target="cpu",
    cache=cache,
)
def __prime_term(
    cos_x: npt.NDArray,
    sin_x: npt.NDArray,
    cot_r: float,
    cos_psi: npt.NDArray | float,
    sin_psi_div_2_sq: npt.NDArray,
    cot_a: npt.NDArray,
    cot_b: npt.NDArray,
    cot_c: npt.NDArray,
    cot_d: npt.NDArray,
    index: npt.NDArray,
):
    psi = np.arccos(cos_psi)
    temp = cos_x + sin_x / cot_r * (
        cos_psi * __f_exp_2(cot_a, cot_r) + sin_psi_div_2_sq * __f_exp_2(cot_b, cot_r)
    ) / (2 - __f_exp(cot_c, cot_r) - psi / np.pi * __f_exp(cot_d, cot_r))
    return temp * index


@vectorize([float64(float64, float64, float64)], target="cpu", cache=cache)
def __cos_s0(
    factor: float,
    cos_x: npt.NDArray,
    cot_rough: float,
):
    sin_x = np.sqrt(1 - cos_x**2)
    cot_x = cos_x / sin_x
    return factor * (
        cos_x
        + sin_x
        / cot_rough
        * __f_exp_2(cot_x, cot_rough)
        / (2.0 - __f_exp(cot_x, cot_rough))
    )


@vectorize(
    [float64(float64, float64, float64, float64, float64, float64, float64)],
    target="cpu",
    cache=cache,
)
def __cos_s(
    factor: float,
    cos_x: npt.NDArray,
    cos_y: npt.NDArray,
    cos_psi: npt.NDArray,
    cot_rough: float,
    ile: npt.NDArray,
    ige: npt.NDArray,
):
    if cos_x == 1.0 or cos_y == 1.0:
        return cos_x

    cos_x_s = cos_x * 0.0
    sin_psi_div_2_sq = np.abs(0.5 - cos_psi / 2)

    cot_x = cos_x / np.sqrt(1 - cos_x**2)
    sin_x = np.sqrt(1 - cos_x**2)
    cot_y = cos_y / np.sqrt(1 - cos_y**2)

    cos_x_s += factor * __prime_term(
        cos_x,
        sin_x,
        cot_rough,
        cos_psi,
        sin_psi_div_2_sq,
        cot_y,
        cot_x,
        cot_y,
        cot_x,
        ile,
    )
    cos_x_s += factor * __prime_term(
        cos_x,
        sin_x,
        cot_rough,
        # np.ones_like(cos_psi),
        1.0,
        -sin_psi_div_2_sq,
        cot_x,
        cot_y,
        cot_x,
        cot_y,
        ige,
    )
    return cos_x_s


@vectorize(
    [float64(float64, float64, float64, float64, float64, float64, float64, float64)],
    target="cpu",
    cache=cache,
)
def __roughness(
    factor: float,
    cos_i: npt.NDArray,
    cos_i_s0: npt.NDArray,
    cos_e: npt.NDArray | float,
    cos_e_s: npt.NDArray,
    cos_e_s0: npt.NDArray,
    f_psi: npt.NDArray,
    ile: npt.NDArray | bool,
) -> npt.NDArray:
    if cos_i == 1.0 or cos_e == 1.0:
        return cos_i * 0.0 + 1.0
    s = factor * (cos_e_s / cos_e_s0) * (cos_i / cos_i_s0)
    if ile:
        div = 1 + f_psi * (factor * (cos_i / cos_i_s0) - 1)
    else:
        div = 1 + f_psi * (factor * (cos_e / cos_e_s0) - 1)
    s /= div
    return s


# @jit(nogil=True, fastmath=True, cache=cache)
@guvectorize(
    [
        (
            float64,
            float64[:, :],
            float64[:, :],
            float64[:, :],
            float64[:],
            float64[:],
            float64[:],
        )
    ],
    "(),(m,n),(m,n),(m,n)->(n),(n),(n)",
    cache=cache,
    # target="parallel",
)
def microscopic_roughness(
    # inputs
    roughness: float,
    incidence_direction: npt.NDArray,
    emission_direction: npt.NDArray,
    surface_orientation: npt.NDArray,
    # output
    s: npt.NDArray,
    cos_i_s: npt.NDArray,
    cos_e_s: npt.NDArray,
):
    # ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
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
    # original_shape = surface_orientation.shape[1:]
    original_shape = surface_orientation.shape[1]

    # Incidence angle
    # cos_i = dot0(incidence_direction, surface_orientation)
    cos_i = np.empty(original_shape, dtype=np.float64)
    dot0(incidence_direction, surface_orientation, cos_i)
    # Emission angle
    # cos_e = dot0(emission_direction, surface_orientation)
    cos_e = np.empty(original_shape, dtype=np.float64)
    dot0(emission_direction, surface_orientation, cos_e)

    if roughness < EPS:
        # If roughness is zero, return default values
        s[:] = 1.0
        cos_i_s[:] = cos_i
        cos_e_s[:] = cos_e
        print("Roughness is zero, returning default values")
        return

    # Projections
    # projection_incidence = (
    #     incidence_direction - np.expand_dims(cos_i, axis=0) * surface_orientation
    # )
    # projection_emission = (
    #     emission_direction - np.expand_dims(cos_e, axis=0) * surface_orientation
    # )

    # projection_incidence = incidence_direction - cos_i * surface_orientation
    # projection_emission = emission_direction - cos_e * surface_orientation
    # projection_incidence /= normalize_keepdims(projection_incidence)
    # projection_emission /= normalize_keepdims(projection_emission)

    # Azicos_eth angle
    cos_psi = np.empty(original_shape)
    dot0(incidence_direction, emission_direction, cos_psi)
    cos_psi = np.clip(
        (cos_psi - cos_i * cos_e)
        /  #
        (np.sqrt(1 - cos_i**2) * np.sqrt(1 - cos_e**2)),
        -1.0 + EPS,  # avoid numerical issues for f_psi
        1.0,
    )
    sin_psi = np.sqrt(1 - cos_psi**2)

    # Macroscopic Roughness
    cot_rough = 1 / np.tan(roughness)
    # Check for cases
    ile = cos_i > cos_e
    ige = cos_i <= cos_e
    # Check for singularities

    factor = 1 / np.sqrt(1 + np.pi / cot_rough**2)
    f_psi = np.exp(-2 * sin_psi / (1 + cos_psi))

    cos_i_s0 = __cos_s0(factor, cos_i, cot_rough)
    cos_e_s0 = __cos_s0(factor, cos_e, cot_rough)

    cos_i_s[:] = __cos_s(  # noqa
        # cos_i_s = __cos_s(
        factor,
        cos_i,
        cos_e,
        cos_psi,
        cot_rough,
        ile,
        ige,
    )

    cos_e_s[:] = __cos_s(
        # cos_e_s = __cos_s(
        factor,
        cos_e,
        cos_i,
        cos_psi,
        cot_rough,
        ige,
        ile,
    )

    s[:] = __roughness(  # noqa
        # s = __roughness(
        factor,
        cos_i,
        cos_i_s0,
        cos_e,
        cos_e_s,
        cos_e_s0,
        f_psi,
        ile,
    )

    # print(s.shape, cos_i_s.shape, cos_e_s.shape)

    # return (
    #     np.asarray(s).reshape(original_shape),
    #     cos_i_s.reshape(original_shape),
    #     cos_e_s.reshape(original_shape),
    # )


# @vectorize([float64(float64, float64, float64, float64)], target="cpu", cache=cache)
# @jit(nogil=True, fastmath=True, cache=cache)
# def microscopic_roughness(
#     roughness: float,
#     incidence_direction: npt.NDArray,
#     emission_direction: npt.NDArray,
#     surface_orientation: npt.NDArray,
# ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
#     r"""Calculates the microscopic roughness factor for Hapke's model.

#     This correction accounts for the effects of sub-resolution roughness on
#     the observed reflectance.

#     Parameters
#     ----------

#     roughness : float
#         The mean slope angle of surface facets, in radians.
#         A value of 0 means a smooth surface.
#     incidence_direction : npt.NDArray
#         Incidence direction vector(s), shape (..., 3). Assumed to be normalized.
#     emission_direction : npt.NDArray
#         Emission direction vector(s), shape (..., 3). Assumed to be normalized.
#     surface_orientation : npt.NDArray
#         Surface normal vector(s), shape (..., 3). Assumed to be normalized.

#     Returns
#     -------
#     s : npt.NDArray
#         The microscopic roughness factor, shape (...).
#     mu_0_prime : npt.NDArray
#         The modified cosine of the incidence angle ($\mu_0^{\prime}$), accounting
#         for roughness, shape (...).
#     mu_prime : npt.NDArray
#         The modified cosine of the emission angle ($\mu^{\prime}$), accounting
#         for roughness, shape (...).

#     Notes
#     -----
#     The calculations are based on Hapke (1984).

#     - The terms $\mu_0^{\prime}$ (mu_0_s0, mu_0_s) and $\mu^{\prime}$ (mu_s0, mu_s)
#       are calculated based on different conditions for incidence angle `i`
#       and emission angle `e`:

#       - For prime-zero terms ($\mu_0^{\prime(0)}$, $\mu^{\prime(0)}$ used in `mu_0_s0`, `mu_s0`):
#         See Hapke (1984, Eqs. 48, 49).
#       - For $\mu_0^{\prime}$ and $\mu^{\prime}$ when $i < e$:
#         See Hapke (1984, Eqs. 46, 47).
#       - For $\mu_0^{\prime}$ and $\mu^{\prime}$ when $i \ge e$:
#         See Hapke (1984, Eqs. 50, 51).

#     - Input vectors (`incidence_direction`, `emission_direction`, `surface_orientation`)
#       are normalized internally.
#     - If `roughness` is 0, `s` is 1, `mu_0_prime` is `cos(i)`, and `mu_prime` is `cos(e)`.

#     """
#     # original_shape = surface_orientation.shape[1:]
#     # incidence_direction = np.ascontiguousarray(incidence_direction).reshape(3, -1)
#     # emission_direction = np.ascontiguousarray(emission_direction).reshape(3, -1)
#     # surface_orientation = np.ascontiguousarray(surface_orientation).reshape(3, -1)
#     # original_shape = surface_orientation.shape[:-1]
#     # incidence_direction = np.ascontiguousarray(incidence_direction).reshape(-1, 3)
#     # emission_direction = np.ascontiguousarray(emission_direction).reshape(-1, 3)
#     # surface_orientation = np.ascontiguousarray(surface_orientation).reshape(-1, 3)

#     # Angles
#     # incidence_direction /= normalize_keepdims(incidence_direction)
#     # emission_direction /= normalize_keepdims(emission_direction)
#     # surface_orientation /= normalize_keepdims(surface_orientation)

#     original_shape = surface_orientation.shape[1:]
#     s = np.ones(original_shape, dtype=np.float64)
#     cos_i_s = np.empty(original_shape, dtype=np.float64)
#     cos_e_s = np.empty(original_shape, dtype=np.float64)
#     if roughness < EPS:
#         # Incidence angle
#         # cos_i = angle_processing(incidence_direction, surface_orientation)
#         dot0(incidence_direction, surface_orientation, cos_i_s)
#         # Emission angle
#         # cos_e = angle_processing(emission_direction, surface_orientation)
#         dot0(emission_direction, surface_orientation, cos_e_s)

#         print("Roughness is zero, returning default values")
#         # return (s, cos_i_s, cos_e_s)
#     else:
#         microscopic_roughness_scalar(
#             roughness,
#             incidence_direction,
#             emission_direction,
#             surface_orientation,
#             s,
#             cos_i_s,
#             cos_e_s,
#         )
#     return s, cos_i_s, cos_e_s
#     # return microscopic_roughness_scalar(
#     #     roughness,
#     #     incidence_direction,
#     #     emission_direction,
#     #     surface_orientation,
#     # )
