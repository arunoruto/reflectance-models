"""
??? info "References"

    1. Hapke, B. (1984). Bidirectional reflectance spectroscopy: 3.
    Correction for macroscopic roughness. Icarus, 59(1), 41-59.
    <https://doi.org/10.1016/0019-1035(84)90054-X>
"""

import numpy as np
import numpy.typing as npt


def __f_exp(x: npt.NDArray, y: float):
    """
    Helper for the micoscopic roughness:
    calculates the exponential function for the given inputs.

    Args:
        x (numpy.ndarray): The input array.
        y (float): The exponential factor.

    Returns:
        (np.ndarray): The result of the exponential function.

    """
    return np.exp(-2 / np.pi * y * x)


def __f_exp_2(x: npt.NDArray, y: float):
    """
    Helper for the micoscopic roughness:
    calculates the exponential function with a squared term.

    Args:
        x (numpy.ndarray): The input array.
        y (float): The value to be squared.

    Returns:
        (np.ndarray): The result of the exponential function.

    """
    return np.exp(-(y**2) * x**2 / np.pi)


def microscopic_roughness(
    roughness: float,
    incidence_direction: npt.NDArray,
    emission_direction: npt.NDArray,
    surface_orientation: npt.NDArray,
):
    r"""
    Calculates the microscopic roughness factor for the Hapke reflectance model.

    Args:
        roughness (float): The roughness parameter.
        incidence_direction (numpy.ndarray): Array of incidence directions.
        emission_direction (numpy.ndarray): Array of emission directions.
        surface_orientation (numpy.ndarray): Array of surface orientations.

    Returns:
        s (numpy.ndarray): The microscopic roughness factor.
        mu_0 (numpy.ndarray): The modified incidence-normal cosine value ($\mu_0^{\prime}$).
        mu (numpy.ndarray): The modified emission-normal cosine value ($\mu^{\prime}$).

    Note:
        - prime-zero terms: equations 48 and 49 in Hapke (1984).
        - $i  < e$: equations 46 and 47 in Hapke (1984).
        - $i >= e$: equations 50 and 51 in Hapke (1984).
    """

    # Angles
    incidence_direction /= np.linalg.norm(
        incidence_direction, axis=-1, keepdims=True
    )
    emission_direction /= np.linalg.norm(
        emission_direction, axis=-1, keepdims=True
    )
    surface_orientation /= np.linalg.norm(
        surface_orientation, axis=-1, keepdims=True
    )

    # Incidence angle
    mu_0 = np.sum(incidence_direction * surface_orientation, axis=-1)
    mu_0[mu_0 > 1] = 1
    mu_0[mu_0 < -1] = -1
    sin_i = np.sqrt(1 - mu_0**2)
    # tan_i = sin_i / mu_0
    # cot_i = np.divide(1, tan_i, out=np.ones_like(tan_i) * np.inf, where=tan_i != 0)
    cot_i = np.divide(
        mu_0, sin_i, out=np.ones_like(mu_0) * np.inf, where=sin_i != 0
    )
    i = np.arccos(mu_0)

    # Emission angle
    mu = np.sum(emission_direction * surface_orientation, axis=-1)
    mu[mu > 1] = 1
    mu[mu < -1] = -1
    sin_e = np.sqrt(1 - mu**2)
    tan_e = sin_e / mu
    cot_e = np.divide(
        1, tan_e, out=np.ones_like(tan_e) * np.inf, where=tan_e != 0
    )
    e = np.arccos(mu)

    if roughness == 0:
        print("Roughness is zero, returning default values")
        return np.ones_like(mu), mu_0, mu

    # Projections
    projection_incidence = (
        incidence_direction
        - np.expand_dims(mu_0, axis=-1) * surface_orientation
    )
    projection_emission = (
        emission_direction - np.expand_dims(mu, axis=-1) * surface_orientation
    )
    projection_incidence_norm = np.linalg.norm(projection_incidence, axis=-1)
    projection_emission_norm = np.linalg.norm(projection_emission, axis=-1)

    # Azimuth angle
    cos_psi = np.divide(
        np.sum(projection_incidence * projection_emission, axis=-1),
        projection_incidence_norm * projection_emission_norm,
        out=np.ones_like(projection_incidence_norm) * np.nan,
        where=projection_incidence_norm * projection_emission_norm != 0,
    )
    cos_psi[cos_psi > 1] = 1
    cos_psi[cos_psi < -1] = -1
    sin_psi = np.sqrt(1 - cos_psi**2)
    sin_psi_div_2_sq = np.abs(0.5 - cos_psi / 2)
    psi = np.arccos(cos_psi)

    # Macroscopic Roughness
    tan_rough = np.tan(roughness)
    cot_rough = 1 / tan_rough
    # Check for cases
    ile = i < e
    ige = i >= e
    # Check for singularities
    index_mu0 = 1 == mu_0
    index_mu = 1 == mu

    factor = 1 / np.sqrt(1 + np.pi * tan_rough**2)
    # f_exp = lambda x: np.exp(-2.0 / np.pi * cot_rough * x)
    # f_exp_2 = lambda x: np.exp(-(cot_rough**2) * x**2 / np.pi)
    # f_psi = np.exp(-2.0 * sin_psi / (1 + cos_psi))
    # f_psi[-1 == cos_psi] = 0
    f_psi = np.exp(
        -2
        * np.divide(
            sin_psi,
            1 + cos_psi,
            out=np.ones_like(sin_psi) * -1,
            where=cos_psi != -1,
        )
    )

    mu_0_s0 = factor * (
        mu_0
        + sin_i
        * tan_rough
        * __f_exp_2(cot_i, cot_rough)
        / (2.0 - __f_exp(cot_i, cot_rough))
    )
    mu_s0 = factor * (
        mu
        + sin_e
        * tan_rough
        * __f_exp_2(cot_e, cot_rough)
        / (2.0 - __f_exp(cot_e, cot_rough))
    )

    mu_0_s = np.zeros_like(mu_0)
    mu_0_s[ile] = factor * (
        mu_0[ile]
        + sin_i[ile]
        * tan_rough
        * (
            cos_psi[ile] * __f_exp_2(cot_e[ile], cot_rough)
            + sin_psi_div_2_sq[ile] * __f_exp_2(cot_i[ile], cot_rough)
        )
        / (
            2
            - __f_exp(cot_e[ile], cot_rough)
            - psi[ile] / np.pi * __f_exp(cot_i[ile], cot_rough)
        )
    )
    mu_0_s[ige] = factor * (
        mu_0[ige]
        + sin_i[ige]
        * tan_rough
        * (
            __f_exp_2(cot_i[ige], cot_rough)
            - sin_psi_div_2_sq[ige] * __f_exp_2(cot_e[ige], cot_rough)
        )
        / (
            2.0
            - __f_exp(cot_i[ige], cot_rough)
            - psi[ige] / np.pi * __f_exp(cot_e[ige], cot_rough)
        )
    )
    mu_0_s[index_mu0 | index_mu] = mu_0[index_mu0 | index_mu]

    mu_s = np.zeros_like(mu)
    mu_s[ile] = factor * (
        mu[ile]
        + sin_e[ile]
        * tan_rough
        * (
            __f_exp_2(cot_e[ile], cot_rough)
            - sin_psi_div_2_sq[ile] * __f_exp_2(cot_i[ile], cot_rough)
        )
        / (
            2.0
            - __f_exp(cot_e[ile], cot_rough)
            - psi[ile] / np.pi * __f_exp(cot_i[ile], cot_rough)
        )
    )
    mu_s[ige] = factor * (
        mu[ige]
        + sin_e[ige]
        * tan_rough
        * (
            cos_psi[ige] * __f_exp_2(cot_i[ige], cot_rough)
            + sin_psi_div_2_sq[ige] * __f_exp_2(cot_e[ige], cot_rough)
        )
        / (
            2.0
            - __f_exp(cot_i[ige], cot_rough)
            - psi[ige] / np.pi * __f_exp(cot_e[ige], cot_rough)
        )
    )
    mu_s[index_mu0 | index_mu] = mu[index_mu0 | index_mu]

    s = factor * (mu_s / mu_s0) * (mu_0 / mu_0_s0)
    s[ile] = s[ile] / (
        1 + f_psi[ile] * (factor * (mu_0[ile] / mu_0_s0[ile]) - 1)
    )
    s[ige] = s[ige] / (1 + f_psi[ige] * (factor * (mu[ige] / mu_s0[ige]) - 1))
    s[index_mu0 | index_mu] = 1

    return np.squeeze(s), np.squeeze(mu_0_s), np.squeeze(mu_s)
