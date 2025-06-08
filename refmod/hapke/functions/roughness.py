"""Surface roughness corrections based on Hapke's model.

This module provides functions to calculate corrections for macroscopic
surface roughness, a key component in photometric modeling as described
by Hapke.

??? info "References"

    Hapke (1984)
"""

import numpy as np
import numpy.typing as npt


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
