from typing import Callable, Tuple, Union

import numpy as np
import numpy.typing as npt

from refmod.hapke.functions import h_function
from refmod.hapke.legendre import coef_a, function_p, value_p
from refmod.hapke.roughness import microscopic_roughness


def amsa(
    incidence_direction: npt.NDArray,
    emission_direction: npt.NDArray,
    surface_orientation: npt.NDArray,
    single_scattering_albedo: npt.NDArray,
    phase_function: Callable[[npt.NDArray], npt.NDArray],
    b_n: npt.NDArray,
    a_n: npt.NDArray = np.empty(1) * np.nan,
    hs: float = 0,
    bs0: float = 0,
    roughness: float = 0,
    hc: float = 0,
    bc0: float = 0,
    nan2zero: bool = False,
    derivative: bool = False,
) -> Union[npt.NDArray, Tuple[npt.NDArray, npt.NDArray]]:
    """
    Calculates the reflectance using the AMSA (Advanced Modified Shadowing and Coherent Backscattering) model.

    Args:
        incidence_direction: Array of shape (number_u, number_v, 3) representing the incidence direction vectors.
        emission_direction: Array of shape (number_u, number_v, 3) representing the emission direction vectors.
        surface_orientation: Array of shape (number_u, number_v, 3) representing the surface orientation vectors.
        single_scattering_albedo: Array of shape (number_u, number_v) representing the single scattering albedo values.
        phase_function: Callable function that takes the cosine of the scattering angle and returns the phase function values.
        b_n: Array of shape (n,) representing the coefficients of the Legendre expansion.
        a_n: Array of shape (n,) representing the coefficients of the Legendre expansion. Defaults to np.empty(1) * np.nan.
        hs: Float representing the shadowing parameter. Defaults to 0.
        bs0: Float representing the shadowing parameter. Defaults to 0.
        roughness: Float representing the surface roughness. Defaults to 0.
        hc: Float representing the coherent backscattering parameter. Defaults to 0.
        bc0: Float representing the coherent backscattering parameter. Defaults to 0.

    Returns:
        Array of shape (number_u, number_v) representing the reflectance values.

    Raises:
        Exception: If at least one reflectance value is not real.

    """
    # Allocate memory
    number_u, number_v = surface_orientation.shape[:2]

    if np.ndim(single_scattering_albedo) == 0:
        single_scattering_albedo *= np.ones((number_u, number_v))

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

    # Roughness
    [s, mu_0, mu] = microscopic_roughness(
        roughness, incidence_direction, emission_direction, surface_orientation
    )

    # Legendre
    # [p_mu_0, p_mu, p] = legendre_expansion(mu_0, mu, 0.18, 1.1, 15)
    if np.any(np.isnan(a_n)):
        a_n = coef_a(b_n.size - 1)
    p_mu_0 = function_p(mu_0, b_n, a_n)
    p_mu = function_p(mu, b_n, a_n)
    p = value_p(b_n, a_n)

    # Alpha angle
    cos_alpha = np.sum(incidence_direction * emission_direction, axis=-1)
    cos_alpha[cos_alpha > 1] = 1
    cos_alpha[cos_alpha < -1] = -1
    sin_alpha = np.sqrt(1 - cos_alpha**2)
    tan_alpha_2 = sin_alpha / (1 + cos_alpha)

    # Phase function values
    p_g = phase_function(cos_alpha)

    # H-Function
    h_mu_0 = h_function(
        mu_0, single_scattering_albedo, level=2, derivative=derivative
    )
    h_mu = h_function(
        mu, single_scattering_albedo, level=2, derivative=derivative
    )

    dm_dw = np.nan
    if derivative:
        dh0_dw = h_mu_0[1]
        h_mu_0 = h_mu_0[0]

        dh_dw = h_mu[1]
        h_mu = h_mu[0]

        # derivative of M term
        dm_dw = (
            p_mu_0 * dh_dw
            + p_mu * dh0_dw
            + p * (dh_dw * (h_mu_0 - 1) + dh0_dw * (h_mu - 1))
        )

    # M term
    m = (
        p_mu_0 * (h_mu - 1)
        + p_mu * (h_mu_0 - 1)
        + p * (h_mu_0 - 1) * (h_mu - 1)
    )

    # Shadow-hiding effect
    b_sh = 1
    if (bs0 != 0) and (hs != 0):
        b_sh += bs0 / (1 + tan_alpha_2 / hs)

    # Coherent backscattering effect
    b_cb = 1
    if (bc0 != 0) and (hc != 0):
        hc_2 = tan_alpha_2 / hc
        bc = 0.5 * (1 + (1 - np.exp(-hc_2)) / hc_2) / (1 + hc_2) ** 2
        b_cb += bc0 * bc

    # Reflectance
    albedo_independent = mu_0 / (mu_0 + mu) * s / (4 * np.pi) * b_cb
    refl = albedo_independent * single_scattering_albedo * (b_sh * p_g + m)
    refl[(mu <= 0) | (mu_0 <= 0)] = np.nan
    refl[refl < 1e-6] = np.nan

    # Final result
    threshold_imag = 0.1
    threshold_error = 1e-4
    arg_rh = np.divide(
        np.imag(refl),
        np.real(refl),
        out=np.zeros_like(refl, dtype=float),
        where=np.real(refl) != 0,
    )
    refl[arg_rh > threshold_imag] /= np.nan

    if np.any(arg_rh >= threshold_error):
        raise Exception("At least one reflectance value is not real!")

    refl = np.nan_to_num(refl) if nan2zero else refl

    if derivative:
        dr_dw = (
            b_sh * p_g + m + single_scattering_albedo * dm_dw
        ) * albedo_independent
        print("Returning derivative too")
        dr_dw = np.nan_to_num(dr_dw) if nan2zero else dr_dw
        return refl, dr_dw

    return refl
