from typing import Callable

import numpy as np
import numpy.typing as npt

from refmod.hapke.functions import h_function
from refmod.hapke.roughness import microscopic_roughness


def imsa(
    incidence_direction: npt.NDArray,
    emission_direction: npt.NDArray,
    surface_orientation: npt.NDArray,
    single_scattering_albedo: npt.NDArray,
    phase_function: Callable[[npt.NDArray], npt.NDArray],
    opposition_effect_h: float = 0,
    oppoistion_effect_b0: float = 0,  # Note: Corrected typo in original arg name for consistency
    roughness: float = 0,
) -> npt.NDArray:
    """Calculates reflectance using the IMSA model.

    IMSA stands for Inversion of Multiple Scattering and Absorption.

    Parameters
    ----------

    incidence_direction : npt.NDArray
        Incidence direction vector(s), shape (..., 3).
    emission_direction : npt.NDArray
        Emission direction vector(s), shape (..., 3).
    surface_orientation : npt.NDArray
        Surface normal vector(s), shape (..., 3).
    single_scattering_albedo : npt.NDArray
        Single scattering albedo, shape (...).
    phase_function : Callable[[npt.NDArray], npt.NDArray]
        Callable that accepts `cos_alpha` (cosine of phase angle) and
        returns phase function values.
    opposition_effect_h : float, optional
        Opposition effect parameter h, by default 0.
    oppoistion_effect_b0 : float, optional
        Opposition effect parameter B0 (b_zero), by default 0.
        Note: Original argument name `oppoistion_effect_b0` kept for API compatibility.
    roughness : float, optional
        Surface roughness parameter, by default 0.

    Returns
    -------
    npt.NDArray
        Calculated reflectance values, shape (...).

    Raises
    ------
    Exception
        If any calculated reflectance value has a significant imaginary part.

    Notes
    -----
    - Input arrays `incidence_direction`, `emission_direction`,
      `surface_orientation`, and `single_scattering_albedo` are expected to
      broadcast together.
    - The `phase_function` should be vectorized to handle arrays of `cos_alpha`.
    - The IMSA model accounts for multiple scattering and absorption.

    References

    ----------

    [IMSAModelPlaceholder]

    """
    # Allocate memory
    if np.ndim(single_scattering_albedo) == 0:
        single_scattering_albedo *= np.ones(surface_orientation.shape[:2])

    # Angles
    incidence_direction /= np.linalg.norm(
        incidence_direction, axis=2, keepdims=True
    )
    emission_direction /= np.linalg.norm(
        emission_direction, axis=2, keepdims=True
    )
    surface_orientation /= np.linalg.norm(
        surface_orientation, axis=2, keepdims=True
    )

    # Phase angle
    cos_alpha = np.sum(incidence_direction * emission_direction, axis=2)
    cos_alpha[cos_alpha > 1] = 1
    cos_alpha[cos_alpha < -1] = -1
    sin_alpha = np.sqrt(1 - cos_alpha**2)

    [s, mu_0, mu] = microscopic_roughness(
        roughness, incidence_direction, emission_direction, surface_orientation
    )

    # Phase function values
    p_g = phase_function(cos_alpha)

    # H-Function
    # TODO: implement derivation of the H-function
    h_mu_0 = h_function(mu_0, single_scattering_albedo)
    h_mu = h_function(mu, single_scattering_albedo)

    # Opposition effect
    b_g = 1.0
    if oppoistion_effect_b0 != 0.0:
        b_g += oppoistion_effect_b0 / (
            1 + sin_alpha / (1 + cos_alpha) / opposition_effect_h
        )

    # Reflectance
    refl = mu_0 / (mu_0 + mu) * s / (4 * np.pi)
    refl *= single_scattering_albedo * (b_g * p_g + h_mu_0 * h_mu - 1)
    refl /= 4 * np.pi
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

    return refl
