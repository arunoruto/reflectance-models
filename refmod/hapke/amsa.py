import numpy as np
import numpy.typing as npt
from refmod.hapke.functions import (
    PhaseFunctionType,
    angle_processing_base,
    h_function_2,
    h_function_2_derivative,
    normalize_keepdims,
    phase_function,
)
from refmod.hapke.legendre import function_p, value_p
from refmod.hapke.roughness import microscopic_roughness


def __amsa_preprocess(
    single_scattering_albedo: npt.NDArray,
    incidence_direction: npt.NDArray,
    emission_direction: npt.NDArray,
    surface_orientation: npt.NDArray,
    phase_function_type: PhaseFunctionType,
    b_n: npt.NDArray,
    a_n: npt.NDArray,
    roughness: float = 0.0,
    hs: float = 0.0,
    bs0: float = 0.0,
    hc: float = 0.0,
    bc0: float = 0.0,
    phase_function_args: tuple = (),
):
    """Preprocesses the inputs for the AMSA model.

    Parameters
    ----------

    single_scattering_albedo : npt.NDArray
        Single scattering albedo.
    incidence_direction : npt.NDArray
        Incidence direction vector(s) of shape (..., 3).
    emission_direction : npt.NDArray
        Emission direction vector(s) of shape (..., 3).
    surface_orientation : npt.NDArray
        Surface orientation vector(s) of shape (..., 3).
    phase_function_type : PhaseFunctionType
        Type of phase function to use.
    b_n : npt.NDArray
        Coefficients of the Legendre expansion.
    a_n : npt.NDArray
        Coefficients of the Legendre expansion.
    roughness : float, optional
        Surface roughness, by default 0.0.
    hs : float, optional
        Shadowing parameter, by default 0.0.
    bs0 : float, optional
        Shadowing parameter, by default 0.0.
    hc : float, optional
        Coherent backscattering parameter, by default 0.0.
    bc0 : float, optional
        Coherent backscattering parameter, by default 0.0.
    phase_function_args : tuple, optional
        Additional arguments for the phase function, by default ().

    Returns
    -------
    tuple
        A tuple containing:
            - albedo_independent : npt.NDArray
                Albedo-independent term.
            - mu_0 : npt.NDArray
                Cosine of the incidence angle.
            - mu : npt.NDArray
                Cosine of the emission angle.
            - p_g : npt.NDArray
                Phase function values.
            - m : npt.NDArray
                M term.
            - p_mu_0 : npt.NDArray
                Legendre polynomial values for mu_0.
            - p_mu : npt.NDArray
                Legendre polynomial values for mu.
            - p : npt.NDArray
                Legendre polynomial values.
            - h_mu_0 : npt.NDArray
                H-function values for mu_0.
            - h_mu : npt.NDArray
                H-function values for mu.
    """
    # Angles
    incidence_direction /= normalize_keepdims(incidence_direction)
    emission_direction /= normalize_keepdims(emission_direction)
    surface_orientation /= normalize_keepdims(surface_orientation)

    # Roughness
    s, mu_0, mu = microscopic_roughness(
        roughness, incidence_direction, emission_direction, surface_orientation
    )

    # Legendre
    p_mu_0 = function_p(mu_0, b_n, a_n)
    p_mu = function_p(mu, b_n, a_n)
    p = value_p(b_n, a_n)

    # Alpha angle
    cos_alpha, sin_alpha = angle_processing_base(
        incidence_direction,
        emission_direction,
    )
    tan_alpha_2 = sin_alpha / (1 + cos_alpha)

    p_g = phase_function(cos_alpha, phase_function_type, phase_function_args)

    # H-Function
    h_mu_0 = h_function_2(mu_0, single_scattering_albedo)
    h_mu = h_function_2(mu, single_scattering_albedo)

    # Shadow-hiding effect
    b_sh = np.ones_like(tan_alpha_2)
    if (bs0 != 0) and (hs != 0):
        b_sh += bs0 / (1 + tan_alpha_2 / hs)
    p_g *= b_sh

    # Coherent backscattering effect
    b_cb = np.ones_like(tan_alpha_2)
    if (bc0 != 0) and (hc != 0):
        hc_2 = tan_alpha_2 / hc
        bc = 0.5 * (1 + (1 - np.exp(-hc_2)) / hc_2) / (1 + hc_2) ** 2
        b_cb += bc0 * bc

    # M term
    m = p_mu_0 * (h_mu - 1) + p_mu * (h_mu_0 - 1) + p * (h_mu_0 - 1) * (h_mu - 1)

    albedo_independent = mu_0 / (mu_0 + mu) * s / (4 * np.pi) * b_cb

    return (
        albedo_independent,
        mu_0,
        mu,
        p_g,
        m,
        p_mu_0,
        p_mu,
        p,
        h_mu_0,
        h_mu,
    )


def amsa(
    incidence_direction: npt.NDArray,
    emission_direction: npt.NDArray,
    surface_orientation: npt.NDArray,
    single_scattering_albedo: npt.NDArray,
    phase_function_type: PhaseFunctionType,
    b_n: npt.NDArray,
    a_n: npt.NDArray,
    hs: float = 0,
    bs0: float = 0,
    roughness: float = 0,
    hc: float = 0,
    bc0: float = 0,
    phase_function_args: tuple = (),
    refl_optimization: npt.NDArray | None = None,
) -> npt.NDArray:
    """Calculates the reflectance using the AMSA model.

    Parameters
    ----------

    incidence_direction : npt.NDArray
        Incidence direction vector(s) of shape (..., 3).
    emission_direction : npt.NDArray
        Emission direction vector(s) of shape (..., 3).
    surface_orientation : npt.NDArray
        Surface orientation vector(s) of shape (..., 3).
    single_scattering_albedo : npt.NDArray
        Single scattering albedo.
    phase_function_type : PhaseFunctionType
        Type of phase function to use.
    b_n : npt.NDArray
        Coefficients of the Legendre expansion.
    a_n : npt.NDArray
        Coefficients of the Legendre expansion.
    hs : float, optional
        Shadowing parameter, by default 0.
    bs0 : float, optional
        Shadowing parameter, by default 0.
    roughness : float, optional
        Surface roughness, by default 0.
    hc : float, optional
        Coherent backscattering parameter, by default 0.
    bc0 : float, optional
        Coherent backscattering parameter, by default 0.
    phase_function_args : tuple, optional
        Additional arguments for the phase function, by default ().
    refl_optimization : npt.NDArray | None, optional
        Reflectance optimization array, by default None.

    Returns
    -------
    npt.NDArray
        Reflectance values.

    Raises
    ------
    Exception
        If at least one reflectance value is not real.

    References

    ----------

    [AMSAModelPlaceholder]
    """
    (
        albedo_independent,
        mu_0,
        mu,
        p_g,
        m,
        _,
        _,
        _,
        _,
        _,
    ) = __amsa_preprocess(
        single_scattering_albedo,
        incidence_direction,
        emission_direction,
        surface_orientation,
        phase_function_type,
        b_n,
        a_n,
        roughness,
        hs,
        bs0,
        hc,
        bc0,
        phase_function_args,
    )
    # Reflectance
    refl = albedo_independent * single_scattering_albedo * (p_g + m)
    refl = np.where((mu <= 0) | (mu_0 <= 0), np.nan, refl)
    refl = np.where(refl < 1e-6, np.nan, refl)

    # Final result
    threshold_imag = 0.1
    threshold_error = 1e-4
    arg_rh = np.where(np.real(refl) == 0, 0, np.imag(refl) / np.real(refl))
    refl = np.where(arg_rh > threshold_imag, np.nan, refl)

    if np.any(arg_rh >= threshold_error):
        raise Exception("At least one reflectance value is not real!")

    if refl_optimization is None:
        return refl
    refl -= refl_optimization
    return refl


def amsa_derivative(
    single_scattering_albedo: npt.NDArray,
    incidence_direction: npt.NDArray,
    emission_direction: npt.NDArray,
    surface_orientation: npt.NDArray,
    phase_function_type: PhaseFunctionType,
    b_n: npt.NDArray,
    a_n: npt.NDArray,
    roughness: float = 0,
    hs: float = 0,
    bs0: float = 0,
    hc: float = 0,
    bc0: float = 0,
    phase_function_args: tuple = (),
    refl_optimization: npt.NDArray | None = None,
) -> npt.NDArray:
    """Calculates the derivative of the reflectance using the AMSA model.

    Parameters
    ----------

    single_scattering_albedo : npt.NDArray
        Single scattering albedo.
    incidence_direction : npt.NDArray
        Incidence direction vector(s) of shape (..., 3).
    emission_direction : npt.NDArray
        Emission direction vector(s) of shape (..., 3).
    surface_orientation : npt.NDArray
        Surface orientation vector(s) of shape (..., 3).
    phase_function_type : PhaseFunctionType
        Type of phase function to use.
    b_n : npt.NDArray
        Coefficients of the Legendre expansion.
    a_n : npt.NDArray
        Coefficients of the Legendre expansion.
    roughness : float, optional
        Surface roughness, by default 0.
    hs : float, optional
        Shadowing parameter, by default 0.
    bs0 : float, optional
        Shadowing parameter, by default 0.
    hc : float, optional
        Coherent backscattering parameter, by default 0.
    bc0 : float, optional
        Coherent backscattering parameter, by default 0.
    phase_function_args : tuple, optional
        Additional arguments for the phase function, by default ().
    refl_optimization : npt.NDArray | None, optional
        Reflectance optimization array, by default None.
        This parameter is not used in the derivative calculation.

    Returns
    -------
    npt.NDArray
        Derivative of the reflectance with respect to single scattering albedo.

    References

    ----------

    [AMSAModelPlaceholder]
    """
    (
        albedo_independent,
        mu_0,
        mu,
        p_g,
        m,
        p_mu_0,
        p_mu,
        p,
        h_mu_0,
        h_mu,
    ) = __amsa_preprocess(
        single_scattering_albedo,
        incidence_direction,
        emission_direction,
        surface_orientation,
        phase_function_type,
        b_n,
        a_n,
        roughness,
        hs,
        bs0,
        hc,
        bc0,
        phase_function_args,
    )

    dh0_dw = h_function_2_derivative(mu_0, single_scattering_albedo)
    dh_dw = h_function_2_derivative(mu, single_scattering_albedo)

    # derivative of M term
    dm_dw = (
        p_mu_0 * dh_dw
        + p_mu * dh0_dw
        + p * (dh_dw * (h_mu_0 - 1) + dh0_dw * (h_mu - 1))
    )

    dr_dw = (p_g + m + single_scattering_albedo * dm_dw) * albedo_independent

    return dr_dw
