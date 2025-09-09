import numpy as np
import numpy.typing as npt
from numba import jit
from refmod.config import cache
from refmod.hapke.functions.h import h_function, h_function_derivative
from refmod.hapke.functions.legendre import function_p, value_p
from refmod.hapke.functions.opposition_effect import (
    coherant_backscattering,
    shadow_hiding,
)
from refmod.hapke.functions.phase import PhaseFunctionType, phase_function
from refmod.hapke.functions.roughness import microscopic_roughness
from refmod.hapke.functions.vectors import angle_processing, dot0, normalize_keepdims


# @jit(nogil=True, fastmath=True, cache=cache)
def __amsa_preprocess(
    single_scattering_albedo: npt.NDArray,
    incidence_direction: npt.NDArray,
    emission_direction: npt.NDArray,
    surface_orientation: npt.NDArray,
    phase_function_type: PhaseFunctionType = "dhg",
    b_n: npt.NDArray | None = None,
    a_n: npt.NDArray | None = None,
    roughness: float = 0.0,
    shadow_hiding_h: float = 0.0,
    shadow_hiding_b0: float = 0.0,
    coherant_backscattering_h: float = 0.0,
    coherant_backscattering_b0: float = 0.0,
    phase_function_args: tuple = (),
    h_level: int = 2,
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
    shadow_hiding_h : float, optional
        Shadowing parameter, by default 0.0.
    shadow_hiding_b0 : float, optional
        Shadowing parameter, by default 0.0.
    coherant_backscattering_h : float, optional
        Coherent backscattering parameter, by default 0.0.
    coherant_backscattering_b0 : float, optional
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
    incidence_direction /= normalize_keepdims(incidence_direction, 0)
    emission_direction /= normalize_keepdims(emission_direction, 0)
    surface_orientation /= normalize_keepdims(surface_orientation, 0)

    # Roughness
    s, mu_0, mu = microscopic_roughness(
        roughness, incidence_direction, emission_direction, surface_orientation
    )

    # Phase angle Alpha
    cos_alpha = angle_processing(incidence_direction, emission_direction)
    sin_alpha = np.sqrt(1 - cos_alpha**2)
    tan_alpha_2 = sin_alpha / (1 + cos_alpha)

    p_g = phase_function(cos_alpha, phase_function_type, phase_function_args)

    # H-Function
    if b_n is None:
        h_level = 1
    h_mu_0 = h_function(mu_0, single_scattering_albedo, level=h_level)
    h_mu = h_function(mu, single_scattering_albedo, level=h_level)

    if b_n is None:
        # If the Legendre terms are not used, the model reduces to IMSA
        p_mu_0 = np.ones_like(h_mu_0)
        p_mu = np.ones_like(h_mu)
        p = 1.0
        m = h_mu_0 * h_mu - 1
    else:
        # Legendre
        p_mu_0 = np.asarray(function_p(mu_0, b_n, a_n))
        p_mu = np.asarray(function_p(mu, b_n, a_n))
        p = value_p(b_n, a_n)

        # M term
        m = p_mu_0 * (h_mu - 1) + p_mu * (h_mu_0 - 1) + p * (h_mu_0 - 1) * (h_mu - 1)

    ## If the following two terms are ignored, we get the MIMSA model
    b_sh = shadow_hiding(
        tan_alpha_2,
        shadow_hiding_h,
        shadow_hiding_b0,
    )
    p_g *= b_sh

    b_cb = coherant_backscattering(
        tan_alpha_2,
        coherant_backscattering_h,
        coherant_backscattering_b0,
    )

    ## Merging into result
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


# @jit(nogil=True, fastmath=True, cache=cache)
def __amsa_scalar(
    single_scattering_albedo: npt.NDArray,
    incidence_direction: npt.NDArray,
    emission_direction: npt.NDArray,
    surface_orientation: npt.NDArray,
    phase_function_type: PhaseFunctionType = "dhg",
    b_n: npt.NDArray | None = None,
    a_n: npt.NDArray | None = None,
    roughness: float = 0,
    shadow_hiding_h: float = 0.0,
    shadow_hiding_b0: float = 0.0,
    coherant_backscattering_h: float = 0.0,
    coherant_backscattering_b0: float = 0.0,
    phase_function_args: tuple = (),
    refl_optimization: npt.NDArray | None = None,
    h_level: int = 2,
) -> npt.NDArray:
    """Calculates the reflectance using the AMSA model.

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
    shadow_hiding_h : float, optional
        Shadowing parameter, by default 0.
    shadow_hiding_b0 : float, optional
        Shadowing parameter, by default 0.
    coherant_backscattering_h : float, optional
        Coherent backscattering parameter, by default 0.
    coherant_backscattering_b0 : float, optional
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
        shadow_hiding_h,
        shadow_hiding_b0,
        coherant_backscattering_h,
        coherant_backscattering_b0,
        phase_function_args,
        h_level,
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

    if refl_optimization is not None:
        refl -= refl_optimization
    return refl


# @jit(nogil=True, fastmath=True, cache=cache)
def amsa(
    single_scattering_albedo: npt.NDArray,
    incidence_direction: npt.NDArray,
    emission_direction: npt.NDArray,
    surface_orientation: npt.NDArray,
    phase_function_type: PhaseFunctionType = "dhg",
    b_n: npt.NDArray | None = None,
    a_n: npt.NDArray | None = None,
    roughness: float = 0,
    shadow_hiding_h: float = 0.0,
    shadow_hiding_b0: float = 0.0,
    coherant_backscattering_h: float = 0.0,
    coherant_backscattering_b0: float = 0.0,
    phase_function_args: tuple = (),
    refl_optimization: npt.NDArray | None = None,
    h_level: int = 2,
) -> npt.NDArray:
    """Calculates the reflectance using the AMSA model.

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
    shadow_hiding_h : float, optional
        Shadowing parameter, by default 0.
    shadow_hiding_b0 : float, optional
        Shadowing parameter, by default 0.
    coherant_backscattering_h : float, optional
        Coherent backscattering parameter, by default 0.
    coherant_backscattering_b0 : float, optional
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
    original_shape = np.array(single_scattering_albedo.shape)
    # single_scattering_albedo = np.ascontiguousarray(single_scattering_albedo).reshape(
    #     -1
    # )
    # incidence_direction = np.ascontiguousarray(incidence_direction).reshape(3, -1)
    # emission_direction = np.ascontiguousarray(emission_direction).reshape(3, -1)
    # surface_orientation = np.ascontiguousarray(surface_orientation).reshape(3, -1)

    space_shape = surface_orientation.shape[1:]
    bands_shape = original_shape[: -len(space_shape)]

    # TODO: maybe make more axis, like 1+np.arange(len(bands_shape))
    incidence_direction = np.tile(
        np.expand_dims(incidence_direction, axis=1),
        (1, *bands_shape, 1, 1),
    ).reshape(3, -1)
    emission_direction = np.tile(
        np.expand_dims(emission_direction, axis=1),
        (1, *bands_shape, 1, 1),
    ).reshape(3, -1)
    surface_orientation = np.tile(
        np.expand_dims(np.ascontiguousarray(surface_orientation), axis=1),
        (1, *bands_shape, 1, 1),
    ).reshape(3, -1)
    single_scattering_albedo = np.ascontiguousarray(single_scattering_albedo).reshape(
        -1
    )

    refl = __amsa_scalar(
        single_scattering_albedo,
        incidence_direction,
        emission_direction,
        surface_orientation,
        phase_function_type,
        b_n,
        a_n,
        roughness,
        shadow_hiding_h,
        shadow_hiding_b0,
        coherant_backscattering_h,
        coherant_backscattering_b0,
        phase_function_args,
        refl_optimization,
        h_level,
    )
    return refl.reshape(original_shape)


# @jit(nogil=True, fastmath=True, cache=cache)
def amsa_derivative(
    single_scattering_albedo: npt.NDArray,
    incidence_direction: npt.NDArray,
    emission_direction: npt.NDArray,
    surface_orientation: npt.NDArray,
    phase_function_type: PhaseFunctionType,
    b_n: npt.NDArray | None = None,
    a_n: npt.NDArray | None = None,
    roughness: float = 0,
    shadow_hiding_h: float = 0.0,
    shadow_hiding_b0: float = 0.0,
    coherant_backscattering_h: float = 0.0,
    coherant_backscattering_b0: float = 0.0,
    phase_function_args: tuple = (),
    refl_optimization: npt.NDArray | None = None,
    h_level: int = 2,
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
        shadow_hiding_h,
        shadow_hiding_b0,
        coherant_backscattering_h,
        coherant_backscattering_b0,
        phase_function_args,
    )

    dh0_dw = h_function_derivative(mu_0, single_scattering_albedo, h_level)
    dh_dw = h_function_derivative(mu, single_scattering_albedo, h_level)

    # derivative of M term
    dm_dw = (
        p_mu_0 * dh_dw
        + p_mu * dh0_dw
        + p * (dh_dw * (h_mu_0 - 1) + dh0_dw * (h_mu - 1))
    )

    dr_dw = (p_g + m + single_scattering_albedo * dm_dw) * albedo_independent

    return dr_dw


# @jit(nogil=True, fastmath=True, cache=cache)
def imsa(
    single_scattering_albedo: npt.NDArray,
    incidence_direction: npt.NDArray,
    emission_direction: npt.NDArray,
    surface_orientation: npt.NDArray,
    phase_function_type: PhaseFunctionType = "dhg",
    roughness: float = 0.0,
    opposition_effect_h: float = 0.0,
    opposition_effect_b0: float = 0.0,
    phase_function_args: tuple = (),
    h_level: int = 2,
) -> npt.NDArray:
    """Calculates reflectance using the IMSA model.

    IMSA stands for Inversion of Multiple Scattering and Absorption.

    Parameters
    ----------

    single_scattering_albedo : npt.NDArray
        Single scattering albedo, shape (...).
    incidence_direction : npt.NDArray
        Incidence direction vector(s), shape (..., 3).
    emission_direction : npt.NDArray
        Emission direction vector(s), shape (..., 3).
    surface_orientation : npt.NDArray
        Surface normal vector(s), shape (..., 3).
    phase_function : Callable[[npt.NDArray], npt.NDArray]
        Callable that accepts `cos_alpha` (cosine of phase angle) and
        returns phase function values.
    roughness : float, optional
        Surface roughness parameter, by default 0.
    opposition_effect_h : float, optional
        Opposition effect parameter h, by default 0.
    oppoistion_effect_b0 : float, optional
        Opposition effect parameter B0 (b_zero), by default 0.
        Note: Original argument name `oppoistion_effect_b0` kept for API compatibility.

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
    """
    return amsa(
        single_scattering_albedo=single_scattering_albedo,
        incidence_direction=incidence_direction,
        emission_direction=emission_direction,
        surface_orientation=surface_orientation,
        phase_function_type=phase_function_type,
        b_n=None,
        a_n=None,
        roughness=roughness,
        shadow_hiding_h=opposition_effect_h,
        shadow_hiding_b0=opposition_effect_b0,
        phase_function_args=phase_function_args,
        h_level=h_level,
    )
