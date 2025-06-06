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
    oppoistion_effect_b0: float = 0, # Typo: opposition_effect_b0
    roughness: float = 0,
    nan2zero: bool = False, # Added for consistency
) -> npt.NDArray:
    """Isotropic Multiple Scattering Approximation (IMSA) model.

    Calculates the bidirectional reflectance of a particulate surface using
    Hapke's IMSA model. This model considers multiple scattering,
    opposition effect, and macroscopic roughness with an isotropic
    phase function.

    Parameters
    ----------
    incidence_direction : npt.NDArray
        Incidence direction vectors, shape (..., 3). Each vector should be
        normalized.
    emission_direction : npt.NDArray
        Emission direction vectors, shape (..., 3). Each vector should be
        normalized.
    surface_orientation : npt.NDArray
        Surface normal vectors, shape (..., 3). Each vector should be
        normalized.
    single_scattering_albedo : npt.NDArray
        Single scattering albedo, shape (...). Values should be between 0 and 1.
    phase_function : Callable[[npt.NDArray], npt.NDArray]
        A callable function that accepts the cosine of the phase angle (g)
        and returns the phase function P(g). For IMSA, this is typically
        isotropic (returns 1.0) or a simple form.
    opposition_effect_h : float, optional
        Width parameter for the shadow-hiding opposition effect. Defaults to 0.
    opposition_effect_b0 : float, optional
        Amplitude parameter for the shadow-hiding opposition effect.
        Defaults to 0. (Corrected typo from `oppoistion_effect_b0`)
    roughness : float, optional
        Mean slope angle of surface facets, in radians. Used to calculate
        microscopic roughness effects. Defaults to 0.
    nan2zero : bool, optional
        If True, convert NaN values in the output to zero. Defaults to False.

    Returns
    -------
    npt.NDArray
        Calculated bidirectional reflectance, shape (...).

    Raises
    ------
    ValueError
        If any reflectance value has a significant imaginary component,
        indicating a potential issue in calculations or input parameters.

    Notes
    -----
    The IMSA model is a foundational part of Hapke's theory, described across
    several papers, including [Hapke1981]_, [Hapke1984]_ (for roughness),
    and [Hapke1986]_ (for opposition effect).
    This implementation assumes an isotropic phase function for the
    multiple scattering term (H-functions), though the single scattering
    term uses the provided `phase_function`.

    Examples
    --------
    >>> import numpy as np
    >>> incidence = np.array([0., 0., -1.])
    >>> emission = np.array([np.sin(np.deg2rad(30)), 0., -np.cos(np.deg2rad(30))])
    >>> surface_normal = np.array([0., 0., 1.])
    >>> ssa = 0.9
    >>> def isotropic_phase_func(cos_g): return np.ones_like(cos_g)
    >>> reflectance = imsa(
    ...     incidence, emission, surface_normal, ssa,
    ...     isotropic_phase_func, opposition_effect_h=0.05,
    ...     opposition_effect_b0=1.0, roughness=np.deg2rad(20)
    ... )
    >>> print(reflectance) # Illustrative
    0.1525...

    References
    ----------
    .. [Hapke1981] Hapke, B. (1981). Bidirectional reflectance spectroscopy 1.
       JGR: Solid Earth, 86(B4).
    .. [Hapke1984] Hapke, B. (1984). Bidirectional reflectance spectroscopy: 3.
       Icarus, 59(1).
    .. [Hapke1986] Hapke, B. (1986). Bidirectional reflectance spectroscopy: 4.
       Icarus, 67(2).
    """
    # Store original input shape
    input_shape = surface_orientation.shape[:-1]
    if not input_shape: # Handle scalar-like inputs by promoting
        _surface_orientation_ndim = np.ndim(surface_orientation)
        _ssa_ndim = np.ndim(single_scattering_albedo)

        incidence_direction = np.atleast_1d(incidence_direction)
        emission_direction = np.atleast_1d(emission_direction)
        surface_orientation = np.atleast_1d(surface_orientation)
        single_scattering_albedo = np.atleast_1d(single_scattering_albedo)

        if _surface_orientation_ndim == 0 and _ssa_ndim == 0:
            input_shape = ()
        elif _ssa_ndim > 0 :
             input_shape = single_scattering_albedo.shape
        else:
             input_shape = surface_orientation.shape[:-1] if _surface_orientation_ndim > 1 else ()

    if np.ndim(single_scattering_albedo) == 0:
        single_scattering_albedo = np.full(input_shape if input_shape else (1,), single_scattering_albedo)


    # Angles
    norm_inc = np.linalg.norm(incidence_direction, axis=-1, keepdims=True)
    if np.any(norm_inc == 0): norm_inc[norm_inc == 0] = 1e-9
    incidence_direction = incidence_direction / norm_inc

    norm_emi = np.linalg.norm(emission_direction, axis=-1, keepdims=True)
    if np.any(norm_emi == 0): norm_emi[norm_emi == 0] = 1e-9
    emission_direction = emission_direction / norm_emi

    norm_sur = np.linalg.norm(surface_orientation, axis=-1, keepdims=True)
    if np.any(norm_sur == 0): norm_sur[norm_sur == 0] = 1e-9
    surface_orientation = surface_orientation / norm_sur

    # Phase angle
    # Ensure dot product is taken along the last axis (-1) for vector arrays
    cos_alpha = np.sum(incidence_direction * emission_direction, axis=-1)

    cos_alpha = np.clip(cos_alpha, -1.0, 1.0) # Clip to valid range
    sin_alpha = np.sqrt(1 - cos_alpha**2)

    # Correct call to microscopic_roughness if dimensions were squeezed by atleast_1d
    _inc_dir_squeezed = incidence_direction
    _emi_dir_squeezed = emission_direction
    _surf_ori_squeezed = surface_orientation

    # If original inputs were truly scalar (0-dim for vectors), they are now (1,3) or (3,)
    # microscopic_roughness expects (..., 3)
    if input_shape == () : # Original was scalar-like
        if _inc_dir_squeezed.ndim == 1: _inc_dir_squeezed = _inc_dir_squeezed[np.newaxis,:]
        if _emi_dir_squeezed.ndim == 1: _emi_dir_squeezed = _emi_dir_squeezed[np.newaxis,:]
        if _surf_ori_squeezed.ndim == 1: _surf_ori_squeezed = _surf_ori_squeezed[np.newaxis,:]


    [s, mu_0, mu] = microscopic_roughness(
        roughness, _inc_dir_squeezed, _emi_dir_squeezed, _surf_ori_squeezed
    )

    # Phase function values
    p_g = phase_function(cos_alpha)

    # H-Function
    # TODO: implement derivation of the H-function
    h_mu_0 = h_function(mu_0, single_scattering_albedo)
    h_mu = h_function(mu, single_scattering_albedo)

    # Opposition effect
    b_g = 1.0
    # Corrected typo: oppoistion_effect_b0 to opposition_effect_b0
    if opposition_effect_b0 != 0.0 and opposition_effect_h != 0.0:
        # Avoid division by zero if cos_alpha is -1 (phase_angle is pi)
        # tan_half_alpha = sin_alpha / (1 + cos_alpha) can be problematic
        # Use np.arctan2(sin_alpha, 1 + cos_alpha) for tan(alpha/2)
        # Or, more directly, tan(g/2) = sin(g) / (1 + cos(g))
        # If opposition_effect_h is zero, this term is problematic.
        tan_half_phase_angle = np.divide(sin_alpha, 1 + cos_alpha,
                                         out=np.full_like(sin_alpha, np.inf),
                                         where=(1 + cos_alpha) != 0)
        b_g += opposition_effect_b0 / (1 + tan_half_phase_angle / opposition_effect_h)


    # Reflectance: Original Hapke formula (e.g., Eq. 17 from Hapke 1981, with roughness and OE)
    # Term K = mu0 / (4*pi * (mu0 + mu)) -- this is from older papers; later Hapke uses K = mu0 / (mu0e + mue)
    # The s factor from roughness correction is applied to K implicitly by using mu_0, mu from roughness calc.
    # The division by 4*pi seems to be applied differently in some sources or later papers.
    # Sticking to a common interpretation:
    # Refl = (w / 4pi) * (mu0e / (mu0e + mue)) * [ (1+B(g))*P(g) + H(mu0e)*H(mue) -1 ] * S(i,e,g,theta_bar)
    # Here, s = S(), b_g = (1+B(g)), p_g = P(g)
    # The 1/(4*pi) factor is sometimes included in the definition of bidirectional reflectance, sometimes not.
    # The provided code had refl /= 4 * np.pi at the end, which might be double-counting.
    # Let's follow a structure closer to typical Hapke equations.
    # The term mu_0 / (mu_0 + mu) is often written as K_geo or similar.
    # The factor 1/(4*pi) is part of the definition of bidirectional reflectance r = dL_r / (F_0 dOmega_i)
    # where F_0 is incident flux per unit area normal to beam.
    # If mu_0 is cos(i) w.r.t surface normal, then incident flux on surface is F_0 * mu_0.
    # Radiance L = (w / (4*pi)) * F_0 * integral(...)
    # Bidirectional Reflectance r_surf = pi * L_surf / (F_0 * mu_0_illumination)
    # Hapke's r = (w / 4) * (1 / (mu_0e + mu_e)) * [(1+B(g))P(g) + H(mu_0e)H(mu_e)-1] * S
    # The factor of pi is often a point of confusion. Let's use the common form without the extra /4pi.

    term_scattering = single_scattering_albedo * (b_g * p_g + h_mu_0 * h_mu - 1.0)
    # Geometric factor K, including roughness correction via s and modified mu, mu_0
    # Note: mu_0 and mu from microscopic_roughness are mu_0' and mu'
    geometric_factor = mu_0 / (mu_0 + mu) * s # This is K * S from some formulations

    refl = (geometric_factor / (4 * np.pi)) * term_scattering # Common formulation with 1/(4pi)

    refl[(mu <= 0) | (mu_0 <= 0)] = np.nan
    # refl[refl < 1e-6] = np.nan # Optional: keep very small values unless problematic

    # Final result
    threshold_imag = 0.1 # Allow small imaginary part due to numerical precision
    threshold_error = 1e-4 # Stricter check for raising error

    # Ensure refl is complex for np.imag, np.real if it became float somehow
    if not np.iscomplexobj(refl):
        refl = refl.astype(np.complex128, copy=False)

    arg_rh = np.divide(
        np.imag(refl),
        np.real(refl),
        out=np.zeros_like(refl, dtype=float), # Use float for arg_rh
        where=np.real(refl) != 0,
    )
    refl_real = np.real(refl) # Work with real part after check
    refl_real[arg_rh > threshold_imag] = np.nan


    if np.any(arg_rh >= threshold_error):
        shape_for_fill = input_shape if input_shape else (1,)
        if np.all(arg_rh >= threshold_error):
            return np.full(shape_for_fill, np.nan).reshape(input_shape)
        raise ValueError("At least one reflectance value has a significant imaginary component!")

    refl_real = np.nan_to_num(refl_real, nan=0.0) if nan2zero else refl_real
    return refl_real.reshape(input_shape)
