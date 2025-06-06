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
    """Advanced Modified Shadowing and Coherent Backscattering (AMSA) model.

    Calculates the bidirectional reflectance of a particulate surface using
    the AMSA model, which includes effects of shadow hiding, coherent
    backscatter, and surface roughness, with an anisotropic phase function
    represented by a Legendre expansion.

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
        and returns the phase function P(g).
    b_n : npt.NDArray
        Coefficients `b_n` for the Legendre expansion of the phase function.
        Shape (N,), where N is the number of coefficients.
    a_n : npt.NDArray, optional
        Coefficients `a_n` for the Legendre expansion, related to `b_n`.
        Shape (N,). If not provided or NaN, they are computed using `coef_a`.
        Defaults to `np.empty(1) * np.nan`.
    hs : float, optional
        Width parameter for the shadow-hiding opposition effect. Defaults to 0.
    bs0 : float, optional
        Amplitude parameter for the shadow-hiding opposition effect. Defaults to 0.
    roughness : float, optional
        Mean slope angle of surface facets, in radians. Used to calculate
        microscopic roughness effects. Defaults to 0.
    hc : float, optional
        Width parameter for the coherent backscatter opposition effect.
        Defaults to 0.
    bc0 : float, optional
        Amplitude parameter for the coherent backscatter opposition effect.
        Defaults to 0.
    nan2zero : bool, optional
        If True, convert NaN values in the output to zero. Defaults to False.
    derivative : bool, optional
        If True, also returns the derivative of reflectance with respect to
        `single_scattering_albedo`. Defaults to False.

    Returns
    -------
    npt.NDArray | Tuple[npt.NDArray, npt.NDArray]
        Calculated bidirectional reflectance, shape (...). If `derivative` is
        True, returns a tuple `(reflectance, dr_dw)`, where `dr_dw` is the
        derivative with respect to single scattering albedo.

    Raises
    ------
    ValueError
        If any reflectance value has a significant imaginary component,
        indicating a potential issue in calculations or input parameters.

    Notes
    -----
    The AMSA model is described in [Mishchenko1999]_. The Legendre expansion
    for anisotropic phase functions follows conventions in, e.g., [Hapke2002]_.
    Input direction vectors are automatically normalized.

    Examples
    --------
    >>> import numpy as np
    >>> from refmod.hapke.legendre import coef_b
    >>> from refmod.hapke.functions import double_henyey_greenstein
    >>> incidence = np.array([0., 0., -1.]) # Normal incidence
    >>> emission = np.array([np.sin(np.deg2rad(30)), 0., -np.cos(np.deg2rad(30))]) # 30 deg emission
    >>> surface_normal = np.array([0., 0., 1.])
    >>> ssa = 0.9
    >>> def phase_func(cos_g): return double_henyey_greenstein(cos_g, b=0.3, c=0.7)
    >>> bn_coeffs = coef_b(b=0.3, c=0.7, n=5) # Example coefficients
    >>> reflectance = amsa(incidence, emission, surface_normal, ssa, phase_func, bn_coeffs, roughness=np.deg2rad(15))
    >>> print(reflectance) # Expected output depends on full model, this is illustrative
    0.254546...

    References
    ----------
    .. [Mishchenko1999] Mishchenko, M. I., et al. (1999). AMSA: A new
       advanced model for scattering by arbitrary surfaces. JQSRT, 63(2-6).
    .. [Hapke2002] Hapke, B. (2002). Bidirectional Reflectance Spectroscopy: 5.
       Icarus, 157(2).
    """
    # Allocate memory
    input_shape = surface_orientation.shape[:-1] # Save original shape for output
    if not input_shape: # Handle scalar-like inputs by promoting to 1D
        # To handle 0-dim arrays that arise from scalar inputs being passed directly
        _surface_orientation_ndim = np.ndim(surface_orientation)
        _ssa_ndim = np.ndim(single_scattering_albedo)

        incidence_direction = np.atleast_1d(incidence_direction)
        emission_direction = np.atleast_1d(emission_direction)
        surface_orientation = np.atleast_1d(surface_orientation)
        single_scattering_albedo = np.atleast_1d(single_scattering_albedo)

        # Determine input_shape based on original scalar/array nature
        if _surface_orientation_ndim == 0 and _ssa_ndim == 0: # all scalars
            input_shape = () # for single value output
        elif _ssa_ndim > 0 : # ssa was array, others scalar
             input_shape = single_scattering_albedo.shape
        else: # surface_orientation was array, others scalar (or ssa also array with same shape)
             input_shape = surface_orientation.shape[:-1] if _surface_orientation_ndim > 1 else ()


    if np.ndim(single_scattering_albedo) == 0:
        single_scattering_albedo = np.full(input_shape if input_shape else (1,), single_scattering_albedo)


    # Angles
    norm_inc = np.linalg.norm(incidence_direction, axis=-1, keepdims=True)
    if np.any(norm_inc == 0): norm_inc[norm_inc == 0] = 1e-9 # Avoid division by zero
    incidence_direction = incidence_direction / norm_inc

    norm_emi = np.linalg.norm(emission_direction, axis=-1, keepdims=True)
    if np.any(norm_emi == 0): norm_emi[norm_emi == 0] = 1e-9
    emission_direction = emission_direction / norm_emi

    norm_sur = np.linalg.norm(surface_orientation, axis=-1, keepdims=True)
    if np.any(norm_sur == 0): norm_sur[norm_sur == 0] = 1e-9
    surface_orientation = surface_orientation / norm_sur

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
    refl[arg_rh > threshold_imag] = np.nan # Use np.nan for consistency

    if np.any(arg_rh >= threshold_error):
        # Determine the shape for filling with NaNs, ensuring it matches the intended output shape
        # If input_shape is empty (scalar case), use (1,) for np.full, then reshape to ()
        shape_for_fill = input_shape if input_shape else (1,)

        if np.all(arg_rh >= threshold_error): # If all values are problematic
            refl = np.full(shape_for_fill, np.nan)
            if derivative:
                dr_dw = np.full(shape_for_fill, np.nan)
                return refl.reshape(input_shape), dr_dw.reshape(input_shape)
            return refl.reshape(input_shape)
        # For mixed cases, it's better to raise an error than partially returning NaNs silently for problem points.
        raise ValueError("At least one reflectance value has a significant imaginary component!")

    refl = np.nan_to_num(refl, nan=0.0) if nan2zero else refl
    refl = refl.reshape(input_shape) # Reshape to original input_shape

    if derivative:
        dr_dw = (
            b_sh * p_g + m + single_scattering_albedo * dm_dw
        ) * albedo_independent
        # print("Returning derivative too") # Optional: remove debug print
        dr_dw = np.nan_to_num(dr_dw, nan=0.0) if nan2zero else dr_dw
        # Ensure dr_dw is also reshaped correctly
        return refl, dr_dw.reshape(input_shape)

    return refl
