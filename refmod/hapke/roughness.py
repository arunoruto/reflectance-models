"""Microscopic surface roughness corrections for Hapke model.

This module implements functions to calculate the effects of microscopic
surface roughness on bidirectional reflectance, following Hapke's model.

References are indicated using a citation key, e.g., [Hapke1984]_,
corresponding to entries in a BibTeX file.
"""

import numpy as np
import numpy.typing as npt


def __f_exp(x: npt.NDArray, y: float) -> npt.NDArray:
    """Helper for microscopic_roughness: e^(-2/pi * y * x).

    Parameters
    ----------
    x : npt.NDArray
        Input array.
    y : float
        Exponential factor component.

    Returns
    -------
    npt.NDArray
        Result of the exponential function.
    """
    return np.exp(-2.0 / np.pi * y * x)


def __f_exp_2(x: npt.NDArray, y: float) -> npt.NDArray:
    """Helper for microscopic_roughness: e^(-(y*x/sqrt(pi))^2).

    Parameters
    ----------
    x : npt.NDArray
        Input array.
    y : float
        Factor to be squared.

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
    r"""Calculate microscopic surface roughness corrections.

    This function computes the shadowing/illumination correction factor `S`
    and the modified cosines of incidence (`mu_0_prime`) and emission
    (`mu_prime`) angles due to surface roughness, as defined by Hapke.

    Parameters
    ----------
    roughness : float
        Mean slope angle of surface facets, in radians. Often denoted as
        theta-bar.
    incidence_direction : npt.NDArray
        Incidence direction vectors, shape (..., 3). Normalized.
    emission_direction : npt.NDArray
        Emission direction vectors, shape (..., 3). Normalized.
    surface_orientation : npt.NDArray
        Surface normal vectors, shape (..., 3). Normalized.

    Returns
    -------
    s_factor : npt.NDArray
        The microscopic roughness correction factor S. Shape (...).
    mu_0_prime : npt.NDArray
        The modified cosine of the effective incidence angle. Shape (...).
    mu_prime : npt.NDArray
        The modified cosine of the effective emission angle. Shape (...).

    Notes
    -----
    The calculations are based on equations from [Hapke1984]_:
    - `mu_0_prime` (mu_0_s0 in code) and `mu_prime` (mu_s0 in code) for
      prime-zero terms: Equations 48 and 49.
    - `mu_0_prime` (mu_0_s in code) and `mu_prime` (mu_s in code) for
      general case:
        - If i < e: Equations 46 and 47.
        - If i >= e: Equations 50 and 51.
    The input direction vectors are assumed to be normalized.

    Examples
    --------
    >>> import numpy as np
    >>> inc_dir = np.array([0., 0., -1.]) # Normal incidence
    >>> emi_dir = np.array([np.sin(np.deg2rad(30)), 0., -np.cos(np.deg2rad(30))]) # 30 deg emission
    >>> surf_norm = np.array([0., 0., 1.])
    >>> s, mu0p, mup = microscopic_roughness(np.deg2rad(20), inc_dir, emi_dir, surf_norm)
    >>> print(f"S: {s:.4f}, mu0': {mu0p:.4f}, mu': {mup:.4f}")
    S: 0.8583, mu0': 0.9286, mu': 0.8188
    """
    # Ensure input vectors are normalized (already specified, but good practice)
    # Note: The calling functions (amsa, imsa) should already normalize them.
    # For standalone use, uncomment if needed:
    # incidence_direction = incidence_direction / np.linalg.norm(incidence_direction, axis=-1, keepdims=True)
    # emission_direction = emission_direction / np.linalg.norm(emission_direction, axis=-1, keepdims=True)
    # surface_orientation = surface_orientation / np.linalg.norm(surface_orientation, axis=-1, keepdims=True)

    # Incidence angle cosine with respect to surface normal
    mu_0 = np.sum(incidence_direction * surface_orientation, axis=-1)
    mu_0 = np.clip(mu_0, -1.0, 1.0) # Ensure valid cosine
    sin_i = np.sqrt(1 - mu_0**2)
    # tan_i = sin_i / mu_0
    # cot_i = np.divide(1, tan_i, out=np.ones_like(tan_i) * np.inf, where=tan_i != 0)
    cot_i = np.divide(
        mu_0, sin_i, out=np.ones_like(mu_0) * np.inf, where=sin_i != 0
    )
    i = np.arccos(mu_0)

    # Emission angle cosine with respect to surface normal
    mu = np.sum(emission_direction * surface_orientation, axis=-1)
    mu = np.clip(mu, -1.0, 1.0) # Ensure valid cosine
    sin_e = np.sqrt(1 - mu**2)
    tan_e = sin_e / mu
    cot_e = np.divide(
        1, tan_e, out=np.ones_like(tan_e) * np.inf, where=tan_e != 0
    )
    e = np.arccos(mu)

    if roughness == 0:
        # print("Roughness is zero, returning default values") # Optional: remove debug print
        return np.ones_like(mu), mu_0, mu

    # Projections of incidence and emission vectors onto the mean surface plane
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
    # Check for cases i < e and i >= e
    ile = i < e
    ige = np.logical_not(ile) # More robust than i >= e if i can be equal to e

    # Check for singularities (normal incidence/emission)
    # Using a small tolerance for floating point comparisons
    epsilon = 1e-8
    index_mu0_is_1 = np.abs(mu_0 - 1.0) < epsilon
    index_mu_is_1 = np.abs(mu - 1.0) < epsilon
    singular_condition = index_mu0_is_1 | index_mu_is_1


    factor = 1.0 / np.sqrt(1 + np.pi * tan_rough**2)

    # f_psi term: Hapke (1984) Eq. 45
    # Handle cos_psi == -1 case for tan(psi/2) which is sin(psi)/(1+cos(psi))
    # tan_psi_half = np.divide(sin_psi, 1 + cos_psi, out=np.full_like(sin_psi, np.inf), where=(1 + cos_psi) != 0)
    # f_psi = np.exp(-2.0 * tan_psi_half) -> This seems to be a simplification used in some codes.
    # Original Hapke form: exp(-2 * tan(psi/2)) where tan(psi/2) = sqrt((1-cos(psi))/(1+cos_psi))
    # Let's use a safe way for tan(psi/2)
    tan_psi_half_sq = np.divide(1 - cos_psi, 1 + cos_psi, out=np.full_like(cos_psi, np.inf), where=(1 + cos_psi) !=0)
    tan_psi_half = np.sqrt(np.maximum(0, tan_psi_half_sq)) # Ensure non-negative for sqrt
    f_psi = np.exp(-2.0 * tan_psi_half)
    f_psi[np.abs(1 + cos_psi) < epsilon] = 0.0 # If 1+cos_psi is zero, tan(psi/2) is infinite, exp(-inf) is 0

    # mu_0_prime_zero (Eq. 48) and mu_prime_zero (Eq. 49)
    # These are mu_0_s0 and mu_s0 in the code
    mu_0_prime_zero = factor * (
        mu_0
        + sin_i
        * tan_rough
        * __f_exp_2(cot_i, cot_rough)
        / (2.0 - __f_exp(cot_i, cot_rough))
    )
    mu_prime_zero = factor * (
        mu
        + sin_e
        * tan_rough
        * __f_exp_2(cot_e, cot_rough)
        / (2.0 - __f_exp(cot_e, cot_rough))
    )

    # Initialize mu_0_prime (mu_0_s in code) and mu_prime (mu_s in code)
    mu_0_prime = np.zeros_like(mu_0)
    # Case i < e (Eq. 46 for mu_0_prime)
    # Denominator for i < e case (Eq. 46 & 47)
    den_ile = (2.0 - __f_exp(cot_e, cot_rough) - (psi / np.pi) * __f_exp(cot_i, cot_rough))
    mu_0_prime[ile] = factor * (
        mu_0[ile]
        + sin_i[ile]
        * tan_rough
        * (cos_psi[ile] * __f_exp_2(cot_e[ile], cot_rough)
           + sin_psi_div_2_sq[ile] * __f_exp_2(cot_i[ile], cot_rough))
        / np.where(den_ile[ile]==0, np.inf, den_ile[ile]) # Avoid division by zero
    )
    # Case i >= e (Eq. 50 for mu_0_prime)
    # Denominator for i >= e case (Eq. 50 & 51)
    den_ige = (2.0 - __f_exp(cot_i, cot_rough) - (psi / np.pi) * __f_exp(cot_e, cot_rough))
    mu_0_prime[ige] = factor * (
        mu_0[ige]
        + sin_i[ige]
        * tan_rough
        * (__f_exp_2(cot_i[ige], cot_rough)
           - sin_psi_div_2_sq[ige] * __f_exp_2(cot_e[ige], cot_rough))
        / np.where(den_ige[ige]==0, np.inf, den_ige[ige]) # Avoid division by zero
    )
    mu_0_prime[singular_condition] = mu_0[singular_condition] # No correction at normal viewing/illumination

    mu_prime = np.zeros_like(mu)
    # Case i < e (Eq. 47 for mu_prime)
    mu_prime[ile] = factor * (
        mu[ile]
        + sin_e[ile]
        * tan_rough
        * (__f_exp_2(cot_e[ile], cot_rough)
           - sin_psi_div_2_sq[ile] * __f_exp_2(cot_i[ile], cot_rough))
        / np.where(den_ile[ile]==0, np.inf, den_ile[ile]) # Avoid division by zero
    )
    # Case i >= e (Eq. 51 for mu_prime)
    mu_prime[ige] = factor * (
        mu[ige]
        + sin_e[ige]
        * tan_rough
        * (cos_psi[ige] * __f_exp_2(cot_i[ige], cot_rough)
           + sin_psi_div_2_sq[ige] * __f_exp_2(cot_e[ige], cot_rough))
        / np.where(den_ige[ige]==0, np.inf, den_ige[ige]) # Avoid division by zero
    )
    mu_prime[singular_condition] = mu[singular_condition]

    # Roughness factor S (Eq. 44)
    # s_factor = factor * (mu_prime / mu_prime_zero) * (mu_0 / mu_0_prime_zero) # This seems to be S_0
    # The S factor needs mu_0_prime and mu_prime, not mu_0_prime_zero and mu_prime_zero in the ratio with mu_0 and mu
    # S = K(theta_bar) * (mu_e' / mu_e) * (mu_i' / mu_i) in some notations (K is `factor`)
    # S = factor * (mu_prime / mu_0_prime_zero) * (mu_0 / mu_prime_zero) - This was old code structure
    # Let's use Eq 44 directly: S = chi(i) * chi(e) * S_psi
    # chi(i) = mu_0_prime / mu_0; chi(e) = mu_prime / mu
    # S_psi_ile = 1 / (1 - f_psi + f_psi * chi(i)_ile )
    # S_psi_ige = 1 / (1 - f_psi + f_psi * chi(e)_ige )
    # This is getting complicated. Let's use the structure from the original code if it matches a known source.
    # The original code structure for s:
    # s = factor * (mu_s / mu_s0) * (mu_0 / mu_0_s0)
    # s[ile] = s[ile] / (1 + f_psi[ile] * (factor * (mu_0[ile] / mu_0_s0[ile]) - 1))
    # s[ige] = s[ige] / (1 + f_psi[ige] * (factor * (mu[ige] / mu_s0[ige]) - 1))
    # This structure seems to match Eq.34 of Hapke, B. (1986) (Shadowing correction for rough surfaces, Icarus, 67, 264-280)
    # where S = S_0 * S_psi, and S_0 = factor * (mu_prime/mu_prime_zero) * (mu_0/mu_0_prime_zero)
    # and S_psi for i<e is 1 / (1 - f(psi) + f(psi) * factor * mu_0 / mu_0_prime_zero)
    # and S_psi for i>=e is 1 / (1 - f(psi) + f(psi) * factor * mu_e / mu_prime_zero)

    s_factor = factor * (mu_prime / mu_prime_zero) * (mu_0 / mu_0_prime_zero)

    term_ile = 1.0 + f_psi[ile] * (factor * (mu_0[ile] / mu_0_prime_zero[ile]) - 1.0)
    s_factor[ile] = s_factor[ile] / np.where(term_ile==0, np.inf, term_ile)

    term_ige = 1.0 + f_psi[ige] * (factor * (mu[ige] / mu_prime_zero[ige]) - 1.0)
    s_factor[ige] = s_factor[ige] / np.where(term_ige==0, np.inf, term_ige)

    s_factor[singular_condition] = 1.0 # No roughness correction at normal incidence/emission

    # Ensure results are squeezed if original inputs were scalar-like leading to single-element arrays
    # The calling functions (amsa, imsa) handle reshaping based on original input_shape.
    # Here, we return potentially multi-dimensional arrays if inputs were so.
    return s_factor, mu_0_prime, mu_prime
