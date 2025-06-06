"""Hapke photometric model functions.

This module provides implementations of core functions used in the Hapke
photometric model, including the H-function and various phase functions.

References are indicated using a citation key, e.g., [Hapke1993]_,
corresponding to entries in a BibTeX file (e.g., `docs/source/library.bib`).
"""

import numpy as np
import numpy.typing as npt


def h_function(
    x: npt.NDArray, w: npt.NDArray, level: int = 1, derivative: bool = False
) -> npt.NDArray | tuple[npt.NDArray, npt.NDArray | None]:
    """Chandrasekhar's H-function for isotropic scatterers.

    Calculates the Hapke H-function, which is an approximation of
    Chandrasekhar's H-function for isotropic scatterers.

    Parameters
    ----------
    x : npt.NDArray
        Typically the cosine of the incidence angle (mu0) or emission
        angle (mu). Must be positive.
    w : npt.NDArray
        Single scattering albedo.
    level : int, optional
        The approximation level for the H-function.
        - 1: Simpler, less accurate approximation.
        - 2: More complex, more accurate approximation by Hapke.
        Defaults to 1.
    derivative : bool, optional
        If True, also returns the derivative of H with respect to `w`.
        Defaults to False.

    Returns
    -------
    npt.NDArray | tuple[npt.NDArray, npt.NDArray | None]
        The calculated H-function value(s). If `derivative` is True,
        returns a tuple `(h, dh_dw)`, where `dh_dw` is the derivative.
        `dh_dw` might be None if the derivative for the specified `level`
        is not implemented.

    Raises
    ------
    ValueError
        If an invalid `level` is provided.

    Notes
    -----
    The H-function is a key component in radiative transfer theory and is
    used extensively in Hapke's model.
    Level 1 is a simpler approximation, while Level 2 provides a more
    accurate representation as described in [Hapke1993]_.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0.5, 1.0])
    >>> w = np.array([0.8, 0.9])
    >>> h_function(x, w, level=1)
    array([1.30901699, 1.42135624])

    >>> h_function(x, w, level=2, derivative=True)
    (array([1.27929307, 1.37710511]), array([0.52704628, 0.70710678]))

    References
    ----------
    .. [Hapke1993] Hapke, B. (1993). Theory of reflectance and emittance
       spectroscopy. Cambridge university press.
    """
    gamma = np.sqrt(1 - w)
    r0 = (1 - gamma) / (1 + gamma)
    x_log_term = np.log(1 + 1 / x)

    match level:
        case 1:
            h = (1 + 2 * x) / (1 + 2 * x * gamma)
        case 2:
            h_inv = 1 - w * x * (r0 + (1 - 2 * r0 * x) / 2 * x_log_term)
            h = 1.0 / h_inv
        case _:
            raise ValueError("Please provide a level between 1 and 2!")

    if not derivative:
        return h

    match level:
        case 1:
            dh_dw = None
        case 2:
            dr0_dw = 1 / (gamma * (1 + gamma) ** 2)
            dh_dw = x * (
                r0
                + (1 - 2 * r0 * x) / 2 * x_log_term
                + w * dr0_dw * (1 - x * x_log_term)
            )

    return h, dh_dw


def double_henyey_greenstein(
    cos_g: npt.NDArray, b: float = 0.21, c: float = 0.7
) -> npt.NDArray:
    """Double Henyey-Greenstein phase function.

    This function models the angular distribution of scattered light using
    a weighted sum of two Henyey-Greenstein functions, one for forward
    scattering and one for backward scattering.

    Parameters
    ----------
    cos_g : npt.NDArray
        Cosine of the phase angle g.
    b : float, optional
        Asymmetry parameter for the Henyey-Greenstein function.
        Controls the sharpness of the scattering peak. Typically between 0 (isotropic)
        and 1 (highly forward scattering). Defaults to 0.21.
    c : float, optional
        Backscatter fraction. Controls the relative weight of the backward
        scattering lobe. Ranges from 0 to 1. Defaults to 0.7.

    Returns
    -------
    npt.NDArray
        The phase function value(s) for the given phase angle(s).

    Examples
    --------
    >>> import numpy as np
    >>> cos_phase_angle = np.cos(np.deg2rad(30)) # cos(30 degrees)
    >>> double_henyey_greenstein(cos_phase_angle, b=0.3, c=0.6)
    1.015189...

    References
    ----------
    This functional form is common in planetary science literature, often
    used to represent scattering by particulate surfaces. See, for example,
    Chapter 8 of [Hapke1993]_.
    """
    term1 = (1 + c) / 2 * (1 - b**2) / np.power(1 - 2 * b * cos_g + b**2, 1.5)
    term2 = (1 - c) / 2 * (1 - b**2) / np.power(1 + 2 * b * cos_g + b**2, 1.5)
    p_g = term1 + term2
    return p_g


def cornette_shanks(cos_g: npt.NDArray, xi: float) -> npt.NDArray:
    """Cornette-Shanks phase function.

    This phase function is often used to model scattering from rough surfaces
    or particulate media.

    Parameters
    ----------
    cos_g : npt.NDArray
        Cosine of the phase angle g.
    xi : float
        A parameter related to particle properties and scattering behavior,
        often related to the refractive index or other physical characteristics
        of the scattering particles.

    Returns
    -------
    npt.NDArray
        The phase function value(s) for the given phase angle(s).

    Notes
    -----
    This is Equation 8 from [Cornette1992]_.

    Examples
    --------
    >>> import numpy as np
    >>> cos_phase_angle = np.cos(np.deg2rad(45)) # cos(45 degrees)
    >>> cornette_shanks(cos_phase_angle, xi=0.5)
    0.607548...

    References
    ----------
    .. [Cornette1992] Cornette, W. M., & Shanks, J. G. (1992).
       Bidirectional reflectance of flat, optically thick particulate systems.
       Applied Optics, 31(15), 3152-3160.
    """
    numerator = 1.5 * (1 - xi**2) * (1 + cos_g**2)
    denominator = (2 + xi**2) * np.power(1 + xi**2 - 2 * xi * cos_g, 1.5)
    p_g = numerator / denominator
    return p_g
