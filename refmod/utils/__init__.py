"""Miscellaneous utilities.

This subpackage contains small, reusable helpers that don't belong to a specific
reflectance model implementation.
"""

from .spectral_continuum import (  # noqa: F401
    continuum_remove_upper_hull,
    smooth_spectrum_m3,
    upper_hull_continuum,
)
from .spectrum_fit import SpectrumFitResult, fit_linear_spectrum_combination  # noqa: F401

__all__ = [
    "smooth_spectrum_m3",
    "upper_hull_continuum",
    "continuum_remove_upper_hull",
    "SpectrumFitResult",
    "fit_linear_spectrum_combination",
]
