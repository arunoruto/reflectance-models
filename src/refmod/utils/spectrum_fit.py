"""Spectrum fitting helpers.

This module provides small utilities for fitting combinations of spectra to a
target spectrum using :func:`scipy.optimize.least_squares`.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.optimize import OptimizeResult, least_squares


@dataclass(frozen=True)
class SpectrumFitResult:
    """Result for a linear spectrum combination fit.

    Attributes
    ----------
    weights:
        Fitted weights for each basis component.
    pred:
        Best-fit prediction (raw spectrum domain).
    pred_continuum_removed:
        Continuum-removed version of the prediction (only if ``continuum_alpha < 1.0``).
    rmse_abs:
        Absolute RMSE between prediction and target (raw domain).
    rmse_rel:
        Relative RMSE between prediction and target (raw domain).
    rmse_abs_continuum_removed:
        Absolute RMSE in the continuum-removed domain (only if ``continuum_alpha < 1.0``).
    rmse_rel_continuum_removed:
        Relative RMSE in the continuum-removed domain (only if ``continuum_alpha < 1.0``).
    alpha:
        The blending factor used (clipped to [0, 1]).
    rmse_rel_weighted:
        Weighted combined relative RMSE:
        ``alpha * rmse_rel + (1 - alpha) * rmse_rel_continuum_removed``.
    optimize_result:
        Full result object from :func:`scipy.optimize.least_squares`.
    """

    weights: np.ndarray
    pred: np.ndarray
    pred_continuum_removed: np.ndarray | None
    rmse_abs: float
    rmse_rel: float
    rmse_abs_continuum_removed: float | None
    rmse_rel_continuum_removed: float | None
    alpha: float
    rmse_rel_weighted: float
    optimize_result: OptimizeResult


def fit_linear_spectrum_combination(
    basis: np.ndarray,
    target: np.ndarray,
    *,
    wavelength: np.ndarray | None = None,
    weights0: np.ndarray | None = None,
    bounds: tuple[np.ndarray | float, np.ndarray | float] = (0.0, np.inf),
    fit_mask: np.ndarray | None = None,
    relative_error: bool = True,
    eps: float = 1e-12,
    continuum_alpha: float = 1.0,
    continuum_smooth: bool = True,
    continuum_smooth_mu: float = 1000.0,
    continuum_method: str = "numpy",
    sum_to_one: bool = False,
    sum_to_one_weight: float = 1.0,
    method: str = "trf",
    loss: str = "linear",
) -> SpectrumFitResult:
    """Fit a linear combination of spectra to a target.

    Solves for weights ``w`` in ``pred = basis @ w`` that best match ``target``.

    Important
    ---------
    If ``relative_error=True`` (recommended for reflectance spectra), the
    optimized residual uses *relative* error:

    ``(pred - target) / max(|target|, eps)``.

    Parameters
    ----------
    basis:
        Basis spectra. Shape ``(n_wavelengths, n_components)`` or
        ``(n_components, n_wavelengths)``.
    target:
        Target spectrum. Shape ``(n_wavelengths,)``.
    weights0:
        Optional initial weights. Defaults to uniform weights.
    bounds:
        Bounds passed to :func:`scipy.optimize.least_squares`.
    fit_mask:
        Optional boolean mask selecting wavelengths used for fitting.
    relative_error:
        If True, residuals are normalized by the target spectrum.
    eps:
        Small value used in the relative error denominator.
    continuum_alpha:
        Weighting between fitting the raw spectrum and the continuum-removed
        spectrum. Values outside ``[0, 1]`` are clipped (with a warning).
        ``1.0`` (default) fits only the raw spectrum.
    continuum_smooth, continuum_smooth_mu, continuum_method:
        Parameters forwarded to :func:`refmod.utils.spectral_continuum.continuum_remove_upper_hull`.
    sum_to_one:
        If True, appends an extra residual enforcing ``sum(weights)=1``.
    sum_to_one_weight:
        Weight of the sum-to-one constraint residual.
    method, loss:
        Passed through to :func:`scipy.optimize.least_squares`.

    Returns
    -------
    SpectrumFitResult
        Contains fitted weights, best-fit prediction, and RMSE metrics.
    """

    alpha_in = float(continuum_alpha)
    if not np.isfinite(alpha_in):
        warnings.warn(
            f"continuum_alpha is not finite ({continuum_alpha!r}); using 1.0",
            RuntimeWarning,
            stacklevel=2,
        )
        alpha = 1.0
    else:
        alpha = float(np.clip(alpha_in, 0.0, 1.0))
        if alpha != alpha_in:
            warnings.warn(
                f"continuum_alpha={alpha_in} is outside [0, 1]; clipped to {alpha}",
                RuntimeWarning,
                stacklevel=2,
            )

    if alpha < 1.0 and not bool(continuum_smooth):
        warnings.warn(
            "continuum_smooth=False with continuum_alpha<1.0 can make the upper-hull "
            "continuum (and thus the objective) more sensitive to noise/outliers and "
            "harder to optimize",
            RuntimeWarning,
            stacklevel=2,
        )

    basis_in = np.asarray(basis, dtype=float)
    target_in = np.asarray(target, dtype=float).reshape(-1)
    if basis_in.ndim != 2:
        raise ValueError("basis must be a 2D array")
    if target_in.ndim != 1:
        raise ValueError("target must be a 1D array")

    if basis_in.shape[0] == target_in.size:
        a = basis_in
    elif basis_in.shape[1] == target_in.size:
        a = basis_in.T
    else:
        raise ValueError(
            "basis must have shape (n_wavelengths, n_components) or (n_components, n_wavelengths)"
        )

    n_wl, n_comp = a.shape
    if n_wl != target_in.size:
        raise ValueError("basis and target wavelength dimension mismatch")

    if wavelength is None:
        wl = np.arange(n_wl, dtype=float)
    else:
        wl = np.asarray(wavelength, dtype=float).reshape(-1)
        if wl.size != n_wl:
            raise ValueError("wavelength must have same length as target")

    continuum_method_s = str(continuum_method).lower().strip()
    if continuum_method_s not in {"numpy", "scipy"}:
        raise ValueError("continuum_method must be 'numpy' or 'scipy'")

    if weights0 is None:
        x0 = np.full(n_comp, 1.0 / float(n_comp), dtype=float)
    else:
        x0 = np.asarray(weights0, dtype=float).reshape(-1)
        if x0.size != n_comp:
            raise ValueError("weights0 must have length n_components")

    finite = np.isfinite(target_in) & np.all(np.isfinite(a), axis=1)
    if fit_mask is not None:
        fm = np.asarray(fit_mask, dtype=bool).reshape(-1)
        if fm.size != target_in.size:
            raise ValueError("fit_mask must have same length as target")
        finite = finite & fm

    if not np.any(finite):
        raise ValueError("No finite samples to fit")

    a_fit = a[finite, :]
    y_fit = target_in[finite]
    n_fit = int(np.count_nonzero(finite))
    fit_scale = 1.0 / float(np.sqrt(n_fit))

    use_continuum = bool(alpha < 1.0)
    if use_continuum:
        from .spectral_continuum import continuum_remove_upper_hull

        y_cr, _, _ = continuum_remove_upper_hull(
            wl,
            target_in,
            eps=float(eps),
            smooth=bool(continuum_smooth),
            smooth_mu=float(continuum_smooth_mu),
            assume_sorted=False,
            method=continuum_method_s,  # type: ignore[arg-type]
        )
        y_cr_fit = np.asarray(y_cr, dtype=float)[finite]
        # If target continuum removal has non-finite values in the fit window,
        # we keep them in the denominator handling below; the finite mask is
        # based on the raw spectrum and basis only.
    else:
        y_cr_fit = None

    def _residual(x: np.ndarray) -> np.ndarray:
        pred_fit = a_fit @ x
        r_raw = pred_fit - y_fit
        if relative_error:
            denom = np.maximum(np.abs(y_fit), float(eps))
            r_raw = r_raw / denom
        r_raw = np.asarray(r_raw, dtype=float).reshape(-1) * fit_scale

        if use_continuum:
            # Compute continuum removal on the full prediction to keep the hull
            # consistent with the input spectrum (then select fit window).
            pred_full = a @ x
            cr_pred, _, _ = continuum_remove_upper_hull(
                wl,
                np.asarray(pred_full, dtype=float).reshape(-1),
                eps=float(eps),
                smooth=bool(continuum_smooth),
                smooth_mu=float(continuum_smooth_mu),
                assume_sorted=False,
                method=continuum_method_s,  # type: ignore[arg-type]
            )
            cr_pred_fit = np.asarray(cr_pred, dtype=float)[finite]
            r_cr = cr_pred_fit - np.asarray(y_cr_fit, dtype=float)
            if relative_error:
                denom_cr = np.maximum(
                    np.abs(np.asarray(y_cr_fit, dtype=float)), float(eps)
                )
                r_cr = r_cr / denom_cr
            r_cr = np.asarray(r_cr, dtype=float).reshape(-1) * fit_scale

            r = np.concatenate(
                [
                    np.sqrt(alpha) * r_raw,
                    np.sqrt(1.0 - alpha) * r_cr,
                ]
            )
        else:
            r = r_raw

        if sum_to_one:
            r = np.concatenate(
                [
                    np.asarray(r, dtype=float).reshape(-1),
                    np.array([float(sum_to_one_weight) * (float(np.sum(x)) - 1.0)]),
                ]
            )
        return np.asarray(r, dtype=float).reshape(-1)

    sol = least_squares(_residual, x0=x0, bounds=bounds, method=method, loss=loss)

    w = np.asarray(sol.x, dtype=float).reshape(-1)
    pred = (a @ w).reshape(-1)

    diff = pred - target_in
    rmse_abs = float(np.sqrt(np.nanmean(diff**2)))
    denom_full = np.maximum(np.abs(target_in), float(eps))
    rmse_rel = float(np.sqrt(np.nanmean((diff / denom_full) ** 2)))

    pred_cr = None
    rmse_abs_cr = None
    rmse_rel_cr = None
    if use_continuum:
        from .spectral_continuum import continuum_remove_upper_hull

        target_cr, _, _ = continuum_remove_upper_hull(
            wl,
            target_in,
            eps=float(eps),
            smooth=bool(continuum_smooth),
            smooth_mu=float(continuum_smooth_mu),
            assume_sorted=False,
            method=continuum_method_s,  # type: ignore[arg-type]
        )
        pred_cr, _, _ = continuum_remove_upper_hull(
            wl,
            pred,
            eps=float(eps),
            smooth=bool(continuum_smooth),
            smooth_mu=float(continuum_smooth_mu),
            assume_sorted=False,
            method=continuum_method_s,  # type: ignore[arg-type]
        )
        target_cr = np.asarray(target_cr, dtype=float).reshape(-1)
        pred_cr = np.asarray(pred_cr, dtype=float).reshape(-1)

        diff_cr = pred_cr - target_cr
        rmse_abs_cr = float(np.sqrt(np.nanmean(diff_cr**2)))
        denom_cr_full = np.maximum(np.abs(target_cr), float(eps))
        rmse_rel_cr = float(np.sqrt(np.nanmean((diff_cr / denom_cr_full) ** 2)))

    rmse_rel_weighted = alpha * float(rmse_rel) + (1.0 - alpha) * float(
        rmse_rel_cr if rmse_rel_cr is not None else rmse_rel
    )

    return SpectrumFitResult(
        weights=w,
        pred=pred,
        pred_continuum_removed=pred_cr,
        rmse_abs=rmse_abs,
        rmse_rel=rmse_rel,
        rmse_abs_continuum_removed=rmse_abs_cr,
        rmse_rel_continuum_removed=rmse_rel_cr,
        alpha=alpha,
        rmse_rel_weighted=float(rmse_rel_weighted),
        optimize_result=sol,
    )
