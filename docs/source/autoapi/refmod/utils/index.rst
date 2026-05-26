refmod.utils
============

.. py:module:: refmod.utils

.. autoapi-nested-parse::

   Miscellaneous utilities.

   This subpackage contains small, reusable helpers that don't belong to a specific
   reflectance model implementation.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/refmod/utils/spectral_continuum/index
   /autoapi/refmod/utils/spectrum_fit/index






Package Contents
----------------

.. py:function:: continuum_remove_upper_hull(wavelength, reflectance, *, eps = 1e-12, smooth = True, smooth_mu = 1000.0, assume_sorted = False, method = 'numpy')

   Remove continuum slope using an upper-hull continuum.

   The continuum-removed spectrum is computed as:
   ``reflectance / max(continuum, eps)``.

   :param wavelength: 1D wavelengths.
   :param reflectance: 1D reflectance values, same shape as wavelength.
   :param eps: Small value to avoid division by zero (or very small continuum values).
   :param smooth: If True, applies M3 smoothing (via :func:`smooth_spectrum_m3`) to the
                  reflectance *before* computing the hull. This reduces the sensitivity of
                  the upper hull to noise.
   :param smooth_mu: Regularization parameter for the M3 smoother (if ``smooth=True``).
   :param assume_sorted: If True, assumes wavelength is already increasing.
   :param method: "numpy" (default) or "scipy".

   :returns: * *continuum_removed* -- The continuum-removed spectrum.
             * *continuum* -- The computed continuum.
             * *hull_indices* -- Indices (into the input arrays) of the hull anchor points.


.. py:function:: smooth_spectrum_m3(wavelength, reflectance, *, mu = 1000.0, assume_sorted = False)

   Smooth a 1D spectrum using the M3 regularized smoother.

   This matches the approach from the M3 MATLAB code: solve a regularized
   least-squares problem with a scaled second-difference operator on an
   irregular wavelength grid.

   .. rubric:: Notes

   - The original MATLAB code assumes wavelength in nanometers with ~20-40 nm
     spacing. To keep ``mu`` comparable, we smooth in nm units (auto-convert if
     input looks like microns).
   - Endpoints are preserved via the identity term in the system.


.. py:function:: upper_hull_continuum(wavelength, reflectance, *, assume_sorted = False, method = 'numpy')

   Compute an upper-hull continuum for a spectrum.

   :param wavelength: 1D wavelengths.
   :param reflectance: 1D reflectance values, same shape as wavelength.
   :param assume_sorted: If True, assumes wavelength is already increasing.
   :param method: "numpy" (default) or "scipy".

   :returns: * *continuum* -- Continuum evaluated at every input wavelength.
             * *hull_indices* -- Indices (into the original input arrays) of hull anchor points.

   .. rubric:: Notes

   - Non-finite samples are ignored.
   - Duplicate wavelengths are collapsed (keeping max reflectance).
   - ``hull_indices`` are approximate when duplicates exist (first occurrence).


.. py:class:: SpectrumFitResult

   Result for a linear spectrum combination fit.

   .. attribute:: weights

      Fitted weights for each basis component.

   .. attribute:: pred

      Best-fit prediction (raw spectrum domain).

   .. attribute:: pred_continuum_removed

      Continuum-removed version of the prediction (only if ``continuum_alpha < 1.0``).

   .. attribute:: rmse_abs

      Absolute RMSE between prediction and target (raw domain).

   .. attribute:: rmse_rel

      Relative RMSE between prediction and target (raw domain).

   .. attribute:: rmse_abs_continuum_removed

      Absolute RMSE in the continuum-removed domain (only if ``continuum_alpha < 1.0``).

   .. attribute:: rmse_rel_continuum_removed

      Relative RMSE in the continuum-removed domain (only if ``continuum_alpha < 1.0``).

   .. attribute:: alpha

      The blending factor used (clipped to [0, 1]).

   .. attribute:: rmse_rel_weighted

      Weighted combined relative RMSE:
      ``alpha * rmse_rel + (1 - alpha) * rmse_rel_continuum_removed``.

   .. attribute:: optimize_result

      Full result object from :func:`scipy.optimize.least_squares`.


   .. py:attribute:: weights
      :type:  numpy.ndarray


   .. py:attribute:: pred
      :type:  numpy.ndarray


   .. py:attribute:: pred_continuum_removed
      :type:  numpy.ndarray | None


   .. py:attribute:: rmse_abs
      :type:  float


   .. py:attribute:: rmse_rel
      :type:  float


   .. py:attribute:: rmse_abs_continuum_removed
      :type:  float | None


   .. py:attribute:: rmse_rel_continuum_removed
      :type:  float | None


   .. py:attribute:: alpha
      :type:  float


   .. py:attribute:: rmse_rel_weighted
      :type:  float


   .. py:attribute:: optimize_result
      :type:  scipy.optimize.OptimizeResult


.. py:function:: fit_linear_spectrum_combination(basis, target, *, wavelength = None, weights0 = None, bounds = (0.0, np.inf), fit_mask = None, relative_error = True, eps = 1e-12, continuum_alpha = 1.0, continuum_smooth = True, continuum_smooth_mu = 1000.0, continuum_method = 'numpy', sum_to_one = False, sum_to_one_weight = 1.0, method = 'trf', loss = 'linear')

   Fit a linear combination of spectra to a target.

   Solves for weights ``w`` in ``pred = basis @ w`` that best match ``target``.

   .. important::

      If ``relative_error=True`` (recommended for reflectance spectra), the
      optimized residual uses *relative* error:

      ``(pred - target) / max(|target|, eps)``.

   :param basis: Basis spectra. Shape ``(n_wavelengths, n_components)`` or
                 ``(n_components, n_wavelengths)``.
   :param target: Target spectrum. Shape ``(n_wavelengths,)``.
   :param weights0: Optional initial weights. Defaults to uniform weights.
   :param bounds: Bounds passed to :func:`scipy.optimize.least_squares`.
   :param fit_mask: Optional boolean mask selecting wavelengths used for fitting.
   :param relative_error: If True, residuals are normalized by the target spectrum.
   :param eps: Small value used in the relative error denominator.
   :param continuum_alpha: Weighting between fitting the raw spectrum and the continuum-removed
                           spectrum. Values outside ``[0, 1]`` are clipped (with a warning).
                           ``1.0`` (default) fits only the raw spectrum.
   :param continuum_smooth: Parameters forwarded to :func:`refmod.utils.spectral_continuum.continuum_remove_upper_hull`.
   :param continuum_smooth_mu: Parameters forwarded to :func:`refmod.utils.spectral_continuum.continuum_remove_upper_hull`.
   :param continuum_method: Parameters forwarded to :func:`refmod.utils.spectral_continuum.continuum_remove_upper_hull`.
   :param sum_to_one: If True, appends an extra residual enforcing ``sum(weights)=1``.
   :param sum_to_one_weight: Weight of the sum-to-one constraint residual.
   :param method: Passed through to :func:`scipy.optimize.least_squares`.
   :param loss: Passed through to :func:`scipy.optimize.least_squares`.

   :returns: Contains fitted weights, best-fit prediction, and RMSE metrics.
   :rtype: SpectrumFitResult


