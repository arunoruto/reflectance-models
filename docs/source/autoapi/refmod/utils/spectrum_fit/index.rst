refmod.utils.spectrum_fit
=========================

.. py:module:: refmod.utils.spectrum_fit

.. autoapi-nested-parse::

   Spectrum fitting helpers.

   This module provides small utilities for fitting combinations of spectra to a
   target spectrum using :func:`scipy.optimize.least_squares`.







Module Contents
---------------

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


