refmod.utils.spectral_continuum
===============================

.. py:module:: refmod.utils.spectral_continuum

.. autoapi-nested-parse::

   Continuum removal utilities.

   Implements an "upper convex hull" continuum commonly used in spectroscopy:

   - Treat each band as a 2D point (x=λ, y=R)
   - Compute the *upper* convex envelope
   - Linearly interpolate this envelope to obtain a continuum curve
   - Divide the spectrum by the continuum to remove the continuum slope

   Two backends are supported:

   - ``method='numpy'`` (default): deterministic monotone-chain upper hull.
   - ``method='scipy'``: uses ``scipy.spatial.ConvexHull`` to obtain the convex
     polygon, then extracts the upper chain.







Module Contents
---------------

.. py:data:: _Method

.. py:function:: _upper_monotone_chain(xv, yv)

   Compute upper monotone chain indices using Andrew's monotone chain algorithm.

   Assumes xv is strictly increasing.


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


