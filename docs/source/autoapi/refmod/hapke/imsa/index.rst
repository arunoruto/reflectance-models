refmod.hapke.imsa
=================

.. py:module:: refmod.hapke.imsa






Module Contents
---------------

.. py:data:: EPS
   :value: 1e-15


.. py:function:: _refl_imsa_scalar(w, b_n, i, e, n, roughness)

   IMSA reflectance for a single pixel.

   Simplest Hapke model: isotropic multiple scattering with a Legendre
   polynomial phase function for single scattering.  No opposition effects
   (SHOE, CBOE) are included.

   :param w: Single scattering albedo (scalar).
   :type w: jax.Array
   :param b_n: Legendre polynomial coefficients for the single-scattering phase
               function.  Shape ``(n_coeffs,)``.
   :type b_n: jax.Array
   :param i: Incidence (illumination) direction vector.  Shape ``(3,)``.
   :type i: jax.Array
   :param e: Emission (viewing) direction vector.  Shape ``(3,)``.
   :type e: jax.Array
   :param n: Surface normal vector.  Shape ``(3,)``.
   :type n: jax.Array
   :param roughness: Macroscopic roughness angle in radians, :math:`\bar{\theta}`.
   :type roughness: float

   :returns: Reflectance (scalar).  Pixels with the source or detector behind the
             local horizon are set to NaN.
   :rtype: jax.Array

   .. rubric:: References

   .. bibliography::
      :filter: False

      Hapke-2012


.. py:data:: _imsa_batched

.. py:function:: imsa(w, b_n, i, e, n, roughness = 0.0)

   Batched IMSA reflectance.

   Vectorised wrapper around :func:`_refl_imsa_scalar` that evaluates the
   isotropic multiple-scattering Hapke model for an arbitrary number of
   pixels sharing the same Legendre coefficients and roughness.

   :param w: Single scattering albedo per pixel.  Shape ``(n_pixels,)``.
   :type w: jax.Array
   :param b_n: Legendre polynomial coefficients.  Shape ``(n_coeffs,)``.
   :type b_n: jax.Array
   :param i: Incidence direction vectors.  Shape ``(n_pixels, 3)``.
   :type i: jax.Array
   :param e: Emission direction vectors.  Shape ``(n_pixels, 3)``.
   :type e: jax.Array
   :param n: Surface normal vectors.  Shape ``(n_pixels, 3)``.
   :type n: jax.Array
   :param roughness: Macroscopic roughness angle in radians, :math:`\bar{\theta}`.
                     Default is 0.0.
   :type roughness: float, optional

   :returns: Reflectance per pixel.  Shape ``(n_pixels,)``.  Pixels where the
             source or detector is behind the local horizon are returned as NaN.
   :rtype: jax.Array

   .. rubric:: References

   .. bibliography::
      :filter: False

      Hapke-2012


