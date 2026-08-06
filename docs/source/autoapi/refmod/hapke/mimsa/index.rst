refmod.hapke.mimsa
==================

.. py:module:: refmod.hapke.mimsa






Module Contents
---------------

.. py:function:: _refl_mimsa_scalar(w, b_n, i, e, n, roughness, a_n = None)

   MIMSA reflectance for a single pixel.

   Modified IMSA model that replaces the isotropic multiple-scattering
   approximation with the full Legendre *P*-term formulation.  No opposition
   effects are included.

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
   :param a_n: Legendre expansion coefficients for the multiple-scattering term.
               When ``None`` (default), they are computed from the order of *b_n*
               using :func:`~._core.coef_a`.
   :type a_n: jax.Array or None, optional

   :returns: Reflectance (scalar).  Pixels with the source or detector behind the
             local horizon are set to NaN.
   :rtype: jax.Array

   .. rubric:: References

   .. bibliography::
      :filter: False

      Hapke-2012
      Hapke-2002


.. py:data:: _mimsa_batched

.. py:function:: mimsa(w, b_n, i, e, n, roughness = 0.0, a_n = None)

   Batched MIMSA reflectance.

   Vectorised wrapper around :func:`_refl_mimsa_scalar` that evaluates the
   modified isotropic multiple-scattering Hapke model for an arbitrary
   number of pixels sharing the same Legendre coefficients and roughness.

   .. note::
       MIMSA is mathematically identical to :func:`refmod.hapke.amsa` with
       all opposition-effect parameters left at zero (asserted by
       ``test_mimsa_equals_amsa_no_opposition``). Prefer :func:`amsa` in
       new code; this wrapper is kept for API stability.

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
   :param a_n: Legendre expansion coefficients for the multiple-scattering term.
               When ``None`` (default), they are computed from the order of *b_n*
               using :func:`~._core.coef_a`.
   :type a_n: jax.Array or None, optional

   :returns: Reflectance per pixel.  Shape ``(n_pixels,)``.  Pixels where the
             source or detector is behind the local horizon are returned as NaN.
   :rtype: jax.Array

   .. rubric:: References

   .. bibliography::
      :filter: False

      Hapke-2012
      Hapke-2002


