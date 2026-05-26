refmod.hapke.amsa
=================

.. py:module:: refmod.hapke.amsa






Module Contents
---------------

.. py:function:: _refl_amsa_scalar(w, b_n, i, e, n, roughness, h_sh, b0_sh, h_cb, b0_cb, a_n = None)

   Compute AMSA reflectance for a single pixel (scalar w, 3-vectors).

   Full anisotropic multiple scattering with shadow hiding and coherent
   backscatter. Handles a single pixel with scalar single-scattering albedo
   and 3-vector incidence, emission, and normal directions.

   :param w: Single-scattering albedo (scalar).
   :type w: Array
   :param b_n: Legendre coefficients of the single-particle phase function, shape (N,).
   :type b_n: Array
   :param i: Incidence direction vector, shape (3,).
   :type i: Array
   :param e: Emission direction vector, shape (3,).
   :type e: Array
   :param n: Surface normal vector, shape (3,).
   :type n: Array
   :param roughness: Surface roughness angle in radians.
   :type roughness: float
   :param h_sh: Shadow-hiding angular width parameter.
   :type h_sh: float
   :param b0_sh: Shadow-hiding opposition amplitude.
   :type b0_sh: float
   :param h_cb: Coherent backscatter angular width parameter.
   :type h_cb: float
   :param b0_cb: Coherent backscatter opposition amplitude.
   :type b0_cb: float
   :param a_n: Precomputed Legendre expansion of the Henyey-Greenstein phase function
               coefficients. Computed from ``b_n`` if None.
   :type a_n: Array or None, optional

   :returns: Reflectance value (scalar). NaN if mu0 <= 0 or mu <= 0.
   :rtype: Array

   .. rubric:: References

   :cite:p:`Hapke-1984`
   :cite:p:`Hapke-2002`
   :cite:p:`Hapke-2012`


.. py:function:: _refl_amsa_scalar_and_grad(w, b_n, i, e, n, roughness, h_sh, b0_sh, h_cb, b0_cb, a_n = None)

   Compute AMSA reflectance and its analytical derivative dR/dw for a single pixel.

   Full anisotropic multiple scattering with shadow hiding and coherent
   backscatter. Returns both reflectance and derivative with respect to the
   single-scattering albedo w, useful for Levenberg-Marquardt optimization.

   :param w: Single-scattering albedo (scalar).
   :type w: Array
   :param b_n: Legendre coefficients of the single-particle phase function, shape (N,).
   :type b_n: Array
   :param i: Incidence direction vector, shape (3,).
   :type i: Array
   :param e: Emission direction vector, shape (3,).
   :type e: Array
   :param n: Surface normal vector, shape (3,).
   :type n: Array
   :param roughness: Surface roughness angle in radians.
   :type roughness: float
   :param h_sh: Shadow-hiding angular width parameter.
   :type h_sh: float
   :param b0_sh: Shadow-hiding opposition amplitude.
   :type b0_sh: float
   :param h_cb: Coherent backscatter angular width parameter.
   :type h_cb: float
   :param b0_cb: Coherent backscatter opposition amplitude.
   :type b0_cb: float
   :param a_n: Precomputed Legendre expansion of the Henyey-Greenstein phase function
               coefficients. Computed from ``b_n`` if None.
   :type a_n: Array or None, optional

   :returns: Reflectance value (scalar) and derivative dR/dw (scalar).
             Reflectance is NaN and derivative is 0.0 if mu0 <= 0 or mu <= 0.
   :rtype: tuple[Array, Array]

   .. rubric:: References

   :cite:p:`Hapke-1984`
   :cite:p:`Hapke-2002`
   :cite:p:`Hapke-2012`


.. py:function:: _precompute_amsa_scalar(b_n, i, e, n, roughness, h_sh, b0_sh, h_cb, b0_cb, a_n = None)

   Precompute all w-independent quantities for fast LM iteration.

   Computes roughness correction, Legendre expansion, single-particle phase
   function, shadow hiding, coherent backscatter, and the albedo-independent
   prefactor. The returned dict can be reused with
   ``_fast_refl_amsa_scalar`` and ``_fast_refl_amsa_scalar_and_grad``
   to avoid recomputing these quantities when only w varies.

   :param b_n: Legendre coefficients of the single-particle phase function, shape (N,).
   :type b_n: Array
   :param i: Incidence direction vector, shape (3,).
   :type i: Array
   :param e: Emission direction vector, shape (3,).
   :type e: Array
   :param n: Surface normal vector, shape (3,).
   :type n: Array
   :param roughness: Surface roughness angle in radians.
   :type roughness: float
   :param h_sh: Shadow-hiding angular width parameter.
   :type h_sh: float
   :param b0_sh: Shadow-hiding opposition amplitude.
   :type b0_sh: float
   :param h_cb: Coherent backscatter angular width parameter.
   :type h_cb: float
   :param b0_cb: Coherent backscatter opposition amplitude.
   :type b0_cb: float
   :param a_n: Precomputed Legendre expansion of the Henyey-Greenstein phase function
               coefficients. Computed from ``b_n`` if None.
   :type a_n: Array or None, optional

   :returns: Dictionary of precomputed values with keys ``albedo_indep``, ``p_sh``,
             ``p_mu0``, ``p_mu``, ``P``, ``mu0``, ``mu``, ``valid``.
   :rtype: dict

   .. rubric:: References

   :cite:p:`Hapke-1984`
   :cite:p:`Hapke-2002`
   :cite:p:`Hapke-2012`


.. py:function:: _fast_refl_amsa_scalar(w, pre)

   Compute AMSA reflectance using precomputed state.

   Only recomputes the H-functions and the multiple scattering term M.
   Uses precomputed w-independent quantities from
   ``_precompute_amsa_scalar`` for faster evaluation during
   Levenberg-Marquardt iterations.

   :param w: Single-scattering albedo (scalar).
   :type w: Array
   :param pre: Precomputed state dict from ``_precompute_amsa_scalar``.
   :type pre: dict

   :returns: Reflectance value (scalar). NaN if not valid.
   :rtype: Array

   .. rubric:: References

   :cite:p:`Hapke-2012`


.. py:function:: _fast_refl_amsa_scalar_and_grad(w, pre)

   Compute AMSA reflectance and analytical derivative dR/dw using precomputed state.

   Only recomputes the H-functions, their derivatives, and the multiple
   scattering term. Uses precomputed w-independent quantities from
   ``_precompute_amsa_scalar`` for faster evaluation during
   Levenberg-Marquardt iterations.

   :param w: Single-scattering albedo (scalar).
   :type w: Array
   :param pre: Precomputed state dict from ``_precompute_amsa_scalar``.
   :type pre: dict

   :returns: Reflectance value (scalar) and derivative dR/dw (scalar).
             Reflectance is NaN and derivative is 0.0 if not valid.
   :rtype: tuple[Array, Array]

   .. rubric:: References

   :cite:p:`Hapke-2012`


.. py:data:: _precompute_amsa_batched

.. py:data:: _fast_refl_amsa_batched
   :value: None


.. py:data:: _fast_refl_amsa_and_grad_batched
   :value: None


.. py:function:: precompute_amsa(b_n, i, e, n, roughness = 0.0, h_sh = 0.0, b0_sh = 0.0, h_cb = 0.0, b0_cb = 0.0, a_n = None)

   Precompute w-independent quantities for batched (N, 3) inputs.

   ``jax.vmap``-wrapped version of ``_precompute_amsa_scalar`` across the
   pixel dimension. ``b_n`` is broadcast across all pixels.

   :param b_n: Legendre coefficients of the single-particle phase function, shape (N,).
   :type b_n: Array
   :param i: Incidence direction vectors, shape (M, 3).
   :type i: Array
   :param e: Emission direction vectors, shape (M, 3).
   :type e: Array
   :param n: Surface normal vectors, shape (M, 3).
   :type n: Array
   :param roughness: Surface roughness angle in radians. Default is 0.0.
   :type roughness: float, optional
   :param h_sh: Shadow-hiding angular width parameter. Default is 0.0.
   :type h_sh: float, optional
   :param b0_sh: Shadow-hiding opposition amplitude. Default is 0.0.
   :type b0_sh: float, optional
   :param h_cb: Coherent backscatter angular width parameter. Default is 0.0.
   :type h_cb: float, optional
   :param b0_cb: Coherent backscatter opposition amplitude. Default is 0.0.
   :type b0_cb: float, optional

   :returns: Dictionary of batched precomputed values with keys ``albedo_indep``,
             ``p_sh``, ``p_mu0``, ``p_mu``, ``P``, ``mu0``, ``mu``, ``valid``.
             Each value has an added leading batch dimension.
   :rtype: dict

   .. rubric:: References

   :cite:p:`Hapke-1984`
   :cite:p:`Hapke-2002`
   :cite:p:`Hapke-2012`


.. py:data:: _amsa_batched

.. py:function:: amsa(w, b_n, i, e, n, roughness = 0.0, h_sh = 0.0, b0_sh = 0.0, h_cb = 0.0, b0_cb = 0.0, a_n = None)

   Public batched AMSA reflectance model.

   Full anisotropic multiple scattering with shadow hiding and coherent
   backscatter. Computes reflectance for a batch of pixels with scalar w,
   broadcast Legendre coefficients ``b_n``, and batched geometry vectors.

   :param w: Single-scattering albedo (scalar, broadcast across pixels).
   :type w: Array
   :param b_n: Legendre coefficients of the single-particle phase function, shape (N,).
   :type b_n: Array
   :param i: Incidence direction vectors, shape (M, 3).
   :type i: Array
   :param e: Emission direction vectors, shape (M, 3).
   :type e: Array
   :param n: Surface normal vectors, shape (M, 3).
   :type n: Array
   :param roughness: Surface roughness angle in radians. Default is 0.0.
   :type roughness: float, optional
   :param h_sh: Shadow-hiding angular width parameter. Default is 0.0.
   :type h_sh: float, optional
   :param b0_sh: Shadow-hiding opposition amplitude. Default is 0.0.
   :type b0_sh: float, optional
   :param h_cb: Coherent backscatter angular width parameter. Default is 0.0.
   :type h_cb: float, optional
   :param b0_cb: Coherent backscatter opposition amplitude. Default is 0.0.
   :type b0_cb: float, optional
   :param a_n: Precomputed Legendre expansion of the Henyey-Greenstein phase function
               coefficients. Computed from ``b_n`` if None.
   :type a_n: Array or None, optional

   :returns: Reflectance values, shape (M,). NaN where mu0 <= 0 or mu <= 0.
   :rtype: Array

   .. rubric:: References

   :cite:p:`Hapke-1984`
   :cite:p:`Hapke-2002`
   :cite:p:`Hapke-2012`


