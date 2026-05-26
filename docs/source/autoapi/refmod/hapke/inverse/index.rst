refmod.hapke.inverse
====================

.. py:module:: refmod.hapke.inverse








Module Contents
---------------

.. py:data:: _BYTES_PER_PIXEL
   :value: 500


.. py:class:: AmsaInversionState

   Reusable geometry-dependent state for AMSA albedo inversion.

   Instances are created with :func:`prepare_amsa_inversion` and consumed by
   :func:`invert_amsa_precomputed`. The state stores one or more precomputed
   chunks so repeated inversions with fixed geometry can skip the expensive
   geometry-only setup.


   .. py:attribute:: chunks
      :type:  tuple[tuple[int, int, dict], Ellipsis]


   .. py:attribute:: n_pixels
      :type:  int


   .. py:attribute:: chunk_size
      :type:  int


.. py:function:: _adaptive_chunk_size(n_pixels)

   Determine a safe per-chunk pixel count based on available memory.

   When a GPU is available, 90 % of the VRAM capacity is used to estimate
   the maximum chunk size.  On CPU-only systems, 75 % of the free RAM
   reported by ``psutil`` is used instead.  The returned chunk size is
   capped at *n_pixels*.

   :param n_pixels: Total number of pixels to be inverted.
   :type n_pixels: int

   :returns: Maximum number of pixels that can be inverted in a single chunk
             without exhausting memory.
   :rtype: int


.. py:function:: _tanh_to_w(x)

.. py:function:: _w_to_tanh(w)

.. py:function:: _d_w_d_x(x)

.. py:function:: _invert_chunk(refl_obs, pre, w0, max_steps = 40)

   Levenberg-Marquardt inversion for a single pixel chunk.

   Iteratively minimises the squared residual between the observed
   reflectance and the AMSA forward model.  The single scattering albedo
   *w* is transformed via :math:`x \mapsto \tanh` to keep it in
   :math:`(0, 1)` during the optimisation.

   Uses precomputed *w*-independent state (radiance-transfer quantities,
   shadowing corrections, etc.) to accelerate each iteration.  Only pixels
   flagged as active (finite observed reflectance) are updated; converged
   pixels are automatically masked out.

   :param refl_obs: Observed reflectance.  Shape ``(n_pixels,)``.
   :type refl_obs: jax.Array
   :param pre: Precomputed geometry-dependent terms from
               :func:`~.amsa.precompute_amsa`.
   :type pre: dict
   :param w0: Initial guess for the single scattering albedo.  Shape
              ``(n_pixels,)``.
   :type w0: jax.Array
   :param max_steps: Maximum number of Levenberg-Marquardt iterations.  Default is 40.
   :type max_steps: int, optional

   :returns: Recovered single scattering albedo.  Shape ``(n_pixels,)``.
   :rtype: jax.Array


.. py:data:: _invert_chunk_jit
   :value: None


.. py:function:: prepare_amsa_inversion(b_n, i, e, n, roughness = 0.0, h_sh = 0.0, b0_sh = 0.0, h_cb = 0.0, b0_cb = 0.0, chunk_size = None)

   Prepare reusable geometry-dependent state for AMSA inversion.

   Use this when the geometry and Hapke phase/opposition parameters stay
   fixed while multiple observed reflectance arrays are inverted.

   :param b_n: Legendre polynomial coefficients for the single-scattering phase
               function. Shape ``(n_coeffs,)``.
   :type b_n: jax.Array
   :param i: Incidence direction vectors. Shape ``(n_pixels, 3)``.
   :type i: jax.Array
   :param e: Emission direction vectors. Shape ``(n_pixels, 3)``.
   :type e: jax.Array
   :param n: Surface normal vectors. Shape ``(n_pixels, 3)``.
   :type n: jax.Array
   :param roughness: AMSA geometry/opposition parameters matching :func:`invert_amsa`.
   :type roughness: float, optional
   :param h_sh: AMSA geometry/opposition parameters matching :func:`invert_amsa`.
   :type h_sh: float, optional
   :param b0_sh: AMSA geometry/opposition parameters matching :func:`invert_amsa`.
   :type b0_sh: float, optional
   :param h_cb: AMSA geometry/opposition parameters matching :func:`invert_amsa`.
   :type h_cb: float, optional
   :param b0_cb: AMSA geometry/opposition parameters matching :func:`invert_amsa`.
   :type b0_cb: float, optional
   :param chunk_size: Number of pixels to precompute per chunk. When ``None``, determined
                      automatically from available memory.
   :type chunk_size: int or None, optional

   :returns: Reusable precomputed state for :func:`invert_amsa_precomputed`.
   :rtype: AmsaInversionState


.. py:function:: invert_amsa_precomputed(refl_obs, state, w0 = None, max_steps = 40)

   Invert AMSA reflectance using precomputed geometry state.

   This is the high-throughput path for repeated inversions with fixed
   geometry. For one-off calls, use :func:`invert_amsa`.

   :param refl_obs: Observed reflectance. Shape ``(n_pixels,)``.
   :type refl_obs: jax.Array
   :param state: Prepared state from :func:`prepare_amsa_inversion`.
   :type state: AmsaInversionState
   :param w0: Initial guess for the single scattering albedo. Shape
              ``(n_pixels,)``. When ``None``, defaults to 1/3 everywhere.
   :type w0: jax.Array or None, optional
   :param max_steps: Maximum number of Levenberg-Marquardt iterations per chunk.
   :type max_steps: int, optional

   :returns: Recovered single scattering albedo. Shape ``(n_pixels,)``.
   :rtype: jax.Array


.. py:function:: invert_amsa(refl_obs, b_n, i, e, n, roughness = 0.0, h_sh = 0.0, b0_sh = 0.0, h_cb = 0.0, b0_cb = 0.0, w0 = None, max_steps = 40, chunk_size = None)

   Invert the AMSA model to recover single scattering albedo from
   observed reflectance.

   Solves the inverse problem for the Hapke AMSA forward model using a
   chunked, batched Levenberg-Marquardt algorithm.  The single scattering
   albedo :math:`w` is constrained to :math:`(0, 1)` via a
   :math:`\tanh`-based variable transformation.  Convergence is tracked
   per pixel through an active-set mask, and large datasets are
   automatically split into memory-safe chunks.

   :param refl_obs: Observed reflectance.  Shape ``(n_pixels,)``.
   :type refl_obs: jax.Array
   :param b_n: Legendre polynomial coefficients for the single-scattering phase
               function.  Shape ``(n_coeffs,)``.
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
   :param h_sh: SHOE angular width parameter.  Default is 0.0.
   :type h_sh: float, optional
   :param b0_sh: SHOE amplitude.  Default is 0.0.
   :type b0_sh: float, optional
   :param h_cb: CBOE angular width parameter.  Default is 0.0.
   :type h_cb: float, optional
   :param b0_cb: CBOE amplitude.  Default is 0.0.
   :type b0_cb: float, optional
   :param w0: Initial guess for the single scattering albedo.  Shape
              ``(n_pixels,)``.  When ``None``, defaults to 1/3 everywhere.
   :type w0: jax.Array or None, optional
   :param max_steps: Maximum number of Levenberg-Marquardt iterations per chunk.
                     Default is 40.
   :type max_steps: int, optional
   :param chunk_size: Number of pixels to process per chunk.  When ``None``, determined
                      automatically by :func:`_adaptive_chunk_size`.
   :type chunk_size: int or None, optional

   :returns: Recovered single scattering albedo.  Shape ``(n_pixels,)``.
   :rtype: jax.Array


