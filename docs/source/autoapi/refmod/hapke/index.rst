refmod.hapke
============

.. py:module:: refmod.hapke


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/refmod/hapke/_core/index
   /autoapi/refmod/hapke/amsa/index
   /autoapi/refmod/hapke/imsa/index
   /autoapi/refmod/hapke/inverse/index
   /autoapi/refmod/hapke/mimsa/index






Package Contents
----------------

.. py:function:: coef_a(n = 15)

   Legendre expansion coefficients :math:`a_n` for the Hapke phase function.

   Computes the coefficients defined by Hapke (2002, Eq. 27):

   .. math::

       a_n = \begin{cases}
           0, & n = 0, 2, 4, \ldots \\
           -\frac{1}{2}, & n = 1 \\
           \frac{2 - n}{n + 1} a_{n-2}, & n = 3, 5, 7, \ldots
       \end{cases}

   :param n: Number of coefficients to compute (default 15). Returns *n+1* values
             indexed 0 through *n*.
   :type n: int, optional

   :returns: Array of :math:`a_n` coefficients of length *n+1*.
   :rtype: jax.Array

   .. rubric:: References

   :cite:p:`Hapke-2002`


.. py:function:: cs_legendre_coefficients(xi, n = 15)

   Legendre expansion coefficients for the Cornette–Shanks phase function.

   :param xi: Asymmetry parameter :math:`\xi`.
   :type xi: float
   :param n: Number of coefficients to compute (default 15). Returns *n+1* values.
   :type n: int, optional

   :returns: Array of Legendre coefficients :math:`b_n` of length *n+1*.
   :rtype: jax.Array

   .. rubric:: References

   :cite:p:`Cornette-1992`


.. py:function:: dhg_legendre_coefficients(b, c, n = 15)

   Legendre expansion coefficients for the Double Henyey–Greenstein phase function.

   Computes the coefficients :math:`b_n`:

   .. math::

       b_n = \begin{cases}
           (2n + 1) b^n, & n \text{ even} \\
           c (2n + 1) b^n, & n \text{ odd}
       \end{cases}

   :param b: Asymmetry parameter.
   :type b: float
   :param c: Backscatter fraction.
   :type c: float
   :param n: Number of coefficients to compute (default 15). Returns *n+1* values.
   :type n: int, optional

   :returns: Array of Legendre coefficients :math:`b_n` of length *n+1*.
   :rtype: jax.Array

   .. rubric:: References

   :cite:p:`Henyey-1941`


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


.. py:function:: mimsa(w, b_n, i, e, n, roughness = 0.0, a_n = None)

   Batched MIMSA reflectance.

   Vectorised wrapper around :func:`_refl_mimsa_scalar` that evaluates the
   modified isotropic multiple-scattering Hapke model for an arbitrary
   number of pixels sharing the same Legendre coefficients and roughness.

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


.. py:class:: Hapke(/, **data)



   !!! abstract "Usage Documentation"
       [Models](../concepts/models.md)

   A base class for creating Pydantic models.

   .. attribute:: __class_vars__

      The names of the class variables defined on the model.

   .. attribute:: __private_attributes__

      Metadata about the private attributes of the model.

   .. attribute:: __signature__

      The synthesized `__init__` [`Signature`][inspect.Signature] of the model.

   .. attribute:: __pydantic_complete__

      Whether model building is completed, or if there are still undefined fields.

   .. attribute:: __pydantic_core_schema__

      The core schema of the model.

   .. attribute:: __pydantic_custom_init__

      Whether the model has a custom `__init__` function.

   .. attribute:: __pydantic_decorators__

      Metadata containing the decorators defined on the model.
      This replaces `Model.__validators__` and `Model.__root_validators__` from Pydantic V1.

   .. attribute:: __pydantic_generic_metadata__

      Metadata for generic models; contains data used for a similar purpose to
      __args__, __origin__, __parameters__ in typing-module generics. May eventually be replaced by these.

   .. attribute:: __pydantic_parent_namespace__

      Parent namespace of the model, used for automatic rebuilding of models.

   .. attribute:: __pydantic_post_init__

      The name of the post-init method for the model, if defined.

   .. attribute:: __pydantic_root_model__

      Whether the model is a [`RootModel`][pydantic.root_model.RootModel].

   .. attribute:: __pydantic_serializer__

      The `pydantic-core` `SchemaSerializer` used to dump instances of the model.

   .. attribute:: __pydantic_validator__

      The `pydantic-core` `SchemaValidator` used to validate instances of the model.

   .. attribute:: __pydantic_fields__

      A dictionary of field names and their corresponding [`FieldInfo`][pydantic.fields.FieldInfo] objects.

   .. attribute:: __pydantic_computed_fields__

      A dictionary of computed field names and their corresponding [`ComputedFieldInfo`][pydantic.fields.ComputedFieldInfo] objects.

   .. attribute:: __pydantic_extra__

      A dictionary containing extra values, if [`extra`][pydantic.config.ConfigDict.extra]
      is set to `'allow'`.

   .. attribute:: __pydantic_fields_set__

      The names of fields explicitly set during instantiation.

   .. attribute:: __pydantic_private__

      Values of private attributes set on the model instance.

   Create a new model by parsing and validating input data from keyword arguments.

   Raises [`ValidationError`][pydantic_core.ValidationError] if the input data cannot be
   validated to form a valid model.

   `self` is explicitly positional-only to allow `self` as a field name.


   .. py:attribute:: single_scattering_albedo
      :type:  numpy.typing.NDArray | None
      :value: None



   .. py:attribute:: legendre_coefficients
      :type:  numpy.typing.NDArray
      :value: None



   .. py:attribute:: incidence_direction
      :type:  numpy.typing.NDArray
      :value: None



   .. py:attribute:: emission_direction
      :type:  numpy.typing.NDArray
      :value: None



   .. py:attribute:: surface_orientation
      :type:  numpy.typing.NDArray
      :value: None



   .. py:attribute:: roughness
      :type:  float
      :value: None



   .. py:attribute:: shadow_hiding_h
      :type:  float
      :value: None



   .. py:attribute:: shadow_hiding_b0
      :type:  float
      :value: None



   .. py:attribute:: coherent_backscattering_h
      :type:  float
      :value: None



   .. py:attribute:: coherent_backscattering_b0
      :type:  float
      :value: None



   .. py:attribute:: model
      :type:  Literal['amsa', 'imsa', 'mimsa']
      :value: None



   .. py:attribute:: model_config

      Configuration for the model, should be a dictionary conforming to [`ConfigDict`][pydantic.config.ConfigDict].


   .. py:method:: model_post_init(__context)

      Override this method to perform additional initialization after `__init__` and `model_construct`.
      This is useful if you want to do some validation that requires the entire model to be initialized.



   .. py:method:: _broadcast_to_shape(a, target_shape)
      :staticmethod:



   .. py:method:: refl()


   .. py:method:: albedo(reflectance, x0 = None)


