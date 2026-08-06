refmod
======

.. py:module:: refmod


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/refmod/api/index
   /autoapi/refmod/config/index
   /autoapi/refmod/dtm_helper/index
   /autoapi/refmod/hapke/index
   /autoapi/refmod/jax/index
   /autoapi/refmod/lambert/index
   /autoapi/refmod/lunar_lambert/index
   /autoapi/refmod/mixing/index
   /autoapi/refmod/shkuratov/index
   /autoapi/refmod/utils/index
   /autoapi/refmod/warmup/index






Package Contents
----------------

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


.. py:function:: lambert(w, i, e, n)

   Batched Lambert reflectance model.

   The model is view-independent — the emission direction *e* is accepted
   for API consistency but ignored.

   :param w: Single scattering albedo per pixel.  Shape ``(n_pixels,)``.
   :type w: jax.Array
   :param i: Incidence direction vectors.  Shape ``(n_pixels, 3)``.
   :type i: jax.Array
   :param e: Emission direction vectors (unused).  Shape ``(n_pixels, 3)``.
   :type e: jax.Array
   :param n: Surface normal vectors.  Shape ``(n_pixels, 3)``.
   :type n: jax.Array

   :returns: Reflectance per pixel.  Shape ``(n_pixels,)``.
   :rtype: jax.Array


.. py:function:: lunar_lambert(rho, th_i, th_e, alpha)

   MATLAB-compatible lunar-Lambert reflectance model.

   This ports the reference ``lunar_lambert_model.m``. It is a lookup-table
   model, tabulated for :math:`\bar{\theta} = 10^\circ` and :math:`w = 0.1`,
   and is intentionally separate from physical Lambertian reflectance
   (:func:`~refmod.lambert.lambert`).


.. py:function:: shkuratov(a_n, mu1 = 0.0, eta = 0.0, i = None, e = None, n = None, m0 = 0.0, mu2 = 0.0)

   Shkuratov photometric model with the Akimov disk function.

   Reflectance is computed as

   .. math::

      r = A_n \, \frac{\phi(\alpha) \, D(\alpha, \beta, \gamma)}{\cos i}

   where :math:`\phi(\alpha)` is the phase function and
   :math:`D(\alpha, \beta, \gamma)` is the Akimov disk function.

   :param a_n: Normal albedo per pixel.  Shape ``(n_pixels,)``.
   :type a_n: jax.Array
   :param mu1: Roughness parameter (exponential phase term).  Default is 0.0.
   :type mu1: float, optional
   :param eta: Fractal deviation parameter controlling the limb-darkening exponent.
               Default is 0.0.
   :type eta: float, optional
   :param i: Incidence direction vectors.  Shape ``(n_pixels, 3)``.
   :type i: jax.Array or None, optional
   :param e: Emission direction vectors.  Shape ``(n_pixels, 3)``.
   :type e: jax.Array or None, optional
   :param n: Surface normal vectors.  Shape ``(n_pixels, 3)``.
   :type n: jax.Array or None, optional
   :param m0: Opposition surge amplitude.  Default is 0.0 (no opposition effect).
   :type m0: float, optional
   :param mu2: Opposition surge width.  Default is 0.0.
   :type mu2: float, optional

   :returns: Reflectance per pixel.  Shape ``(n_pixels,)``.
   :rtype: jax.Array

   .. rubric:: References

   :cite:p:`Shkuratov-2011`


