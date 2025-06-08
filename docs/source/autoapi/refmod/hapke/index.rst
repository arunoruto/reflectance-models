refmod.hapke
============

.. py:module:: refmod.hapke


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/refmod/hapke/amsa/index
   /autoapi/refmod/hapke/functions/index
   /autoapi/refmod/hapke/imsa/index
   /autoapi/refmod/hapke/mimsa/index






Package Contents
----------------

.. py:function:: amsa(single_scattering_albedo, incidence_direction, emission_direction, surface_orientation, phase_function_type, b_n = None, a_n = None, hs = 0, bs0 = 0, roughness = 0, hc = 0, bc0 = 0, phase_function_args = (), refl_optimization = None)

   Calculates the reflectance using the AMSA model.

   :param single_scattering_albedo: Single scattering albedo.
   :type single_scattering_albedo: npt.NDArray
   :param incidence_direction: Incidence direction vector(s) of shape (..., 3).
   :type incidence_direction: npt.NDArray
   :param emission_direction: Emission direction vector(s) of shape (..., 3).
   :type emission_direction: npt.NDArray
   :param surface_orientation: Surface orientation vector(s) of shape (..., 3).
   :type surface_orientation: npt.NDArray
   :param phase_function_type: Type of phase function to use.
   :type phase_function_type: PhaseFunctionType
   :param b_n: Coefficients of the Legendre expansion.
   :type b_n: npt.NDArray
   :param a_n: Coefficients of the Legendre expansion.
   :type a_n: npt.NDArray
   :param hs: Shadowing parameter, by default 0.
   :type hs: float, optional
   :param bs0: Shadowing parameter, by default 0.
   :type bs0: float, optional
   :param roughness: Surface roughness, by default 0.
   :type roughness: float, optional
   :param hc: Coherent backscattering parameter, by default 0.
   :type hc: float, optional
   :param bc0: Coherent backscattering parameter, by default 0.
   :type bc0: float, optional
   :param phase_function_args: Additional arguments for the phase function, by default ().
   :type phase_function_args: tuple, optional
   :param refl_optimization: Reflectance optimization array, by default None.
   :type refl_optimization: npt.NDArray | None, optional

   :returns: Reflectance values.
   :rtype: npt.NDArray

   :raises Exception: If at least one reflectance value is not real.
   :raises References:
   :raises ----------:
   :raises [AMSAModelPlaceholder]:


.. py:function:: imsa(single_scattering_albedo, incidence_direction, emission_direction, surface_orientation, phase_function, opposition_effect_h = 0, oppoistion_effect_b0 = 0, roughness = 0)

   Calculates reflectance using the IMSA model.

   IMSA stands for Inversion of Multiple Scattering and Absorption.

   :param single_scattering_albedo: Single scattering albedo, shape (...).
   :type single_scattering_albedo: npt.NDArray
   :param incidence_direction: Incidence direction vector(s), shape (..., 3).
   :type incidence_direction: npt.NDArray
   :param emission_direction: Emission direction vector(s), shape (..., 3).
   :type emission_direction: npt.NDArray
   :param surface_orientation: Surface normal vector(s), shape (..., 3).
   :type surface_orientation: npt.NDArray
   :param phase_function: Callable that accepts `cos_alpha` (cosine of phase angle) and
                          returns phase function values.
   :type phase_function: Callable[[npt.NDArray], npt.NDArray]
   :param opposition_effect_h: Opposition effect parameter h, by default 0.
   :type opposition_effect_h: float, optional
   :param oppoistion_effect_b0: Opposition effect parameter B0 (b_zero), by default 0.
                                Note: Original argument name `oppoistion_effect_b0` kept for API compatibility.
   :type oppoistion_effect_b0: float, optional
   :param roughness: Surface roughness parameter, by default 0.
   :type roughness: float, optional

   :returns: Calculated reflectance values, shape (...).
   :rtype: npt.NDArray

   :raises Exception: If any calculated reflectance value has a significant imaginary part.

   .. rubric:: Notes

   - Input arrays `incidence_direction`, `emission_direction`,
     `surface_orientation`, and `single_scattering_albedo` are expected to
     broadcast together.
   - The `phase_function` should be vectorized to handle arrays of `cos_alpha`.
   - The IMSA model accounts for multiple scattering and absorption.

   References

   ----------

   [IMSAModelPlaceholder]


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
      :type:  numpy.typing.NDArray


   .. py:attribute:: incidence_direction
      :type:  numpy.typing.NDArray


   .. py:attribute:: emission_direction
      :type:  numpy.typing.NDArray


   .. py:attribute:: surface_orientation
      :type:  numpy.typing.NDArray


   .. py:attribute:: phase_function
      :type:  Callable[[numpy.typing.NDArray], numpy.typing.NDArray]


   .. py:attribute:: opposition_effect_h
      :type:  float
      :value: None



   .. py:attribute:: oppoistion_effect_b0
      :type:  float
      :value: None



   .. py:attribute:: roughness
      :type:  float
      :value: None



   .. py:attribute:: model_config

      Configuration for the model, should be a dictionary conforming to [`ConfigDict`][pydantic.config.ConfigDict].


