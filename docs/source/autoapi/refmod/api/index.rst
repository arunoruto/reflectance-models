refmod.api
==========

.. py:module:: refmod.api

.. autoapi-nested-parse::

   High-level, image-shaped interface to the reflectance models.

   The functions in :mod:`refmod.hapke` and friends operate on flat
   ``(n_pixels, 3)`` geometry and scalar-per-pixel albedo. This module wraps
   them for the common application case:

   - :class:`HapkeAmsaParams` and siblings collect model parameters and accept
     external configuration dictionaries via ``from_dict``.
   - :func:`reflectance_image` evaluates a model while preserving
     ``(height, width)`` image shape and offers explicit invalid-geometry
     handling.
   - :func:`reflectance_normal_jacobian` and
     :func:`reflectance_gradient_jacobian` give per-pixel derivatives with
     respect to the surface normal and to the surface gradients ``(p, q)``.
   - :func:`invert_albedo_multi` retrieves one albedo per pixel from one or
     more observations of the same scene.









Module Contents
---------------

.. py:data:: InvalidMode

.. py:data:: ModelName

.. py:class:: HapkeAmsaParams

   .. py:attribute:: b
      :type:  float


   .. py:attribute:: c
      :type:  float


   .. py:attribute:: hs
      :type:  float
      :value: 0.0



   .. py:attribute:: Bs0
      :type:  float
      :value: 0.0



   .. py:attribute:: tb
      :type:  float
      :value: 0.0



   .. py:attribute:: hc
      :type:  float | None
      :value: None



   .. py:attribute:: Bc0
      :type:  float | None
      :value: None



   .. py:attribute:: n_order
      :type:  int
      :value: 15



   .. py:method:: from_dict(config)
      :classmethod:



   .. py:attribute:: from_matlab_config


   .. py:property:: legendre_coefficients
      :type: jax.Array


      DHG Legendre coefficients using refmod's public ``c`` convention.

      Pass external configuration ``c`` values directly to this object.

      The MATLAB reference implementation negates ``c`` into an internal
      variable (``c_hapke = -c``) and simultaneously swaps the signs in the
      two Henyey-Greenstein denominators. Those two flips cancel, so the
      value carried in configuration files equals the ``c`` used here and
      needs no conversion. Verified against the reference sources
      (``hapke_amsa_fast.m``) at the source level.


   .. py:property:: coherent_backscatter
      :type: tuple[float, float]



.. py:class:: HapkeImsaParams

   .. py:attribute:: b
      :type:  float


   .. py:attribute:: c
      :type:  float


   .. py:attribute:: h
      :type:  float
      :value: 0.0



   .. py:attribute:: b0
      :type:  float
      :value: 0.0



   .. py:attribute:: tb
      :type:  float
      :value: 0.0



   .. py:attribute:: n_order
      :type:  int
      :value: 15



   .. py:method:: from_dict(config)
      :classmethod:



   .. py:attribute:: from_matlab_config


   .. py:property:: legendre_coefficients
      :type: jax.Array



.. py:class:: HapkeCornetteParams

   .. py:attribute:: xi
      :type:  float


   .. py:attribute:: hs
      :type:  float
      :value: 0.0



   .. py:attribute:: Bs0
      :type:  float
      :value: 0.0



   .. py:attribute:: tb
      :type:  float
      :value: 0.0



   .. py:attribute:: hc
      :type:  float | None
      :value: None



   .. py:attribute:: Bc0
      :type:  float | None
      :value: None



   .. py:method:: from_dict(config)
      :classmethod:



   .. py:attribute:: from_matlab_config


   .. py:property:: coherent_backscatter
      :type: tuple[float, float]



.. py:class:: LunarLambertParams

   .. py:method:: from_dict(config = None)
      :classmethod:



   .. py:attribute:: from_matlab_config


.. py:class:: MultiImageInversionResult

   .. py:attribute:: parameters
      :type:  numpy.typing.NDArray


   .. py:attribute:: residuals
      :type:  numpy.typing.NDArray


   .. py:attribute:: converged
      :type:  numpy.typing.NDArray[numpy.bool_]


   .. py:attribute:: iterations
      :type:  numpy.typing.NDArray[numpy.int_]


.. py:function:: amsa_dhg(w, s, v, n, b, c, hs = 0.0, Bs0 = 0.0, tb = 0.0, hc = None, Bc0 = None, invalid = 'zero')

.. py:function:: imsa_modified_h_dhg(w, s, v, n, b, c, h = 0.0, b0 = 0.0, tb = 0.0, invalid = 'zero')

.. py:function:: reflectance_image(model, w, s, v, n, params = None, invalid = 'nan')

   Evaluate a reflectance model and preserve image-shaped inputs.

   Geometry can be supplied as ``(..., 3)``, ``(3, ...)``, ``(pixels, 3)``,
   or a single 3-vector. Scalar albedo is broadcast to the geometry shape.


.. py:function:: reflectance_normal_jacobian(model, w, s, v, n, params = None, invalid = 'zero')

   Return per-pixel reflectance derivatives with respect to normal vectors.


.. py:function:: reflectance_gradient_jacobian(model, w, s, v, p, q, params = None, invalid = 'zero')

   Return derivatives with respect to surface gradients ``p`` and ``q``.

   Uses the surface-gradient parametrisation
   ``n = normalize([-p, -q, 1])``, i.e. ``p = -n_x/n_z``, ``q = -n_y/n_z``.


.. py:function:: _reflectance_flat_jax(model, w_flat, s_flat, v_flat, n_flat, params)

.. py:function:: _reflectance_one_jax(model, w_value, s_value, v_value, n_value, params)

.. py:function:: _prepare_params(model, params)

.. py:function:: invert_albedo_multi(reflectance, s, v, n, params, initial_w = 1.0 / 3.0, mask = None, model = 'amsa', sigma = 0.0, max_steps = 40, return_info = False)

   Estimate one albedo value per pixel from one or more images.

   ``sigma=0`` runs pixelwise. ``sigma>0`` applies local-area
   preconditioning by smoothing observations and geometry with mask-aware
   Gaussian filters before the pixelwise solve, which stabilises the
   estimate where single-pixel observations are poorly conditioned.


.. py:function:: _invert_lunar_lambert_multi(refl_flat, s_flat, v_flat, n_flat, active, out_shape, return_info)

.. py:function:: _optional_float(value)

.. py:function:: _normalize_model(model)

.. py:function:: _vectors_to_flat(v)

.. py:function:: _infer_output_shape(w, *geometry_shapes)

.. py:function:: _broadcast_vectors(v, n_pixels, name)

.. py:function:: _field_to_flat(field, out_shape, n_pixels, name)

.. py:function:: _valid_geometry(s, v, n)

.. py:function:: _normalize_np(v)

.. py:function:: _geometry_angles(s, v, n)

.. py:function:: _geometry_angles_jax(s, v, n)

.. py:function:: _gradient_normals(p, q)

.. py:function:: _normalize_jnp(v)

.. py:function:: _format_invalid(refl, valid, out_shape, invalid)

.. py:function:: _image_stack_to_flat(stack, name)

.. py:function:: _as_per_image_vectors(geometry, n_images)

   Return geometry as ``(n_images, 3)`` when it is constant per image.


.. py:function:: _geometry_stack_to_flat(stack, n_images, n_pixels, name)

.. py:function:: _active_mask_stack(reflectance, mask)

.. py:function:: _smooth_inversion_inputs(reflectance, s, v, n, active, sigma)

.. py:function:: _smooth_geometry_stack(geometry, active, image_shape, sigma, n_images)

.. py:function:: _smooth_geometry_image(geometry, valid, image_shape, sigma)

.. py:function:: _masked_gaussian(values, valid, sigma)

.. py:function:: _amsa_residual_x(x, refl_obs, s, v, n, active, b_n, tb, hs, Bs0, hc, Bc0)

.. py:function:: _imsa_modified_h_residual_x(x, refl_obs, s, v, n, active, b, c, tb, h, b0)

.. py:function:: _invert_multi_pixel(refl_obs, s, v, n, active, w0, b_n, tb, hs, Bs0, hc, Bc0, max_steps)

.. py:data:: _invert_multi_amsa_jit
   :value: None


.. py:data:: _invert_multi_amsa_shared_jit
   :value: None


.. py:function:: _invert_multi_imsa_modified_h_pixel(refl_obs, s, v, n, active, w0, b, c, tb, h, b0, max_steps)

.. py:data:: _invert_multi_imsa_modified_h_jit
   :value: None


.. py:data:: _invert_multi_imsa_modified_h_shared_jit
   :value: None


