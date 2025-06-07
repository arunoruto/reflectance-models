refmod.hapke.functions
======================

.. py:module:: refmod.hapke.functions

.. autoapi-nested-parse::

   ??? info "References"

       1. Cornette and Shanks (1992). Bidirectional reflectance
       of flat, optically thick particulate systems. Applied Optics, 31(15),
       3152-3160. <https://doi.org/10.1364/AO.31.003152>







Module Contents
---------------

.. py:data:: PhaseFunctionType

.. py:function:: h_function_1(x, w)

   Calculates the H-function (level 1).

   :param x: Input parameter.
   :type x: npt.NDArray
   :param w: Single scattering albedo.
   :type w: npt.NDArray

   :returns: H-function values.
   :rtype: npt.NDArray

   .. rubric:: References

   Hapke (1993, p. 121, Eq. 8.31a).


.. py:function:: h_function_2(x, w)

   Calculates the H-function (level 2).

   :param x: Input parameter, often mu or mu_0 (cosine of angles).
   :type x: npt.NDArray
   :param w: Single scattering albedo.
   :type w: npt.NDArray

   :returns: H-function values.
   :rtype: npt.NDArray

   .. rubric:: References

   Cornette and Shanks (1992)


.. py:function:: h_function_2_derivative(x, w)

   Calculates the derivative of the H-function (level 2) with respect to w.

   :param x: Input parameter, often mu or mu_0 (cosine of angles).
   :type x: npt.NDArray
   :param w: Single scattering albedo.
   :type w: npt.NDArray

   :returns: Derivative of the H-function (level 2) with respect to w.
   :rtype: npt.NDArray


.. py:function:: h_function(x, w, level = 1)

   Calculates the Hapke H-function.

   This function can compute two different versions (levels) of the H-function.

   :param x: Input parameter, often mu or mu_0 (cosine of angles).
   :type x: npt.NDArray
   :param w: Single scattering albedo.
   :type w: npt.NDArray
   :param level: Level of the H-function to calculate (1 or 2), by default 1.
                 Level 1 refers to `h_function_1`.
                 Level 2 refers to `h_function_2`.
   :type level: int, optional

   :returns: Calculated H-function values.
   :rtype: npt.NDArray

   :raises Exception: If an invalid level (not 1 or 2) is provided.


.. py:function:: h_function_derivative(x, w, level = 1)

   Calculates the derivative of the Hapke H-function with respect to w.

   This function can compute the derivative for two different versions (levels)
   of the H-function.

   :param x: Input parameter, often mu or mu_0 (cosine of angles).
   :type x: npt.NDArray
   :param w: Single scattering albedo.
   :type w: npt.NDArray
   :param level: Level of the H-function derivative to calculate (1 or 2), by default 1.
                 Level 1 derivative is not implemented.
                 Level 2 refers to `h_function_2_derivative`.
   :type level: int, optional

   :returns: Calculated H-function derivative values.
   :rtype: npt.NDArray

   :raises NotImplementedError: If level 1 is selected, as its derivative is not implemented.
   :raises Exception: If an invalid level (not 1 or 2) is provided.


.. py:function:: double_henyey_greenstein(cos_g, b = 0.21, c = 0.7)

   Calculates the Double Henyey-Greenstein phase function.

   :param cos_g: Cosine of the scattering angle (g).
   :type cos_g: npt.NDArray
   :param b: Asymmetry parameter, by default 0.21.
   :type b: float, optional
   :param c: Backscatter fraction, by default 0.7.
   :type c: float, optional

   :returns: Phase function values.
   :rtype: npt.NDArray


.. py:function:: cornette_shanks(cos_g, xi)

   Calculates the Cornette-Shanks phase function.

   :param cos_g: Cosine of the scattering angle (g).
   :type cos_g: npt.NDArray
   :param xi: Asymmetry parameter, related to the average scattering angle.
              Note: This `xi` is different from the single scattering albedo `w`.
   :type xi: float

   :returns: Phase function values.
   :rtype: npt.NDArray

   .. rubric:: References

   Cornette and Shanks (1992, Eq. 8).


.. py:function:: phase_function(cos_g, type, args)

   Selects and evaluates a phase function.

   :param cos_g: Cosine of the scattering angle (g).
   :type cos_g: npt.NDArray
   :param type: Type of phase function to use.
                Valid options are:
                - "dhg" or "double_henyey_greenstein": Double Henyey-Greenstein
                - "cs" or "cornette" or "cornette_shanks": Cornette-Shanks
   :type type: PhaseFunctionType
   :param args: Arguments for the selected phase function.
                - For "dhg": (b, c) where b is asymmetry and c is backscatter fraction.
                - For "cs": (xi,) where xi is the Cornette-Shanks asymmetry parameter.
   :type args: tuple

   :returns: Calculated phase function values.
   :rtype: npt.NDArray

   :raises Exception: If an unsupported `type` is provided.


.. py:function:: normalize(x, axis = -1)

   Normalizes a vector or a batch of vectors.

   Calculates the L2 norm (Euclidean norm) of the input array along the
   specified axis.

   :param x: Input array representing a vector or a batch of vectors.
   :type x: npt.NDArray
   :param axis: Axis along which to compute the norm, by default -1.
   :type axis: int, optional

   :returns: The L2 norm of the input array. If `x` is a batch of vectors,
             the output will be an array of norms.
   :rtype: npt.NDArray


.. py:function:: normalize_keepdims(x, axis = -1)

   Normalizes a vector or batch of vectors, keeping dimensions.

   Calculates the L2 norm of the input array along the specified axis,
   then expands the dimensions of the output to match the input array's
   dimension along the normalization axis. This is useful for broadcasting
   the norm for division.

   :param x: Input array representing a vector or a batch of vectors.
   :type x: npt.NDArray
   :param axis: Axis along which to compute the norm, by default -1.
   :type axis: int, optional

   :returns: The L2 norm of the input array, with dimensions kept for broadcasting.
   :rtype: npt.NDArray


.. py:function:: angle_processing_base(vec_a, vec_b, axis = -1)

   Computes cosine and sine of the angle between two vectors.

   :param vec_a: First vector or batch of vectors.
   :type vec_a: npt.NDArray
   :param vec_b: Second vector or batch of vectors. Must have the same shape as vec_a.
   :type vec_b: npt.NDArray
   :param axis: Axis along which the dot product is performed, by default -1.
   :type axis: int, optional

   :returns:

             A tuple containing:
                 - cos_phi : npt.NDArray
                     Cosine of the angle(s) between vec_a and vec_b.
                 - sin_phi : npt.NDArray
                     Sine of the angle(s) between vec_a and vec_b.
   :rtype: tuple[npt.NDArray, npt.NDArray]


.. py:function:: angle_processing(vec_a, vec_b, axis = -1)

   Computes various trigonometric quantities related to the angle between two vectors.

   :param vec_a: First vector or batch of vectors.
   :type vec_a: npt.NDArray
   :param vec_b: Second vector or batch of vectors. Must have the same shape as vec_a.
   :type vec_b: npt.NDArray
   :param axis: Axis along which the dot product is performed, by default -1.
   :type axis: int, optional

   :returns:

             A tuple containing:
                 - cos_phi : npt.NDArray
                     Cosine of the angle(s) between vec_a and vec_b.
                 - sin_phi : npt.NDArray
                     Sine of the angle(s) between vec_a and vec_b.
                 - cot_phi : npt.NDArray
                     Cotangent of the angle(s) between vec_a and vec_b.
                     (Returns np.inf where sin_phi is 0).
                 - i : npt.NDArray
                     The angle(s) in radians between vec_a and vec_b (i.e., arccos(cos_phi)).
   :rtype: tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]


