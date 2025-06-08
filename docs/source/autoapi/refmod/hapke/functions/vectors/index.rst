refmod.hapke.functions.vectors
==============================

.. py:module:: refmod.hapke.functions.vectors




Module Contents
---------------

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


