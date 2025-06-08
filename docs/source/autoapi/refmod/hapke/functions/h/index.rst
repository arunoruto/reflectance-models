refmod.hapke.functions.h
========================

.. py:module:: refmod.hapke.functions.h




Module Contents
---------------

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


