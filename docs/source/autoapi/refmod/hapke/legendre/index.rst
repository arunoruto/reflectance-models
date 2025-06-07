refmod.hapke.legendre
=====================

.. py:module:: refmod.hapke.legendre

.. autoapi-nested-parse::

   Reference for Hapke's Legendre polynomial coefficients and functions.

   This module implements coefficients and functions related to Legendre polynomial
   expansions as described by Hapke. These are primarily used for modeling
   anisotropic scattering and phase functions.

   ??? info "References"

       Hapke (2002)





Module Contents
---------------

.. py:function:: coef_a(n = 15)

   Calculates coefficients 'a_n' for Legendre polynomial series.

   These coefficients are used in Hapke's photometric model.

   :param n: The number of coefficients to calculate (degree of Legendre polynomial),
             by default 15. The resulting array will have `n + 1` elements.
   :type n: int, optional

   :returns: Array of 'a_n' coefficients, shape (n + 1,).
   :rtype: npt.NDArray

   .. rubric:: References

   Hapke (2002, Eq. 27).


.. py:function:: coef_b(b = 0.21, c = 0.7, n = 15)

   Calculates coefficients 'b_n' for Legendre polynomial expansion.

   These coefficients are used in Hapke's photometric model, specifically
   for the phase function representation.

   :param b: Asymmetry parameter for the Henyey-Greenstein phase function component,
             by default 0.21.
   :type b: float, optional
   :param c: Parameter determining the mixture of Henyey-Greenstein functions or
             a single function if NaN, by default 0.7.
             If `c` is `np.nan`, a single Henyey-Greenstein function is assumed.
   :type c: float, optional
   :param n: The number of coefficients to calculate (degree of Legendre polynomial),
             by default 15. The resulting array will have `n + 1` elements.
   :type n: int, optional

   :returns: Array of 'b_n' coefficients, shape (n + 1,).
   :rtype: npt.NDArray

   .. rubric:: Notes

   The calculation method depends on whether `c` is NaN.
   The first element `b_n[0]` is set to 1 if `c` is not NaN, which differs
   from the direct formula application for that term.

   .. rubric:: References

   Hapke (2002, p. 530).


.. py:function:: function_p(x, b_n, a_n = np.empty(1) * np.nan)

   Calculates the P function from Hapke's model.

   This function relates to the integrated phase function and accounts for
   anisotropic scattering.

   :param x: Input array, typically cosine of angles (e.g., mu, mu0).
   :type x: npt.NDArray
   :param b_n: Array of 'b_n' coefficients.
   :type b_n: npt.NDArray
   :param a_n: Array of 'a_n' coefficients. If not provided or NaN, they are
               calculated using `coef_a(b_n.size)`, by default `np.empty(1) * np.nan`.
   :type a_n: npt.NDArray, optional

   :returns: Calculated P function values. The shape will match `x` after broadcasting.
   :rtype: npt.NDArray

   .. rubric:: References

   Hapke (2002, Eqs. 23, 24).


.. py:function:: value_p(b_n, a_n = np.empty(1) * np.nan)

   Calculates the scalar value P from Hapke's model.

   This value is used in the expression for single particle phase function.

   :param b_n: Array of 'b_n' coefficients.
   :type b_n: npt.NDArray
   :param a_n: Array of 'a_n' coefficients. If not provided or NaN, they are
               calculated using `coef_a(b_n.size)`, by default `np.empty(1) * np.nan`.
   :type a_n: npt.NDArray, optional

   :returns: The calculated scalar value P.
   :rtype: float

   .. rubric:: References

   Hapke (2002, Eq. 25).


