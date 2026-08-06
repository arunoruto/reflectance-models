refmod.jax
==========

.. py:module:: refmod.jax


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/refmod/jax/hapke/index




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

   .. rubric:: Notes

   When *b* and *c* are plain scalars, the truncation error of the series
   is checked against ``DHG_TRUNCATION_WARN_THRESHOLD`` and a warning with
   a recommended order is emitted if the reconstruction of the phase
   function would be too inaccurate (relevant for strongly peaked phase
   functions, roughly :math:`|b| \gtrsim 0.4` at the default order).

   .. rubric:: References

   :cite:p:`Henyey-1941`


.. py:function:: jax_amsa(w, i, e, n, b_n, a_n = None, roughness = 0.0, h_sh = 0.0, b0_sh = 0.0, h_cb = 0.0, b0_cb = 0.0)

   Legacy image-shaped wrapper for the pre-1.0 ``refmod.jax.hapke`` API.


