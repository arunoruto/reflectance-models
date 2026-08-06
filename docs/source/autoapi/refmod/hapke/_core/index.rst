refmod.hapke._core
==================

.. py:module:: refmod.hapke._core






Module Contents
---------------

.. py:data:: DHG_TRUNCATION_WARN_THRESHOLD
   :value: 0.0001


   Truncation-error bound above which :func:`dhg_legendre_coefficients` warns.

.. py:function:: normalize(v)

   Normalize a vector to unit length using the L2 norm.

   :param v: Input vector (or batch of vectors).
   :type v: jax.Array

   :returns: Normalized vector(s) with unit L2 norm.
   :rtype: jax.Array


.. py:function:: cos_angle(a, b)

   Compute the cosine of the angle between two vectors.

   The result is clamped to :math:`[-1, 1]` for numerical stability.

   :param a: First vector.
   :type a: jax.Array
   :param b: Second vector.
   :type b: jax.Array

   :returns: Dot product of *a* and *b*, clamped to :math:`[-1, 1]`.
   :rtype: jax.Array


.. py:function:: h_function(x, w)

   Hapke isotropic multiple-scattering H-function.

   Computes the Ambartsumian–Chandrasekhar H-function using Hapke's
   level-2 approximation (Hapke 2002, Eq. 13):

   .. math::

       H(x, w) = \frac{1}{1 - w x \left[r_0 + \frac{1 - 2 r_0 x}{2}
       \ln\frac{1 + x}{x}\right]}

   where :math:`\gamma = \sqrt{1 - w}` and
   :math:`r_0 = (1 - \gamma) / (1 + \gamma)`.

   :param x: Direction cosine :math:`\mu` or :math:`\mu_0`.
   :type x: jax.Array
   :param w: Single-scattering albedo.
   :type w: jax.Array

   :returns: Value of the H-function :math:`H(x, w)`.
   :rtype: jax.Array

   .. rubric:: References

   :cite:p:`Hapke-2002`


.. py:function:: h_function_derivative(x, w)

   Derivative of the H-function with respect to single-scattering albedo.

   Computes :math:`\partial H(x, w) / \partial w` of Hapke's level-2
   approximation (see :func:`h_function`).

   Note: kept alongside JAX autodiff on purpose — it matches the MATLAB
   ``hapke_amsa.m`` derivative exactly and is marginally faster than
   ``jax.jvp`` of the forward model.

   :param x: Direction cosine :math:`\mu` or :math:`\mu_0`.
   :type x: jax.Array
   :param w: Single-scattering albedo.
   :type w: jax.Array

   :returns: Derivative :math:`\partial H / \partial w`.
   :rtype: jax.Array

   .. rubric:: References

   :cite:p:`Hapke-2002`


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


.. py:function:: dhg_truncation_error(b, c, n)

   Upper bound on the truncation error of the DHG Legendre expansion.

   The DHG phase function has the exact expansion
   :math:`p(x) = \sum_k b_k P_k(x)` with
   :math:`|b_k| \leq \max(1, |c|)\,(2k+1)\,|b|^k` and :math:`|P_k(x)| \leq 1`,
   so the absolute error of truncating after order *n* is bounded by the
   tail sum

   .. math::

       \epsilon_n \leq \max(1, |c|) \sum_{k=n+1}^{\infty} (2k+1)\,|b|^k .

   :param b: Asymmetry parameter (:math:`|b| < 1`).
   :type b: float
   :param c: Backscatter fraction.
   :type c: float
   :param n: Truncation order.
   :type n: int

   :returns: Upper bound on the max absolute error of the reconstructed phase
             function. ``inf`` if :math:`|b| \geq 1`.
   :rtype: float


.. py:function:: recommended_dhg_order(b, c, tol = DHG_TRUNCATION_WARN_THRESHOLD, max_order = 300)

   Smallest Legendre order whose truncation-error bound is below *tol*.

   See :func:`dhg_truncation_error` for the bound.

   :param b: Asymmetry parameter.
   :type b: float
   :param c: Backscatter fraction.
   :type c: float
   :param tol: Target error bound.
   :type tol: float, optional
   :param max_order: Upper limit for the search (returned if no order satisfies *tol*).
   :type max_order: int, optional

   :returns: Recommended number of Legendre orders.
   :rtype: int


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


.. py:function:: cs_legendre_coefficients(xi, n = 15)

   Legendre expansion coefficients for the Cornette–Shanks phase function.

   .. deprecated:: 1.1
       Unvalidated and likely misaligned: the returned coefficients start
       at order 1 (not 0), so they are shifted by one order relative to
       what :func:`legendre_eval` and :func:`function_p` expect. Use
       :func:`refmod.hapke.cornette.cornette_legendre_coefficients` (the
       MATLAB-derived variant) if Cornette support is needed. Kept for
       reference until the Cornette models are validated.

   :param xi: Asymmetry parameter :math:`\xi`.
   :type xi: float
   :param n: Number of coefficients to compute (default 15). Returns *n+1* values.
   :type n: int, optional

   :returns: Array of Legendre coefficients :math:`b_n` of length *n+1*.
   :rtype: jax.Array

   .. rubric:: References

   :cite:p:`Cornette-1992`


.. py:function:: double_henyey_greenstein(cos_g, b, c)

   Double Henyey–Greenstein (DHG) phase function.

   .. math::

       P(\cos g) = \frac{1 + c}{2}
       \frac{1 - b^2}{(1 - 2b\cos g + b^2)^{3/2}}
       + \frac{1 - c}{2}
       \frac{1 - b^2}{(1 + 2b\cos g + b^2)^{3/2}}

   :param cos_g: Cosine of the phase angle :math:`\cos g`.
   :type cos_g: jax.Array
   :param b: Asymmetry parameter.
   :type b: float
   :param c: Backscatter fraction.
   :type c: float

   :returns: Phase function value(s) at the given angle(s).
   :rtype: jax.Array

   .. rubric:: References

   :cite:p:`Henyey-1941`


.. py:function:: cornette_shanks(cos_g, xi)

   Cornette–Shanks phase function.

   .. math::

       P(\cos g) = \frac{3}{2} \frac{1 - \xi^2}{2 + \xi^2}
       \frac{1 + \cos^2 g}{(1 + \xi^2 - 2\xi\cos g)^{3/2}}

   :param cos_g: Cosine of the phase angle :math:`\cos g`.
   :type cos_g: jax.Array
   :param xi: Asymmetry parameter :math:`\xi`.
   :type xi: float

   :returns: Phase function value(s) at the given angle(s).
   :rtype: jax.Array

   .. rubric:: References

   :cite:p:`Cornette-1992`


.. py:function:: legendre_eval(x, b_n)

   Evaluate a Legendre polynomial series via the Bonnet recurrence.

   Computes :math:`\sum_{n=0}^{N} b_n P_n(x)` using the three-term
   recurrence :math:`n P_n = (2n-1) x P_{n-1} - (n-1) P_{n-2}`.

   :param x: Argument :math:`x` where :math:`|x| \leq 1`.
   :type x: jax.Array
   :param b_n: Coefficients :math:`b_n` of the Legendre series.
   :type b_n: jax.Array

   :returns: Value of the Legendre series at *x*.
   :rtype: jax.Array

   .. rubric:: References

   :cite:p:`Hapke-2002`


.. py:function:: function_p(x, b_n, a_n)

   Hapke P-function for anisotropic multiple scattering.

   Computes the single-scattering phase function contribution to multiple
   scattering (Hapke 2002, Eqs. 23–24):

   .. math::

       P(\cos g) = 1 + \sum_{n=0}^{N} a_n b_n P_n(\cos g)

   where :math:`a_n` are the Hapke coefficients and :math:`b_n` are the
   Legendre expansion coefficients of the single-particle phase function.

   :param x: Cosine of the phase angle :math:`\cos g`.
   :type x: jax.Array
   :param b_n: Legendre expansion coefficients of the phase function.
   :type b_n: jax.Array
   :param a_n: Hapke coefficients :math:`a_n`.
   :type a_n: jax.Array

   :returns: Value of the P-function at *x*.
   :rtype: jax.Array

   .. rubric:: References

   :cite:p:`Hapke-2002`


.. py:function:: value_p(b_n, a_n)

   Scalar value of the Hapke P-function.

   Computes the scalar P-value (Hapke 2002, Eq. 25):

   .. math::

       \langle P \rangle = 1 + \sum_{n=0}^{N} a_n^2 b_n

   :param b_n: Legendre expansion coefficients of the phase function.
   :type b_n: jax.Array
   :param a_n: Hapke coefficients :math:`a_n`.
   :type a_n: jax.Array

   :returns: Scalar :math:`\langle P \rangle`.
   :rtype: jax.Array

   .. rubric:: References

   :cite:p:`Hapke-2002`


.. py:function:: shadow_hiding(tan_alpha_2, h, b0)

   Shadow-hiding opposition effect :math:`B_{SH}`.

   .. math::

       B_{SH}(\alpha) = 1 + \frac{B_0}{1 + \tan(\alpha/2) / h}

   where :math:`B_0` is the opposition surge amplitude and *h* is the
   angular width of the opposition effect (Hapke 1984).

   :param tan_alpha_2: Tangent of half the phase angle, :math:`\tan(\alpha/2)`.
   :type tan_alpha_2: jax.Array
   :param h: Angular width parameter.
   :type h: float
   :param b0: Opposition surge amplitude :math:`B_0`.
   :type b0: float

   :returns: Shadow-hiding opposition effect factor :math:`B_{SH}`.
   :rtype: jax.Array

   .. rubric:: References

   :cite:p:`Hapke-1984`


.. py:function:: coherent_backscatter(tan_alpha_2, h, b0)

   Coherent backscatter opposition effect :math:`B_{CB}`.

   Computes the coherent backscatter enhancement factor (Hapke 2002):

   .. math::

       B_{CB}(\alpha) = 1 + B_0 \,
       \frac{1}{2}
       \frac{1 + (1 - e^{-x}) / x}{(1 + x)^2},
       \quad x = \frac{\tan(\alpha/2)}{h}

   :param tan_alpha_2: Tangent of half the phase angle, :math:`\tan(\alpha/2)`.
   :type tan_alpha_2: jax.Array
   :param h: Angular width parameter.
   :type h: float
   :param b0: Opposition surge amplitude :math:`B_0`.
   :type b0: float

   :returns: Coherent backscatter opposition effect factor :math:`B_{CB}`.
   :rtype: jax.Array

   .. rubric:: References

   :cite:p:`Hapke-2002`


.. py:function:: _fe(x, y)

   Exponential factor for the roughness correction.

   :param x: Cotangent or transformed angle argument.
   :type x: jax.Array
   :param y: Scale argument.
   :type y: jax.Array

   :returns: :math:`\exp(-2 y x / \pi)` or 0 where *x* is infinite.
   :rtype: jax.Array


.. py:function:: _fe2(x, y)

   Gaussian factor for the roughness correction.

   :param x: Cotangent or transformed angle argument.
   :type x: jax.Array
   :param y: Scale argument.
   :type y: jax.Array

   :returns: :math:`\exp(-y^2 x^2 / \pi)` or 0 where *x* is infinite.
   :rtype: jax.Array


.. py:function:: roughness_correction(roughness, i, e, n)

   Microscopic roughness shadowing correction.

   Computes the macroscopic roughness correction factor :math:`S` and
   effective cosines :math:`\mu_{0e}` and :math:`\mu_e` following
   Hapke (1984).

   :param roughness: RMS slope angle :math:`\bar{\theta}` in radians.
   :type roughness: float
   :param i: Unit vector toward the light source.
   :type i: jax.Array
   :param e: Unit vector toward the observer.
   :type e: jax.Array
   :param n: Unit surface normal vector.
   :type n: jax.Array

   :returns: A tuple ``(S, mu0, mu)`` where:

             - *S* is the roughness shadowing correction factor.
             - *mu0* is the effective cosine of the incidence angle.
             - *mu* is the effective cosine of the emission angle.
   :rtype: tuple[jax.Array, jax.Array, jax.Array]

   .. rubric:: References

   :cite:p:`Hapke-1984`


.. py:function:: _roughness_impl(roughness, i, e, n, cos_i, cos_e)

   Implementation of the microscopic roughness shadowing correction.

   Computes the roughness correction factor and effective cosines for
   non-zero roughness values. For zero or near-zero roughness, the
   identity correction is returned by the caller
   :func:`roughness_correction`.

   :param roughness: RMS slope angle :math:`\bar{\theta}` in radians (assumed non-zero).
   :type roughness: float
   :param i: Unit vector toward the light source.
   :type i: jax.Array
   :param e: Unit vector toward the observer.
   :type e: jax.Array
   :param n: Unit surface normal vector.
   :type n: jax.Array
   :param cos_i: Pre-computed cosine of incidence angle.
   :type cos_i: jax.Array
   :param cos_e: Pre-computed cosine of emission angle.
   :type cos_e: jax.Array

   :returns: A tuple ``(S, mu0, mu)`` as in :func:`roughness_correction`.
   :rtype: tuple[jax.Array, jax.Array, jax.Array]

   .. rubric:: References

   :cite:p:`Hapke-1984`


