refmod.shkuratov
================

.. py:module:: refmod.shkuratov






Module Contents
---------------

.. py:function:: _shkuratov_scalar(a_n, mu1, eta, i, e, n, m0 = 0.0, mu2 = 0.0)

.. py:data:: _shkuratov_batched

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


