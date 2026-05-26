refmod.lambert
==============

.. py:module:: refmod.lambert






Module Contents
---------------

.. py:function:: _lambert_scalar(w, i, n)

   Lambert reflectance for a single pixel.

   Computes :math:`w \cos i \,/\, \pi`.

   :param w: Single scattering albedo (scalar).
   :type w: jax.Array
   :param i: Incidence (illumination) direction vector.  Shape ``(3,)``.
   :type i: jax.Array
   :param n: Surface normal vector.  Shape ``(3,)``.
   :type n: jax.Array

   :returns: Reflectance (scalar).  Zero when the source is behind the surface.
   :rtype: jax.Array


.. py:data:: _lambert_batched

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


