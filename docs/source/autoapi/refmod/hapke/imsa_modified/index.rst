refmod.hapke.imsa_modified
==========================

.. py:module:: refmod.hapke.imsa_modified






Module Contents
---------------

.. py:function:: _modified_h(x, w)

.. py:function:: _refl_imsa_modified_h_scalar(w, b, c, i, e, n, roughness, h_sh, b0_sh)

   MATLAB-compatible IMSA modified-H reflectance for one pixel.


.. py:data:: _imsa_modified_h_batched

.. py:function:: imsa_modified_h(w, b, c, i, e, n, roughness = 0.0, h_sh = 0.0, b0_sh = 0.0)

   Batched IMSA reflectance using Hapke's modified H-function.

   Combines a Double Henyey-Greenstein phase function, the shadow-hiding
   opposition effect, the macroscopic roughness correction, and Hapke's
   modified H-function approximation
   :math:`H(x) \approx (1 + 2x) / (1 + 2x\gamma)`.

   Matches the reference implementation ``hapke_imsa_modifiedH.m``. Note
   that the reference ``hapke_imsa.m`` implements this same model rather
   than the isotropic :func:`~refmod.hapke.imsa` in this package.


