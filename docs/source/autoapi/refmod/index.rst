refmod
======

.. py:module:: refmod


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/refmod/dtm_helper/index
   /autoapi/refmod/hapke/index
   /autoapi/refmod/inverse/index




Package Contents
----------------

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


