refmod.lunar_lambert
====================

.. py:module:: refmod.lunar_lambert






Module Contents
---------------

.. py:data:: _POINTS

.. py:data:: _LUT_X

.. py:data:: _LUT_Y

.. py:function:: lunar_lambert(rho, th_i, th_e, alpha)

   MATLAB-compatible lunar-Lambert reflectance model.

   This ports the reference ``lunar_lambert_model.m``. It is a lookup-table
   model, tabulated for :math:`\bar{\theta} = 10^\circ` and :math:`w = 0.1`,
   and is intentionally separate from physical Lambertian reflectance
   (:func:`~refmod.lambert.lambert`).


