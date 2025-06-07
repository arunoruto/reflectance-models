refmod.hapke
============

.. py:module:: refmod.hapke


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/refmod/hapke/amsa/index
   /autoapi/refmod/hapke/functions/index
   /autoapi/refmod/hapke/imsa/index
   /autoapi/refmod/hapke/legendre/index
   /autoapi/refmod/hapke/mimsa/index
   /autoapi/refmod/hapke/roughness/index






Package Contents
----------------

.. py:function:: amsa(incidence_direction, emission_direction, surface_orientation, single_scattering_albedo, phase_function_type, b_n, a_n, hs = 0, bs0 = 0, roughness = 0, hc = 0, bc0 = 0, phase_function_args = (), refl_optimization = None)

   Calculates the reflectance using the AMSA model.

   :param incidence_direction: Incidence direction vector(s) of shape (..., 3).
   :type incidence_direction: npt.NDArray
   :param emission_direction: Emission direction vector(s) of shape (..., 3).
   :type emission_direction: npt.NDArray
   :param surface_orientation: Surface orientation vector(s) of shape (..., 3).
   :type surface_orientation: npt.NDArray
   :param single_scattering_albedo: Single scattering albedo.
   :type single_scattering_albedo: npt.NDArray
   :param phase_function_type: Type of phase function to use.
   :type phase_function_type: PhaseFunctionType
   :param b_n: Coefficients of the Legendre expansion.
   :type b_n: npt.NDArray
   :param a_n: Coefficients of the Legendre expansion.
   :type a_n: npt.NDArray
   :param hs: Shadowing parameter, by default 0.
   :type hs: float, optional
   :param bs0: Shadowing parameter, by default 0.
   :type bs0: float, optional
   :param roughness: Surface roughness, by default 0.
   :type roughness: float, optional
   :param hc: Coherent backscattering parameter, by default 0.
   :type hc: float, optional
   :param bc0: Coherent backscattering parameter, by default 0.
   :type bc0: float, optional
   :param phase_function_args: Additional arguments for the phase function, by default ().
   :type phase_function_args: tuple, optional
   :param refl_optimization: Reflectance optimization array, by default None.
   :type refl_optimization: npt.NDArray | None, optional

   :returns: Reflectance values.
   :rtype: npt.NDArray

   :raises Exception: If at least one reflectance value is not real.
   :raises References:
   :raises ----------:
   :raises [AMSAModelPlaceholder]:


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


.. py:function:: imsa(incidence_direction, emission_direction, surface_orientation, single_scattering_albedo, phase_function, opposition_effect_h = 0, oppoistion_effect_b0 = 0, roughness = 0)

   Calculates reflectance using the IMSA model.

   IMSA stands for Inversion of Multiple Scattering and Absorption.

   :param incidence_direction: Incidence direction vector(s), shape (..., 3).
   :type incidence_direction: npt.NDArray
   :param emission_direction: Emission direction vector(s), shape (..., 3).
   :type emission_direction: npt.NDArray
   :param surface_orientation: Surface normal vector(s), shape (..., 3).
   :type surface_orientation: npt.NDArray
   :param single_scattering_albedo: Single scattering albedo, shape (...).
   :type single_scattering_albedo: npt.NDArray
   :param phase_function: Callable that accepts `cos_alpha` (cosine of phase angle) and
                          returns phase function values.
   :type phase_function: Callable[[npt.NDArray], npt.NDArray]
   :param opposition_effect_h: Opposition effect parameter h, by default 0.
   :type opposition_effect_h: float, optional
   :param oppoistion_effect_b0: Opposition effect parameter B0 (b_zero), by default 0.
                                Note: Original argument name `oppoistion_effect_b0` kept for API compatibility.
   :type oppoistion_effect_b0: float, optional
   :param roughness: Surface roughness parameter, by default 0.
   :type roughness: float, optional

   :returns: Calculated reflectance values, shape (...).
   :rtype: npt.NDArray

   :raises Exception: If any calculated reflectance value has a significant imaginary part.

   .. rubric:: Notes

   - Input arrays `incidence_direction`, `emission_direction`,
     `surface_orientation`, and `single_scattering_albedo` are expected to
     broadcast together.
   - The `phase_function` should be vectorized to handle arrays of `cos_alpha`.
   - The IMSA model accounts for multiple scattering and absorption.

   References

   ----------

   [IMSAModelPlaceholder]


.. py:class:: Hapke(incidence_direction, emission_direction, surface_orientation, single_scattering_albedo, phase_function, opposition_effect_h = 0, oppoistion_effect_b0 = 0, roughness = 0)

   .. py:attribute:: sun


   .. py:attribute:: cam


   .. py:attribute:: normal


   .. py:attribute:: w


   .. py:attribute:: p


   .. py:attribute:: h
      :value: 0



   .. py:attribute:: b0
      :value: 0



   .. py:attribute:: tb
      :value: 0



