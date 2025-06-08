refmod.hapke.imsa
=================

.. py:module:: refmod.hapke.imsa




Module Contents
---------------

.. py:function:: imsa(single_scattering_albedo, incidence_direction, emission_direction, surface_orientation, phase_function, opposition_effect_h = 0, oppoistion_effect_b0 = 0, roughness = 0)

   Calculates reflectance using the IMSA model.

   IMSA stands for Inversion of Multiple Scattering and Absorption.

   :param single_scattering_albedo: Single scattering albedo, shape (...).
   :type single_scattering_albedo: npt.NDArray
   :param incidence_direction: Incidence direction vector(s), shape (..., 3).
   :type incidence_direction: npt.NDArray
   :param emission_direction: Emission direction vector(s), shape (..., 3).
   :type emission_direction: npt.NDArray
   :param surface_orientation: Surface normal vector(s), shape (..., 3).
   :type surface_orientation: npt.NDArray
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


