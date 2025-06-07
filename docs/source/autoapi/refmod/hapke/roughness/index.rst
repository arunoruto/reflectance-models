refmod.hapke.roughness
======================

.. py:module:: refmod.hapke.roughness

.. autoapi-nested-parse::

   Surface roughness corrections based on Hapke's model.

   This module provides functions to calculate corrections for macroscopic
   surface roughness, a key component in photometric modeling as described
   by Hapke.

   ??? info "References"

       Hapke (1984)





Module Contents
---------------

.. py:function:: __f_exp(x, y)

   Helper function for microscopic roughness calculation.

   Calculates `exp(-2 * y * x / pi)`.

   :param x: Input array.
   :type x: npt.NDArray
   :param y: Factor, typically related to cot(roughness).
   :type y: float

   :returns: Result of the exponential function.
   :rtype: npt.NDArray


.. py:function:: __f_exp_2(x, y)

   Helper function for microscopic roughness calculation.

   Calculates `exp(-(y^2 * x^2) / pi)`.

   :param x: Input array.
   :type x: npt.NDArray
   :param y: Factor, typically related to cot(roughness), which is squared.
   :type y: float

   :returns: Result of the exponential function.
   :rtype: npt.NDArray


.. py:function:: microscopic_roughness(roughness, incidence_direction, emission_direction, surface_orientation)

   Calculates the microscopic roughness factor for Hapke's model.

   This correction accounts for the effects of sub-resolution roughness on
   the observed reflectance.

   :param roughness: The mean slope angle of surface facets, in radians.
                     A value of 0 means a smooth surface.
   :type roughness: float
   :param incidence_direction: Incidence direction vector(s), shape (..., 3). Assumed to be normalized.
   :type incidence_direction: npt.NDArray
   :param emission_direction: Emission direction vector(s), shape (..., 3). Assumed to be normalized.
   :type emission_direction: npt.NDArray
   :param surface_orientation: Surface normal vector(s), shape (..., 3). Assumed to be normalized.
   :type surface_orientation: npt.NDArray

   :returns: * **s** (*npt.NDArray*) -- The microscopic roughness factor, shape (...).
             * **mu_0_prime** (*npt.NDArray*) -- The modified cosine of the incidence angle ($\mu_0^{\prime}$), accounting
               for roughness, shape (...).
             * **mu_prime** (*npt.NDArray*) -- The modified cosine of the emission angle ($\mu^{\prime}$), accounting
               for roughness, shape (...).

   .. rubric:: Notes

   The calculations are based on Hapke (1984).

   - The terms $\mu_0^{\prime}$ (mu_0_s0, mu_0_s) and $\mu^{\prime}$ (mu_s0, mu_s)
     are calculated based on different conditions for incidence angle `i`
     and emission angle `e`:

     - For prime-zero terms ($\mu_0^{\prime(0)}$, $\mu^{\prime(0)}$ used in `mu_0_s0`, `mu_s0`):
       See Hapke (1984, Eqs. 48, 49).
     - For $\mu_0^{\prime}$ and $\mu^{\prime}$ when $i < e$:
       See Hapke (1984, Eqs. 46, 47).
     - For $\mu_0^{\prime}$ and $\mu^{\prime}$ when $i \ge e$:
       See Hapke (1984, Eqs. 50, 51).

   - Input vectors (`incidence_direction`, `emission_direction`, `surface_orientation`)
     are normalized internally.
   - If `roughness` is 0, `s` is 1, `mu_0_prime` is `cos(i)`, and `mu_prime` is `cos(e)`.

   .. rubric:: References

   Hapke (1984)


