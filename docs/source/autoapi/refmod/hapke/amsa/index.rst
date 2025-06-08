refmod.hapke.amsa
=================

.. py:module:: refmod.hapke.amsa




Module Contents
---------------

.. py:function:: __amsa_preprocess(single_scattering_albedo, incidence_direction, emission_direction, surface_orientation, phase_function_type, b_n = None, a_n = None, roughness = 0.0, hs = 0.0, bs0 = 0.0, hc = 0.0, bc0 = 0.0, phase_function_args = ())

   Preprocesses the inputs for the AMSA model.

   :param single_scattering_albedo: Single scattering albedo.
   :type single_scattering_albedo: npt.NDArray
   :param incidence_direction: Incidence direction vector(s) of shape (..., 3).
   :type incidence_direction: npt.NDArray
   :param emission_direction: Emission direction vector(s) of shape (..., 3).
   :type emission_direction: npt.NDArray
   :param surface_orientation: Surface orientation vector(s) of shape (..., 3).
   :type surface_orientation: npt.NDArray
   :param phase_function_type: Type of phase function to use.
   :type phase_function_type: PhaseFunctionType
   :param b_n: Coefficients of the Legendre expansion.
   :type b_n: npt.NDArray
   :param a_n: Coefficients of the Legendre expansion.
   :type a_n: npt.NDArray
   :param roughness: Surface roughness, by default 0.0.
   :type roughness: float, optional
   :param hs: Shadowing parameter, by default 0.0.
   :type hs: float, optional
   :param bs0: Shadowing parameter, by default 0.0.
   :type bs0: float, optional
   :param hc: Coherent backscattering parameter, by default 0.0.
   :type hc: float, optional
   :param bc0: Coherent backscattering parameter, by default 0.0.
   :type bc0: float, optional
   :param phase_function_args: Additional arguments for the phase function, by default ().
   :type phase_function_args: tuple, optional

   :returns:

             A tuple containing:
                 - albedo_independent : npt.NDArray
                     Albedo-independent term.
                 - mu_0 : npt.NDArray
                     Cosine of the incidence angle.
                 - mu : npt.NDArray
                     Cosine of the emission angle.
                 - p_g : npt.NDArray
                     Phase function values.
                 - m : npt.NDArray
                     M term.
                 - p_mu_0 : npt.NDArray
                     Legendre polynomial values for mu_0.
                 - p_mu : npt.NDArray
                     Legendre polynomial values for mu.
                 - p : npt.NDArray
                     Legendre polynomial values.
                 - h_mu_0 : npt.NDArray
                     H-function values for mu_0.
                 - h_mu : npt.NDArray
                     H-function values for mu.
   :rtype: tuple


.. py:function:: amsa(single_scattering_albedo, incidence_direction, emission_direction, surface_orientation, phase_function_type, b_n = None, a_n = None, hs = 0, bs0 = 0, roughness = 0, hc = 0, bc0 = 0, phase_function_args = (), refl_optimization = None)

   Calculates the reflectance using the AMSA model.

   :param single_scattering_albedo: Single scattering albedo.
   :type single_scattering_albedo: npt.NDArray
   :param incidence_direction: Incidence direction vector(s) of shape (..., 3).
   :type incidence_direction: npt.NDArray
   :param emission_direction: Emission direction vector(s) of shape (..., 3).
   :type emission_direction: npt.NDArray
   :param surface_orientation: Surface orientation vector(s) of shape (..., 3).
   :type surface_orientation: npt.NDArray
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


.. py:function:: amsa_derivative(single_scattering_albedo, incidence_direction, emission_direction, surface_orientation, phase_function_type, b_n = None, a_n = None, roughness = 0, hs = 0, bs0 = 0, hc = 0, bc0 = 0, phase_function_args = (), refl_optimization = None)

   Calculates the derivative of the reflectance using the AMSA model.

   :param single_scattering_albedo: Single scattering albedo.
   :type single_scattering_albedo: npt.NDArray
   :param incidence_direction: Incidence direction vector(s) of shape (..., 3).
   :type incidence_direction: npt.NDArray
   :param emission_direction: Emission direction vector(s) of shape (..., 3).
   :type emission_direction: npt.NDArray
   :param surface_orientation: Surface orientation vector(s) of shape (..., 3).
   :type surface_orientation: npt.NDArray
   :param phase_function_type: Type of phase function to use.
   :type phase_function_type: PhaseFunctionType
   :param b_n: Coefficients of the Legendre expansion.
   :type b_n: npt.NDArray
   :param a_n: Coefficients of the Legendre expansion.
   :type a_n: npt.NDArray
   :param roughness: Surface roughness, by default 0.
   :type roughness: float, optional
   :param hs: Shadowing parameter, by default 0.
   :type hs: float, optional
   :param bs0: Shadowing parameter, by default 0.
   :type bs0: float, optional
   :param hc: Coherent backscattering parameter, by default 0.
   :type hc: float, optional
   :param bc0: Coherent backscattering parameter, by default 0.
   :type bc0: float, optional
   :param phase_function_args: Additional arguments for the phase function, by default ().
   :type phase_function_args: tuple, optional
   :param refl_optimization: Reflectance optimization array, by default None.
                             This parameter is not used in the derivative calculation.
   :type refl_optimization: npt.NDArray | None, optional

   :returns: * *npt.NDArray* -- Derivative of the reflectance with respect to single scattering albedo.
             * *References*
             * *----------*
             * *[AMSAModelPlaceholder]*


