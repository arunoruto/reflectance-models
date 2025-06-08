refmod.hapke.functions.phase
============================

.. py:module:: refmod.hapke.functions.phase

.. autoapi-nested-parse::

   ??? info "References"

       1. Cornette and Shanks (1992). Bidirectional reflectance
       of flat, optically thick particulate systems. Applied Optics, 31(15),
       3152-3160. <https://doi.org/10.1364/AO.31.003152>







Module Contents
---------------

.. py:data:: PhaseFunctionType

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


.. py:function:: cornette_shanks(cos_g, xi)

   Calculates the Cornette-Shanks phase function.

   :param cos_g: Cosine of the scattering angle (g).
   :type cos_g: npt.NDArray
   :param xi: Asymmetry parameter, related to the average scattering angle.
              Note: This `xi` is different from the single scattering albedo `w`.
   :type xi: float

   :returns: Phase function values.
   :rtype: npt.NDArray

   .. rubric:: References

   Cornette and Shanks (1992, Eq. 8).


.. py:function:: phase_function(cos_g, type, args)

   Selects and evaluates a phase function.

   :param cos_g: Cosine of the scattering angle (g).
   :type cos_g: npt.NDArray
   :param type: Type of phase function to use.
                Valid options are:
                - "dhg" or "double_henyey_greenstein": Double Henyey-Greenstein
                - "cs" or "cornette" or "cornette_shanks": Cornette-Shanks
   :type type: PhaseFunctionType
   :param args: Arguments for the selected phase function.
                - For "dhg": (b, c) where b is asymmetry and c is backscatter fraction.
                - For "cs": (xi,) where xi is the Cornette-Shanks asymmetry parameter.
   :type args: tuple

   :returns: Calculated phase function values.
   :rtype: npt.NDArray

   :raises Exception: If an unsupported `type` is provided.


