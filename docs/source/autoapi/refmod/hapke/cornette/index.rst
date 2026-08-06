refmod.hapke.cornette
=====================

.. py:module:: refmod.hapke.cornette






Module Contents
---------------

.. py:function:: cornette_legendre_coefficients(xi, n = 11)

   Cornette-Shanks Legendre expansion coefficients.

   Implemented from the reference ``hapke_amsa_cornette.m``. These have not
   yet been fixture-validated against the reference output; prefer the
   Double Henyey-Greenstein models for validated results.


.. py:function:: _cornette_multiple_scattering_terms(mu0, mu, xi, n_order)

.. py:function:: _refl_amsa_cornette_scalar(w, xi, i, e, n, roughness, h_sh, b0_sh, h_cb, b0_cb, n_order = 11)

.. py:function:: _refl_imsa_cornette_scalar(w, xi, i, e, n, roughness, h_sh, b0_sh)

.. py:data:: _amsa_cornette_batched

.. py:data:: _imsa_cornette_batched

.. py:function:: amsa_cornette(w, xi, i, e, n, roughness = 0.0, h_sh = 0.0, b0_sh = 0.0, h_cb = 0.0, b0_cb = 0.0, n_order = 11)

.. py:function:: imsa_cornette(w, xi, i, e, n, roughness = 0.0, h_sh = 0.0, b0_sh = 0.0)

