# The AMSA (Advanced Modified Shadowing and Coherent Backscattering) Model

The AMSA model is an advanced formulation within the Hapke framework, designed to provide a more comprehensive treatment of opposition effects and anisotropic scattering. It builds upon earlier Hapke models by incorporating more detailed terms for shadow hiding, coherent backscattering, and the Legendre polynomial expansion of the phase function.

The functions `refmod.hapke.amsa.amsa` and `refmod.hapke.amsa.amsa_derivative` implement this model.

## Key Components of AMSA

The AMSA model typically combines the following elements:

1.  **Single-Scattering Albedo ($)**: As in all Hapke models, this is fundamental.
2.  **Particle Phase Function ((g)$)**:
    - Often represented by a Legendre polynomial expansion: (g) = 1 + \sum b_n P_n(\cos g)$.
    - The coefficients $ are inputs (see `refmod.hapke.legendre.coef_b`).
    - Alternatively, other phase functions like Double Henyey-Greenstein or Cornette-Shanks can be used if the formulation is adapted. The `refmod.hapke.amsa.amsa` function takes a `phase_function_type` argument.
3.  **Anisotropic Scattering Corrections (Hapke, 2002)**:
    - Uses auxiliary coefficients $ (see `refmod.hapke.legendre.coef_a`).
    - Involves functions (\mu_0)$, (\mu)$ and a constant $ derived from $ and $ (see `refmod.hapke.legendre.function_p` and `refmod.hapke.legendre.value_p`).
    - These are used to construct an $ term:  = P(\mu_0)(H(\mu)-1) + P(\mu)(H(\mu_0)-1) + P_0(H(\mu_0)-1)(H(\mu)-1)$.
4.  **Chandrasekhar's H-functions ((\mu, w)$)**:
    - Typically uses Hapke's second approximation (see `refmod.hapke.functions.h_function_2`).
5.  **Shadow-Hiding Opposition Effect (SHOE)**:
    - Modeled with an amplitude {s0}$ and an angular width parameter $.
    - The term is often {SH}(g) = 1 + \frac{B_{s0}}{1 + \tan(g/2)/h_s}$.
6.  **Coherent Backscatter Opposition Effect (CBOE)**:
    - Modeled with an amplitude {c0}$ and an angular width parameter $.
    - The term {CB}(g)$ has a more complex form, e.g., {CB}(g) = 1 + B_{c0} \frac{1/2 (1 + (1 - e^{-x})/x )}{(1+x)^2}$ where  = \tan(g/2)/h_c$. (This is one form, exact implementation details should be checked in code).
7.  **Macroscopic Roughness Correction ($)**:
    - Applied as described by Hapke (1984) using `refmod.hapke.roughness.microscopic_roughness`.

## AMSA Reflectance Equation Sketch

A simplified conceptual form of the AMSA reflectance might look like:

2348 r(i, e, g) = \frac{w}{4\pi} \frac{\mu_0}{\mu_0 + \mu} S \cdot B_{CB}(g) \cdot \left[ p(g)B_{SH}(g) + M \right] 2348

Note: $\mu_0$ and $\mu$ here are the cosines of effective incidence and emission angles after roughness correction. The exact structure and combination of terms can be intricate and should be verified against a primary reference for the specific AMSA variant implemented. The `__amsa_preprocess` internal function in `refmod.hapke.amsa` sets up many of these terms.

## Derivative

The `refmod.hapke.amsa.amsa_derivative` function calculates /dw$, the derivative of the AMSA reflectance with respect to the single-scattering albedo. This is crucial for model inversion and parameter fitting.

## References

-   A specific primary reference for the AMSA model variant implemented in `refmod` should be identified. For now, we use **[AMSAModelPlaceholder]**.
-   Hapke, B. (2002). Bidirectional Reflectance Spectroscopy: 5. The Coherent Backscatter Opposition Effect and Anisotropic Scattering. *Icarus*, 157(2), 523–534. (Provides basis for anisotropic scattering and CBOE).
-   Hapke, B. (1984). Bidirectional reflectance spectroscopy: 3. Correction for macroscopic roughness. *Icarus*, 59(1), 41-59.
-   See also [references.md](references.md).
