# The AMSA (Anisotropic Multiple Scattering Approximation / Advanced Model)

The AMSA model, as implemented in `refmod.hapke.amsa.amsa` and `refmod.hapke.amsa.amsa_derivative`, represents an advanced and comprehensive formulation within the Hapke framework. The acronym AMSA often stands for "Anisotropic Multiple Scattering Approximation" in Hapke's literature {cite}`Hapke-2002`, which emphasizes a more rigorous treatment of how anisotropic single-particle scattering affects multiple scattering. The descriptive name "Advanced Modified Shadowing and Coherent Backscattering" aptly captures this implementation's inclusion of detailed opposition effects alongside anisotropic scattering.

This model builds upon earlier Hapke models by incorporating more sophisticated terms for shadow hiding, coherent backscattering, and often utilizes the Legendre polynomial expansion of the phase function for greater flexibility.

## Key Components of AMSA (based on `refmod` implementation)

The AMSA model, as reflected in `refmod`, typically combines the following elements:

1.  **Single-Scattering Albedo ($w$)**: As in all Hapke models, this is a fundamental parameter representing the probability of scattering per particle interaction.
2.  **Particle Phase Function ($P(g)$)**:
    - Often represented by a Legendre polynomial expansion: $P(g) = 1 + \sum_{n=1}^{N} b_n P_n(\cos g)$ (or a similar form with $c_n$ coefficients).
    - The Legendre coefficients $b_n$ (or $c_n$) are typically inputs (see `refmod.hapke.legendre.coef_b`).
    - The `refmod.hapke.amsa.amsa` function also accepts a `phase_function_type` argument, suggesting it can dispatch to other predefined phase functions (e.g., Double Henyey-Greenstein, Cornette-Shanks) if the overall formulation is adapted or if the Legendre coefficients are derived from them.
3.  **Anisotropic Multiple Scattering Corrections ({cite:t}`Hapke-2002`)**:
    - Utilizes auxiliary coefficients $a_n$ derived from Legendre polynomials (see `refmod.hapke.legendre.coef_a`).
    - Involves functions $\chi(\mu_0)$ and $\chi(\mu)$, and a constant $\chi_0$, which are derived from $a_n$ and $b_n$ coefficients (see `refmod.hapke.legendre.function_p` and `refmod.hapke.legendre.value_p`).
    - These are used to construct a multiple scattering term, $M_{aniso}$, that accounts for anisotropy. A common form for this term, before incorporating H-functions directly, involves these $\chi$ functions. The full expression for the multiply scattered radiance often takes the form $M = H(\mu_0)H(\mu) \left(1 + \sum_{n=1}^N \frac{b_n a_n P_n(\mu_0)P_n(\mu)}{\chi_0 - b_n a_n^2} \right) - 1$, or similar complex expressions involving $\chi(\mu_0)$, $\chi(\mu)$, and $\chi_0$ to modify the isotropic H-function product. _(The specific expression for $M$ needs to be carefully checked against the primary AMSA reference, e.g., Hapke (2002) or later papers, as the "M" term you sketched seems simplified or symbolic.)_
4.  **Chandrasekhar's H-functions ($H(\mu, w)$)**:
    - The model typically employs accurate approximations for H-functions, such as Hapke's second approximation (see `refmod.hapke.functions.h_function_2`), which are essential for calculating multiple scattering contributions.
5.  **Shadow-Hiding Opposition Effect (SHOE, $B_{SH}(g)$)**:
    - Modeled with an amplitude $B_{S0}$ and an angular width parameter $h_S$.
    - The term is often expressed as:
      $$
      B_{SH}(g) = 1 + \frac{B_{S0}}{1 + \frac{1}{h_S} \tan(g/2)}
      $$
6.  **Coherent Backscatter Opposition Effect (CBOE, $B_{CB}(g)$)**:
    - Modeled with an amplitude $B_{C0}$ and an angular width parameter $h_C$.
    - The CBOE term $B_{CB}(g)$ has a more complex functional form, for example, one common expression is:
      $$
      B_{CB}(g) = 1 + B_{C0} \frac{1/2 \left(1 + \frac{1 - e^{-x}}{x} \right)}{(1+x)^2}
      $$
      where $x = \frac{1}{h_C} \tan(g/2)$. _(The exact implementation details should be verified in the `refmod` code or the primary AMSA reference.)_
7.  **Macroscopic Roughness Correction ($S$)**:
    - Applied as described by {cite:t}`Hapke-1984`, presumably using a function like `refmod.hapke.roughness.macroscopic_roughness`.

## AMSA Reflectance Equation Sketch

The AMSA model combines these components into a comprehensive reflectance equation. A conceptual (highly simplified) sketch might be:

$$
r(i, e, g) = \frac{w}{4\pi} \frac{\mu_{0e}}{\mu_{0e} + \mu_e} S \cdot B_{CB}(g) \cdot \left[ P(g)B_{SH}(g) + M_{total} \right]
$$

Where:

- $\mu_{0e}$ and $\mu_e$ are effective cosines of incidence and emission angles, potentially modified by $S$.
- $S$ is the macroscopic roughness correction.
- $B_{CB}(g)$ is the coherent backscatter opposition effect.
- $P(g)$ is the single-particle phase function.
- $B_{SH}(g)$ is the shadow-hiding opposition effect.
- $M_{total}$ represents the complete multiple scattering term, incorporating H-functions and corrections for anisotropy (as alluded to in point 3 above). This term is significantly more complex than a simple $H(\mu_0)H(\mu)-1$.

**Note:** The exact structure, the order of application of correction factors (especially $B_{CB}$), and the precise form of the multiple scattering term $M_{total}$ are intricate and can vary between different detailed AMSA formulations. Verifying against a primary reference like {cite:t}`Hapke-2002` or subsequent papers (e.g., Hapke 2008, 2012) is crucial for an exact representation. The internal function `__amsa_preprocess` in `refmod.hapke.amsa` likely handles the setup of many of these complex terms.

## Derivative Function

The presence of `refmod.hapke.amsa.amsa_derivative` is highly significant. This function, by calculating $\partial r / \partial w$ (the derivative of the AMSA reflectance with respect to the single-scattering albedo $w$), provides essential information for:

- **Model Inversion:** Efficiently fitting the model to observed reflectance data to retrieve $w$ and other physical parameters.
- **Sensitivity Analysis:** Understanding how sensitive the reflectance is to changes in $w$.

This capability makes the AMSA implementation in `refmod` particularly powerful for quantitative analysis of remote sensing data.
