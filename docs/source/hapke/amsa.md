# The AMSA (Anisotropic Multiple Scattering Approximation) Model

The AMSA model, an acronym for **Anisotropic Multiple Scattering Approximation**, represents an advanced formulation of Hapke's theory designed to more accurately account for the effects of anisotropic single-particle scattering on the multiple-scattering term. The definitive form of this model, which also incorporates a sophisticated treatment of opposition effects, is detailed by {cite:t}`Hapke-2002`.

The functions `refmod.hapke.amsa.amsa` and `refmod.hapke.amsa.amsa_derivative` in this library implement this comprehensive and powerful model.

## AMSA Reflectance Equation

The final expression for the AMSA model, as given by {cite:t}`Hapke-2002` (Eq. 38), combines the single-scattering term, the anisotropic multiple-scattering term, and both major opposition effects. When combined with the macroscopic roughness correction, the full equation is:

$$
r(i, e, g) = \frac{w}{4\pi} \frac{\mu_{0e}}{\mu_{0e} + \mu_e} \left[ p(g) B_{SH}(g) + M(\mu_{0e}, \mu_e) \right] B_{CB}(g) \cdot S(i, e, g, \bar{\theta})
$$

The components are broken down as follows:

1.  **Single-Scattering Term ($p(g) B_{SH}(g)$)**

    - $p(g)$ is the single-particle phase function (e.g., a Legendre polynomial expansion).
    - $B_{SH}(g)$ is the **Shadow-Hiding Opposition Effect (SHOE)**, which multiplies _only_ the single-scattering term. It is given by (Eq. 28, 29):
      $$ B*{SH}(g) = 1 + \frac{B*{S0}}{1 + \frac{1}{h_S} \tan(g/2)} $$

2.  **Anisotropic Multiple-Scattering Term ($M(\mu_{0e}, \mu_e)$)**

    - This is the core improvement of AMSA, replacing the simpler $H(\mu_0)H(\mu)-1$ term from IMSA. It is defined by {cite:t}`Hapke-2002` (Eq. 17):
      $$ M(\mu_0, \mu) = P(\mu_0)[H(\mu) - 1] + P(\mu)[H(\mu_0) - 1] + \bar{P}[H(\mu) - 1][H(\mu_0) - 1] $$
    - The functions $P(\mu_0)$, $P(\mu)$, and $\bar{P}$ are averaged phase functions defined in terms of Legendre coefficients.
    - The H-functions used here should be the more accurate "level 2" approximations (Eq. 13 in the paper; `h_function_2` in `refmod`).

3.  **Coherent Backscatter Opposition Effect ($B_{CB}(g)$)**

    - This term multiplies the _entire_ reflectance (both single and multiple scattering components). It is defined by (Eq. 32):
      $$ B*{CB}(g) = 1 + B*{C0} \cdot B_C(g) $$
        where $B_C(g)$ is a complex function modeling the coherent backscatter peak.

4.  **Macroscopic Roughness ($S$)**
    - The standard roughness correction {cite}`Hapke-1984` is applied to the final reflectance. The effective angles $\mu_{0e}$ and $\mu_e$ are used as inputs to the main scattering function.

## Derivative Function

The presence of `refmod.hapke.amsa.amsa_derivative` is highly significant. This function calculates $\partial r / \partial w$ (the derivative of the AMSA reflectance with respect to the single-scattering albedo $w$). This capability is crucial for model inversion and sensitivity analysis, making the AMSA implementation in `refmod` particularly powerful for quantitative analysis of remote sensing data.
