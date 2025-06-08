# The IMSA (Isotropic Multiple Scattering Approximation) Model

The term IMSA in the Hapke framework historically stands for **Isotropic Multiple Scattering Approximation**. It refers to a foundational version of the theory where the multiple-scattering component of light is treated as isotropic, even if the single-particle phase function itself is anisotropic {cite}`Hapke-2002`.

The name "Inversion of Multiple Scattering and Absorption," which may be suggested by a module's name, can describe the intended application of this particular variant, which is well-suited for inverting reflectance data to derive physical parameters.

The function `refmod.hapke.imsa.imsa` implements a model consistent with this classic IMSA formulation, augmented with terms for opposition effects and macroscopic roughness.

## IMSA Reflectance Equation

The fundamental IMSA equation, as given by {cite:t}`Hapke-2002` (Eq. 1), is:

$$
r(i, e, g) = \frac{w}{4\pi} \frac{\mu_0}{\mu_0 + \mu} \left[ p(g) + H(\mu_0)H(\mu) - 1 \right]
$$

This base equation is then typically modified with a simplified opposition effect (like SHOE) and a standard macroscopic roughness correction.

## Key Features

- **Multiple Scattering**: Treated as isotropic, using the standard product of Chandrasekhar's H-functions, $H(\mu_0)H(\mu)$. Simpler H-function approximations (like Eq. 2 in {cite:t}`Hapke-2002`) are often associated with this model.
- **Phase Function ($p(g)$)**: The `refmod` implementation allows for flexible, user-supplied callable phase functions.
- **Simplicity for Inversion**: Due to its relative simplicity and fewer free parameters compared to the full AMSA model, the IMSA model is well-suited for use in inversion routines that fit observational data to derive physical parameters.

> **Note on `refmod` Normalization**
>
> An important implementation detail in `refmod.hapke.imsa.imsa` is an additional division by $4\pi$ (`refl /= 4*np.pi`). This is a significant deviation from the published formula (Eq. 1 in {cite:t}`Hapke-2002`) and should be treated with care when comparing `refmod` outputs to results from other standard Hapke models.
