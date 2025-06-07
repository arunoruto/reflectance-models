# The IMSA (Inversion of Multiple Scattering and Absorption) Model

The IMSA model is another variant within the Hapke family of reflectance models. The name suggests a focus on inverting reflectance data to derive physical parameters, particularly the single-scattering albedo ($) and potentially phase function parameters.

The function `refmod.hapke.imsa.imsa` implements this model.

## Key Features and Distinctions

While sharing core components with other Hapke models, IMSA might have specific simplifications or emphases:

1.  **Single-Scattering Albedo ($)**: Remains a central parameter.
2.  **Particle Phase Function ((g)$)**:
    - The `imsa` function in `refmod` takes a callable `phase_function` argument, allowing flexibility. This differs from `amsa` which uses `phase_function_type` and coefficients.
3.  **Chandrasekhar's H-functions ((\mu, w)$)**:
    - The model uses H-functions (e.g., `refmod.hapke.functions.h_function`).
4.  **Opposition Effect**:
    - The `imsa` function includes parameters `opposition_effect_h` and `oppoistion_effect_b0` (note: potential typo in original, likely $). This suggests a simplified opposition effect term, possibly combining or focusing on the shadow-hiding component:
      (g) = 1 + \frac{B_0}{1 + \tan(g/2)/h}$.
    - This is simpler than the separate SHOE and CBOE terms often detailed in more advanced models like AMSA.
5.  **Macroscopic Roughness Correction ($)**:
    - Uses the Hapke (1984) correction via `refmod.hapke.roughness.microscopic_roughness`.

## IMSA Reflectance Equation Sketch

A general form for the IMSA model, as suggested by the `refmod.hapke.imsa.imsa` implementation, might be:

2348 r(i, e, g) = \frac{w}{4\pi} \frac{\mu_0}{\mu_0 + \mu} S \cdot \left[ B_G(g) p(g) + H(\mu_0, w)H(\mu, w) - 1 \right] 2348

Where:

- $\mu_0$ and $\mu$ are effective cosines of incidence/emission angles after roughness correction.
- (g)$ is the opposition effect term.
- (g)$ is the particle phase function.
- $ is the macroscopic roughness correction.

The term /(4\pi)$ appearing as an additional divisor in the `refmod.hapke.imsa.imsa` code (`refl /= 4*np.pi`) is unusual compared to standard Hapke formulations (where the /(4\pi)$ is typically at the beginning of the /(4\pi)$ term). This might be a specific normalization or convention in this IMSA variant.

## Inversion Focus

The name "Inversion of Multiple Scattering and Absorption" suggests that this model might be structured or simplified in a way that is particularly amenable to inversion algorithms (e.g., iterative fitting to find $, phase function parameters, etc., from measured reflectance data). However, the provided code is a forward model.
