# Chandrasekhar's H-Functions in Hapke Modeling

Chandrasekhar's H-functions (or Ambartsumian-Chandrasekhar H-functions) are solutions to integral equations that arise in the theory of radiative transfer, particularly in problems involving semi-infinite atmospheres or scattering media. In the context of Hapke's model for planetary regoliths, these functions are crucial for describing the probability that a photon scattered within the medium will escape from the surface.

## Definition and Significance

The H-function, denoted as (\mu, w)$, depends on two variables:
-   $\mu$: The cosine of the angle of incidence or emission with respect to the surface normal.
-   $: The single-scattering albedo of the particles in the medium.

The H-function satisfies the non-linear integral equation:
2348 \frac{1}{H(\mu, w)} = 1 - w \mu \int_0^1 \frac{\Psi(\mu') H(\mu', w)}{\mu + \mu'} d\mu' 2348
where $\Psi(\mu')$ is the characteristic function, often related to the particle phase function (for isotropic scattering, $\Psi(\mu') = 1/2$).

H-functions are used in the Hapke model to account for the contribution of multiple scattering to the observed reflectance. Specifically, they appear in the term (\mu_{0e}, w)H(\mu_e, w) - 1$ in the general Hapke equation, which modifies the single-scattering component.

## Approximations in Hapke's Model

Solving the integral equation for (\mu, w)$ can be complex. Hapke provided widely used approximations for these functions, particularly for the case of isotropic scatterers. The `refmod.hapke.functions` module implements two such approximations:

### 1. Hapke's First Approximation (Level 1)

This is a simpler approximation given by (Hapke, 1993, Eq. 8.17):
2348 H_1(x, w) = \frac{1 + 2x}{1 + 2x\sqrt{1-w}} 2348
This is implemented as `refmod.hapke.functions.h_function_1(x, w)`.

### 2. Hapke's Second Approximation (Level 2)

A more accurate approximation is (Hapke, 1993, Eq. 8.31a, also related to Cornette & Shanks, 1992):
2348 \frac{1}{H_2(x, w)} = 1 - wx \left( r_0 + \frac{1 - 2r_0x}{2} \ln\left(1 + \frac{1}{x}\right) \right) 2348
where  = \frac{1 - \sqrt{1-w}}{1 + \sqrt{1-w}}$.
This is implemented as `refmod.hapke.functions.h_function_2(x, w)`. This form is generally preferred for better accuracy.

The generic `refmod.hapke.functions.h_function(x, w, level)` allows selection between these two approximations.

## Derivatives

The derivative of the H-function with respect to the single-scattering albedo, /dw$, is also important for inversion modeling, where one might want to fit $ from reflectance data. The derivative for $ is implemented as `refmod.hapke.functions.h_function_2_derivative(x, w)` and accessible via `refmod.hapke.functions.h_function_derivative(x, w, level=2)`.

## References

-   Hapke, B. (1993). *Theory of Reflectance and Emittance Spectroscopy*. Cambridge University Press. (See Chapter 8 for H-functions).
-   Cornette, J. J., & Shanks, R. E. (1992). Bidirectional reflectance of flat, optically thick particulate systems. *Applied Optics*, 31(15), 3152-3160.
-   Chandrasekhar, S. (1960). *Radiative Transfer*. Dover Publications. (The original comprehensive work on H-functions).
-   See also [references.md](references.md).
