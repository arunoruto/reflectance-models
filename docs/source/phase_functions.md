# Particle Phase Functions in Hapke Modeling

The particle phase function, denoted as (g)$ or (g)$ (where $ is the phase angle), describes the angular distribution of light scattered by a single particle or an average particle within a particulate medium. It is a fundamental component of the Hapke model, influencing the overall brightness and angular reflectance characteristics of the surface.

The phase angle $ is the angle between the incident light ray and the scattered light ray. It ranges from -bash^\circ$ (exact backscatter, opposition) to 80^\circ$ (exact forward scatter).

## Common Phase Function Models

Several analytical functions are commonly used to represent particle scattering behavior in Hapke models. The `refmod.hapke.functions` module implements some of these:

### 1. Double Henyey-Greenstein (DHG)

The Double Henyey-Greenstein function is a versatile two-parameter function that can model a wide range of scattering behaviors, including those with both forward and backward scattering lobes. It is defined as:

2348 p_{DHG}(g, b, c) = \frac{1+c}{2} \frac{1-b^2}{(1 - 2b\cos(g) + b^2)^{3/2}} + \frac{1-c}{2} \frac{1-b^2}{(1 + 2b\cos(g) + b^2)^{3/2}} 2348

Where:
-   $ is the phase angle.
-   $ is an asymmetry parameter (-bash < b < 1$) controlling the sharpness of the scattering lobes.
-   $ is a parameter ($ -1 < c < 1 $) controlling the relative strength of the backscattering ( \approx 0$) versus forward scattering ( \approx \pi$) lobes.

This function is implemented in `refmod.hapke.functions.double_henyey_greenstein(cos_g, b, c)`.

### 2. Cornette-Shanks

The Cornette-Shanks phase function is another empirical model often used for particulate surfaces. It is given by:

2348 p_{CS}(g, \xi) = \frac{3}{2} \frac{1-\xi^2}{2+\xi^2} \frac{1+\cos^2(g)}{(1 + \xi^2 - 2\xi\cos(g))^{3/2}} 2348

Where:
-   $ is the phase angle.
-   $\xi$ is an asymmetry parameter, related to the average scattering angle. (Note: The original paper by Cornette & Shanks (1992) uses $ for this parameter, but $\xi$ or other symbols are often used in implementations to avoid confusion with phase angle $).

This function is implemented in `refmod.hapke.functions.cornette_shanks(cos_g, xi)`. It is noted as Equation 8 from Cornette and Shanks (1992).

### 3. Legendre Polynomial Expansion

Phase functions can also be represented as an expansion in Legendre polynomials (\cos g)$:

2348 p(g) = 1 + \sum_{n=1}^{N} b_n P_n(\cos g) 2348

Where:
-   $ are the expansion coefficients. These coefficients are related to the physical scattering properties of the particles.
-   Hapke (2002) discusses how to derive these coefficients (see `refmod.hapke.legendre.coef_b`).

The `refmod.hapke.phase_function` utility function can select between different phase function types based on input.

## Role in Hapke Model

The chosen phase function (g)$ is a key input to the main Hapke reflectance equation. It directly scales the single-scattered component of light and influences the multiple-scattering terms through its integral properties.

## References

-   Cornette, J. J., & Shanks, R. E. (1992). Bidirectional reflectance of flat, optically thick particulate systems. *Applied Optics*, 31(15), 3152-3160.
-   Hapke, B. (2002). Bidirectional Reflectance Spectroscopy: 5. The Coherent Backscatter Opposition Effect and Anisotropic Scattering. *Icarus*, 157(2), 523–534.
-   See also [references.md](references.md).
