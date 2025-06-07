# Particle Phase Functions in Hapke Modeling

The particle phase function, commonly denoted as $P(g)$ or $p(g)$ (where $g$ is the phase angle), describes the angular distribution of light scattered by a single particle or an ensemble average of particles within a particulate medium. It is a fundamental component of radiative transfer models, including Hapke's model {cite}`Hapke-2012`, significantly influencing the overall brightness and angular reflectance characteristics of the modeled surface.

The phase angle $g$ is the angle between the direction of incident light and the direction of scattered light. It typically ranges from $0^\circ$ (exact forward scatter) to $180^\circ$ (exact backscatter). _Correction: Your text said $0^\circ$ (backscatter) to $180^\circ$ (forward scatter); conventionally, $g=0$ is forward and $g=180^\circ$ (or $\pi$ radians) is backscatter. I've adjusted this._

## Common Phase Function Models

Several analytical functions are commonly used to represent particle scattering behavior in Hapke models. The `refmod.hapke.functions` module implements some of these:

### 1. Double Henyey-Greenstein (DHG)

The Double Henyey-Greenstein (DHG) function is a versatile two-parameter function that can model a wide range of scattering behaviors, including those with distinct forward and backward scattering lobes. It is defined as a weighted sum of two Henyey-Greenstein functions:

$$
P_{DHG}(g; b, c) = \frac{1+c}{2} \frac{1-b^2}{(1 - 2b\cos(g) + b^2)^{3/2}} + \frac{1-c}{2} \frac{1-b^2}{(1 + 2b\cos(g) + b^2)^{3/2}}
$$

Where:

- $g$ is the phase angle.
- $b$ is an asymmetry parameter ($0 \le b < 1$) controlling the sharpness of the scattering lobes. A value of $b=0$ corresponds to isotropic scattering for each lobe.
- $c$ is a parameter ($-1 \le c \le 1$) controlling the relative strength of the forward-scattering lobe (first term, peaking near $g=0$) versus the backward-scattering lobe (second term, peaking near $g=\pi$). If $c=1$, only the forward lobe contributes; if $c=-1$, only the backward lobe contributes.

This function is implemented in `refmod.hapke.functions.double_henyey_greenstein(cos_g, b, c)`.

### 2. Cornette-Shanks

The Cornette-Shanks phase function is another empirical model often used for particulate surfaces, known for its ability to fit a variety of scattering patterns. It is given by ({cite:t}`Cornette-1992`, Eq. 8):

$$
P_{CS}(g; \xi) = \frac{3}{2} \frac{1-\xi^2}{2+\xi^2} \frac{1+\cos^2(g)}{(1 + \xi^2 - 2\xi\cos(g))^{3/2}}
$$

Where:

- $g$ is the phase angle.
- $\xi$ is an asymmetry parameter (often denoted $g$ in the original paper, but $\xi$ or other symbols like $g_{CS}$ are frequently used in implementations to avoid confusion with the phase angle $g$). It is related to the average cosine of the scattering angle.

This function is implemented in `refmod.hapke.functions.cornette_shanks(cos_g, xi)`.

### 3. Legendre Polynomial Expansion

As detailed in a separate section (see {doc}`legendre_polynomials`), phase functions can also be represented as an expansion in Legendre polynomials $P_n(\cos g)$:

$$
P(g) = \sum_{n=0}^{N} c_n P_n(\cos g)
$$

Or, in a common alternative form:

$$
P(g) = 1 + \sum_{n=1}^{N} b_n P_n(\cos g)
$$

Where:

- $c_n$ or $b_n$ are the expansion coefficients. These coefficients are related to the physical scattering properties of the particles and can be derived from more fundamental scattering theories (like Mie theory) or fitted to experimental data.
- {cite:t}`Hapke-2002` discusses the derivation and use of these coefficients in the context of his model (see also the `refmod.hapke.legendre.coef_b` function).

The `refmod.hapke.phase_function` utility function within your library likely serves as a dispatcher, selecting and computing the phase function based on the chosen model type and its corresponding parameters.

## Role in Hapke Model

The chosen particle phase function $P(g)$ is a critical input to the main Hapke reflectance equation. It directly scales the single-scattered component of light. Furthermore, its integral properties (e.g., the asymmetry factor, or the full set of Legendre coefficients) influence the calculation of multiple-scattering terms, often through modifications to Chandrasekhar's H-functions or other parts of the radiative transfer solution.
