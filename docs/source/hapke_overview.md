# Overview of Hapke's Bidirectional Reflectance Theory

Bruce Hapke's model is a widely used analytical model in planetary science and remote sensing to describe the bidirectional reflectance of a particulate surface. It relates the observed reflectance to the physical properties of the surface, such as single-scattering albedo, particle phase function, and macroscopic roughness.

## Core Concepts

The Hapke model is based on radiative transfer theory but provides an approximate analytical solution, making it computationally more tractable than full multiple scattering solutions. Key components and assumptions often include:

1.  **Isotropic Scatterers Approximation:** Initially, particles are often assumed to scatter isotropically, with corrections applied for anisotropic scattering.
2.  **Single-Scattering Albedo (SSA, $):** This represents the probability that a photon interacting with a particle is scattered rather than absorbed. It's a crucial parameter derived from reflectance spectra.
3.  **Particle Phase Function ((g)$ or (g)$):** Describes the angular distribution of light scattered by a single particle. $ is the phase angle. Various analytical forms are used, such as Henyey-Greenstein, Legendre polynomials, or more complex functions like Cornette-Shanks.
4.  **Opposition Effect:** The non-linear surge in brightness observed at small phase angles. This is often modeled with two components:
    *   **Shadow Hiding Opposition Effect (SHOE):** Caused by the hiding of shadows at zero phase angle. Modeled with parameters {s0}$ (amplitude) and $ (angular width).
    *   **Coherent Backscatter Opposition Effect (CBOE):** Arises from constructive interference of light waves traveling reciprocal paths. Modeled with parameters {c0}$ (amplitude) and $ (angular width).
5.  **Macroscopic Roughness ($\bar{\theta}$):** Accounts for large-scale surface undulations that affect local incidence and emission angles, and cast shadows. This is modeled as an average surface slope.
6.  **Chandrasekhar's H-functions:** These functions (e.g., (\mu, w)$) are solutions to auxiliary equations in radiative transfer and are used to describe the escape probability of photons from the surface. Approximations by Hapke are commonly used.

## General Form of the Hapke Equation

A general form of the Hapke equation for bidirectional reflectance (i, e, g)$ (where $ is incidence angle, $ is emission angle, $ is phase angle) can be expressed as:

2348 r(i, e, g) = K \frac{w}{4\pi} \frac{\mu_{0e}}{\mu_{0e} + \mu_e} \left[ (1 + B(g))p(g) + H(\mu_{0e}, w)H(\mu_e, w) - 1 \right] S(i, e, g, \bar{\theta}) 2348

Where:
-   $ is a normalization factor (often assumed to be 1).
-   $\mu_{0e}$ and $\mu_e$ are effective cosines of incidence and emission angles, often modified by local topography or particle properties. (In some formulations, these are just $\mu_0 = \cos(i)$ and $\mu = \cos(e)$ before roughness corrections).
-   (g)$ represents the opposition effect term (combining SHOE and CBOE).
-   (\mu, w)$ are the Chandrasekhar H-functions (or Hapke's approximations).
-   (i, e, g, \bar{\theta})$ is the macroscopic roughness correction factor.

The specific formulation can vary between different versions and extensions of the Hapke model (e.g., IMSA, AMSA).

## Applications

Hapke modeling is extensively used for:
-   Deriving physical properties of planetary regoliths (e.g., Moon, Mars, asteroids).
-   Correcting photometric effects in remote sensing data.
-   Understanding light scattering in particulate media.

## Further Reading

For detailed derivations and explanations, please consult the works listed in our [references.md](references.md), particularly Hapke (1993).
