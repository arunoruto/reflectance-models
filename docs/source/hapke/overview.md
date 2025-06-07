# Overview of Hapke's Bidirectional Reflectance Theory

Bruce Hapke's model is a widely used analytical model in planetary science and remote sensing to describe the bidirectional reflectance of a particulate surface {cite}`Hapke-2012`. It relates the observed reflectance to the physical properties of the surface, such as single-scattering albedo, particle phase function, and macroscopic roughness.

## Core Concepts

The Hapke model, comprehensively detailed in his book {cite}`Hapke-2012`, is based on radiative transfer theory but provides an approximate analytical solution, making it computationally more tractable than full multiple scattering solutions. Key components and assumptions often include:

1.  **Isotropic Scatterers Approximation:** Initially, particles are often assumed to scatter isotropically, with corrections applied for anisotropic scattering.
2.  **Single-Scattering Albedo (SSA, $w$):** This represents the probability that a photon interacting with a particle is scattered rather than absorbed. It's a crucial parameter derived from reflectance spectra. The symbol $w$ is commonly used for SSA.
3.  **Particle Phase Function ($P(g)$):** Describes the angular distribution of light scattered by a single particle. Here, $g$ is the phase angle (the angle between the incident and scattered light directions). Various analytical forms are used, such as the Henyey-Greenstein function, expansions in Legendre polynomials, or more complex functions like the Cornette-Shanks function {cite}`Cornette-1992`.
4.  **Opposition Effect:** The non-linear surge in brightness observed at small phase angles (when the illumination source is nearly directly behind the observer). This is often modeled with two components, as discussed in later developments of the theory (e.g., {cite}`Hapke-2002`):
    - **Shadow Hiding Opposition Effect (SHOE):** Caused by the hiding of shadows between particles at zero phase angle. Modeled with parameters such as $B_{S0}$ (amplitude) and $h_S$ (angular width).
    - **Coherent Backscatter Opposition Effect (CBOE):** Arises from constructive interference of light waves traveling reciprocal paths within the scattering medium. Modeled with parameters such as $B_{C0}$ (amplitude) and $h_C$ (angular width).
5.  **Macroscopic Roughness ($\bar{\theta}$):** Accounts for large-scale surface undulations that affect local incidence and emission angles, and cast shadows. This is often modeled as an average surface slope angle $\bar{\theta}$, with corrections detailed by {cite:t}`Hapke-1984`.
6.  **Chandrasekhar's H-functions:** These functions (e.g., $H(\mu, w)$) are solutions to auxiliary equations in radiative transfer theory and are used to describe the escape probability of photons from the surface after multiple scattering events. Approximations developed by Hapke are commonly used for computational efficiency {cite}`Hapke-2012`.

## General Form of the Hapke Equation

A general form of the Hapke equation for bidirectional reflectance $r(i, e, g)$, where $i$ is the incidence angle, $e$ is the emission angle, and $g$ is the phase angle, can be expressed as (see, e.g., {cite}`Hapke-2002,Hapke-2012`):

$$
r(i, e, g) = \frac{w}{4\pi} \frac{\mu_{0e}}{\mu_{0e} + \mu_e} \left[ (1 + B(g))P(g) + H(\mu_{0e}, w)H(\mu_e, w) - 1 \right] S(i, e, g, \bar{\theta})
$$

Where:

- $w$ is the single-scattering albedo.
- $\mu_{0e}$ and $\mu_e$ are effective cosines of the incidence and emission angles, respectively. These are often related to the direct cosines $\mu_0 = \cos(i)$ and $\mu = \cos(e)$ but can be modified by particle properties or local topography (if not handled by $S$).
- $P(g)$ is the single particle phase function.
- $B(g)$ represents the opposition effect term, often a sum of SHOE and CBOE contributions: $B(g) = B_S(g) + B_C(g)$.
- $H(\mu, w)$ are the Chandrasekhar H-functions (or Hapke's approximations).
- $S(i, e, g, \bar{\theta})$ is the macroscopic roughness correction factor, a key contribution outlined in {cite}`Hapke-1984`.

The specific formulation and the exact form of these terms can vary between different versions and extensions of the Hapke model (e.g., IMSA - Isotropic Multiple Scattering Approximation, AMSA - Anisotropic Multiple Scattering Approximation).

## Applications

Hapke modeling is extensively used for:

- Deriving physical properties of planetary regoliths (e.g., on the Moon, Mars, asteroids, and other Solar System bodies) {cite}`Hapke-2012`.
- Correcting photometric effects in remote sensing data to standardize observations made under different viewing and illumination geometries.
- Understanding light scattering processes in various particulate media.
