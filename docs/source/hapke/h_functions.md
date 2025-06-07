# Chandrasekhar's H-Functions

Chandrasekhar's H-functions are a set of special functions that arise in the theory of radiative transfer, particularly in problems involving semi-infinite, plane-parallel atmospheres or scattering media. They were introduced by Subrahmanyan Chandrasekhar in his seminal work on radiative transfer {cite}`Chandrasekhar-2013`.

## Definition and Significance

The H-function, typically denoted as $H(\mu, w)$, depends on two main parameters:

- $\mu$: The cosine of an angle (often the angle of emergence or incidence of radiation with respect to the surface normal). It ranges from 0 to 1.
- $w$: The single-scattering albedo of the medium, representing the probability of scattering per interaction. It also ranges from 0 to 1 (where $w=1$ implies conservative scattering, i.e., no absorption).

The H-function is defined as the solution to the following non-linear integral equation for isotropic scattering:

$$
H(\mu, w) = 1 + \mu H(\mu, w) \int_0^1 \frac{\Psi_0(\mu') H(\mu', w)}{\mu + \mu'} d\mu'
$$

where $\Psi_0(\mu') = w/2$. An alternative exact expression, often used as a starting point for approximations or iterative solutions, is:

$$
\frac{1}{H(\mu, w)} = 1 - \frac{w \mu}{2} \int_0^1 \frac{H(\mu', w)}{\mu + \mu'} d\mu'
$$

**Physical Interpretation:**
The H-function $H(\mu, w)$ is fundamentally related to the probability that a photon, after undergoing multiple scattering events within a semi-infinite medium, will escape from the surface in a direction whose cosine with the normal is $\mu$. It quantifies the angular distribution of diffusely reflected or transmitted radiation.

## Properties

Some key properties of H-functions include:

- $H(\mu, w)$ is a monotonically increasing function of $\mu$ for a fixed $w$.
- $H(\mu, w)$ is a monotonically increasing function of $w$ for a fixed $\mu$.
- For conservative scattering ($w=1$), $H(\mu, 1)$ can be significantly larger than for $w < 1$.
- $H(0, w) = 1$.

Moments of the H-function are also important:

$$
\alpha_n = \int_0^1 \mu^n H(\mu, w) d\mu
$$

For example, the zeroth moment $\alpha_0$ is related to the albedo of the semi-infinite medium. For isotropic scattering and $w < 1$:

$$
\alpha_0 = \int_0^1 H(\mu, w) d\mu = \frac{2}{w} (1 - \sqrt{1-w})
$$

## Approximations and Usage in Hapke's Model

In Hapke's bidirectional reflectance model {cite}`Hapke-2012`, H-functions (or their approximations) play a crucial role in describing the multiple scattering component of light within the particulate surface. Specifically, terms like $H(\mu_{0e}, w)H(\mu_e, w)$ appear in the equation to account for the probability of incident photons scattering multiple times and then emerging towards the observer.

Due to the complexity of solving the integral equation for $H(\mu, w)$ directly, Hapke and others have developed accurate analytical approximations.

### Hapke's Approximation (Level 1)

A common and simple approximation by Hapke for isotropic scatterers is {cite}`Hapke-1984,Hapke-2012`:

$$
H_1(x, w) \approx \frac{1 + 2x}{1 + 2x\sqrt{1-w}}
$$

where $x$ can be $\mu$ or $\mu_0$. This is often referred to as a "level 1" approximation.

### Iterative Approximation (Level 2)

A more accurate "level 2" approximation (implemented in this library as `h_function_2`), uses a slightly different approach or further refinement based on the integral form. The resulting approximation is:
Let $\gamma = \sqrt{1-w}$ and $r_0 = (1-\gamma)/(1+\gamma)$. Then,

$$
H_2(x, w) = \left[ 1 - w x \left( r_0 + \frac{1 - 2 r_0 x}{2} \ln\left(1 + \frac{1}{x}\right) \right) \right]^{-1}
$$

This $H_2(x, w)$ approximation is generally more accurate than the simpler $H_1(x, w)$ across a wider range of parameters.

## Computation

While analytical approximations like $H_1$ and $H_2$ are common, H-functions can also be computed numerically using iterative methods directly on their defining integral equations or by employing sophisticated numerical schemes. The choice of method depends on the required accuracy and computational efficiency.
