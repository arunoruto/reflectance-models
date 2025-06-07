# Legendre Polynomials in Hapke Phase Function Expansion

In Hapke modeling, particularly for describing anisotropic scattering from particles, the single-particle phase function $P(g)$ is often expanded as a series of Legendre polynomials. This provides a flexible way to represent complex scattering behaviors that cannot be adequately described by simpler analytical functions like the Henyey-Greenstein function.

## The Legendre Expansion

The phase function $P(g)$ can be written as:

$$
P(g) = 1 + \sum_{n=1}^{N} b_n P_n(\cos g)
$$

This form is common when the phase function is normalized such that its average over $4\pi$ steradians is 1, and $P_0(\cos g) = 1$ is the isotropic term. The $b_n$ coefficients then describe the anisotropic deviations.

Alternatively, a more general expansion (often used when the normalization constant is absorbed into the coefficients) is:

$$
P(g) = \sum_{n=0}^{N} c_n P_n(\cos g)
$$

Here, $c_0$ would typically be 1 if $P(g)$ is normalized such that its integral over $4\pi$ steradians, divided by $4\pi$, is 1 (i.e., $\frac{1}{4\pi} \int_{4\pi} P(g) d\Omega = 1$).

Where:

- $g$ is the phase angle.
- $P_n(\cos g)$ is the Legendre polynomial of degree $n$.
- $b_n$ (or $c_n$) are the expansion coefficients. These coefficients depend on the particle's size, shape, and composition. For instance, $c_1 = 3g_{asym}$ where $g_{asym} = \langle \cos g \rangle$ is the asymmetry factor. If using the first form, $b_1$ is related to $3g_{asym}$.
- $N$ is the order of the expansion, typically chosen based on the complexity of the phase function and the available data to fit the coefficients.

## Hapke's Formulation for Anisotropic Scattering

{cite:t}`Hapke-2002` provides a detailed framework for incorporating anisotropically scattering particles into his bidirectional reflectance model, making extensive use of Legendre polynomial expansions for the phase function. This is particularly important for accurately modeling the multiple scattering contributions.

The module `refmod.hapke.legendre` implements several functions based on this work:

- **`coef_b(b, c, n)`**: This function calculates the coefficients $b_n$ (or $c_n$ depending on context and implementation details) for the phase function expansion. The exact calculation method can depend on underlying physical models or assumptions for particle scattering.
  For example, {cite:t}`Hapke-2002` (p. 530, in the context of CBOE parameters) discusses coefficients like $c_n = c(2n+1)b^n$ for one type of exponential phase function and $c_n = (2n+1)(-b)^n$ for another (where $b$ and $c$ in Hapke's paper are specific model parameters, not to be confused with the general expansion coefficients $b_n$ or $c_n$). The `refmod` implementation's arguments `b` and `c` should be understood in the specific context of how they are used to derive the Legendre series coefficients.

- **`coef_a(n)`**: Calculates auxiliary coefficients $a_n$, as defined in {cite:t}`Hapke-2002` (Eq. 27), which are used in modifying the H-functions for anisotropic scattering. These are given by $a_n = -P_{n-1}(0) / (n+1)$ for $n \ge 1$, and $a_0 = 0$.
  _Note: $P_{odd}(0)=0$ and $P_{even}(0) = (-1)^{k} \frac{(2k-1)!!}{(2k)!!}$ where $n-1=2k$. So $a_n$ is non-zero only for even $n$.\_
  The `refmod` implementation maps these to an array, likely starting the index at $n=0$.

- **`function_p(x, b_n, a_n)`**: Calculates an auxiliary function $\chi(x)$ (referred to as $\Pi(x)$ or similar in some contexts, see {cite:t}`Hapke-2002`, Eqs. 23, 24), which is defined as $\chi(x) = 1 + \sum_{n=1}^{N} a_n b_n P_n(x)$ (assuming $a_0=0$). This function appears in the expressions for the anisotropic H-functions.

- **`value_p(b_n, a_n)`**: Calculates a scalar value $\chi_0$ (see {cite:t}`Hapke-2002`, Eq. 25), defined as $\chi_0 = 1 + \sum_{n=1}^{N} a_n^2 b_n$. This value also contributes to the corrections for anisotropic scattering within the Hapke model.

## Significance

Using a Legendre polynomial expansion for the particle phase function offers several advantages:

- **Flexibility:** It provides a systematic and mathematically convenient way to represent arbitrarily complex phase functions.
- **Analytical Tractability:** Legendre polynomials have useful orthogonality properties, which can simplify the integration of the phase function in radiative transfer equations, particularly in multiple scattering calculations.
- **Physical Insight:** The expansion coefficients ($b_n$ or $c_n$) can sometimes be related to physical parameters of the scattering particles or can be derived from theoretical scattering models (e.g., Mie theory).

This approach allows for a more accurate representation of real-world particulate surfaces compared to models restricted to simpler, fixed-form phase functions.
