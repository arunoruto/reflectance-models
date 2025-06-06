# Legendre Polynomials in Hapke Phase Function Expansion

In Hapke modeling, particularly for describing anisotropic scattering from particles, the single-particle phase function (g)$ is often expanded as a series of Legendre polynomials. This provides a flexible way to represent complex scattering behaviors.

## The Legendre Expansion

The phase function (g)$ can be written as:

2348 p(g) = 1 + \sum_{n=1}^{N} b_n P_n(\cos g) 2348

Or, more generally, if the -bash^{th}$ order term is not fixed at 1:

2348 p(g) = \sum_{n=0}^{N} c_n P_n(\cos g) 2348
(where $ would typically be 1 if (g)$ is normalized such that its integral over \pi$ steradians, divided by \pi$, is 1).

Where:
-   $ is the phase angle.
-   (\cos g)$ is the Legendre polynomial of degree $.
-   $ (or $) are the expansion coefficients. These coefficients depend on the particle's size, shape, and composition. For example,  = 3 \langle \cos g \rangle$ is related to the asymmetry factor  = \langle \cos g \rangle$.
-   $ is the order of the expansion, typically chosen based on the complexity of the phase function and the available data.

## Hapke's Formulation (Hapke, 2002)

Hapke (2002) provides a detailed framework for using Legendre polynomials in the context of bidirectional reflectance, especially concerning the coherent backscatter opposition effect and anisotropic scattering.

The module `refmod.hapke.legendre` implements functions based on this work:

-   **`coef_b(b, c, n)`**: Calculates the coefficients $ for the phase function expansion. The calculation method can vary based on the input parameters $ and $ (which are related to specific phase function models, like the Double Henyey-Greenstein, though Hapke (2002) provides a more general context for $).
    -   For example, Hapke (2002, p. 530) describes  = c(2n+1)b^n$ for a certain type of scattering, and  = (2n+1)(-b)^n$ for another. (Note: The `refmod` implementation might have specific interpretations of , c$ here).

-   **`coef_a(n)`**: Calculates auxiliary coefficients $ (Hapke, 2002, Eq. 27) used in further calculations involving the H-functions and the overall reflectance when an anisotropic phase function is considered. These are given by  = -P_{n-1}(0) / (n+1)$ for  \ge 1$, and =0$. (The `refmod` implementation maps this to an array starting at =0$).

-   **`function_p(x, b_n, a_n)`**: Calculates an auxiliary function (x)$ (Hapke, 2002, Eqs. 23, 24), which is defined as (x) = 1 + \sum_{n=0}^{N} a_n b_n P_n(x)$. This function is part of the correction for anisotropic scattering in the Hapke model.

-   **`value_p(b_n, a_n)`**: Calculates a scalar value $ (Hapke, 2002, Eq. 25),  = 1 + \sum_{n=0}^{N} a_n^2 b_n$. This also contributes to the anisotropic scattering corrections.

## Significance

Using a Legendre expansion allows for:
-   A systematic way to represent arbitrary phase functions.
-   Easier integration of the phase function in multiple scattering calculations.
-   Relating scattering behavior to physical parameters through the coefficients $.

## References

-   Hapke, B. (2002). Bidirectional Reflectance Spectroscopy: 5. The Coherent Backscatter Opposition Effect and Anisotropic Scattering. *Icarus*, 157(2), 523–534.
-   See also [references.md](references.md).
