# Macroscopic Roughness Correction in Hapke Modeling

Macroscopic roughness refers to large-scale irregularities on a particulate surface that are significantly larger than the individual particles themselves but typically smaller than the area resolved by a single remote sensing measurement. Examples include hills, valleys, craters, or even centimeter-scale roughness elements like rocks and clumps. These features significantly influence the observed reflectance by:

1.  Altering local incidence ($i'$) and emission ($e'$) angles on surface facets relative to the mean surface normal.
2.  Casting shadows, thereby reducing the fraction of the surface that is both illuminated by the source and visible to the observer.

{cite:t}`Hapke-1984` introduced a widely adopted correction factor, $S(i, e, g, \bar{\theta})$, to account for these effects in bidirectional reflectance models.

## The Roughness Parameter $\bar{\theta}$

The primary parameter describing macroscopic roughness in this formulation of Hapke's model is $\bar{\theta}$. This parameter represents the **average slope angle** of the surface facets relative to the mean horizontal plane.

- A surface with $\bar{\theta} = 0^\circ$ is macroscopically smooth.
- Larger values of $\bar{\theta}$ indicate statistically rougher surfaces with steeper average slopes.

## Conceptual Basis of the Correction

The correction factor $S$ is derived by statistically averaging the reflectance from a distribution of facets, each with its own orientation (slope and azimuth) relative to the mean surface. Hapke's model specifically considers how these tilted facets affect:

- **Illumination:** The amount of incident sunlight received by a facet, considering its local tilt.
- **Viewing:** The amount of scattered light from a facet that is directed towards the observer.
- **Shadowing:** The probability that a facet is shadowed by adjacent topography, both from the illumination source (self-shadowing of the incident beam) and from the viewer's perspective (self-shadowing of the emergent beam).

## Implementation in `refmod.hapke.roughness`

The function `refmod.hapke.roughness.macroscopic_roughness` (if the name was `microscopic_roughness` as previously noted, it should ideally be `macroscopic_roughness` to match its function as per {cite:t}`Hapke-1984`) calculates several key quantities:

- **$S(i, e, g, \bar{\theta})$**: The overall roughness correction factor. This factor multiplies the reflectance calculated for a smooth surface to give the reflectance of the rough surface.
- **$\mu_0'$** (often returned as `mu0_s`, `mu0_prime`, or similar in code): The effective cosine of the incidence angle, averaged over the illuminated and visible portions of the facet distribution.
- **$\mu'$** (often returned as `mu_s`, `mu_prime`, or similar in code): The effective cosine of the emission angle, averaged over the illuminated and visible portions of the facet distribution.

The derivation involves complex geometric considerations and statistical averaging, which are detailed extensively in {cite:t}`Hapke-1984`. The key equations from this seminal paper (e.g., Eqs. 46, 47, 48, 49, 50, and 51 in the 1984 Icarus paper) describe how these modified (effective) cosines and the overall $S$ factor are calculated based on the macroscopic incidence angle $i$, emission angle $e$, relative azimuth (which, together with $i$ and $e$, defines the phase angle $g$), and the mean slope angle $\bar{\theta}$.

## Impact on Reflectance

The macroscopic roughness correction factor $S$ typically:

- **Reduces reflectance** at large incidence or emission angles. This is primarily due to increased shadowing effects – more of the surface is hidden from the light source or the viewer.
- Can **increase or decrease reflectance near opposition** ($g \approx 0^\circ$). For very rough surfaces, $S$ can contribute to the opposition surge as inter-particle shadows become hidden. However, the exact behavior depends on the interplay with other opposition effect mechanisms (like SHOE and CBOE).
- Affects the overall photometric behavior, generally making the surface appear **darker at oblique viewing geometries** compared to a macroscopically smooth surface with the same intrinsic single-scattering albedo and phase function.
- Modifies the apparent limb darkening or brightening of a planetary body.

Understanding and correctly applying this roughness correction is crucial for accurately interpreting remote sensing data and deriving meaningful physical parameters from planetary surfaces.
