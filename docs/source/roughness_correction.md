# Macroscopic Roughness Correction in Hapke Modeling

Macroscopic roughness refers to large-scale irregularities on a particulate surface, such as hills, valleys, craters, or even cm-scale roughness elements like rocks and clumps, that are larger than the individual particles but smaller than the area resolved by a single measurement. These features significantly influence the observed reflectance by:

1.  Altering local incidence and emission angles relative to the mean surface normal.
2.  Casting shadows, thereby reducing the illuminated and/or visible area.

Hapke (1984) introduced a widely used correction factor, (i, e, g, \bar{\theta})$, to account for these effects.

## The Roughness Parameter $\bar{\theta}$

The primary parameter describing macroscopic roughness in Hapke's model is $\bar{\theta}$, which represents the average slope of the surface facets relative to the mean horizontal plane. A surface with $\bar{\theta} = 0^\circ$ is perfectly smooth, while larger values indicate rougher surfaces.

## Conceptual Basis of the Correction

The correction involves statistically averaging the reflectance from a distribution of facets with varying orientations. Hapke's model considers how these tilted facets affect:
-   The amount of incident sunlight received.
-   The amount of scattered light directed towards the observer.
-   The probability that a facet is shadowed by adjacent topography, both from the illumination source and from the viewer's perspective.

## Implementation in `refmod.hapke.roughness`

The function `refmod.hapke.roughness.microscopic_roughness` (despite its name, it implements the *macroscopic* roughness correction as described by Hapke, 1984) calculates:
-   $: The overall roughness correction factor.
-   $\mu_0'$ (returned as `mu_0_s` or `mu_0_prime` in some contexts): The modified cosine of the incidence angle, averaged over the facet distribution.
-   $\mu'$ (returned as `mu_s` or `mu_prime`): The modified cosine of the emission angle, averaged over the facet distribution.

The derivation involves complex geometric considerations and statistical averaging, detailed in Hapke (1984). The key equations from this paper (e.g., Eqs. 46, 47, 48, 49, 50, 51) describe how these modified cosines and the $ factor are calculated based on the incidence angle $, emission angle $, phase angle $ (implicitly through the relative azimuth), and the mean slope angle $\bar{\theta}$.

## Impact on Reflectance

The roughness correction $ typically:
-   Reduces reflectance at large incidence or emission angles due to shadowing.
-   Can increase reflectance near opposition for rough surfaces, as shadows are minimized.
-   Affects the overall photometric behavior of the surface, often making it appear darker at oblique viewing geometries than a smooth surface with the same intrinsic properties.

## References

-   Hapke, B. (1984). Bidirectional reflectance spectroscopy: 3. Correction for macroscopic roughness. *Icarus*, 59(1), 41-59.
-   See also [references.md](references.md).
