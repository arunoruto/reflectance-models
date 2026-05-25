# The AMSA (Anisotropic Multiple Scattering Approximation) Model

The AMSA model, an acronym for **Anisotropic Multiple Scattering Approximation**, represents an advanced formulation of Hapke's theory designed to more accurately account for the effects of anisotropic single-particle scattering on the multiple-scattering term. The definitive form of this model, which also incorporates a sophisticated treatment of opposition effects, is detailed by {cite:t}`Hapke-2002`.

The functions `refmod.hapke.amsa.amsa` and `refmod.hapke.amsa.amsa_derivative` in this library implement this comprehensive and powerful model.

## AMSA Reflectance Equation

The final expression for the AMSA model, as given by {cite:t}`Hapke-2002` (Eq. 38), combines the single-scattering term, the anisotropic multiple-scattering term, and both major opposition effects. When combined with the macroscopic roughness correction, the full equation is:

$$
r(i, e, g) = \frac{w}{4\pi} \frac{\mu_{0e}}{\mu_{0e} + \mu_e} \left[ p(g) B_{SH}(g) + M(\mu_{0e}, \mu_e) \right] B_{CB}(g) \cdot S(i, e, g, \bar{\theta})
$$

The components are broken down as follows:

1.  **Single-Scattering Term ($p(g) B_{SH}(g)$)**

    - $p(g)$ is the single-particle phase function (e.g., a Legendre polynomial expansion).
    - $B_{SH}(g)$ is the **Shadow-Hiding Opposition Effect (SHOE)**, which multiplies _only_ the single-scattering term. It is given by (Eq. 28, 29):
      $$ B*{SH}(g) = 1 + \frac{B*{S0}}{1 + \frac{1}{h_S} \tan(g/2)} $$

2.  **Anisotropic Multiple-Scattering Term ($M(\mu_{0e}, \mu_e)$)**

    - This is the core improvement of AMSA, replacing the simpler $H(\mu_0)H(\mu)-1$ term from IMSA. It is defined by {cite:t}`Hapke-2002` (Eq. 17):
      $$ M(\mu_0, \mu) = P(\mu_0)[H(\mu) - 1] + P(\mu)[H(\mu_0) - 1] + \bar{P}[H(\mu) - 1][H(\mu_0) - 1] $$
    - The functions $P(\mu_0)$, $P(\mu)$, and $\bar{P}$ are averaged phase functions defined in terms of Legendre coefficients.
    - The H-functions used here should be the more accurate "level 2" approximations (Eq. 13 in the paper; `h_function_2` in `refmod`).

3.  **Coherent Backscatter Opposition Effect ($B_{CB}(g)$)**

    - This term multiplies the _entire_ reflectance (both single and multiple scattering components). It is defined by (Eq. 32):
      $$ B*{CB}(g) = 1 + B*{C0} \cdot B_C(g) $$
        where $B_C(g)$ is a complex function modeling the coherent backscatter peak.

4.  **Macroscopic Roughness ($S$)**
    - The standard roughness correction {cite}`Hapke-1984` is applied to the final reflectance. The effective angles $\mu_{0e}$ and $\mu_e$ are used as inputs to the main scattering function.

## Derivative Function

The presence of `refmod.hapke.amsa.amsa_derivative` is highly significant. This function calculates $\partial r / \partial w$ (the derivative of the AMSA reflectance with respect to the single-scattering albedo $w$). This capability is crucial for model inversion and sensitivity analysis, making the AMSA implementation in `refmod` particularly powerful for quantitative analysis of remote sensing data.

## Efficient Repeated Inversion

The public function `refmod.hapke.invert_amsa` is the simplest way to recover the single-scattering albedo from observed AMSA reflectance. It prepares the geometry-dependent terms and then solves the inverse problem in one call:

```python
from refmod.hapke import invert_amsa

w = invert_amsa(
    refl_obs,
    b_n,
    incidence_direction,
    emission_direction,
    surface_normal,
    roughness=roughness,
    h_sh=h_sh,
    b0_sh=b0_sh,
    h_cb=h_cb,
    b0_cb=b0_cb,
)
```

For repeated inversions with the same geometry and Hapke parameters, use the prepared-state API instead. The geometry-dependent terms are computed once by `prepare_amsa_inversion`, and each subsequent call to `invert_amsa_precomputed` only solves for the albedo:

```python
from refmod.hapke import prepare_amsa_inversion, invert_amsa_precomputed

state = prepare_amsa_inversion(
    b_n,
    incidence_direction,
    emission_direction,
    surface_normal,
    roughness=roughness,
    h_sh=h_sh,
    b0_sh=b0_sh,
    h_cb=h_cb,
    b0_cb=b0_cb,
)

w_1 = invert_amsa_precomputed(refl_obs_1, state)
w_2 = invert_amsa_precomputed(refl_obs_2, state)
```

This is useful when illumination, viewing geometry, surface normals, roughness, and phase-function parameters are fixed, but the observed reflectance changes. Examples include repeated inversion over multiple scenes, multiple wavelengths with shared geometry, or iterative workflows that solve many reflectance arrays against one terrain/geometry setup.

The convenience API remains the recommended choice for one-off inversions. The prepared-state API is intended for throughput-sensitive workflows where the one-time preparation cost is amortized across many inversions.

### Performance Notes

On the Hopper AMSA test image (1,332,870 pixels), the prepared-state API substantially reduces repeated inversion time after the one-time preparation step:

| Backend | `invert_amsa` steady-state | `prepare_amsa_inversion` once | `invert_amsa_precomputed` steady-state |
|---|---:|---:|---:|
| CPU | 3.26 s | 0.76 s | 2.45 s |
| GPU | 0.41 s | 0.31 s | 0.11 s |

Exact timings depend on hardware, JAX/XLA versions, array shapes, and whether the persistent JAX compilation cache is warm.

### JAX Compilation Cache

JAX does not use Python `__pycache__` directories for compiled XLA programs. For repeated runs, enable the persistent JAX compilation cache before the first compilation. In this repository, the `devenv` shell configures it with environment variables:

```sh
JAX_COMPILATION_CACHE_DIR="$DEVENV_STATE/jax-cache"
JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=all
```

Run Python commands through `devenv shell -- uv run ...` so these environment variables are present.
