import jax
import jax.numpy as jnp

from ._core import (
    cos_angle,
    double_henyey_greenstein,
    normalize,
    roughness_correction,
    shadow_hiding,
)


def _modified_h(x: jax.Array, w: jax.Array) -> jax.Array:
    gamma = jnp.sqrt(1.0 - w)
    return (1.0 + 2.0 * x) / (1.0 + 2.0 * x * gamma)


def _refl_imsa_modified_h_scalar(
    w: jax.Array,
    b: float,
    c: float,
    i: jax.Array,
    e: jax.Array,
    n: jax.Array,
    roughness: float,
    h_sh: float,
    b0_sh: float,
) -> jax.Array:
    r"""MATLAB-compatible IMSA modified-H reflectance for one pixel."""
    i = normalize(i)
    e = normalize(e)
    n = normalize(n)
    # Raw-cosine mask: the roughness correction is undefined for
    # back-facing geometry and can yield spurious positive cosines.
    invalid = (cos_angle(i, n) <= 0.0) | (cos_angle(e, n) <= 0.0)
    s, mu0, mu = roughness_correction(roughness, i, e, n)
    invalid = invalid | (mu <= 0.0) | (mu0 <= 0.0)
    cos_alpha = cos_angle(i, e)
    sin_alpha = jnp.sqrt(jnp.maximum(1.0 - cos_alpha**2, 0.0))
    tan_alpha_2 = sin_alpha / (1.0 + cos_alpha)
    phase = double_henyey_greenstein(cos_alpha, b, c)
    b_sh = shadow_hiding(tan_alpha_2, h_sh, b0_sh)
    h_mu0 = _modified_h(mu0, w)
    h_mu = _modified_h(mu, w)
    prefactor = mu0 / (mu0 + mu) * s / (4.0 * jnp.pi)
    refl = w * (b_sh * phase + h_mu0 * h_mu - 1.0) * prefactor
    return jnp.where(invalid, jnp.nan, refl)


_imsa_modified_h_batched = jax.vmap(
    _refl_imsa_modified_h_scalar,
    in_axes=(0, None, None, 0, 0, 0, None, None, None),
)


def imsa_modified_h(
    w: jax.Array,
    b: float,
    c: float,
    i: jax.Array,
    e: jax.Array,
    n: jax.Array,
    roughness: float = 0.0,
    h_sh: float = 0.0,
    b0_sh: float = 0.0,
) -> jax.Array:
    r"""Batched IMSA reflectance using the Hapke (1981) H-function.

    Combines a Double Henyey-Greenstein phase function, the shadow-hiding
    opposition effect, the macroscopic roughness correction, and the Hapke
    (1981) H-function approximation
    :math:`H(x) \approx (1 + 2x) / (1 + 2x\gamma)`.

    Notes
    -----
    Despite this function's name, its counterpart in the reference MATLAB
    toolbox is ``hapke_imsa.m`` — **not** ``hapke_imsa_modifiedH.m``. There,
    "modified H" denotes Hapke's *2002 modification*, which is the more
    accurate closed form
    :math:`H(x) = [1 - w x (r_0 + \frac{1 - 2 r_0 x}{2}\ln\frac{1+x}{x})]^{-1}`
    — the same H used by :func:`~refmod.hapke.amsa` and
    :func:`~refmod.hapke.imsa`. That exact-H variant of IMSA has no port in
    this package.

    Note also that the reference ``hapke_imsa.m`` historically applied a
    spurious second factor of :math:`1/(4\pi)`; this implementation does not.
    That defect is fixed upstream as of the toolbox's 2.0.0 release, so the
    two now agree, and the ``imsa_approx_h`` fixture pins it.

    Porting the exact-H variant is one decision away, not one function call
    away. Swapping :func:`~refmod.hapke._core.h_function` in here reproduces
    the ``imsa_exact_h`` fixture to 2e-16 at five of its six samples -- and
    fails the sixth, ``i = e = 0``, by 1.8e-3 relative, which is five orders
    above the fixture's 1e-8 tolerance.

    The cause is not the H-function. ``hapke_imsa_modifiedH.m`` uses a
    *different* roughness formulation from every other MATLAB model: it omits
    the ``cos(i) == 1 | cos(e) == 1`` guard that
    :func:`~refmod.hapke._core.roughness_correction` reproduces, so at exactly
    normal incidence or emission it returns Hapke's limit
    :math:`\mu_{0e} \to \chi(\bar\theta)\cos i` where this package returns the
    uncorrected cosine. At :math:`\bar\theta = 8^\circ` that factor
    :math:`\chi` is 0.9703, which is the whole discrepancy.

    So the blocker is agreeing which limit is right, not writing the model.
    See the toolbox's ``docs/known-issues.md`` entry 10 -- neither side has
    taken that decision yet, and taking it moves published numbers on both.
    """
    return _imsa_modified_h_batched(w, b, c, i, e, n, roughness, h_sh, b0_sh)
