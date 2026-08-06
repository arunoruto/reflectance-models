import jax
import jax.numpy as jnp

from ._core import (
    coef_a,
    coherent_backscatter,
    cos_angle,
    function_p,
    h_function,
    h_function_derivative,
    legendre_eval,
    normalize,
    roughness_correction,
    shadow_hiding,
    value_p,
)


def _refl_amsa_scalar(
    w: jax.Array,
    b_n: jax.Array,
    i: jax.Array,
    e: jax.Array,
    n: jax.Array,
    roughness: float,
    h_sh: float,
    b0_sh: float,
    h_cb: float,
    b0_cb: float,
    a_n: jax.Array | None = None,
) -> jax.Array:
    r"""Compute AMSA reflectance for a single pixel (scalar w, 3-vectors).

    Full anisotropic multiple scattering with shadow hiding and coherent
    backscatter. Handles a single pixel with scalar single-scattering albedo
    and 3-vector incidence, emission, and normal directions.

    Parameters
    ----------
    w : Array
        Single-scattering albedo (scalar).
    b_n : Array
        Legendre coefficients of the single-particle phase function, shape (N,).
    i : Array
        Incidence direction vector, shape (3,).
    e : Array
        Emission direction vector, shape (3,).
    n : Array
        Surface normal vector, shape (3,).
    roughness : float
        Surface roughness angle in radians.
    h_sh : float
        Shadow-hiding angular width parameter.
    b0_sh : float
        Shadow-hiding opposition amplitude.
    h_cb : float
        Coherent backscatter angular width parameter.
    b0_cb : float
        Coherent backscatter opposition amplitude.
    a_n : Array or None, optional
        Precomputed Legendre expansion of the Henyey-Greenstein phase function
        coefficients. Computed from ``b_n`` if None.

    Returns
    -------
    Array
        Reflectance value (scalar). NaN if mu0 <= 0 or mu <= 0.

    References
    ----------
    :cite:p:`Hapke-1984`
    :cite:p:`Hapke-2002`
    :cite:p:`Hapke-2012`
    """
    i = normalize(i)
    e = normalize(e)
    n = normalize(n)
    # Mask on the raw cosines too: the roughness correction is only defined
    # for 0 <= i, e <= 90 deg and produces spurious positive effective
    # cosines for back-facing geometry.
    invalid = (cos_angle(i, n) <= 0.0) | (cos_angle(e, n) <= 0.0)
    s, mu0, mu = roughness_correction(roughness, i, e, n)
    invalid = invalid | (mu <= 0.0) | (mu0 <= 0.0)
    cos_alpha = cos_angle(i, e)
    tan_alpha_2 = jnp.sqrt(1.0 - cos_alpha**2) / (1.0 + cos_alpha)
    p = legendre_eval(cos_alpha, b_n)
    b_sh = shadow_hiding(tan_alpha_2, h_sh, b0_sh)
    p_sh = p * b_sh
    h_mu0 = h_function(mu0, w)
    h_mu = h_function(mu, w)
    if a_n is None:
        a_n = coef_a(b_n.shape[0] - 1)
    p_mu0 = function_p(mu0, b_n, a_n)
    p_mu = function_p(mu, b_n, a_n)
    P = value_p(b_n, a_n)
    m = p_mu0 * (h_mu - 1.0) + p_mu * (h_mu0 - 1.0) + P * (h_mu0 - 1.0) * (h_mu - 1.0)
    b_cb = coherent_backscatter(tan_alpha_2, h_cb, b0_cb)
    albedo_indep = mu0 / (mu0 + mu) * s / (4.0 * jnp.pi) * b_cb
    refl = albedo_indep * w * (p_sh + m)
    return jnp.where(invalid, jnp.nan, refl)


def _refl_amsa_scalar_and_grad(
    w: jax.Array,
    b_n: jax.Array,
    i: jax.Array,
    e: jax.Array,
    n: jax.Array,
    roughness: float,
    h_sh: float,
    b0_sh: float,
    h_cb: float,
    b0_cb: float,
    a_n: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    r"""Compute AMSA reflectance and its analytical derivative dR/dw for a single pixel.

    Full anisotropic multiple scattering with shadow hiding and coherent
    backscatter. Returns both reflectance and derivative with respect to the
    single-scattering albedo w, useful for Levenberg-Marquardt optimization.

    Parameters
    ----------
    w : Array
        Single-scattering albedo (scalar).
    b_n : Array
        Legendre coefficients of the single-particle phase function, shape (N,).
    i : Array
        Incidence direction vector, shape (3,).
    e : Array
        Emission direction vector, shape (3,).
    n : Array
        Surface normal vector, shape (3,).
    roughness : float
        Surface roughness angle in radians.
    h_sh : float
        Shadow-hiding angular width parameter.
    b0_sh : float
        Shadow-hiding opposition amplitude.
    h_cb : float
        Coherent backscatter angular width parameter.
    b0_cb : float
        Coherent backscatter opposition amplitude.
    a_n : Array or None, optional
        Precomputed Legendre expansion of the Henyey-Greenstein phase function
        coefficients. Computed from ``b_n`` if None.

    Returns
    -------
    tuple[Array, Array]
        Reflectance value (scalar) and derivative dR/dw (scalar).
        Reflectance is NaN and derivative is 0.0 if mu0 <= 0 or mu <= 0.

    References
    ----------
    :cite:p:`Hapke-1984`
    :cite:p:`Hapke-2002`
    :cite:p:`Hapke-2012`
    """
    i = normalize(i)
    e = normalize(e)
    n = normalize(n)
    invalid = (cos_angle(i, n) <= 0.0) | (cos_angle(e, n) <= 0.0)
    s, mu0, mu = roughness_correction(roughness, i, e, n)
    invalid = invalid | (mu <= 0.0) | (mu0 <= 0.0)
    cos_alpha = cos_angle(i, e)
    tan_alpha_2 = jnp.sqrt(1.0 - cos_alpha**2) / (1.0 + cos_alpha)
    p = legendre_eval(cos_alpha, b_n)
    b_sh = shadow_hiding(tan_alpha_2, h_sh, b0_sh)
    p_sh = p * b_sh
    h_mu0 = h_function(mu0, w)
    h_mu = h_function(mu, w)
    dh_mu0 = h_function_derivative(mu0, w)
    dh_mu = h_function_derivative(mu, w)
    if a_n is None:
        a_n = coef_a(b_n.shape[0] - 1)
    p_mu0 = function_p(mu0, b_n, a_n)
    p_mu = function_p(mu, b_n, a_n)
    P = value_p(b_n, a_n)
    hm0_1 = h_mu0 - 1.0
    hm_1 = h_mu - 1.0
    m = p_mu0 * hm_1 + p_mu * hm0_1 + P * hm0_1 * hm_1
    dm_dw = p_mu0 * dh_mu + p_mu * dh_mu0 + P * (dh_mu * hm0_1 + dh_mu0 * hm_1)
    b_cb = coherent_backscatter(tan_alpha_2, h_cb, b0_cb)
    albedo_indep = mu0 / (mu0 + mu) * s / (4.0 * jnp.pi) * b_cb
    bracket = p_sh + m
    refl = albedo_indep * w * bracket
    d_refl = albedo_indep * (bracket + w * dm_dw)
    refl = jnp.where(invalid, jnp.nan, refl)
    d_refl = jnp.where(invalid, 0.0, d_refl)
    return refl, d_refl


#
# Precomputed w-independent state for fast LM iterations
#


def _precompute_amsa_scalar(
    b_n: jax.Array,
    i: jax.Array,
    e: jax.Array,
    n: jax.Array,
    roughness: float,
    h_sh: float,
    b0_sh: float,
    h_cb: float,
    b0_cb: float,
    a_n: jax.Array | None = None,
) -> dict:
    r"""Precompute all w-independent quantities for fast LM iteration.

    Computes roughness correction, Legendre expansion, single-particle phase
    function, shadow hiding, coherent backscatter, and the albedo-independent
    prefactor. The returned dict can be reused with
    ``_fast_refl_amsa_scalar`` and ``_fast_refl_amsa_scalar_and_grad``
    to avoid recomputing these quantities when only w varies.

    Parameters
    ----------
    b_n : Array
        Legendre coefficients of the single-particle phase function, shape (N,).
    i : Array
        Incidence direction vector, shape (3,).
    e : Array
        Emission direction vector, shape (3,).
    n : Array
        Surface normal vector, shape (3,).
    roughness : float
        Surface roughness angle in radians.
    h_sh : float
        Shadow-hiding angular width parameter.
    b0_sh : float
        Shadow-hiding opposition amplitude.
    h_cb : float
        Coherent backscatter angular width parameter.
    b0_cb : float
        Coherent backscatter opposition amplitude.
    a_n : Array or None, optional
        Precomputed Legendre expansion of the Henyey-Greenstein phase function
        coefficients. Computed from ``b_n`` if None.

    Returns
    -------
    dict
        Dictionary of precomputed values with keys ``albedo_indep``, ``p_sh``,
        ``p_mu0``, ``p_mu``, ``P``, ``mu0``, ``mu``, ``valid``.

    References
    ----------
    :cite:p:`Hapke-1984`
    :cite:p:`Hapke-2002`
    :cite:p:`Hapke-2012`
    """
    i = normalize(i)
    e = normalize(e)
    n = normalize(n)
    valid_raw = (cos_angle(i, n) > 0.0) & (cos_angle(e, n) > 0.0)
    s, mu0, mu = roughness_correction(roughness, i, e, n)
    cos_alpha = cos_angle(i, e)
    tan_alpha_2 = jnp.sqrt(1.0 - cos_alpha**2) / (1.0 + cos_alpha)
    p = legendre_eval(cos_alpha, b_n)
    b_sh = shadow_hiding(tan_alpha_2, h_sh, b0_sh)
    p_sh = p * b_sh
    if a_n is None:
        a_n = coef_a(b_n.shape[0] - 1)
    p_mu0 = function_p(mu0, b_n, a_n)
    p_mu = function_p(mu, b_n, a_n)
    P = value_p(b_n, a_n)
    b_cb = coherent_backscatter(tan_alpha_2, h_cb, b0_cb)
    albedo_indep = mu0 / (mu0 + mu) * s / (4.0 * jnp.pi) * b_cb
    return dict(
        albedo_indep=albedo_indep,
        p_sh=p_sh,
        p_mu0=p_mu0,
        p_mu=p_mu,
        P=P,
        mu0=mu0,
        mu=mu,
        valid=valid_raw & (mu > 0.0) & (mu0 > 0.0),
    )


def _fast_refl_amsa_scalar(
    w: jax.Array,
    pre: dict,
) -> jax.Array:
    r"""Compute AMSA reflectance using precomputed state.

    Only recomputes the H-functions and the multiple scattering term M.
    Uses precomputed w-independent quantities from
    ``_precompute_amsa_scalar`` for faster evaluation during
    Levenberg-Marquardt iterations.

    Parameters
    ----------
    w : Array
        Single-scattering albedo (scalar).
    pre : dict
        Precomputed state dict from ``_precompute_amsa_scalar``.

    Returns
    -------
    Array
        Reflectance value (scalar). NaN if not valid.

    References
    ----------
    :cite:p:`Hapke-2012`
    """
    h_mu0 = h_function(pre["mu0"], w)
    h_mu = h_function(pre["mu"], w)
    hm0_1 = h_mu0 - 1.0
    hm_1 = h_mu - 1.0
    m = pre["p_mu0"] * hm_1 + pre["p_mu"] * hm0_1 + pre["P"] * hm0_1 * hm_1
    refl = pre["albedo_indep"] * w * (pre["p_sh"] + m)
    return jnp.where(pre["valid"], refl, jnp.nan)


def _fast_refl_amsa_scalar_and_grad(
    w: jax.Array,
    pre: dict,
) -> tuple[jax.Array, jax.Array]:
    r"""Compute AMSA reflectance and analytical derivative dR/dw using precomputed state.

    Only recomputes the H-functions, their derivatives, and the multiple
    scattering term. Uses precomputed w-independent quantities from
    ``_precompute_amsa_scalar`` for faster evaluation during
    Levenberg-Marquardt iterations.

    Parameters
    ----------
    w : Array
        Single-scattering albedo (scalar).
    pre : dict
        Precomputed state dict from ``_precompute_amsa_scalar``.

    Returns
    -------
    tuple[Array, Array]
        Reflectance value (scalar) and derivative dR/dw (scalar).
        Reflectance is NaN and derivative is 0.0 if not valid.

    References
    ----------
    :cite:p:`Hapke-2012`
    """
    h_mu0 = h_function(pre["mu0"], w)
    h_mu = h_function(pre["mu"], w)
    dh_mu0 = h_function_derivative(pre["mu0"], w)
    dh_mu = h_function_derivative(pre["mu"], w)
    hm0_1 = h_mu0 - 1.0
    hm_1 = h_mu - 1.0
    m = pre["p_mu0"] * hm_1 + pre["p_mu"] * hm0_1 + pre["P"] * hm0_1 * hm_1
    dm_dw = (
        pre["p_mu0"] * dh_mu
        + pre["p_mu"] * dh_mu0
        + pre["P"] * (dh_mu * hm0_1 + dh_mu0 * hm_1)
    )
    bracket = pre["p_sh"] + m
    refl = pre["albedo_indep"] * w * bracket
    d_refl = pre["albedo_indep"] * (bracket + w * dm_dw)
    refl = jnp.where(pre["valid"], refl, jnp.nan)
    d_refl = jnp.where(pre["valid"], d_refl, 0.0)
    return refl, d_refl


_precompute_amsa_batched = jax.vmap(
    _precompute_amsa_scalar,
    in_axes=(None, 0, 0, 0, None, None, None, None, None, None),
)

_fast_refl_amsa_batched = jax.jit(
    jax.vmap(
        _fast_refl_amsa_scalar,
        in_axes=(0, 0),
    )
)

_fast_refl_amsa_and_grad_batched = jax.jit(
    jax.vmap(
        _fast_refl_amsa_scalar_and_grad,
        in_axes=(0, 0),
    )
)


def precompute_amsa(
    b_n: jax.Array,
    i: jax.Array,
    e: jax.Array,
    n: jax.Array,
    roughness: float = 0.0,
    h_sh: float = 0.0,
    b0_sh: float = 0.0,
    h_cb: float = 0.0,
    b0_cb: float = 0.0,
    a_n: jax.Array | None = None,
) -> dict:
    r"""Precompute w-independent quantities for batched (N, 3) inputs.

    ``jax.vmap``-wrapped version of ``_precompute_amsa_scalar`` across the
    pixel dimension. ``b_n`` is broadcast across all pixels.

    Parameters
    ----------
    b_n : Array
        Legendre coefficients of the single-particle phase function, shape (N,).
    i : Array
        Incidence direction vectors, shape (M, 3).
    e : Array
        Emission direction vectors, shape (M, 3).
    n : Array
        Surface normal vectors, shape (M, 3).
    roughness : float, optional
        Surface roughness angle in radians. Default is 0.0.
    h_sh : float, optional
        Shadow-hiding angular width parameter. Default is 0.0.
    b0_sh : float, optional
        Shadow-hiding opposition amplitude. Default is 0.0.
    h_cb : float, optional
        Coherent backscatter angular width parameter. Default is 0.0.
    b0_cb : float, optional
        Coherent backscatter opposition amplitude. Default is 0.0.

    Returns
    -------
    dict
        Dictionary of batched precomputed values with keys ``albedo_indep``,
        ``p_sh``, ``p_mu0``, ``p_mu``, ``P``, ``mu0``, ``mu``, ``valid``.
        Each value has an added leading batch dimension.

    References
    ----------
    :cite:p:`Hapke-1984`
    :cite:p:`Hapke-2002`
    :cite:p:`Hapke-2012`
    """
    return _precompute_amsa_batched(
        b_n, i, e, n, roughness, h_sh, b0_sh, h_cb, b0_cb, a_n
    )


#
# Batched (for standalone use, not in LM loop)
#

_amsa_batched = jax.vmap(
    _refl_amsa_scalar,
    in_axes=(0, None, 0, 0, 0, None, None, None, None, None, None),
)


def amsa(
    w: jax.Array,
    b_n: jax.Array,
    i: jax.Array,
    e: jax.Array,
    n: jax.Array,
    roughness: float = 0.0,
    h_sh: float = 0.0,
    b0_sh: float = 0.0,
    h_cb: float = 0.0,
    b0_cb: float = 0.0,
    a_n: jax.Array | None = None,
) -> jax.Array:
    r"""Public batched AMSA reflectance model.

    Full anisotropic multiple scattering with shadow hiding and coherent
    backscatter. Computes reflectance for a batch of pixels with scalar w,
    broadcast Legendre coefficients ``b_n``, and batched geometry vectors.

    Parameters
    ----------
    w : Array
        Single-scattering albedo (scalar, broadcast across pixels).
    b_n : Array
        Legendre coefficients of the single-particle phase function, shape (N,).
    i : Array
        Incidence direction vectors, shape (M, 3).
    e : Array
        Emission direction vectors, shape (M, 3).
    n : Array
        Surface normal vectors, shape (M, 3).
    roughness : float, optional
        Surface roughness angle in radians. Default is 0.0.
    h_sh : float, optional
        Shadow-hiding angular width parameter. Default is 0.0.
    b0_sh : float, optional
        Shadow-hiding opposition amplitude. Default is 0.0.
    h_cb : float, optional
        Coherent backscatter angular width parameter. Default is 0.0.
    b0_cb : float, optional
        Coherent backscatter opposition amplitude. Default is 0.0.
    a_n : Array or None, optional
        Precomputed Legendre expansion of the Henyey-Greenstein phase function
        coefficients. Computed from ``b_n`` if None.

    Returns
    -------
    Array
        Reflectance values, shape (M,). NaN where mu0 <= 0 or mu <= 0.

    References
    ----------
    :cite:p:`Hapke-1984`
    :cite:p:`Hapke-2002`
    :cite:p:`Hapke-2012`
    """
    return _amsa_batched(w, b_n, i, e, n, roughness, h_sh, b0_sh, h_cb, b0_cb, a_n)
