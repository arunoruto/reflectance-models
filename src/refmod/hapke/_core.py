import warnings

import jax
import jax.numpy as jnp
import numpy as np

from refmod.config import EPS

DHG_TRUNCATION_WARN_THRESHOLD = 1e-4
"""Truncation-error bound above which :func:`dhg_legendre_coefficients` warns."""


def normalize(v: jax.Array) -> jax.Array:
    r"""Normalize a vector to unit length using the L2 norm.

    Parameters
    ----------
    v : jax.Array
        Input vector (or batch of vectors).

    Returns
    -------
    jax.Array
        Normalized vector(s) with unit L2 norm.
    """
    norm = jnp.sqrt(jnp.sum(v**2, axis=-1, keepdims=True))
    norm = jnp.maximum(norm, 1e-12)
    return v / norm


def cos_angle(a: jax.Array, b: jax.Array) -> jax.Array:
    r"""Compute the cosine of the angle between two vectors.

    The result is clamped to :math:`[-1, 1]` for numerical stability.

    Parameters
    ----------
    a : jax.Array
        First vector.
    b : jax.Array
        Second vector.

    Returns
    -------
    jax.Array
        Dot product of *a* and *b*, clamped to :math:`[-1, 1]`.
    """
    return jnp.clip(jnp.dot(a, b), -1.0, 1.0)


def h_function(x: jax.Array, w: jax.Array) -> jax.Array:
    r"""Hapke isotropic multiple-scattering H-function.

    Computes the Ambartsumian–Chandrasekhar H-function using Hapke's
    level-2 approximation (Hapke 2002, Eq. 13):

    .. math::

        H(x, w) = \frac{1}{1 - w x \left[r_0 + \frac{1 - 2 r_0 x}{2}
        \ln\frac{1 + x}{x}\right]}

    where :math:`\gamma = \sqrt{1 - w}` and
    :math:`r_0 = (1 - \gamma) / (1 + \gamma)`.

    Parameters
    ----------
    x : jax.Array
        Direction cosine :math:`\mu` or :math:`\mu_0`.
    w : jax.Array
        Single-scattering albedo.

    Returns
    -------
    jax.Array
        Value of the H-function :math:`H(x, w)`.

    References
    ----------
    :cite:p:`Hapke-2002`
    """
    gamma = jnp.sqrt(1.0 - w)
    r0 = (1.0 - gamma) / (1.0 + gamma)
    h_inv = 1.0 - w * x * (r0 + (1.0 - 2.0 * r0 * x) / 2.0 * jnp.log((1.0 + x) / x))
    return 1.0 / h_inv


def h_function_derivative(x: jax.Array, w: jax.Array) -> jax.Array:
    r"""Derivative of the H-function with respect to single-scattering albedo.

    Computes :math:`\partial H(x, w) / \partial w` of Hapke's level-2
    approximation (see :func:`h_function`).

    Note: kept alongside JAX autodiff on purpose — it matches the MATLAB
    ``hapke_amsa.m`` derivative exactly and is marginally faster than
    ``jax.jvp`` of the forward model.

    Parameters
    ----------
    x : jax.Array
        Direction cosine :math:`\mu` or :math:`\mu_0`.
    w : jax.Array
        Single-scattering albedo.

    Returns
    -------
    jax.Array
        Derivative :math:`\partial H / \partial w`.

    References
    ----------
    :cite:p:`Hapke-2002`
    """
    gamma = jnp.sqrt(1.0 - w)
    r0 = (1.0 - gamma) / (1.0 + gamma)
    x_log = jnp.log((1.0 + x) / x)
    dr0_dw = 1.0 / (gamma * (1.0 + gamma) ** 2)
    h = h_function(x, w)
    return (
        h**2
        * x
        * (r0 + (1.0 - 2.0 * r0 * x) / 2.0 * x_log + w * dr0_dw * (1.0 - x * x_log))
    )


def coef_a(n: int = 15) -> jax.Array:
    r"""Legendre expansion coefficients :math:`a_n` for the Hapke phase function.

    Computes the coefficients defined by Hapke (2002, Eq. 27):

    .. math::

        a_n = \begin{cases}
            0, & n = 0, 2, 4, \ldots \\
            -\frac{1}{2}, & n = 1 \\
            \frac{2 - n}{n + 1} a_{n-2}, & n = 3, 5, 7, \ldots
        \end{cases}

    Parameters
    ----------
    n : int, optional
        Number of coefficients to compute (default 15). Returns *n+1* values
        indexed 0 through *n*.

    Returns
    -------
    jax.Array
        Array of :math:`a_n` coefficients of length *n+1*.

    References
    ----------
    :cite:p:`Hapke-2002`
    """
    a_n = jnp.zeros(n + 1)
    a_n = a_n.at[1].set(-0.5)
    for i in range(3, n + 1, 2):
        a_n = a_n.at[i].set((2 - i) / (i + 1) * a_n[i - 2])
    return a_n


def dhg_truncation_error(b: float, c: float, n: int) -> float:
    r"""Upper bound on the truncation error of the DHG Legendre expansion.

    The DHG phase function has the exact expansion
    :math:`p(x) = \sum_k b_k P_k(x)` with
    :math:`|b_k| \leq \max(1, |c|)\,(2k+1)\,|b|^k` and :math:`|P_k(x)| \leq 1`,
    so the absolute error of truncating after order *n* is bounded by the
    tail sum

    .. math::

        \epsilon_n \leq \max(1, |c|) \sum_{k=n+1}^{\infty} (2k+1)\,|b|^k .

    Parameters
    ----------
    b : float
        Asymmetry parameter (:math:`|b| < 1`).
    c : float
        Backscatter fraction.
    n : int
        Truncation order.

    Returns
    -------
    float
        Upper bound on the max absolute error of the reconstructed phase
        function. ``inf`` if :math:`|b| \geq 1`.
    """
    t = abs(b)
    if t >= 1.0:
        return float("inf")
    if t == 0.0:
        return 0.0
    m = n + 1
    s1 = t**m / (1.0 - t)
    s2 = t**m * (m - (m - 1) * t) / (1.0 - t) ** 2
    return max(1.0, abs(c)) * (2.0 * s2 + s1)


def recommended_dhg_order(
    b: float,
    c: float,
    tol: float = DHG_TRUNCATION_WARN_THRESHOLD,
    max_order: int = 300,
) -> int:
    r"""Smallest Legendre order whose truncation-error bound is below *tol*.

    See :func:`dhg_truncation_error` for the bound.

    Parameters
    ----------
    b : float
        Asymmetry parameter.
    c : float
        Backscatter fraction.
    tol : float, optional
        Target error bound.
    max_order : int, optional
        Upper limit for the search (returned if no order satisfies *tol*).

    Returns
    -------
    int
        Recommended number of Legendre orders.
    """
    for k in range(1, max_order + 1):
        if dhg_truncation_error(b, c, k) < tol:
            return k
    return max_order


def dhg_legendre_coefficients(b: float, c: float, n: int = 15) -> jax.Array:
    r"""Legendre expansion coefficients for the Double Henyey–Greenstein phase function.

    Computes the coefficients :math:`b_n`:

    .. math::

        b_n = \begin{cases}
            (2n + 1) b^n, & n \text{ even} \\
            c (2n + 1) b^n, & n \text{ odd}
        \end{cases}

    Parameters
    ----------
    b : float
        Asymmetry parameter.
    c : float
        Backscatter fraction.
    n : int, optional
        Number of coefficients to compute (default 15). Returns *n+1* values.

    Returns
    -------
    jax.Array
        Array of Legendre coefficients :math:`b_n` of length *n+1*.

    Notes
    -----
    When *b* and *c* are plain scalars, the truncation error of the series
    is checked against ``DHG_TRUNCATION_WARN_THRESHOLD`` and a warning with
    a recommended order is emitted if the reconstruction of the phase
    function would be too inaccurate (relevant for strongly peaked phase
    functions, roughly :math:`|b| \gtrsim 0.4` at the default order).

    References
    ----------
    :cite:p:`Henyey-1941`
    """
    if isinstance(b, (int, float, np.floating)) and isinstance(
        c, (int, float, np.floating)
    ):
        err = dhg_truncation_error(float(b), float(c), n)
        if err > DHG_TRUNCATION_WARN_THRESHOLD:
            warnings.warn(
                f"Truncating the DHG Legendre expansion at order {n} for "
                f"b={float(b):g}, c={float(c):g} leaves a phase-function "
                f"error bound of {err:.2e}. Consider n="
                f"{recommended_dhg_order(float(b), float(c))} for an error "
                f"below {DHG_TRUNCATION_WARN_THRESHOLD:g}.",
                stacklevel=2,
            )
    range_n = jnp.arange(n + 1, dtype=jnp.float64)
    b_n = (2.0 * range_n + 1.0) * jnp.power(b, range_n)
    b_n = b_n.at[1::2].multiply(c)
    return b_n


def cs_legendre_coefficients(xi: float, n: int = 15) -> jax.Array:
    r"""Legendre expansion coefficients for the Cornette–Shanks phase function.

    .. deprecated:: 1.1
        Unvalidated and likely misaligned: the returned coefficients start
        at order 1 (not 0), so they are shifted by one order relative to
        what :func:`legendre_eval` and :func:`function_p` expect. Use
        :func:`refmod.hapke.cornette.cornette_legendre_coefficients` (the
        MATLAB-derived variant) if Cornette support is needed. Kept for
        reference until the Cornette models are validated.

    Parameters
    ----------
    xi : float
        Asymmetry parameter :math:`\xi`.
    n : int, optional
        Number of coefficients to compute (default 15). Returns *n+1* values.

    Returns
    -------
    jax.Array
        Array of Legendre coefficients :math:`b_n` of length *n+1*.

    References
    ----------
    :cite:p:`Cornette-1992`
    """
    range_n = jnp.arange(n + 1, dtype=jnp.float64) + 1.0
    b_n = (2.0 * range_n + 1.0) * jnp.power(-xi, range_n)
    return b_n


def double_henyey_greenstein(cos_g: jax.Array, b: float, c: float) -> jax.Array:
    r"""Double Henyey–Greenstein (DHG) phase function.

    .. math::

        P(\cos g) = \frac{1 + c}{2}
        \frac{1 - b^2}{(1 - 2b\cos g + b^2)^{3/2}}
        + \frac{1 - c}{2}
        \frac{1 - b^2}{(1 + 2b\cos g + b^2)^{3/2}}

    Parameters
    ----------
    cos_g : jax.Array
        Cosine of the phase angle :math:`\cos g`.
    b : float
        Asymmetry parameter.
    c : float
        Backscatter fraction.

    Returns
    -------
    jax.Array
        Phase function value(s) at the given angle(s).

    References
    ----------
    :cite:p:`Henyey-1941`
    """
    return (1.0 + c) / 2.0 * (1.0 - b**2) / (1.0 - 2.0 * b * cos_g + b**2) ** 1.5 + (
        1.0 - c
    ) / 2.0 * (1.0 - b**2) / (1.0 + 2.0 * b * cos_g + b**2) ** 1.5


def cornette_shanks(cos_g: jax.Array, xi: float) -> jax.Array:
    r"""Cornette–Shanks phase function.

    .. math::

        P(\cos g) = \frac{3}{2} \frac{1 - \xi^2}{2 + \xi^2}
        \frac{1 + \cos^2 g}{(1 + \xi^2 - 2\xi\cos g)^{3/2}}

    Parameters
    ----------
    cos_g : jax.Array
        Cosine of the phase angle :math:`\cos g`.
    xi : float
        Asymmetry parameter :math:`\xi`.

    Returns
    -------
    jax.Array
        Phase function value(s) at the given angle(s).

    References
    ----------
    :cite:p:`Cornette-1992`
    """
    return (
        1.5
        * (1.0 - xi**2)
        / (2.0 + xi**2)
        * (1.0 + cos_g**2)
        / (1.0 + xi**2 - 2.0 * xi * cos_g) ** 1.5
    )


def legendre_eval(x: jax.Array, b_n: jax.Array) -> jax.Array:
    r"""Evaluate a Legendre polynomial series via the Bonnet recurrence.

    Computes :math:`\sum_{n=0}^{N} b_n P_n(x)` using the three-term
    recurrence :math:`n P_n = (2n-1) x P_{n-1} - (n-1) P_{n-2}`.

    Parameters
    ----------
    x : jax.Array
        Argument :math:`x` where :math:`|x| \leq 1`.
    b_n : jax.Array
        Coefficients :math:`b_n` of the Legendre series.

    Returns
    -------
    jax.Array
        Value of the Legendre series at *x*.

    References
    ----------
    :cite:p:`Hapke-2002`
    """
    p_n_2 = 1.0
    p_n_1 = x
    res = b_n[0] + b_n[1] * x
    for i in range(2, b_n.shape[0]):
        p_n = (2.0 - 1.0 / i) * x * p_n_1 - (1.0 - 1.0 / i) * p_n_2
        res = res + p_n * b_n[i]
        p_n_2 = p_n_1
        p_n_1 = p_n
    return res


def function_p(x: jax.Array, b_n: jax.Array, a_n: jax.Array) -> jax.Array:
    r"""Hapke P-function for anisotropic multiple scattering.

    Computes the single-scattering phase function contribution to multiple
    scattering (Hapke 2002, Eqs. 23–24):

    .. math::

        P(\cos g) = 1 + \sum_{n=0}^{N} a_n b_n P_n(\cos g)

    where :math:`a_n` are the Hapke coefficients and :math:`b_n` are the
    Legendre expansion coefficients of the single-particle phase function.

    Parameters
    ----------
    x : jax.Array
        Cosine of the phase angle :math:`\cos g`.
    b_n : jax.Array
        Legendre expansion coefficients of the phase function.
    a_n : jax.Array
        Hapke coefficients :math:`a_n`.

    Returns
    -------
    jax.Array
        Value of the P-function at *x*.

    References
    ----------
    :cite:p:`Hapke-2002`
    """
    ab_n = a_n * b_n
    p_n_2 = 1.0
    p_n_1 = x
    res = ab_n[0] + x * ab_n[1]
    for i in range(2, b_n.shape[0]):
        p_n = (2.0 - 1.0 / i) * x * p_n_1 - (1.0 - 1.0 / i) * p_n_2
        res = res + p_n * ab_n[i]
        p_n_2 = p_n_1
        p_n_1 = p_n
    return res + 1.0


def value_p(b_n: jax.Array, a_n: jax.Array) -> jax.Array:
    r"""Scalar value of the Hapke P-function.

    Computes the scalar P-value (Hapke 2002, Eq. 25):

    .. math::

        \langle P \rangle = 1 + \sum_{n=0}^{N} a_n^2 b_n

    Parameters
    ----------
    b_n : jax.Array
        Legendre expansion coefficients of the phase function.
    a_n : jax.Array
        Hapke coefficients :math:`a_n`.

    Returns
    -------
    jax.Array
        Scalar :math:`\langle P \rangle`.

    References
    ----------
    :cite:p:`Hapke-2002`
    """
    return 1.0 + jnp.sum(a_n**2 * b_n)


def shadow_hiding(tan_alpha_2: jax.Array, h: float, b0: float) -> jax.Array:
    r"""Shadow-hiding opposition effect :math:`B_{SH}`.

    .. math::

        B_{SH}(\alpha) = 1 + \frac{B_0}{1 + \tan(\alpha/2) / h}

    where :math:`B_0` is the opposition surge amplitude and *h* is the
    angular width of the opposition effect (Hapke 1984).

    Parameters
    ----------
    tan_alpha_2 : jax.Array
        Tangent of half the phase angle, :math:`\tan(\alpha/2)`.
    h : float
        Angular width parameter.
    b0 : float
        Opposition surge amplitude :math:`B_0`.

    Returns
    -------
    jax.Array
        Shadow-hiding opposition effect factor :math:`B_{SH}`.

    References
    ----------
    :cite:p:`Hapke-1984`
    """
    b_sh = 1.0
    return jnp.where(
        (b0 > 0.0) & (h > 0.0),
        b_sh + b0 / (1.0 + tan_alpha_2 / h),
        b_sh,
    )


def coherent_backscatter(tan_alpha_2: jax.Array, h: float, b0: float) -> jax.Array:
    r"""Coherent backscatter opposition effect :math:`B_{CB}`.

    Computes the coherent backscatter enhancement factor (Hapke 2002):

    .. math::

        B_{CB}(\alpha) = 1 + B_0 \,
        \frac{1}{2}
        \frac{1 + (1 - e^{-x}) / x}{(1 + x)^2},
        \quad x = \frac{\tan(\alpha/2)}{h}

    Parameters
    ----------
    tan_alpha_2 : jax.Array
        Tangent of half the phase angle, :math:`\tan(\alpha/2)`.
    h : float
        Angular width parameter.
    b0 : float
        Opposition surge amplitude :math:`B_0`.

    Returns
    -------
    jax.Array
        Coherent backscatter opposition effect factor :math:`B_{CB}`.

    References
    ----------
    :cite:p:`Hapke-2002`
    """
    hc_2 = tan_alpha_2 / jnp.maximum(h, EPS)
    # (1 - exp(-x)) / x -> 1 as x -> 0; guard the division so the limit is
    # exact at opposition (double-where keeps gradients NaN-free).
    hc_2_safe = jnp.where(hc_2 > EPS, hc_2, 1.0)
    ratio = jnp.where(hc_2 > EPS, -jnp.expm1(-hc_2_safe) / hc_2_safe, 1.0)
    bc = 0.5 * (1.0 + ratio) / (1.0 + hc_2) ** 2
    return jnp.where((b0 != 0.0) & (h != 0.0), 1.0 + b0 * bc, 1.0)


def _fe(x: jax.Array, y: jax.Array) -> jax.Array:
    r"""Exponential factor for the roughness correction.

    Parameters
    ----------
    x : jax.Array
        Cotangent or transformed angle argument.
    y : jax.Array
        Scale argument.

    Returns
    -------
    jax.Array
        :math:`\exp(-2 y x / \pi)` or 0 where *x* is infinite.
    """
    return jnp.where(jnp.isinf(x), 0.0, jnp.exp(-2.0 / jnp.pi * y * x))


def _fe2(x: jax.Array, y: jax.Array) -> jax.Array:
    r"""Gaussian factor for the roughness correction.

    Parameters
    ----------
    x : jax.Array
        Cotangent or transformed angle argument.
    y : jax.Array
        Scale argument.

    Returns
    -------
    jax.Array
        :math:`\exp(-y^2 x^2 / \pi)` or 0 where *x* is infinite.
    """
    return jnp.where(jnp.isinf(x), 0.0, jnp.exp(-(y**2) * x**2 / jnp.pi))


def roughness_correction(
    roughness: float,
    i: jax.Array,
    e: jax.Array,
    n: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Microscopic roughness shadowing correction.

    Computes the macroscopic roughness correction factor :math:`S` and
    effective cosines :math:`\mu_{0e}` and :math:`\mu_e` following
    Hapke (1984).

    Parameters
    ----------
    roughness : float
        RMS slope angle :math:`\bar{\theta}` in radians.
    i : jax.Array
        Unit vector toward the light source.
    e : jax.Array
        Unit vector toward the observer.
    n : jax.Array
        Unit surface normal vector.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array]
        A tuple ``(S, mu0, mu)`` where:

        - *S* is the roughness shadowing correction factor.
        - *mu0* is the effective cosine of the incidence angle.
        - *mu* is the effective cosine of the emission angle.

    References
    ----------
    :cite:p:`Hapke-1984`
    """
    cos_i = cos_angle(i, n)
    cos_e = cos_angle(e, n)
    safe_r = jnp.maximum(roughness, EPS)
    s_r, mu0_r, mu_r = _roughness_impl(safe_r, i, e, n, cos_i, cos_e)
    s = jnp.where(roughness < EPS, 1.0, s_r)
    mu0 = jnp.where(roughness < EPS, cos_i, mu0_r)
    mu = jnp.where(roughness < EPS, cos_e, mu_r)
    return s, mu0, mu


def _roughness_impl(
    roughness: float,
    i: jax.Array,
    e: jax.Array,
    n: jax.Array,
    cos_i: jax.Array,
    cos_e: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Implementation of the microscopic roughness shadowing correction.

    Computes the roughness correction factor and effective cosines for
    non-zero roughness values. For zero or near-zero roughness, the
    identity correction is returned by the caller
    :func:`roughness_correction`.

    Parameters
    ----------
    roughness : float
        RMS slope angle :math:`\bar{\theta}` in radians (assumed non-zero).
    i : jax.Array
        Unit vector toward the light source.
    e : jax.Array
        Unit vector toward the observer.
    n : jax.Array
        Unit surface normal vector.
    cos_i : jax.Array
        Pre-computed cosine of incidence angle.
    cos_e : jax.Array
        Pre-computed cosine of emission angle.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array]
        A tuple ``(S, mu0, mu)`` as in :func:`roughness_correction`.

    References
    ----------
    :cite:p:`Hapke-1984`
    """
    sin_i = jnp.sqrt(1.0 - cos_i**2)
    sin_e = jnp.sqrt(1.0 - cos_e**2)

    cot_i = jnp.where(sin_i > 0.0, cos_i / sin_i, 0.0)
    cot_e = jnp.where(sin_e > 0.0, cos_e / sin_e, 0.0)

    cos_psi_raw = jnp.dot(i, e)
    cos_psi = jnp.clip(
        (cos_psi_raw - cos_i * cos_e) / (sin_i * sin_e),
        -1.0 + EPS,
        1.0,
    )
    psi = jnp.arccos(cos_psi)
    sin_psi = jnp.sin(psi)
    sin_psi_div_2_sq = jnp.abs(0.5 - cos_psi / 2.0)

    cot_rough = 1.0 / jnp.tan(roughness)
    factor = 1.0 / jnp.sqrt(1.0 + jnp.pi / cot_rough**2)
    f_psi = jnp.exp(-2.0 * sin_psi / (1.0 + cos_psi))

    cos_i_s0 = factor * (
        cos_i
        + sin_i / cot_rough * _fe2(cot_i, cot_rough) / (2.0 - _fe(cot_i, cot_rough))
    )
    cos_e_s0 = factor * (
        cos_e
        + sin_e / cot_rough * _fe2(cot_e, cot_rough) / (2.0 - _fe(cot_e, cot_rough))
    )

    ile = cos_i >= cos_e

    def _cos_s_one(cos_x, sin_x, _cos_psi, _sin_psi_div_2_sq, cot_a, cot_b):
        return factor * (
            cos_x
            + sin_x
            / cot_rough
            * (
                _cos_psi * _fe2(cot_a, cot_rough)
                + _sin_psi_div_2_sq * _fe2(cot_b, cot_rough)
            )
            / (2.0 - _fe(cot_a, cot_rough) - psi / jnp.pi * _fe(cot_b, cot_rough))
        )

    # Hapke 1984, Eqs. 47-50. The i>=e and i<e cases are the same expression
    # with the two angles exchanged, so selecting the *inputs* on ``ile``
    # evaluates each effective cosine once instead of computing both branches
    # and discarding one.
    #
    # Worth calibrating expectations: this removes half the calls but only
    # about 10 % of the emitted HLO and 7 % of CPU runtime, because XLA
    # already shared most of the duplicated subexpressions -- the two branches
    # differ in argument order, not in the underlying `_fe`/`_fe2` terms.
    # Whether it helps the pathological CUDA compile time for `tb > 0`
    # (benchmark/README.md) is untested; no PTX toolchain was available here.
    #
    # The reverse-mode gradient is a clearer win: `jnp.where` on outputs routes
    # a zero cotangent into the discarded branch, and 0 * inf is NaN if that
    # branch has a singular derivative. Selecting inputs never evaluates it.
    cot_near, cot_far = (
        jnp.where(ile, cot_e, cot_i),
        jnp.where(ile, cot_i, cot_e),
    )
    # Whichever angle is larger takes the cos_psi-weighted numerator; the other
    # takes the unweighted one with the half-angle term negated.
    cos_psi_i = jnp.where(ile, cos_psi, 1.0)
    cos_psi_e = jnp.where(ile, 1.0, cos_psi)
    half_i = jnp.where(ile, sin_psi_div_2_sq, -sin_psi_div_2_sq)

    cos_i_s = _cos_s_one(cos_i, sin_i, cos_psi_i, half_i, cot_near, cot_far)
    cos_e_s = _cos_s_one(cos_e, sin_e, cos_psi_e, -half_i, cot_near, cot_far)

    s = factor * (cos_e_s / cos_e_s0) * (cos_i / cos_i_s0)

    div = 1.0 + f_psi * (
        factor * jnp.where(ile, cos_i / cos_i_s0, cos_e / cos_e_s0) - 1.0
    )

    s = s / div
    s = jnp.where((cos_i == 1.0) | (cos_e == 1.0), 1.0, s)
    cos_i_s = jnp.where((cos_i == 1.0) | (cos_e == 1.0), cos_i, cos_i_s)
    cos_e_s = jnp.where((cos_i == 1.0) | (cos_e == 1.0), cos_e, cos_e_s)

    return s, cos_i_s, cos_e_s
