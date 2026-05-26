import jax
import jax.numpy as jnp

EPS = 1e-15


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
    norm = jnp.sqrt(jnp.sum(v**2))
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

    Computes the Ambartsumian–Chandrasekhar H-function using the
    Cornette & Shanks level 2 approximation:

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
    :cite:p:`Cornette-1992`
    """
    gamma = jnp.sqrt(1.0 - w)
    r0 = (1.0 - gamma) / (1.0 + gamma)
    h_inv = 1.0 - w * x * (r0 + (1.0 - 2.0 * r0 * x) / 2.0 * jnp.log((1.0 + x) / x))
    return 1.0 / h_inv


def h_function_derivative(x: jax.Array, w: jax.Array) -> jax.Array:
    r"""Derivative of the H-function with respect to single-scattering albedo.

    Computes :math:`\partial H(x, w) / \partial w` using the Cornette &
    Shanks level 2 approximation.

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
    :cite:p:`Cornette-1992`
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

    References
    ----------
    :cite:p:`Henyey-1941`
    """
    range_n = jnp.arange(n + 1, dtype=jnp.float64)
    b_n = (2.0 * range_n + 1.0) * jnp.power(b, range_n)
    b_n = b_n.at[1::2].multiply(c)
    return b_n


def cs_legendre_coefficients(xi: float, n: int = 15) -> jax.Array:
    r"""Legendre expansion coefficients for the Cornette–Shanks phase function.

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
    return (
        (1.0 + c) / 2.0 * (1.0 - b**2) / (1.0 - 2.0 * b * cos_g + b**2) ** 1.5
        + (1.0 - c) / 2.0 * (1.0 - b**2) / (1.0 + 2.0 * b * cos_g + b**2) ** 1.5
    )


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
    r"""Evaluate a Legendre polynomial series using Clenshaw recurrence.

    Computes :math:`\sum_{n=0}^{N} b_n P_n(x)`.

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
    bc = 0.5 * (1.0 + (1.0 - jnp.exp(-hc_2)) / (hc_2 + EPS)) / (1.0 + hc_2) ** 2
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
        cos_i + sin_i / cot_rough * _fe2(cot_i, cot_rough) / (2.0 - _fe(cot_i, cot_rough))
    )
    cos_e_s0 = factor * (
        cos_e + sin_e / cot_rough * _fe2(cot_e, cot_rough) / (2.0 - _fe(cot_e, cot_rough))
    )

    ile = cos_i >= cos_e

    def _cos_s_one(cos_x, sin_x, _cos_psi, _psi, _sin_psi_div_2_sq, cot_a, cot_b, cot_c, cot_d):
        return factor * (
            cos_x + sin_x / cot_rough * (
                _cos_psi * _fe2(cot_a, cot_rough)
                + _sin_psi_div_2_sq * _fe2(cot_b, cot_rough)
            ) / (2.0 - _fe(cot_c, cot_rough) - _psi / jnp.pi * _fe(cot_d, cot_rough))
        )

    cos_i_s_ile = _cos_s_one(cos_i, sin_i, cos_psi, psi, sin_psi_div_2_sq, cot_e, cot_i, cot_e, cot_i)
    cos_i_s_ige = _cos_s_one(cos_i, sin_i, 1.0, 0.0, -sin_psi_div_2_sq, cot_i, cot_e, cot_i, cot_e)
    cos_i_s = jnp.where(ile, cos_i_s_ile, cos_i_s_ige)

    cos_e_s_ige = _cos_s_one(cos_e, sin_e, cos_psi, psi, sin_psi_div_2_sq, cot_i, cot_e, cot_i, cot_e)
    cos_e_s_ile = _cos_s_one(cos_e, sin_e, 1.0, 0.0, -sin_psi_div_2_sq, cot_e, cot_i, cot_e, cot_i)
    cos_e_s = jnp.where(ile, cos_e_s_ile, cos_e_s_ige)

    s = factor * (cos_e_s / cos_e_s0) * (cos_i / cos_i_s0)

    div_ile = 1.0 + f_psi * (factor * (cos_i / cos_i_s0) - 1.0)
    div_ige = 1.0 + f_psi * (factor * (cos_e / cos_e_s0) - 1.0)
    div = jnp.where(ile, div_ile, div_ige)

    s = s / div
    s = jnp.where((cos_i == 1.0) | (cos_e == 1.0), 1.0, s)
    cos_i_s = jnp.where((cos_i == 1.0) | (cos_e == 1.0), cos_i, cos_i_s)
    cos_e_s = jnp.where((cos_i == 1.0) | (cos_e == 1.0), cos_e, cos_e_s)

    return s, cos_i_s, cos_e_s
