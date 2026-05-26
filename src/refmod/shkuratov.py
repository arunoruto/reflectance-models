import jax
import jax.numpy as jnp


def _shkuratov_scalar(
    a_n: jax.Array,
    mu1: float,
    eta: float,
    i: jax.Array,
    e: jax.Array,
    n: jax.Array,
    m0: float = 0.0,
    mu2: float = 0.0,
) -> jax.Array:
    i_n = i / jnp.maximum(jnp.sqrt(jnp.sum(i**2)), 1e-12)
    e_n = e / jnp.maximum(jnp.sqrt(jnp.sum(e**2)), 1e-12)
    n_n = n / jnp.maximum(jnp.sqrt(jnp.sum(n**2)), 1e-12)

    mu0 = jnp.dot(i_n, n_n)
    mu = jnp.dot(e_n, n_n)
    cos_alpha = jnp.clip(jnp.dot(i_n, e_n), -1.0, 1.0)
    alpha = jnp.arccos(cos_alpha)
    sin_alpha = jnp.sqrt(1.0 - cos_alpha**2)

    tan_gamma = jnp.where(
        sin_alpha > 1e-12,
        (mu0 / jnp.maximum(mu, 1e-12) - cos_alpha) / sin_alpha,
        0.0,
    )
    gamma = jnp.arctan(tan_gamma)

    cos_beta = jnp.where(
        jnp.abs(jnp.cos(gamma)) > 1e-12,
        mu / jnp.cos(gamma),
        0.0,
    )
    cos_beta = jnp.clip(cos_beta, -1.0, 1.0)

    phase = (jnp.exp(-mu1 * alpha) + m0 * jnp.exp(-mu2 * alpha)) / (1.0 + m0)

    pi_minus_alpha = jnp.maximum(jnp.pi - alpha, 1e-12)
    exponent = eta * alpha * pi_minus_alpha
    cos_beta_pow = jnp.where(
        jnp.abs(cos_beta) > 1e-12,
        jnp.abs(cos_beta) ** exponent,
        0.0,
    )

    disk = (
        jnp.cos(alpha / 2.0)
        * jnp.cos(jnp.pi / pi_minus_alpha * (gamma - alpha / 2.0))
        * cos_beta_pow
        / jnp.maximum(jnp.cos(gamma), 1e-12)
    )

    a = a_n * phase * disk
    fr = a / jnp.maximum(mu0, 1e-12)
    return jnp.where((mu0 > 0.0) & (mu > 0.0), fr, 0.0)


_shkuratov_batched = jax.vmap(
    _shkuratov_scalar,
    in_axes=(0, None, None, 0, 0, 0, None, None),
)


def shkuratov(
    a_n: jax.Array,
    mu1: float = 0.0,
    eta: float = 0.0,
    i: jax.Array | None = None,
    e: jax.Array | None = None,
    n: jax.Array | None = None,
    m0: float = 0.0,
    mu2: float = 0.0,
) -> jax.Array:
    r"""Shkuratov photometric model with the Akimov disk function.

    Reflectance is computed as

    .. math::

       r = A_n \, \frac{\phi(\alpha) \, D(\alpha, \beta, \gamma)}{\cos i}

    where :math:`\phi(\alpha)` is the phase function and
    :math:`D(\alpha, \beta, \gamma)` is the Akimov disk function.

    Parameters
    ----------
    a_n : jax.Array
        Normal albedo per pixel.  Shape ``(n_pixels,)``.
    mu1 : float, optional
        Roughness parameter (exponential phase term).  Default is 0.0.
    eta : float, optional
        Fractal deviation parameter controlling the limb-darkening exponent.
        Default is 0.0.
    i : jax.Array or None, optional
        Incidence direction vectors.  Shape ``(n_pixels, 3)``.
    e : jax.Array or None, optional
        Emission direction vectors.  Shape ``(n_pixels, 3)``.
    n : jax.Array or None, optional
        Surface normal vectors.  Shape ``(n_pixels, 3)``.
    m0 : float, optional
        Opposition surge amplitude.  Default is 0.0 (no opposition effect).
    mu2 : float, optional
        Opposition surge width.  Default is 0.0.

    Returns
    -------
    jax.Array
        Reflectance per pixel.  Shape ``(n_pixels,)``.

    References
    ----------
    :cite:p:`Shkuratov-2011`
    """
    return _shkuratov_batched(a_n, mu1, eta, i, e, n, m0, mu2)
