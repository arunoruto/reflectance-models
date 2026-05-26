import jax
import jax.numpy as jnp


def _lambert_scalar(w: jax.Array, i: jax.Array, n: jax.Array) -> jax.Array:
    r"""Lambert reflectance for a single pixel.

    Computes :math:`w \cos i \,/\, \pi`.

    Parameters
    ----------
    w : jax.Array
        Single scattering albedo (scalar).
    i : jax.Array
        Incidence (illumination) direction vector.  Shape ``(3,)``.
    n : jax.Array
        Surface normal vector.  Shape ``(3,)``.

    Returns
    -------
    jax.Array
        Reflectance (scalar).  Zero when the source is behind the surface.
    """
    mu0 = jnp.dot(i, n)
    return w * jnp.maximum(mu0, 0.0) / jnp.pi


_lambert_batched = jax.vmap(_lambert_scalar, in_axes=(0, 0, 0))


def lambert(
    w: jax.Array,
    i: jax.Array,
    e: jax.Array,
    n: jax.Array,
) -> jax.Array:
    r"""Batched Lambert reflectance model.

    The model is view-independent — the emission direction *e* is accepted
    for API consistency but ignored.

    Parameters
    ----------
    w : jax.Array
        Single scattering albedo per pixel.  Shape ``(n_pixels,)``.
    i : jax.Array
        Incidence direction vectors.  Shape ``(n_pixels, 3)``.
    e : jax.Array
        Emission direction vectors (unused).  Shape ``(n_pixels, 3)``.
    n : jax.Array
        Surface normal vectors.  Shape ``(n_pixels, 3)``.

    Returns
    -------
    jax.Array
        Reflectance per pixel.  Shape ``(n_pixels,)``.
    """
    return _lambert_batched(w, i, n)
