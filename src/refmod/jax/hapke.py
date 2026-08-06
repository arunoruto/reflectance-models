import jax
import jax.numpy as jnp

from refmod.hapke import amsa, coef_a, dhg_legendre_coefficients


def jax_amsa(
    w: jax.Array,
    i: jax.Array,
    e: jax.Array,
    n: jax.Array,
    b_n: jax.Array,
    a_n: jax.Array | None = None,
    roughness: float = 0.0,
    h_sh: float = 0.0,
    b0_sh: float = 0.0,
    h_cb: float = 0.0,
    b0_cb: float = 0.0,
) -> jax.Array:
    r"""Legacy image-shaped wrapper for the pre-1.0 ``refmod.jax.hapke`` API."""
    w_arr = jnp.asarray(w)
    out_shape = w_arr.shape
    return amsa(
        w_arr.reshape(-1),
        b_n,
        jnp.asarray(i).reshape(-1, 3),
        jnp.asarray(e).reshape(-1, 3),
        jnp.asarray(n).reshape(-1, 3),
        roughness,
        h_sh,
        b0_sh,
        h_cb,
        b0_cb,
        a_n,
    ).reshape(out_shape)


__all__ = ["coef_a", "dhg_legendre_coefficients", "jax_amsa"]
