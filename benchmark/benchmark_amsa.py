import os
import timeit

os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax.numpy as jnp
import numpy as np
from astropy.io import fits

from refmod.dtm_helper import dtm2grad
from refmod.hapke import amsa as jax_amsa
from refmod.hapke._core import coef_a as jax_coef_a
from refmod.hapke._core import dhg_legendre_coefficients as jax_dhg_legendre_coefficients


def load_data():
    file_name = "test/data/hopper_amsa.fits"
    f = fits.open(file_name)

    result = f["result"].data.astype(np.float64)
    i_deg = f["result"].header["i"]
    e_deg = f["result"].header["e"]
    b = f["result"].header["b"]
    c = f["result"].header["c"]
    hs = f["result"].header["hs"]
    bs0 = f["result"].header["bs0"]
    tb = f["result"].header["tb"]
    hc = f["result"].header["hc"]
    bc0 = f["result"].header["bc0"]
    albedo = f["albedo"].data.astype(np.float64)
    dtm = f["dtm"].data.astype(np.float64)
    resolution = f["dtm"].header["res"]

    n = dtm2grad(dtm, resolution, normalize=False)

    u, v = result.shape
    i_rad = np.deg2rad(i_deg)
    e_rad = np.deg2rad(e_deg)

    i_flat = np.tile(np.array([np.sin(i_rad), 0, np.cos(i_rad)]), (u * v, 1))
    e_flat = np.tile(np.array([np.sin(e_rad), 0, np.cos(e_rad)]), (u * v, 1))
    n_flat = n.reshape(-1, 3)
    w_flat = albedo.reshape(-1)

    f.close()
    return w_flat, i_flat, e_flat, n_flat, b, c, hs, bs0, tb, hc, bc0


def run_jax(albedo, i, e, n, b, c, hs, bs0, tb, hc, bc0):
    w = jnp.array(albedo)
    i_j = jnp.array(i)
    e_j = jnp.array(e)
    n_j = jnp.array(n)
    b_n = jax_dhg_legendre_coefficients(b, c)
    a_n = jax_coef_a()

    return jax_amsa(
        w=w,
        b_n=b_n,
        i=i_j,
        e=e_j,
        n=n_j,
        roughness=tb,
        h_sh=hs,
        b0_sh=bs0,
        h_cb=hc,
        b0_cb=bc0,
        a_n=a_n,
    ).block_until_ready()


if __name__ == "__main__":
    albedo, i, e, n, b, c, hs, bs0, tb, hc, bc0 = load_data()

    print("Warming up JAX...")
    run_jax(albedo, i, e, n, b, c, hs, bs0, tb, hc, bc0)

    n_runs = 10
    jax_time = timeit.timeit(
        lambda: run_jax(albedo, i, e, n, b, c, hs, bs0, tb, hc, bc0), number=n_runs
    )

    print(f"JAX took: {jax_time / n_runs:.6f} seconds per run")
