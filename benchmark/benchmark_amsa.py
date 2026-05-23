import timeit

import jax.numpy as jnp
import numpy as np
from astropy.io import fits

# from jax import config
from refmod.dtm_helper import dtm2grad
from refmod.hapke.models import amsa
from refmod.jax_gemini.hapke import jax_amsa, jax_coef_a, jax_dhg_legendre_coefficients

# config.update("jax_enable_x64", False)


def load_data():
    file_name = "test/data/hopper_amsa.fits"
    f = fits.open(file_name)

    result = f["result"].data.astype(np.float64)  # pyright: ignore [reportAttributeAccessIssue]
    i = np.deg2rad(f["result"].header["i"])  # pyright: ignore [reportAttributeAccessIssue]
    e = np.deg2rad(f["result"].header["e"])  # pyright: ignore [reportAttributeAccessIssue]
    b = f["result"].header["b"]  # pyright: ignore [reportAttributeAccessIssue]
    c = f["result"].header["c"]  # pyright: ignore [reportAttributeAccessIssue]
    hs = f["result"].header["hs"]  # pyright: ignore [reportAttributeAccessIssue]
    bs0 = f["result"].header["bs0"]  # pyright: ignore [reportAttributeAccessIssue]
    tb = f["result"].header["tb"]  # pyright: ignore [reportAttributeAccessIssue]
    hc = f["result"].header["hc"]  # pyright: ignore [reportAttributeAccessIssue]
    bc0 = f["result"].header["bc0"]  # pyright: ignore [reportAttributeAccessIssue]
    albedo = f["albedo"].data.astype(np.float64)  # pyright: ignore [reportAttributeAccessIssue]
    dtm = f["dtm"].data.astype(np.float64)  # pyright: ignore [reportAttributeAccessIssue]
    resolution = f["dtm"].header["res"]  # pyright: ignore [reportAttributeAccessIssue]

    n = dtm2grad(dtm, resolution, normalize=False)

    u = result.shape[0]
    v = result.shape[1]

    i = np.reshape([np.sin(i), 0, np.cos(i)], [-1, 1, 1])
    e = np.reshape([np.sin(e), 0, np.cos(e)], [-1, 1, 1])
    i = np.tile(i, (1, u, v))
    e = np.tile(e, (1, u, v))
    n = np.moveaxis(n, -1, 0)

    return albedo, i, e, n, b, c, hs, bs0, tb, hc, bc0


def run_numba(albedo, i, e, n, b, c, hs, bs0, tb, hc, bc0):
    from refmod.hapke.functions.legendre import coef_a, dhg_legendre_coefficients

    a_n = coef_a()
    b_n = dhg_legendre_coefficients(b, c)

    return amsa(
        albedo,
        b_n,
        i,
        e,
        n,
        a_n,
        tb,
        hs,
        bs0,
        hc,
        bc0,
    )


def run_jax(albedo, i, e, n, b, c, hs, bs0, tb, hc, bc0):
    albedo = jnp.array(albedo)
    i = jnp.array(i)
    e = jnp.array(e)
    n = jnp.array(n)
    b_n = jax_dhg_legendre_coefficients(b, c)
    a_n = jax_coef_a()

    return jax_amsa(
        single_scattering_albedo=albedo,
        phase_function_legendre=b_n,
        incidence_direction=i,
        emission_direction=e,
        surface_orientation=n,
        a_n=a_n,
        roughness=tb,
        shadow_hiding_h=hs,
        shadow_hiding_b0=bs0,
        coherant_backscattering_h=hc,
        coherant_backscattering_b0=bc0,
    ).block_until_ready()


if __name__ == "__main__":
    albedo, i, e, n, b, c, hs, bs0, tb, hc, bc0 = load_data()

    # Warm up JAX
    print("Warming up JAX...")
    run_jax(albedo, i, e, n, b, c, hs, bs0, tb, hc, bc0)

    n_runs = 10
    numba_time = timeit.timeit(
        lambda: run_numba(albedo, i, e, n, b, c, hs, bs0, tb, hc, bc0), number=n_runs
    )
    jax_time = timeit.timeit(
        lambda: run_jax(albedo, i, e, n, b, c, hs, bs0, tb, hc, bc0), number=n_runs
    )

    print(f"Numba took: {numba_time / n_runs:.6f} seconds per run")
    print(f"JAX took:   {jax_time / n_runs:.6f} seconds per run")

    if jax_time < numba_time:
        print(f"JAX is {numba_time / jax_time:.2f}x faster")
    else:
        print(f"Numba is {jax_time / numba_time:.2f}x faster")
