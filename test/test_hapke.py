import numpy as np
import jax.numpy as jnp
from astropy.io import fits

from refmod.dtm_helper import dtm2grad
from refmod.hapke import amsa, mimsa, dhg_legendre_coefficients

DATA_DIR = "test/data"


def test_amsa_hopper():
    file_name = f"{DATA_DIR}/hopper_amsa.fits"
    f = fits.open(file_name)

    result = f["result"].data.astype(float)
    i_deg = f["result"].header["i"]
    e_deg = f["result"].header["e"]
    b = f["result"].header["b"]
    c = f["result"].header["c"]
    hs = f["result"].header["hs"]
    bs0 = f["result"].header["bs0"]
    tb = f["result"].header["tb"]
    hc = f["result"].header["hc"]
    bc0 = f["result"].header["bc0"]
    albedo = f["albedo"].data.astype(float)
    dtm = f["dtm"].data.astype(float)
    resolution = f["dtm"].header["res"]

    n = dtm2grad(dtm, resolution, normalize=False)

    u, v = result.shape
    i_rad = np.deg2rad(i_deg)
    e_rad = np.deg2rad(e_deg)

    i_flat = np.tile(np.array([np.sin(i_rad), 0, np.cos(i_rad)]), (u * v, 1))
    e_flat = np.tile(np.array([np.sin(e_rad), 0, np.cos(e_rad)]), (u * v, 1))
    n_flat = n.reshape(-1, 3)
    w_flat = albedo.reshape(-1)

    b_n = dhg_legendre_coefficients(b, c)

    refl = amsa(
        jnp.array(w_flat),
        b_n,
        jnp.array(i_flat),
        jnp.array(e_flat),
        jnp.array(n_flat),
        roughness=tb,
        h_sh=hs,
        b0_sh=bs0,
        h_cb=hc,
        b0_cb=bc0,
    )

    refl = np.array(refl).reshape(u, v)
    result[np.isnan(refl)] = np.nan

    np.testing.assert_allclose(refl, result)
    f.close()


def test_mimsa_equals_amsa_no_opposition():
    b_n = dhg_legendre_coefficients(0.21, 0.7, 15)

    w_batch = jnp.array([0.3, 0.5, 0.7])
    i_vec = jnp.array([0.0, jnp.sin(jnp.deg2rad(30.0)), jnp.cos(jnp.deg2rad(30.0))])
    e_vec = jnp.array([0.0, jnp.sin(jnp.deg2rad(10.0)), jnp.cos(jnp.deg2rad(10.0))])
    n_vec = jnp.array([0.0, 0.0, 1.0])

    i_batch = jnp.tile(i_vec, (3, 1))
    e_batch = jnp.tile(e_vec, (3, 1))
    n_batch = jnp.tile(n_vec, (3, 1))

    r_mimsa = mimsa(w_batch, b_n, i_batch, e_batch, n_batch)
    r_amsa = amsa(w_batch, b_n, i_batch, e_batch, n_batch)

    np.testing.assert_allclose(
        np.array(r_mimsa),
        np.array(r_amsa),
        err_msg="MIMSA should equal AMSA when opposition effects are zero",
    )
