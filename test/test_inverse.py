import numpy as np
import jax.numpy as jnp
import pytest
from astropy.io import fits

from refmod.dtm_helper import dtm2grad
from refmod.hapke import (
    Hapke,
    amsa,
    dhg_legendre_coefficients,
    invert_amsa_precomputed,
    prepare_amsa_inversion,
)
from refmod.hapke.inverse import invert_amsa

DATA_DIR = "test/data"


def test_inverse_amsa_small():
    """Test that inverting the forward model recovers the original albedo."""
    b_n = dhg_legendre_coefficients(0.21, 0.7, 15)

    w_true = jnp.array([0.3, 0.5, 0.7])
    i_vec = jnp.array([0.0, jnp.sin(jnp.deg2rad(30.0)), jnp.cos(jnp.deg2rad(30.0))])
    e_vec = jnp.array([0.0, jnp.sin(jnp.deg2rad(10.0)), jnp.cos(jnp.deg2rad(10.0))])
    n_vec = jnp.array([0.0, 0.0, 1.0])

    i_batch = jnp.tile(i_vec, (3, 1))
    e_batch = jnp.tile(e_vec, (3, 1))
    n_batch = jnp.tile(n_vec, (3, 1))

    refl = amsa(w_true, b_n, i_batch, e_batch, n_batch)
    w_recon = invert_amsa(refl, b_n, i_batch, e_batch, n_batch)

    np.testing.assert_allclose(
        np.array(w_recon),
        np.array(w_true),
        rtol=1e-4,
        err_msg="Inversion should recover original albedo",
    )


def test_precomputed_inverse_matches_convenience_api():
    b_n = dhg_legendre_coefficients(0.21, 0.7, 15)
    w_true = jnp.array([0.2, 0.4, 0.6, 0.8])
    i_vec = jnp.array([0.0, jnp.sin(jnp.deg2rad(35.0)), jnp.cos(jnp.deg2rad(35.0))])
    e_vec = jnp.array([0.0, jnp.sin(jnp.deg2rad(15.0)), jnp.cos(jnp.deg2rad(15.0))])
    n_vec = jnp.array([0.0, 0.0, 1.0])
    i_batch = jnp.tile(i_vec, (w_true.shape[0], 1))
    e_batch = jnp.tile(e_vec, (w_true.shape[0], 1))
    n_batch = jnp.tile(n_vec, (w_true.shape[0], 1))

    params = dict(roughness=0.1, h_sh=0.03, b0_sh=0.2, h_cb=0.04, b0_cb=0.1)
    refl = amsa(w_true, b_n, i_batch, e_batch, n_batch, **params)

    state = prepare_amsa_inversion(b_n, i_batch, e_batch, n_batch, **params)
    w_precomputed = invert_amsa_precomputed(refl, state)
    w_convenience = invert_amsa(refl, b_n, i_batch, e_batch, n_batch, **params)

    np.testing.assert_allclose(
        np.array(w_precomputed), np.array(w_convenience), rtol=1e-8
    )
    np.testing.assert_allclose(np.array(w_precomputed), np.array(w_true), rtol=1e-4)


def test_precomputed_inverse_state_can_be_reused():
    b_n = dhg_legendre_coefficients(0.15, 0.4, 12)
    w_a = jnp.array([0.25, 0.45, 0.65])
    w_b = jnp.array([0.3, 0.5, 0.7])
    i_batch = jnp.tile(jnp.array([0.0, 0.5, 0.8660254]), (3, 1))
    e_batch = jnp.tile(jnp.array([0.0, 0.2, 0.9797959]), (3, 1))
    n_batch = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (3, 1))

    state = prepare_amsa_inversion(b_n, i_batch, e_batch, n_batch)
    refl_a = amsa(w_a, b_n, i_batch, e_batch, n_batch)
    refl_b = amsa(w_b, b_n, i_batch, e_batch, n_batch)

    np.testing.assert_allclose(
        np.array(invert_amsa_precomputed(refl_a, state)), np.array(w_a), rtol=1e-4
    )
    np.testing.assert_allclose(
        np.array(invert_amsa_precomputed(refl_b, state)), np.array(w_b), rtol=1e-4
    )


def test_precomputed_inverse_forced_chunking():
    b_n = dhg_legendre_coefficients(0.2, 0.6, 12)
    w_true = jnp.array([0.2, 0.35, 0.5, 0.65, 0.8])
    i_batch = jnp.tile(jnp.array([0.0, 0.4, 0.9165151]), (w_true.shape[0], 1))
    e_batch = jnp.tile(jnp.array([0.0, 0.1, 0.9949874]), (w_true.shape[0], 1))
    n_batch = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (w_true.shape[0], 1))

    refl = amsa(w_true, b_n, i_batch, e_batch, n_batch)
    state = prepare_amsa_inversion(b_n, i_batch, e_batch, n_batch, chunk_size=2)
    assert len(state.chunks) == 3

    w_recon = invert_amsa_precomputed(refl, state)
    np.testing.assert_allclose(np.array(w_recon), np.array(w_true), rtol=1e-4)


def test_precomputed_inverse_rejects_reflectance_length_mismatch():
    b_n = dhg_legendre_coefficients(0.15, 0.4, 12)
    w = jnp.array([0.25, 0.45, 0.65])
    i_batch = jnp.tile(jnp.array([0.0, 0.5, 0.8660254]), (3, 1))
    e_batch = jnp.tile(jnp.array([0.0, 0.2, 0.9797959]), (3, 1))
    n_batch = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (3, 1))

    state = prepare_amsa_inversion(b_n, i_batch, e_batch, n_batch)
    refl = amsa(w, b_n, i_batch, e_batch, n_batch)

    with pytest.raises(ValueError, match="reflectance length"):
        invert_amsa_precomputed(jnp.concatenate([refl, refl[:1]]), state)


def test_precomputed_inverse_rejects_w0_length_mismatch():
    b_n = dhg_legendre_coefficients(0.15, 0.4, 12)
    w = jnp.array([0.25, 0.45, 0.65])
    i_batch = jnp.tile(jnp.array([0.0, 0.5, 0.8660254]), (3, 1))
    e_batch = jnp.tile(jnp.array([0.0, 0.2, 0.9797959]), (3, 1))
    n_batch = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (3, 1))

    state = prepare_amsa_inversion(b_n, i_batch, e_batch, n_batch)
    refl = amsa(w, b_n, i_batch, e_batch, n_batch)

    with pytest.raises(ValueError, match="initial guess length"):
        invert_amsa_precomputed(refl, state, w0=jnp.ones(state.n_pixels + 1))


def test_inverse_amsa_masks_nonfinite_reflectance():
    b_n = dhg_legendre_coefficients(0.21, 0.7, 15)
    w_true = jnp.array([0.3, 0.5, 0.7])
    i_vec = jnp.array([0.0, jnp.sin(jnp.deg2rad(30.0)), jnp.cos(jnp.deg2rad(30.0))])
    e_vec = jnp.array([0.0, jnp.sin(jnp.deg2rad(10.0)), jnp.cos(jnp.deg2rad(10.0))])
    n_vec = jnp.array([0.0, 0.0, 1.0])
    i_batch = jnp.tile(i_vec, (3, 1))
    e_batch = jnp.tile(e_vec, (3, 1))
    n_batch = jnp.tile(n_vec, (3, 1))

    refl = amsa(w_true, b_n, i_batch, e_batch, n_batch)
    refl_mixed = refl.at[1].set(jnp.nan)
    state = prepare_amsa_inversion(b_n, i_batch, e_batch, n_batch)

    w_direct = invert_amsa(refl_mixed, b_n, i_batch, e_batch, n_batch)
    w_precomputed = invert_amsa_precomputed(refl_mixed, state)

    finite_idx = np.array([0, 2])
    np.testing.assert_allclose(
        np.array(w_direct)[finite_idx], np.array(w_true)[finite_idx], rtol=1e-4
    )
    np.testing.assert_allclose(
        np.array(w_precomputed)[finite_idx], np.array(w_true)[finite_idx], rtol=1e-4
    )
    np.testing.assert_allclose(np.array(w_direct[1]), 0.5)
    np.testing.assert_allclose(np.array(w_precomputed[1]), 0.5)


def test_inverse_amsa_hopper():
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

    # Subset for speed
    r = 5
    uc = u // 2 + np.arange(-r, r)
    vc = v // 2 + np.arange(-r, r)

    albedo_sub = albedo[uc, :][:, vc]

    b_n = dhg_legendre_coefficients(b, c)

    model = Hapke(
        single_scattering_albedo=albedo_sub,
        legendre_coefficients=np.array(b_n),
        incidence_direction=np.array([np.sin(i_rad), 0, np.cos(i_rad)]).reshape(
            3, 1, 1
        ),
        emission_direction=np.array([np.sin(e_rad), 0, np.cos(e_rad)]).reshape(3, 1, 1),
        surface_orientation=n[uc, :, :][:, vc, :].transpose(2, 0, 1),
        roughness=tb,
        shadow_hiding_h=hs,
        shadow_hiding_b0=bs0,
        coherent_backscattering_h=hc,
        coherent_backscattering_b0=bc0,
    )

    refl = model.refl()
    albedo_recon = model.albedo(np.array(refl))

    np.testing.assert_allclose(
        albedo_recon,
        albedo_sub,
        rtol=1e-4,
        err_msg="Hapke model inversion should recover albedo",
    )
    f.close()
