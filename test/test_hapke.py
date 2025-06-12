import numpy as np
from astropy.io import fits
from refmod.dtm_helper import dtm2grad
from refmod.hapke.functions.legendre import coef_a, coef_b
from refmod.hapke.models import amsa, imsa

DATA_DIR = "test/data"
EXTENSION = "fits"
# EXTENSION = "mat"


def test_imsa_hopper():
    file_name = f"{DATA_DIR}/hopper_imsa.fits"
    f = fits.open(file_name)

    result = f["result"].data.astype(float)
    i = np.deg2rad(f["result"].header["i"])
    e = np.deg2rad(f["result"].header["e"])
    b = f["result"].header["b"]
    c = f["result"].header["c"]
    h = f["result"].header["hs"]
    b0 = f["result"].header["bs0"]
    tb = f["result"].header["tb"]
    albedo = f["albedo"].data.astype(float)
    dtm = f["dtm"].data.astype(float)
    resolution = f["dtm"].header["res"]

    n = dtm2grad(dtm, resolution, normalize=False)

    u = result.shape[0]
    v = result.shape[1]

    i = np.reshape([np.sin(i), 0, np.cos(i)], [-1, 1, 1])
    e = np.reshape([np.sin(e), 0, np.cos(e)], [-1, 1, 1])
    i = np.tile(i, (1, u, v))
    e = np.tile(e, (1, u, v))
    n = np.moveaxis(n, -1, 0)
    # i = np.reshape([np.sin(i), 0, np.cos(i)], [1, 1, -1])
    # e = np.reshape([np.sin(e), 0, np.cos(e)], [1, 1, -1])
    # i = np.tile(i, (u, v, 1))
    # e = np.tile(e, (u, v, 1))

    refl = imsa(
        single_scattering_albedo=albedo,
        incidence_direction=i,
        emission_direction=e,
        surface_orientation=n,
        phase_function_type="dhg",
        roughness=tb,
        opposition_effect_h=h,
        opposition_effect_b0=b0,
        phase_function_args=(b, c),
        h_level=1,
    )
    result[np.isnan(refl)] = np.nan

    np.testing.assert_allclose(refl, result * 4 * np.pi)


def test_amsa_hopper():
    file_name = f"{DATA_DIR}/hopper_amsa.fits"
    f = fits.open(file_name)

    result = f["result"].data.astype(float)
    i = np.deg2rad(f["result"].header["i"])
    e = np.deg2rad(f["result"].header["e"])
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

    u = result.shape[0]
    v = result.shape[1]

    i = np.reshape([np.sin(i), 0, np.cos(i)], [-1, 1, 1])
    e = np.reshape([np.sin(e), 0, np.cos(e)], [-1, 1, 1])
    i = np.tile(i, (1, u, v))
    e = np.tile(e, (1, u, v))
    n = np.moveaxis(n, -1, 0)
    # i = np.reshape([np.sin(i), 0, np.cos(i)], [1, 1, -1])
    # e = np.reshape([np.sin(e), 0, np.cos(e)], [1, 1, -1])
    # i = np.tile(i, (u, v, 1))
    # e = np.tile(e, (u, v, 1))

    a_n = coef_a()
    b_n = coef_b(b, c)

    refl = amsa(
        albedo,
        i,
        e,
        n,
        "dhg",
        b_n,
        a_n,
        tb,
        hs,
        bs0,
        hc,
        bc0,
        (b, c),
    )
    result[np.isnan(refl)] = np.nan
    np.testing.assert_allclose(refl, result)
    # np.testing.assert_allclose(refl, result, rtol=1e-20)
