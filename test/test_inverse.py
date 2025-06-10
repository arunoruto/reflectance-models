import numpy as np
from astropy.io import fits
from refmod.dtm_helper import dtm2grad
from refmod.hapke.functions.legendre import coef_a, coef_b
from refmod.hapke.functions.phase import double_henyey_greenstein
from refmod.hapke.inverse import inverse_model
from refmod.hapke.models import amsa

DATA_DIR = "test/data"


def test_inverse_amsa():
    file_name = f"{DATA_DIR}/hopper_amsa.fits"
    f = fits.open(file_name)

    result = f["result"].data
    i = np.deg2rad(f["result"].header["i"])
    e = np.deg2rad(f["result"].header["e"])
    b = f["result"].header["b"]
    c = f["result"].header["c"]
    hs = f["result"].header["hs"]
    bs0 = f["result"].header["bs0"]
    tb = f["result"].header["tb"]
    hc = f["result"].header["hc"]
    bc0 = f["result"].header["bc0"]
    albedo = f["albedo"].data
    dtm = f["dtm"].data
    resolution = f["dtm"].header["res"]

    n = dtm2grad(dtm, resolution, normalize=False)

    u = result.shape[0]
    v = result.shape[1]

    i = np.reshape([np.sin(i), 0, np.cos(i)], [1, 1, -1])
    e = np.reshape([np.sin(e), 0, np.cos(e)], [1, 1, -1])
    i = np.tile(i, (u, v, 1))
    e = np.tile(e, (u, v, 1))

    r = 20
    uc = u // 2 + np.arange(-r, r)
    vc = v // 2 + np.arange(-r, r)
    albedo = albedo[uc, :][:, vc]
    i = i[uc, :, :][:, vc, :]
    e = e[uc, :, :][:, vc, :]
    n = n[uc, :, :][:, vc, :]

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
    # refl = refl.flatten()
    albedo_recon = np.zeros_like(refl)
    for k in range(refl.shape[0] * refl.shape[1]):
        row = k % refl.shape[0]
        col = k // refl.shape[0]
        albedo_recon[row, col, ...] = inverse_model(
            refl[row, col, ...],
            i[row, col, :],
            e[row, col, :],
            n[row, col, :],
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

    np.testing.assert_allclose(albedo_recon, albedo)
