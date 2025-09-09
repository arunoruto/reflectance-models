import numpy as np
from astropy.io import fits
from refmod.dtm_helper import dtm2grad
from refmod.hapke.functions.legendre import coef_a, coef_b
from refmod.hapke.inverse import inverse_model
from refmod.hapke.models import amsa

DATA_DIR = "test/data"


def test_inverse_amsa():
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

    i = np.array([np.sin(i), 0, np.cos(i)]).reshape(3, 1, 1)
    e = np.array([np.sin(e), 0, np.cos(e)]).reshape(3, 1, 1)
    i = np.tile(i, (1, u, v))
    e = np.tile(e, (1, u, v))
    n = np.moveaxis(n, -1, 0)

    r = 10 * 1
    uc = u // 2 + np.arange(-r, r)
    vc = v // 2 + np.arange(-r, r)
    albedo = albedo[uc, :][:, vc]
    i = i[:, uc, :][:, :, vc]
    e = e[:, uc, :][:, :, vc]
    n = n[:, uc, :][:, :, vc]

    albedo = np.stack(
        [
            albedo,
            0.01 * albedo,
        ],
        axis=0,
    )

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
    # albedo_recon = np.zeros_like(refl)
    # for k in range(refl.shape[0] * refl.shape[1]):
    #     row = k % refl.shape[0]
    #     col = k // refl.shape[0]
    #     albedo_recon[row, col, ...] = inverse_model(
    #         refl[row, col, ...],
    #         i[row, col, :],
    #         e[row, col, :],
    #         n[row, col, :],
    #         "dhg",
    #         b_n,
    #         a_n,
    #         tb,
    #         hs,
    #         bs0,
    #         hc,
    #         bc0,
    #         (b, c),
    #     )

    albedo_recon = inverse_model(
        refl,
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

    np.testing.assert_allclose(albedo_recon, albedo)
