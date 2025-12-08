import numpy as np
from astropy.io import fits
from scipy.io import loadmat

from refmod.dtm_helper import dtm2grad
from refmod.hapke.functions.legendre import coef_a, dhg_legendre_coefficients
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

    a_n = coef_a()
    b_n = dhg_legendre_coefficients(b, c)

    refl = imsa(
        single_scattering_albedo=albedo,
        b_n=b_n,
        incidence_direction=i,
        emission_direction=e,
        surface_orientation=n,
        a_n=a_n,
        roughness=tb,
        opposition_effect_h=h,
        opposition_effect_b0=b0,
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

    b_n = dhg_legendre_coefficients(b, c)
    a_n = coef_a()
    # b_n_actual = dhg_legendre_coefficients(b, c)
    # range_n = np.arange(15 + 1)
    # b_n = (c * (2 * range_n + 1) * np.power(b, range_n)).reshape(-1, 1, 1)
    # print(b_n_actual.squeeze())
    # print(b_n.squeeze())
    # print(a_n.squeeze())
    # print(a_n.squeeze() * b_n_actual.squeeze())
    # print(a_n.squeeze() * b_n.squeeze())

    refl = amsa(
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
        # phase_function_type="dhg",
        # phase_function_args=(b, c),
    )
    result[np.isnan(refl)] = np.nan
    np.testing.assert_allclose(refl, result)
    # np.testing.assert_allclose(refl, result, rtol=1e-20)


# def test_amsa_olivine():
#     file_name = f"{DATA_DIR}/olivine_shackelford.mat"
#     f = loadmat(file_name)

#     i = np.deg2rad(f["incidence_angle"][0, 0])
#     e = np.deg2rad(f["emission_angle"][0, 0])
#     b = f["b"][0, 0]
#     c = f["c"][0, 0]
#     tb = f["tb"][0, 0]
#     hs = np.nan_to_num(f["hs"][0, 0])
#     bs0 = np.nan_to_num(f["Bs0"][0, 0])
#     hc = np.nan_to_num(f["hc"][0, 0])
#     bc0 = np.nan_to_num(f["Bc0"][0, 0])

#     types = ["fresh", "mature"]
#     for t in types:
#         spectrum = f[f"spectrum_{t}"]
#         albedo = f[f"albedo_{t}"]
#         refl = amsa(
#             albedo,
#             coef_b(b, c),
#             np.reshape([np.sin(i), 0, np.cos(i)], [-1, 1, 1]),
#             np.reshape([np.sin(e), 0, np.cos(e)], [-1, 1, 1]),
#             np.array([0.0, 0.0, 1.0]).reshape([-1, 1, 1]),
#             coef_a(),
#             tb,
#             hs,
#             bs0,
#             hc,
#             bc0,
#         )
#         print(spectrum.shape, albedo.shape, refl.shape)
#         print(np.mean((refl - spectrum) / spectrum))
#         np.testing.assert_allclose(spectrum.flatten(), refl.flatten())
