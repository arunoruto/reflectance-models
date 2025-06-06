# import numpy as np
# from astropy.io import fits
# from scipy.optimize import least_squares

# from refmod.dtm_helper import dtm2grad
# from refmod.hapke import double_henyey_greenstein
# from refmod.hapke.amsa import amsa
# from refmod.hapke.imsa import imsa
# from refmod.hapke.legendre import coef_a, coef_b
# from refmod.inverse import inverse_model

# DATA_DIR = "test/data"


# def test_inverse_amsa():
#     file_name = f"{DATA_DIR}/hopper_amsa.fits"
#     f = fits.open(file_name)

#     result = f["result"].data
#     i = np.deg2rad(f["result"].header["i"])
#     e = np.deg2rad(f["result"].header["e"])
#     b = f["result"].header["b"]
#     c = f["result"].header["c"]
#     hs = f["result"].header["hs"]
#     bs0 = f["result"].header["bs0"]
#     tb = f["result"].header["tb"]
#     hc = f["result"].header["hc"]
#     bc0 = f["result"].header["bc0"]
#     albedo = f["albedo"].data
#     dtm = f["dtm"].data
#     resolution = f["dtm"].header["res"]

#     n = dtm2grad(dtm, resolution, normalize=False)

#     u, v = result.shape

#     i = np.reshape([np.sin(i), 0, np.cos(i)], [1, 1, -1])
#     e = np.reshape([np.sin(e), 0, np.cos(e)], [1, 1, -1])
#     i = np.tile(i, (u, v, 1))
#     e = np.tile(e, (u, v, 1))

#     a_n = coef_a()
#     b_n = coef_b(b, c)
#     refl = amsa(
#         i,
#         e,
#         n,
#         albedo,
#         lambda x: double_henyey_greenstein(x, b, c),
#         b_n,
#         a_n,
#         hs,
#         bs0,
#         tb,
#         hc,
#         bc0,
#     )
#     refl = refl.flatten()
#     albedo_recon = inverse_model(
#         refl,
#         i,
#         e,
#         n,
#         lambda x: double_henyey_greenstein(x, b, c),
#         b_n,
#         a_n,
#         hs,
#         bs0,
#         tb,
#         hc,
#         bc0,
#     )

#     np.testing.assert_allclose(albedo_recon, albedo)
