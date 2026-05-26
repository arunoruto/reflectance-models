import numpy as np

from refmod.hapke import Hapke, dhg_legendre_coefficients


def test_hapke_refl_accepts_runtime_1d_vectors_regression():
    model = Hapke(
        single_scattering_albedo=np.array([0.25, 0.5, 0.75]),
        legendre_coefficients=np.array(dhg_legendre_coefficients(0.2, 0.5, 8)),
        incidence_direction=np.array([0.0, 0.0, 1.0]),
        emission_direction=np.array([0.0, 0.0, 1.0]),
        surface_orientation=np.array([0.0, 0.0, 1.0]),
        model="imsa",
    )

    model.incidence_direction = np.array([0.0, 0.0, 1.0])
    model.emission_direction = np.array([0.0, 0.0, 1.0])
    model.surface_orientation = np.array([0.0, 0.0, 1.0])

    refl = model.refl()
    assert refl.shape == (3,)
    assert np.all(np.isfinite(refl))


def test_hapke_refl_broadcasts_single_geometry_to_albedo_shape():
    ssa = np.array([[0.3, 0.6], [0.1, 0.8]])
    model = Hapke(
        single_scattering_albedo=ssa,
        legendre_coefficients=np.array(dhg_legendre_coefficients(0.21, 0.7, 10)),
        incidence_direction=np.array([0.0, 0.0, 1.0]),
        emission_direction=np.array([0.0, 0.0, 1.0]),
        surface_orientation=np.array([0.0, 0.0, 1.0]),
        model="mimsa",
    )

    refl = model.refl()
    assert refl.shape == ssa.shape
    assert np.all(np.isfinite(refl))


def test_hapke_albedo_inverse_roundtrip_synthetic():
    ssa = np.array([0.2, 0.4, 0.65, 0.85])
    model = Hapke(
        single_scattering_albedo=ssa,
        legendre_coefficients=np.array(dhg_legendre_coefficients(0.2, 0.6, 12)),
        incidence_direction=np.array([0.0, np.sin(np.deg2rad(35.0)), np.cos(np.deg2rad(35.0))]),
        emission_direction=np.array([0.0, np.sin(np.deg2rad(15.0)), np.cos(np.deg2rad(15.0))]),
        surface_orientation=np.array([0.0, 0.0, 1.0]),
        model="amsa",
        roughness=np.deg2rad(10.0),
        shadow_hiding_h=0.05,
        shadow_hiding_b0=0.2,
        coherent_backscattering_h=0.08,
        coherent_backscattering_b0=0.1,
    )

    refl = model.refl()
    recon = model.albedo(refl)

    np.testing.assert_allclose(recon, ssa, rtol=3e-4, atol=1e-6)


def test_hapke_albedo_accepts_broadcast_x0():
    ssa = np.array([0.2, 0.4, 0.65, 0.85])
    model = Hapke(
        single_scattering_albedo=ssa,
        legendre_coefficients=np.array(dhg_legendre_coefficients(0.2, 0.6, 12)),
        incidence_direction=np.array([0.0, np.sin(np.deg2rad(35.0)), np.cos(np.deg2rad(35.0))]),
        emission_direction=np.array([0.0, np.sin(np.deg2rad(15.0)), np.cos(np.deg2rad(15.0))]),
        surface_orientation=np.array([0.0, 0.0, 1.0]),
        model="amsa",
    )

    refl = model.refl()
    default_recon = model.albedo(refl)
    x0 = np.full_like(refl, 0.5)[np.newaxis, :]
    x0_recon = model.albedo(refl, x0=x0)

    assert np.all(np.isfinite(x0_recon))
    np.testing.assert_allclose(x0_recon, default_recon, rtol=1e-5, atol=1e-6)
