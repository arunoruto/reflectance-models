import numpy as np

from refmod.mixing import __linear_mixing


def test_random_and_zero():
    rng = np.random.default_rng(42)

    C = 3
    N = 10
    K = 3
    random_albedo = rng.random((N, 1))
    random_coefficients = rng.random((C, N, 1))
    bulk_density = rng.random(K + 1)
    albedo = np.concat(
        (
            random_albedo,
            np.zeros((N, K)),
        ),
        axis=1,
    )
    phase_function_coefficients = np.concat(
        (
            random_coefficients,
            np.zeros((C, N, K)),
        ),
        axis=2,
    )
    mixed_albedo, mixed_phase_function_coefficients = __linear_mixing(
        bulk_density=bulk_density,
        albedo=albedo,
        phase_function_coefficients=phase_function_coefficients,
    )

    mixed_albedo /= bulk_density[0] / np.sum(bulk_density)

    np.testing.assert_allclose(
        mixed_albedo,
        random_albedo,
    )
    np.testing.assert_allclose(
        mixed_phase_function_coefficients,
        random_coefficients,
    )


def test_same_random():
    rng = np.random.default_rng(42)

    C = 3
    N = 10
    random_albedo = rng.random(N)
    random_coefficients = rng.random((C, N))
    albedo = np.stack(
        (
            random_albedo,
            random_albedo,
        ),
        axis=1,
    )
    phase_function_coefficients = np.stack(
        (
            random_coefficients,
            random_coefficients,
        ),
        axis=2,
    )
    mixed_albedo, mixed_phase_function_coefficients = __linear_mixing(
        bulk_density=np.array([1, 1]),
        albedo=albedo,
        phase_function_coefficients=phase_function_coefficients,
    )

    np.testing.assert_allclose(
        mixed_albedo[:, 0],
        random_albedo,
    )
    np.testing.assert_allclose(
        mixed_phase_function_coefficients[:, :, 0],
        random_coefficients,
    )


def test_manual_complex_mixing():
    """
    Manually verify the mixing formulas (Equations 13 and 17) using explicit loops
    to ensure the vectorized implementation handles weights (density, radius, extinction) correctly.
    """
    rng = np.random.default_rng(123)

    # Dimensions
    K = 5  # Endmembers
    N = 4  # Wavelengths
    C = 3  # Legendre Coefficients

    # 1. Generate Random Inputs
    bulk_density = rng.uniform(0.5, 2.0, size=K)
    # Use (K, 1) for explicit property shapes as typically expected for per-endmember properties
    solid_density = rng.uniform(2000, 4000, size=(K, 1))
    radius = rng.uniform(1e-6, 1e-4, size=(K, 1))

    albedo = rng.uniform(0.1, 0.9, size=(N, K))
    extinction_efficiency = rng.uniform(0.5, 2.5, size=(N, K))
    phase_function_coefficients = rng.uniform(-0.5, 0.5, size=(C, N, K))

    # 2. Run the vectorized function
    mixed_albedo, mixed_phase_coeffs = __linear_mixing(
        bulk_density=bulk_density,
        albedo=albedo,
        phase_function_coefficients=phase_function_coefficients,
        extinction_efficiency=extinction_efficiency,
        solid_density=solid_density,
        radius=radius,
    )

    # 3. Calculate Manually
    expected_albedo = np.zeros((N, 1))
    expected_phase_coeffs = np.zeros((C, N, 1))

    # Geometric weighting factor C_j = M_j / (rho_j * r_j)
    # Note: vector function reshapes bulk_density to (K,1) internally if 1D.
    geometric_factors = bulk_density / (solid_density.flatten() * radius.flatten())

    for n in range(N):
        # --- Albedo Mixing (Eq 13) ---
        # w_mix = Sum( C_j * Q_j * w_j ) / Sum( C_j * Q_j )
        numerator_alb = 0.0
        denominator_alb = 0.0

        for k in range(K):
            geo = geometric_factors[k]
            Q = extinction_efficiency[n, k]
            w = albedo[n, k]

            term = geo * Q
            numerator_alb += term * w
            denominator_alb += term

        expected_albedo[n, 0] = numerator_alb / denominator_alb

        # --- Phase Function Mixing (Eq 17) ---
        # p_mix = Sum( C_j * Q_j * w_j * p_j ) / Sum( C_j * Q_j * w_j )
        # Denominator is exactly the numerator of the albedo mixing!
        denominator_phase = numerator_alb

        for c in range(C):
            numerator_phase = 0.0
            for k in range(K):
                geo = geometric_factors[k]
                Q = extinction_efficiency[n, k]
                w = albedo[n, k]
                p_val = phase_function_coefficients[c, n, k]

                # Weighting by scattering cross section (proportional to w * Q * geo)
                term = geo * Q * w
                numerator_phase += term * p_val

            expected_phase_coeffs[c, n, 0] = numerator_phase / denominator_phase

    # 4. Assert
    np.testing.assert_allclose(mixed_albedo, expected_albedo, err_msg="Albedo mismatch")
    np.testing.assert_allclose(
        mixed_phase_coeffs, expected_phase_coeffs, err_msg="Phase Coeff mismatch"
    )
