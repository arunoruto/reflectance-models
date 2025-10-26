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
