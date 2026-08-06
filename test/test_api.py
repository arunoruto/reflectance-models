import jax
import jax.numpy as jnp
import numpy as np
import pytest

from refmod.lunar_lambert import lunar_lambert
from refmod.api import (
    HapkeAmsaParams,
    HapkeCornetteParams,
    HapkeImsaParams,
    amsa_dhg,
    imsa_modified_h_dhg,
    invert_albedo_multi,
    reflectance_image,
    reflectance_gradient_jacobian,
    reflectance_jax,
    reflectance_normal_jacobian,
    reflectance_pq_and_grad_jax,
    reflectance_pq_jax,
)


def test_amsa_params_accept_matlab_aliases_and_nan_backscatter():
    params = HapkeAmsaParams.from_matlab_config(
        {
            "b": 0.21,
            "c": 0.7,
            "h": 0.05,
            "b0": 0.2,
            "tb": 0.1,
            "hc": np.nan,
            "Bc0": np.nan,
        }
    )

    assert params.hs == 0.05
    assert params.Bs0 == 0.2
    assert params.coherent_backscatter == (0.0, 0.0)


def test_reflectance_image_preserves_image_shape_and_zeroes_invalid_geometry():
    w = np.full((2, 2), 0.4)
    s = np.tile(np.array([0.0, 0.0, 1.0]), (2, 2, 1))
    v = np.tile(np.array([0.0, 0.0, 1.0]), (2, 2, 1))
    n = np.tile(np.array([0.0, 0.0, 1.0]), (2, 2, 1))
    s[0, 0] = np.array([0.0, 0.0, -1.0])

    refl = amsa_dhg(w, s, v, n, b=0.21, c=0.7, invalid="zero")

    assert refl.shape == w.shape
    assert refl[0, 0] == 0.0
    assert np.all(np.isfinite(refl))
    assert np.all(refl[1:, :] > 0.0)


def test_reflectance_image_can_return_valid_mask():
    w = np.array([0.4, 0.4])
    s = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    v = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    n = np.array([0.0, 0.0, 1.0])

    refl, valid = reflectance_image(
        "amsa", w, s, v, n, HapkeAmsaParams(0.21, 0.7), invalid="mask"
    )

    assert refl.shape == (2,)
    np.testing.assert_array_equal(valid, [True, False])


def test_lunar_lambert_matches_matlab_zero_phase_identity():
    rho = np.full((2, 2), 0.35)
    result = lunar_lambert(rho, 0.0, 0.0, 0.0)

    np.testing.assert_allclose(np.asarray(result), rho)


def test_lunar_lambert_image_wrapper_uses_vector_geometry():
    rho = np.full((2, 2), 0.35)
    s = np.array([0.0, 0.0, 1.0])
    v = np.array([0.0, 0.0, 1.0])
    n = np.tile(np.array([0.0, 0.0, 1.0]), (2, 2, 1))

    refl = reflectance_image("lunar-lambert", rho, s, v, n, invalid="zero")

    np.testing.assert_allclose(refl, rho)


def test_imsa_modified_h_matches_closed_form_normal_geometry():
    w = np.array([0.4])
    b = 0.21
    c = 0.7
    b0 = 0.3
    s = np.array([[0.0, 0.0, 1.0]])
    v = np.array([[0.0, 0.0, 1.0]])
    n = np.array([[0.0, 0.0, 1.0]])

    refl = imsa_modified_h_dhg(
        w, s, v, n, b=b, c=c, h=0.1, b0=b0, tb=0.0, invalid="nan"
    )

    phase = (
        0.5 * (1.0 + c) * (1.0 - b**2) / (1.0 - 2.0 * b + b**2) ** 1.5
        + 0.5 * (1.0 - c) * (1.0 - b**2) / (1.0 + 2.0 * b + b**2) ** 1.5
    )
    h_value = 3.0 / (1.0 + 2.0 * np.sqrt(1.0 - w[0]))
    expected = w[0] * ((1.0 + b0) * phase + h_value**2 - 1.0) / (8.0 * np.pi)

    np.testing.assert_allclose(refl, [expected], rtol=1e-6)


def test_reflectance_normal_jacobian_matches_finite_difference():
    params = HapkeAmsaParams(b=0.21, c=0.7, hs=0.05, Bs0=0.2)
    w = np.array([0.4])
    s = np.array([[0.0, np.sin(np.deg2rad(30.0)), np.cos(np.deg2rad(30.0))]])
    v = np.array([[0.0, 0.0, 1.0]])
    n = np.array([[0.1, 0.0, np.sqrt(0.99)]])
    eps = 1e-4

    jac = reflectance_normal_jacobian("amsa", w, s, v, n, params, invalid="zero")
    n_plus = n.copy()
    n_minus = n.copy()
    n_plus[0, 0] += eps
    n_minus[0, 0] -= eps
    r_plus = reflectance_image("amsa", w, s, v, n_plus, params, invalid="zero")
    r_minus = reflectance_image("amsa", w, s, v, n_minus, params, invalid="zero")
    finite = (r_plus - r_minus) / (2.0 * eps)

    np.testing.assert_allclose(jac[:, 0], finite, rtol=2e-3, atol=2e-5)


def test_reflectance_gradient_jacobian_shape_and_finiteness():
    params = HapkeImsaParams(b=0.21, c=0.7, h=0.05, b0=0.2)
    w = np.full((2, 2), 0.4)
    p = np.zeros((2, 2))
    q = np.zeros((2, 2))
    s = np.tile(
        np.array([0.0, np.sin(np.deg2rad(30.0)), np.cos(np.deg2rad(30.0))]), (2, 2, 1)
    )
    v = np.tile(np.array([0.0, 0.0, 1.0]), (2, 2, 1))

    jac = reflectance_gradient_jacobian("imsa_modified_h", w, s, v, p, q, params)

    assert jac.shape == (2, 2, 2)
    assert np.all(np.isfinite(jac))


def test_cornette_forward_wrappers_preserve_shape_and_are_finite():
    w = np.full((2, 2), 0.4)
    s = np.tile(
        np.array([0.0, np.sin(np.deg2rad(30.0)), np.cos(np.deg2rad(30.0))]), (2, 2, 1)
    )
    v = np.tile(np.array([0.0, 0.0, 1.0]), (2, 2, 1))
    n = np.tile(np.array([0.0, 0.0, 1.0]), (2, 2, 1))
    params = HapkeCornetteParams(xi=0.2, hs=0.05, Bs0=0.2, tb=0.0)

    amsa_refl = reflectance_image("amsa_cornette", w, s, v, n, params, invalid="zero")
    imsa_refl = reflectance_image("imsa_cornette", w, s, v, n, params, invalid="zero")

    assert amsa_refl.shape == w.shape
    assert imsa_refl.shape == w.shape
    assert np.all(np.isfinite(amsa_refl))
    assert np.all(np.isfinite(imsa_refl))
    assert np.all(amsa_refl > 0.0)
    assert np.all(imsa_refl > 0.0)


def test_invert_albedo_multi_roundtrips_synthetic_amsa_observations():
    w_true = np.array([[0.25, 0.55], [0.35, 0.75]])
    n = np.tile(np.array([0.0, 0.0, 1.0]), (2, 2, 1))
    v = np.stack(
        [
            np.tile(np.array([0.0, 0.0, 1.0]), (2, 2, 1)),
            np.tile(
                np.array([0.0, np.sin(np.deg2rad(10.0)), np.cos(np.deg2rad(10.0))]),
                (2, 2, 1),
            ),
        ]
    )
    s = np.stack(
        [
            np.tile(
                np.array([0.0, np.sin(np.deg2rad(25.0)), np.cos(np.deg2rad(25.0))]),
                (2, 2, 1),
            ),
            np.tile(
                np.array([0.0, np.sin(np.deg2rad(45.0)), np.cos(np.deg2rad(45.0))]),
                (2, 2, 1),
            ),
        ]
    )
    params = HapkeAmsaParams(b=0.21, c=0.7, hs=0.05, Bs0=0.2, tb=0.0)
    reflectance = np.stack(
        [
            reflectance_image("amsa", w_true, s[k], v[k], n, params, invalid="zero")
            for k in range(2)
        ]
    )

    result = invert_albedo_multi(
        reflectance, s, v, n, params, initial_w=0.4, return_info=True
    )

    np.testing.assert_allclose(result.parameters, w_true, rtol=3e-4, atol=1e-5)
    assert np.all(result.converged)


def test_invert_albedo_multi_roundtrips_synthetic_imsa_modified_h_observations():
    w_true = np.array([[0.25, 0.55], [0.35, 0.75]])
    n = np.tile(np.array([0.0, 0.0, 1.0]), (2, 2, 1))
    s = np.stack(
        [
            np.tile(
                np.array([0.0, np.sin(np.deg2rad(25.0)), np.cos(np.deg2rad(25.0))]),
                (2, 2, 1),
            ),
            np.tile(
                np.array([0.0, np.sin(np.deg2rad(45.0)), np.cos(np.deg2rad(45.0))]),
                (2, 2, 1),
            ),
        ]
    )
    v = np.stack([np.tile(np.array([0.0, 0.0, 1.0]), (2, 2, 1)) for _ in range(2)])
    params = HapkeImsaParams(b=0.21, c=0.7, h=0.05, b0=0.2, tb=0.0)
    reflectance = np.stack(
        [
            reflectance_image(
                "imsa_modified_h", w_true, s[k], v[k], n, params, invalid="zero"
            )
            for k in range(2)
        ]
    )

    result = invert_albedo_multi(
        reflectance,
        s,
        v,
        n,
        params,
        initial_w=0.4,
        model="imsa_modified_h",
        return_info=True,
    )

    np.testing.assert_allclose(result.parameters, w_true, rtol=3e-4, atol=1e-5)
    assert np.all(result.converged)


def test_invert_albedo_multi_sigma_smoothing_keeps_constant_field_stable():
    w_true = np.full((4, 4), 0.4)
    n = np.tile(np.array([0.0, 0.0, 1.0]), (4, 4, 1))
    s = np.stack(
        [
            np.tile(
                np.array([0.0, np.sin(np.deg2rad(30.0)), np.cos(np.deg2rad(30.0))]),
                (4, 4, 1),
            )
        ]
    )
    v = np.stack([np.tile(np.array([0.0, 0.0, 1.0]), (4, 4, 1))])
    params = HapkeAmsaParams(b=0.21, c=0.7, hs=0.05, Bs0=0.2, tb=0.0)
    reflectance = np.stack(
        [reflectance_image("amsa", w_true, s[0], v[0], n, params, invalid="zero")]
    )
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 0] = False

    result = invert_albedo_multi(
        reflectance, s, v, n, params, initial_w=0.4, mask=mask, sigma=1.0
    )

    assert result.shape == w_true.shape
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result[mask], w_true[mask], rtol=2e-3, atol=2e-4)


def test_invert_albedo_multi_recovers_lunar_lambert_rho():
    rho_true = np.array([[0.25, 0.55], [0.35, 0.75]])
    n = np.tile(np.array([0.0, 0.0, 1.0]), (2, 2, 1))
    s = np.stack(
        [
            np.tile(
                np.array([0.0, np.sin(np.deg2rad(20.0)), np.cos(np.deg2rad(20.0))]),
                (2, 2, 1),
            ),
            np.tile(
                np.array([0.0, np.sin(np.deg2rad(40.0)), np.cos(np.deg2rad(40.0))]),
                (2, 2, 1),
            ),
        ]
    )
    v = np.stack([np.tile(np.array([0.0, 0.0, 1.0]), (2, 2, 1)) for _ in range(2)])
    reflectance = np.stack(
        [
            reflectance_image("lunar-lambert", rho_true, s[k], v[k], n, invalid="zero")
            for k in range(2)
        ]
    )

    result = invert_albedo_multi(
        reflectance, s, v, n, None, model="lunar-lambert", return_info=True
    )

    np.testing.assert_allclose(result.parameters, rho_true, rtol=1e-6, atol=1e-8)
    assert np.all(result.converged)


def test_refmod_exposes_api_namespace():
    import refmod

    assert refmod.api.HapkeAmsaParams is HapkeAmsaParams


# ---------------------------------------------------------------------------
# Device-resident (``*_jax``) entry points
# ---------------------------------------------------------------------------

# The roughness used by the shape-from-shading configuration this API serves
# (11 degrees), so the tests exercise the ``tb > 0`` path rather than the
# cheaper smooth short-circuit.
_SFS_PARAMS = HapkeAmsaParams(b=0.21, c=0.7, hs=0.11, Bs0=3.1, tb=np.deg2rad(11.0))


def _pq_scene(height=6, width=5, seed=0):
    """A small surface-gradient field with oblique illumination."""
    rng = np.random.default_rng(seed)
    p = rng.normal(0.0, 0.05, (height, width))
    q = rng.normal(0.0, 0.05, (height, width))
    theta = np.deg2rad(35.0)
    s = np.tile(np.array([0.0, np.sin(theta), np.cos(theta)]), (height, width, 1))
    v = np.tile(np.array([0.0, 0.0, 1.0]), (height, width, 1))
    w = np.full((height, width), 0.4)
    return w, s, v, p, q


@pytest.mark.parametrize(
    ("model", "params"),
    [
        ("amsa", _SFS_PARAMS),
        (
            "imsa-modified-h",
            HapkeImsaParams(b=0.21, c=0.7, h=0.11, b0=3.1, tb=np.deg2rad(11.0)),
        ),
        ("lunar-lambert", None),
    ],
)
def test_reflectance_jax_agrees_with_reflectance_image(model, params):
    w, s, v, p, q = _pq_scene()
    norm = np.sqrt(1.0 + p**2 + q**2)
    n = np.stack((-p / norm, -q / norm, 1.0 / norm), axis=-1)

    expected = reflectance_image(model, w, s, v, n, params, invalid="nan")
    actual = reflectance_jax(model, w, s, v, n, params)

    assert isinstance(actual, jax.Array)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-12, atol=0.0)


def test_reflectance_jax_is_jittable_with_params_held_static():
    w, s, v, p, q = _pq_scene()
    norm = np.sqrt(1.0 + p**2 + q**2)
    n = np.stack((-p / norm, -q / norm, 1.0 / norm), axis=-1)

    jitted = jax.jit(reflectance_jax, static_argnums=(0, 5))
    result = jitted(
        "amsa",
        jnp.asarray(w),
        jnp.asarray(s),
        jnp.asarray(v),
        jnp.asarray(n),
        _SFS_PARAMS,
    )

    np.testing.assert_allclose(
        np.asarray(result),
        np.asarray(reflectance_jax("amsa", w, s, v, n, _SFS_PARAMS)),
        rtol=1e-12,
        atol=0.0,
    )


def test_reflectance_pq_and_grad_jax_matches_central_differences():
    w, s, v, p, q = _pq_scene()
    step = 1e-6

    refl, d_dp, d_dq = reflectance_pq_and_grad_jax("amsa", w, s, v, p, q, _SFS_PARAMS)

    def value(p_arr, q_arr):
        return np.asarray(
            reflectance_pq_jax("amsa", w, s, v, p_arr, q_arr, _SFS_PARAMS)
        )

    fd_p = (value(p + step, q) - value(p - step, q)) / (2.0 * step)
    fd_q = (value(p, q + step) - value(p, q - step)) / (2.0 * step)

    np.testing.assert_allclose(np.asarray(refl), value(p, q), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(np.asarray(d_dp), fd_p, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(np.asarray(d_dq), fd_q, rtol=1e-6, atol=1e-9)


def test_reflectance_pq_and_grad_jax_agrees_with_gradient_jacobian():
    w, s, v, p, q = _pq_scene()

    _, d_dp, d_dq = reflectance_pq_and_grad_jax("amsa", w, s, v, p, q, _SFS_PARAMS)
    reference = reflectance_gradient_jacobian("amsa", w, s, v, p, q, _SFS_PARAMS)

    np.testing.assert_allclose(
        np.asarray(d_dp), reference[..., 0], rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(d_dq), reference[..., 1], rtol=1e-12, atol=0.0
    )


def test_reflectance_pq_jax_broadcasts_scalar_albedo_and_shared_geometry():
    _, _, _, p, q = _pq_scene()
    theta = np.deg2rad(35.0)

    refl = reflectance_pq_jax(
        "amsa",
        0.4,
        np.array([0.0, np.sin(theta), np.cos(theta)]),
        np.array([0.0, 0.0, 1.0]),
        p,
        q,
        _SFS_PARAMS,
    )

    assert refl.shape == p.shape
    assert np.all(np.isfinite(np.asarray(refl)))
