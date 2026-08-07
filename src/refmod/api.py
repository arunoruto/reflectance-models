r"""High-level, image-shaped interface to the reflectance models.

The functions in :mod:`refmod.hapke` and friends operate on flat
``(n_pixels, 3)`` geometry and scalar-per-pixel albedo. This module wraps
them for the common application case:

- :class:`HapkeAmsaParams` and siblings collect model parameters and accept
  external configuration dictionaries via ``from_dict``.
- :func:`reflectance_image` evaluates a model while preserving
  ``(height, width)`` image shape and offers explicit invalid-geometry
  handling.
- :func:`reflectance_normal_jacobian` and
  :func:`reflectance_gradient_jacobian` give per-pixel derivatives with
  respect to the surface normal and to the surface gradients ``(p, q)``.
- :func:`invert_albedo_multi` retrieves one albedo per pixel from one or
  more observations of the same scene.

Every function above returns NumPy and is meant to be called a handful of
times. Iterative callers -- shape-from-shading in particular, which evaluates
a model and its gradients once per iteration for hundreds of iterations --
need the opposite: results that stay on the device so the whole loop can live
inside a single :func:`jax.jit`. That is what the ``*_jax`` family provides:

- :func:`reflectance_jax` is :func:`reflectance_image` without the NumPy
  round-trip.
- :func:`reflectance_pq_jax` parametrises the surface by its gradients
  ``(p, q)`` instead of by the normal.
- :func:`reflectance_pq_and_grad_jax` returns reflectance together with
  ``dR/dp`` and ``dR/dq`` at the same point, which is the inner-loop
  primitive for shape-from-shading.

These are ordinary JAX functions: traceable, jittable, differentiable, and
usable under ``vmap``. They return ``jax.Array``; wrap a call in
``np.asarray`` if you want NumPy back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter

from refmod.hapke import (
    amsa,
    amsa_cornette,
    dhg_legendre_coefficients,
    imsa,
    imsa_cornette,
    imsa_modified_h,
)
from refmod.lunar_lambert import lunar_lambert

InvalidMode = Literal["nan", "zero", "mask"]
ModelName = Literal[
    "amsa",
    "imsa",
    "imsa-modified-h",
    "imsa_modified_h",
    "imsa-matlab",
    "amsa-cornette",
    "amsa_cornette",
    "imsa-cornette",
    "imsa_cornette",
    "ll",
    "lunar-lambert",
    "lunar_lambert",
]


@dataclass(frozen=True)
class HapkeAmsaParams:
    b: float
    c: float
    hs: float = 0.0
    Bs0: float = 0.0
    tb: float = 0.0
    hc: float | None = None
    Bc0: float | None = None
    n_order: int = 15

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "HapkeAmsaParams":
        return cls(
            b=float(config["b"]),
            c=float(config["c"]),
            hs=float(config.get("hs", config.get("h", 0.0))),
            Bs0=float(config.get("Bs0", config.get("b0", 0.0))),
            tb=float(config.get("tb", 0.0)),
            hc=_optional_float(config.get("hc")),
            Bc0=_optional_float(config.get("Bc0")),
            n_order=int(config.get("n_order", 15)),
        )

    from_matlab_config = from_dict

    @property
    def legendre_coefficients(self) -> jax.Array:
        """DHG Legendre coefficients using refmod's public ``c`` convention.

        Pass external configuration ``c`` values directly to this object.

        The MATLAB reference implementation used to negate ``c`` into an
        internal variable (``c_hapke = -c``) and simultaneously swap the signs
        in the two Henyey-Greenstein denominators. Those two flips cancelled,
        so the value carried in configuration files equalled the ``c`` used
        here and needed no conversion.

        As of the toolbox's 2.0.0 release that double negation is gone: its
        ``hapke_phase_dhg.m`` takes ``c`` in this same convention directly, so
        the two sources now agree by inspection rather than by two errors
        cancelling. The conclusion for callers is unchanged -- no conversion.
        """
        return dhg_legendre_coefficients(self.b, self.c, self.n_order)

    @property
    def coherent_backscatter(self) -> tuple[float, float]:
        if self.hc is None or self.Bc0 is None:
            return 0.0, 0.0
        if np.isnan(self.hc) or np.isnan(self.Bc0):
            return 0.0, 0.0
        return self.hc, self.Bc0


@dataclass(frozen=True)
class HapkeImsaParams:
    b: float
    c: float
    h: float = 0.0
    b0: float = 0.0
    tb: float = 0.0
    n_order: int = 15

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "HapkeImsaParams":
        return cls(
            b=float(config["b"]),
            c=float(config["c"]),
            h=float(config.get("h", 0.0)),
            b0=float(config.get("b0", 0.0)),
            tb=float(config.get("tb", 0.0)),
            n_order=int(config.get("n_order", 15)),
        )

    from_matlab_config = from_dict

    @property
    def legendre_coefficients(self) -> jax.Array:
        return dhg_legendre_coefficients(self.b, self.c, self.n_order)


@dataclass(frozen=True)
class HapkeCornetteParams:
    xi: float
    hs: float = 0.0
    Bs0: float = 0.0
    tb: float = 0.0
    hc: float | None = None
    Bc0: float | None = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "HapkeCornetteParams":
        return cls(
            xi=float(config["xi"]),
            hs=float(config.get("hs", config.get("h", 0.0))),
            Bs0=float(config.get("Bs0", config.get("b0", 0.0))),
            tb=float(config.get("tb", 0.0)),
            hc=_optional_float(config.get("hc")),
            Bc0=_optional_float(config.get("Bc0")),
        )

    from_matlab_config = from_dict

    @property
    def coherent_backscatter(self) -> tuple[float, float]:
        if self.hc is None or self.Bc0 is None:
            return 0.0, 0.0
        if np.isnan(self.hc) or np.isnan(self.Bc0):
            return 0.0, 0.0
        return self.hc, self.Bc0


@dataclass(frozen=True)
class LunarLambertParams:
    @classmethod
    def from_dict(cls, config: dict[str, Any] | None = None) -> "LunarLambertParams":
        return cls()

    from_matlab_config = from_dict


@dataclass(frozen=True)
class MultiImageInversionResult:
    parameters: npt.NDArray
    residuals: npt.NDArray
    converged: npt.NDArray[np.bool_]
    iterations: npt.NDArray[np.int_]


def amsa_dhg(
    w: npt.ArrayLike,
    s: npt.ArrayLike,
    v: npt.ArrayLike,
    n: npt.ArrayLike,
    b: float,
    c: float,
    hs: float = 0.0,
    Bs0: float = 0.0,
    tb: float = 0.0,
    hc: float | None = None,
    Bc0: float | None = None,
    invalid: InvalidMode = "zero",
) -> npt.NDArray | tuple[npt.NDArray, npt.NDArray[np.bool_]]:
    params = HapkeAmsaParams(b=b, c=c, hs=hs, Bs0=Bs0, tb=tb, hc=hc, Bc0=Bc0)
    return reflectance_image("amsa", w, s, v, n, params, invalid=invalid)


def imsa_modified_h_dhg(
    w: npt.ArrayLike,
    s: npt.ArrayLike,
    v: npt.ArrayLike,
    n: npt.ArrayLike,
    b: float,
    c: float,
    h: float = 0.0,
    b0: float = 0.0,
    tb: float = 0.0,
    invalid: InvalidMode = "zero",
) -> npt.NDArray | tuple[npt.NDArray, npt.NDArray[np.bool_]]:
    params = HapkeImsaParams(b=b, c=c, h=h, b0=b0, tb=tb)
    return reflectance_image("imsa_modified_h", w, s, v, n, params, invalid=invalid)


def reflectance_image(
    model: ModelName,
    w: npt.ArrayLike,
    s: npt.ArrayLike,
    v: npt.ArrayLike,
    n: npt.ArrayLike,
    params: HapkeAmsaParams
    | HapkeImsaParams
    | HapkeCornetteParams
    | LunarLambertParams
    | dict[str, Any]
    | None = None,
    invalid: InvalidMode = "nan",
) -> npt.NDArray | tuple[npt.NDArray, npt.NDArray[np.bool_]]:
    r"""Evaluate a reflectance model and preserve image-shaped inputs.

    Geometry can be supplied as ``(..., 3)``, ``(3, ...)``, ``(pixels, 3)``,
    or a single 3-vector. Scalar albedo is broadcast to the geometry shape.
    """
    model = _normalize_model(model)
    if invalid not in ("nan", "zero", "mask"):
        raise ValueError("invalid must be one of 'nan', 'zero', or 'mask'")

    s_flat, s_shape = _vectors_to_flat(s)
    v_flat, v_shape = _vectors_to_flat(v)
    n_flat, n_shape = _vectors_to_flat(n)
    out_shape = _infer_output_shape(np.asarray(w), s_shape, v_shape, n_shape)
    n_pixels = int(np.prod(out_shape)) if out_shape else 1

    s_flat = _broadcast_vectors(s_flat, n_pixels, "s")
    v_flat = _broadcast_vectors(v_flat, n_pixels, "v")
    n_flat = _broadcast_vectors(n_flat, n_pixels, "n")
    w_flat = _field_to_flat(w, out_shape, n_pixels, "w")
    valid = _valid_geometry(s_flat, v_flat, n_flat)

    refl = _reflectance_flat_jax(model, w_flat, s_flat, v_flat, n_flat, params)
    return _format_invalid(refl, valid, out_shape, invalid)


def reflectance_normal_jacobian(
    model: ModelName,
    w: npt.ArrayLike,
    s: npt.ArrayLike,
    v: npt.ArrayLike,
    n: npt.ArrayLike,
    params: HapkeAmsaParams
    | HapkeImsaParams
    | HapkeCornetteParams
    | LunarLambertParams
    | dict[str, Any]
    | None = None,
    invalid: InvalidMode = "zero",
) -> npt.NDArray | tuple[npt.NDArray, npt.NDArray[np.bool_]]:
    r"""Return per-pixel reflectance derivatives with respect to normal vectors."""
    model = _normalize_model(model)
    s_flat, s_shape = _vectors_to_flat(s)
    v_flat, v_shape = _vectors_to_flat(v)
    n_flat, n_shape = _vectors_to_flat(n)
    out_shape = _infer_output_shape(np.asarray(w), s_shape, v_shape, n_shape)
    n_pixels = int(np.prod(out_shape)) if out_shape else 1
    s_flat = _broadcast_vectors(s_flat, n_pixels, "s")
    v_flat = _broadcast_vectors(v_flat, n_pixels, "v")
    n_flat = _broadcast_vectors(n_flat, n_pixels, "n")
    w_flat = _field_to_flat(w, out_shape, n_pixels, "w")
    valid = _valid_geometry(s_flat, v_flat, n_flat)
    prepared = _prepare_params(model, params)

    def one(w_value, s_value, v_value, n_value):
        return _reflectance_one_jax(model, w_value, s_value, v_value, n_value, prepared)

    jac = jax.vmap(jax.jacfwd(one, argnums=3))(
        jnp.asarray(w_flat),
        jnp.asarray(s_flat),
        jnp.asarray(v_flat),
        jnp.asarray(n_flat),
    )
    return _format_invalid(jac, valid[:, None], out_shape + (3,), invalid)


def reflectance_gradient_jacobian(
    model: ModelName,
    w: npt.ArrayLike,
    s: npt.ArrayLike,
    v: npt.ArrayLike,
    p: npt.ArrayLike,
    q: npt.ArrayLike,
    params: HapkeAmsaParams
    | HapkeImsaParams
    | HapkeCornetteParams
    | LunarLambertParams
    | dict[str, Any]
    | None = None,
    invalid: InvalidMode = "zero",
) -> npt.NDArray | tuple[npt.NDArray, npt.NDArray[np.bool_]]:
    r"""Return derivatives with respect to surface gradients ``p`` and ``q``.

    Uses the surface-gradient parametrisation
    ``n = normalize([-p, -q, 1])``, i.e. ``p = -n_x/n_z``, ``q = -n_y/n_z``.
    """
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    out_shape = np.broadcast_shapes(
        np.asarray(w).shape if np.asarray(w).size != 1 else (), p_arr.shape, q_arr.shape
    )
    n_pixels = int(np.prod(out_shape)) if out_shape else 1
    p_flat = _field_to_flat(np.broadcast_to(p_arr, out_shape), out_shape, n_pixels, "p")
    q_flat = _field_to_flat(np.broadcast_to(q_arr, out_shape), out_shape, n_pixels, "q")
    n_flat = _gradient_normals(p_flat, q_flat)
    s_flat, _ = _vectors_to_flat(s)
    v_flat, _ = _vectors_to_flat(v)
    s_flat = _broadcast_vectors(s_flat, n_pixels, "s")
    v_flat = _broadcast_vectors(v_flat, n_pixels, "v")
    w_flat = _field_to_flat(w, out_shape, n_pixels, "w")
    valid = _valid_geometry(s_flat, v_flat, n_flat)
    model = _normalize_model(model)
    prepared = _prepare_params(model, params)

    def one(w_value, s_value, v_value, gradients):
        p_value, q_value = gradients
        n_value = _normalize_jnp(jnp.array([-p_value, -q_value, 1.0]))
        return _reflectance_one_jax(model, w_value, s_value, v_value, n_value, prepared)

    gradients = jnp.stack((jnp.asarray(p_flat), jnp.asarray(q_flat)), axis=-1)
    jac = jax.vmap(jax.jacfwd(one, argnums=3))(
        jnp.asarray(w_flat), jnp.asarray(s_flat), jnp.asarray(v_flat), gradients
    )
    return _format_invalid(jac, valid[:, None], out_shape + (2,), invalid)


def reflectance_jax(
    model: ModelName,
    w: jax.Array,
    s: jax.Array,
    v: jax.Array,
    n: jax.Array,
    params: HapkeAmsaParams
    | HapkeImsaParams
    | HapkeCornetteParams
    | LunarLambertParams
    | dict[str, Any]
    | None = None,
) -> jax.Array:
    r"""Evaluate a reflectance model without leaving the device.

    The device-resident counterpart of :func:`reflectance_image`. Traceable,
    jittable and differentiable, so an iterative caller can keep an entire
    solver loop inside one :func:`jax.jit` instead of paying a host round-trip
    per evaluation.

    Unlike :func:`reflectance_image` this does not accept ``(3, ...)`` geometry
    and does not offer ``invalid=`` handling: geometry must be ``(..., 3)``,
    and invalid geometry yields NaN, which callers mask themselves. Both
    restrictions keep the traced graph free of host-side branching.

    Parameters
    ----------
    model : ModelName
        Reflectance model to evaluate.
    w : jax.Array
        Single-scattering albedo, broadcast to the batch shape.
    s, v, n : jax.Array
        Incidence, emission and surface-normal vectors, shape ``(..., 3)``.
        ``s`` and ``v`` broadcast against the batch shape of ``n``.
    params : HapkeAmsaParams or HapkeImsaParams or HapkeCornetteParams or LunarLambertParams or dict, optional
        Model parameters. The frozen parameter dataclasses are hashable and so
        may be passed as ``static_argnums`` of an enclosing ``jit``.

    Returns
    -------
    jax.Array
        Reflectance with the batch shape of ``n``, i.e. ``n.shape[:-1]``.

    See Also
    --------
    reflectance_image : NumPy-returning equivalent with invalid-geometry modes.
    reflectance_pq_and_grad_jax : Fused value and surface-gradient derivatives.
    """
    resolved = _normalize_model(model)
    prepared = _prepare_params(resolved, params)
    n_j = jnp.asarray(n)
    batch = n_j.shape[:-1]
    refl = _reflectance_flat_jax(
        resolved,
        jnp.reshape(jnp.broadcast_to(jnp.asarray(w), batch), (-1,)),
        jnp.reshape(jnp.broadcast_to(jnp.asarray(s), (*batch, 3)), (-1, 3)),
        jnp.reshape(jnp.broadcast_to(jnp.asarray(v), (*batch, 3)), (-1, 3)),
        jnp.reshape(n_j, (-1, 3)),
        prepared,
    )
    return jnp.reshape(refl, batch)


def _normals_from_pq(p: jax.Array, q: jax.Array) -> jax.Array:
    r"""Surface normals from gradients, ``n = normalize([-p, -q, 1])``."""
    norm = jnp.sqrt(1.0 + p**2 + q**2)
    return jnp.stack((-p / norm, -q / norm, jnp.ones_like(p) / norm), axis=-1)


def reflectance_pq_jax(
    model: ModelName,
    w: jax.Array,
    s: jax.Array,
    v: jax.Array,
    p: jax.Array,
    q: jax.Array,
    params: HapkeAmsaParams
    | HapkeImsaParams
    | HapkeCornetteParams
    | LunarLambertParams
    | dict[str, Any]
    | None = None,
) -> jax.Array:
    r"""Evaluate a reflectance model as a function of surface gradients.

    As :func:`reflectance_jax`, but the surface is given by its gradients
    under the shape-from-shading parametrisation ``n = normalize([-p, -q, 1])``
    rather than by an explicit normal.

    Parameters
    ----------
    model : ModelName
        Reflectance model to evaluate.
    w : jax.Array
        Single-scattering albedo, broadcast to the batch shape.
    s, v : jax.Array
        Incidence and emission vectors, shape ``(..., 3)``.
    p, q : jax.Array
        Surface gradients, broadcast against each other to give the batch shape.
    params : HapkeAmsaParams or HapkeImsaParams or HapkeCornetteParams or LunarLambertParams or dict, optional
        Model parameters.

    Returns
    -------
    jax.Array
        Reflectance with the broadcast batch shape of ``p`` and ``q``.
    """
    p_j, q_j = jnp.broadcast_arrays(jnp.asarray(p), jnp.asarray(q))
    return reflectance_jax(model, w, s, v, _normals_from_pq(p_j, q_j), params)


def reflectance_pq_and_grad_jax(
    model: ModelName,
    w: jax.Array,
    s: jax.Array,
    v: jax.Array,
    p: jax.Array,
    q: jax.Array,
    params: HapkeAmsaParams
    | HapkeImsaParams
    | HapkeCornetteParams
    | LunarLambertParams
    | dict[str, Any]
    | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Reflectance and its surface-gradient derivatives at the same point.

    Returns ``R``, ``dR/dp`` and ``dR/dq`` under the shape-from-shading
    parametrisation ``n = normalize([-p, -q, 1])``. This is the inner-loop
    primitive of a shape-from-shading iteration, which needs the model value
    and both partials evaluated at one surface estimate.

    The derivatives come from forward-mode autodiff, and are exact. The MATLAB
    reference this package was ported from used central differences with a
    fixed step of ``1e-6``, costing four extra model evaluations per image per
    iteration; two ``jvp`` passes are both cheaper and free of truncation
    error.

    Because reflectance is elementwise in ``(p, q)``, a tangent of ones
    recovers the per-pixel partial directly -- no Jacobian is ever
    materialised, and this stays O(1) in memory per pixel rather than the
    O(n_pixels) a naive ``jacfwd`` over the whole field would need.

    Parameters
    ----------
    model : ModelName
        Reflectance model to evaluate.
    w : jax.Array
        Single-scattering albedo, broadcast to the batch shape.
    s, v : jax.Array
        Incidence and emission vectors, shape ``(..., 3)``.
    p, q : jax.Array
        Surface gradients, broadcast against each other to give the batch shape.
    params : HapkeAmsaParams or HapkeImsaParams or HapkeCornetteParams or LunarLambertParams or dict, optional
        Model parameters.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array]
        ``(reflectance, dR/dp, dR/dq)``, each with the broadcast batch shape
        of ``p`` and ``q``.

    See Also
    --------
    reflectance_gradient_jacobian : NumPy-returning equivalent, value excluded.
    """
    p_j, q_j = jnp.broadcast_arrays(jnp.asarray(p), jnp.asarray(q))
    ones = jnp.ones_like(p_j)

    refl, d_dp = jax.jvp(
        lambda arg: reflectance_pq_jax(model, w, s, v, arg, q_j, params),
        (p_j,),
        (ones,),
    )
    _, d_dq = jax.jvp(
        lambda arg: reflectance_pq_jax(model, w, s, v, p_j, arg, params),
        (q_j,),
        (ones,),
    )
    return refl, d_dp, d_dq


def _reflectance_flat_jax(
    model: str,
    w_flat: np.ndarray | jax.Array,
    s_flat: np.ndarray | jax.Array,
    v_flat: np.ndarray | jax.Array,
    n_flat: np.ndarray | jax.Array,
    params: HapkeAmsaParams
    | HapkeImsaParams
    | HapkeCornetteParams
    | LunarLambertParams
    | dict[str, Any]
    | None,
) -> jax.Array:
    params = _prepare_params(model, params)
    w_j = jnp.asarray(w_flat)
    s_j = jnp.asarray(s_flat)
    v_j = jnp.asarray(v_flat)
    n_j = jnp.asarray(n_flat)

    if model == "amsa":
        if params is None:
            raise ValueError("AMSA requires HapkeAmsaParams or a config dict")
        h_cb, b0_cb = params.coherent_backscatter
        return amsa(
            w_j,
            params.legendre_coefficients,
            s_j,
            v_j,
            n_j,
            params.tb,
            params.hs,
            params.Bs0,
            h_cb,
            b0_cb,
        )
    elif model == "imsa":
        if params is None:
            raise ValueError("IMSA requires HapkeImsaParams or a config dict")
        return imsa(
            w_j,
            params.legendre_coefficients,
            s_j,
            v_j,
            n_j,
            params.tb,
        )
    elif model == "imsa-modified-h":
        if params is None:
            raise ValueError(
                "IMSA modified-H requires HapkeImsaParams or a config dict"
            )
        return imsa_modified_h(
            w_j, params.b, params.c, s_j, v_j, n_j, params.tb, params.h, params.b0
        )
    elif model == "amsa-cornette":
        if params is None:
            raise ValueError(
                "AMSA Cornette requires HapkeCornetteParams or a config dict"
            )
        h_cb, b0_cb = params.coherent_backscatter
        return amsa_cornette(
            w_j, params.xi, s_j, v_j, n_j, params.tb, params.hs, params.Bs0, h_cb, b0_cb
        )
    elif model == "imsa-cornette":
        if params is None:
            raise ValueError(
                "IMSA Cornette requires HapkeCornetteParams or a config dict"
            )
        return imsa_cornette(
            w_j, params.xi, s_j, v_j, n_j, params.tb, params.hs, params.Bs0
        )
    else:
        th_i, th_e, alpha = _geometry_angles_jax(s_j, v_j, n_j)
        return lunar_lambert(w_j, th_i, th_e, alpha)


def _reflectance_one_jax(
    model: str,
    w_value: jax.Array,
    s_value: jax.Array,
    v_value: jax.Array,
    n_value: jax.Array,
    params: HapkeAmsaParams
    | HapkeImsaParams
    | HapkeCornetteParams
    | LunarLambertParams,
) -> jax.Array:
    return _reflectance_flat_jax(
        model,
        jnp.asarray([w_value]),
        jnp.asarray([s_value]),
        jnp.asarray([v_value]),
        jnp.asarray([n_value]),
        params,
    )[0]


def _prepare_params(
    model: str,
    params: HapkeAmsaParams
    | HapkeImsaParams
    | HapkeCornetteParams
    | LunarLambertParams
    | dict[str, Any]
    | None,
) -> HapkeAmsaParams | HapkeImsaParams | HapkeCornetteParams | LunarLambertParams:
    if model == "amsa":
        if isinstance(params, dict):
            return HapkeAmsaParams.from_dict(params)
        if isinstance(params, HapkeAmsaParams):
            return params
        raise TypeError("AMSA requires HapkeAmsaParams")
    if model in ("imsa", "imsa-modified-h"):
        if isinstance(params, dict):
            return HapkeImsaParams.from_dict(params)
        if isinstance(params, HapkeImsaParams):
            return params
        raise TypeError("IMSA requires HapkeImsaParams")
    if model in ("amsa-cornette", "imsa-cornette"):
        if isinstance(params, dict):
            return HapkeCornetteParams.from_dict(params)
        if isinstance(params, HapkeCornetteParams):
            return params
        raise TypeError("Cornette models require HapkeCornetteParams")
    if params is None or isinstance(params, dict):
        return LunarLambertParams.from_dict(params)
    if isinstance(params, LunarLambertParams):
        return params
    raise TypeError("lunar-Lambert requires LunarLambertParams or no params")


def invert_albedo_multi(
    reflectance: npt.ArrayLike,
    s: npt.ArrayLike,
    v: npt.ArrayLike,
    n: npt.ArrayLike,
    params: HapkeAmsaParams
    | HapkeImsaParams
    | LunarLambertParams
    | dict[str, Any]
    | None,
    initial_w: npt.ArrayLike | float = 1.0 / 3.0,
    mask: npt.ArrayLike | None = None,
    model: Literal[
        "amsa",
        "imsa_modified_h",
        "imsa-modified-h",
        "imsa-matlab",
        "ll",
        "lunar-lambert",
        "lunar_lambert",
    ] = "amsa",
    sigma: float = 0.0,
    max_steps: int = 40,
    return_info: bool = False,
) -> npt.NDArray | MultiImageInversionResult:
    r"""Estimate one albedo value per pixel from one or more images.

    ``sigma=0`` runs pixelwise. ``sigma>0`` applies local-area
    preconditioning by smoothing observations and geometry with mask-aware
    Gaussian filters before the pixelwise solve, which stabilises the
    estimate where single-pixel observations are poorly conditioned.
    """
    model = _normalize_model(model)
    if model not in ("amsa", "imsa-modified-h", "lunar-lambert"):
        raise NotImplementedError(
            "invert_albedo_multi supports 'amsa', 'imsa_modified_h', and 'lunar-lambert'"
        )
    params = _prepare_params(model, params)

    reflectance_arr = np.asarray(reflectance, dtype=float)
    if reflectance_arr.ndim < 2:
        raise ValueError(
            f"reflectance must have shape (n_images, ...), got {reflectance_arr.shape}"
        )
    out_shape = reflectance_arr.shape[1:]
    active_stack = _active_mask_stack(reflectance_arr, mask)
    if sigma > 0:
        reflectance_arr, s, v, n, active_stack = _smooth_inversion_inputs(
            reflectance_arr, s, v, n, active_stack, float(sigma)
        )

    refl_flat, out_shape = _image_stack_to_flat(reflectance_arr, "reflectance")
    n_images, n_pixels = refl_flat.shape
    s_per_image = _as_per_image_vectors(s, n_images)
    v_per_image = _as_per_image_vectors(v, n_images)
    s_flat = _geometry_stack_to_flat(s, n_images, n_pixels, "s")
    v_flat = _geometry_stack_to_flat(v, n_images, n_pixels, "v")
    n_flat, n_shape = _vectors_to_flat(n)
    n_flat = _broadcast_vectors(n_flat, n_pixels, "n")
    w0_flat = _field_to_flat(initial_w, out_shape, n_pixels, "initial_w")

    active = active_stack.reshape(n_images, n_pixels) & np.isfinite(refl_flat)
    # Validate per image: avoids materialising (n_images * n_pixels, 3)
    # copies of the geometry (s_flat/v_flat are usually broadcast views).
    for k in range(n_images):
        active[k] &= _valid_geometry(s_flat[k], v_flat[k], n_flat)

    if model == "lunar-lambert":
        return _invert_lunar_lambert_multi(
            refl_flat, s_flat, v_flat, n_flat, active, out_shape, return_info
        )

    # When sun/view are constant per image, share the (n_images, 3) arrays
    # across all pixels (in_axes=None). This avoids materialising
    # (n_pixels, n_images, 3) geometry on host and device.
    shared_geometry = s_per_image is not None and v_per_image is not None
    if shared_geometry:
        s_dev = jnp.asarray(s_per_image)
        v_dev = jnp.asarray(v_per_image)
    else:
        s_dev = jnp.asarray(s_flat.transpose(1, 0, 2))
        v_dev = jnp.asarray(v_flat.transpose(1, 0, 2))

    if model == "amsa":
        h_cb, b0_cb = params.coherent_backscatter
        solver = (
            _invert_multi_amsa_shared_jit if shared_geometry else _invert_multi_amsa_jit
        )
        model_params = (
            params.legendre_coefficients,
            float(params.tb),
            float(params.hs),
            float(params.Bs0),
            float(h_cb),
            float(b0_cb),
        )
    else:
        solver = (
            _invert_multi_imsa_modified_h_shared_jit
            if shared_geometry
            else _invert_multi_imsa_modified_h_jit
        )
        model_params = (
            float(params.b),
            float(params.c),
            float(params.tb),
            float(params.h),
            float(params.b0),
        )

    w_sol, residuals, converged, iterations = solver(
        jnp.asarray(refl_flat.T),
        s_dev,
        v_dev,
        jnp.asarray(n_flat),
        jnp.asarray(active.T),
        jnp.asarray(w0_flat),
        *model_params,
        int(max_steps),
    )

    w_np = np.asarray(w_sol).reshape(out_shape)
    residuals_np = np.asarray(residuals).reshape(out_shape)
    if not return_info:
        return w_np
    return MultiImageInversionResult(
        parameters=w_np,
        residuals=residuals_np,
        converged=np.asarray(converged).reshape(out_shape),
        iterations=np.asarray(iterations).reshape(out_shape),
    )


def _invert_lunar_lambert_multi(
    refl_flat: np.ndarray,
    s_flat: np.ndarray,
    v_flat: np.ndarray,
    n_flat: np.ndarray,
    active: np.ndarray,
    out_shape: tuple[int, ...],
    return_info: bool,
) -> npt.NDArray | MultiImageInversionResult:
    n_images, n_pixels = refl_flat.shape
    n_stack = np.broadcast_to(n_flat[None, :, :], (n_images, n_pixels, 3))
    th_i, th_e, alpha = _geometry_angles_jax(
        jnp.asarray(s_flat), jnp.asarray(v_flat), jnp.asarray(n_stack)
    )
    factors = np.asarray(lunar_lambert(jnp.ones_like(th_i), th_i, th_e, alpha))
    weights = active & np.isfinite(factors) & np.isfinite(refl_flat)
    numerator = np.sum(np.where(weights, refl_flat * factors, 0.0), axis=0)
    denominator = np.sum(np.where(weights, factors**2, 0.0), axis=0)
    rho = np.where(denominator > 0.0, numerator / denominator, np.nan)
    model_refl = rho[None, :] * factors
    residuals = 0.5 * np.sum(
        np.where(weights, (model_refl - refl_flat) ** 2, 0.0), axis=0
    )
    rho_image = rho.reshape(out_shape)
    if not return_info:
        return rho_image
    return MultiImageInversionResult(
        parameters=rho_image,
        residuals=residuals.reshape(out_shape),
        converged=(denominator > 0.0).reshape(out_shape),
        iterations=np.zeros(out_shape, dtype=int),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if np.isnan(value) else value


def _normalize_model(model: str) -> str:
    model = model.lower().replace("_", "-")
    if model in ("ll", "lunar-lambert"):
        return "lunar-lambert"
    if model in ("imsa-modified-h", "imsa-matlab"):
        return "imsa-modified-h"
    if model in ("amsa-cornette",):
        return "amsa-cornette"
    if model in ("imsa-cornette",):
        return "imsa-cornette"
    if model in ("amsa", "imsa"):
        return model
    raise ValueError(f"Unsupported reflectance model: {model}")


def _vectors_to_flat(v: npt.ArrayLike) -> tuple[np.ndarray, tuple[int, ...]]:
    arr = np.asarray(v, dtype=float)
    if arr.ndim == 1 and arr.shape[0] == 3:
        return arr.reshape(1, 3), ()
    if arr.ndim >= 2 and arr.shape[-1] == 3:
        return np.ascontiguousarray(arr.reshape(-1, 3)), arr.shape[:-1]
    if arr.ndim >= 2 and arr.shape[0] == 3:
        moved = np.moveaxis(arr, 0, -1)
        return np.ascontiguousarray(moved.reshape(-1, 3)), moved.shape[:-1]
    raise ValueError(
        f"Expected vector array with a length-3 axis, got shape {arr.shape}"
    )


def _infer_output_shape(
    w: np.ndarray, *geometry_shapes: tuple[int, ...]
) -> tuple[int, ...]:
    if w.ndim > 0 and w.size != 1:
        return w.shape
    for shape in geometry_shapes:
        if shape:
            return shape
    return ()


def _broadcast_vectors(v: np.ndarray, n_pixels: int, name: str) -> np.ndarray:
    if v.shape[0] == n_pixels:
        return v
    if v.shape[0] == 1:
        return np.broadcast_to(v, (n_pixels, 3))
    raise ValueError(f"{name} has {v.shape[0]} vectors, expected 1 or {n_pixels}")


def _field_to_flat(
    field: npt.ArrayLike, out_shape: tuple[int, ...], n_pixels: int, name: str
) -> np.ndarray:
    arr = np.asarray(field, dtype=float)
    if arr.size == 1:
        return np.full(n_pixels, float(arr.reshape(-1)[0]))
    if arr.shape == out_shape:
        return np.ascontiguousarray(arr.reshape(-1))
    if arr.size == n_pixels:
        return np.ascontiguousarray(arr.reshape(-1))
    raise ValueError(
        f"{name} has shape {arr.shape}, expected scalar, {out_shape}, or {n_pixels} values"
    )


def _valid_geometry(s: np.ndarray, v: np.ndarray, n: np.ndarray) -> np.ndarray:
    s = _normalize_np(s)
    v = _normalize_np(v)
    n = _normalize_np(n)
    mu0 = np.sum(s * n, axis=-1)
    mu = np.sum(v * n, axis=-1)
    return (mu0 > 0.0) & (mu > 0.0)


def _normalize_np(v: np.ndarray) -> np.ndarray:
    norm = np.maximum(np.sqrt(np.sum(v**2, axis=-1, keepdims=True)), 1e-12)
    return v / norm


def _geometry_angles(
    s: np.ndarray, v: np.ndarray, n: np.ndarray
) -> tuple[jax.Array, jax.Array, jax.Array]:
    # Deprecated: unused NumPy convenience wrapper around
    # _geometry_angles_jax; kept for potential external callers.
    return _geometry_angles_jax(jnp.asarray(s), jnp.asarray(v), jnp.asarray(n))


def _geometry_angles_jax(
    s: jax.Array, v: jax.Array, n: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    s_j = _normalize_jnp(s)
    v_j = _normalize_jnp(v)
    n_j = _normalize_jnp(n)
    th_i = jnp.arccos(jnp.clip(jnp.sum(s_j * n_j, axis=-1), -1.0, 1.0))
    th_e = jnp.arccos(jnp.clip(jnp.sum(v_j * n_j, axis=-1), -1.0, 1.0))
    alpha = jnp.arccos(jnp.clip(jnp.sum(s_j * v_j, axis=-1), -1.0, 1.0))
    return th_i, th_e, alpha


def _gradient_normals(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    normals = np.stack((-p, -q, np.ones_like(p)), axis=-1)
    return _normalize_np(normals)


def _normalize_jnp(v: jax.Array) -> jax.Array:
    norm = jnp.maximum(jnp.sqrt(jnp.sum(v**2, axis=-1, keepdims=True)), 1e-12)
    return v / norm


def _format_invalid(
    refl: jax.Array,
    valid: np.ndarray,
    out_shape: tuple[int, ...],
    invalid: InvalidMode,
) -> npt.NDArray | tuple[npt.NDArray, npt.NDArray[np.bool_]]:
    valid_j = jnp.asarray(valid)
    if invalid == "zero":
        out = jnp.where(valid_j & jnp.isfinite(refl), refl, 0.0)
    else:
        out = jnp.where(valid_j, refl, jnp.nan)
    out_np = np.asarray(out).reshape(out_shape)
    if invalid == "mask":
        return out_np, np.asarray(valid).reshape(out_shape)
    return out_np


def _image_stack_to_flat(
    stack: npt.ArrayLike, name: str
) -> tuple[np.ndarray, tuple[int, ...]]:
    arr = np.asarray(stack, dtype=float)
    if arr.ndim < 2:
        raise ValueError(f"{name} must have shape (n_images, ...), got {arr.shape}")
    return np.ascontiguousarray(arr.reshape(arr.shape[0], -1)), arr.shape[1:]


def _as_per_image_vectors(geometry: npt.ArrayLike, n_images: int) -> np.ndarray | None:
    """Return geometry as ``(n_images, 3)`` when it is constant per image."""
    arr = np.asarray(geometry, dtype=float)
    if arr.ndim == 1 and arr.shape == (3,):
        return np.broadcast_to(arr, (n_images, 3))
    if arr.ndim == 2 and arr.shape == (n_images, 3):
        return arr
    return None


def _geometry_stack_to_flat(
    stack: npt.ArrayLike, n_images: int, n_pixels: int, name: str
) -> np.ndarray:
    arr = np.asarray(stack, dtype=float)
    if arr.ndim == 1 and arr.shape == (3,):
        return np.broadcast_to(arr, (n_images, n_pixels, 3))
    if arr.ndim == 2 and arr.shape == (n_images, 3):
        return np.broadcast_to(arr[:, None, :], (n_images, n_pixels, 3))
    if arr.ndim >= 3 and arr.shape[0] == n_images and arr.shape[-1] == 3:
        flat = arr.reshape(n_images, -1, 3)
        if flat.shape[1] == 1:
            return np.broadcast_to(flat, (n_images, n_pixels, 3))
        if flat.shape[1] == n_pixels:
            return np.ascontiguousarray(flat)
    if arr.ndim >= 3 and arr.shape[-1] == 3:
        flat = arr.reshape(1, -1, 3)
        if flat.shape[1] == n_pixels:
            return np.broadcast_to(flat, (n_images, n_pixels, 3))
    raise ValueError(
        f"{name} geometry has shape {arr.shape}, expected image stack geometry"
    )


def _active_mask_stack(
    reflectance: np.ndarray, mask: npt.ArrayLike | None
) -> np.ndarray:
    active = np.isfinite(reflectance)
    if mask is None:
        return active
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.shape == reflectance.shape:
        return active & mask_arr
    if mask_arr.shape == reflectance.shape[1:]:
        return active & np.broadcast_to(mask_arr, reflectance.shape)
    if mask_arr.size == reflectance[0].size:
        return active & np.broadcast_to(
            mask_arr.reshape(reflectance.shape[1:]), reflectance.shape
        )
    raise ValueError(
        f"mask has shape {mask_arr.shape}, expected {reflectance.shape} or {reflectance.shape[1:]}"
    )


def _smooth_inversion_inputs(
    reflectance: np.ndarray,
    s: npt.ArrayLike,
    v: npt.ArrayLike,
    n: npt.ArrayLike,
    active: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_images = reflectance.shape[0]
    image_shape = reflectance.shape[1:]
    reflectance_smooth = np.empty_like(reflectance, dtype=float)
    active_smooth = np.empty_like(active, dtype=bool)
    for k in range(n_images):
        reflectance_smooth[k] = _masked_gaussian(reflectance[k], active[k], sigma)
        active_smooth[k] = active[k] & np.isfinite(reflectance_smooth[k])

    normal_smooth = _smooth_geometry_image(
        n, np.any(active, axis=0), image_shape, sigma
    )
    sun_smooth = _smooth_geometry_stack(s, active, image_shape, sigma, n_images)
    view_smooth = _smooth_geometry_stack(v, active, image_shape, sigma, n_images)
    return reflectance_smooth, sun_smooth, view_smooth, normal_smooth, active_smooth


def _smooth_geometry_stack(
    geometry: npt.ArrayLike,
    active: np.ndarray,
    image_shape: tuple[int, ...],
    sigma: float,
    n_images: int,
) -> np.ndarray:
    geometry_arr = np.asarray(geometry, dtype=float)
    # Constant-per-image geometry is unaffected by smoothing; keep it
    # compact instead of materialising (n_images, *image_shape, 3) copies.
    if geometry_arr.ndim == 1 and geometry_arr.shape == (3,):
        return geometry_arr
    if geometry_arr.ndim == 2 and geometry_arr.shape == (n_images, 3):
        return geometry_arr
    if geometry_arr.shape == (*image_shape, 3):
        geometry_arr = np.broadcast_to(
            geometry_arr[None, ...], (n_images, *image_shape, 3)
        ).copy()
    if geometry_arr.shape != (n_images, *image_shape, 3):
        raise ValueError(
            f"geometry has shape {geometry_arr.shape}, expected (3,), ({n_images}, 3), "
            f"{(*image_shape, 3)}, or {(n_images, *image_shape, 3)}"
        )
    return np.stack(
        [
            _smooth_geometry_image(geometry_arr[k], active[k], image_shape, sigma)
            for k in range(n_images)
        ]
    )


def _smooth_geometry_image(
    geometry: npt.ArrayLike,
    valid: np.ndarray,
    image_shape: tuple[int, ...],
    sigma: float,
) -> np.ndarray:
    geometry_arr = np.asarray(geometry, dtype=float)
    if geometry_arr.ndim == 1 and geometry_arr.shape == (3,):
        return np.broadcast_to(
            geometry_arr.reshape((1,) * len(image_shape) + (3,)), (*image_shape, 3)
        ).copy()
    if geometry_arr.shape != (*image_shape, 3):
        raise ValueError(
            f"geometry has shape {geometry_arr.shape}, expected {(*image_shape, 3)}"
        )
    geometry_arr = _normalize_np(geometry_arr)
    nz = geometry_arr[..., 2]
    slope_valid = valid & np.isfinite(geometry_arr).all(axis=-1) & (np.abs(nz) > 1e-12)
    p = -geometry_arr[..., 0] / np.where(np.abs(nz) > 1e-12, nz, np.nan)
    q = -geometry_arr[..., 1] / np.where(np.abs(nz) > 1e-12, nz, np.nan)
    p_smooth = _masked_gaussian(p, slope_valid, sigma)
    q_smooth = _masked_gaussian(q, slope_valid, sigma)
    fallback = _normalize_np(geometry_arr.reshape(-1, 3)).reshape(geometry_arr.shape)
    smooth = _gradient_normals(p_smooth.reshape(-1), q_smooth.reshape(-1)).reshape(
        geometry_arr.shape
    )
    return np.where(np.isfinite(smooth).all(axis=-1, keepdims=True), smooth, fallback)


def _masked_gaussian(values: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    valid = valid & np.isfinite(values)
    weights = gaussian_filter(
        valid.astype(float), sigma=sigma, mode="constant", cval=0.0
    )
    numerator = gaussian_filter(
        np.where(valid, values, 0.0), sigma=sigma, mode="constant", cval=0.0
    )
    return np.where(weights > 1e-12, numerator / weights, np.nan)


# Single source of truth for the (0, 1) <-> R albedo transform lives in
# refmod.hapke.inverse; re-exported here for the multi-image solvers.
from refmod.hapke.inverse import _tanh_to_w, _w_to_tanh  # noqa: E402


def _amsa_residual_x(
    x: jax.Array,
    refl_obs: jax.Array,
    s: jax.Array,
    v: jax.Array,
    n: jax.Array,
    active: jax.Array,
    b_n: jax.Array,
    tb: float,
    hs: float,
    Bs0: float,
    hc: float,
    Bc0: float,
) -> jax.Array:
    w = jnp.full((refl_obs.shape[0],), _tanh_to_w(x))
    n_stack = jnp.broadcast_to(n, s.shape)
    model = amsa(w, b_n, s, v, n_stack, tb, hs, Bs0, hc, Bc0)
    return jnp.where(active & jnp.isfinite(model), model - refl_obs, 0.0)


def _imsa_modified_h_residual_x(
    x: jax.Array,
    refl_obs: jax.Array,
    s: jax.Array,
    v: jax.Array,
    n: jax.Array,
    active: jax.Array,
    b: float,
    c: float,
    tb: float,
    h: float,
    b0: float,
) -> jax.Array:
    w = jnp.full((refl_obs.shape[0],), _tanh_to_w(x))
    n_stack = jnp.broadcast_to(n, s.shape)
    model = imsa_modified_h(w, b, c, s, v, n_stack, tb, h, b0)
    return jnp.where(active & jnp.isfinite(model), model - refl_obs, 0.0)


def _invert_multi_pixel(
    refl_obs: jax.Array,
    s: jax.Array,
    v: jax.Array,
    n: jax.Array,
    active: jax.Array,
    w0: jax.Array,
    b_n: jax.Array,
    tb: float,
    hs: float,
    Bs0: float,
    hc: float,
    Bc0: float,
    max_steps: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    x0 = jnp.where(jnp.any(active), _w_to_tanh(w0), 0.0)
    lam0 = jnp.array(1e-3)

    def body(carry):
        x, lam, _last_loss, converged, step = carry
        residual_fun = lambda x_value: _amsa_residual_x(
            x_value, refl_obs, s, v, n, active, b_n, tb, hs, Bs0, hc, Bc0
        )
        residual = residual_fun(x)
        jac = jax.jacfwd(residual_fun)(x)
        loss = 0.5 * jnp.sum(residual**2)
        grad = jnp.sum(jac * residual)
        hess = jnp.sum(jac**2)
        delta = grad / (hess + lam)
        x_new = x - delta
        residual_new = residual_fun(x_new)
        loss_new = 0.5 * jnp.sum(residual_new**2)
        improved = loss_new < loss
        x = jnp.where(improved, x_new, x)
        lam = jnp.where(
            improved, jnp.maximum(lam * 0.3, 1e-7), jnp.minimum(lam * 2.0, 1e7)
        )
        converged = converged | (jnp.abs(grad) < 1e-10) | (jnp.abs(delta) < 1e-10)
        return x, lam, jnp.where(improved, loss_new, loss), converged, step + 1

    def cond(carry):
        _x, _lam, _last_loss, converged, step = carry
        return jnp.any(active) & (~converged) & (step < max_steps)

    x, _lam, loss, converged, steps = jax.lax.while_loop(
        cond,
        body,
        (x0, lam0, jnp.array(jnp.inf), False, jnp.array(0)),
    )
    return _tanh_to_w(x), loss, converged | ~jnp.any(active), steps


_invert_multi_amsa_jit = jax.jit(
    jax.vmap(
        _invert_multi_pixel,
        in_axes=(0, 0, 0, 0, 0, 0, None, None, None, None, None, None, None),
    ),
    static_argnums=(12,),
)

# Variant sharing the per-image (n_images, 3) sun/view arrays across pixels;
# used when the geometry is constant per image (the common case).
_invert_multi_amsa_shared_jit = jax.jit(
    jax.vmap(
        _invert_multi_pixel,
        in_axes=(0, None, None, 0, 0, 0, None, None, None, None, None, None, None),
    ),
    static_argnums=(12,),
)


def _invert_multi_imsa_modified_h_pixel(
    refl_obs: jax.Array,
    s: jax.Array,
    v: jax.Array,
    n: jax.Array,
    active: jax.Array,
    w0: jax.Array,
    b: float,
    c: float,
    tb: float,
    h: float,
    b0: float,
    max_steps: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    x0 = jnp.where(jnp.any(active), _w_to_tanh(w0), 0.0)
    lam0 = jnp.array(1e-3)

    def body(carry):
        x, lam, _last_loss, converged, step = carry
        residual_fun = lambda x_value: _imsa_modified_h_residual_x(
            x_value, refl_obs, s, v, n, active, b, c, tb, h, b0
        )
        residual = residual_fun(x)
        jac = jax.jacfwd(residual_fun)(x)
        loss = 0.5 * jnp.sum(residual**2)
        grad = jnp.sum(jac * residual)
        hess = jnp.sum(jac**2)
        delta = grad / (hess + lam)
        x_new = x - delta
        residual_new = residual_fun(x_new)
        loss_new = 0.5 * jnp.sum(residual_new**2)
        improved = loss_new < loss
        x = jnp.where(improved, x_new, x)
        lam = jnp.where(
            improved, jnp.maximum(lam * 0.3, 1e-7), jnp.minimum(lam * 2.0, 1e7)
        )
        converged = converged | (jnp.abs(grad) < 1e-10) | (jnp.abs(delta) < 1e-10)
        return x, lam, jnp.where(improved, loss_new, loss), converged, step + 1

    def cond(carry):
        _x, _lam, _last_loss, converged, step = carry
        return jnp.any(active) & (~converged) & (step < max_steps)

    x, _lam, loss, converged, steps = jax.lax.while_loop(
        cond,
        body,
        (x0, lam0, jnp.array(jnp.inf), False, jnp.array(0)),
    )
    return _tanh_to_w(x), loss, converged | ~jnp.any(active), steps


_invert_multi_imsa_modified_h_jit = jax.jit(
    jax.vmap(
        _invert_multi_imsa_modified_h_pixel,
        in_axes=(0, 0, 0, 0, 0, 0, None, None, None, None, None, None),
    ),
    static_argnums=(11,),
)

_invert_multi_imsa_modified_h_shared_jit = jax.jit(
    jax.vmap(
        _invert_multi_imsa_modified_h_pixel,
        in_axes=(0, None, None, 0, 0, 0, None, None, None, None, None, None),
    ),
    static_argnums=(11,),
)
