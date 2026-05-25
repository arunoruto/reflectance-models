from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from ._core import coef_a, cs_legendre_coefficients, dhg_legendre_coefficients
from .amsa import amsa
from .imsa import imsa
from .inverse import AmsaInversionState, invert_amsa, invert_amsa_precomputed, prepare_amsa_inversion
from .mimsa import mimsa

__all__ = [
    "amsa",
    "imsa",
    "mimsa",
    "Hapke",
    "coef_a",
    "dhg_legendre_coefficients",
    "cs_legendre_coefficients",
    "AmsaInversionState",
    "invert_amsa",
    "invert_amsa_precomputed",
    "prepare_amsa_inversion",
]


def _to_flat_vectors(v: np.ndarray) -> np.ndarray:
    if v.ndim < 2:
        return v.reshape(1, 3)
    elif v.shape[-1] == 3:
        return v.reshape(-1, 3)
    else:
        v = np.moveaxis(v, 0, -1)
        return v.reshape(-1, 3)


class Hapke(BaseModel):
    single_scattering_albedo: npt.NDArray | None = Field(default=None)
    legendre_coefficients: npt.NDArray = Field(default=np.array([1.0, 0.0, 0.5]))
    incidence_direction: npt.NDArray = Field(default=np.array(0.0))
    emission_direction: npt.NDArray = Field(default=np.array(0.0))
    surface_orientation: npt.NDArray = Field(default=np.array([0.0, 0.0, 1.0]))
    roughness: float = Field(default=0.0)
    shadow_hiding_h: float = Field(default=0.0)
    shadow_hiding_b0: float = Field(default=0.0)
    coherent_backscattering_h: float = Field(default=0.0)
    coherent_backscattering_b0: float = Field(default=0.0)

    model: Literal["amsa", "imsa", "mimsa"] = Field(default="amsa")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        if self.incidence_direction.ndim < 2:
            self.incidence_direction = np.ascontiguousarray(
                self.incidence_direction
            ).reshape(1, 1, 3)
        if self.emission_direction.ndim < 2:
            self.emission_direction = np.ascontiguousarray(
                self.emission_direction
            ).reshape(1, 1, 3)
        if self.surface_orientation.ndim < 2:
            self.surface_orientation = np.ascontiguousarray(
                self.surface_orientation
            ).reshape(1, 1, 3)

    @staticmethod
    def _broadcast_to_shape(a: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
        if a.ndim >= len(target_shape) + 1:
            return a
        a = a.squeeze()
        tile_dims = []
        for i, (s, t) in enumerate(zip(a.shape, target_shape)):
            if s == 1 and t > 1:
                tile_dims.append((i, t))
        for axis, reps in tile_dims:
            a = np.tile(np.expand_dims(a, axis=axis), (1,) * axis + (reps,) + (1,) * (a.ndim - axis))
        return a

    def refl(self) -> npt.NDArray:
        if self.single_scattering_albedo is None:
            raise ValueError("single_scattering_albedo must be provided")

        ssa = np.asarray(self.single_scattering_albedo)
        i_flat = _to_flat_vectors(np.asarray(self.incidence_direction))
        e_flat = _to_flat_vectors(np.asarray(self.emission_direction))
        n_flat = _to_flat_vectors(np.asarray(self.surface_orientation))
        w_flat = np.ascontiguousarray(ssa).reshape(-1)

        n_pixels = max(i_flat.shape[0], e_flat.shape[0], n_flat.shape[0], w_flat.shape[0])
        w_flat = np.broadcast_to(w_flat, (n_pixels,))
        if i_flat.shape[0] == 1:
            i_flat = np.broadcast_to(i_flat, (n_pixels, 3))
        if e_flat.shape[0] == 1:
            e_flat = np.broadcast_to(e_flat, (n_pixels, 3))
        if n_flat.shape[0] == 1:
            n_flat = np.broadcast_to(n_flat, (n_pixels, 3))

        w_jax = jnp.asarray(w_flat)
        i_jax = jnp.asarray(i_flat)
        e_jax = jnp.asarray(e_flat)
        n_jax = jnp.asarray(n_flat)
        b_n_jax = jnp.asarray(self.legendre_coefficients)

        if self.model == "amsa":
            refl_jax = amsa(
                w_jax,
                b_n_jax,
                i_jax,
                e_jax,
                n_jax,
                self.roughness,
                self.shadow_hiding_h,
                self.shadow_hiding_b0,
                self.coherent_backscattering_h,
                self.coherent_backscattering_b0,
            )
        elif self.model == "imsa":
            refl_jax = imsa(
                w_jax,
                b_n_jax,
                i_jax,
                e_jax,
                n_jax,
                self.roughness,
            )
        else:
            refl_jax = mimsa(
                w_jax,
                b_n_jax,
                i_jax,
                e_jax,
                n_jax,
                self.roughness,
            )

        return np.asarray(refl_jax).reshape(ssa.shape)

    def albedo(
        self,
        reflectance: npt.NDArray,
        x0: npt.NDArray | None = None,
    ) -> npt.NDArray:
        if self.model != "amsa":
            raise ValueError("Albedo inversion is only implemented for 'amsa' model.")

        refl_arr = np.asarray(reflectance)
        original_shape = refl_arr.shape

        i_flat = _to_flat_vectors(np.asarray(self.incidence_direction))
        e_flat = _to_flat_vectors(np.asarray(self.emission_direction))
        n_flat = _to_flat_vectors(np.asarray(self.surface_orientation))
        refl_flat = np.ascontiguousarray(refl_arr).reshape(-1)

        n_pixels = max(i_flat.shape[0], e_flat.shape[0], n_flat.shape[0], refl_flat.shape[0])
        refl_flat = np.broadcast_to(refl_flat, (n_pixels,))
        if i_flat.shape[0] == 1:
            i_flat = np.broadcast_to(i_flat, (n_pixels, 3))
        if e_flat.shape[0] == 1:
            e_flat = np.broadcast_to(e_flat, (n_pixels, 3))
        if n_flat.shape[0] == 1:
            n_flat = np.broadcast_to(n_flat, (n_pixels, 3))

        w0_flat = np.full(n_pixels, 1.0 / 3.0) if x0 is None else np.broadcast_to(np.asarray(x0).reshape(-1), (n_pixels,))

        refl_jax = jnp.asarray(refl_flat)
        i_jax = jnp.asarray(i_flat)
        e_jax = jnp.asarray(e_flat)
        n_jax = jnp.asarray(n_flat)
        b_n_jax = jnp.asarray(self.legendre_coefficients)
        w0_jax = jnp.asarray(w0_flat)

        w_sol = invert_amsa(
            refl_jax,
            b_n_jax,
            i_jax,
            e_jax,
            n_jax,
            self.roughness,
            self.shadow_hiding_h,
            self.shadow_hiding_b0,
            self.coherent_backscattering_h,
            self.coherent_backscattering_b0,
            w0_jax,
        )

        self.single_scattering_albedo = np.asarray(w_sol).reshape(original_shape)
        return self.single_scattering_albedo
