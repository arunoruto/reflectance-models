from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import least_squares

from .functions.legendre import coef_a, coef_b
from .functions.phase import PhaseFunctionType
from .models import amsa, imsa

__all__ = ["amsa", "imsa", "Hapke"]


class Hapke(BaseModel):
    single_scattering_albedo: npt.NDArray | None = Field(default=None)
    incidence_direction: npt.NDArray = Field(default=np.array(0.0))
    emission_direction: npt.NDArray = Field(default=np.array(0.0))
    surface_orientation: npt.NDArray = Field(default=np.array([0.0, 0.0, 1.0]))
    phase_function_type: PhaseFunctionType = Field(default="dhg")
    roughness: float = Field(0.0)
    shadow_hiding_h: float = Field(0.0)
    shadow_hiding_b0: float = Field(0.0)
    coherant_backscattering_h: float = Field(0.0)
    coherant_backscattering_b0: float = Field(0.0)
    phase_function_args: tuple = Field(default=())
    legendre_expansion: int = Field(default=15)

    model: Literal["amsa", "imsa"] = Field(default="amsa")
    h_level: Literal[1, 2] = Field(default=2)
    # backend: Literal["numpy", "numba"] = Field(default="numpy")

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

    def refl(self) -> npt.NDArray:
        """
        Calculate the reflectance using the specified Hapke model.

        Returns:
            npt.NDArray: The calculated reflectance.
        """
        if self.single_scattering_albedo is None:
            raise ValueError(
                "single_scattering_albedo must be provided for reflectance calculations"
            )
        if self.model == "amsa":
            return amsa(
                incidence_direction=self.incidence_direction,
                emission_direction=self.emission_direction,
                surface_orientation=self.surface_orientation,
                single_scattering_albedo=self.single_scattering_albedo,
                phase_function_type=self.phase_function_type,
                roughness=self.roughness,
                shadow_hiding_h=self.shadow_hiding_h,
                shadow_hiding_b0=self.shadow_hiding_b0,
                coherant_backscattering_h=self.coherant_backscattering_h,
                coherant_backscattering_b0=self.coherant_backscattering_b0,
                h_level=self.h_level,
            )
        else:
            return imsa(
                incidence_direction=self.incidence_direction,
                emission_direction=self.emission_direction,
                surface_orientation=self.surface_orientation,
                single_scattering_albedo=self.single_scattering_albedo,
                phase_function_type=self.phase_function_type,
                roughness=self.roughness,
                opposition_effect_h=self.shadow_hiding_h,
                opposition_effect_b0=self.shadow_hiding_b0,
                h_level=self.h_level,
            )

    def albedo(self, reflectance: npt.NDArray) -> npt.NDArray:
        if self.model != "amsa":
            raise ValueError(
                "Albedo inversion is only implemented for the 'amsa' model."
            )

        if reflectance.ndim <= 1:
            reflectance = reflectance.reshape(-1, 1, 1)
        elif reflectance.ndim == 2:
            reflectance = np.expand_dims(reflectance, axis=0)
        elif reflectance.ndim > 3:
            raise Exception(
                "The reflectanceectance array must be 2D or 3D, it is: ",
                reflectance.shape,
            )

        a_n = coef_a(n=self.legendre_expansion)
        b_n = coef_b(*self.phase_function_args, n=self.legendre_expansion)

        space_shape = self.surface_orientation.shape[1:]
        bands_shape = reflectance.shape[: len(space_shape) - 1]

        original_shape = np.array(reflectance.shape)
        incidence_direction = np.tile(
            np.expand_dims(self.incidence_direction, axis=1),
            (1, *bands_shape, 1, 1),
        ).reshape(3, -1)
        emission_direction = np.tile(
            np.expand_dims(self.emission_direction, axis=1),
            (1, *bands_shape, 1, 1),
        ).reshape(3, -1)
        surface_orientation = np.tile(
            np.expand_dims(self.surface_orientation, axis=1),
            (1, *bands_shape, 1, 1),
        ).reshape(3, -1)
        reflectance = reflectance.reshape(-1)

        albedo_recon = least_squares(
            amsa,
            np.ones_like(reflectance) / 3,
            method="lm",
            # verbose=2,
            kwargs=dict(
                incidence_direction=incidence_direction,
                emission_direction=emission_direction,
                surface_orientation=surface_orientation,
                phase_function_type=self.phase_function_type,
                b_n=b_n,
                a_n=a_n,
                roughness=self.roughness,
                shadow_hiding_h=self.shadow_hiding_h,
                shadow_hiding_b0=self.shadow_hiding_b0,
                coherant_backscattering_h=self.coherant_backscattering_h,
                coherant_backscattering_b0=self.coherant_backscattering_b0,
                phase_function_args=self.phase_function_args,
                refl_optimization=reflectance,
                h_level=self.h_level,
            ),
        )
        self.single_scattering_albedo = np.array(albedo_recon.x.reshape(original_shape))

        return self.single_scattering_albedo
