from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from .functions.phase import PhaseFunctionType
from .models import amsa, imsa

__all__ = ["amsa", "imsa", "Hapke"]


class Hapke(BaseModel):
    single_scattering_albedo: npt.NDArray
    incidence_direction: npt.NDArray
    emission_direction: npt.NDArray
    surface_orientation: npt.NDArray
    phase_function_type: PhaseFunctionType = Field(default="dhg")
    roughness: float = Field(0.0)
    shadow_hiding_h: float = Field(0.0)
    shadow_hiding_b0: float = Field(0.0)
    coherant_backscattering_h: float = Field(0.0)
    coherant_backscattering_b0: float = Field(0.0)
    phase_function_args: tuple = Field(default=())

    model: Literal["amsa", "imsa"] = Field(default="amsa")
    h_level: Literal[1, 2] = Field(default=2)
    backend: Literal["numpy", "numba"] = Field(default="numpy")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def refl(self) -> npt.NDArray:
        """
        Calculate the reflectance using the specified Hapke model.

        Returns:
            npt.NDArray: The calculated reflectance.
        """
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
