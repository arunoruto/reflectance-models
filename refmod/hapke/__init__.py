from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import least_squares

from .functions.legendre import coef_a
from .models import amsa, amsa_scalar, imsa

__all__ = ["amsa", "imsa", "Hapke"]


class Hapke(BaseModel):
    single_scattering_albedo: npt.NDArray | None = Field(default=None)
    legendre_coefficients: npt.NDArray = Field(default=np.array([1.0, 0.0, 0.5]))
    incidence_direction: npt.NDArray = Field(default=np.array(0.0))
    emission_direction: npt.NDArray = Field(default=np.array(0.0))
    surface_orientation: npt.NDArray = Field(default=np.array([0.0, 0.0, 1.0]))
    roughness: float = Field(default=0.0)
    shadow_hiding_h: float = Field(default=0.0)
    shadow_hiding_b0: float = Field(default=0.0)
    coherant_backscattering_h: float = Field(default=0.0)
    coherant_backscattering_b0: float = Field(default=0.0)

    model: Literal["amsa", "imsa"] = Field(default="amsa")
    h_level: Literal[1, 2] = Field(default=2)
    # backend: Literal["numpy", "numba"] = Field(default="numpy")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization hook to ensure array shapes are correct.

        Ensures incidence_direction, emission_direction, and surface_orientation
        have at least 3 dimensions (batch dimensions + vector dimension).
        """
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
                single_scattering_albedo=self.single_scattering_albedo,
                phase_function_legendre=self.legendre_coefficients,
                incidence_direction=self.incidence_direction,
                emission_direction=self.emission_direction,
                surface_orientation=self.surface_orientation,
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
                b_n=self.legendre_coefficients,
                roughness=self.roughness,
                opposition_effect_h=self.shadow_hiding_h,
                opposition_effect_b0=self.shadow_hiding_b0,
                h_level=self.h_level,
            )

    def albedo(
        self,
        reflectance: npt.NDArray,
        least_squares_param: dict = {"method": "lm"},
        x0: npt.NDArray | None = None,
    ) -> npt.NDArray:
        """Invert the Hapke model to estimate single scattering albedo.

        Parameters
        ----------
        reflectance : npt.NDArray
            Observed reflectance values.
        least_squares_param : dict, optional
            Additional parameters passed to scipy.optimize.least_squares,
            by default {"method": "lm"}.
        x0 : npt.NDArray | None, optional
            Initial guess for single scattering albedo. If None, defaults to 1/3.

        Returns
        -------
        npt.NDArray
            Estimated single scattering albedo.

        Raises
        ------
        ValueError
            If the model is not 'amsa'.
        Exception
            If the reflectance array has invalid dimensions (>3).
        """
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

        a_n = coef_a(n=self.legendre_coefficients.shape[0] - 1)

        # Direction vectors are shaped as (..., 3). The last axis is always the
        # vector component axis and should not be treated as a spatial dimension.
        space_shape = self.surface_orientation.shape[1:-1]
        bands_shape = reflectance.shape[: len(space_shape)]

        original_shape = np.array(reflectance.shape)
        # Expand/tile vectors to match the flattened reflectance.
        # Vectors are stored as (..., 3); reshape to (3, N) directly.
        incidence_direction = (
            np.tile(
                np.expand_dims(self.incidence_direction, axis=1),
                (1, *bands_shape, 1, 1),
            )
            .reshape(-1, 3)
            .T
        )
        emission_direction = (
            np.tile(
                np.expand_dims(self.emission_direction, axis=1),
                (1, *bands_shape, 1, 1),
            )
            .reshape(-1, 3)
            .T
        )
        surface_orientation = (
            np.tile(
                np.expand_dims(self.surface_orientation, axis=1),
                (1, *bands_shape, 1, 1),
            )
            .reshape(-1, 3)
            .T
        )
        reflectance = reflectance.reshape(-1)

        initial_guess = np.ones_like(reflectance) / 3 if x0 is None else np.asarray(x0)
        if initial_guess.shape != reflectance.shape:
            raise ValueError(
                f"x0 must have the same shape as reflectance; got {initial_guess.shape} vs {reflectance.shape}"
            )

        albedo_recon = least_squares(
            amsa_scalar,
            initial_guess,
            # method="lm",
            # verbose=2,
            kwargs=dict(
                b_n=self.legendre_coefficients,
                incidence_direction=incidence_direction,
                emission_direction=emission_direction,
                surface_orientation=surface_orientation,
                a_n=a_n,
                roughness=self.roughness,
                shadow_hiding_h=self.shadow_hiding_h,
                shadow_hiding_b0=self.shadow_hiding_b0,
                coherant_backscattering_h=self.coherant_backscattering_h,
                coherant_backscattering_b0=self.coherant_backscattering_b0,
                refl_optimization=reflectance,
                h_level=self.h_level,
            ),
            **least_squares_param,
        )
        self.single_scattering_albedo = np.array(albedo_recon.x.reshape(original_shape))

        return self.single_scattering_albedo

    # single_scattering_albedo: npt.NDArray,
    # b_n: npt.NDArray,
    # incidence_direction: npt.NDArray,
    # emission_direction: npt.NDArray,
    # surface_orientation: npt.NDArray,
    # a_n: npt.NDArray | None = None,
    # roughness: float = 0,
    # shadow_hiding_h: float = 0.0,
    # shadow_hiding_b0: float = 0.0,
    # coherant_backscattering_h: float = 0.0,
    # coherant_backscattering_b0: float = 0.0,
    # refl_optimization: npt.NDArray | None = None,
    # h_level: int = 2,
    # imsa: bool = False,
