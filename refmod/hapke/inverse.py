from typing import Callable, Tuple, Union

import numpy as np
import numpy.typing as npt
from refmod.hapke import amsa
from scipy.optimize import least_squares

from .functions.phase import PhaseFunctionType


def inverse_model(
    refl: npt.NDArray,
    incidence_direction: npt.NDArray,
    emission_direction: npt.NDArray,
    surface_orientation: npt.NDArray,
    phase_function_type: PhaseFunctionType = "dhg",
    b_n: npt.NDArray | None = None,
    a_n: npt.NDArray | None = None,
    roughness: float = 0,
    shadow_hiding_h: float = 0.0,
    shadow_hiding_b0: float = 0.0,
    coherant_backscattering_h: float = 0.0,
    coherant_backscattering_b0: float = 0.0,
    phase_function_args: tuple = (),
    h_level: int = 2,
) -> Union[npt.NDArray, Tuple[npt.NDArray, npt.NDArray]]:
    # incidence_direction = incidence_direction.reshape(-1, 1, 3)
    # emission_direction = emission_direction.reshape(-1, 1, 3)
    # surface_orientation = surface_orientation.reshape(-1, 1, 3)
    incidence_direction = incidence_direction.reshape(-1, 3)
    emission_direction = emission_direction.reshape(-1, 3)
    surface_orientation = surface_orientation.reshape(-1, 3)

    # refl_temp = (
    #     amsa(
    #         np.ones(surface_orientation.shape[0]),
    #         incidence_direction,
    #         emission_direction,
    #         surface_orientation,
    #         phase_function_type,
    #         b_n,
    #         a_n,
    #         roughness,
    #         shadow_hiding_h,
    #         shadow_hiding_b0,
    #         coherant_backscattering_h,
    #         coherant_backscattering_b0,
    #         phase_function_args,
    #         h_level=h_level,
    #     ),
    # )
    # print(f"{refl.shape=}")
    # print(f"{refl_temp[0].shape=}")
    # refl = np.nan_to_num(refl)
    # print(f"{np.sum(np.isfinite(refl))=}")
    # print(f"{np.ones(surface_orientation.shape[0]).shape=}")

    albedo_recon = least_squares(
        lambda w: (
            amsa(
                w,
                incidence_direction,
                emission_direction,
                surface_orientation,
                phase_function_type,
                b_n,
                a_n,
                roughness,
                shadow_hiding_h,
                shadow_hiding_b0,
                coherant_backscattering_h,
                coherant_backscattering_b0,
                phase_function_args,
                refl,
                h_level=h_level,
            )
        ),
        np.ones(surface_orientation.shape[0]) / 3,
        # method="dogbox",
        # verbose=2,
    )

    return albedo_recon.x
