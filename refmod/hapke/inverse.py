import numpy as np
import numpy.typing as npt
from numba import jit
from refmod.config import cache
from refmod.hapke import amsa
from refmod.hapke.models import amsa_derivative
from scipy.optimize import least_squares

from .functions.phase import PhaseFunctionType


@jit(nogil=True, cache=cache)
def __amsa_wrapper(ssa, *args):
    return amsa(ssa, *args)


# @jit(nogil=True, cache=True)
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
) -> npt.NDArray:
    if incidence_direction.ndim < 2:
        incidence_direction = np.ascontiguousarray(incidence_direction).reshape(1, 1, 3)
    if emission_direction.ndim < 2:
        emission_direction = np.ascontiguousarray(emission_direction).reshape(1, 1, 3)
    if surface_orientation.ndim < 2:
        surface_orientation = np.ascontiguousarray(surface_orientation).reshape(1, 1, 3)

    if refl.ndim <= 1:
        refl = refl.reshape(-1, 1, 1)
    elif refl.ndim == 2:
        refl = np.expand_dims(refl, axis=0)
    elif refl.ndim > 3:
        raise Exception("The reflectance array must be 2D or 3D, it is: ", refl.shape)

    # __amsa_wrapper = lambda ssa, *args: amsa(ssa, *args)  # noqa
    space_shape = surface_orientation.shape[1:]
    bands_shape = refl.shape[: len(space_shape) - 1]

    original_shape = np.array(refl.shape)
    incidence_direction = np.tile(
        np.expand_dims(incidence_direction, axis=1),
        (1, *bands_shape, 1, 1),
    ).reshape(3, -1)
    emission_direction = np.tile(
        np.expand_dims(emission_direction, axis=1),
        (1, *bands_shape, 1, 1),
    ).reshape(3, -1)
    surface_orientation = np.tile(
        np.expand_dims(surface_orientation, axis=1),
        (1, *bands_shape, 1, 1),
    ).reshape(3, -1)
    refl = refl.reshape(-1)

    # albedo_recon, _, _, _, _ = levenberg_marquardt_core(
    #     func=__amsa_wrapper,
    #     p0=np.ones(refl.size) / 3,
    #     target_y=refl.flatten(),
    #     args=(
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
    #         # refl,
    #     ),
    # )

    # return albedo_recon.reshape(original_shape)

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
        np.ones_like(refl) / 3,
        # jac=lambda w: (  # pyright: ignore
        #     amsa_derivative(
        #         w,
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
        #         # refl,
        #         h_level=h_level,
        #     )
        # ),
        method="lm",
        # verbose=2,
    )

    return albedo_recon.x.reshape(original_shape)
