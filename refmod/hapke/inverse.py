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
    """Invert the AMSA model to estimate single scattering albedo from reflectance.

    Parameters
    ----------
    refl : npt.NDArray
        Observed reflectance values.
    incidence_direction : npt.NDArray
        Incidence direction vector(s).
    emission_direction : npt.NDArray
        Emission direction vector(s).
    surface_orientation : npt.NDArray
        Surface orientation vector(s).
    phase_function_type : PhaseFunctionType, optional
        Type of phase function to use, by default "dhg".
    b_n : npt.NDArray | None, optional
        Legendre coefficients, by default None.
    a_n : npt.NDArray | None, optional
        Coefficients of the Legendre expansion, by default None.
    roughness : float, optional
        Surface roughness, by default 0.
    shadow_hiding_h : float, optional
        Shadowing parameter h, by default 0.0.
    shadow_hiding_b0 : float, optional
        Shadowing parameter b0, by default 0.0.
    coherant_backscattering_h : float, optional
        Coherent backscattering parameter h, by default 0.0.
    coherant_backscattering_b0 : float, optional
        Coherent backscattering parameter b0, by default 0.0.
    phase_function_args : tuple, optional
        Additional arguments for the phase function, by default ().
    h_level : int, optional
        Level of the H-function approximation, by default 2.

    Returns
    -------
    npt.NDArray
        Estimated single scattering albedo.

    Raises
    ------
    Exception
        If the reflectance array has invalid dimensions (>3).
    """
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

    albedo_recon = least_squares(
        amsa,
        np.ones_like(refl) / 3,
        # jac=amsa_derivative,  # pyright: ignore
        method="lm",
        # verbose=2,
        kwargs=dict(
            incidence_direction=incidence_direction,
            emission_direction=emission_direction,
            surface_orientation=surface_orientation,
            phase_function_type=phase_function_type,
            b_n=b_n,
            a_n=a_n,
            roughness=roughness,
            shadow_hiding_h=shadow_hiding_h,
            shadow_hiding_b0=shadow_hiding_b0,
            coherant_backscattering_h=coherant_backscattering_h,
            coherant_backscattering_b0=coherant_backscattering_b0,
            phase_function_args=phase_function_args,
            refl_optimization=refl,
            h_level=h_level,
        ),
    )

    return albedo_recon.x.reshape(original_shape)
