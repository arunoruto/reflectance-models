import numpy as np
import numpy.typing as npt
from numba import jit
from scipy.optimize import least_squares


def __linear_mixing(
    bulk_density: npt.NDArray,
    albedo: npt.NDArray,
    phase_function_coefficients: npt.NDArray,
    extinction_efficiency: npt.NDArray | None = None,
    solid_density: npt.NDArray | None = None,
    radius: npt.NDArray | None = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    # bulk density
    if bulk_density.shape[0] != albedo.shape[1]:
        raise ValueError(
            "Number of end-members in bulk_density (mixing coefficients) and albedo need to be the same!"
        )
    # phase function
    if phase_function_coefficients.shape[1:] != albedo.shape:
        raise ValueError(
            "Phase function (except the first dimension) and albedo should have the same shape"
        )
    # extinction efficiency
    if extinction_efficiency is None:
        extinction_efficiency = np.ones((albedo.shape[1], 1))
    elif extinction_efficiency.shape[0] != albedo.shape[1]:
        raise ValueError(
            "Extinction efficienceis and number of end-members should be the same!"
        )
    # solid density
    if solid_density is None:
        solid_density = np.ones((albedo.shape[1], 1))
    elif solid_density.shape[0] != albedo.shape[1]:
        raise ValueError("Solid density and number of end-members should be the same!")
    # radius
    if radius is None:
        radius = np.ones((albedo.shape[1], 1))
    elif radius.shape[0] != albedo.shape[1]:
        raise ValueError("Radius and number of end-members should be the same!")

    coefficients = bulk_density * extinction_efficiency / solid_density / radius
    coefficients /= np.sum(coefficients, axis=0, keepdims=True)

    # resulting_albedo = coefficients * albedo
    resulting_albedo = albedo @ coefficients
    resulting_phase_function_coefficients = (
        phase_function_coefficients * resulting_albedo / np.atleast_3d(resulting_albedo)
    )

    return resulting_albedo, resulting_phase_function_coefficients


def __list_to_legendre(
    phase_function: list[npt.NDArray],
    legendre_expansion: int = 15,
) -> npt.NDArray:
    legendre_coefficients = np.empty((legendre_expansion, len(phase_function)))
    for i, p in enumerate(phase_function):
        # TODO: this is quite an assumption, but it works here
        # Maybe let the user provide a list of theta values in the future!
        cos_g = np.cos(np.linspace(0, np.pi, p.size))
        legendre_coefficients[:, i] = np.polynomial.Legendre.fit(
            cos_g,
            p,
            deg=legendre_expansion,
            domain=[-1, 1],
        ).coef
    return legendre_coefficients


def linear_mixing(
    bulk_density: npt.NDArray,
    albedo: npt.NDArray,
    phase_function: list[npt.NDArray],
    extinction_efficiency: npt.NDArray | None = None,
    solid_density: npt.NDArray | None = None,
    radius: npt.NDArray | None = None,
    legendre_expansion: int = 15,
    theta_elements: int = 1 * 180 + 1,
) -> tuple[npt.NDArray, npt.NDArray]:
    legendre_coefficients = __list_to_legendre(
        phase_function,
        legendre_expansion,
    )

    mixed_albedo, mixed_coefficients = __linear_mixing(
        bulk_density,
        albedo,
        legendre_coefficients,
        extinction_efficiency,
        solid_density,
        radius,
    )

    mixed_phase_function = np.polynomial.legendre.legval(
        np.cos(np.linspace(0, np.pi, theta_elements)),
        mixed_coefficients,
    )

    return mixed_albedo, mixed_phase_function


# def linear_unmixing(
#     measurements: npt.NDArray,
#     endmembers: npt.NDArray,
# ):
#     lstsq = np.linalg.lstsq(endmembers, measurements, rcond=None)
#     abundance = lstsq[0]
#     return abundance


# NOTE:
# - Solve problem of minimizing |calculated - measured|^2
# - measured is a reflectance here
# - calculated is the reflectance from a function (amsa, imsa, etc)
# - the function takes parameters - provide them as a dict?
# - the function also takes an albedo and phase function (usually params, but need to change that!)
# - input albedo and phase function are a super-position of endmembers
# - calculate first the linear combination
# - IDEA: we have a linear_mixing function ->
#   - create a mixed_reflectance function which uses the mixture and passes it to
def nonlinear_unmixing(
    reflectance: npt.NDArray,
    albedo_endmembers: npt.NDArray,
    phase_function_endmembers: list[npt.NDArray],
    extinction_efficiency: npt.NDArray | None = None,
    solid_density: npt.NDArray | None = None,
    radius: npt.NDArray | None = None,
    legendre_expansion: int = 15,
) -> npt.NDArray | None:
    legendre_coefficients = __list_to_legendre(
        phase_function_endmembers,
        legendre_expansion,
    )
