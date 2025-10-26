import numpy as np
import numpy.typing as npt


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
    if bulk_density.ndim == 1:
        bulk_density = bulk_density[:, np.newaxis]
    # phase function
    if phase_function_coefficients.shape[1:] != albedo.shape:
        raise ValueError(
            "Phase function (except the first dimension) and albedo should have the same shape"
        )
    # extinction efficiency
    if extinction_efficiency is None:
        extinction_efficiency = np.ones_like(albedo)
    elif extinction_efficiency.shape != albedo.shape:
        raise ValueError(
            f"""
            Extinction efficienceis and number of end-members should be the same!
            {extinction_efficiency.shape=} {albedo.shape=}
            """
        )
    # solid density
    if solid_density is None:
        solid_density = np.ones((albedo.shape[1], 1))
    elif solid_density.shape[0] != solid_density.size != albedo.shape[1]:
        raise ValueError("Solid density and number of end-members should be the same!")
    if solid_density.ndim == 1:
        solid_density = solid_density[:, np.newaxis]
    # radius
    if radius is None:
        radius = np.ones((albedo.shape[1], 1))
    elif radius.shape[0] != radius.size != albedo.shape[1]:
        raise ValueError("Radius and number of end-members should be the same!")
    if radius.ndim == 1:
        radius = radius[:, np.newaxis]

    coefficients = bulk_density / solid_density / radius
    # print(albedo.shape)
    # print(bulk_density.shape, solid_density.shape, radius.shape)
    # print(coefficients.shape)
    # print(extinction_efficiency.shape)
    # print(albedo * extinction_efficiency == albedo)

    resulting_albedo = (albedo * extinction_efficiency) @ coefficients
    resulting_albedo /= extinction_efficiency @ coefficients
    # print(f"""
    #     {extinction_efficiency.shape=},
    #     {coefficients.shape=},
    #     {(extinction_efficiency @ coefficients).shape=},
    #     {albedo.shape=}
    #     {resulting_albedo.shape=}
    # """)
    # print(f"""
    #     {(extinction_efficiency @ coefficients)=},
    # """)

    albedo = albedo[np.newaxis, :, :]
    extinction_efficiency = extinction_efficiency[np.newaxis, :, :]

    resulting_phase_function_coefficients = (
        phase_function_coefficients * albedo * extinction_efficiency
    ) @ coefficients
    resulting_phase_function_coefficients /= (
        albedo * extinction_efficiency
    ) @ coefficients
    return resulting_albedo, resulting_phase_function_coefficients


def __list_to_legendre(
    phase_function: list[npt.NDArray],
    legendre_expansion: int = 15,
) -> npt.NDArray:
    if phase_function[0].ndim != 2:
        raise ValueError("Phase function arrays must be 2D! (angles x wavelengths)")
    num_wavelengths = phase_function[0].shape[1]
    legendre_coefficients = np.empty(
        (legendre_expansion + 1, num_wavelengths, len(phase_function))
    )
    for i, p in enumerate(phase_function):
        if p.ndim != 2:
            # TODO: if ndim is 1, repeat it for every "wavelength"
            raise ValueError("Phase function arrays must be 2D! (angles x wavelengths)")
        for j in range(num_wavelengths):
            if p.shape[1] != num_wavelengths:
                raise ValueError(
                    "All phase function arrays must have the same number of wavelengths!"
                )
            # TODO: this is quite an assumption, but it works here
            # Maybe let the user provide a list of theta values in the future!
            cos_g = np.cos(np.linspace(0, np.pi, p.shape[0]))
            print(f"{cos_g.shape=} {p.shape=}")
            legendre_coefficients[:, j, i] = np.polynomial.Legendre.fit(
                cos_g,
                p[:, j],
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
    phase_function_coefficients_endmembers: npt.NDArray,
    extinction_efficiency: npt.NDArray | None = None,
    solid_density: npt.NDArray | None = None,
    radius: npt.NDArray | None = None,
) -> npt.NDArray | None: ...
