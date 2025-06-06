from typing import Callable, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from refmod.hapke import amsa


def inverse_model(
    refl: npt.NDArray,
    incidence_direction: npt.NDArray,
    emission_direction: npt.NDArray,
    surface_orientation: npt.NDArray,
    phase_function: Callable[[npt.NDArray], npt.NDArray],
    b_n: npt.NDArray,
    a_n: npt.NDArray = np.empty(1) * np.nan,
    hs: float = 0,
    bs0: float = 0,
    roughness: float = 0,
    hc: float = 0,
    bc0: float = 0,
) -> Union[npt.NDArray, Tuple[npt.NDArray, npt.NDArray]]:
    # incidence_direction = incidence_direction.reshape(-1, 1, 3)
    # emission_direction = emission_direction.reshape(-1, 1, 3)
    # surface_orientation = surface_orientation.reshape(-1, 1, 3)
    incidence_direction = incidence_direction.reshape(-1, 3)
    emission_direction = emission_direction.reshape(-1, 3)
    surface_orientation = surface_orientation.reshape(-1, 3)

    refl_temp = (
        amsa(
            incidence_direction,
            emission_direction,
            surface_orientation,
            np.ones(surface_orientation.shape[0]),
            lambda x: phase_function(x),
            b_n,
            a_n,
            hs,
            bs0,
            roughness,
            hc,
            bc0,
        ),
    )
    print(f"{refl.shape=}")
    print(f"{refl_temp[0].shape=}")
    refl = np.nan_to_num(refl)
    print(f"{np.sum(np.isfinite(refl))=}")
    print(f"{np.ones(surface_orientation.shape[0]).shape=}")

    albedo_recon = least_squares(
        lambda w: (
            amsa(
                incidence_direction,
                emission_direction,
                surface_orientation,
                w,
                lambda x: phase_function(x),
                b_n,
                a_n,
                hs,
                bs0,
                roughness,
                hc,
                bc0,
                nan2zero=True,
            )[0]
            - refl
        ),
        np.ones(surface_orientation.shape[0]) / 3,
        # method="dogbox",
        verbose=2,
    )

    return albedo_recon
