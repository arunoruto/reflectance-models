import numpy as np
import numpy.typing as npt


def normalize(x: npt.NDArray, axis: int = -1) -> npt.NDArray:
    """Normalizes a vector or a batch of vectors.

    Calculates the L2 norm (Euclidean norm) of the input array along the
    specified axis.

    Parameters
    ----------

    x : npt.NDArray
        Input array representing a vector or a batch of vectors.
    axis : int, optional
        Axis along which to compute the norm, by default -1.

    Returns
    -------
    npt.NDArray
        The L2 norm of the input array. If `x` is a batch of vectors,
        the output will be an array of norms.
    """
    temp = np.sum(x**2, axis=axis)
    if isinstance(temp, float):
        temp = np.array([temp])
    return np.sqrt(temp)


def normalize_keepdims(x: npt.NDArray, axis: int = -1) -> npt.NDArray:
    """Normalizes a vector or batch of vectors, keeping dimensions.

    Calculates the L2 norm of the input array along the specified axis,
    then expands the dimensions of the output to match the input array's
    dimension along the normalization axis. This is useful for broadcasting
    the norm for division.

    Parameters
    ----------

    x : npt.NDArray
        Input array representing a vector or a batch of vectors.
    axis : int, optional
        Axis along which to compute the norm, by default -1.

    Returns
    -------
    npt.NDArray
        The L2 norm of the input array, with dimensions kept for broadcasting.
    """
    temp = np.sqrt(np.sum(x**2, axis=axis))
    if isinstance(temp, float):
        temp = np.array(temp)
        # temp = np.array([temp])
    return np.expand_dims(temp, axis=axis)


def angle_processing_base(
    vec_a: npt.NDArray, vec_b: npt.NDArray, axis: int = -1
) -> tuple[npt.NDArray, npt.NDArray]:
    """Computes cosine and sine of the angle between two vectors.

    Parameters
    ----------

    vec_a : npt.NDArray
        First vector or batch of vectors.
    vec_b : npt.NDArray
        Second vector or batch of vectors. Must have the same shape as vec_a.
    axis : int, optional
        Axis along which the dot product is performed, by default -1.

    Returns
    -------
    tuple[npt.NDArray, npt.NDArray]
        A tuple containing:
            - cos_phi : npt.NDArray
                Cosine of the angle(s) between vec_a and vec_b.
            - sin_phi : npt.NDArray
                Sine of the angle(s) between vec_a and vec_b.
    """
    cos_phi = np.sum(vec_a * vec_b, axis=axis)
    cos_phi = np.array([cos_phi]) if isinstance(cos_phi, float) else cos_phi
    cos_phi = np.clip(cos_phi, -1, 1)
    sin_phi = np.sqrt(1 - cos_phi**2)
    return cos_phi, sin_phi


def angle_processing(
    vec_a: npt.NDArray, vec_b: npt.NDArray, axis: int = -1
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Computes various trigonometric quantities related to the angle between two vectors.

    Parameters
    ----------

    vec_a : npt.NDArray
        First vector or batch of vectors.
    vec_b : npt.NDArray
        Second vector or batch of vectors. Must have the same shape as vec_a.
    axis : int, optional
        Axis along which the dot product is performed, by default -1.

    Returns
    -------
    tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]
        A tuple containing:
            - cos_phi : npt.NDArray
                Cosine of the angle(s) between vec_a and vec_b.
            - sin_phi : npt.NDArray
                Sine of the angle(s) between vec_a and vec_b.
            - cot_phi : npt.NDArray
                Cotangent of the angle(s) between vec_a and vec_b.
                (Returns np.inf where sin_phi is 0).
            - i : npt.NDArray
                The angle(s) in radians between vec_a and vec_b (i.e., arccos(cos_phi)).
    """
    cos_phi, sin_phi = angle_processing_base(vec_a, vec_b, axis)
    cot_phi = np.where(sin_phi == 0, np.inf, cos_phi / sin_phi)
    i = np.arccos(cos_phi)
    return cos_phi, sin_phi, cot_phi, i
