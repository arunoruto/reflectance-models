"""Continuum removal utilities.

Implements an "upper convex hull" continuum commonly used in spectroscopy:

- Treat each band as a 2D point (x=λ, y=R)
- Compute the *upper* convex envelope
- Linearly interpolate this envelope to obtain a continuum curve
- Divide the spectrum by the continuum to remove the continuum slope

Two backends are supported:

- ``method='numpy'`` (default): deterministic monotone-chain upper hull.
- ``method='scipy'``: uses ``scipy.spatial.ConvexHull`` to obtain the convex
  polygon, then extracts the upper chain.
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np

_Method = Literal["numpy", "scipy"]


def _upper_monotone_chain(xv: np.ndarray, yv: np.ndarray) -> np.ndarray:
    """Compute upper monotone chain indices using Andrew's monotone chain algorithm.

    Assumes xv is strictly increasing.
    """
    n = xv.size
    hull = np.empty(n, dtype=np.int64)
    h_len = 0

    for i in range(n):
        while h_len >= 2:
            # cross(o, a, b) >= 0 checks if the turn o->a->b is left or straight
            o = hull[h_len - 2]
            a = hull[h_len - 1]
            b = i

            # Cross product of (a-o) and (b-o)
            val = (xv[a] - xv[o]) * (yv[b] - yv[o]) - (yv[a] - yv[o]) * (xv[b] - xv[o])

            if val >= 0.0:
                h_len -= 1
            else:
                break

        hull[h_len] = i
        h_len += 1

    return hull[:h_len]


def smooth_spectrum_m3(
    wavelength: np.ndarray,
    reflectance: np.ndarray,
    *,
    mu: float = 1000.0,
    assume_sorted: bool = False,
) -> np.ndarray:
    """Smooth a 1D spectrum using the M3 regularized smoother.

    This matches the approach from the M3 MATLAB code: solve a regularized
    least-squares problem with a scaled second-difference operator on an
    irregular wavelength grid.

    Notes
    -----
    - The original MATLAB code assumes wavelength in nanometers with ~20-40 nm
      spacing. To keep ``mu`` comparable, we smooth in nm units (auto-convert if
      input looks like microns).
    - Endpoints are preserved via the identity term in the system.
    """

    wl_in = np.asarray(wavelength, dtype=float).reshape(-1)
    r_in = np.asarray(reflectance, dtype=float).reshape(-1)
    if wl_in.shape != r_in.shape:
        raise ValueError("wavelength and reflectance must have same shape")

    finite = np.isfinite(wl_in) & np.isfinite(r_in)
    if not np.any(finite):
        raise ValueError("No finite samples")

    wl0 = wl_in[finite]
    r0 = r_in[finite]

    if assume_sorted:
        order0 = np.arange(wl0.size)
    else:
        order0 = np.argsort(wl0)
        wl0 = wl0[order0]
        r0 = r0[order0]

    n = int(wl0.size)
    if n < 3:
        out = np.full_like(wl_in, np.nan, dtype=float)
        if assume_sorted:
            out[finite] = r0
        else:
            unsorted = np.empty_like(r0)
            unsorted[order0] = r0
            out[finite] = unsorted
        return out

    wl_nm = wl0 * (1000.0 if float(np.nanmax(wl0)) <= 10.0 else 1.0)

    dist = np.zeros(n, dtype=float)
    dist[1:] = np.diff(wl_nm)

    chann_mean = np.zeros(n, dtype=float)
    chann_mean[1:] = 0.5 * (wl_nm[1:] + wl_nm[:-1])

    # Build the tri-diagonal second-difference-like operator.
    main = np.zeros(n, dtype=float)
    off_l = np.zeros(n - 1, dtype=float)
    off_r = np.zeros(n - 1, dtype=float)
    for i in range(1, n - 1):
        di = float(dist[i])
        dip1 = float(dist[i + 1])
        if di <= 0.0 or dip1 <= 0.0:
            raise ValueError("wavelength must be strictly increasing")
        main[i] = (1.0 / di) + (1.0 / dip1)
        off_l[i - 1] = -1.0 / di
        off_r[i] = -1.0 / dip1

    dmat = np.diag(main) + np.diag(off_r, 1) + np.diag(off_l, -1)

    # Scale interior rows like MATLAB:
    # mu ./ (chann_mean(2:end-1) - chann_mean(1:end-2))
    denom = chann_mean[1 : n - 1] - chann_mean[0 : n - 2]
    denom = np.where(np.abs(denom) > 0.0, denom, np.nan)
    scale = float(mu) / denom
    scale = np.where(np.isfinite(scale), scale, 0.0)

    a_top = np.eye(n, dtype=float)
    a_bottom = (scale[:, None]) * dmat[1 : n - 1, :]
    a = np.vstack([a_top, a_bottom])
    b = np.concatenate([r0, np.zeros(n - 2, dtype=float)])

    r_smooth0, *_ = np.linalg.lstsq(a, b, rcond=None)
    r_smooth0 = np.asarray(r_smooth0, dtype=float).reshape(-1)

    out = np.full_like(wl_in, np.nan, dtype=float)
    unsorted = np.empty_like(r_smooth0)
    unsorted[order0] = r_smooth0
    out[finite] = unsorted
    return out


def upper_hull_continuum(
    wavelength: np.ndarray,
    reflectance: np.ndarray,
    *,
    assume_sorted: bool = False,
    method: _Method = "numpy",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute an upper-hull continuum for a spectrum.

    Parameters
    ----------
    wavelength:
        1D wavelengths.
    reflectance:
        1D reflectance values, same shape as wavelength.
    assume_sorted:
        If True, assumes wavelength is already increasing.
    method:
        "numpy" (default) or "scipy".

    Returns
    -------
    continuum:
        Continuum evaluated at every input wavelength.
    hull_indices:
        Indices (into the original input arrays) of hull anchor points.

    Notes
    -----
    - Non-finite samples are ignored.
    - Duplicate wavelengths are collapsed (keeping max reflectance).
    - ``hull_indices`` are approximate when duplicates exist (first occurrence).
    """

    wl_in = np.asarray(wavelength, dtype=float).reshape(-1)
    r_in = np.asarray(reflectance, dtype=float).reshape(-1)
    if wl_in.shape != r_in.shape:
        raise ValueError("wavelength and reflectance must have same shape")

    method_s: str = str(method).lower().strip()
    if method_s not in {"numpy", "scipy"}:
        raise ValueError("method must be 'numpy' or 'scipy'")
    method = cast(_Method, method_s)

    finite = np.isfinite(wl_in) & np.isfinite(r_in)
    if not np.any(finite):
        raise ValueError("No finite samples")

    wl0 = wl_in[finite]
    r0 = r_in[finite]

    if assume_sorted:
        order0 = np.arange(wl0.size)
    else:
        order0 = np.argsort(wl0)
        wl0 = wl0[order0]
        r0 = r0[order0]

    if wl0.size == 1:
        continuum = np.full_like(wl_in, np.nan, dtype=float)
        continuum[finite] = r0
        hull_indices = np.array([np.where(finite)[0][order0[0]]], dtype=int)
        return continuum, hull_indices

    # Collapse duplicates by keeping the maximum reflectance at each wavelength.
    x, inv = np.unique(wl0, return_inverse=True)
    y = np.full(x.shape, -np.inf, dtype=float)
    np.maximum.at(y, inv, r0)

    if method == "scipy":
        try:
            from scipy.spatial import ConvexHull  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("method='scipy' requires SciPy installed") from e

        pts = np.column_stack([x, y])
        if pts.shape[0] < 3:
            upper_idx = np.arange(pts.shape[0], dtype=int)
        else:
            hull = ConvexHull(pts)
            verts = np.asarray(hull.vertices, dtype=int)

            left = int(np.argmin(x))
            right = int(np.argmax(x))

            pos_left = int(np.where(verts == left)[0][0])
            pos_right = int(np.where(verts == right)[0][0])

            def walk(a: int, b: int) -> np.ndarray:
                if a <= b:
                    out = verts[a : b + 1]
                else:
                    out = np.concatenate([verts[a:], verts[: b + 1]])
                return np.asarray(out, dtype=int)

            chain1 = walk(pos_left, pos_right)
            chain2 = walk(pos_right, pos_left)

            def chain_interp(chain: np.ndarray) -> np.ndarray:
                xs = x[chain]
                ys = y[chain]
                o = np.argsort(xs)
                xs = xs[o]
                ys = ys[o]
                ux, invu = np.unique(xs, return_inverse=True)
                my = np.full(ux.shape, -np.inf, dtype=float)
                np.maximum.at(my, invu, ys)
                return np.interp(x, ux, my)

            y1 = chain_interp(chain1)
            y2 = chain_interp(chain2)
            upper = chain1 if float(np.mean(y1 - y2)) >= 0.0 else chain2

            upper_idx = np.unique(upper)
            upper_idx = upper_idx[np.argsort(x[upper_idx])]

        # Enforce convexity and remove any degenerate points.
        upper_x = x[upper_idx]
        upper_y = y[upper_idx]
        if upper_x.size > 1 and np.any(np.diff(upper_x) <= 0):
            upper_x, uniq = np.unique(upper_x, return_index=True)
            upper_y = upper_y[uniq]
        rel = _upper_monotone_chain(upper_x, upper_y)
        hull_x = upper_x[rel]
        hull_y = upper_y[rel]
    else:
        hull_idx = _upper_monotone_chain(x, y)
        hull_x = x[hull_idx]
        hull_y = y[hull_idx]

    # Ensure hull includes endpoints so interpolation spans wl0.
    if hull_x[0] > wl0.min():
        hull_x = np.insert(hull_x, 0, wl0.min())
        hull_y = np.insert(hull_y, 0, float(np.interp(wl0.min(), wl0, r0)))
    if hull_x[-1] < wl0.max():
        hull_x = np.append(hull_x, wl0.max())
        hull_y = np.append(hull_y, float(np.interp(wl0.max(), wl0, r0)))

    # Deduplicate x for stable interpolation.
    ux, invu = np.unique(hull_x, return_inverse=True)
    my = np.full(ux.shape, -np.inf, dtype=float)
    np.maximum.at(my, invu, hull_y)
    hull_x = ux
    hull_y = my

    continuum0 = np.interp(wl0, hull_x, hull_y)

    continuum = np.full_like(wl_in, np.nan, dtype=float)
    cont_unsorted = np.empty_like(continuum0)
    cont_unsorted[order0] = continuum0
    continuum[finite] = cont_unsorted

    # Approximate hull indices w.r.t. original input ordering (first occurrence).
    original_sorted_idx = np.where(finite)[0][order0]
    hull_indices = np.array(
        [
            original_sorted_idx[int(np.searchsorted(wl0, hx, side="left"))]
            for hx in hull_x
        ],
        dtype=int,
    )

    return continuum, hull_indices


def continuum_remove_upper_hull(
    wavelength: np.ndarray,
    reflectance: np.ndarray,
    *,
    eps: float = 1e-12,
    smooth: bool = True,
    smooth_mu: float = 1000.0,
    assume_sorted: bool = False,
    method: _Method = "numpy",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove continuum slope using an upper-hull continuum.

    The continuum-removed spectrum is computed as:
    ``reflectance / max(continuum, eps)``.

    Parameters
    ----------
    wavelength:
        1D wavelengths.
    reflectance:
        1D reflectance values, same shape as wavelength.
    eps:
        Small value to avoid division by zero (or very small continuum values).
    smooth:
        If True, applies M3 smoothing (via :func:`smooth_spectrum_m3`) to the
        reflectance *before* computing the hull. This reduces the sensitivity of
        the upper hull to noise.
    smooth_mu:
        Regularization parameter for the M3 smoother (if ``smooth=True``).
    assume_sorted:
        If True, assumes wavelength is already increasing.
    method:
        "numpy" (default) or "scipy".

    Returns
    -------
    continuum_removed:
        The continuum-removed spectrum.
    continuum:
        The computed continuum.
    hull_indices:
        Indices (into the input arrays) of the hull anchor points.
    """

    r = np.asarray(reflectance, dtype=float)
    r_for_hull = (
        smooth_spectrum_m3(
            wavelength, r, mu=float(smooth_mu), assume_sorted=assume_sorted
        )
        if bool(smooth)
        else r
    )

    continuum, hull_idx = upper_hull_continuum(
        wavelength, r_for_hull, assume_sorted=assume_sorted, method=method
    )
    cont = np.asarray(continuum, dtype=float)
    denom = np.maximum(cont, float(eps))
    cr = r / denom
    return cr, cont, hull_idx
