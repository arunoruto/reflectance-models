import argparse
import os
import time

os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax
import jax.numpy as jnp
import numpy as np
from astropy.io import fits

from refmod.dtm_helper import dtm2grad
from refmod.hapke import (
    amsa,
    dhg_legendre_coefficients,
    invert_amsa_precomputed,
    prepare_amsa_inversion,
)
from refmod.hapke.inverse import invert_amsa


def _load_hopper(pixel_side: int | None):
    with fits.open("test/data/hopper_amsa.fits") as f:
        result = f["result"].data.astype(np.float64)
        i_deg = f["result"].header["i"]
        e_deg = f["result"].header["e"]
        b = f["result"].header["b"]
        c = f["result"].header["c"]
        hs = f["result"].header["hs"]
        bs0 = f["result"].header["bs0"]
        tb = f["result"].header["tb"]
        hc = f["result"].header["hc"]
        bc0 = f["result"].header["bc0"]
        albedo = f["albedo"].data.astype(np.float64)
        dtm = f["dtm"].data.astype(np.float64)
        resolution = f["dtm"].header["res"]

    n = dtm2grad(dtm, resolution, normalize=False)
    u, v = result.shape

    if pixel_side is not None:
        half = pixel_side // 2
        uc = u // 2 + np.arange(-half, half)
        vc = v // 2 + np.arange(-half, half)
        albedo = albedo[uc, :][:, vc]
        n = n[uc, :, :][:, vc, :]

    i_rad = np.deg2rad(i_deg)
    e_rad = np.deg2rad(e_deg)
    w = albedo.reshape(-1)
    n_flat = n.reshape(-1, 3)
    i_flat = np.tile(np.array([np.sin(i_rad), 0.0, np.cos(i_rad)]), (w.shape[0], 1))
    e_flat = np.tile(np.array([np.sin(e_rad), 0.0, np.cos(e_rad)]), (w.shape[0], 1))

    return (
        jnp.asarray(w),
        dhg_legendre_coefficients(b, c),
        jnp.asarray(i_flat),
        jnp.asarray(e_flat),
        jnp.asarray(n_flat),
        dict(roughness=tb, h_sh=hs, b0_sh=bs0, h_cb=hc, b0_cb=bc0),
    )


def _time_once(fn):
    start = time.perf_counter()
    out = fn()
    if hasattr(out, "block_until_ready"):
        out.block_until_ready()
    return time.perf_counter() - start, out


def _time_repeated(label: str, fn, repeats: int):
    samples = []
    for _ in range(repeats):
        elapsed, _ = _time_once(fn)
        samples.append(elapsed)
    arr = np.array(samples)
    print(
        f"{label}: mean={arr.mean():.6f}s median={np.median(arr):.6f}s "
        f"min={arr.min():.6f}s max={arr.max():.6f}s n={repeats}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pixel-side", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    print(f"backend={jax.default_backend()}")
    print(f"devices={jax.devices()}")
    print(f"x64={jax.config.jax_enable_x64}")
    print(f"jax_cache={os.environ.get('JAX_COMPILATION_CACHE_DIR')}")

    w, b_n, i, e, n, params = _load_hopper(args.pixel_side)
    print(f"pixels={w.shape[0]}")

    first_forward, refl = _time_once(lambda: amsa(w, b_n, i, e, n, **params))
    print(f"forward_first: {first_forward:.6f}s")
    _time_repeated(
        "forward_steady", lambda: amsa(w, b_n, i, e, n, **params), args.repeats
    )

    first_inverse, w_rec = _time_once(lambda: invert_amsa(refl, b_n, i, e, n, **params))
    print(f"inverse_first: {first_inverse:.6f}s")
    _time_repeated(
        "inverse_steady",
        lambda: invert_amsa(refl, b_n, i, e, n, **params),
        args.repeats,
    )

    prepare_time, state = _time_once(
        lambda: prepare_amsa_inversion(b_n, i, e, n, **params)
    )
    print(f"prepare_inversion: {prepare_time:.6f}s")
    first_precomputed, w_rec_pre = _time_once(
        lambda: invert_amsa_precomputed(refl, state)
    )
    print(f"inverse_precomputed_first: {first_precomputed:.6f}s")
    _time_repeated(
        "inverse_precomputed_steady",
        lambda: invert_amsa_precomputed(refl, state),
        args.repeats,
    )

    err = float(jnp.nanmax(jnp.abs(w_rec - w)))
    err_pre = float(jnp.nanmax(jnp.abs(w_rec_pre - w)))
    print(f"max_inverse_error={err:.3e}")
    print(f"max_precomputed_inverse_error={err_pre:.3e}")


if __name__ == "__main__":
    main()
