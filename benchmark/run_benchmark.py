"""Time the reflectance models on the workload shared with the MATLAB toolbox.

Reads the same ``benchmark_spec.json`` the MATLAB side reads and writes the
same result schema, so the two JSON files can be compared directly by
``compare.py``.

Method, and the reasons behind it:

* **JAX dispatches asynchronously.** Timing a call without
  ``block_until_ready()`` measures the time to enqueue work, not to do it, and
  produces impressively meaningless numbers. Every timed region blocks.
* **Compilation is reported separately, not amortised into the steady state.**
  The first call to a jitted function compiles; that cost is real but is paid
  once, so folding it into a per-call average would misrepresent both the
  first call and the rest. ``compile_ms`` is measured explicitly.
* **The minimum over repeats is reported**, matching the MATLAB harness.
  Scheduler and cache noise can only make a run slower, so the minimum is the
  cleanest estimate of the underlying cost. Mean and standard deviation are
  recorded too, so a noisy case stays visible.
* **Inputs are built once per case and moved to the device before timing**, so
  host-to-device transfer is not counted. This mirrors MATLAB, where the array
  is already in the process's memory.
* Geometry is generated from the spec seed with the same construction as the
  MATLAB harness, so both time numerically identical work.

Usage::

    devenv shell -- uv run python benchmark/run_benchmark.py
    devenv shell -- uv run python benchmark/run_benchmark.py --spec path/to/spec.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from refmod.hapke import (  # noqa: E402
    amsa,
    dhg_legendre_coefficients,
    imsa_modified_h,
)

DEFAULT_SPEC = (
    Path(__file__).resolve().parents[2]
    / "ReflectanceModels"
    / "matlab"
    / "benchmark"
    / "benchmark_spec.json"
)


def make_inputs(spec: dict, n_pixels: int):
    """Deterministic geometry matching the MATLAB harness construction."""
    rng = np.random.RandomState(spec["seed"])

    theta_s = np.deg2rad(spec["sun_zenith_deg"])
    theta_v = np.deg2rad(spec["view_zenith_deg"])

    s = np.zeros((n_pixels, 3))
    s[:, 1] = np.sin(theta_s)
    s[:, 2] = np.cos(theta_s)

    v = np.zeros((n_pixels, 3))
    v[:, 0] = np.sin(theta_v)
    v[:, 2] = np.cos(theta_v)

    n = np.zeros((n_pixels, 3))
    n[:, 0] = 0.2 * (2 * rng.rand(n_pixels) - 1)
    n[:, 1] = 0.2 * (2 * rng.rand(n_pixels) - 1)
    n[:, 2] = 1.0
    n /= np.linalg.norm(n, axis=1, keepdims=True)

    w = 0.1 + 0.8 * rng.rand(n_pixels)

    # Move to the device up front: transfer is not part of what we measure.
    return (
        jax.device_put(jnp.asarray(s)),
        jax.device_put(jnp.asarray(v)),
        jax.device_put(jnp.asarray(n)),
        jax.device_put(jnp.asarray(w)),
    )


def build_callable(spec: dict, case: dict):
    """Return a zero-argument jitted callable for one benchmark case."""
    p = spec["params"]
    tb = float(np.deg2rad(case["tb_deg"]))
    s, v, n, w = make_inputs(spec, case["pixels"])

    model = case["model"]
    if model in ("hapke_amsa", "hapke_amsa_fast"):
        # refmod has a single AMSA implementation. MATLAB keeps a
        # loop-optimised twin; both map here, and the comparison report notes
        # that the Python column is the same code in both rows.
        b_n = dhg_legendre_coefficients(float(p["b"]), float(p["c"]), 15)
        fn = jax.jit(
            lambda w_, s_, v_, n_: amsa(
                w_, b_n, s_, v_, n_, tb, float(p["hs"]), float(p["Bs0"]),
                float(p["hc"]), float(p["Bc0"]),
            )
        )
    elif model == "hapke_imsa":
        fn = jax.jit(
            lambda w_, s_, v_, n_: imsa_modified_h(
                w_, float(p["b"]), float(p["c"]), s_, v_, n_,
                tb, float(p["hs"]), float(p["Bs0"]),
            )
        )
    else:
        raise ValueError(f"no benchmark mapping for {model}")

    return lambda: fn(w, s, v, n)


def time_case(spec: dict, case: dict) -> dict:
    run = build_callable(spec, case)

    # Compilation: first call includes tracing + XLA compile.
    t0 = time.perf_counter()
    jax.block_until_ready(run())
    compile_ms = (time.perf_counter() - t0) * 1000

    for _ in range(int(spec["warmup"])):
        jax.block_until_ready(run())

    times_ms = []
    for _ in range(int(spec["repeats"])):
        t0 = time.perf_counter()
        jax.block_until_ready(run())
        times_ms.append((time.perf_counter() - t0) * 1000)

    times = np.asarray(times_ms)
    return {
        "name": case["name"],
        "model": case["model"],
        "pixels": case["pixels"],
        "tb_deg": case["tb_deg"],
        "min_ms": float(times.min()),
        "mean_ms": float(times.mean()),
        "std_ms": float(times.std(ddof=1)) if times.size > 1 else 0.0,
        "compile_ms": float(compile_ms),
        "mpixels_per_s": float(case["pixels"] / (times.min() / 1000) / 1e6),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument(
        "--out", type=Path, default=Path(__file__).parent / "benchmark-python.json"
    )
    args = ap.parse_args()

    spec = json.loads(args.spec.read_text())
    device = jax.devices()[0]

    print(
        f"Benchmarking {len(spec['cases'])} case(s), {spec['repeats']} repeat(s) "
        f"after {spec['warmup']} warmup(s) on {device.platform}:{device.device_kind}"
    )

    results = []
    for case in spec["cases"]:
        r = time_case(spec, case)
        results.append(r)
        print(
            f"  {r['name']:<20} {r['pixels']:>10d} px  {r['min_ms']:9.3f} ms  "
            f"{r['mpixels_per_s']:8.2f} Mpx/s  (compile {r['compile_ms']:.0f} ms)"
        )

    out = {
        "schema": "reflectance-benchmark-result/1",
        "implementation": "python-jax",
        "version": f"jax {jax.__version__}",
        "platform": f"{platform.machine()}/{device.platform}:{device.device_kind}",
        "threads": None,
        "repeats": spec["repeats"],
        "warmup": spec["warmup"],
        "x64": bool(jax.config.jax_enable_x64),
        "cases": results,
    }
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
