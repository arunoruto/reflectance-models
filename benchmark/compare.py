"""Compare MATLAB and Python benchmark results side by side.

Both harnesses write ``reflectance-benchmark-result/1`` JSON from the same
workload spec, so this only has to join on case name and format the outcome.

Usage::

    devenv shell -- uv run python benchmark/compare.py \
        ../ReflectanceModels/matlab/build/benchmark-matlab.json \
        benchmark/benchmark-python.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    d = json.loads(path.read_text())
    if d.get("schema") != "reflectance-benchmark-result/1":
        raise SystemExit(f"{path}: unexpected schema {d.get('schema')!r}")
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("matlab", type=Path)
    ap.add_argument("python", type=Path)
    args = ap.parse_args()

    ml, py = load(args.matlab), load(args.python)
    ml_cases = {c["name"]: c for c in ml["cases"]}
    py_cases = {c["name"]: c for c in py["cases"]}

    print(f"MATLAB : {ml['version']} on {ml['platform']}, {ml['threads']} threads")
    print(f"Python : {py['version']} on {py['platform']}, x64={py.get('x64')}")
    print(f"Both   : min of {ml['repeats']} repeats after {ml['warmup']} warmups\n")

    hdr = (
        f"{'case':<20} {'pixels':>9} {'MATLAB ms':>10} {'JAX ms':>10} "
        f"{'speedup':>8}  {'JAX compile':>11}"
    )
    print(hdr)
    print("-" * len(hdr))

    speedups = []
    for name in [c["name"] for c in ml["cases"]]:
        m, p = ml_cases.get(name), py_cases.get(name)
        if not m or not p:
            continue
        ratio = m["min_ms"] / p["min_ms"]
        speedups.append((name, ratio))
        faster = "JAX" if ratio > 1 else "MATLAB"
        # MATLAB's jsonencode emits all numbers as floats, so coerce.
        print(
            f"{name:<20} {int(m['pixels']):>9d} {m['min_ms']:>10.3f} "
            f"{p['min_ms']:>10.3f} {ratio:>7.2f}x {faster:<7} "
            f"{p['compile_ms']:>7.0f} ms"
        )

    print()
    if speedups:
        best = max(speedups, key=lambda kv: kv[1])
        worst = min(speedups, key=lambda kv: kv[1])
        geo = 1.0
        for _, r in speedups:
            geo *= r
        geo **= 1.0 / len(speedups)
        print(f"geometric mean speedup (JAX vs MATLAB): {geo:.2f}x")
        print(f"  best  for JAX: {best[0]} ({best[1]:.2f}x)")
        print(f"  worst for JAX: {worst[0]} ({worst[1]:.2f}x)")

    print(
        "\nNote: JAX compile time is one-off per shape and is excluded from the\n"
        "per-call figures. For a pipeline that calls a model repeatedly at a\n"
        "fixed size it is amortised to nothing; for a single call on a new\n"
        "shape it dominates. Both numbers are reported so neither reading is\n"
        "hidden."
    )


if __name__ == "__main__":
    main()
