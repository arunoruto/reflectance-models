# Cross-implementation benchmarks

Times the same reflectance models in this package and in the MATLAB toolbox
(`MatlabToolboxes/ReflectanceModels`) on an identical workload, so the two are
directly comparable.

Both harnesses read one shared spec —
`ReflectanceModels/matlab/benchmark/benchmark_spec.json` — and emit the same
`reflectance-benchmark-result/1` JSON schema. Geometry is generated from the
same seed with the same construction on both sides, so they time numerically
identical work rather than merely similar-looking work.

## Running

```sh
# MATLAB side (needs the image and the campus network)
cd ../ReflectanceModels/matlab && devenv shell -- benchmark

# Python side
devenv shell -- uv run python benchmark/run_benchmark.py --out benchmark/benchmark-python-cpu.json

# Compare
devenv shell -- uv run python benchmark/compare.py \
    ../ReflectanceModels/matlab/build/benchmark-matlab.json \
    benchmark/benchmark-python-cpu.json
```

## Methodology

Microbenchmarks are easy to get wrong in ways that flatter one side, so the
choices here are deliberate and symmetric:

- **JAX dispatches asynchronously.** Timing a call without
  `block_until_ready()` measures how long it takes to *enqueue* work, not to do
  it, and yields impressively meaningless numbers. Every timed region blocks.
- **Warmup iterations are discarded on both sides.** MATLAB JIT-compiles on
  first execution; JAX traces and compiles. Neither first call is
  representative.
- **JAX compile time is reported separately, never amortised.** Folding it into
  a per-call mean would misrepresent both the first call and every subsequent
  one. It is a one-off cost *per shape*: for a pipeline calling a model
  repeatedly at fixed size it rounds to nothing; for a single call on a fresh
  shape it dominates. Both numbers appear in the output so neither reading can
  be hidden.
- **The minimum over repeats is reported.** Scheduling and cache noise can only
  make a run slower, so the minimum is the cleanest estimate of the underlying
  cost. Mean and standard deviation are recorded too, so a pathologically noisy
  case stays visible.
- **Inputs are built once per case and placed on the device before timing**, so
  neither allocation nor host-to-device transfer is counted. This matches
  MATLAB, where the array is already in the process's memory.
- Both run in **double precision** (`jax_enable_x64=True`), because the models
  need it and MATLAB has no single-precision mode here.

## Reading the results

Two caveats worth keeping in mind:

1. **`amsa_fast` is one function in Python and two in MATLAB.** The toolbox
   keeps `hapke_amsa` and a loop-optimised `hapke_amsa_fast`; this package has
   a single AMSA implementation, so both rows run the same Python code. A lower
   "speedup" on the `amsa_fast` rows therefore means MATLAB got faster, not
   that Python got slower.
2. **Per-call time is not the whole story for the inverse problem.** The
   Levenberg-Marquardt solver calls the forward model tens of times per pixel
   inside a compiled loop, where JAX's advantage compounds and the compile cost
   is paid once. A forward-model microbenchmark understates that.

## Results

Measured on this development host: AMD x86_64, 8 MATLAB compute threads,
MATLAB R2025a Update 1, JAX 0.8.1, double precision.

| case | pixels | MATLAB (ms) | JAX CPU (ms) | speedup |
|---|---:|---:|---:|---:|
| `amsa_smooth_1k` | 1 000 | 0.785 | 0.072 | 10.96× |
| `amsa_smooth_10k` | 10 000 | 3.520 | 0.432 | 8.15× |
| `amsa_smooth_100k` | 100 000 | 40.191 | 2.604 | 15.44× |
| `amsa_smooth_1m` | 1 000 000 | 548.123 | 40.910 | 13.40× |
| `amsa_rough_10k` | 10 000 | 5.497 | 0.733 | 7.50× |
| `amsa_rough_100k` | 100 000 | 46.651 | 4.007 | 11.64× |
| `amsa_rough_1m` | 1 000 000 | 679.392 | 56.683 | 11.99× |
| `amsa_fast_100k` | 100 000 | 29.287 | 3.936 | 7.44× |
| `amsa_fast_1m` | 1 000 000 | 377.561 | 56.710 | 6.66× |
| `imsa_rough_100k` | 100 000 | 27.015 | 4.485 | 6.02× |
| `imsa_rough_1m` | 1 000 000 | 348.396 | 53.469 | 6.52× |

**Geometric mean: 9.15× in favour of JAX on CPU.** JAX compile time was
210–430 ms per shape.

### GPU (partial)

On an NVIDIA TITAN X (Pascal) — a 2016 card whose FP64 throughput is 1/32 of
FP32, so this is double precision on hardware that is actively bad at it:

| case | pixels | MATLAB (ms) | JAX CPU (ms) | JAX GPU (ms) | GPU vs MATLAB |
|---|---:|---:|---:|---:|---:|
| `amsa_smooth_1k` | 1 000 | 0.785 | 0.072 | 0.095 | 8.3× |
| `amsa_smooth_10k` | 10 000 | 3.520 | 0.432 | 0.177 | 19.9× |
| `amsa_smooth_100k` | 100 000 | 40.191 | 2.604 | 0.696 | 57.7× |
| `amsa_smooth_1m` | 1 000 000 | 548.123 | 40.910 | 5.218 | 105.0× |

The crossover matters more than the peak: **below roughly 1 000 pixels the CPU
is faster**, because kernel-launch overhead dominates the actual work. At 1 M
pixels the GPU reaches 192 Mpx/s.

### A trap worth knowing about: JAX preallocates the GPU

**JAX reserves ~75 % of VRAM on first use and holds it for the process
lifetime.** Two consequences bit this benchmark before the cause was
understood:

- A benchmark process that hangs or is killed uncleanly keeps its reservation.
  A second run then finds only scraps, completes the small cases, and dies on
  a larger one — with no traceback and no CUDA diagnostic, because the failure
  is a signal rather than an exception. It reads exactly like a compiler
  pathology and is not one.
- `pgrep -f run_benchmark.py` does **not** find such a process, because under
  `uv run` the process name is `uv`. Use
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` to see who is
  actually holding the device.

Before trusting any GPU timing, check the device is idle:

```sh
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
```

Running with `XLA_PYTHON_CLIENT_PREALLOCATE=false` makes the harness allocate
on demand, which is friendlier when anything else shares the GPU, at the cost
of some allocator churn.

### The roughness kernel does not compile on this GPU

Separately from the leak above — and confirmed only after ruling it out — the
`tb > 0` cases genuinely do not finish compiling for the CUDA backend:

| kernel | GPU compile time |
|---|---|
| AMSA, `tb = 0` (smooth) | 237–291 ms |
| AMSA, `tb = 20°` (rough) | **> 31 min, abandoned** |

Measured on an idle GPU with preallocation disabled: 31 min of CPU time,
99.8 % on a single thread, GPU utilisation 7 %. The work is entirely in XLA's
compiler, not on the device. The same kernel compiles for the CPU backend in
about 430 ms.

The plausible cause is the `ile`/`ige` branch structure in the roughness
correction. Both branches are evaluated and selected with `jnp.where`, which
roughly doubles an already large straight-line graph, and the smooth path
avoids it entirely by short-circuiting on `roughness < EPS`.

This is worth treating as a real limitation rather than a curiosity: it means
**GPU execution is currently only usable for smooth-surface work**, which for
planetary applications is the less interesting half. It is also a concrete
optimisation target — restructuring the roughness branch to reduce graph size
would help compile time on both backends.

Observations worth acting on:

- MATLAB's own `hapke_amsa_fast` optimisation is real and measurable: 29.3 ms
  versus 40.2 ms at 100 k pixels, about 27 % faster. It earns its place.
- The roughness correction costs MATLAB about 16 % (40.2 → 46.7 ms at 100 k)
  and JAX about 54 % (2.6 → 4.0 ms). Roughness is a larger *relative* cost in
  the compiled version, which suggests the branchy `ile`/`ige` structure
  vectorises less well than the straight-line smooth path.
- Throughput peaks near 100 k pixels on both sides and drops at 1 M, on both
  implementations — a cache-capacity effect rather than anything
  language-specific.
