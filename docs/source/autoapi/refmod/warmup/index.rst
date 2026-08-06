refmod.warmup
=============

.. py:module:: refmod.warmup

.. autoapi-nested-parse::

   Ahead-of-time compilation warmup and persistent compilation caching.

   XLA compilation of the refmod kernels can be expensive on GPU — pathologically
   so when the device memory is nearly full, because XLA's autotuner allocates
   real scratch buffers while timing kernel candidates. Two mitigations:

   1. :func:`enable_compilation_cache` — persist compiled kernels to disk so any
      compile is paid at most once per machine.
   2. :func:`warmup` — trigger the compiles up front (ideally right after import,
      while the GPU is still empty), so they never interleave with real work.

   The Levenberg-Marquardt inversion kernel is specialized on the *pixel count*
   and ``max_steps`` only; all Hapke parameters (roughness, opposition effects,
   phase-function coefficients) enter through runtime arrays. A single warmup per
   chunk shape therefore covers every parameter combination. Chunk sizes produced
   by :func:`refmod.hapke.prepare_amsa_inversion` are powers of two, so warming
   the expected chunk size (or a small ladder of sizes) is enough.

   Usage::

       import refmod.warmup
       refmod.warmup.enable_compilation_cache()   # before any jit executes
       refmod.warmup.warmup(n_pixels=(1 << 20,))  # while the GPU is still empty

   or from the command line::

       python -m refmod.warmup --pixels 1048576 --max-steps 40





Module Contents
---------------

.. py:function:: enable_compilation_cache(cache_dir = None, min_compile_time_secs = 0.0)

   Enable JAX's persistent compilation cache (if not already enabled).

   If a cache directory is already configured — e.g. via the
   ``JAX_COMPILATION_CACHE_DIR`` environment variable, as this project's
   devenv does — and no explicit *cache_dir* is passed, the existing
   configuration is kept and returned. Passing an explicit *cache_dir*
   overrides it (resetting an already-initialized cache).

   :param cache_dir: Cache location. Defaults to the already-configured directory, or
                     ``$XDG_CACHE_HOME/refmod/jax`` (``~/.cache/refmod/jax``) when none
                     is configured.
   :type cache_dir: path-like or None, optional
   :param min_compile_time_secs: Only compiles slower than this are persisted. Default 0.0 (persist
                                 everything), so even the ~2 s CPU compiles are amortized.
   :type min_compile_time_secs: float, optional

   :returns: The cache directory in use.
   :rtype: pathlib.Path


.. py:function:: warmup(n_pixels = (1 << 20, ), max_steps = 40, cache = True, verbose = True)

   Compile the AMSA forward/gradient/inversion kernels ahead of time.

   Run this right after importing, before loading data onto the device:
   compiling while device memory is nearly full can slow XLA's autotuner
   down by orders of magnitude.

   :param n_pixels: Pixel counts (chunk shapes) to compile for. Use the chunk size your
                    inversions will run at; :func:`refmod.hapke.prepare_amsa_inversion`
                    quantizes adaptive chunk sizes to powers of two. Default: ``2**20``.
   :type n_pixels: iterable of int, optional
   :param max_steps: ``max_steps`` value the inversion will be called with (part of the
                     compiled kernel). Default 40, matching :func:`refmod.hapke.invert_amsa`.
   :type max_steps: int, optional
   :param cache: Also enable the persistent compilation cache (default True), so the
                 warmup outlives this process.
   :type cache: bool, optional
   :param verbose: Print per-kernel compile times.
   :type verbose: bool, optional


