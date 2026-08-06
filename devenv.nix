{
  config,
  pkgs,
  lib,
  inputs,
  ...
}:
let
  cuda-combined = pkgs.symlinkJoin {
    name = "cuda-combined-lib";
    paths = with pkgs.cudaPackages_12; [
      (lib.getLib cuda_cudart)
      (lib.getLib cuda_cupti)
      (lib.getLib libcublas)
      (lib.getLib libcufft)
      (lib.getLib libcusolver)
      (lib.getLib libcusparse)
      (lib.getLib cudnn)
      cuda_nvcc
    ];
  };
in
{
  overlays = [
    (final: prev: {
      # unstable = import inputs.nixpkgs-unstable {
      unstable = import inputs.nixpkgs {
        inherit (final.stdenv.hostPlatform) system;
        config = {
          allowUnfree = true;
          nvidia.acceptLicense = true;
        };
      };
    })
  ];

  env = {
    UV_PYTHON = toString config.languages.python.package.interpreter;
    # LD_LIBRARY_PATH = lib.makeLibraryPath [
    #   pkgs.stdenv.cc.cc.lib
    #   pkgs.zlib
    #   cuda-combined
    #   "/run/opengl-driver"
    # ];
    XLA_FLAGS = "--xla_gpu_cuda_data_dir=${cuda-combined}";

    # JAX_ENABLE_X64 = "True";
    # JAX_PLATFORMS = "cpu";
    # JAX_PLATFORMS = "cpu";
    JAX_ENABLE_X64 = "True";
    JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES = -1;
    JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS = 0;
    JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES = "all";
  };

  packages = [
    pkgs.unstable.beads
    pkgs.git
    cuda-combined
  ];

  scripts = {
    hello.exec = ''
      echo x64 enabled: $JAX_ENABLE_X64
    '';
    pytest.exec = ''uv run pytest "$@"'';
    ensure-beads.exec = ''
      if [ -d "$DEVENV_ROOT/.beads" ]; then
        # Check if daemon is responding, otherwise start it
        if ! bd status >/dev/null 2>&1; then
          echo "🔮 Starting Beads daemon..."
          bd daemon --start --log "$DEVENV_ROOT/.beads/daemon.log"
        fi
      fi
    '';
  };

  enterShell = ''
    export JAX_COMPILATION_CACHE_DIR="$DEVENV_STATE/jax-cache"
    mkdir -p "$JAX_COMPILATION_CACHE_DIR"
    hello
    git --version
    if [ ! -L "$DEVENV_ROOT/.venv" ]; then
        ln -s "$DEVENV_STATE/venv/" "$DEVENV_ROOT/.venv"
    fi
    # ensure-beads
  '';

  enterTest = ''
    echo "Running tests"
    git --version | grep --color=auto "${pkgs.git.version}"
  '';

  languages.python = {
    enable = true;
    # version = "3.12";

    uv = {
      enable = true;
      sync = {
        enable = true;
        groups = [
          "test"
          "docs"
          "profiling"
        ];
        extras = [ "jax-gpu" ];
      };
    };

    libraries = [
      pkgs.zlib
      "/run/opengl-driver"
      cuda-combined
    ];
  };
}
