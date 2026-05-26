import jax


def pytest_configure() -> None:
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "Tests require JAX x64 mode. Run via `devenv shell -- uv run pytest` "
            "or set JAX_ENABLE_X64=True before starting pytest."
        )
