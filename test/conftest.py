import jax


def pytest_configure() -> None:
    jax.config.update("jax_enable_x64", True)
