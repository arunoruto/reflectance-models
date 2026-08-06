from importlib.metadata import version as get_version

from . import api
from .hapke import Hapke
from .lambert import lambert
from .lunar_lambert import lunar_lambert
from .shkuratov import shkuratov

__all__ = [
    "Hapke",
    "api",
    "enable_compilation_cache",
    "lambert",
    "lunar_lambert",
    "shkuratov",
]


def __getattr__(name: str):
    # Lazy import so `python -m refmod.warmup` does not double-import the
    # warmup module (runpy warning).
    if name == "enable_compilation_cache":
        from .warmup import enable_compilation_cache

        return enable_compilation_cache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ = get_version(__package__) if __package__ else "0.0.0"
# __version__ = "0.1.0"
