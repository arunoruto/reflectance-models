import jax

jax.config.update("jax_enable_x64", True)

from importlib.metadata import version as get_version

from .hapke import Hapke
from .lambert import lambert
from .shkuratov import shkuratov

__all__ = ["Hapke", "lambert", "shkuratov"]

__version__ = get_version(__package__) if __package__ else "0.0.0"
# __version__ = "0.1.0"
