from importlib.metadata import version as get_version

from .hapke import Hapke

__all__ = ["Hapke"]

__version__ = get_version(__package__) if __package__ else "0.0.0"
# __version__ = "0.1.0"
