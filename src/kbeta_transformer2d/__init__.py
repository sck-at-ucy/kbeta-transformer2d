"""2‑D Heat‑Diffusion Transformer demo for the Kourkoutas‑β paper."""
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version(__name__)
except PackageNotFoundError:          # building an sdist
    __version__ = "0.0.0.dev0"

from .data import generate_datasets  # noqa: F401,E402
from .model import HeatDiffusionModel  # noqa: F401,E402

__all__ = ["HeatDiffusionModel", "generate_datasets"]

