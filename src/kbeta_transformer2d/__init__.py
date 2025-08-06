"""2‑D Heat‑Diffusion Transformer demo for the Kourkoutas‑β paper."""

from .data import generate_datasets  # noqa: F401,E402
from .model import HeatDiffusionModel  # noqa: F401,E402

__all__ = ["HeatDiffusionModel", "generate_datasets"]
__version__ = "0.1.0a4"
