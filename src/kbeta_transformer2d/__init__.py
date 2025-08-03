__version__ = "0.0.0a0"
"""2‑D Heat‑Diffusion Transformer demo for the Kourkoutas‑β paper."""
from importlib.metadata import version as _v

from .data import generate_datasets
from .model import HeatDiffusionModel

__all__ = ["HeatDiffusionModel", "generate_datasets"]
