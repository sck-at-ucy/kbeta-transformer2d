__version__ = "0.0.0a0"
"""2‑D Heat‑Diffusion Transformer demo for the Kourkoutas‑β paper."""
from importlib.metadata import version as _v

from .model import HeatDiffusionModel
from .data import generate_datasets

__all__ = ["HeatDiffusionModel", "generate_datasets"]
