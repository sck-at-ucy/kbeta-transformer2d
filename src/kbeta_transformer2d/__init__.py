"""kbeta-transformer2d: 2-D Heat-Diffusion Transformer demo

Companion workload for the Kourkoutas-β optimiser paper.
Includes model definition, dataset generators, and training helpers.
"""
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version(__name__)
except PackageNotFoundError:          # building an sdist
    __version__ = "1.0.0"

from .data import generate_datasets  # noqa: F401,E402
from .model import HeatDiffusionModel  # noqa: F401,E402

__all__ = ["HeatDiffusionModel", "generate_datasets"]

