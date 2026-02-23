"""Arrow Field Embeddings: preserving high-dimensional information in 3D."""

from .core import ArrowFieldEmbedding
from .backends import PaCMAPBackend, ManualBackend, DREAMSBackend
from .export import export_for_viewer

__version__ = "0.1.0"

__all__ = [
    "ArrowFieldEmbedding",
    "PaCMAPBackend",
    "ManualBackend",
    "DREAMSBackend",
    "export_for_viewer",
]
