"""Arrow Field Embeddings: preserving high-dimensional information in 3D."""

from .core import ArrowFieldEmbedding
from .backends import (
    PaCMAPBackend, ManualBackend, DREAMSBackend,
    UMAPBackend, TSNEBackend, TriMAPBackend,
)
from .export import export_for_viewer
from .gap_analysis import SpatialInformationGapAnalyzer
from .attribution import arrow_dim_labels, get_arrow_attributions
from .evaluation import (
    arrow_consistency,
    arrow_knn_recall,
    arrow_spatial_information_gain,
    knn_recall,
    recon_knn_recall,
    reconstruction_error,
    spatial_information_gap,
)
from ._version import __version__

__all__ = [
    "ArrowFieldEmbedding",
    "PaCMAPBackend",
    "ManualBackend",
    "DREAMSBackend",
    "UMAPBackend",
    "TSNEBackend",
    "TriMAPBackend",
    "SpatialInformationGapAnalyzer",
    "arrow_consistency",
    "arrow_dim_labels",
    "arrow_knn_recall",
    "arrow_spatial_information_gain",
    "export_for_viewer",
    "get_arrow_attributions",
    "knn_recall",
    "recon_knn_recall",
    "reconstruction_error",
    "spatial_information_gap",
    "__version__",
]
