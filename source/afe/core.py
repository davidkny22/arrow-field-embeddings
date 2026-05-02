"""Core ArrowFieldEmbedding class."""

import logging
import numpy as np
from sklearn.base import BaseEstimator
from typing import Optional, Union, Literal, Dict

from .backends import SpatialBackend, get_backend
from .gap_analysis import SpatialInformationGapAnalyzer
from .encoding import get_encoder
from .normalization import ArrowNormalizer
from .reconstruction import Reconstructor
from .attribution import get_arrow_attributions
from .reproducibility import RESULT_SCHEMA_VERSION

_logger = logging.getLogger(__name__)


class ArrowFieldEmbedding(BaseEstimator):
    """Arrow Field Embedding: dimensionality reduction with directional annotations.

    Augments standard 3D spatial embedding with k arrows per point, where each
    arrow encodes 3 additional dimensions (azimuth, elevation, magnitude) to
    preserve high-dimensional information beyond spatial position.

    Parameters
    ----------
    n_arrows : int, default=2
        Number of arrows per embedded point. Each arrow encodes 3 additional
        effective dimensions.

    encoding_mode : {'adaptive', 'pca', 'direct'}, default='direct'
        How residual dimensions are mapped to arrows.

    backend : str, ndarray, or SpatialBackend, default='pacmap'
        Spatial layout method. Built-in: 'pacmap', 'dreams'.
        Can also pass a pre-computed (n, 3) array or SpatialBackend instance.

    backend_kwargs : dict, optional
        Keyword arguments forwarded to the spatial backend.

    correlation_threshold : float, default=0.3
        Minimum |correlation| for a dimension to be considered captured
        by the spatial layout.

    eigenvalue_gap_threshold : float, default=1.5
        Minimum eigenvalue ratio for gap detection (adaptive mode).

    normalize_arrows : bool, default=True
        Whether to normalize arrow components per-arrow across all points.

    random_state : int or None, default=None
        Random seed for reproducibility.

    verbose : bool, default=False
        Print progress information.
    """

    def __init__(
        self,
        n_arrows: int = 2,
        encoding_mode: Literal["adaptive", "pca", "direct"] = "direct",
        backend: Union[str, np.ndarray, SpatialBackend] = "pacmap",
        backend_kwargs: Optional[dict] = None,
        correlation_threshold: float = 0.3,
        eigenvalue_gap_threshold: float = 1.5,
        normalize_arrows: bool = True,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.n_arrows = n_arrows
        self.encoding_mode = encoding_mode
        self.backend = backend
        self.backend_kwargs = backend_kwargs
        self.correlation_threshold = correlation_threshold
        self.eigenvalue_gap_threshold = eigenvalue_gap_threshold
        self.normalize_arrows = normalize_arrows
        self.random_state = random_state
        self.verbose = verbose

        # Fitted state
        self._backend_instance = None
        self._gap_analyzer = None
        self._encoder = None
        self._normalizer = None
        self._reconstructor = None
        self._spatial = None
        self._arrows = None
        self._arrows_raw = None
        self._gap_report = None
        self._n_features_original = None

    def fit(self, X: np.ndarray, y=None) -> "ArrowFieldEmbedding":
        """Compute the spatial embedding and arrow field from X.

        Parameters
        ----------
        X : ndarray (n_samples, n_features)
        y : ignored
        """
        if isinstance(self.n_arrows, bool) or not isinstance(self.n_arrows, (int, np.integer)):
            raise ValueError("n_arrows must be an integer >= 1 for AFE.")
        if int(self.n_arrows) < 1:
            raise ValueError(
                "n_arrows must be >= 1 for AFE. Use a standalone spatial "
                "backend result for the zero-arrow/normal-DR baseline."
            )
        self.n_arrows_ = int(self.n_arrows)

        X = np.asarray(X, dtype=np.float32)
        self._n_features_original = X.shape[1]

        # Step 1: Spatial layout
        if self.verbose:
            _logger.info("Step 1: Computing spatial layout...")
        self._backend_instance = get_backend(
            self.backend,
            backend_kwargs=self.backend_kwargs,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self._spatial = self._backend_instance.fit_transform(X)

        # Step 2: Spatial information gap analysis
        if self.verbose:
            _logger.info("Step 2: Analyzing spatial information gap...")
        self._gap_analyzer = SpatialInformationGapAnalyzer(
            correlation_threshold=self.correlation_threshold,
            verbose=self.verbose,
        )
        self._gap_report = self._gap_analyzer.analyze(
            X, self._spatial, n_arrows=self.n_arrows_
        )

        # Step 3: Arrow encoding
        if self.verbose:
            _logger.info("Step 3: Encoding arrows (%s mode)...", self.encoding_mode)

        encoder_kwargs = {"verbose": self.verbose}
        if self.encoding_mode == "adaptive":
            encoder_kwargs["eigenvalue_gap_threshold"] = self.eigenvalue_gap_threshold

        self._encoder = get_encoder(self.encoding_mode, **encoder_kwargs)
        self._encoder.fit(
            self._gap_report["residual_data"],
            self.n_arrows_,
            self._gap_report,
        )
        self._arrows_raw = self._encoder.encode(self._gap_report["residual_data"])

        # Step 4: Normalization
        if self.normalize_arrows:
            if self.verbose:
                _logger.info("Step 4: Normalizing arrows...")
            self._normalizer = ArrowNormalizer()
            self._arrows = self._normalizer.fit_transform(self._arrows_raw)
        else:
            self._arrows = self._arrows_raw

        # Fit reconstructor
        self._reconstructor = Reconstructor()
        self._reconstructor.fit(
            X, self._spatial, self._arrows_raw, self._gap_report, self._encoder
        )

        return self

    def fit_transform(self, X: np.ndarray, y=None) -> Dict:
        """Fit and return the embedding result.

        Returns
        -------
        result : dict with keys:
            spatial  : ndarray (n, 3)
            arrows   : ndarray (n, n_arrows, 3) — (azimuth, elevation, magnitude)
            metadata : dict — gap analysis results, encoding info
        """
        self.fit(X, y)
        return self._build_result()

    def get_spatial(self) -> np.ndarray:
        """Return the (n, 3) spatial positions."""
        self._check_fitted()
        return self._spatial.copy()

    def get_arrows(self) -> np.ndarray:
        """Return the (n, n_arrows, 3) arrow array."""
        self._check_fitted()
        return self._arrows.copy()

    def get_gap_report(self) -> Dict:
        """Return the spatial information gap analysis report."""
        self._check_fitted()
        import copy
        return copy.deepcopy(self._gap_report)

    def get_arrow_attributions(self, feature_names=None, top_n: int = 8) -> list:
        """Return how each arrow maps back to residual dimensions/components."""
        self._check_fitted()
        return get_arrow_attributions(self, feature_names=feature_names, top_n=top_n)

    def reconstruct(self) -> np.ndarray:
        """Reconstruct approximate HD vectors from the embedding."""
        self._check_fitted()
        return self._reconstructor.reconstruct(self._spatial, self._arrows_raw)

    def _build_result(self) -> Dict:
        residual_dims = list(self._gap_report["residual_dims"])
        captured_dims = list(self._gap_report["captured_dims"])
        arrow_capacity = {
            "spatial_dims": 3,
            "n_arrows": int(self.n_arrows_),
            "dims_per_arrow": 3,
            "arrow_channel_capacity": int(self.n_arrows_ * 3),
            "total_representation_dims": int(3 + self.n_arrows_ * 3),
            "n_residual_dims": len(residual_dims),
            "n_mapped_residual_dims": int(
                min(len(residual_dims), self.n_arrows_ * 3)
                if self.encoding_mode == "direct"
                else min(len(residual_dims), self.n_arrows_)
            ),
        }
        return {
            "spatial": self._spatial.copy(),
            "arrows": self._arrows.copy(),
            "metadata": {
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "gap_report": self._gap_report,
                "spatial_information_gap": self._gap_report["spatial_information_gap"],
                "encoding_mode": self.encoding_mode,
                "n_arrows": self.n_arrows_,
                "arrow_capacity": arrow_capacity,
                "arrow_attributions": get_arrow_attributions(self),
                "residual_selector": {
                    "method": "spatial_axis_max_abs_pearson_correlation",
                    "correlation_threshold": self.correlation_threshold,
                    "minimum_residual_dims": min(self.n_arrows_ * 3, self._n_features_original),
                    "residual_order": "variance_descending_after_capture_split",
                },
                "feature_scaling": {
                    "input": "as_provided_float32",
                    "arrow_normalization": (
                        "per_arrow_spherical_minmax" if self.normalize_arrows else "none"
                    ),
                },
                "backend": self._backend_metadata(),
                "n_features_original": self._n_features_original,
                "captured_dims": captured_dims,
                "residual_dims": residual_dims,
                "n_captured_dims": len(captured_dims),
                "n_residual_dims": len(residual_dims),
                "dims_per_arrow": 3,
                "total_encoded_dims": arrow_capacity["total_representation_dims"],
            },
        }

    def _backend_metadata(self) -> Dict:
        backend_value = self.backend
        if isinstance(backend_value, np.ndarray):
            backend_name = "manual"
            backend_params = {"source": "ndarray"}
        elif isinstance(backend_value, SpatialBackend):
            backend_name = backend_value.__class__.__name__
            backend_params = {}
        else:
            backend_name = str(backend_value)
            backend_params = dict(self.backend_kwargs or {})
        return {
            "name": backend_name,
            "params": backend_params,
            "random_state": self.random_state,
            "spatial_coordinates_fixed": isinstance(backend_value, np.ndarray),
        }

    def _check_fitted(self):
        if self._spatial is None:
            raise RuntimeError(
                "ArrowFieldEmbedding has not been fitted. Call fit() first."
            )


__all__ = ["ArrowFieldEmbedding"]
