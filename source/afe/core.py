"""Core ArrowFieldEmbedding class."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Union, Literal, Dict

from .backends import SpatialBackend, get_backend
from .gap_analysis import InformationGapAnalyzer
from .encoding import get_encoder
from .normalization import ArrowNormalizer
from .reconstruction import Reconstructor


class ArrowFieldEmbedding(BaseEstimator, TransformerMixin):
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
        self._X_original = None

    def fit(self, X: np.ndarray, y=None) -> "ArrowFieldEmbedding":
        """Compute the spatial embedding and arrow field from X.

        Parameters
        ----------
        X : ndarray (n_samples, n_features)
        y : ignored
        """
        X = np.asarray(X, dtype=np.float32)
        self._X_original = X

        # Step 1: Spatial layout
        if self.verbose:
            print("Step 1: Computing spatial layout...")
        self._backend_instance = get_backend(
            self.backend,
            backend_kwargs=self.backend_kwargs,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self._spatial = self._backend_instance.fit_transform(X)

        # Step 2: Information gap analysis
        if self.verbose:
            print("Step 2: Analyzing information gap...")
        self._gap_analyzer = InformationGapAnalyzer(
            correlation_threshold=self.correlation_threshold,
            verbose=self.verbose,
        )
        self._gap_report = self._gap_analyzer.analyze(X, self._spatial)

        # Step 3: Arrow encoding
        if self.verbose:
            print(f"Step 3: Encoding arrows ({self.encoding_mode} mode)...")

        encoder_kwargs = {"verbose": self.verbose}
        if self.encoding_mode == "adaptive":
            encoder_kwargs["eigenvalue_gap_threshold"] = self.eigenvalue_gap_threshold

        self._encoder = get_encoder(self.encoding_mode, **encoder_kwargs)
        self._encoder.fit(
            self._gap_report["residual_data"],
            self.n_arrows,
            self._gap_report,
        )
        self._arrows_raw = self._encoder.encode(self._gap_report["residual_data"])

        # Step 4: Normalization
        if self.normalize_arrows:
            if self.verbose:
                print("Step 4: Normalizing arrows...")
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
        """Return the information gap analysis report."""
        self._check_fitted()
        return self._gap_report.copy()

    def reconstruct(self) -> np.ndarray:
        """Reconstruct approximate HD vectors from the embedding."""
        self._check_fitted()
        return self._reconstructor.reconstruct(self._spatial, self._arrows_raw)

    def _build_result(self) -> Dict:
        return {
            "spatial": self._spatial.copy(),
            "arrows": self._arrows.copy(),
            "metadata": {
                "gap_report": self._gap_report,
                "encoding_mode": self.encoding_mode,
                "n_arrows": self.n_arrows,
                "n_features_original": self._X_original.shape[1],
                "n_residual_dims": len(self._gap_report["residual_dims"]),
                "dims_per_arrow": 3,
                "total_encoded_dims": 3 + self.n_arrows * 3,
            },
        }

    def _check_fitted(self):
        if self._spatial is None:
            raise RuntimeError(
                "ArrowFieldEmbedding has not been fitted. Call fit() first."
            )
