"""HD reconstruction from spatial + arrow representation."""

import numpy as np


class Reconstructor:
    """Reconstruct approximate HD vectors from AFE embedding.

    Captured dimensions are recovered from the spatial layout via
    linear regression. Residual dimensions are recovered from arrow
    decoding. No double-counting: each dimension uses exactly one
    source.
    """

    def __init__(self):
        self._spatial_weights = None  # (3, n_captured)
        self._spatial_bias = None     # (n_captured,)
        self._encoder = None
        self._gap_report = None
        self._n_features = None

    def fit(
        self,
        X_original: np.ndarray,
        spatial: np.ndarray,
        arrows: np.ndarray,
        gap_report: dict,
        encoder,
    ) -> "Reconstructor":
        """Learn the reconstruction mapping."""
        self._gap_report = gap_report
        self._encoder = encoder
        self._n_features = X_original.shape[1]

        captured_dims = gap_report["captured_dims"]

        if captured_dims:
            # Linear regression: spatial -> captured dims only
            X_captured = X_original[:, captured_dims]
            n = len(spatial)
            spatial_aug = np.column_stack([spatial, np.ones(n)])  # (n, 4)
            result = np.linalg.lstsq(spatial_aug, X_captured, rcond=None)
            wb = result[0]  # (4, n_captured)
            self._spatial_weights = wb[:3]  # (3, n_captured)
            self._spatial_bias = wb[3]      # (n_captured,)
        else:
            self._spatial_weights = None
            self._spatial_bias = None

        return self

    def reconstruct(
        self, spatial: np.ndarray, arrows: np.ndarray
    ) -> np.ndarray:
        """Produce approximate HD vectors.

        Returns ndarray (n, d_original).
        """
        n = len(spatial)
        d = self._n_features
        X_recon = np.zeros((n, d), dtype=np.float32)

        # Captured dims from spatial regression
        captured_dims = self._gap_report["captured_dims"]
        if captured_dims and self._spatial_weights is not None:
            X_captured = spatial @ self._spatial_weights + self._spatial_bias
            for j, dim_idx in enumerate(captured_dims):
                X_recon[:, dim_idx] = X_captured[:, j]

        # Residual dims from arrow decoding
        residual_dims = self._gap_report["residual_dims"]
        if residual_dims:
            residual_decoded = self._encoder.decode(arrows)
            for j, dim_idx in enumerate(residual_dims):
                if j < residual_decoded.shape[1]:
                    X_recon[:, dim_idx] = residual_decoded[:, j]

        return X_recon


__all__ = ["Reconstructor"]
