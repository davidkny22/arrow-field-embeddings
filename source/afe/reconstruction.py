"""HD reconstruction from spatial + arrow representation."""

import numpy as np


class Reconstructor:
    """Reconstruct approximate HD vectors from AFE embedding.

    Combines spatial inverse mapping (linear regression from 3D -> HD)
    with arrow decoding to produce approximate original-space vectors.
    """

    def __init__(self):
        self._spatial_weights = None  # (3, d) linear mapping
        self._spatial_bias = None     # (d,)
        self._encoder = None
        self._gap_report = None

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

        # Linear regression: spatial -> HD (captures the "captured" dimensions)
        # X_original ≈ spatial @ W + b
        # Solve via least squares
        n = len(spatial)
        spatial_aug = np.column_stack([spatial, np.ones(n)])  # (n, 4)
        result = np.linalg.lstsq(spatial_aug, X_original, rcond=None)
        wb = result[0]  # (4, d)
        self._spatial_weights = wb[:3]  # (3, d)
        self._spatial_bias = wb[3]      # (d,)

        return self

    def reconstruct(
        self, spatial: np.ndarray, arrows: np.ndarray
    ) -> np.ndarray:
        """Produce approximate HD vectors.

        Returns ndarray (n, d_original).
        """
        if self._spatial_weights is None:
            raise RuntimeError("Reconstructor must be fit first.")

        # Spatial contribution
        X_spatial = spatial @ self._spatial_weights + self._spatial_bias

        # Arrow contribution (decode residuals)
        residual_decoded = self._encoder.decode(arrows)

        # Place decoded residuals into the correct dimension slots
        residual_dims = self._gap_report["residual_dims"]
        n = len(spatial)
        d = X_spatial.shape[1]
        X_arrow = np.zeros((n, d), dtype=np.float32)
        for i, dim_idx in enumerate(residual_dims):
            if i < residual_decoded.shape[1]:
                X_arrow[:, dim_idx] = residual_decoded[:, i]

        return (X_spatial + X_arrow).astype(np.float32)
