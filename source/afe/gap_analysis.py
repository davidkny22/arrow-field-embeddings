"""Spatial information gap analysis: what does the spatial layout miss?"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional

from .evaluation import correlation_matrix_to_spatial

_logger = logging.getLogger(__name__)


class SpatialInformationGapAnalyzer:
    """Measure what the 3D spatial layout fails to capture.

    Computes Pearson correlation between each original dimension and the
    3 spatial coordinates. All dimensions are ranked by how well the
    spatial layout captures them. The least-captured dimensions become
    residuals for arrow encoding.

    The correlation threshold controls the captured/residual split for
    reporting, but min_residual_dims ensures arrows always have
    something to encode — even when correlations are universally high
    (e.g. well-clustered data where every dimension correlates with
    cluster membership).
    """

    def __init__(
        self,
        correlation_threshold: float = 0.3,
        min_residual_dims: Optional[int] = None,
        verbose: bool = False,
    ):
        self.correlation_threshold = correlation_threshold
        self.min_residual_dims = min_residual_dims
        self.verbose = verbose

    def analyze(
        self,
        X_high: np.ndarray,
        X_3d: np.ndarray,
        n_arrows: int = 0,
    ) -> Dict:
        """Run the gap analysis.

        Parameters
        ----------
        X_high : ndarray (n, d)
            Original high-dimensional data.
        X_3d : ndarray (n, 3)
            3D spatial embedding.
        n_arrows : int
            Number of arrows requested. Used to compute min_residual_dims
            if not explicitly set (ensures at least n_arrows * 3 residuals
            when possible).

        Returns
        -------
        report : dict with keys:
            correlation_matrix : ndarray (d, 3)
            max_abs_correlation : ndarray (d,)
            captured_dims : list[int]
            residual_dims : list[int]  (sorted: worst-captured first)
            residual_data : ndarray (n, n_residual)
            spatial_information_gap : float
        """
        d = X_high.shape[1]
        corr_matrix = self._compute_correlation_matrix(X_high, X_3d)
        max_abs_corr = np.max(np.abs(corr_matrix), axis=1)

        # Determine minimum residual dims
        min_res = self.min_residual_dims
        if min_res is None and n_arrows > 0:
            # Ensure enough residual dims to fill the arrows
            min_res = min(n_arrows * 3, d)

        captured, residual = self._classify_dimensions(max_abs_corr, min_res, d)

        # Filter out zero-variance (constant) dims — they carry no information
        dim_var = np.var(X_high, axis=0)
        residual = [d for d in residual if dim_var[d] > 1e-10]

        # Sort residual dims by variance descending (most informative first)
        # so encoders that only use the first N dims get the best ones
        residual_sorted = sorted(residual, key=lambda dim: dim_var[dim], reverse=True)

        residual_data = (
            X_high[:, residual_sorted]
            if residual_sorted
            else np.empty((len(X_high), 0), dtype=np.float32)
        )

        gap = float(1.0 - np.mean(max_abs_corr))

        if self.verbose:
            _logger.info(
                "Spatial information gap analysis: %d captured, %d residual dims (gap: %.3f)",
                len(captured),
                len(residual_sorted),
                gap,
            )

        return {
            "correlation_matrix": corr_matrix,
            "max_abs_correlation": max_abs_corr,
            "captured_dims": captured,
            "residual_dims": residual_sorted,
            "residual_data": residual_data,
            "spatial_information_gap": gap,
            "gap_definition": "1 - mean_j max_l |corr(X_j, Y_l)|",
            "gap_readout": "linear_feature_visibility",
        }

    def _compute_correlation_matrix(
        self, X_high: np.ndarray, X_3d: np.ndarray
    ) -> np.ndarray:
        """Pearson correlation between each HD dim and 3D coords.

        Returns ndarray of shape (n_features, 3).
        """
        return correlation_matrix_to_spatial(X_high, X_3d)

    def _classify_dimensions(
        self,
        max_abs_corr: np.ndarray,
        min_residual: Optional[int],
        total_dims: int,
    ) -> Tuple[List[int], List[int]]:
        """Split dims into captured vs residual.

        Uses the correlation threshold first, then promotes the
        least-correlated captured dims to residual if min_residual
        is not met.
        """
        captured = []
        residual = []
        for i, corr in enumerate(max_abs_corr):
            if corr >= self.correlation_threshold:
                captured.append(i)
            else:
                residual.append(i)

        # Ensure minimum residual count by promoting least-correlated
        # captured dims
        if min_residual is not None and len(residual) < min_residual:
            needed = min_residual - len(residual)
            # Sort captured by correlation ascending (least correlated first)
            captured_sorted = sorted(captured, key=lambda d: max_abs_corr[d])
            promote = captured_sorted[:needed]
            residual.extend(promote)
            captured = [d for d in captured if d not in set(promote)]

        return captured, residual


__all__ = ["SpatialInformationGapAnalyzer"]
