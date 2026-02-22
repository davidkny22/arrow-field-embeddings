"""Information gap analysis: what does the spatial layout miss?"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class InformationGapAnalyzer:
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
            information_gap_score : float
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

        # Sort residual dims by how poorly they're captured (lowest correlation first)
        residual_sorted = sorted(residual, key=lambda dim: max_abs_corr[dim])

        residual_data = (
            X_high[:, residual_sorted]
            if residual_sorted
            else np.empty((len(X_high), 0), dtype=np.float32)
        )

        gap_score = 1.0 - np.mean(max_abs_corr)

        if self.verbose:
            print(
                f"Gap analysis: {len(captured)} captured, "
                f"{len(residual_sorted)} residual dims "
                f"(gap score: {gap_score:.3f})"
            )

        return {
            "correlation_matrix": corr_matrix,
            "max_abs_correlation": max_abs_corr,
            "captured_dims": captured,
            "residual_dims": residual_sorted,
            "residual_data": residual_data,
            "information_gap_score": gap_score,
        }

    def _compute_correlation_matrix(
        self, X_high: np.ndarray, X_3d: np.ndarray
    ) -> np.ndarray:
        """Pearson correlation between each HD dim and 3D coords.

        Returns ndarray of shape (n_features, 3).
        """
        n = len(X_high)

        # Center both
        X_h = X_high - X_high.mean(axis=0, keepdims=True)
        X_s = X_3d - X_3d.mean(axis=0, keepdims=True)

        # Standard deviations
        std_h = np.std(X_high, axis=0, ddof=0)
        std_s = np.std(X_3d, axis=0, ddof=0)

        # Avoid division by zero for constant dimensions
        std_h = np.where(std_h == 0, 1.0, std_h)
        std_s = np.where(std_s == 0, 1.0, std_s)

        # Correlation: (d, 3) = (d, n) @ (n, 3) / n, then normalize
        corr = (X_h.T @ X_s) / n
        corr = corr / (std_h[:, None] * std_s[None, :])

        return corr.astype(np.float32)

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
