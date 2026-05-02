"""Arrow normalization utilities."""

import numpy as np


class ArrowNormalizer:
    """Per-arrow normalization across all points.

    Each arrow index is normalized independently but consistently across
    all points. Arrow k uses the same scale on every point.
    """

    def __init__(self):
        self._mins = None  # (n_arrows, 3)
        self._maxs = None  # (n_arrows, 3)

    def fit(self, arrows: np.ndarray) -> "ArrowNormalizer":
        """Compute normalization statistics.

        Parameters
        ----------
        arrows : ndarray (n, n_arrows, 3)
        """
        # Min/max per arrow index, across all points
        self._mins = arrows.min(axis=0)  # (n_arrows, 3)
        self._maxs = arrows.max(axis=0)  # (n_arrows, 3)
        return self

    def transform(self, arrows: np.ndarray) -> np.ndarray:
        """Normalize arrows to target ranges.

        Channel 0 (azimuth)  -> [-pi, pi]
        Channel 1 (elevation) -> [-pi/2, pi/2]
        Channel 2 (magnitude) -> [0, 1]
        """
        if self._mins is None:
            raise RuntimeError("ArrowNormalizer must be fit before transform.")

        ranges = self._maxs - self._mins
        # Avoid division by zero for constant channels
        ranges = np.where(ranges == 0, 1.0, ranges)

        # Normalize to [0, 1]
        normalized = (arrows - self._mins) / ranges

        # Scale to target ranges
        result = np.empty_like(normalized)
        result[:, :, 0] = normalized[:, :, 0] * (2 * np.pi) - np.pi  # [-pi, pi]
        result[:, :, 1] = normalized[:, :, 1] * np.pi - (np.pi / 2)  # [-pi/2, pi/2]
        result[:, :, 2] = normalized[:, :, 2]  # [0, 1]

        return result.astype(np.float32)

    def inverse_transform(self, arrows_normed: np.ndarray) -> np.ndarray:
        """Reverse normalization."""
        if self._mins is None:
            raise RuntimeError("ArrowNormalizer must be fit before inverse_transform.")

        ranges = self._maxs - self._mins
        ranges = np.where(ranges == 0, 1.0, ranges)

        # Reverse target range scaling to [0, 1]
        unit = np.empty_like(arrows_normed)
        unit[:, :, 0] = (arrows_normed[:, :, 0] + np.pi) / (2 * np.pi)
        unit[:, :, 1] = (arrows_normed[:, :, 1] + np.pi / 2) / np.pi
        unit[:, :, 2] = arrows_normed[:, :, 2]

        # Reverse [0, 1] to original scale
        return (unit * ranges + self._mins).astype(np.float32)

    def fit_transform(self, arrows: np.ndarray) -> np.ndarray:
        return self.fit(arrows).transform(arrows)


__all__ = ["ArrowNormalizer"]
