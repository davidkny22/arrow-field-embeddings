"""Arrow encoding modes: how residual dimensions map to arrows."""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.linalg import eigh


class ArrowEncoder(ABC):
    """Abstract base for arrow encoding strategies."""

    @abstractmethod
    def fit(
        self, residual_data: np.ndarray, n_arrows: int, gap_report: dict
    ) -> "ArrowEncoder":
        """Fit the encoder to residual data."""
        ...

    @abstractmethod
    def encode(self, residual_data: np.ndarray) -> np.ndarray:
        """Map residual data to arrow vectors.

        Returns
        -------
        arrows : ndarray (n_samples, n_arrows, 3)
            Each arrow is (azimuth, elevation, magnitude).
        """
        ...

    @abstractmethod
    def decode(self, arrows: np.ndarray) -> np.ndarray:
        """Reconstruct residual data from arrows (for evaluation)."""
        ...


class DirectMappingEncoder(ArrowEncoder):
    """1-to-1 mapping: residual dimensions directly to arrow channels.

    Each arrow gets exactly 3 dimensions mapped to (azimuth, elevation,
    magnitude). Dimension ordering follows the priority ranking from
    gap analysis (worst-captured first).
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._n_arrows = None
        self._n_residual = None
        self._n_mapped = None

    def fit(
        self, residual_data: np.ndarray, n_arrows: int, gap_report: dict
    ) -> "DirectMappingEncoder":
        self._n_residual = residual_data.shape[1]
        self._n_arrows = n_arrows
        self._n_mapped = min(n_arrows * 3, self._n_residual)

        if self.verbose:
            print(
                f"DirectMapping: {self._n_residual} residual dims -> "
                f"{self._n_arrows} arrows ({self._n_mapped} dims mapped)"
            )
        return self

    def encode(self, residual_data: np.ndarray) -> np.ndarray:
        n = len(residual_data)
        arrows = np.zeros((n, self._n_arrows, 3), dtype=np.float32)

        for a in range(self._n_arrows):
            for c in range(3):
                dim_idx = a * 3 + c
                if dim_idx < self._n_residual:
                    arrows[:, a, c] = residual_data[:, dim_idx]
        return arrows

    def decode(self, arrows: np.ndarray) -> np.ndarray:
        n = arrows.shape[0]
        residual = np.zeros((n, self._n_residual), dtype=np.float32)

        for a in range(self._n_arrows):
            for c in range(3):
                dim_idx = a * 3 + c
                if dim_idx < self._n_residual:
                    residual[:, dim_idx] = arrows[:, a, c]
        return residual


class PCAResidualEncoder(ArrowEncoder):
    """PCA on residuals: top principal components -> arrows.

    Each arrow corresponds to one PC. The PC score for each point
    becomes the arrow's magnitude. The PC loading vector direction
    (projected to 2D via its first two elements) defines azimuth
    and elevation.

    This means arrows are ordered by variance explained.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._pca = None
        self._n_arrows = None
        self._n_residual = None
        self._loading_angles = None  # (n_arrows, 2) — azimuth, elevation per arrow

    def fit(
        self, residual_data: np.ndarray, n_arrows: int, gap_report: dict
    ) -> "PCAResidualEncoder":
        self._n_residual = residual_data.shape[1]
        self._n_arrows = n_arrows

        if self._n_residual == 0:
            self._pca = None
            self._loading_angles = np.empty((0, 2), dtype=np.float32)
            if self.verbose:
                print("PCAResidual: 0 residual dims, arrows will be zero")
            return self

        n_components = min(n_arrows, self._n_residual)

        self._pca = PCA(n_components=n_components)
        self._pca.fit(residual_data)

        # Compute loading angles: each PC defines a direction in residual space.
        # We project this direction to angular coordinates.
        self._loading_angles = np.zeros((n_components, 2), dtype=np.float32)
        for i in range(n_components):
            loading = self._pca.components_[i]
            # Use first two loading elements for azimuth, third for elevation
            if len(loading) >= 2:
                self._loading_angles[i, 0] = np.arctan2(loading[1], loading[0])
            if len(loading) >= 3:
                norm_xy = np.sqrt(loading[0] ** 2 + loading[1] ** 2)
                self._loading_angles[i, 1] = np.arctan2(loading[2], norm_xy)

        if self.verbose:
            explained = self._pca.explained_variance_ratio_
            total = np.sum(explained)
            print(
                f"PCAResidual: {self._n_residual} residual dims -> "
                f"{n_components} PCs ({total:.1%} variance explained)"
            )
        return self

    def encode(self, residual_data: np.ndarray) -> np.ndarray:
        n = len(residual_data)
        if self._pca is None:
            return np.zeros((n, self._n_arrows, 3), dtype=np.float32)

        n_components = self._pca.n_components_
        scores = self._pca.transform(residual_data)  # (n, n_components)

        arrows = np.zeros((n, self._n_arrows, 3), dtype=np.float32)
        for i in range(min(self._n_arrows, n_components)):
            arrows[:, i, 0] = self._loading_angles[i, 0]  # azimuth (same for all points)
            arrows[:, i, 1] = self._loading_angles[i, 1]  # elevation (same for all points)
            arrows[:, i, 2] = scores[:, i]                 # magnitude (varies per point)
        return arrows

    def decode(self, arrows: np.ndarray) -> np.ndarray:
        n = arrows.shape[0]
        if self._pca is None:
            return np.zeros((n, self._n_residual), dtype=np.float32)

        n_components = self._pca.n_components_
        # Extract scores from magnitude channel
        scores = np.zeros((n, n_components), dtype=np.float32)
        for i in range(min(self._n_arrows, n_components)):
            scores[:, i] = arrows[:, i, 2]

        # Inverse PCA
        return self._pca.inverse_transform(scores).astype(np.float32)


class AdaptiveGroupingEncoder(ArrowEncoder):
    """Eigenvalue gap detection + hierarchical clustering + validation.

    Groups correlated residual dimensions onto shared arrows.
    Independent dimensions get their own arrows.

    Pipeline:
    1. Eigenvalue gap detection: how many groups exist?
    2. Hierarchical clustering: which dims share an arrow?
    3. Correlation validation: did the grouping make sense?
    4. Per-group mapping to (azimuth, elevation, magnitude).
    """

    def __init__(
        self,
        eigenvalue_gap_threshold: float = 1.5,
        correlation_validation_threshold: float = 0.2,
        verbose: bool = False,
    ):
        self.eigenvalue_gap_threshold = eigenvalue_gap_threshold
        self.correlation_validation_threshold = correlation_validation_threshold
        self.verbose = verbose
        self._n_arrows = None
        self._n_residual = None
        self._groups = None  # list of lists: group_i -> [dim indices]
        self._group_pcas = None  # PCA per group for mapping
        self._group_angles = None  # (n_groups, 2) loading angles

    def fit(
        self, residual_data: np.ndarray, n_arrows: int, gap_report: dict
    ) -> "AdaptiveGroupingEncoder":
        self._n_residual = residual_data.shape[1]
        self._n_arrows = n_arrows

        if self._n_residual == 0:
            self._groups = []
            self._group_pcas = []
            self._group_angles = np.empty((0, 2), dtype=np.float32)
            return self

        # Layer 1: Eigenvalue gap detection
        n_groups = self._detect_n_groups(residual_data)
        n_groups = min(n_groups, n_arrows, self._n_residual)
        n_groups = max(n_groups, 1)

        # Layer 2: Hierarchical clustering
        groups = self._cluster_dimensions(residual_data, n_groups)

        # Layer 3: Correlation validation
        groups = self._validate_groups(residual_data, groups)

        # Trim to n_arrows (merge smallest groups if too many)
        while len(groups) > n_arrows:
            # Merge the two smallest groups
            sizes = [len(g) for g in groups]
            i1 = np.argmin(sizes)
            sizes[i1] = float("inf")
            i2 = np.argmin(sizes)
            merged = sorted(groups[i1] + groups[i2])
            groups = [g for idx, g in enumerate(groups) if idx not in (i1, i2)]
            groups.append(merged)

        self._groups = groups

        # Fit per-group PCA for mapping
        self._group_pcas = []
        self._group_angles = np.zeros((len(groups), 2), dtype=np.float32)

        for gi, group_dims in enumerate(groups):
            group_data = residual_data[:, group_dims]

            if len(group_dims) == 1:
                # Single dimension: no PCA needed
                self._group_pcas.append(None)
            else:
                n_comp = min(3, len(group_dims))
                pca = PCA(n_components=n_comp)
                pca.fit(group_data)
                self._group_pcas.append(pca)

                # Loading angles from first PC
                loading = pca.components_[0]
                if len(loading) >= 2:
                    self._group_angles[gi, 0] = np.arctan2(loading[1], loading[0])
                if len(loading) >= 3:
                    norm_xy = np.sqrt(loading[0] ** 2 + loading[1] ** 2)
                    self._group_angles[gi, 1] = np.arctan2(loading[2], norm_xy)

        if self.verbose:
            group_sizes = [len(g) for g in groups]
            print(
                f"AdaptiveGrouping: {self._n_residual} residual dims -> "
                f"{len(groups)} groups (sizes: {group_sizes})"
            )
        return self

    def encode(self, residual_data: np.ndarray) -> np.ndarray:
        n = len(residual_data)
        arrows = np.zeros((n, self._n_arrows, 3), dtype=np.float32)

        for gi, group_dims in enumerate(self._groups):
            if gi >= self._n_arrows:
                break

            group_data = residual_data[:, group_dims]

            if self._group_pcas[gi] is None:
                # Single dimension: direct mapping
                arrows[:, gi, 0] = 0.0  # azimuth = 0 (single dim has no direction)
                arrows[:, gi, 1] = 0.0  # elevation = 0
                arrows[:, gi, 2] = group_data[:, 0]  # magnitude = value
            else:
                pca = self._group_pcas[gi]
                scores = pca.transform(group_data)
                # Azimuth from loading direction (consistent across points)
                arrows[:, gi, 0] = self._group_angles[gi, 0]
                # Elevation: second PC score if available, else loading angle
                if scores.shape[1] >= 2:
                    arrows[:, gi, 1] = scores[:, 1]
                else:
                    arrows[:, gi, 1] = self._group_angles[gi, 1]
                # Magnitude from first PC score
                arrows[:, gi, 2] = scores[:, 0]
        return arrows

    def decode(self, arrows: np.ndarray) -> np.ndarray:
        n = arrows.shape[0]
        residual = np.zeros((n, self._n_residual), dtype=np.float32)

        for gi, group_dims in enumerate(self._groups):
            if gi >= self._n_arrows:
                break

            if self._group_pcas[gi] is None:
                # Single dim: magnitude is the value
                residual[:, group_dims[0]] = arrows[:, gi, 2]
            else:
                pca = self._group_pcas[gi]
                n_comp = pca.n_components_
                scores = np.zeros((n, n_comp), dtype=np.float32)
                scores[:, 0] = arrows[:, gi, 2]  # magnitude -> PC1 score
                if n_comp >= 2:
                    scores[:, 1] = arrows[:, gi, 1]  # elevation -> PC2 score
                reconstructed = pca.inverse_transform(scores)
                for j, dim_idx in enumerate(group_dims):
                    residual[:, dim_idx] = reconstructed[:, j]
        return residual

    def _detect_n_groups(self, residual_data: np.ndarray) -> int:
        """Layer 1: Detect number of natural groups via eigenvalue gaps."""
        if residual_data.shape[1] <= 1:
            return 1

        cov = np.cov(residual_data, rowvar=False)
        if cov.ndim == 0:
            return 1
        eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]

        # Remove near-zero eigenvalues
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        if len(eigenvalues) <= 1:
            return 1

        # Compute ratios between consecutive eigenvalues
        ratios = eigenvalues[:-1] / eigenvalues[1:]

        # Count significant gaps
        gaps = np.where(ratios > self.eigenvalue_gap_threshold)[0]
        n_groups = len(gaps) + 1

        if self.verbose:
            print(f"  Eigenvalue gaps: {len(gaps)} detected "
                  f"(threshold={self.eigenvalue_gap_threshold})")

        return n_groups

    def _cluster_dimensions(
        self, residual_data: np.ndarray, n_groups: int
    ) -> List[List[int]]:
        """Layer 2: Hierarchical clustering on dimension correlations."""
        n_dims = residual_data.shape[1]

        if n_dims <= n_groups:
            return [[i] for i in range(n_dims)]

        # Distance = 1 - |correlation|
        corr = np.corrcoef(residual_data, rowvar=False)
        # Handle NaN correlations (constant dims)
        corr = np.nan_to_num(corr, nan=0.0)
        distance = 1.0 - np.abs(corr)

        # Convert to condensed distance matrix
        condensed = []
        for i in range(n_dims):
            for j in range(i + 1, n_dims):
                condensed.append(distance[i, j])
        condensed = np.array(condensed)

        # Ward linkage
        Z = linkage(condensed, method="ward")
        labels = fcluster(Z, t=n_groups, criterion="maxclust")

        # Build groups
        groups = []
        for g in range(1, n_groups + 1):
            group_dims = list(np.where(labels == g)[0])
            if group_dims:
                groups.append(group_dims)

        return groups

    def _validate_groups(
        self, residual_data: np.ndarray, groups: List[List[int]]
    ) -> List[List[int]]:
        """Layer 3: Validate that grouped dims are actually correlated."""
        validated = []
        for group_dims in groups:
            if len(group_dims) <= 1:
                validated.append(group_dims)
                continue

            # Check within-group mean |correlation|
            group_data = residual_data[:, group_dims]
            corr = np.corrcoef(group_data, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0)
            # Mean of off-diagonal |correlations|
            mask = ~np.eye(len(group_dims), dtype=bool)
            mean_corr = np.mean(np.abs(corr[mask]))

            if mean_corr >= self.correlation_validation_threshold:
                validated.append(group_dims)
            else:
                # Split: each dim becomes its own group
                if self.verbose:
                    print(f"  Splitting weak group (mean |corr|={mean_corr:.3f}): "
                          f"dims {group_dims}")
                for d in group_dims:
                    validated.append([d])

        return validated


def get_encoder(mode: str, **kwargs) -> ArrowEncoder:
    """Factory: mode string -> ArrowEncoder instance."""
    encoders = {
        "adaptive": AdaptiveGroupingEncoder,
        "pca": PCAResidualEncoder,
        "direct": DirectMappingEncoder,
    }
    if mode not in encoders:
        raise ValueError(
            f"Unknown encoding mode '{mode}'. Choose from {list(encoders.keys())}"
        )
    return encoders[mode](**kwargs)
