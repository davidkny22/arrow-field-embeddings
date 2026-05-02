"""Stable evaluation metrics for Arrow Field Embeddings.

This module is the public home for paper-facing metrics. Benchmark scripts,
examples, tests, and exports should import from here directly.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


def _effective_k(n_samples: int, k: int) -> int:
    if n_samples <= 1:
        return 0
    return max(1, min(int(k), n_samples - 1))


def _safe_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    corr = np.corrcoef(a, b)[0, 1]
    return float(corr) if np.isfinite(corr) else np.nan


def _nanmean_or_zero(values) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return 0.0
    return float(np.mean(finite))


def correlation_matrix_to_spatial(X_high: np.ndarray, X_spatial: np.ndarray) -> np.ndarray:
    """Pearson correlations between each HD dimension and spatial coordinate."""
    X_high = np.asarray(X_high, dtype=np.float64)
    X_spatial = np.asarray(X_spatial, dtype=np.float64)
    if X_high.ndim != 2 or X_spatial.ndim != 2:
        raise ValueError("X_high and X_spatial must both be 2D arrays.")
    if len(X_high) != len(X_spatial):
        raise ValueError("X_high and X_spatial must have the same number of rows.")

    n = len(X_high)
    X_h = X_high - X_high.mean(axis=0, keepdims=True)
    X_s = X_spatial - X_spatial.mean(axis=0, keepdims=True)
    std_h = np.std(X_high, axis=0, ddof=0)
    std_s = np.std(X_spatial, axis=0, ddof=0)
    std_h = np.where(std_h == 0, 1.0, std_h)
    std_s = np.where(std_s == 0, 1.0, std_s)

    corr = (X_h.T @ X_s) / n
    corr = corr / (std_h[:, None] * std_s[None, :])
    return np.nan_to_num(corr, nan=0.0).astype(np.float32)


def spatial_information_gap(X_high: np.ndarray, X_spatial: np.ndarray) -> float:
    """Spatial information gap under a linear feature-visibility readout.

    Defined as ``1 - mean_j max_l |corr(X_j, Y_l)|``. This measures how much
    original feature variation is not linearly visible in any 3D spatial axis.
    It is not a Shannon mutual-information estimate.
    """
    corr = correlation_matrix_to_spatial(X_high, X_spatial)
    max_abs_corr = np.max(np.abs(corr), axis=1)
    return float(1.0 - np.mean(max_abs_corr))


def knn_recall(X_high: np.ndarray, X_low: np.ndarray, k: int = 10) -> float:
    """Fraction of original k-nearest neighbors preserved in a representation."""
    k = _effective_k(len(X_high), k)
    if k == 0:
        return 1.0

    nn_high = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X_high)
    nn_low = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X_low)

    _, idx_high = nn_high.kneighbors(X_high)
    _, idx_low = nn_low.kneighbors(X_low)
    idx_high = idx_high[:, 1:]
    idx_low = idx_low[:, 1:]

    recalls = [
        len(set(idx_high[i]) & set(idx_low[i])) / k
        for i in range(len(X_high))
    ]
    return float(np.mean(recalls))


def recon_knn_recall(
    X_high: np.ndarray, X_reconstructed: np.ndarray, k: int = 10
) -> float:
    """kNN recall on reconstructed HD vectors versus original HD vectors."""
    return knn_recall(X_high, X_reconstructed, k=k)


def spearman_distance_correlation(
    X_high: np.ndarray, X_low: np.ndarray, n_sample: int = 5000, random_state: int = 42
) -> float:
    """Spearman rank correlation between pairwise distances."""
    n = len(X_high)
    if n > n_sample:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, n_sample, replace=False)
        X_high = X_high[idx]
        X_low = X_low[idx]

    if len(X_high) < 3:
        return 0.0

    corr, _ = spearmanr(pdist(X_high), pdist(X_low))
    return float(corr) if np.isfinite(corr) else 0.0


def random_triplet_accuracy(
    X_high: np.ndarray,
    X_low: np.ndarray,
    n_triplets: int = 50000,
    random_state: int = 42,
) -> float:
    """Fraction of random triplets with preserved distance ordering."""
    n = len(X_high)
    if n < 3:
        return 1.0
    rng = np.random.RandomState(random_state)

    idx = np.empty((n_triplets, 3), dtype=int)
    for col in range(3):
        idx[:, col] = rng.randint(0, n, size=n_triplets)

    bad = (
        (idx[:, 0] == idx[:, 1])
        | (idx[:, 0] == idx[:, 2])
        | (idx[:, 1] == idx[:, 2])
    )
    while bad.any():
        n_bad = int(bad.sum())
        for col in range(3):
            idx[bad, col] = rng.randint(0, n, size=n_bad)
        bad = (
            (idx[:, 0] == idx[:, 1])
            | (idx[:, 0] == idx[:, 2])
            | (idx[:, 1] == idx[:, 2])
        )

    i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
    d_high_ij = np.linalg.norm(X_high[i] - X_high[j], axis=1)
    d_high_ik = np.linalg.norm(X_high[i] - X_high[k], axis=1)
    d_low_ij = np.linalg.norm(X_low[i] - X_low[j], axis=1)
    d_low_ik = np.linalg.norm(X_low[i] - X_low[k], axis=1)

    non_tie = d_high_ij != d_high_ik
    if not np.any(non_tie):
        return 1.0
    preserved = (d_high_ij < d_high_ik) == (d_low_ij < d_low_ik)
    return float(np.mean(preserved[non_tie]))


def centroid_triplet_accuracy(
    X_high: np.ndarray, X_low: np.ndarray, labels: Optional[np.ndarray]
) -> Optional[float]:
    """Fraction of label-centroid triplets with preserved distance ordering."""
    if labels is None:
        return None

    from scipy.spatial.distance import cdist

    classes = np.unique(labels)
    if len(classes) < 3:
        return None

    centroids_high = np.array([X_high[labels == c].mean(axis=0) for c in classes])
    centroids_low = np.array([X_low[labels == c].mean(axis=0) for c in classes])
    d_high = cdist(centroids_high, centroids_high, metric="euclidean")
    d_low = cdist(centroids_low, centroids_low, metric="euclidean")

    preserved = 0
    total = 0
    for i in range(len(classes)):
        for j in range(len(classes)):
            for kk in range(j + 1, len(classes)):
                if j == i or kk == i or d_high[i, j] == d_high[i, kk]:
                    continue
                same_order = (d_high[i, j] < d_high[i, kk]) == (
                    d_low[i, j] < d_low[i, kk]
                )
                preserved += int(same_order)
                total += 1
    return float(preserved / total) if total > 0 else None


def trustworthiness(X_high: np.ndarray, X_low: np.ndarray, k: int = 10) -> float:
    """Trustworthiness: false-neighbor penalty for the low-dimensional embedding."""
    from sklearn.manifold import trustworthiness as sklearn_trust

    n = len(X_high)
    if n < 3:
        return 1.0
    k_eff = min(_effective_k(n, k), max(1, (n - 1) // 2))
    return float(sklearn_trust(X_high, X_low, n_neighbors=k_eff))


def continuity(X_high: np.ndarray, X_low: np.ndarray, k: int = 10) -> float:
    """Continuity: missed-neighbor penalty, dual to trustworthiness."""
    from sklearn.manifold import trustworthiness as sklearn_trust

    n = len(X_high)
    if n < 3:
        return 1.0
    k_eff = min(_effective_k(n, k), max(1, (n - 1) // 2))
    return float(sklearn_trust(X_low, X_high, n_neighbors=k_eff))


def silhouette(X_low: np.ndarray, labels: Optional[np.ndarray]) -> Optional[float]:
    """Silhouette score in the embedding space, or None when not defined."""
    if labels is None:
        return None

    n_classes = len(np.unique(labels))
    if n_classes < 2 or n_classes >= len(labels):
        return None

    from sklearn.metrics import silhouette_score

    return float(silhouette_score(X_low, labels))


def normalized_stress(
    X_high: np.ndarray, X_low: np.ndarray, n_sample: int = 5000, random_state: int = 42
) -> float:
    """Scale-invariant pairwise distance distortion with Procrustes scaling."""
    n = len(X_high)
    if n > n_sample:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, n_sample, replace=False)
        X_high = X_high[idx]
        X_low = X_low[idx]

    if len(X_high) < 3:
        return 0.0

    d_high = pdist(X_high)
    d_low = pdist(X_low)
    denom_scale = np.sum(d_low ** 2)
    if denom_scale > 0:
        d_low = d_low * (np.sum(d_high * d_low) / denom_scale)

    denominator = np.sum(d_high ** 2)
    if denominator == 0:
        return 0.0
    return float(np.sum((d_high - d_low) ** 2) / denominator)


def reconstruction_error(
    X_high: np.ndarray, X_reconstructed: np.ndarray, metric: str = "mse"
) -> float:
    """Reconstruction error from AFE representation back to HD space."""
    if metric == "mse":
        return float(np.mean((X_high - X_reconstructed) ** 2))
    if metric == "cosine":
        norms_h = np.linalg.norm(X_high, axis=1, keepdims=True)
        norms_r = np.linalg.norm(X_reconstructed, axis=1, keepdims=True)
        norms_h = np.where(norms_h == 0, 1.0, norms_h)
        norms_r = np.where(norms_r == 0, 1.0, norms_r)
        cos_sim = np.sum(X_high / norms_h * X_reconstructed / norms_r, axis=1)
        return float(1.0 - np.mean(cos_sim))
    if metric == "relative":
        norms = np.linalg.norm(X_high, axis=1)
        norms = np.where(norms == 0, 1.0, norms)
        return float(np.mean(np.linalg.norm(X_high - X_reconstructed, axis=1) / norms))
    raise ValueError(f"Unknown metric: {metric}")


def arrow_spatial_information_gain(
    X_high: np.ndarray, X_spatial_recon: np.ndarray, X_full_recon: np.ndarray
) -> float:
    """Mean per-dimension spatial-information recovery contributed by arrows."""
    corrs_spatial = []
    corrs_full = []
    for j in range(X_high.shape[1]):
        corrs_spatial.append(_safe_corr_1d(X_high[:, j], X_spatial_recon[:, j]))
        corrs_full.append(_safe_corr_1d(X_high[:, j], X_full_recon[:, j]))
    mean_spatial = _nanmean_or_zero(corrs_spatial)
    mean_full = _nanmean_or_zero(corrs_full)
    return mean_full - mean_spatial


def arrow_knn_recall(
    X_high: np.ndarray, spatial: np.ndarray, arrows: np.ndarray, k: int = 10
) -> float:
    """kNN recall using concatenated standardized spatial and arrow channels."""
    n = len(spatial)
    arrows_flat = arrows.reshape(n, -1)
    spatial_std = np.std(spatial, axis=0, keepdims=True)
    arrows_std = np.std(arrows_flat, axis=0, keepdims=True)
    spatial_std = np.where(spatial_std == 0, 1.0, spatial_std)
    arrows_std = np.where(arrows_std == 0, 1.0, arrows_std)
    combined = np.column_stack([spatial / spatial_std, arrows_flat / arrows_std])
    return knn_recall(X_high, combined, k=k)


def arrow_consistency(arrows: np.ndarray, X_high: np.ndarray, k: int = 10) -> float:
    """Mean cosine similarity of arrows among HD nearest neighbors."""
    k = _effective_k(len(X_high), k)
    if k == 0:
        return 1.0

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X_high)
    _, idx = nn.kneighbors(X_high)
    idx = idx[:, 1:]

    arrows_flat = arrows.reshape(len(arrows), -1)
    norms = np.linalg.norm(arrows_flat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    arrows_normed = arrows_flat / norms

    consistencies = [
        float(np.mean(arrows_normed[idx[i]] @ arrows_normed[i]))
        for i in range(len(arrows))
    ]
    return float(np.mean(consistencies))


def knn_classification_metrics(
    representation: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 10,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Dict[str, float]:
    """Downstream kNN classification accuracy and macro-F1."""
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return {"accuracy": 1.0, "macro_f1": 1.0}

    counts = np.unique(labels, return_counts=True)[1]
    stratify = labels if np.min(counts) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        representation,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    k_eff = _effective_k(len(X_train), n_neighbors)
    clf = KNeighborsClassifier(n_neighbors=k_eff)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
    }


def knn_classification_accuracy(
    representation: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 10,
    test_size: float = 0.3,
    random_state: int = 42,
) -> float:
    """Convenience wrapper returning only downstream kNN accuracy."""
    return knn_classification_metrics(
        representation,
        labels,
        n_neighbors=n_neighbors,
        test_size=test_size,
        random_state=random_state,
    )["accuracy"]


def flattened_afe_representation(spatial: np.ndarray, arrows: np.ndarray) -> np.ndarray:
    """Return the raw concatenated AFE representation ``[spatial, arrows]``."""
    return np.column_stack([spatial, arrows.reshape(len(spatial), -1)])


def dimension_coverage(n_features: int, n_arrows: int, spatial_dims: int = 3) -> Dict[str, float]:
    """Representation budget summary for a 3D+k-arrow AFE encoding."""
    arrow_dims = int(n_arrows) * 3
    total_dims = int(spatial_dims) + arrow_dims
    return {
        "n_features": int(n_features),
        "spatial_dims": int(spatial_dims),
        "n_arrows": int(n_arrows),
        "arrow_channel_capacity": arrow_dims,
        "total_representation_dims": total_dims,
        "coverage_fraction": float(min(total_dims, int(n_features)) / max(1, int(n_features))),
    }


__all__ = [
    "arrow_consistency",
    "arrow_spatial_information_gain",
    "arrow_knn_recall",
    "centroid_triplet_accuracy",
    "continuity",
    "correlation_matrix_to_spatial",
    "dimension_coverage",
    "flattened_afe_representation",
    "knn_classification_accuracy",
    "knn_classification_metrics",
    "knn_recall",
    "normalized_stress",
    "random_triplet_accuracy",
    "recon_knn_recall",
    "reconstruction_error",
    "silhouette",
    "spearman_distance_correlation",
    "spatial_information_gap",
    "trustworthiness",
]
