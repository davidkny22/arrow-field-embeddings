"""Evaluation metrics for AFE: standard DR metrics + AFE-specific.

Standard metrics adapted from continuous-pacmap-public/benchmarks/metrics.py.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist


# ── Standard DR Metrics ──────────────────────────────────────────────────

def knn_recall(X_high, X_low, k=10):
    """Fraction of true k-NN preserved in embedding."""
    nn_high = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X_high)
    nn_low = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X_low)

    _, idx_high = nn_high.kneighbors(X_high)
    _, idx_low = nn_low.kneighbors(X_low)

    idx_high = idx_high[:, 1:]
    idx_low = idx_low[:, 1:]

    recalls = []
    for i in range(len(X_high)):
        overlap = len(set(idx_high[i]) & set(idx_low[i]))
        recalls.append(overlap / k)
    return np.mean(recalls)


def spearman_distance_correlation(X_high, X_low, n_sample=5000):
    """Spearman rank correlation between pairwise distances."""
    n = len(X_high)
    if n > n_sample:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, n_sample, replace=False)
        X_high = X_high[idx]
        X_low = X_low[idx]

    d_high = pdist(X_high)
    d_low = pdist(X_low)
    corr, _ = spearmanr(d_high, d_low)
    return corr


def random_triplet_accuracy(X_high, X_low, n_triplets=50000, random_state=42):
    """Fraction of random triplets with preserved distance ordering."""
    n = len(X_high)
    rng = np.random.RandomState(random_state)

    idx = np.empty((n_triplets, 3), dtype=int)
    for col in range(3):
        idx[:, col] = rng.randint(0, n, size=n_triplets)
    bad = (idx[:, 0] == idx[:, 1]) | (idx[:, 0] == idx[:, 2]) | (idx[:, 1] == idx[:, 2])
    while bad.any():
        n_bad = bad.sum()
        for col in range(3):
            idx[bad, col] = rng.randint(0, n, size=n_bad)
        bad = (idx[:, 0] == idx[:, 1]) | (idx[:, 0] == idx[:, 2]) | (idx[:, 1] == idx[:, 2])

    i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]

    d_high_ij = np.linalg.norm(X_high[i] - X_high[j], axis=1)
    d_high_ik = np.linalg.norm(X_high[i] - X_high[k], axis=1)
    d_low_ij = np.linalg.norm(X_low[i] - X_low[j], axis=1)
    d_low_ik = np.linalg.norm(X_low[i] - X_low[k], axis=1)

    preserved = (d_high_ij < d_high_ik) == (d_low_ij < d_low_ik)
    non_tie = d_high_ij != d_high_ik
    return float(np.mean(preserved[non_tie]))


def centroid_triplet_accuracy(X_high, X_low, labels):
    """Fraction of centroid triplets with preserved distance ordering."""
    if labels is None:
        return None

    from scipy.spatial.distance import cdist

    classes = np.unique(labels)
    n_classes = len(classes)
    if n_classes < 3:
        return None

    centroids_high = np.array([X_high[labels == c].mean(axis=0) for c in classes])
    centroids_low = np.array([X_low[labels == c].mean(axis=0) for c in classes])

    d_high = cdist(centroids_high, centroids_high, metric='euclidean')
    d_low = cdist(centroids_low, centroids_low, metric='euclidean')

    preserved = 0
    total = 0
    for i in range(n_classes):
        for j in range(n_classes):
            for k in range(j + 1, n_classes):
                if j == i or k == i:
                    continue
                if d_high[i, j] == d_high[i, k]:
                    continue
                same_order = (d_high[i, j] < d_high[i, k]) == (d_low[i, j] < d_low[i, k])
                preserved += int(same_order)
                total += 1

    return float(preserved / total) if total > 0 else None


# ── AFE-Specific Metrics ─────────────────────────────────────────────────

def reconstruction_error(X_high, X_reconstructed, metric='mse'):
    """Reconstruction error from spatial + arrows back to HD.

    Parameters
    ----------
    metric : {'mse', 'cosine', 'relative'}
    """
    if metric == 'mse':
        return float(np.mean((X_high - X_reconstructed) ** 2))
    elif metric == 'cosine':
        # Mean cosine distance
        norms_h = np.linalg.norm(X_high, axis=1, keepdims=True)
        norms_r = np.linalg.norm(X_reconstructed, axis=1, keepdims=True)
        norms_h = np.where(norms_h == 0, 1.0, norms_h)
        norms_r = np.where(norms_r == 0, 1.0, norms_r)
        cos_sim = np.sum(X_high / norms_h * X_reconstructed / norms_r, axis=1)
        return float(1.0 - np.mean(cos_sim))
    elif metric == 'relative':
        norms = np.linalg.norm(X_high, axis=1)
        norms = np.where(norms == 0, 1.0, norms)
        return float(np.mean(np.linalg.norm(X_high - X_reconstructed, axis=1) / norms))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def arrow_information_gain(X_high, X_spatial_recon, X_full_recon):
    """How much information arrows add beyond spatial alone.

    Computes improvement in correlation between original and reconstructed.
    """
    corr_spatial = np.corrcoef(X_high.ravel(), X_spatial_recon.ravel())[0, 1]
    corr_full = np.corrcoef(X_high.ravel(), X_full_recon.ravel())[0, 1]
    return float(corr_full - corr_spatial)


def arrow_knn_recall(X_high, spatial, arrows, k=10):
    """KNN recall using combined spatial + arrow representation.

    Builds a combined feature vector [spatial, flattened_arrows] and
    measures kNN recall against HD space.
    """
    n = len(spatial)
    arrows_flat = arrows.reshape(n, -1)
    # Normalize both to unit variance before combining
    spatial_std = np.std(spatial, axis=0, keepdims=True)
    spatial_std = np.where(spatial_std == 0, 1.0, spatial_std)
    arrows_std = np.std(arrows_flat, axis=0, keepdims=True)
    arrows_std = np.where(arrows_std == 0, 1.0, arrows_std)

    combined = np.column_stack([
        spatial / spatial_std,
        arrows_flat / arrows_std,
    ])
    return knn_recall(X_high, combined, k=k)


def arrow_consistency(arrows, X_high, k=10):
    """Do nearby points in HD space have similar arrow orientations?

    For each point's k nearest neighbors in HD, compute the mean
    cosine similarity of their arrow configurations.
    """
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X_high)
    _, idx = nn.kneighbors(X_high)
    idx = idx[:, 1:]  # exclude self

    n = len(arrows)
    arrows_flat = arrows.reshape(n, -1)

    # Normalize arrow vectors
    norms = np.linalg.norm(arrows_flat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    arrows_normed = arrows_flat / norms

    consistencies = []
    for i in range(n):
        neighbors = idx[i]
        cos_sims = arrows_normed[neighbors] @ arrows_normed[i]
        consistencies.append(np.mean(cos_sims))

    return float(np.mean(consistencies))
