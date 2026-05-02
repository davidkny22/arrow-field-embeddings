"""Shared metric computation for benchmark runners.

Eliminates duplication between standalone (baseline) and AFE-enhanced runs.
"""

import numpy as np

from afe.backends import get_backend
from afe.evaluation import (
    knn_recall,
    knn_classification_metrics,
    flattened_afe_representation,
    spearman_distance_correlation,
    random_triplet_accuracy,
    centroid_triplet_accuracy,
    reconstruction_error,
    arrow_spatial_information_gain,
    arrow_knn_recall,
    recon_knn_recall,
    arrow_consistency,
    trustworthiness,
    continuity,
    silhouette,
    normalized_stress,
)


def compute_spatial_embedding(X, backend_name, seed, backend_params=None):
    """Compute one fixed spatial embedding for a dataset/backend/seed."""
    backend = get_backend(backend_name, backend_kwargs=backend_params, random_state=seed)
    return backend.fit_transform(X)


def _linear_reconstruct(X, Y):
    """Reconstruct X from Y via linear regression. Returns (X_recon, wb)."""
    n = len(Y)
    Y_aug = np.column_stack([Y, np.ones(n)])
    wb = np.linalg.lstsq(Y_aug, X, rcond=None)[0]
    X_recon = Y_aug @ wb
    return X_recon, wb


def _compute_standard_metrics(X, Y, labels=None, seed=0):
    """Compute standard DR metrics shared by standalone and AFE runs.

    Parameters
    ----------
    X : ndarray (n, d)
        Original high-dimensional data.
    Y : ndarray (n, 3)
        3D spatial embedding.
    labels : ndarray (n,), optional
        Per-point labels for classification/triplet metrics.
    seed : int
        Random seed for reproducible classification split.

    Returns
    -------
    metrics : dict
        Dictionary with all standard DR metrics.
    """
    metrics = {
        "knn_recall_k10": knn_recall(X, Y, k=10),
        "knn_recall_k50": knn_recall(X, Y, k=50),
        "spearman_dist_corr": spearman_distance_correlation(X, Y),
        "random_triplet_acc": random_triplet_accuracy(X, Y),
        "trustworthiness_k10": trustworthiness(X, Y, k=10),
        "continuity_k10": continuity(X, Y, k=10),
        "normalized_stress": normalized_stress(X, Y),
    }

    # Reconstruction via linear regression
    X_recon, _ = _linear_reconstruct(X, Y)
    metrics["reconstruction_mse"] = reconstruction_error(X, X_recon, metric='mse')
    metrics["reconstruction_cosine"] = reconstruction_error(X, X_recon, metric='cosine')

    # Label-dependent metrics
    if labels is not None:
        ct = centroid_triplet_accuracy(X, Y, labels)
        if ct is not None:
            metrics["centroid_triplet_acc"] = ct
        sil = silhouette(Y, labels)
        if sil is not None:
            metrics["silhouette_score"] = sil
        clf_hd = knn_classification_metrics(X, labels, n_neighbors=10, random_state=seed)
        metrics["knn_class_acc_hd"] = clf_hd["accuracy"]
        metrics["knn_class_f1_hd"] = clf_hd["macro_f1"]
        clf_spatial = knn_classification_metrics(Y, labels, n_neighbors=10, random_state=seed)
        metrics["knn_class_acc_spatial"] = clf_spatial["accuracy"]
        metrics["knn_class_f1_spatial"] = clf_spatial["macro_f1"]

    return metrics, X_recon


def _compute_afe_metrics(X, afe, spatial, arrows, labels=None, seed=0):
    """Compute all metrics for an AFE-enhanced run.

    Calls _compute_standard_metrics for the spatial baseline, then adds
    AFE-specific metrics (arrow KNN, reconstruction, classification on
    flat AFE and reconstructed HD).

    Parameters
    ----------
    X : ndarray (n, d)
        Original high-dimensional data.
    afe : ArrowFieldEmbedding
        Fitted AFE instance.
    spatial : ndarray (n, 3)
        3D spatial positions.
    arrows : ndarray (n, k, 3)
        Arrow field.
    labels : ndarray (n,), optional
        Per-point labels.
    seed : int
        Random seed for classification split.

    Returns
    -------
    metrics : dict
        Complete metric dictionary for the AFE run.
    """
    metrics, _ = _compute_standard_metrics(X, spatial, labels=labels, seed=seed)

    # AFE-enhanced metrics
    metrics["arrow_knn_recall_k10"] = arrow_knn_recall(X, spatial, arrows, k=10)
    metrics["arrow_knn_recall_k50"] = arrow_knn_recall(X, spatial, arrows, k=50)
    metrics["arrow_consistency"] = arrow_consistency(arrows, X, k=10)

    # Reconstruction from AFE
    X_recon = afe.reconstruct()
    metrics["reconstruction_mse"] = reconstruction_error(X, X_recon, metric='mse')
    metrics["reconstruction_cosine"] = reconstruction_error(X, X_recon, metric='cosine')
    metrics["recon_knn_recall_k10"] = recon_knn_recall(X, X_recon, k=10)
    metrics["recon_knn_recall_k50"] = recon_knn_recall(X, X_recon, k=50)

    # Spatial-only reconstruction for spatial information gain
    X_spatial_recon, _ = _linear_reconstruct(X, spatial)
    metrics["arrow_spatial_information_gain"] = arrow_spatial_information_gain(
        X, X_spatial_recon, X_recon
    )

    # Extended label-dependent metrics
    if labels is not None:
        afe_flat = flattened_afe_representation(spatial, arrows)
        clf_flat = knn_classification_metrics(afe_flat, labels, n_neighbors=10, random_state=seed)
        metrics["knn_class_acc_flat"] = clf_flat["accuracy"]
        metrics["knn_class_f1_flat"] = clf_flat["macro_f1"]
        clf_recon = knn_classification_metrics(X_recon, labels, n_neighbors=10, random_state=seed)
        metrics["knn_class_acc_recon"] = clf_recon["accuracy"]
        metrics["knn_class_f1_recon"] = clf_recon["macro_f1"]

    return metrics
