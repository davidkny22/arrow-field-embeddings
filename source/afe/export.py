"""Export AFE results to viewer-compatible JSON format."""

import json
import gzip
import numpy as np
from pathlib import Path
from typing import Optional, Union

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


def _compute_clusters(labels: np.ndarray, positions: np.ndarray):
    """Build cluster metadata from labels and positions."""
    unique_labels = np.unique(labels)
    clusters = []
    for label_id in unique_labels:
        mask = labels == label_id
        cluster_positions = positions[mask]
        centroid = cluster_positions.mean(axis=0).tolist()
        clusters.append({
            "id": int(label_id),
            "label": str(label_id),
            "size": int(mask.sum()),
            "centroid": centroid,
        })
    return clusters


def _safe_float(v):
    """Convert to float, replacing NaN/Inf with 0."""
    f = float(v)
    if np.isnan(f) or np.isinf(f):
        return 0.0
    return f


def _sanitize(obj):
    """json.dumps default handler — turn non-serializable values into 0."""
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return 0.0
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _compute_per_point_recon_error(X_original, X_reconstructed):
    """Per-point MSE between original and reconstructed."""
    diff = X_original - X_reconstructed
    return (diff ** 2).mean(axis=1)


def export_for_viewer(
    afe,
    X: np.ndarray,
    labels: Optional[np.ndarray] = None,
    path: str = "dataset.json.gz",
    dataset_name: str = "Untitled",
    label_names: Optional[list] = None,
    compress: bool = True,
    compute_metrics: bool = True,
):
    """Export a fitted ArrowFieldEmbedding to viewer-compatible JSON.

    Parameters
    ----------
    afe : ArrowFieldEmbedding
        A fitted AFE instance (must have called fit_transform already).
    X : ndarray (n, d)
        The original high-dimensional data (needed for metrics/reconstruction).
    labels : ndarray (n,), optional
        Per-point labels. If None, all points get label 0.
    path : str
        Output path. If compress=True, should end in .json.gz.
    dataset_name : str
        Human-readable name for the dataset.
    label_names : list of str, optional
        Unique label strings. If None, derived from labels.
    compress : bool
        Whether to gzip the output.
    compute_metrics : bool
        Whether to compute kNN recall, reconstruction MSE, etc.
    """
    spatial = afe.get_spatial()       # (n, 3)
    arrows = afe.get_arrows()         # (n, k, 3)
    gap_report = afe.get_gap_report()
    n_points = spatial.shape[0]
    n_arrows = arrows.shape[1]

    # Labels
    if labels is None:
        labels = np.zeros(n_points, dtype=int)

    # Build label name mapping
    unique_labels = np.unique(labels)
    if label_names is None:
        label_names_list = [str(l) for l in unique_labels]
    else:
        label_names_list = list(label_names)

    # Map original labels to indices into label_names_list
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    label_indices = np.array([label_to_idx[l] for l in labels], dtype=int)

    # Clusters
    clusters = _compute_clusters(label_indices, spatial)
    # Update cluster labels with actual names
    for c in clusters:
        if c["id"] < len(label_names_list):
            c["label"] = label_names_list[c["id"]]

    # Reconstruction error
    recon_error = None
    recon_error_spatial = None
    try:
        X_recon = afe.reconstruct()
        recon_error = _compute_per_point_recon_error(X, X_recon).tolist()

        # Spatial-only reconstruction: use reconstructor with zero arrows
        if hasattr(afe, '_reconstructor') and afe._reconstructor is not None:
            zero_arrows = np.zeros_like(arrows)
            X_spatial_recon = afe._reconstructor.reconstruct(spatial, zero_arrows)
            recon_error_spatial = _compute_per_point_recon_error(
                X, X_spatial_recon
            ).tolist()
    except Exception:
        pass

    # Metrics
    metrics = {
        "knn_recall_k10": 0.0,
        "arrow_knn_recall_k10": 0.0,
        "reconstruction_mse": 0.0,
        "arrow_info_gain": 0.0,
        "spearman_dist_corr": 0.0,
    }

    if compute_metrics:
        try:
            from benchmarks.metrics import (
                knn_recall,
                spearman_distance_correlation,
                reconstruction_error as recon_err_metric,
                arrow_knn_recall,
                arrow_information_gain,
            )

            metrics["knn_recall_k10"] = float(knn_recall(X, spatial, k=10))
            metrics["spearman_dist_corr"] = float(
                spearman_distance_correlation(X, spatial)
            )

            X_recon_full = afe.reconstruct()
            metrics["reconstruction_mse"] = float(
                recon_err_metric(X, X_recon_full, metric="mse")
            )

            metrics["arrow_knn_recall_k10"] = float(
                arrow_knn_recall(X, spatial, arrows, k=10)
            )

            # Spatial-only reconstruction for info gain
            try:
                zero_arr = np.zeros_like(arrows)
                X_spatial_recon = afe._reconstructor.reconstruct(
                    spatial, zero_arr
                )
                metrics["arrow_info_gain"] = float(
                    arrow_information_gain(X, X_spatial_recon, X_recon_full)
                )
            except Exception:
                pass
        except ImportError:
            pass
        except Exception as e:
            print(f"Warning: metric computation failed: {e}")

    # Build the dataset object
    dataset = {
        "version": 1,
        "dataset": dataset_name,
        "encoding_mode": str(getattr(afe, 'encoding_mode', 'unknown')),
        "n_points": n_points,
        "n_arrows": n_arrows,
        "embedding_dim": int(X.shape[1]),
        "positions": spatial.flatten().tolist(),
        "arrows": arrows.reshape(n_points, n_arrows * 3).flatten().tolist(),
        "label_indices": label_indices.tolist(),
        "label_names": label_names_list,
        "clusters": clusters,
        "gap_report": {
            "information_gap_score": float(
                gap_report.get("information_gap_score", 0)
            ),
            "n_residual_dims": len(gap_report.get("residual_dims", [])),
            "n_captured_dims": len(gap_report.get("captured_dims", [])),
        },
        "metrics": {k: _safe_float(v) for k, v in metrics.items()},
    }

    if recon_error is not None:
        dataset["recon_error"] = recon_error
    if recon_error_spatial is not None:
        dataset["recon_error_spatial"] = recon_error_spatial

    # Write
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    json_bytes = json.dumps(
        dataset, separators=(",", ":"), allow_nan=False, default=_sanitize
    ).encode("utf-8")

    if compress:
        with gzip.open(out_path, "wb") as f:
            f.write(json_bytes)
    else:
        with open(out_path, "wb") as f:
            f.write(json_bytes)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Exported {n_points} points × {n_arrows} arrows to {out_path} ({size_mb:.1f} MB)")
    return out_path
