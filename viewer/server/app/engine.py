"""AFE computation engine -- wraps the AFE library for the server."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Make the AFE package importable by adding the project root to sys.path.
#   engine.py -> app/ -> server/ -> viewer/ -> project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_source_dir = str(PROJECT_ROOT / "source")
_benchmarks_dir = str(PROJECT_ROOT)

for _p in (_source_dir, _benchmarks_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from afe import ArrowFieldEmbedding  # noqa: E402
from afe.export import (  # noqa: E402
    _compute_clusters,
    _compute_per_point_recon_error,
)

from sklearn.neighbors import NearestNeighbors  # noqa: E402


class AFEEngine:
    """Stateful wrapper around the AFE library.

    Keeps the most recently computed AFE result so that follow-up
    endpoints (/neighbors, /reconstruct) can reference it.
    """

    def __init__(self) -> None:
        self.afe: ArrowFieldEmbedding | None = None
        self.X: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.last_result: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Core AFE computation
    # ------------------------------------------------------------------

    def run_afe(
        self,
        X: np.ndarray,
        labels: np.ndarray | None = None,
        n_arrows: int = 2,
        encoding_mode: str = "direct",
        backend: str = "pacmap",
        random_state: int | None = 42,
        dataset_name: str = "Untitled",
    ) -> dict[str, Any]:
        """Run AFE on *X* and return a viewer-compatible dataset dict."""
        X = np.asarray(X, dtype=np.float32)
        self.X = X

        afe = ArrowFieldEmbedding(
            n_arrows=n_arrows,
            encoding_mode=encoding_mode,
            backend=backend,
            random_state=random_state,
            verbose=True,
        )
        afe.fit_transform(X)
        self.afe = afe

        spatial = afe.get_spatial()       # (n, 3)
        arrows = afe.get_arrows()         # (n, k, 3)
        gap_report = afe.get_gap_report()
        n_points = spatial.shape[0]
        n_arr = arrows.shape[1]

        # Labels
        if labels is None:
            labels = np.zeros(n_points, dtype=int)
        self.labels = labels

        unique_labels = np.unique(labels)
        label_names_list = [str(l) for l in unique_labels]
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        label_indices = np.array([label_to_idx[l] for l in labels], dtype=int)

        # Clusters
        clusters = _compute_clusters(label_indices, spatial)
        for c in clusters:
            if c["id"] < len(label_names_list):
                c["label"] = label_names_list[c["id"]]

        # Reconstruction error
        recon_error = None
        recon_error_spatial = None
        try:
            X_recon = afe.reconstruct()
            recon_error = _compute_per_point_recon_error(X, X_recon).tolist()

            if afe._reconstructor is not None:
                zero_arrows = np.zeros_like(afe._arrows_raw)
                X_spatial_recon = afe._reconstructor.reconstruct(
                    spatial, zero_arrows
                )
                recon_error_spatial = _compute_per_point_recon_error(
                    X, X_spatial_recon
                ).tolist()
        except Exception:
            pass

        dataset: dict[str, Any] = {
            "version": 1,
            "dataset": dataset_name,
            "encoding_mode": encoding_mode,
            "n_points": n_points,
            "n_arrows": n_arr,
            "embedding_dim": int(X.shape[1]),
            "positions": spatial.flatten().tolist(),
            "arrows": arrows.reshape(n_points, n_arr * 3).flatten().tolist(),
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
            "metrics": {
                "knn_recall_k10": 0.0,
                "arrow_knn_recall_k10": 0.0,
                "reconstruction_mse": 0.0,
                "arrow_info_gain": 0.0,
                "spearman_dist_corr": 0.0,
            },
        }

        if recon_error is not None:
            dataset["recon_error"] = recon_error
        if recon_error_spatial is not None:
            dataset["recon_error_spatial"] = recon_error_spatial

        self.last_result = dataset
        return dataset

    # ------------------------------------------------------------------
    # k-NN neighbours on the 3D spatial positions
    # ------------------------------------------------------------------

    def get_neighbors(
        self, positions: np.ndarray, index: int, k: int = 10
    ) -> dict[str, Any]:
        """Return the *k* nearest neighbours of point *index*.

        Parameters
        ----------
        positions : ndarray (n, 3)
            3-D point positions.
        index : int
            Index of the query point.
        k : int
            Number of neighbours to return.

        Returns
        -------
        dict with ``indices`` (list[int]) and ``distances`` (list[float]).
        """
        positions = np.asarray(positions, dtype=np.float32)
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
        nn.fit(positions)
        distances, indices = nn.kneighbors(positions[index : index + 1])
        # Exclude the query point itself (always at distance 0)
        mask = indices[0] != index
        return {
            "indices": indices[0][mask][:k].tolist(),
            "distances": distances[0][mask][:k].tolist(),
        }

    # ------------------------------------------------------------------
    # Per-point reconstruction detail
    # ------------------------------------------------------------------

    def get_reconstruction(
        self, index: int
    ) -> dict[str, Any]:
        """Per-dimension reconstruction detail for a single point.

        Returns
        -------
        dict with:
            original        : list[float]   -- HD vector
            reconstructed   : list[float]   -- full (spatial+arrows) reconstruction
            spatial_only    : list[float]   -- spatial-only reconstruction
            per_dim_error   : list[float]   -- |original - reconstructed| per dim
        """
        if self.afe is None or self.X is None:
            raise RuntimeError("No AFE result available. Run /run_afe first.")

        afe = self.afe
        X = self.X
        original = X[index].astype(float)

        # Full reconstruction
        X_recon_full = afe.reconstruct()
        reconstructed = X_recon_full[index].astype(float)

        # Spatial-only reconstruction (zero out arrows)
        spatial = afe.get_spatial()
        zero_arrows = np.zeros_like(afe._arrows_raw)
        X_spatial_recon = afe._reconstructor.reconstruct(spatial, zero_arrows)
        spatial_only = X_spatial_recon[index].astype(float)

        per_dim_error = np.abs(original - reconstructed)

        return {
            "original": original.tolist(),
            "reconstructed": reconstructed.tolist(),
            "spatial_only": spatial_only.tolist(),
            "per_dim_error": per_dim_error.tolist(),
        }


# Module-level singleton so routes can share state.
engine = AFEEngine()
