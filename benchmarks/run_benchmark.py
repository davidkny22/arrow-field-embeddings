"""Benchmark harness for AFE.

Usage:
    python benchmarks/run_benchmark.py [--datasets swiss_roll,mnist] [--modes direct,pca,adaptive]
"""

import argparse
import json
import time
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from afe import ArrowFieldEmbedding
from benchmarks.datasets import DATASETS, LABELED_DATASETS
from benchmarks.metrics import (
    knn_recall,
    spearman_distance_correlation,
    random_triplet_accuracy,
    centroid_triplet_accuracy,
    reconstruction_error,
    arrow_information_gain,
    arrow_knn_recall,
    arrow_consistency,
)


def run_single(X, labels, encoding_mode, n_arrows, seed):
    """Run a single AFE benchmark."""
    afe = ArrowFieldEmbedding(
        n_arrows=n_arrows,
        encoding_mode=encoding_mode,
        backend="pacmap",
        random_state=seed,
        normalize_arrows=False,  # raw values for metrics
    )

    t0 = time.time()
    result = afe.fit_transform(X)
    elapsed = time.time() - t0

    spatial = result["spatial"]
    arrows = result["arrows"]

    # Standard DR metrics (spatial only)
    metrics = {
        "encoding_mode": encoding_mode,
        "n_arrows": n_arrows,
        "seed": seed,
        "time_seconds": elapsed,
        "knn_recall_spatial": knn_recall(X, spatial, k=10),
        "spearman_spatial": spearman_distance_correlation(X, spatial),
        "triplet_spatial": random_triplet_accuracy(X, spatial),
    }

    # AFE-specific metrics
    metrics["arrow_knn_recall"] = arrow_knn_recall(X, spatial, arrows, k=10)
    metrics["arrow_consistency"] = arrow_consistency(arrows, X, k=10)

    # Reconstruction
    X_recon = afe.reconstruct()
    metrics["reconstruction_mse"] = reconstruction_error(X, X_recon, metric='mse')
    metrics["reconstruction_cosine"] = reconstruction_error(X, X_recon, metric='cosine')

    # Centroid triplet if labeled
    if labels is not None:
        ct = centroid_triplet_accuracy(X, spatial, labels)
        if ct is not None:
            metrics["centroid_triplet_spatial"] = ct

    # Metadata
    gap = result["metadata"]["gap_report"]
    metrics["n_residual_dims"] = len(gap["residual_dims"])
    metrics["info_gap_score"] = gap["information_gap_score"]

    return metrics


def run_benchmark(dataset_name, modes=None, n_arrows=3, n_seeds=3):
    """Run benchmark across modes and seeds for a dataset."""
    if modes is None:
        modes = ["direct", "pca", "adaptive"]

    loader = DATASETS[dataset_name]
    X, y = loader()
    labels = y if dataset_name in LABELED_DATASETS else None

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} ({X.shape[0]} x {X.shape[1]})")
    print(f"{'='*60}")

    all_results = []
    for mode in modes:
        print(f"\n  Mode: {mode}")
        for seed in range(n_seeds):
            metrics = run_single(X, labels, mode, n_arrows, seed)
            metrics["dataset"] = dataset_name
            all_results.append(metrics)
            print(f"    Seed {seed}: kNN={metrics['knn_recall_spatial']:.3f}, "
                  f"arrow_kNN={metrics['arrow_knn_recall']:.3f}, "
                  f"recon_MSE={metrics['reconstruction_mse']:.4f}, "
                  f"time={metrics['time_seconds']:.1f}s")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="AFE Benchmark")
    parser.add_argument("--datasets", type=str, default="swiss_roll,hierarchical_gaussians",
                        help="Comma-separated dataset names")
    parser.add_argument("--modes", type=str, default="direct,pca,adaptive",
                        help="Comma-separated encoding modes")
    parser.add_argument("--n-arrows", type=int, default=3)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]

    all_results = []
    for ds in datasets:
        if ds not in DATASETS:
            print(f"Warning: unknown dataset '{ds}', skipping")
            continue
        results = run_benchmark(ds, modes=modes, n_arrows=args.n_arrows,
                                n_seeds=args.n_seeds)
        all_results.extend(results)

    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = [
        {k: _convert(v) for k, v in row.items()} for row in all_results
    ]

    # Save
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
