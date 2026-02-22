"""Compare AFE vs standard PaCMAP 3D on information preservation.

Measures how much additional information AFE's arrows capture beyond
what spatial position alone provides.

Usage:
    python benchmarks/compare_methods.py [--datasets ...] [--n-arrows 3] [--n-seeds 3]
    python benchmarks/compare_methods.py --category scrna --n-arrows 5,25,50
    python benchmarks/compare_methods.py --list-datasets
"""

import argparse
import json
import time
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from afe import ArrowFieldEmbedding
from benchmarks.datasets import (
    DATASETS, LABELED_DATASETS,
    DATASETS_GENERAL, DATASETS_SCRNA, DATASET_CATEGORIES,
)
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


def run_pacmap_baseline(X, labels, seed):
    """Run standard PaCMAP 3D as baseline."""
    import pacmap

    t0 = time.time()
    reducer = pacmap.PaCMAP(n_components=3, random_state=seed)
    Y = reducer.fit_transform(X).astype(np.float32)
    elapsed = time.time() - t0

    metrics = {
        "method": "pacmap_3d",
        "seed": seed,
        "time_seconds": elapsed,
        "knn_recall_k10": knn_recall(X, Y, k=10),
        "knn_recall_k50": knn_recall(X, Y, k=50),
        "spearman_dist_corr": spearman_distance_correlation(X, Y),
        "random_triplet_acc": random_triplet_accuracy(X, Y),
    }

    # Reconstruction via linear regression (same as AFE uses)
    n = len(Y)
    Y_aug = np.column_stack([Y, np.ones(n)])
    wb = np.linalg.lstsq(Y_aug, X, rcond=None)[0]
    X_recon = Y_aug @ wb
    metrics["reconstruction_mse"] = reconstruction_error(X, X_recon, metric='mse')
    metrics["reconstruction_cosine"] = reconstruction_error(X, X_recon, metric='cosine')

    if labels is not None:
        ct = centroid_triplet_accuracy(X, Y, labels)
        if ct is not None:
            metrics["centroid_triplet_acc"] = ct

    return metrics


def run_afe(X, labels, encoding_mode, n_arrows, seed):
    """Run AFE with specified mode."""
    t0 = time.time()
    afe = ArrowFieldEmbedding(
        n_arrows=n_arrows,
        encoding_mode=encoding_mode,
        backend="pacmap",
        random_state=seed,
        normalize_arrows=False,
    )
    result = afe.fit_transform(X)
    elapsed = time.time() - t0

    spatial = result["spatial"]
    arrows = result["arrows"]

    # Standard DR metrics (spatial only — same as PaCMAP baseline)
    metrics = {
        "method": f"afe_{encoding_mode}_{n_arrows}arr",
        "encoding_mode": encoding_mode,
        "n_arrows": n_arrows,
        "seed": seed,
        "time_seconds": elapsed,
        "knn_recall_k10": knn_recall(X, spatial, k=10),
        "knn_recall_k50": knn_recall(X, spatial, k=50),
        "spearman_dist_corr": spearman_distance_correlation(X, spatial),
        "random_triplet_acc": random_triplet_accuracy(X, spatial),
    }

    # AFE-enhanced metrics (spatial + arrows)
    metrics["arrow_knn_recall_k10"] = arrow_knn_recall(X, spatial, arrows, k=10)
    metrics["arrow_knn_recall_k50"] = arrow_knn_recall(X, spatial, arrows, k=50)
    metrics["arrow_consistency"] = arrow_consistency(arrows, X, k=10)

    # Reconstruction
    X_recon = afe.reconstruct()
    metrics["reconstruction_mse"] = reconstruction_error(X, X_recon, metric='mse')
    metrics["reconstruction_cosine"] = reconstruction_error(X, X_recon, metric='cosine')

    # Spatial-only reconstruction for information gain
    n = len(spatial)
    sp_aug = np.column_stack([spatial, np.ones(n)])
    wb = np.linalg.lstsq(sp_aug, X, rcond=None)[0]
    X_spatial_recon = sp_aug @ wb
    metrics["arrow_info_gain"] = arrow_information_gain(X, X_spatial_recon, X_recon)

    if labels is not None:
        ct = centroid_triplet_accuracy(X, spatial, labels)
        if ct is not None:
            metrics["centroid_triplet_acc"] = ct

    # Metadata
    gap = result["metadata"]["gap_report"]
    metrics["n_residual_dims"] = len(gap["residual_dims"])
    metrics["info_gap_score"] = gap["information_gap_score"]

    return metrics


def compare_on_dataset(dataset_name, arrow_counts=(3,), n_seeds=3,
                       modes=("direct", "pca", "adaptive")):
    """Run full comparison on one dataset across multiple arrow counts."""
    loader = DATASETS[dataset_name]
    X, y = loader()
    labels = y if dataset_name in LABELED_DATASETS else None
    n, d = X.shape

    print(f"\n{'='*70}")
    print(f"  {dataset_name}  ({n} x {d})")
    print(f"{'='*70}")

    all_results = []

    # Baseline: PaCMAP 3D (run once, shared across arrow counts)
    print(f"\n  PaCMAP 3D (baseline)")
    baseline_metrics = []
    for seed in range(n_seeds):
        m = run_pacmap_baseline(X, labels, seed)
        m["dataset"] = dataset_name
        all_results.append(m)
        baseline_metrics.append(m)

    mean_bl = {
        k: np.mean([r[k] for r in baseline_metrics])
        for k in ["knn_recall_k10", "spearman_dist_corr", "random_triplet_acc",
                   "reconstruction_mse"]
    }
    print(f"    kNN@10: {mean_bl['knn_recall_k10']:.3f}  "
          f"Spearman: {mean_bl['spearman_dist_corr']:.3f}  "
          f"Triplet: {mean_bl['random_triplet_acc']:.3f}  "
          f"Recon MSE: {mean_bl['reconstruction_mse']:.4f}")

    # AFE modes x arrow counts
    for n_arrows in arrow_counts:
        for mode in modes:
            print(f"\n  AFE ({mode}, {n_arrows} arrows)")
            mode_metrics = []
            for seed in range(n_seeds):
                m = run_afe(X, labels, mode, n_arrows, seed)
                m["dataset"] = dataset_name
                all_results.append(m)
                mode_metrics.append(m)

            mean_m = {
                k: np.mean([r[k] for r in mode_metrics if k in r])
                for k in ["knn_recall_k10", "arrow_knn_recall_k10", "spearman_dist_corr",
                           "random_triplet_acc", "reconstruction_mse", "arrow_info_gain",
                           "arrow_consistency"]
            }
            knn_gain = mean_m["arrow_knn_recall_k10"] - mean_bl["knn_recall_k10"]
            recon_gain = mean_bl["reconstruction_mse"] - mean_m["reconstruction_mse"]

            print(f"    kNN@10 (spatial):  {mean_m['knn_recall_k10']:.3f}  "
                  f"(same backend)")
            print(f"    kNN@10 (+ arrows): {mean_m['arrow_knn_recall_k10']:.3f}  "
                  f"({'+'if knn_gain>=0 else ''}{knn_gain:.3f} vs baseline)")
            print(f"    Recon MSE:         {mean_m['reconstruction_mse']:.4f}  "
                  f"({'+'if recon_gain>=0 else ''}{recon_gain:.4f} improvement)")
            print(f"    Arrow info gain:   {mean_m['arrow_info_gain']:.4f}")
            print(f"    Arrow consistency: {mean_m['arrow_consistency']:.3f}")

    return all_results


def print_summary(all_results):
    """Print a summary table across all datasets."""
    print(f"\n\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}\n")

    datasets = sorted(set(r["dataset"] for r in all_results))
    methods = sorted(set(r["method"] for r in all_results))

    # Header
    header = f"{'Method':<30} {'kNN@10':>8} {'Spearman':>9} {'Recon MSE':>10}"
    if any("arrow_knn_recall_k10" in r for r in all_results):
        header += f" {'Arr kNN@10':>11} {'Info Gain':>10}"
    print(header)
    print("-" * len(header))

    for ds in datasets:
        print(f"\n  [{ds}]")
        for method in methods:
            rows = [r for r in all_results if r["dataset"] == ds and r["method"] == method]
            if not rows:
                continue
            knn = np.mean([r["knn_recall_k10"] for r in rows])
            sp = np.mean([r["spearman_dist_corr"] for r in rows])
            mse = np.mean([r["reconstruction_mse"] for r in rows])
            line = f"  {method:<28} {knn:>8.3f} {sp:>9.3f} {mse:>10.4f}"

            if "arrow_knn_recall_k10" in rows[0]:
                aknn = np.mean([r["arrow_knn_recall_k10"] for r in rows])
                aig = np.mean([r.get("arrow_info_gain", 0) for r in rows])
                line += f" {aknn:>11.3f} {aig:>10.4f}"

            print(line)


def list_datasets():
    """Print all available datasets grouped by category."""
    print("\nAvailable datasets:\n")
    for cat_name, cat_dict in DATASET_CATEGORIES.items():
        print(f"  [{cat_name}] ({len(cat_dict)} datasets)")
        for name in sorted(cat_dict.keys()):
            labeled = " (labeled)" if name in LABELED_DATASETS else ""
            print(f"    {name}{labeled}")
        print()
    print(f"Total: {len(DATASETS)} datasets")


def main():
    parser = argparse.ArgumentParser(description="AFE vs PaCMAP comparison")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names")
    parser.add_argument("--category", type=str, default=None,
                        choices=["general", "scrna", "all"],
                        help="Run all datasets in a category")
    parser.add_argument("--n-arrows", type=str, default="3",
                        help="Comma-separated arrow counts (e.g., 5,25,50)")
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--modes", type=str, default="direct,pca,adaptive")
    parser.add_argument("--output", type=str, default="comparison_results.json")
    parser.add_argument("--list-datasets", action="store_true",
                        help="Print available datasets and exit")
    args = parser.parse_args()

    if args.list_datasets:
        list_datasets()
        return

    # Resolve dataset list
    if args.category:
        if args.category == "all":
            datasets = list(DATASETS.keys())
        else:
            datasets = list(DATASET_CATEGORIES[args.category].keys())
    elif args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
    else:
        datasets = ["swiss_roll", "hierarchical_gaussians"]

    modes = tuple(m.strip() for m in args.modes.split(","))
    arrow_counts = tuple(int(x.strip()) for x in args.n_arrows.split(","))

    all_results = []
    for ds in datasets:
        if ds not in DATASETS:
            print(f"Warning: unknown dataset '{ds}', skipping")
            continue
        results = compare_on_dataset(ds, arrow_counts=arrow_counts,
                                     n_seeds=args.n_seeds, modes=modes)
        all_results.extend(results)

    print_summary(all_results)

    # Save
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = [{k: _convert(v) for k, v in row.items()} for row in all_results]
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
