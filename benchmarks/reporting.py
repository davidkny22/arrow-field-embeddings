"""Reporting and summary output for benchmark results."""

import numpy as np

from benchmarks.config import ALL_BACKENDS, ALL_MODES
from benchmarks.datasets import (
    DATASETS, DATASET_CATEGORIES,
)


def print_summary(all_results):
    """Print a summary table across all datasets."""
    # Filter out significance records
    bench_results = [r for r in all_results if r.get("type") != "significance"]
    if not bench_results:
        return

    print(f"\n\n{'='*100}")
    print("  SUMMARY")
    print(f"{'='*100}\n")

    datasets = sorted(set(r["dataset"] for r in bench_results))
    methods = sorted(set(r["method"] for r in bench_results))

    header = (f"{'Method':<35} {'kNN@10':>7} {'kNN@50':>7} "
              f"{'Trust':>6} {'Cont':>6} {'Spear':>6} "
              f"{'Triplet':>7} {'Stress':>7} {'ReconMSE':>9}")
    has_arrows = any("arrow_knn_recall_k10" in r for r in bench_results)
    if has_arrows:
        header += f" {'ArrKNN':>7} {'InfoGn':>7}"
    has_sil = any("silhouette_score" in r for r in bench_results)
    if has_sil:
        header += f" {'Silh':>6}"
    print(header)
    print("-" * len(header))

    for ds in datasets:
        print(f"\n  [{ds}]")
        for method in methods:
            rows = [r for r in bench_results
                    if r["dataset"] == ds and r["method"] == method]
            if not rows:
                continue

            def _mean(key):
                vals = [r[key] for r in rows if key in r]
                return np.mean(vals) if vals else float('nan')

            knn10 = _mean("knn_recall_k10")
            knn50 = _mean("knn_recall_k50")
            trust = _mean("trustworthiness_k10")
            cont = _mean("continuity_k10")
            spear = _mean("spearman_dist_corr")
            trip = _mean("random_triplet_acc")
            stress = _mean("normalized_stress")
            mse = _mean("reconstruction_mse")

            line = (f"  {method:<33} {knn10:>7.3f} {knn50:>7.3f} "
                    f"{trust:>6.3f} {cont:>6.3f} {spear:>6.3f} "
                    f"{trip:>7.3f} {stress:>7.4f} {mse:>9.4f}")

            if has_arrows and "arrow_knn_recall_k10" in rows[0]:
                aknn = _mean("arrow_knn_recall_k10")
                aig = _mean("arrow_spatial_information_gain")
                line += f" {aknn:>7.3f} {aig:>7.4f}"
            elif has_arrows:
                line += f" {'':>7} {'':>7}"

            if has_sil:
                sil_val = _mean("silhouette_score")
                if not np.isnan(sil_val):
                    line += f" {sil_val:>6.3f}"
                else:
                    line += f" {'':>6}"

            print(line)


def print_significance_summary(sig_results, aggregated):
    """Print significance testing results."""
    if not sig_results:
        print("\nNo significance results to display.")
        return

    print(f"\n\n{'='*100}")
    print("  SIGNIFICANCE TESTING")
    print(f"{'='*100}\n")

    # Group by backend
    backends = sorted(set(r["baseline"].replace("_3d", "") for r in sig_results))

    for backend in backends:
        backend_results = [r for r in sig_results if r["baseline"] == f"{backend}_3d"]
        if not backend_results:
            continue

        print(f"\n  Backend: {backend.upper()}")
        print(f"  {'Enhanced Method':<40} {'Metric':<22} "
              f"{'Base':>7} {'Enh':>7} {'Diff':>7} "
              f"{'Cohen d':>8} {'p-val':>8} {'q-val':>8} {'Sig':>4}")
        print("  " + "-" * 120)

        for r in sorted(backend_results,
                        key=lambda x: (x["enhanced"], x["metric"])):
            sig_marker = "*" if r.get("significant", False) else ""
            print(f"  {r['enhanced']:<40} {r['metric']:<22} "
                  f"{r['baseline_mean']:>7.3f} {r['enhanced_mean']:>7.3f} "
                  f"{r['mean_diff']:>+7.3f} "
                  f"{r['cohens_d']:>8.2f} {r['p_value']:>8.4f} "
                  f"{r.get('q_value', 1.0):>8.4f} {sig_marker:>4}")

    # Print aggregated win/tie/loss
    if aggregated:
        print(f"\n\n  {'='*80}")
        print("  WIN / TIE / LOSS SUMMARY (across datasets, q < 0.05)")
        print(f"  {'='*80}\n")

        print(f"  {'Method':<40} {'Metric':<22} "
              f"{'W':>3} {'T':>3} {'L':>3} "
              f"{'CombP':>8} {'EffSize':>8}")
        print("  " + "-" * 90)

        for a in sorted(aggregated,
                        key=lambda x: (x["method"], x["metric"])):
            print(f"  {a['method']:<40} {a['metric']:<22} "
                  f"{a['wins']:>3} {a['ties']:>3} {a['losses']:>3} "
                  f"{a['combined_p']:>8.4f} {a['mean_effect_size']:>+8.2f}")


def list_datasets():
    """Print all available datasets grouped by category."""
    from benchmarks.datasets import LABELED_DATASETS
    print("\nAvailable datasets:\n")
    for cat_name, cat_dict in DATASET_CATEGORIES.items():
        print(f"  [{cat_name}] ({len(cat_dict)} datasets)")
        for name in sorted(cat_dict.keys()):
            labeled = " (labeled)" if name in LABELED_DATASETS else ""
            print(f"    {name}{labeled}")
        print()
    print(f"Total: {len(DATASETS)} datasets")
