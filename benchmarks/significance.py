"""Significance testing for benchmark results.

Paired Wilcoxon signed-rank + t-tests, bootstrap CIs, Cohen's d,
and Benjamini-Hochberg FDR correction.
"""

import numpy as np
from collections import defaultdict

from afe.reproducibility import RESULT_SCHEMA_VERSION


SIGNIFICANCE_METRICS = [
    "knn_recall_k10", "knn_recall_k50", "spearman_dist_corr",
    "random_triplet_acc", "trustworthiness_k10", "continuity_k10",
    "normalized_stress", "reconstruction_mse", "reconstruction_cosine",
    "recon_knn_recall_k10", "recon_knn_recall_k50",
    "silhouette_score", "centroid_triplet_acc",
    "knn_class_acc_hd", "knn_class_acc_spatial",
    "knn_class_acc_flat", "knn_class_acc_recon",
    "knn_class_f1_hd", "knn_class_f1_spatial",
    "knn_class_f1_flat", "knn_class_f1_recon",
]

# Metrics where lower is better (for correct direction interpretation)
LOWER_IS_BETTER = {"normalized_stress", "reconstruction_mse", "reconstruction_cosine"}


def _cohens_d(a, b):
    """Compute Cohen's d effect size for paired samples."""
    diff = np.array(a) - np.array(b)
    if len(diff) < 2 or np.std(diff) == 0:
        return 0.0
    return float(np.mean(diff) / np.std(diff, ddof=1))


def _bootstrap_ci(a, b, n_bootstrap=10000, alpha=0.05):
    """Bootstrap 95% CI on the mean difference."""
    rng = np.random.RandomState(42)
    diffs = np.array(a) - np.array(b)
    n = len(diffs)
    boot_means = np.array([
        np.mean(rng.choice(diffs, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return ci_lower, ci_upper


def _benjamini_hochberg(p_values):
    """Apply Benjamini-Hochberg FDR correction. Returns q-values."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    q_values = [0.0] * n
    min_q = 1.0
    for rank_idx in range(n - 1, -1, -1):
        orig_idx, p = indexed[rank_idx]
        rank = rank_idx + 1
        q = p * n / rank
        min_q = min(min_q, q)
        q_values[orig_idx] = min(min_q, 1.0)
    return q_values


def compute_significance(all_results):
    """Comprehensive significance testing: AFE-enhanced vs standalone baselines.

    For each (dataset, backend) pair, compares every AFE variant against the
    standalone baseline across all metrics using paired tests, bootstrap CIs,
    effect sizes, and FDR correction.
    """
    from scipy.stats import wilcoxon, ttest_rel

    # Group results by (dataset, method) -> list of per-seed metric dicts
    grouped = defaultdict(list)
    for r in all_results:
        if r.get("type") == "significance":
            continue
        grouped[(r["dataset"], r["method"])].append(r)

    # Identify all (dataset, backend, afe_method) comparison pairs
    comparisons = []
    for (ds, method), rows in grouped.items():
        if rows[0].get("afe_enhanced", False):
            backend = rows[0]["backend"]
            baseline_key = (ds, f"{backend}_3d")
            if baseline_key in grouped:
                comparisons.append((ds, baseline_key, (ds, method)))

    sig_results = []
    all_p_values = []  # For FDR across all comparisons per dataset

    # Per-dataset FDR groups
    dataset_p_indices = defaultdict(list)

    for ds, baseline_key, enhanced_key in comparisons:
        baseline_rows = grouped[baseline_key]
        enhanced_rows = grouped[enhanced_key]

        # Match seeds
        baseline_by_seed = {r["seed"]: r for r in baseline_rows}
        enhanced_by_seed = {r["seed"]: r for r in enhanced_rows}
        common_seeds = sorted(set(baseline_by_seed) & set(enhanced_by_seed))
        n_seeds = len(common_seeds)

        if n_seeds < 2:
            continue

        for metric in SIGNIFICANCE_METRICS:
            baseline_vals = []
            enhanced_vals = []
            for seed in common_seeds:
                bv = baseline_by_seed[seed].get(metric)
                ev = enhanced_by_seed[seed].get(metric)
                if bv is not None and ev is not None:
                    baseline_vals.append(bv)
                    enhanced_vals.append(ev)

            if len(baseline_vals) < 2:
                continue

            baseline_arr = np.array(baseline_vals)
            enhanced_arr = np.array(enhanced_vals)
            diff = enhanced_arr - baseline_arr

            # Skip if no variation
            if np.all(diff == 0):
                continue

            # Statistical test
            if n_seeds >= 6:
                try:
                    stat, p_value = wilcoxon(enhanced_arr, baseline_arr,
                                             alternative='two-sided')
                    test_used = "wilcoxon"
                except ValueError:
                    stat, p_value = ttest_rel(enhanced_arr, baseline_arr)
                    test_used = "paired_t"
            else:
                stat, p_value = ttest_rel(enhanced_arr, baseline_arr)
                test_used = "paired_t"

            p_value = float(p_value) if not np.isnan(p_value) else 1.0

            # Bootstrap CI
            ci_lower, ci_upper = _bootstrap_ci(enhanced_vals, baseline_vals)

            # Effect size
            cohens_d = _cohens_d(enhanced_vals, baseline_vals)

            idx = len(all_p_values)
            all_p_values.append(p_value)
            dataset_p_indices[ds].append(idx)

            sig_results.append({
                "type": "significance",
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "dataset": ds,
                "baseline": baseline_key[1],
                "enhanced": enhanced_key[1],
                "metric": metric,
                "baseline_mean": float(np.mean(baseline_arr)),
                "baseline_std": float(np.std(baseline_arr, ddof=1)) if len(baseline_arr) > 1 else 0.0,
                "enhanced_mean": float(np.mean(enhanced_arr)),
                "enhanced_std": float(np.std(enhanced_arr, ddof=1)) if len(enhanced_arr) > 1 else 0.0,
                "mean_diff": float(np.mean(diff)),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "cohens_d": cohens_d,
                "p_value": p_value,
                "test_used": test_used,
                "n_seeds": len(baseline_vals),
            })

    # Apply BH-FDR correction per dataset
    for ds, indices in dataset_p_indices.items():
        p_vals = [all_p_values[i] for i in indices]
        q_vals = _benjamini_hochberg(p_vals)
        for idx_in_group, global_idx in enumerate(indices):
            sig_results[global_idx]["q_value"] = q_vals[idx_in_group]
            sig_results[global_idx]["significant"] = q_vals[idx_in_group] < 0.05

    return sig_results


def aggregate_significance(sig_results):
    """Aggregate significance across datasets: win/tie/loss counts and Fisher's method."""
    from scipy.stats import combine_pvalues

    method_metric_results = defaultdict(list)
    for r in sig_results:
        key = (r["enhanced"], r["metric"])
        method_metric_results[key].append(r)

    aggregated = []
    for (method, metric), results in method_metric_results.items():
        p_values = [r["p_value"] for r in results]
        effect_sizes = [r["cohens_d"] for r in results]

        # Win/tie/loss at q < 0.05
        wins = sum(1 for r in results
                   if r.get("significant", False) and
                   (r["mean_diff"] > 0 if metric not in LOWER_IS_BETTER
                    else r["mean_diff"] < 0))
        losses = sum(1 for r in results
                     if r.get("significant", False) and
                     (r["mean_diff"] < 0 if metric not in LOWER_IS_BETTER
                      else r["mean_diff"] > 0))
        ties = len(results) - wins - losses

        # Fisher's method for combining p-values
        valid_p = [p for p in p_values if 0 < p < 1]
        if len(valid_p) >= 2:
            _, combined_p = combine_pvalues(valid_p, method='fisher')
        elif len(valid_p) == 1:
            combined_p = valid_p[0]
        else:
            combined_p = 1.0

        # Mean effect size with CI
        if len(effect_sizes) > 1:
            es_mean = float(np.mean(effect_sizes))
            es_ci_lower, es_ci_upper = _bootstrap_ci(
                effect_sizes, [0] * len(effect_sizes)
            )
        else:
            es_mean = effect_sizes[0] if effect_sizes else 0.0
            es_ci_lower = es_ci_upper = es_mean

        aggregated.append({
            "method": method,
            "metric": metric,
            "n_datasets": len(results),
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "combined_p": float(combined_p),
            "mean_effect_size": es_mean,
            "effect_size_ci_lower": es_ci_lower,
            "effect_size_ci_upper": es_ci_upper,
        })

    return aggregated
