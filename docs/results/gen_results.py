#!/usr/bin/env python3
"""Generate readable benchmark result files from JSONL data.

Outputs per-dataset, per-backend tables with one row per method,
values averaged across seeds (mean +/- std). All metrics included.
"""

import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path):
    bench, sig = [], []
    for line in open(path):
        r = json.loads(line)
        if r.get("type") == "benchmark":
            bench.append(r)
        elif r.get("type") == "significance":
            sig.append(r)
    return bench, sig


def mean(vals):
    return sum(vals) / len(vals) if vals else None


def std(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def fmt_pm(vals):
    if not vals:
        return ""
    m = mean(vals)
    s = std(vals)
    return f"{m:.4f} +/- {s:.4f}"


METHOD_ORDER = {"baseline": 0, "direct": 1, "pca": 2, "adaptive": 3}


def generate(jsonl_path, output_path, has_recon_knn):
    bench, sig = load_jsonl(jsonl_path)

    metric_defs = [
        ("knn_k10", "knn_recall_k10"),
        ("knn_k50", "knn_recall_k50"),
        ("spearman", "spearman_dist_corr"),
        ("trust", "trustworthiness_k10"),
        ("contin", "continuity_k10"),
        ("stress", "normalized_stress"),
    ]
    if has_recon_knn:
        metric_defs.append(("recon_knn_k10", "recon_knn_recall_k10"))
    metric_defs.extend([
        ("recon_mse", "reconstruction_mse"),
        ("recon_cos", "reconstruction_cosine"),
        ("arrow_knn_k10", "arrow_knn_recall_k10"),
        ("arrow_knn_k50", "arrow_knn_recall_k50"),
        ("arrow_consist", "arrow_consistency"),
        ("arrow_info_gain", "arrow_info_gain"),
        ("spatial_info_gap", "info_gap_score"),
        ("time_s", "time_seconds"),
    ])

    headers = [m[0] for m in metric_defs]
    keys = [m[1] for m in metric_defs]

    # Group: (dataset, backend, method_label) -> list of records (one per seed)
    groups = defaultdict(list)
    dataset_meta = {}

    for r in bench:
        ds = r.get("dataset", "?")
        be = r.get("backend") or "none"
        mode = r.get("encoding_mode")
        afe = r.get("afe_enhanced", False)
        label = mode if afe and mode else "baseline"

        groups[(ds, be, label)].append(r)

        if ds not in dataset_meta:
            dataset_meta[ds] = {
                "n_samples": r.get("n_samples", "?"),
                "n_features": r.get("n_features", "?"),
                "n_arrows": r.get("n_arrows", 0),
            }
        if afe and r.get("n_arrows", 0) > dataset_meta[ds].get("n_arrows", 0):
            dataset_meta[ds]["n_arrows"] = r.get("n_arrows", 0)

    datasets = sorted(set(k[0] for k in groups))
    all_backends = sorted(set(k[1] for k in groups if k[1] != "none"))
    methods = ["baseline", "direct", "pca", "adaptive"]

    with open(output_path, "w", encoding="utf-8") as f:
        fname = Path(jsonl_path).name
        f.write(f"# Benchmark Results: {fname}\n\n")
        f.write(f"**{len(bench)}** benchmark records, **{len(sig)}** significance tests\n")
        f.write(f"**{len(datasets)}** datasets, **{len(all_backends)}** backends ({', '.join(all_backends)})\n")
        f.write(f"**10** seeds (0-9), values below are mean +/- std across seeds\n\n")
        f.write("---\n\n")

        for ds in datasets:
            meta = dataset_meta.get(ds, {})
            f.write(f"## {ds} ({meta.get('n_samples','?')} samples, "
                    f"{meta.get('n_features','?')}D, {meta.get('n_arrows','?')} arrows)\n\n")

            ds_backends = [be for be in all_backends if any(
                k[0] == ds and k[1] == be for k in groups
            )]

            for be in ds_backends:
                f.write(f"### {be.upper()}\n\n")
                f.write(f"| method | {' | '.join(headers)} |\n")
                f.write(f"|--------|{'|'.join(['------'] * len(headers))}|\n")

                for method in methods:
                    records = groups.get((ds, be, method), [])
                    if not records:
                        continue
                    cells = []
                    for key in keys:
                        vals = [r.get(key) for r in records if r.get(key) is not None]
                        cells.append(fmt_pm(vals))
                    f.write(f"| {method} | {' | '.join(cells)} |\n")

                f.write("\n")
            f.write("---\n\n")

        # Significance tests
        if sig:
            f.write("## Significance Tests\n\n")
            scols = ["dataset", "baseline", "enhanced", "metric",
                     "mean_diff", "p_value", "q_value", "cohens_d",
                     "significant", "test_used"]
            f.write(f"| {' | '.join(scols)} |\n")
            f.write(f"|{'|'.join(['---'] * len(scols))}|\n")
            for r in sorted(sig, key=lambda x: (
                x.get("dataset", ""), x.get("metric", ""), x.get("enhanced", "")
            )):
                vals = []
                for c in scols:
                    v = r.get(c)
                    if v is None:
                        vals.append("")
                    elif isinstance(v, float):
                        vals.append(f"{v:.2e}" if abs(v) < 0.001 and v != 0 else f"{v:.4f}")
                    else:
                        vals.append(str(v))
                f.write(f"| {' | '.join(vals)} |\n")

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    generate(
        "benchmarks/full_benchmark_results.jsonl",
        "docs/results/benchmark-run-1.md",
        has_recon_knn=False,
    )
    generate(
        "benchmarks/full_benchmark_results_v2.jsonl",
        "docs/results/benchmark-run-2.md",
        has_recon_knn=True,
    )
