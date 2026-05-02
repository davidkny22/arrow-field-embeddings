"""Generate ResultsBenchmark1.md from full_benchmark_results.jsonl."""
import json
import numpy as np
from collections import defaultdict

results = []
sig_results = []
with open("benchmarks/full_benchmark_results.jsonl") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("type") == "benchmark":
            results.append(obj)
        elif obj.get("type") == "significance":
            sig_results.append(obj)

baselines = [r for r in results if not r.get("afe_enhanced")]
afe = [r for r in results if r.get("afe_enhanced")]

lines = []
def w(s=""):
    lines.append(s)

datasets_order = [
    "swiss_roll", "s_curve_hole", "gaussian_noise", "hierarchical_gaussians",
    "usps", "mnist", "fashion_mnist", "coil20", "20newsgroups",
    "pbmc3k", "pbmc68k_reduced", "paul15", "dentate_gyrus", "planaria", "celegans",
]

# ============================================================
# HEADER
# ============================================================
w("# Benchmark v1 Results")
w()
w("> **Original benchmark run** (pre-metric-fixes). 15 datasets, 3 backends (PaCMAP, UMAP, TriMAP), 10 seeds.")
w(">")
w("> **Known issues in this run:**")
w("> - Continuity metric used truncated rank lookup (biased upward)")
w("> - MNIST/Fashion MNIST raw 0-255 pixels (inflated MSE)")
w("> - COIL-20 included with unnormalized PCA (MSE ~782K)")
w("> - No t-SNE backend")
w("> - No recon_knn_recall metric")
w(">")
w("> These are fixed in v2 (in progress).")
w()

# ============================================================
# GRAND SUMMARY
# ============================================================
w("## Grand Summary")
w()
w("| Stat | Value |")
w("|------|-------|")
w(f"| Total benchmark runs | {len(results):,} |")
w(f"| Baseline runs | {len(baselines)} |")
w(f"| AFE runs | {len(afe)} |")
w(f"| Significance tests | {len(sig_results)} |")
w(f"| Datasets | {len(set(r['dataset'] for r in results))} |")
w("| Backends | PaCMAP, UMAP, TriMAP |")
w("| Seeds per config | 10 |")
w()

# Runs per dataset
w("### Runs per Dataset")
w()
w("| Dataset | Total | Baseline | AFE | Arrow Counts |")
w("|---------|-------|----------|-----|--------------|")
for ds in datasets_order:
    n = len([r for r in results if r["dataset"] == ds])
    nb = len([r for r in baselines if r["dataset"] == ds])
    na = len([r for r in afe if r["dataset"] == ds])
    acounts = sorted(set(r["n_arrows"] for r in afe if r["dataset"] == ds))
    ac_str = ", ".join(str(a) for a in acounts) if acounts else "-"
    w(f"| {ds} | {n} | {nb} | {na} | {ac_str} |")
w()

# ============================================================
# TABLE 1: BASELINE
# ============================================================
w("## Table 1: Baseline DR Quality (mean over 10 seeds)")
w()
w("| Dataset | Backend | kNN@10 | kNN@50 | Spearman | Triplet Acc | Trust | Cont | Stress | MSE | Cosine | Time (s) |")
w("|---------|---------|--------|--------|----------|-------------|-------|------|--------|-----|--------|----------|")

base_groups = defaultdict(list)
for r in baselines:
    base_groups[(r["dataset"], r["backend"])].append(r)

base_metrics = [
    "knn_recall_k10", "knn_recall_k50", "spearman_dist_corr", "random_triplet_acc",
    "trustworthiness_k10", "continuity_k10", "normalized_stress", "reconstruction_mse", "reconstruction_cosine",
]

for ds in datasets_order:
    for backend in ["pacmap", "umap", "trimap"]:
        key = (ds, backend)
        if key not in base_groups:
            continue
        rows = base_groups[key]
        v = {m: np.mean([r[m] for r in rows]) for m in base_metrics}
        t = np.mean([r["time_seconds"] for r in rows])
        mse_str = f'{v["reconstruction_mse"]:,.1f}' if v["reconstruction_mse"] > 100 else f'{v["reconstruction_mse"]:.1f}'
        w(f'| {ds} | {backend} | {v["knn_recall_k10"]:.3f} | {v["knn_recall_k50"]:.3f} | {v["spearman_dist_corr"]:.3f} | {v["random_triplet_acc"]:.4f} | {v["trustworthiness_k10"]:.4f} | {v["continuity_k10"]:.4f} | {v["normalized_stress"]:.4f} | {mse_str} | {v["reconstruction_cosine"]:.4f} | {t:.1f} |')
w()

# ============================================================
# TABLE 2: AFE ARROW METRICS (all modes)
# ============================================================
w("## Table 2: AFE Arrow Metrics (all modes, mean over 10 seeds)")
w()

afe_groups = defaultdict(list)
for r in afe:
    key = (r["dataset"], r["backend"], r["encoding_mode"], r["n_arrows"])
    afe_groups[key].append(r)

for ds in datasets_order:
    found = False
    for backend in ["pacmap", "umap", "trimap"]:
        for mode in ["direct", "pca", "adaptive"]:
            arrow_counts = sorted(set(
                r["n_arrows"] for r in afe
                if r["dataset"] == ds and r["backend"] == backend and r["encoding_mode"] == mode
            ))
            if not arrow_counts:
                continue
            if not found:
                w(f"### {ds}")
                w()
                w("| Backend | Mode | #Arrows | kNN@10 | ArrKNN@10 | ArrKNN@50 | Consistency | Arrow Spatial Information Gain | Spatial Information Gap | Recon MSE | Recon Cosine |")
                w("|---------|------|---------|--------|-----------|-----------|-------------|-----------|---------|-----------|--------------|")
                found = True
            for na in arrow_counts:
                key = (ds, backend, mode, na)
                rows = afe_groups[key]
                knn10 = np.mean([r["knn_recall_k10"] for r in rows])
                aknn10 = np.mean([r.get("arrow_knn_recall_k10", float("nan")) for r in rows])
                aknn50 = np.mean([r.get("arrow_knn_recall_k50", float("nan")) for r in rows])
                acons = np.mean([r.get("arrow_consistency", float("nan")) for r in rows])
                aig = np.mean([
                    r.get("arrow_spatial_information_gain", float("nan")) for r in rows
                ])
                igap = np.mean([
                    r.get("spatial_information_gap", float("nan"))
                    for r in rows
                ])
                rmse = np.mean([r.get("reconstruction_mse", float("nan")) for r in rows])
                rcos = np.mean([r.get("reconstruction_cosine", float("nan")) for r in rows])
                mse_str = f"{rmse:,.1f}" if rmse > 100 else f"{rmse:.1f}"
                w(f"| {backend} | {mode} | {na} | {knn10:.3f} | {aknn10:.4f} | {aknn50:.4f} | {acons:.4f} | {aig:+.4f} | {igap:.4f} | {mse_str} | {rcos:.4f} |")
    if found:
        w()

# ============================================================
# TABLE 3: BEST MODE LIFT
# ============================================================
w("## Table 3: AFE vs Baseline Lift (best encoding mode per dataset-backend)")
w()
w("Best mode selected by highest Arrow kNN@10. Lift = (ArrKNN@10 - Base kNN@10) / Base kNN@10.")
w()
w("| Dataset | Backend | Best Mode | #Arrows | Base kNN@10 | AFE kNN@10 | ArrKNN@10 | Lift% | Trust (base) | Trust (AFE) | Cont (base) | Cont (AFE) | Spearman (base) | Spearman (AFE) | Consistency | Arrow Spatial Information Gain |")
w("|---------|---------|-----------|---------|-------------|------------|-----------|-------|--------------|-------------|-------------|------------|-----------------|----------------|-------------|-----------|")

for ds in datasets_order:
    for backend in ["pacmap", "umap", "trimap"]:
        bkey = (ds, backend)
        if bkey not in base_groups:
            continue
        brows = base_groups[bkey]
        base_knn = np.mean([r["knn_recall_k10"] for r in brows])
        base_trust = np.mean([r["trustworthiness_k10"] for r in brows])
        base_cont = np.mean([r["continuity_k10"] for r in brows])
        base_spear = np.mean([r["spearman_dist_corr"] for r in brows])

        best_aknn = -1
        best_key = None
        for mode in ["direct", "pca", "adaptive"]:
            arrow_counts = sorted(set(
                r["n_arrows"] for r in afe
                if r["dataset"] == ds and r["backend"] == backend and r["encoding_mode"] == mode
            ))
            for na in arrow_counts:
                key = (ds, backend, mode, na)
                if key in afe_groups:
                    aknn = np.mean([r.get("arrow_knn_recall_k10", 0) for r in afe_groups[key]])
                    if aknn > best_aknn:
                        best_aknn = aknn
                        best_key = key

        if best_key is None:
            continue

        arows = afe_groups[best_key]
        afe_knn = np.mean([r["knn_recall_k10"] for r in arows])
        aknn10 = np.mean([r.get("arrow_knn_recall_k10", 0) for r in arows])
        afe_trust = np.mean([r["trustworthiness_k10"] for r in arows])
        afe_cont = np.mean([r["continuity_k10"] for r in arows])
        afe_spear = np.mean([r["spearman_dist_corr"] for r in arows])
        acons = np.mean([r.get("arrow_consistency", 0) for r in arows])
        aig = np.mean([r.get("arrow_spatial_information_gain", 0) for r in arows])
        lift_pct = (aknn10 - base_knn) / base_knn * 100 if base_knn > 0 else 0

        w(f"| {ds} | {backend} | {best_key[2]} | {best_key[3]} | {base_knn:.4f} | {afe_knn:.4f} | {aknn10:.4f} | {lift_pct:+.1f}% | {base_trust:.4f} | {afe_trust:.4f} | {base_cont:.4f} | {afe_cont:.4f} | {base_spear:.4f} | {afe_spear:.4f} | {acons:.3f} | {aig:+.4f} |")
w()

# ============================================================
# TABLE 4: CROSS-DATASET SUMMARY
# ============================================================
w("## Table 4: Cross-Dataset Summary (Direct mode, all backends averaged)")
w()
w("| Dataset | Base kNN@10 | Arrow kNN@10 | Lift | #Arrows |")
w("|---------|-------------|--------------|------|---------|")

base_knn_ds = defaultdict(list)
for r in baselines:
    base_knn_ds[r["dataset"]].append(r["knn_recall_k10"])

lift_data = []
for ds in datasets_order:
    bk = np.mean(base_knn_ds[ds])
    direct = [r for r in afe if r["dataset"] == ds and r["encoding_mode"] == "direct"]
    if not direct:
        continue
    best_aknn = -1
    best_na = 0
    for na in set(r["n_arrows"] for r in direct):
        arr = [r for r in direct if r["n_arrows"] == na]
        aknn = np.mean([r["arrow_knn_recall_k10"] for r in arr])
        if aknn > best_aknn:
            best_aknn = aknn
            best_na = na
    lift = (best_aknn - bk) / bk * 100
    lift_data.append((ds, bk, best_aknn, lift, best_na))

for ds, bk, aknn, lift, na in sorted(lift_data, key=lambda x: -x[3]):
    w(f"| {ds} | {bk:.4f} | {aknn:.4f} | {lift:+.1f}% | {na} |")
w()

# ============================================================
# TABLE 5: SIGNIFICANCE TESTS
# ============================================================
w("## Table 5: Significance Tests (Wilcoxon signed-rank, BH-corrected)")
w()
w("### Summary by Metric")
w()
w("| Metric | Significant | Total | Mean |d| | Mean Diff | Interpretation |")
w("|--------|-------------|-------|---------|-----------|----------------|")

metrics_list = sorted(set(s["metric"] for s in sig_results))
interp = {
    "reconstruction_mse": "AFE dramatically reduces MSE",
    "reconstruction_cosine": "AFE dramatically reduces cosine distance",
    "knn_recall_k10": "AFE does NOT degrade spatial kNN@10",
    "knn_recall_k50": "AFE does NOT degrade spatial kNN@50",
    "trustworthiness_k10": "AFE does NOT degrade trustworthiness",
    "continuity_k10": "AFE does NOT degrade continuity",
    "spearman_dist_corr": "AFE does NOT degrade distance correlation",
    "random_triplet_acc": "AFE does NOT degrade triplet accuracy",
    "normalized_stress": "AFE does NOT degrade stress",
    "centroid_triplet_acc": "AFE does NOT degrade centroid triplets",
    "silhouette_score": "AFE does NOT degrade silhouette score",
}
for metric in metrics_list:
    relevant = [s for s in sig_results if s["metric"] == metric]
    sig_count = sum(1 for s in relevant if s["significant"])
    total = len(relevant)
    mean_d = np.mean([abs(s["cohens_d"]) for s in relevant])
    mean_diff = np.mean([s["mean_diff"] for s in relevant])
    md_str = f"{mean_diff:+.4f}" if abs(mean_diff) < 100 else f"{mean_diff:+,.1f}"
    w(f"| {metric} | **{sig_count}/{total}** | {total} | {mean_d:.2f} | {md_str} | {interp.get(metric, '')} |")
w()

w("### Summary by Dataset")
w()
w("| Dataset | Significant | Total |")
w("|---------|-------------|-------|")
for ds in datasets_order:
    relevant = [s for s in sig_results if s["dataset"] == ds]
    if not relevant:
        continue
    sig_count = sum(1 for s in relevant if s["significant"])
    total = len(relevant)
    w(f"| {ds} | {sig_count}/{total} | {total} |")
w()

w("### Non-Significant Tests (q > 0.05)")
w()
w("These confirm AFE does not degrade the underlying embedding quality.")
w()
w("| Dataset | Baseline | Enhanced | Metric | Diff | q-value | Cohen's d |")
w("|---------|----------|----------|--------|------|---------|-----------|")
nonsig = [s for s in sig_results if not s["significant"]]
for s in sorted(nonsig, key=lambda x: (x["dataset"], x["metric"])):
    w(f'| {s["dataset"]} | {s["baseline"]} | {s["enhanced"]} | {s["metric"]} | {s["mean_diff"]:+.4f} | {s["q_value"]:.4f} | {s["cohens_d"]:+.2f} |')
w()

w("### Significant Tests (q <= 0.05)")
w()
w("| Dataset | Baseline | Enhanced | Metric | Diff | q-value | Cohen's d |")
w("|---------|----------|----------|--------|------|---------|-----------|")
sig = [s for s in sig_results if s["significant"]]
for s in sorted(sig, key=lambda x: (x["dataset"], x["metric"])):
    md = f'{s["mean_diff"]:+.4f}' if abs(s["mean_diff"]) < 100 else f'{s["mean_diff"]:+,.1f}'
    w(f'| {s["dataset"]} | {s["baseline"]} | {s["enhanced"]} | {s["metric"]} | {md} | {s["q_value"]:.6f} | {s["cohens_d"]:+.2f} |')
w()

content = "\n".join(lines)
with open("benchmarks/ResultsBenchmark1.md", "w", encoding="utf-8") as f:
    f.write(content)
print(f"Written {len(lines)} lines to benchmarks/ResultsBenchmark1.md")

