# Arrow Field Embeddings — Agent Guide

## Project Overview

Arrow Field Embeddings (AFE) is a Python library that augments standard 3D dimensionality reduction with **arrow fields** — per-point directional annotations that encode residual high-dimensional information without moving points in space.

**Primary contribution:** A new additive residual-field representation for 3D embeddings. AFE is not a DR optimizer; it wraps existing methods (t-SNE, UMAP, PaCMAP, TriMAP) and preserves their spatial layouts exactly.

**Current state:** Post-implementation, pre-submission. The core library, benchmarks (2,720 runs), tests (100 passing), figures (8 generated), and results manuscript are complete. Remaining work centers on paper writing (LaTeX), theory formalization, and any requested ablations.

## Environment

- **Python:** 3.11+ (currently runs on 3.12 as well)
- **NumPy:** `>=1.23, <2.4` (pinned because optional Numba backends fail under NumPy 2.4)
- **Core deps:** scikit-learn, scipy, plotly
- **Optional backends:** pacmap, umap-learn, trimap
- **Benchmark deps:** matplotlib, pandas, scanpy, scvelo, datasets, sentence-transformers, plus Bioconductor bridge packages

### Install

```bash
pip install -e ".[dev,pacmap]"        # dev + one backend
pip install -e ".[benchmark,backends-all]"  # full benchmark stack
```

## Testing

```bash
pytest test/ -v
```

- 100 tests pass, 5 skip cleanly when optional backends are missing
- All tests must pass before claiming work is complete

## Running Benchmarks

```bash
# Quick smoke test
python benchmarks/compare_methods.py --datasets swiss_roll --backends tsne --n-seeds 1 --modes direct --n-arrows 1 --skip-significance

# Full paper benchmark (sequential — slow)
python benchmarks/compare_methods.py --category all --n-seeds 10

# Full paper benchmark (parallel — recommended on Linux/Kaggle)
python benchmarks/compare_methods.py --category all --n-seeds 10 --n-jobs -1
```

The benchmark runner:
- Computes each spatial embedding **once** per dataset/backend/seed
- Reuses fixed coordinates for baseline and all AFE variants
- Writes `afe-benchmark-v3` JSONL records (the schema string is internal; do not use "v3" in public-facing text)
- Supports resumability via `--output` JSONL files
- Supports parallel dataset execution via `--n-jobs N` (one dataset per worker process)

### Platform-specific execution notes

**Windows:** Parallel execution (`--n-jobs > 1`) **does not work** on Windows due to an OpenBLAS threading deadlock when `numpy`/`scipy` are used inside `multiprocessing` ([scipy#20294](https://github.com/scipy/scipy/issues/20294)). The runner sets `OPENBLAS_NUM_THREADS=1` as a workaround, but this only prevents the deadlock for sequential execution. On Windows, run sequential (`--n-jobs 1`).

**Linux / Kaggle / Colab:** Parallel execution works cleanly. `--n-jobs -1` uses all CPU cores (4 on Kaggle, 2 on Colab free). This is the fastest way to run the full benchmark (~4–6 hours on Kaggle vs. ~24+ hours sequential). The benchmark is CPU-bound; GPU accelerators provide no speedup.

**Recommended platform:** Kaggle (4-core Xeon, ~30GB RAM). Select **None** for the GPU accelerator.

## Architecture

```
source/afe/
  core.py            ArrowFieldEmbedding sklearn estimator
  backends.py        Spatial backends (PaCMAP, UMAP, t-SNE, TriMAP, DREAMS, Manual)
  encoding.py        Three encoders: direct, pca, adaptive
  evaluation.py      Paper-facing metrics (KNN recall, stress, trust, classification, etc.)
  gap_analysis.py    SpatialInformationGapAnalyzer — what does the 3D layout miss?
  reconstruction.py  HD vector reconstruction from spatial + arrows
  attribution.py     Arrow-to-dimension attribution mapping
  reproducibility.py Schema versioning, spatial cache, machine info
  viewer.py          Self-contained HTML viewer generator
  visualization.py   Plotly 3D scatter + cone arrows
```

## Key Conventions and Warnings

### "v3" is internal language

The benchmark schema identifier `afe-benchmark-v3` is fine as a machine-readable constant. **Never** use "v3" in public-facing text (README, paper, documentation). Use "paper benchmark" or "current benchmark" instead.

### Spatial information gap, not correlation gap

The metric `spatial_information_gap` (defined as `1 - mean_j max_l |corr(X_j, Y_l)|`) is the correct name. An earlier plan referred to it as "correlation gap score"; that framing was updated. Keep the current name.

### Spatial invariance is structural, not statistical

When AFE wraps an existing embedding via `backend=np.ndarray`, the spatial coordinates are preserved **exactly** (byte-for-byte or array-equal). Do not test this with stochastic significance claims — test with exact equality checks.

### Downstream classification

For labeled datasets, the benchmark runner now computes kNN classification accuracy and macro-F1 on:
- Original HD (`knn_class_acc_hd`, `knn_class_f1_hd`)
- 3D spatial (`knn_class_acc_spatial`, `knn_class_f1_spatial`)
- Flattened AFE (`knn_class_acc_flat`, `knn_class_f1_flat`)
- Reconstructed HD (`knn_class_acc_recon`, `knn_class_f1_recon`)

### Encoding modes

- **direct:** Raw residual components → arrow channels. Lossless when capacity covers residuals.
- **pca:** PCA on residuals; each arrow = one principal component.
- **adaptive:** Eigenvalue gap detection + hierarchical grouping. Report failures honestly.

### Biology validation status

Current scRNA preprocessing produces PCA benchmark matrices (50D). Gene-level attribution requires preserving HVG names, PCA loadings, and marker metadata. Do **not** claim gene-level interpretation from current cached data unless those mappings are explicitly preserved.

## What is Already Done (Strong)

- Complete pip-installable library with sklearn-compatible API
- Massive empirical evaluation: 2,720 runs, 12 datasets, 4 backends, 3 encodings, 10 seeds
- Full statistical testing: Wilcoxon signed-rank + Benjamini-Hochberg FDR correction
- Interactive viewer: React/Three.js frontend + FastAPI backend
- 8 publication figures generated (PDF + PNG at 300 DPI)
- Results manuscript in Markdown with exhaustive tables
- Backend agnosticism proven across t-SNE, UMAP, PaCMAP, TriMAP
- Zero spatial degradation verified via shared-coordinate protocol
- Attribution helpers for direct/PCA/adaptive modes
- Provenance: GPG-signed commits with OpenTimestamps

## What Remains (Gaps)

- **LaTeX paper source** — only Markdown drafts exist
- **Theory section** — MI/RD framing, complexity analysis, rate-distortion connections (exploratory unless robustness checks pass)
- **Systematic arrow count ablations** — some data exists but not a clean sweep across all datasets
- **Biological validation** — gene/program mapping requires updated preprocessing
- **Limitations section** — planned but not drafted
- **User study / task evaluation** — not required for ML-methods paper but needed for VIS adaptation

## Figure Inventory

| Figure | Status | File |
|--------|--------|------|
| Method schematic | Conceptual | Not generated |
| Hero figure (with/without arrows) | Needs viewer export | Not generated |
| ReconKNN@10 main result | Generated | `fig4_recon_knn_bar` |
| Spatial preservation proof | Generated | `fig5_spatial_preservation` |
| Encoding comparison | Generated | `fig6_encoding_comparison` |
| MSE reduction | Generated | `fig8_mse_reduction` |
| Backend heatmap | Generated | `fig10_backend_heatmap` |
| Arrow information gain | Generated | `fig_info_gain` |
| Runtime overhead | Generated | `fig_runtime_overhead` |
| Arrow count ablation | Generated | `fig_ablation_arrow_count` |

All figures are produced by `paper/gen_figures.py` from `benchmarks/full_benchmark_results_v2.jsonl`.

## Venue Positioning

- **NeurIPS/ICML/TMLR** (current target): Lead with method, theory, broad benchmarks, ablations, reproducibility.
- **IEEE VIS/EuroVis**: Lead with interactive visual augmentation, glyph design, analyst workflows. Reuse benchmark as technical support.
- **Bioinformatics/Nature Methods**: Lead only after gene/program attribution is solid. Focus on scRNA datasets, marker recovery, cell-type interpretation.

## Important Rules

1. **Make minimal changes.** The codebase is mature; prefer surgical edits.
2. **Tests must pass.** Run `pytest test/ -v` before claiming completion.
3. **Do not overclaim.** "AFE recovers high-dimensional neighborhoods lost in 3D projection" is strong but honest. Do not say "recovers all information" or "DR destroys X%" without MI/RD support.
4. **Preserve spatial invariance.** Never modify spatial coordinates in the AFE pipeline.
5. **Optional backends must skip gracefully.** Use `pytest.importorskip` in tests; catch `ImportError` in benchmarks.
