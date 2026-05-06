# Arrow Field Embeddings (AFE)

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-139%20passing-brightgreen.svg)]()

**Additive residual fields for information-preserving 3D embeddings.**

Standard 3D dimensionality reduction discards recoverable high-dimensional structure. AFE augments any existing 3D embedding with per-point arrow fields without moving a single point. Each arrow captures 3 additional dimensions via its geometric properties (azimuth, elevation, magnitude), so a point with k arrows preserves 3 + 3k dimensions total, with no theoretical ceiling. The k-th arrow represents the same original dimensions on every point, enabling direct cross-point comparison of high-dimensional structure that the spatial layout alone cannot express. The idea connects to fiber bundles and tangent spaces in differential geometry, but it arrived through intuition first.

*This project started with a question. When you project high-dimensional data to 3D, some structure is preserved, but most of the original dimensions are gone. Two points can end up next to each other despite diverging across dozens of original dimensions that the projection collapsed. But a dimension is an axis, a direction along which points have values. And an arrow is just a direction with a magnitude. When you have to remove an axis from the coordinate system, an arrow can reintroduce it at the point, without expanding the space. What if you gave every point arrows for the dimensions the projection dropped?*

## Quick Start

```python
from afe import ArrowFieldEmbedding
from afe.visualization import plot_afe

# Fit AFE on your high-dimensional data
afe = ArrowFieldEmbedding(
    n_arrows=3,
    encoding_mode="direct",  # or "pca", "adaptive"
    backend="tsne",  # or "pacmap", "umap", "trimap" (requires extras)
    random_state=42,
)
result = afe.fit_transform(X)

# Standalone interactive viewer (zero dependencies)
from afe.viewer import save_viewer
save_viewer(result["spatial"], result["arrows"], labels=y, path="embedding.html")

# Or export for the full React/Three.js application
from afe import export_for_viewer
export_for_viewer(result, labels=y, path="embedding.json.gz")
```

CLI benchmark:
```bash
python benchmarks/compare_methods.py --datasets swiss_roll --backends tsne --n-seeds 1 --modes direct --n-arrows 1 --skip-significance
```

## Module Reference

Every module can be used standalone. Below are practical examples for each domain.

### Core: Arrow Field Embedding

```python
from afe import ArrowFieldEmbedding

afe = ArrowFieldEmbedding(
    n_arrows=5,
    encoding_mode="direct",
    backend="tsne",
    random_state=42,
)
result = afe.fit_transform(X)

# Access components
spatial = result["spatial"]       # (n, 3) -- preserved exactly from backend
arrows = result["arrows"]         # (n, n_arrows, 3) -- azimuth, elevation, magnitude
metadata = result["metadata"]     # gap report, capacity, attributions

# Reconstruct approximate HD vectors
X_recon = afe.reconstruct()
```

### Spatial Information Gap Analysis

```python
from afe.gap_analysis import SpatialInformationGapAnalyzer

analyzer = SpatialInformationGapAnalyzer(correlation_threshold=0.3)
report = analyzer.analyze(X_high=X, X_3d=spatial, n_arrows=5)

print(f"Gap: {report['spatial_information_gap']:.3f}")
print(f"Captured dims: {len(report['captured_dims'])}")
print(f"Residual dims: {len(report['residual_dims'])}")
```

### Evaluation Metrics

```python
from afe.evaluation import (
    knn_recall,
    recon_knn_recall,
    reconstruction_error,
    spatial_information_gap,
    arrow_knn_recall,
    arrow_consistency,
    knn_classification_metrics,
)

# Standard DR metric (spatial only)
knn = knn_recall(X, spatial, k=10)

# AFE metric: spatial + arrows combined
aknn = arrow_knn_recall(X, spatial, arrows, k=10)

# Reconstruction quality
X_recon = afe.reconstruct()
rknn = recon_knn_recall(X, X_recon, k=10)
mse = reconstruction_error(X, X_recon)

# Downstream classification on reconstructed data
metrics = knn_classification_metrics(X_recon, labels)
print(f"Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['macro_f1']:.3f}")
```

### Encoding Modes

```python
from afe.encoding import get_encoder

# Direct: 1-to-1 residual dimension mapping (lossless up to capacity)
encoder = get_encoder("direct")

# PCA: each arrow = one principal component of residuals
encoder = get_encoder("pca")

# Adaptive: eigenvalue gap detection + hierarchical grouping
encoder = get_encoder("adaptive", eigenvalue_gap_threshold=1.5)
```

### Attribution

```python
from afe.attribution import get_arrow_attributions

# Map each arrow back to original dimensions
attrs = afe.get_arrow_attributions(feature_names=feature_names, top_n=8)
for attr in attrs:
    print(f"Arrow {attr['arrow_index']}: {attr['label']}")
```

### Visualization

```python
from afe.visualization import plot_afe, plot_info_gap

# 3D scatter with arrow cones
fig = plot_afe(spatial, arrows, labels=y, title="AFE Visualization")
fig.show()

# Spatial information gap bar chart
fig = plot_info_gap(report)
fig.show()
```

### Viewers

AFE provides two viewer options with different trade-offs. They are not interchangeable.

**Standalone HTML viewer** (`save_viewer`): zero dependencies, single file.
- Quick inspection, email attachments, notebook embedding, sharing with collaborators
- Generated entirely in Python; no build step, no server, no Node.js
- Features: WASD fly-through, labeled point cloud, toggleable arrows, hover tooltips, touch support
- Limitations: no search, no reconstruction panel, no metrics bar, no dataset switching

```python
from afe.viewer import save_viewer

save_viewer(
    result["spatial"],
    result["arrows"],
    labels=y,
    path="embedding.html",
    arrow_scale=1.0,
)
```

**React/Three.js viewer** (`export_for_viewer` + `viewer/` app): full application.
- Deep exploration, presentations, interactive analysis
- Requires building the React app (`npm run build`) or running the FastAPI backend server
- Features: search, reconstruction metrics, GPU picking, camera bookmarks, dataset presets, arrow filtering by magnitude, rectangle selection, shareable URLs
- Limitations: requires Node.js + npm for building; data must be exported as JSON.gz first

```python
from afe import export_for_viewer

export_for_viewer(result, labels=y, path="embedding.json.gz")
```

Then serve the viewer app and open the exported dataset:
```bash
cd viewer
npm install
npm run build
python -m http.server 8080 --directory dist
# Open http://localhost:8080 and load embedding.json.gz
```

## How It Works

AFE is a post-hoc additive layer, not a replacement for dimensionality reduction. It wraps any existing 3D embedding method and adds directional annotations without modifying spatial coordinates.

The pipeline has four stages:

| Stage | What it does |
|-------|-------------|
| **Spatial Layout** | Compute 3D embedding via backend (t-SNE, UMAP, PaCMAP, TriMAP, or manual array) |
| **Gap Analysis** | Compute Pearson correlation between each original dimension and the 3 spatial axes. Classify dimensions as captured (max abs corr >= threshold) or residual. Compute spatial information gap = 1 - mean(max abs correlation) |
| **Arrow Encoding** | Map residual dimensions onto arrows using direct, PCA, or adaptive mode |
| **Reconstruction** | Recover approximate HD vectors: captured dims via linear regression from spatial coordinates, residual dims via encoder decoding from arrows. No double-counting |

**Spatial invariance guarantee:** When wrapping an existing embedding via `backend=np.ndarray`, spatial coordinates are preserved exactly. All improvements come from the arrow field, not from moving points.

**Arrow capacity:** Each arrow carries 3 channels (azimuth, elevation, magnitude). Total residual capacity = 3 x n_arrows dimensions. The default arrow count is ceil((d - 3) / 3), providing enough capacity to cover all residual dimensions.

**Encoding modes:**

| Mode | How It Works |
|------|--------------|
| **direct** | Raw residual values assigned to arrow channels. Dimension i*3+c maps to arrow i, channel c. Lossless when capacity covers residuals |
| **pca** | PCA on residuals. PC loadings encoded as angular directions (azimuth, elevation), PC scores as magnitude. Constant direction per arrow, varying magnitude per point |
| **adaptive** | Eigenvalue gap detection determines group count. Hierarchical clustering assigns dimensions to groups. Correlation validation splits weak groups. Per-group PCA encodes each group onto one arrow |

## Installation

AFE targets Python 3.11+ and pins `numpy<2.4` because optional Numba-backed embedding backends are not yet stable under NumPy 2.4.

```bash
git clone https://github.com/davidkny22/arrow-field-embeddings.git
cd arrow-field-embeddings
pip install -e ".[pacmap]"
```

For development:
```bash
pip install -e ".[dev,pacmap]"
```

For full benchmark runs (includes scRNA-seq loaders, text embeddings, etc.):
```bash
pip install -e ".[benchmark,backends-all]"
```

The DREAMS backend is experimental and requires the compatible berenslab openTSNE fork. Standard `openTSNE` is not sufficient.

**Windows note:** Set `OPENBLAS_NUM_THREADS=1` before running benchmarks. OpenBLAS has a known threading deadlock inside `multiprocessing` on Windows ([scipy#20294](https://github.com/scipy/scipy/issues/20294)). Parallel benchmark execution (`--n-jobs > 1`) is Linux-only.

## Project Structure

```
source/afe/
  core.py              ArrowFieldEmbedding sklearn-compatible estimator
  backends.py          Spatial backends (PaCMAP, UMAP, t-SNE, TriMAP, DREAMS, Manual)
  encoding.py          Three encoders: direct, pca, adaptive
  evaluation.py        Metrics (KNN recall, ReconKNN, stress, trust, classification, etc.)
  gap_analysis.py      SpatialInformationGapAnalyzer
  reconstruction.py    HD vector reconstruction from spatial + arrows
  attribution.py       Arrow-to-dimension attribution mapping
  normalization.py     Per-arrow spherical min-max normalization
  reproducibility.py   Schema versioning, spatial cache, machine info
  information_theory.py  MI estimation (Gaussian proxy, kNN/KSG)
  biology.py           scRNA-seq arrow-to-gene mapping
  viewer.py            Standalone HTML viewer generator (zero-dependency single file)
  visualization.py     Plotly 3D scatter + cone arrows
  export.py            JSON.gz export for the React/Three.js application viewer

benchmarks/
  compare_methods.py   Benchmark runner: (dataset, backend) scheduling, JSONL output, significance
  datasets.py          34 dataset loaders (synthetic, image, text, scRNA-seq)
  config.py            Dataset categories and default configurations
  metrics.py           Metric computation for benchmark records
  io.py                JSONL I/O, resumability, spatial embedding cache
  significance.py      Wilcoxon signed-rank tests with BH correction
  reporting.py         Results manuscript generation

docs/
  results/
    experiment-log.md  Experiment log with results from two benchmark rounds
    benchmark-run-1.md Full per-seed results from initial benchmark
    benchmark-run-2.md Full per-seed results from expanded benchmark
    gen_results.py     Script to regenerate result tables from JSONL

paper/
  gen_figures.py       Publication figure generation from benchmark results

test/
  test_*.py            139 tests covering all modules, integration, benchmarks, visualization

viewer/                React/Three.js web app + FastAPI backend
  src/                 Frontend components (WASD controls, orbit, info panels, bookmarks)
  server/              FastAPI server with live AFE computation engine
```

## Running Benchmarks

```bash
# Quick smoke test
python benchmarks/compare_methods.py --datasets swiss_roll --backends tsne --n-seeds 1 --modes direct --n-arrows 1 --skip-significance

# Full benchmark (sequential)
python benchmarks/compare_methods.py --category all --n-seeds 10

# Precache datasets before parallel execution (prevents OOM)
python benchmarks/compare_methods.py --category all --precache

# Full benchmark (parallel, Linux only)
python benchmarks/compare_methods.py --category all --n-seeds 10 --n-jobs -1

# Custom backend parameters
python benchmarks/compare_methods.py --datasets mnist --backends umap --backend-params '{"n_neighbors": 30}'
```

The benchmark runner:
- Schedules work at (dataset, backend) granularity for parallel execution
- Computes each spatial embedding once per dataset/backend/seed
- Reuses fixed coordinates for baseline and all AFE variants
- Writes JSONL records with full metric coverage
- Supports resumability via `--output` JSONL files
- Includes paired Wilcoxon signed-rank tests with Benjamini-Hochberg FDR correction

## Testing

```bash
pytest test/ -v
```

139 tests covering:
- Core estimator initialization, fitting, transformation, reconstruction
- All three encoding modes (direct, PCA, adaptive) with roundtrip validation
- All spatial backends (Manual, t-SNE, PaCMAP, UMAP, TriMAP, DREAMS)
- Evaluation metrics: KNN recall, spatial information gap, reconstruction, classification
- Information theory: MI estimation, rate-distortion
- Spatial cache reuse and reproducibility
- Benchmark runner safeguards and metric coverage
- Plotly visualization figure generation
- Arrow attribution mapping for all encoding modes

5 tests skip cleanly when optional backends (PaCMAP, UMAP, TriMAP) are not installed.

## Empirical Evaluation

Two benchmark rounds have been completed across 15 and 17 datasets, 3 and 4 DR backends, 3 encoding modes, and 10 seeds per configuration. A third evaluation with shared-coordinate protocol and expanded metric coverage is in progress.

Results from completed rounds are documented in the [experiment log](docs/results/experiment-log.md), with full per-seed data in [benchmark-run-1.md](docs/results/benchmark-run-1.md) and [benchmark-run-2.md](docs/results/benchmark-run-2.md).

## Paper

The experiment log and results manuscript are in `docs/results/`. Publication figures are generated by `paper/gen_figures.py` (pending regeneration from the current benchmark round).

## Citation

```bibtex
@software{kogan2026afe,
  author = {Kogan, David},
  title = {{Arrow Field Embeddings}: Additive Residual Fields for Information-Preserving 3D Embeddings},
  year = {2026},
  url = {https://github.com/davidkny22/arrow-field-embeddings}
}
```

## License

[AGPL-3.0-or-later](LICENSE). Derivative works and network services must release source under the same license.
