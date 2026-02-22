# Arrow Field Embeddings (AFE)

**Preserving what dimensionality reduction destroys.**

AFE augments standard 3D dimensionality reduction with *arrow fields* — each embedded point receives *k* arrows, where each arrow encodes 3 additional dimensions via its geometric properties (azimuth, elevation, magnitude). Arrow *k* means the same thing on every point, enabling direct cross-point comparison of high-dimensional structure.

## Installation

```bash
pip install -e ".[pacmap]"
```

For development:

```bash
pip install -e ".[dev,pacmap]"
```

## Quick Start

```python
from afe import ArrowFieldEmbedding
from afe.visualization import plot_afe

# Fit AFE on your high-dimensional data
afe = ArrowFieldEmbedding(
    n_arrows=3,
    encoding_mode="adaptive",  # or "pca", "direct"
    backend="pacmap",
    random_state=42,
)
result = afe.fit_transform(X)

# Visualize interactively in 3D
fig = plot_afe(result["spatial"], result["arrows"], labels=y)
fig.show()
```

## How It Works

1. **Spatial Layout** — Points are placed in 3D via a backend (PaCMAP, DREAMS, or custom).
2. **Information Gap Analysis** — Measures which original dimensions the spatial layout captures vs misses.
3. **Arrow Encoding** — Maps residual (missed) dimensions onto arrows using one of three modes.
4. **Visualization** — Interactive 3D scatter with colored arrow cones per point.

## Encoding Modes

| Mode | Best For | How It Works |
|------|----------|--------------|
| **adaptive** | General use | Eigenvalue gap detection + hierarchical clustering groups correlated dims onto shared arrows |
| **pca** | High-dim embeddings | PCA on residuals; each arrow = one principal component |
| **direct** | Low-dim tabular data | 1-to-1 mapping of dimensions to arrow channels |

## Spatial Backends

| Backend | Install | Description |
|---------|---------|-------------|
| `pacmap` (default) | `pip install pacmap` | Fast manifold learning |
| `dreams` | [berenslab/DREAMS](https://github.com/berenslab/DREAMS) | t-SNE regularized toward PCA for balanced local/global structure |
| Manual | Pass `(n, 3)` array | Use any pre-computed 3D embedding |

## Evaluation

```python
from benchmarks.metrics import knn_recall, arrow_knn_recall, reconstruction_error

# Standard DR metric (spatial only)
knn = knn_recall(X, result["spatial"], k=10)

# AFE metric (spatial + arrows)
knn_afe = arrow_knn_recall(X, result["spatial"], result["arrows"], k=10)

# Reconstruction quality
X_recon = afe.reconstruct()
mse = reconstruction_error(X, X_recon)
```

## Running Benchmarks

```bash
python benchmarks/run_benchmark.py --datasets swiss_roll,hierarchical_gaussians --n-arrows 3
python benchmarks/compare_methods.py --datasets hierarchical_gaussians,mnist
```

## Testing

```bash
pytest test/ -v
```

## Project Structure

```
source/afe/         Core library (pipeline, backends, encoding, visualization)
benchmarks/         Datasets, metrics, benchmark runner, method comparison
test/               Comprehensive test suite
examples/           Quickstart, Swiss Roll demo, encoding mode comparison
```
