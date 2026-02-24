#!/usr/bin/env python3
"""Generate precomputed dataset presets for the AFE viewer.

Usage:
    python viewer/scripts/generate_presets.py

Generates JSON.gz files in viewer/public/presets/ and an index.json manifest.
"""

import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from source.afe import ArrowFieldEmbedding
from source.afe.export import export_for_viewer

PRESETS_DIR = PROJECT_ROOT / "viewer" / "public" / "presets"


def generate_swiss_roll():
    """Swiss roll: 3k points, 1 arrow (only 1 residual dim for this simple manifold)."""
    import numpy as np
    from benchmarks.datasets import load_swiss_roll

    print("\n=== Swiss Roll (3k pts, 1 arrow, direct) ===")
    X, y = load_swiss_roll(n_samples=3000)

    # Bin continuous position into 10 bands for meaningful cluster colors
    n_bins = 10
    bins = np.linspace(y.min(), y.max() + 1e-8, n_bins + 1)
    y_binned = np.digitize(y, bins) - 1  # 0-indexed bin labels

    afe = ArrowFieldEmbedding(
        n_arrows=1,
        encoding_mode="direct",
        backend="pacmap",
        random_state=42,
        verbose=True,
    )
    afe.fit_transform(X)

    # Swiss roll has 3D spatial coordinates
    feature_names = [f"x{i}" for i in range(X.shape[1])]
    label_names = [f"Band {i+1}" for i in range(n_bins)]
    export_for_viewer(
        afe, X,
        labels=y_binned,
        path=str(PRESETS_DIR / "swiss_roll_direct_1arr.json.gz"),
        dataset_name="Swiss Roll",
        label_names=label_names,
        feature_names=feature_names,
    )
    return {
        "id": "swiss_roll_direct_1arr",
        "label": "Swiss Roll (3k, 1 arrow)",
        "url": "/presets/swiss_roll_direct_1arr.json.gz",
    }


def generate_mnist():
    """MNIST: 10k digits, 10 classes, adaptive mode, 25 arrows."""
    from benchmarks.datasets import load_mnist

    print("\n=== MNIST (10k pts, 25 arrows, adaptive) ===")
    X, y = load_mnist(n_samples=10000)

    afe = ArrowFieldEmbedding(
        n_arrows=25,
        encoding_mode="adaptive",
        backend="pacmap",
        random_state=42,
        verbose=True,
    )
    afe.fit_transform(X)

    # MNIST: 784 pixel features
    feature_names = [f"pixel_{i}" for i in range(X.shape[1])]
    label_names = [str(i) for i in range(10)]
    export_for_viewer(
        afe, X,
        labels=y,
        path=str(PRESETS_DIR / "mnist_adaptive_25arr.json.gz"),
        dataset_name="MNIST Digits",
        label_names=label_names,
        feature_names=feature_names,
    )
    return {
        "id": "mnist_adaptive_25arr",
        "label": "MNIST (10k, 25 arrows, adaptive)",
        "url": "/presets/mnist_adaptive_25arr.json.gz",
    }


def generate_tabula_muris():
    """Tabula Muris: ~20k cells, 56 types, direct mode, 25 arrows."""
    from benchmarks.datasets import load_tabula_muris

    print("\n=== Tabula Muris (20k pts, 25 arrows, direct) ===")
    X, y = load_tabula_muris()

    afe = ArrowFieldEmbedding(
        n_arrows=25,
        encoding_mode="direct",
        backend="pacmap",
        random_state=42,
        verbose=True,
    )
    afe.fit_transform(X)

    # Tabula Muris is PCA-50 transformed; label dims as PC components
    feature_names = [f"PC{i+1}" for i in range(X.shape[1])]
    export_for_viewer(
        afe, X,
        labels=y,
        path=str(PRESETS_DIR / "tabula_muris_direct_25arr.json.gz"),
        dataset_name="Tabula Muris",
        feature_names=feature_names,
    )
    return {
        "id": "tabula_muris_direct_25arr",
        "label": "Tabula Muris (20k cells, 25 arrows)",
        "url": "/presets/tabula_muris_direct_25arr.json.gz",
    }


def generate_ag_news():
    """AG News: 10k sentences embedded with all-MiniLM-L6-v2, 4 classes, direct, 25 arrows."""
    import numpy as np

    print("\n=== AG News / MiniLM (10k pts, 25 arrows, direct) ===")

    cache_path = PROJECT_ROOT / "data" / "ag_news_minilm.npz"
    if cache_path.exists():
        print(f"Loading cached embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        X, y = data["X"], data["y"]
        label_names = data["label_names"].tolist()
    else:
        from datasets import load_dataset
        from sentence_transformers import SentenceTransformer

        print("Downloading AG News and embedding with MiniLM...")
        ds = load_dataset("ag_news", split="train")
        rng = np.random.default_rng(42)
        indices = rng.choice(len(ds), size=10000, replace=False)
        indices.sort()
        subset = ds.select(indices.tolist())

        texts = subset["text"]
        y = np.array(subset["label"])
        label_names = ["World", "Sports", "Business", "Sci/Tech"]

        model = SentenceTransformer("all-MiniLM-L6-v2")
        X = np.array(model.encode(texts, show_progress_bar=True, batch_size=256), dtype=np.float32)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y,
                            texts=np.array(texts, dtype=object),
                            label_names=np.array(label_names))
        print(f"Cached embeddings to {cache_path}")

    afe = ArrowFieldEmbedding(
        n_arrows=25,
        encoding_mode="direct",
        backend="pacmap",
        random_state=42,
        verbose=True,
    )
    afe.fit_transform(X)

    feature_names = [f"emb_{i}" for i in range(X.shape[1])]
    export_for_viewer(
        afe, X,
        labels=y,
        path=str(PRESETS_DIR / "ag_news_minilm_direct_25arr.json.gz"),
        dataset_name="AG News (MiniLM)",
        label_names=label_names,
        feature_names=feature_names,
    )
    return {
        "id": "ag_news_minilm_direct_25arr",
        "label": "AG News / MiniLM (10k, 25 arrows)",
        "url": "/presets/ag_news_minilm_direct_25arr.json.gz",
    }


def main():
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)

    index = []

    # Generate all presets
    index.append(generate_swiss_roll())
    index.append(generate_mnist())
    index.append(generate_tabula_muris())
    index.append(generate_ag_news())

    # Write index
    index_path = PRESETS_DIR / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nWrote {len(index)} presets to {PRESETS_DIR}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
