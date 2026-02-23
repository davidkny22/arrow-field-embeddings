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
    """Swiss roll: 3k points, no meaningful labels, 3 arrows."""
    from benchmarks.datasets import load_swiss_roll

    print("\n=== Swiss Roll (3k pts, 3 arrows, direct) ===")
    X, y = load_swiss_roll(n_samples=3000)

    afe = ArrowFieldEmbedding(
        n_arrows=3,
        encoding_mode="direct",
        backend="pacmap",
        random_state=42,
        verbose=True,
    )
    afe.fit_transform(X)

    export_for_viewer(
        afe, X,
        labels=y,
        path=str(PRESETS_DIR / "swiss_roll_direct_3arr.json.gz"),
        dataset_name="Swiss Roll",
    )
    return {
        "id": "swiss_roll_direct_3arr",
        "label": "Swiss Roll (3k, 3 arrows)",
        "url": "/presets/swiss_roll_direct_3arr.json.gz",
    }


def generate_mnist():
    """MNIST: 10k digits, 10 classes, PCA mode, 25 arrows."""
    from benchmarks.datasets import load_mnist

    print("\n=== MNIST (10k pts, 25 arrows, pca) ===")
    X, y = load_mnist(n_samples=10000)

    afe = ArrowFieldEmbedding(
        n_arrows=25,
        encoding_mode="pca",
        backend="pacmap",
        random_state=42,
        verbose=True,
    )
    afe.fit_transform(X)

    label_names = [str(i) for i in range(10)]
    export_for_viewer(
        afe, X,
        labels=y,
        path=str(PRESETS_DIR / "mnist_pca_25arr.json.gz"),
        dataset_name="MNIST Digits",
        label_names=label_names,
    )
    return {
        "id": "mnist_pca_25arr",
        "label": "MNIST (10k, 25 arrows, PCA)",
        "url": "/presets/mnist_pca_25arr.json.gz",
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

    export_for_viewer(
        afe, X,
        labels=y,
        path=str(PRESETS_DIR / "tabula_muris_direct_25arr.json.gz"),
        dataset_name="Tabula Muris",
    )
    return {
        "id": "tabula_muris_direct_25arr",
        "label": "Tabula Muris (20k cells, 25 arrows)",
        "url": "/presets/tabula_muris_direct_25arr.json.gz",
    }


def main():
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)

    index = []

    # Generate all presets
    index.append(generate_swiss_roll())
    index.append(generate_mnist())
    index.append(generate_tabula_muris())

    # Write index
    index_path = PRESETS_DIR / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nWrote {len(index)} presets to {PRESETS_DIR}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
