"""Smoke tests for AFE export functionality."""

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from afe import ArrowFieldEmbedding
from afe.export import export_for_viewer


@pytest.fixture
def fitted_afe():
    """Return a fitted AFE instance with minimal data."""
    rng = np.random.RandomState(42)
    X = rng.randn(50, 10).astype(np.float32)
    spatial = rng.randn(50, 3).astype(np.float32)
    afe = ArrowFieldEmbedding(n_arrows=2, encoding_mode="direct", backend=spatial)
    afe.fit(X)
    return afe, X


def test_export_for_viewer_creates_valid_gzip_json(tmp_path, fitted_afe):
    """export_for_viewer produces a readable gzip-compressed JSON file."""
    afe, X = fitted_afe
    out_path = tmp_path / "dataset.json.gz"

    result_path = export_for_viewer(afe, X, path=str(out_path), dataset_name="test")

    assert result_path.exists()
    assert result_path.suffixes == [".json", ".gz"]

    with gzip.open(result_path, "rt") as f:
        data = json.load(f)

    assert data["version"] == 1
    assert data["dataset"] == "test"
    assert data["n_points"] == 50
    assert data["n_arrows"] == 2
    assert data["embedding_dim"] == 10
    assert "positions" in data
    assert "arrows" in data
    assert "label_indices" in data
    assert "label_names" in data
    assert "clusters" in data
    assert "gap_report" in data
    assert "metrics" in data
    assert "arrow_dim_labels" in data
    assert "arrow_attributions" in data


def test_export_for_viewer_with_labels(tmp_path, fitted_afe):
    """export_for_viewer handles explicit labels correctly."""
    afe, X = fitted_afe
    labels = np.array([0] * 25 + [1] * 25)
    out_path = tmp_path / "labeled.json.gz"

    export_for_viewer(afe, X, labels=labels, path=str(out_path))

    with gzip.open(out_path, "rt") as f:
        data = json.load(f)

    assert len(data["label_names"]) == 2
    assert len(data["clusters"]) == 2
    assert set(data["label_indices"]) == {0, 1}


def test_export_for_viewer_without_compression(tmp_path, fitted_afe):
    """export_for_viewer supports uncompressed output."""
    afe, X = fitted_afe
    out_path = tmp_path / "dataset.json"

    export_for_viewer(afe, X, path=str(out_path), compress=False)

    assert out_path.exists()
    with open(out_path) as f:
        data = json.load(f)
    assert data["n_points"] == 50
