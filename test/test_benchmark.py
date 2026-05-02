"""Tests for benchmark runner safeguards and metric coverage."""

import json
from pathlib import Path

import numpy as np
import pytest

from afe.reproducibility import RESULT_SCHEMA_VERSION
from benchmarks.compare_methods import run_standalone, run_afe
from benchmarks.io import load_completed, load_all_results


def test_load_completed_requires_current_schema(tmp_path):
    path = tmp_path / "old_schema_completed.jsonl"
    path.write_text(
        json.dumps({"type": "benchmark", "dataset": "d", "method": "m", "seed": 0})
        + "\n"
    )

    with pytest.raises(ValueError, match=RESULT_SCHEMA_VERSION):
        load_completed(path)


def test_load_all_results_requires_current_schema(tmp_path):
    path = tmp_path / "old_schema_results.jsonl"
    path.write_text(
        json.dumps({"type": "benchmark", "dataset": "d", "method": "m", "seed": 0})
        + "\n"
    )

    with pytest.raises(ValueError, match=RESULT_SCHEMA_VERSION):
        load_all_results(path)


def test_load_completed_accepts_current_rows(tmp_path):
    path = tmp_path / "v3_completed.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "benchmark",
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "dataset": "d",
                "method": "m",
                "seed": 0,
            }
        )
        + "\n"
    )

    assert load_completed(path) == {("d", "m", 0)}


def test_run_standalone_includes_classification_for_labeled_data():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 10).astype(np.float32)
    labels = (X[:, 0] > 0).astype(int)
    spatial = rng.randn(100, 3).astype(np.float32)
    metrics = run_standalone(
        X, labels, backend_name="manual", seed=42, spatial=spatial
    )
    assert "knn_class_acc_hd" in metrics
    assert "knn_class_f1_hd" in metrics
    assert "knn_class_acc_spatial" in metrics
    assert "knn_class_f1_spatial" in metrics
    assert 0.0 <= metrics["knn_class_acc_hd"] <= 1.0
    assert 0.0 <= metrics["knn_class_acc_spatial"] <= 1.0


def test_run_standalone_omits_classification_for_unlabeled_data():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 10).astype(np.float32)
    spatial = rng.randn(100, 3).astype(np.float32)
    metrics = run_standalone(
        X, None, backend_name="manual", seed=42, spatial=spatial
    )
    assert "knn_class_acc_hd" not in metrics
    assert "knn_class_acc_spatial" not in metrics


def test_run_afe_includes_classification_and_recon_knn50():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 10).astype(np.float32)
    labels = (X[:, 0] > 0).astype(int)
    spatial = rng.randn(100, 3).astype(np.float32)
    metrics = run_afe(
        X, labels, backend_name="manual", encoding_mode="direct",
        n_arrows=2, seed=42, spatial=spatial
    )
    assert "knn_class_acc_hd" in metrics
    assert "knn_class_f1_hd" in metrics
    assert "knn_class_acc_spatial" in metrics
    assert "knn_class_f1_spatial" in metrics
    assert "knn_class_acc_flat" in metrics
    assert "knn_class_f1_flat" in metrics
    assert "knn_class_acc_recon" in metrics
    assert "knn_class_f1_recon" in metrics
    assert "recon_knn_recall_k50" in metrics
    assert 0.0 <= metrics["knn_class_acc_hd"] <= 1.0
    assert 0.0 <= metrics["recon_knn_recall_k50"] <= 1.0


def test_run_afe_omits_classification_for_unlabeled_data():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 10).astype(np.float32)
    spatial = rng.randn(100, 3).astype(np.float32)
    metrics = run_afe(
        X, None, backend_name="manual", encoding_mode="direct",
        n_arrows=2, seed=42, spatial=spatial
    )
    assert "knn_class_acc_hd" not in metrics
    assert "knn_class_acc_flat" not in metrics
    assert "recon_knn_recall_k50" in metrics
