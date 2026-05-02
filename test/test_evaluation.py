"""Tests for public evaluation metrics."""

import numpy as np

from afe.evaluation import (
    arrow_spatial_information_gain,
    centroid_triplet_accuracy,
    continuity,
    dimension_coverage,
    flattened_afe_representation,
    knn_classification_metrics,
    knn_recall,
    normalized_stress,
    random_triplet_accuracy,
    recon_knn_recall,
    reconstruction_error,
    silhouette,
    spatial_information_gap,
    trustworthiness,
)


def test_spatial_information_gap_is_zero_for_identity_spatial():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 3).astype(np.float32)
    assert spatial_information_gap(X, X) < 1e-6


def test_recon_knn_recall_matches_knn_recall():
    rng = np.random.RandomState(42)
    X = rng.randn(50, 5).astype(np.float32)
    assert recon_knn_recall(X, X, k=5) == knn_recall(X, X, k=5)


def test_arrow_spatial_information_gain_positive_for_better_reconstruction():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 4).astype(np.float32)
    spatial_recon = np.zeros_like(X)
    full_recon = X.copy()
    assert arrow_spatial_information_gain(X, spatial_recon, full_recon) > 0.9


def test_reconstruction_error_and_dimension_coverage():
    X = np.ones((10, 4), dtype=np.float32)
    assert reconstruction_error(X, X, metric="mse") == 0.0
    coverage = dimension_coverage(n_features=10, n_arrows=2)
    assert coverage["total_representation_dims"] == 9
    assert coverage["coverage_fraction"] == 0.9


def test_flattened_representation_and_classification():
    spatial = np.zeros((20, 3), dtype=np.float32)
    arrows = np.zeros((20, 2, 3), dtype=np.float32)
    rep = flattened_afe_representation(spatial, arrows)
    assert rep.shape == (20, 9)

    labels = np.array([0] * 10 + [1] * 10)
    rep[:, 0] = labels
    metrics = knn_classification_metrics(rep, labels, n_neighbors=3)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_trustworthiness_and_continuity_are_bounded():
    rng = np.random.RandomState(42)
    X = rng.randn(50, 10).astype(np.float32)
    Y = rng.randn(50, 3).astype(np.float32)

    t = trustworthiness(X, Y, k=5)
    c = continuity(X, Y, k=5)

    assert 0.0 <= t <= 1.0
    assert 0.0 <= c <= 1.0


def test_trustworthiness_is_high_for_identity():
    rng = np.random.RandomState(42)
    X = rng.randn(30, 5).astype(np.float32)
    # Identity mapping should have high trustworthiness
    t = trustworthiness(X, X[:, :3], k=3)
    assert t > 0.5


def test_silhouette_with_valid_labels():
    rng = np.random.RandomState(42)
    X = rng.randn(40, 3).astype(np.float32)
    labels = np.array([0] * 20 + [1] * 20)

    score = silhouette(X, labels)
    assert score is not None
    assert -1.0 <= score <= 1.0


def test_silhouette_returns_none_for_invalid_labels():
    rng = np.random.RandomState(42)
    X = rng.randn(40, 3).astype(np.float32)

    assert silhouette(X, None) is None
    assert silhouette(X, np.zeros(40)) is None  # single class
    assert silhouette(X, np.arange(40)) is None  # all unique


def test_normalized_stress_is_nonnegative():
    rng = np.random.RandomState(42)
    X = rng.randn(50, 10).astype(np.float32)
    Y = rng.randn(50, 3).astype(np.float32)

    stress = normalized_stress(X, Y)
    assert stress >= 0.0


def test_random_triplet_accuracy_is_bounded():
    rng = np.random.RandomState(42)
    X = rng.randn(30, 10).astype(np.float32)
    Y = rng.randn(30, 3).astype(np.float32)

    acc = random_triplet_accuracy(X, Y, n_triplets=1000)
    assert 0.0 <= acc <= 1.0


def test_random_triplet_accuracy_for_identity():
    rng = np.random.RandomState(42)
    X = rng.randn(20, 5).astype(np.float32)
    # Perfect preservation should give ~1.0
    acc = random_triplet_accuracy(X, X, n_triplets=500)
    assert acc == 1.0


def test_centroid_triplet_accuracy_with_valid_labels():
    rng = np.random.RandomState(42)
    X = rng.randn(60, 10).astype(np.float32)
    labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
    Y = rng.randn(60, 3).astype(np.float32)

    ct = centroid_triplet_accuracy(X, Y, labels)
    assert ct is not None
    assert 0.0 <= ct <= 1.0


def test_centroid_triplet_accuracy_returns_none_for_few_classes():
    rng = np.random.RandomState(42)
    X = rng.randn(20, 5).astype(np.float32)
    Y = rng.randn(20, 3).astype(np.float32)

    assert centroid_triplet_accuracy(X, Y, None) is None
    assert centroid_triplet_accuracy(X, Y, np.zeros(20)) is None
    assert centroid_triplet_accuracy(X, Y, np.array([0] * 10 + [1] * 10)) is None
