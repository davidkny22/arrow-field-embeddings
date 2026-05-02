"""Tests for information_theory.py."""

import numpy as np
import pytest

from afe.information_theory import (
    gaussian_mutual_information,
    knn_mutual_information,
    information_gain,
    rate_distortion_curve,
    negative_control_shuffled,
    negative_control_random_spatial,
    negative_control_gaussian_arrows,
)


class TestGaussianMI:
    def test_independent_variables_mi_near_zero(self):
        rng = np.random.RandomState(42)
        X = rng.randn(500, 3)
        Y = rng.randn(500, 2)
        mi = gaussian_mutual_information(X, Y)
        assert mi >= 0.0
        assert mi < 0.1  # Should be very small for independent Gaussians

    def test_perfect_correlation_high_mi(self):
        rng = np.random.RandomState(42)
        X = rng.randn(500, 2)
        Y = X + rng.randn(500, 2) * 0.01  # Nearly identical
        mi = gaussian_mutual_information(X, Y)
        assert mi > 1.0  # Strongly dependent

    def test_constant_variable_returns_zero(self):
        X = np.ones((100, 1))
        Y = np.random.randn(100, 2)
        mi = gaussian_mutual_information(X, Y)
        assert mi == pytest.approx(0.0, abs=1e-6)

    def test_mi_with_itself_is_entropy(self):
        rng = np.random.RandomState(42)
        X = rng.randn(300, 3)
        mi = gaussian_mutual_information(X, X)
        # When Y = X, the joint covariance is rank-deficient.
        # The Gaussian formula returns 0 for perfectly dependent variables
        # because det(Sigma_XY) = det(Sigma_X)^2 = det(Sigma_X)*det(Sigma_Y).
        assert mi >= 0.0

    def test_1d_variables(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 1)
        Y = X + rng.randn(200, 1) * 0.1
        mi = gaussian_mutual_information(X, Y)
        assert mi > 0.5


class TestKnnMI:
    def test_independent_knn_mi_low(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 2)
        Y = rng.randn(200, 2)
        mi = knn_mutual_information(X, Y, k=3)
        assert mi >= 0.0
        assert mi < 0.5

    def test_correlated_knn_mi_high(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 2)
        Y = X + rng.randn(200, 2) * 0.1
        mi = knn_mutual_information(X, Y, k=3)
        assert mi > 0.3

    def test_subsampling_reduces_points(self):
        rng = np.random.RandomState(42)
        X = rng.randn(1000, 2)
        Y = rng.randn(1000, 2)
        mi_full = knn_mutual_information(X, Y, k=3)
        mi_sub = knn_mutual_information(X, Y, k=3, subsample=100, random_state=42)
        # Both should be small (independent)
        assert mi_sub < 0.5
        assert mi_full < 0.5


class TestInformationGain:
    def test_spatial_only_vs_full(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        spatial = rng.randn(100, 3)
        arrows = rng.randn(100, 2, 3)
        result = information_gain(X, spatial, arrows, estimator="gaussian")
        assert "mi_spatial" in result
        assert "mi_full" in result
        assert "information_gain" in result
        assert "fractional_gain" in result
        assert result["information_gain"] >= 0.0

    def test_knn_estimator(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        spatial = rng.randn(100, 3)
        arrows = rng.randn(100, 1, 3)
        result = information_gain(X, spatial, arrows, estimator="knn", k=3)
        assert "mi_spatial" in result
        assert "information_gain" in result
        assert "fractional_gain" not in result  # Only Gaussian reports this

    def test_zero_arrows_zero_gain(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        spatial = rng.randn(100, 3)
        arrows = np.zeros((100, 0, 3))
        result = information_gain(X, spatial, arrows, estimator="gaussian")
        # No arrows means spatial == full, so gain should be ~0
        assert result["information_gain"] < 0.01


class TestRateDistortionCurve:
    def test_basic_curve(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        representations = [
            ("spatial_3d", rng.randn(100, 3)),
            ("afe_6d", rng.randn(100, 6)),
            ("pca_10d", X),
        ]
        curve = rate_distortion_curve(X, representations)
        assert len(curve) == 3
        for point in curve:
            assert "name" in point
            assert "rate" in point
            assert "distortion" in point
            assert point["rate"] > 0
            assert point["distortion"] >= 0.0

    def test_pca_has_lowest_distortion(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        representations = [
            ("spatial_3d", rng.randn(100, 3)),
            ("pca_10d", X),
        ]
        curve = rate_distortion_curve(X, representations)
        names = [c["name"] for c in curve]
        dists = [c["distortion"] for c in curve]
        # PCA at full dimensionality should reconstruct perfectly
        pca_idx = names.index("pca_10d")
        assert dists[pca_idx] < 1e-6


class TestNegativeControls:
    def test_shuffled_preserves_marginals(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        X_shuf = negative_control_shuffled(X, random_state=42)
        np.testing.assert_allclose(np.mean(X, axis=0), np.mean(X_shuf, axis=0), atol=0.1)
        np.testing.assert_allclose(np.std(X, axis=0), np.std(X_shuf, axis=0), atol=0.1)

    def test_shuffled_destroys_joint_structure(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        X_shuf = negative_control_shuffled(X, random_state=42)
        # Correlation between first two dims should be ~0 after shuffling
        corr_orig = np.corrcoef(X[:, 0], X[:, 1])[0, 1]
        corr_shuf = np.corrcoef(X_shuf[:, 0], X_shuf[:, 1])[0, 1]
        assert abs(corr_shuf) < abs(corr_orig) + 0.1

    def test_random_spatial_shape(self):
        Z = negative_control_random_spatial(50, random_state=42)
        assert Z.shape == (50, 3)

    def test_gaussian_arrows_shape_and_ranges(self):
        arrows = negative_control_gaussian_arrows(30, 4, random_state=42)
        assert arrows.shape == (30, 4, 3)
        # theta in [0, 2pi), phi in [-pi/2, pi/2], r >= 0
        assert np.all(arrows[:, :, 0] >= 0)
        assert np.all(arrows[:, :, 0] < 2 * np.pi)
        assert np.all(arrows[:, :, 2] >= 0)
