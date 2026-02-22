"""Tests for information gap analysis."""

import numpy as np
import pytest
from afe.gap_analysis import InformationGapAnalyzer


class TestInformationGapAnalyzer:
    def test_basic_analysis(self):
        rng = np.random.RandomState(42)
        n = 100
        X_3d = rng.randn(n, 3).astype(np.float32)
        # First 3 dims = spatial coords (perfectly captured)
        # Last 3 dims = independent noise (not captured)
        X_high = np.column_stack([X_3d, rng.randn(n, 3)]).astype(np.float32)

        analyzer = InformationGapAnalyzer(correlation_threshold=0.3)
        report = analyzer.analyze(X_high, X_3d)

        assert "correlation_matrix" in report
        assert "residual_dims" in report
        assert "captured_dims" in report
        assert report["correlation_matrix"].shape == (6, 3)

    def test_captured_dims_detected(self):
        rng = np.random.RandomState(42)
        n = 200
        X_3d = rng.randn(n, 3).astype(np.float32)
        # HD data: first 3 dims = exact copies of spatial
        X_high = np.column_stack([X_3d, rng.randn(n, 5) * 0.01]).astype(np.float32)

        analyzer = InformationGapAnalyzer(correlation_threshold=0.3)
        report = analyzer.analyze(X_high, X_3d)

        # First 3 dims should be captured
        captured = set(report["captured_dims"])
        assert 0 in captured
        assert 1 in captured
        assert 2 in captured

    def test_residual_dims_detected(self):
        rng = np.random.RandomState(42)
        n = 200
        X_3d = rng.randn(n, 3).astype(np.float32)
        X_high = np.column_stack([X_3d, rng.randn(n, 4)]).astype(np.float32)

        analyzer = InformationGapAnalyzer(correlation_threshold=0.3)
        report = analyzer.analyze(X_high, X_3d)

        # Last 4 dims should be residual
        residual = set(report["residual_dims"])
        for d in [3, 4, 5, 6]:
            assert d in residual

    def test_residual_data_shape(self):
        rng = np.random.RandomState(42)
        n = 100
        X_3d = rng.randn(n, 3).astype(np.float32)
        X_high = np.column_stack([X_3d, rng.randn(n, 5)]).astype(np.float32)

        analyzer = InformationGapAnalyzer(correlation_threshold=0.3)
        report = analyzer.analyze(X_high, X_3d)

        n_residual = len(report["residual_dims"])
        assert report["residual_data"].shape == (n, n_residual)

    def test_gap_score_range(self):
        rng = np.random.RandomState(42)
        n = 100
        X_3d = rng.randn(n, 3).astype(np.float32)
        X_high = rng.randn(n, 10).astype(np.float32)

        analyzer = InformationGapAnalyzer()
        report = analyzer.analyze(X_high, X_3d)
        assert 0.0 <= report["information_gap_score"] <= 1.0

    def test_all_captured_edge_case(self):
        """If all dims correlate with spatial, no residuals."""
        rng = np.random.RandomState(42)
        n = 200
        X_3d = rng.randn(n, 3).astype(np.float32)
        # All dims are linear combos of spatial
        X_high = X_3d @ rng.randn(3, 5).astype(np.float32)

        analyzer = InformationGapAnalyzer(correlation_threshold=0.3)
        report = analyzer.analyze(X_high, X_3d)
        assert len(report["residual_dims"]) == 0
        assert report["residual_data"].shape[1] == 0

    def test_constant_dim_handled(self):
        """Constant dimensions (zero variance) should not crash."""
        rng = np.random.RandomState(42)
        n = 100
        X_3d = rng.randn(n, 3).astype(np.float32)
        X_high = np.column_stack([
            rng.randn(n, 3),
            np.ones((n, 1)),  # constant dim
            rng.randn(n, 2),
        ]).astype(np.float32)

        analyzer = InformationGapAnalyzer()
        report = analyzer.analyze(X_high, X_3d)
        assert report["correlation_matrix"].shape == (6, 3)
        # Should not contain NaN
        assert not np.any(np.isnan(report["correlation_matrix"]))
