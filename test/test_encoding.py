"""Tests for arrow encoding modes."""

import numpy as np
import pytest
from afe.encoding import (
    DirectMappingEncoder,
    PCAResidualEncoder,
    AdaptiveGroupingEncoder,
    get_encoder,
)


@pytest.fixture
def residual_10d():
    """10-dimensional residual data with structure."""
    rng = np.random.RandomState(42)
    n = 200
    base1 = rng.randn(n, 1)
    base2 = rng.randn(n, 1)

    data = np.empty((n, 10), dtype=np.float32)
    # Group 1: correlated dims 0-2
    data[:, 0] = (base1 + rng.randn(n, 1) * 0.1).ravel()
    data[:, 1] = (base1 + rng.randn(n, 1) * 0.2).ravel()
    data[:, 2] = (base1 + rng.randn(n, 1) * 0.15).ravel()
    # Group 2: correlated dims 3-5
    data[:, 3] = (base2 + rng.randn(n, 1) * 0.1).ravel()
    data[:, 4] = (base2 + rng.randn(n, 1) * 0.2).ravel()
    data[:, 5] = (base2 + rng.randn(n, 1) * 0.15).ravel()
    # Independent dims 6-9
    for i in range(6, 10):
        data[:, i] = rng.randn(n).astype(np.float32)

    return data


@pytest.fixture
def gap_report_stub():
    return {"residual_dims": list(range(10))}


# --- Factory ---

class TestGetEncoder:
    def test_direct(self):
        enc = get_encoder("direct")
        assert isinstance(enc, DirectMappingEncoder)

    def test_pca(self):
        enc = get_encoder("pca")
        assert isinstance(enc, PCAResidualEncoder)

    def test_adaptive(self):
        enc = get_encoder("adaptive")
        assert isinstance(enc, AdaptiveGroupingEncoder)

    def test_invalid(self):
        with pytest.raises(ValueError, match="Unknown encoding mode"):
            get_encoder("nonexistent")


# --- Direct Mapping ---

class TestDirectMapping:
    def test_encode_shape(self, residual_10d, gap_report_stub):
        enc = DirectMappingEncoder()
        enc.fit(residual_10d, n_arrows=3, gap_report=gap_report_stub)
        arrows = enc.encode(residual_10d)
        assert arrows.shape == (200, 3, 3)

    def test_encode_values(self, residual_10d, gap_report_stub):
        enc = DirectMappingEncoder()
        enc.fit(residual_10d, n_arrows=2, gap_report=gap_report_stub)
        arrows = enc.encode(residual_10d)
        # Arrow 0 should map dims 0, 1, 2
        np.testing.assert_allclose(arrows[:, 0, 0], residual_10d[:, 0])
        np.testing.assert_allclose(arrows[:, 0, 1], residual_10d[:, 1])
        np.testing.assert_allclose(arrows[:, 0, 2], residual_10d[:, 2])

    def test_roundtrip(self, residual_10d, gap_report_stub):
        enc = DirectMappingEncoder()
        enc.fit(residual_10d, n_arrows=4, gap_report=gap_report_stub)
        arrows = enc.encode(residual_10d)
        decoded = enc.decode(arrows)
        # First 12 dims should roundtrip perfectly (4 arrows * 3 channels)
        np.testing.assert_allclose(decoded[:, :10], residual_10d, atol=1e-6)

    def test_more_arrows_than_dims(self, gap_report_stub):
        rng = np.random.RandomState(42)
        data = rng.randn(50, 4).astype(np.float32)
        enc = DirectMappingEncoder()
        enc.fit(data, n_arrows=3, gap_report=gap_report_stub)
        arrows = enc.encode(data)
        assert arrows.shape == (50, 3, 3)
        # Extra channels should be zero
        np.testing.assert_allclose(arrows[:, 1, 1:], 0.0)


# --- PCA Residual ---

class TestPCAResidual:
    def test_encode_shape(self, residual_10d, gap_report_stub):
        enc = PCAResidualEncoder()
        enc.fit(residual_10d, n_arrows=3, gap_report=gap_report_stub)
        arrows = enc.encode(residual_10d)
        assert arrows.shape == (200, 3, 3)

    def test_magnitude_is_score(self, residual_10d, gap_report_stub):
        enc = PCAResidualEncoder()
        enc.fit(residual_10d, n_arrows=3, gap_report=gap_report_stub)
        arrows = enc.encode(residual_10d)
        # Magnitude channel (index 2) should have non-zero variance
        for i in range(3):
            assert np.std(arrows[:, i, 2]) > 0.01

    def test_azimuth_elevation_consistent(self, residual_10d, gap_report_stub):
        """Azimuth and elevation should be the same for every point
        (they come from loading direction, not per-point)."""
        enc = PCAResidualEncoder()
        enc.fit(residual_10d, n_arrows=3, gap_report=gap_report_stub)
        arrows = enc.encode(residual_10d)
        for i in range(3):
            # All points should have same azimuth for arrow i
            assert np.std(arrows[:, i, 0]) < 1e-6
            assert np.std(arrows[:, i, 1]) < 1e-6

    def test_roundtrip(self, residual_10d, gap_report_stub):
        enc = PCAResidualEncoder()
        enc.fit(residual_10d, n_arrows=10, gap_report=gap_report_stub)
        arrows = enc.encode(residual_10d)
        decoded = enc.decode(arrows)
        # With enough arrows, reconstruction should be close
        corr = np.corrcoef(residual_10d.ravel(), decoded.ravel())[0, 1]
        assert corr > 0.95

    def test_variance_ordering(self, residual_10d, gap_report_stub):
        """First PC should explain more variance than second."""
        enc = PCAResidualEncoder()
        enc.fit(residual_10d, n_arrows=5, gap_report=gap_report_stub)
        ratios = enc._pca.explained_variance_ratio_
        for i in range(len(ratios) - 1):
            assert ratios[i] >= ratios[i + 1]


# --- Adaptive Grouping ---

class TestAdaptiveGrouping:
    def test_encode_shape(self, residual_10d, gap_report_stub):
        enc = AdaptiveGroupingEncoder()
        enc.fit(residual_10d, n_arrows=3, gap_report=gap_report_stub)
        arrows = enc.encode(residual_10d)
        assert arrows.shape == (200, 3, 3)

    def test_detects_groups(self, residual_10d, gap_report_stub):
        """Should detect at least 2 groups from the correlated structure."""
        enc = AdaptiveGroupingEncoder(verbose=False)
        enc.fit(residual_10d, n_arrows=5, gap_report=gap_report_stub)
        assert len(enc._groups) >= 2

    def test_groups_cover_all_dims(self, residual_10d, gap_report_stub):
        """All residual dims should be assigned to some group."""
        enc = AdaptiveGroupingEncoder()
        enc.fit(residual_10d, n_arrows=10, gap_report=gap_report_stub)
        all_dims = set()
        for group in enc._groups:
            all_dims.update(group)
        assert all_dims == set(range(10))

    def test_roundtrip_not_garbage(self, residual_10d, gap_report_stub):
        enc = AdaptiveGroupingEncoder()
        enc.fit(residual_10d, n_arrows=5, gap_report=gap_report_stub)
        arrows = enc.encode(residual_10d)
        decoded = enc.decode(arrows)
        corr = np.corrcoef(residual_10d.ravel(), decoded.ravel())[0, 1]
        assert corr > 0.3

    def test_single_dim_residual(self, gap_report_stub):
        """Edge case: only 1 residual dimension."""
        rng = np.random.RandomState(42)
        data = rng.randn(50, 1).astype(np.float32)
        enc = AdaptiveGroupingEncoder()
        enc.fit(data, n_arrows=2, gap_report=gap_report_stub)
        arrows = enc.encode(data)
        assert arrows.shape == (50, 2, 3)

    def test_merge_when_too_many_groups(self, gap_report_stub):
        """If detection finds more groups than n_arrows, groups are merged."""
        rng = np.random.RandomState(42)
        # All independent dims -> each becomes its own group
        data = rng.randn(100, 8).astype(np.float32)
        enc = AdaptiveGroupingEncoder()
        enc.fit(data, n_arrows=3, gap_report=gap_report_stub)
        assert len(enc._groups) <= 3

    def test_empty_residual(self, gap_report_stub):
        data = np.empty((50, 0), dtype=np.float32)
        enc = AdaptiveGroupingEncoder()
        enc.fit(data, n_arrows=2, gap_report=gap_report_stub)
        arrows = enc.encode(data)
        assert arrows.shape == (50, 2, 3)
        np.testing.assert_allclose(arrows, 0.0)
