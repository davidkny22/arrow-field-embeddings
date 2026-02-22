"""Tests for spatial backends."""

import numpy as np
import pytest
from afe.backends import (
    PaCMAPBackend,
    DREAMSBackend,
    ManualBackend,
    get_backend,
    SpatialBackend,
)


class TestManualBackend:
    def test_fit_transform(self):
        emb = np.random.randn(50, 3).astype(np.float32)
        X = np.random.randn(50, 10).astype(np.float32)
        backend = ManualBackend(emb)
        result = backend.fit_transform(X)
        np.testing.assert_allclose(result, emb)

    def test_shape_mismatch(self):
        emb = np.random.randn(50, 3).astype(np.float32)
        X = np.random.randn(60, 10).astype(np.float32)
        backend = ManualBackend(emb)
        with pytest.raises(ValueError, match="50 points"):
            backend.fit_transform(X)

    def test_wrong_dims(self):
        with pytest.raises(ValueError, match="expects \\(n, 3\\)"):
            ManualBackend(np.random.randn(50, 2))

    def test_transform_not_supported(self):
        emb = np.random.randn(50, 3).astype(np.float32)
        backend = ManualBackend(emb)
        with pytest.raises(NotImplementedError):
            backend.transform(np.random.randn(10, 10))


class TestPaCMAPBackend:
    def test_fit_transform_shape(self):
        X = np.random.randn(200, 15).astype(np.float32)
        backend = PaCMAPBackend(random_state=42)
        result = backend.fit_transform(X)
        assert result.shape == (200, 3)
        assert result.dtype == np.float32

    def test_not_fitted_raises(self):
        backend = PaCMAPBackend()
        with pytest.raises(RuntimeError, match="must be fit"):
            backend.transform(np.random.randn(10, 15))


class TestDREAMSBackend:
    def test_import_error_message(self):
        """DREAMS should give a clear import error if openTSNE not installed."""
        # This test passes if DREAMS IS installed (it just runs),
        # or if it ISN'T installed (it raises ImportError with instructions).
        backend = DREAMSBackend(random_state=42)
        X = np.random.randn(50, 10).astype(np.float32)
        try:
            backend.fit_transform(X)
        except ImportError as e:
            assert "berenslab" in str(e)

    def test_transform_not_supported(self):
        backend = DREAMSBackend()
        with pytest.raises(NotImplementedError):
            backend.transform(np.random.randn(10, 10))


class TestGetBackend:
    def test_pacmap_string(self):
        backend = get_backend("pacmap", random_state=42)
        assert isinstance(backend, PaCMAPBackend)

    def test_dreams_string(self):
        backend = get_backend("dreams", random_state=42)
        assert isinstance(backend, DREAMSBackend)

    def test_ndarray(self):
        emb = np.random.randn(50, 3).astype(np.float32)
        backend = get_backend(emb)
        assert isinstance(backend, ManualBackend)

    def test_instance_passthrough(self):
        original = PaCMAPBackend(random_state=42)
        backend = get_backend(original)
        assert backend is original

    def test_unknown_string(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent")

    def test_manual_string_error(self):
        with pytest.raises(ValueError, match="numpy array"):
            get_backend("manual")

    def test_bad_type(self):
        with pytest.raises(TypeError):
            get_backend(42)
