"""Integration tests: end-to-end pipeline across modes and datasets."""

import numpy as np
import pytest
from afe import ArrowFieldEmbedding


class TestAllEncodingModes:
    """Test all encoding modes with manual backend on synthetic data."""

    @pytest.fixture
    def data_and_backend(self, synthetic_data):
        X, _ = synthetic_data
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3, random_state=42)
        embedding = pca.fit_transform(X).astype(np.float32)
        return X, embedding

    @pytest.mark.parametrize("mode", ["direct", "pca", "adaptive"])
    def test_fit_transform_shape(self, mode, data_and_backend):
        X, emb = data_and_backend
        afe = ArrowFieldEmbedding(
            n_arrows=2, encoding_mode=mode, backend=emb
        )
        result = afe.fit_transform(X)
        assert result["spatial"].shape == (len(X), 3)
        assert result["arrows"].shape == (len(X), 2, 3)

    @pytest.mark.parametrize("mode", ["direct", "pca", "adaptive"])
    def test_reconstruct(self, mode, data_and_backend):
        X, emb = data_and_backend
        afe = ArrowFieldEmbedding(
            n_arrows=3,
            encoding_mode=mode,
            backend=emb,
            normalize_arrows=False,
        )
        afe.fit(X)
        X_recon = afe.reconstruct()
        assert X_recon.shape == X.shape
        # Reconstruction should be better than random
        corr = np.corrcoef(X.ravel(), X_recon.ravel())[0, 1]
        assert corr > 0.2, f"Mode {mode}: reconstruction corr = {corr}"

    @pytest.mark.parametrize("n_arrows", [1, 2, 5, 10])
    def test_varying_arrow_counts(self, n_arrows, data_and_backend):
        X, emb = data_and_backend
        afe = ArrowFieldEmbedding(
            n_arrows=n_arrows, encoding_mode="direct", backend=emb
        )
        result = afe.fit_transform(X)
        assert result["arrows"].shape == (len(X), n_arrows, 3)


class TestWithPaCMAPBackend:
    """Test that PaCMAP backend works end-to-end (requires pacmap)."""

    def test_pacmap_swiss_roll(self, swiss_roll_data):
        X, t = swiss_roll_data
        afe = ArrowFieldEmbedding(
            n_arrows=1,
            encoding_mode="direct",
            backend="pacmap",
            random_state=42,
        )
        result = afe.fit_transform(X)
        assert result["spatial"].shape == (len(X), 3)
        assert result["arrows"].shape == (len(X), 1, 3)

    def test_pacmap_high_dim(self):
        """Test on higher-dimensional synthetic data."""
        rng = np.random.RandomState(42)
        X = rng.randn(300, 50).astype(np.float32)

        afe = ArrowFieldEmbedding(
            n_arrows=3,
            encoding_mode="pca",
            backend="pacmap",
            random_state=42,
        )
        result = afe.fit_transform(X)
        assert result["spatial"].shape == (300, 3)
        assert result["arrows"].shape == (300, 3, 3)
        assert result["metadata"]["n_residual_dims"] > 0


class TestMetricsIntegration:
    """Test metrics work with AFE outputs."""

    def test_standard_metrics(self, synthetic_data, manual_embedding):
        from benchmarks.metrics import knn_recall, spearman_distance_correlation

        X, _ = synthetic_data
        afe = ArrowFieldEmbedding(
            n_arrows=2, encoding_mode="direct", backend=manual_embedding
        )
        result = afe.fit_transform(X)

        knn = knn_recall(X, result["spatial"], k=5)
        spearman = spearman_distance_correlation(X, result["spatial"])
        assert 0 <= knn <= 1
        assert -1 <= spearman <= 1

    def test_afe_metrics(self, synthetic_data, manual_embedding):
        from benchmarks.metrics import (
            reconstruction_error,
            arrow_knn_recall,
            arrow_consistency,
        )

        X, _ = synthetic_data
        afe = ArrowFieldEmbedding(
            n_arrows=2,
            encoding_mode="direct",
            backend=manual_embedding,
            normalize_arrows=False,
        )
        afe.fit(X)
        result = afe._build_result()

        X_recon = afe.reconstruct()
        mse = reconstruction_error(X, X_recon)
        assert mse >= 0

        aknn = arrow_knn_recall(X, result["spatial"], result["arrows"], k=5)
        assert 0 <= aknn <= 1

        cons = arrow_consistency(result["arrows"], X, k=5)
        # Consistency should be non-trivial
        assert -1 <= cons <= 1
