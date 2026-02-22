"""Tests for the core ArrowFieldEmbedding class."""

import numpy as np
import pytest
from afe import ArrowFieldEmbedding, ManualBackend


class TestArrowFieldEmbeddingInit:
    def test_default_init(self):
        afe = ArrowFieldEmbedding()
        assert afe.n_arrows == 2
        assert afe.encoding_mode == "direct"
        assert afe.backend == "pacmap"
        assert afe.normalize_arrows is True

    def test_custom_init(self):
        afe = ArrowFieldEmbedding(
            n_arrows=5,
            encoding_mode="direct",
            backend="pacmap",
            correlation_threshold=0.5,
            random_state=42,
        )
        assert afe.n_arrows == 5
        assert afe.correlation_threshold == 0.5
        assert afe.random_state == 42

    def test_invalid_encoding_mode(self, synthetic_data, manual_embedding):
        X, _ = synthetic_data
        afe = ArrowFieldEmbedding(encoding_mode="invalid", backend=manual_embedding)
        with pytest.raises(ValueError, match="Unknown encoding mode"):
            afe.fit(X)


class TestFitTransformWithManualBackend:
    """Test the full pipeline using a manual (PCA) backend to avoid PaCMAP dep."""

    def test_fit_returns_self(self, synthetic_data, manual_embedding):
        X, _ = synthetic_data
        afe = ArrowFieldEmbedding(
            n_arrows=2,
            encoding_mode="direct",
            backend=manual_embedding,
        )
        result = afe.fit(X)
        assert result is afe

    def test_fit_transform_returns_dict(self, synthetic_data, manual_embedding):
        X, _ = synthetic_data
        afe = ArrowFieldEmbedding(
            n_arrows=2,
            encoding_mode="direct",
            backend=manual_embedding,
        )
        result = afe.fit_transform(X)
        assert isinstance(result, dict)
        assert "spatial" in result
        assert "arrows" in result
        assert "metadata" in result

    def test_spatial_shape(self, synthetic_data, manual_embedding):
        X, _ = synthetic_data
        afe = ArrowFieldEmbedding(
            n_arrows=2,
            encoding_mode="direct",
            backend=manual_embedding,
        )
        result = afe.fit_transform(X)
        assert result["spatial"].shape == (len(X), 3)

    def test_arrows_shape(self, synthetic_data, manual_embedding):
        X, _ = synthetic_data
        n_arrows = 3
        afe = ArrowFieldEmbedding(
            n_arrows=n_arrows,
            encoding_mode="direct",
            backend=manual_embedding,
        )
        result = afe.fit_transform(X)
        assert result["arrows"].shape == (len(X), n_arrows, 3)

    def test_metadata_contents(self, synthetic_data, manual_embedding):
        X, _ = synthetic_data
        afe = ArrowFieldEmbedding(
            n_arrows=2,
            encoding_mode="direct",
            backend=manual_embedding,
        )
        result = afe.fit_transform(X)
        meta = result["metadata"]
        assert meta["encoding_mode"] == "direct"
        assert meta["n_arrows"] == 2
        assert meta["n_features_original"] == X.shape[1]
        assert meta["dims_per_arrow"] == 3

    def test_get_spatial(self, synthetic_data, manual_embedding):
        X, _ = synthetic_data
        afe = ArrowFieldEmbedding(
            n_arrows=2,
            encoding_mode="direct",
            backend=manual_embedding,
        )
        afe.fit(X)
        spatial = afe.get_spatial()
        assert spatial.shape == (len(X), 3)

    def test_get_arrows(self, synthetic_data, manual_embedding):
        X, _ = synthetic_data
        afe = ArrowFieldEmbedding(
            n_arrows=2,
            encoding_mode="direct",
            backend=manual_embedding,
        )
        afe.fit(X)
        arrows = afe.get_arrows()
        assert arrows.shape == (len(X), 2, 3)

    def test_not_fitted_raises(self):
        afe = ArrowFieldEmbedding()
        with pytest.raises(RuntimeError, match="not been fitted"):
            afe.get_spatial()


class TestDeterminism:
    def test_deterministic_with_same_seed(self, synthetic_data, manual_embedding):
        X, _ = synthetic_data
        results = []
        for _ in range(2):
            afe = ArrowFieldEmbedding(
                n_arrows=2,
                encoding_mode="direct",
                backend=manual_embedding,
                random_state=42,
            )
            results.append(afe.fit_transform(X))
        np.testing.assert_allclose(
            results[0]["arrows"], results[1]["arrows"], atol=1e-6
        )


class TestNormalization:
    def test_normalized_arrow_ranges(self, synthetic_data, manual_embedding):
        X, _ = synthetic_data
        afe = ArrowFieldEmbedding(
            n_arrows=2,
            encoding_mode="direct",
            backend=manual_embedding,
            normalize_arrows=True,
        )
        result = afe.fit_transform(X)
        arrows = result["arrows"]

        # Azimuth should be in [-pi, pi]
        assert arrows[:, :, 0].min() >= -np.pi - 1e-6
        assert arrows[:, :, 0].max() <= np.pi + 1e-6

        # Elevation should be in [-pi/2, pi/2]
        assert arrows[:, :, 1].min() >= -np.pi / 2 - 1e-6
        assert arrows[:, :, 1].max() <= np.pi / 2 + 1e-6

        # Magnitude should be in [0, 1]
        assert arrows[:, :, 2].min() >= -1e-6
        assert arrows[:, :, 2].max() <= 1.0 + 1e-6

    def test_unnormalized_arrows(self, synthetic_data, manual_embedding):
        X, _ = synthetic_data
        afe = ArrowFieldEmbedding(
            n_arrows=2,
            encoding_mode="direct",
            backend=manual_embedding,
            normalize_arrows=False,
        )
        result = afe.fit_transform(X)
        arrows = result["arrows"]
        # Raw arrows are just the residual values, no range constraint
        assert arrows.shape == (len(X), 2, 3)


class TestReconstruction:
    def test_reconstruct_shape(self, synthetic_data, manual_embedding):
        X, _ = synthetic_data
        afe = ArrowFieldEmbedding(
            n_arrows=2,
            encoding_mode="direct",
            backend=manual_embedding,
        )
        afe.fit(X)
        X_recon = afe.reconstruct()
        assert X_recon.shape == X.shape

    def test_reconstruct_not_garbage(self, synthetic_data, manual_embedding):
        X, _ = synthetic_data
        afe = ArrowFieldEmbedding(
            n_arrows=3,
            encoding_mode="direct",
            backend=manual_embedding,
            normalize_arrows=False,
        )
        afe.fit(X)
        X_recon = afe.reconstruct()
        # Reconstruction should correlate with original (not random)
        corr = np.corrcoef(X.ravel(), X_recon.ravel())[0, 1]
        assert corr > 0.3, f"Reconstruction correlation too low: {corr}"
