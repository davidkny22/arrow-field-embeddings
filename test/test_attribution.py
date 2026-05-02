"""Tests for fitted arrow attribution metadata."""

import numpy as np

from afe import ArrowFieldEmbedding


class TestArrowAttribution:
    def test_direct_attribution_maps_channels(self, synthetic_data, manual_embedding):
        X, _ = synthetic_data
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        afe = ArrowFieldEmbedding(
            n_arrows=2,
            encoding_mode="direct",
            backend=manual_embedding,
            normalize_arrows=False,
        )
        afe.fit(X)

        attrs = afe.get_arrow_attributions()
        assert len(attrs) == 2
        assert attrs[0]["kind"] == "direct"
        assert [c["channel"] for c in attrs[0]["channels"]] == [
            "azimuth",
            "elevation",
            "magnitude",
        ]

        from afe.attribution import arrow_dim_labels

        labels = arrow_dim_labels(afe, feature_names)
        assert len(labels) == 2
        assert all(name.startswith("feature_") for name in labels[0])

    def test_pca_attribution_reports_components(self, synthetic_data, manual_embedding):
        X, _ = synthetic_data
        afe = ArrowFieldEmbedding(
            n_arrows=3,
            encoding_mode="pca",
            backend=manual_embedding,
            normalize_arrows=False,
        )
        afe.fit(X)

        attrs = afe.get_arrow_attributions()
        non_empty = [a for a in attrs if a["kind"] == "pca_component"]
        assert non_empty
        assert "variance_explained" in non_empty[0]
        assert "top_features" in non_empty[0]

    def test_adaptive_attribution_reports_groups(self, synthetic_data, manual_embedding):
        X, _ = synthetic_data
        afe = ArrowFieldEmbedding(
            n_arrows=3,
            encoding_mode="adaptive",
            backend=manual_embedding,
            normalize_arrows=False,
        )
        afe.fit(X)

        attrs = afe.get_arrow_attributions()
        groups = [a for a in attrs if a["kind"] == "adaptive_group"]
        assert groups
        assert "dimensions" in groups[0]
