"""Smoke tests for the visualization module."""

import numpy as np
import pytest


class TestPlotAfe:
    def test_returns_figure(self, synthetic_data, manual_embedding):
        from afe import ArrowFieldEmbedding
        from afe.visualization import plot_afe

        X, labels = synthetic_data
        afe = ArrowFieldEmbedding(
            n_arrows=2,
            encoding_mode="direct",
            backend=manual_embedding,
        )
        result = afe.fit_transform(X)

        fig = plot_afe(result["spatial"], result["arrows"], labels=labels)
        assert fig is not None
        assert hasattr(fig, "data")
        # Points trace + 2 arrow cone traces = 3 traces
        assert len(fig.data) == 3

    def test_no_arrows(self, synthetic_data, manual_embedding):
        from afe import ArrowFieldEmbedding
        from afe.visualization import plot_afe

        X, labels = synthetic_data
        afe = ArrowFieldEmbedding(
            n_arrows=2,
            encoding_mode="direct",
            backend=manual_embedding,
        )
        result = afe.fit_transform(X)

        fig = plot_afe(result["spatial"], result["arrows"], show_arrows=False)
        assert len(fig.data) == 1  # Points only

    def test_selective_arrows(self, synthetic_data, manual_embedding):
        from afe import ArrowFieldEmbedding
        from afe.visualization import plot_afe

        X, _ = synthetic_data
        afe = ArrowFieldEmbedding(
            n_arrows=3,
            encoding_mode="direct",
            backend=manual_embedding,
        )
        result = afe.fit_transform(X)

        fig = plot_afe(result["spatial"], result["arrows"], show_arrows=[0, 2])
        assert len(fig.data) == 3  # Points + 2 selected arrows


class TestPlotInfoGap:
    def test_returns_figure(self, synthetic_data, manual_embedding):
        from afe import ArrowFieldEmbedding
        from afe.visualization import plot_info_gap

        X, _ = synthetic_data
        afe = ArrowFieldEmbedding(
            n_arrows=2,
            encoding_mode="direct",
            backend=manual_embedding,
        )
        afe.fit(X)
        gap = afe.get_gap_report()

        fig = plot_info_gap(gap)
        assert fig is not None
        assert hasattr(fig, "data")
