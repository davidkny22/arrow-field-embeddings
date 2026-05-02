"""Tests for biology.py."""

import numpy as np
import pytest

from afe.biology import (
    map_arrows_to_genes,
    marker_gene_enrichment,
    pathway_enrichment,
    validate_against_markers,
    arrow_gene_program_summary,
)


class TestMapArrowsToGenes:
    def test_direct_mapping_basic(self):
        from afe import ArrowFieldEmbedding
        rng = np.random.RandomState(42)
        X = rng.randn(50, 20)
        y = np.zeros(50, dtype=int)
        afe = ArrowFieldEmbedding(
            n_arrows=2, encoding_mode="direct", backend="tsne", random_state=42
        )
        afe.fit_transform(X)

        pca_loadings = rng.randn(20, 20)
        hvg_names = [f"gene_{i}" for i in range(20)]

        result = map_arrows_to_genes(afe, pca_loadings, hvg_names, top_n_genes=5)
        assert len(result) == 2
        for r in result:
            assert "arrow_index" in r
            assert "kind" in r
            assert "top_genes" in r
            assert "gene_scores" in r
            assert len(r["top_genes"]) <= 5

    def test_pca_mapping_basic(self):
        from afe import ArrowFieldEmbedding
        rng = np.random.RandomState(42)
        X = rng.randn(50, 20)
        afe = ArrowFieldEmbedding(
            n_arrows=3, encoding_mode="pca", backend="tsne", random_state=42
        )
        afe.fit_transform(X)

        pca_loadings = rng.randn(20, 20)
        hvg_names = [f"gene_{i}" for i in range(20)]

        result = map_arrows_to_genes(afe, pca_loadings, hvg_names, top_n_genes=5)
        assert len(result) == 3
        for r in result:
            assert r["kind"] == "pca_component"
            assert len(r["top_genes"]) <= 5

    def test_empty_arrows(self):
        from afe import ArrowFieldEmbedding
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        afe = ArrowFieldEmbedding(
            n_arrows=1, encoding_mode="direct", backend="tsne", random_state=42
        )
        afe.fit_transform(X)

        pca_loadings = rng.randn(10, 10)
        hvg_names = [f"g{i}" for i in range(10)]

        result = map_arrows_to_genes(afe, pca_loadings, hvg_names)
        # With 1 arrow, we get 1 result
        assert len(result) == 1
        assert result[0]["arrow_index"] == 0


class TestMarkerGeneEnrichment:
    def test_perfect_enrichment(self):
        rng = np.random.RandomState(42)
        n_genes = 100
        arrow_scores = np.zeros(n_genes)
        # Top 10 genes are known markers for cell type A
        marker_genes = {
            "cell_A": [f"gene_{i}" for i in range(10)],
            "cell_B": [f"gene_{i}" for i in range(90, 100)],
        }
        hvg_names = [f"gene_{i}" for i in range(n_genes)]
        # Give cell_A markers high scores
        for i in range(10):
            arrow_scores[i] = 10.0 - i

        result = marker_gene_enrichment(arrow_scores, marker_genes, hvg_names, top_n=20)
        assert "cell_A" in result
        assert result["cell_A"]["overlap"] == 10
        assert result["cell_A"]["p_value"] < 0.01
        assert result["cell_A"]["odds_ratio"] > 1.0

    def test_no_overlap(self):
        n_genes = 100
        arrow_scores = np.zeros(n_genes)
        marker_genes = {
            "cell_A": [f"gene_{i}" for i in range(50, 60)],
        }
        hvg_names = [f"gene_{i}" for i in range(n_genes)]
        # Top genes are completely different
        arrow_scores[:10] = 10.0

        result = marker_gene_enrichment(arrow_scores, marker_genes, hvg_names, top_n=10)
        assert result["cell_A"]["overlap"] == 0
        assert result["cell_A"]["p_value"] == 1.0 or result["cell_A"]["odds_ratio"] == 0.0

    def test_markers_not_in_background(self):
        n_genes = 50
        arrow_scores = np.zeros(n_genes)
        marker_genes = {
            "cell_A": ["missing_gene_1", "missing_gene_2"],
        }
        hvg_names = [f"gene_{i}" for i in range(n_genes)]

        result = marker_gene_enrichment(arrow_scores, marker_genes, hvg_names)
        assert "cell_A" not in result  # No markers in background


class TestPathwayEnrichment:
    def test_pathway_same_math_as_markers(self):
        n_genes = 100
        arrow_scores = np.zeros(n_genes)
        pathways = {
            "GO_0001": [f"gene_{i}" for i in range(10)],
        }
        hvg_names = [f"gene_{i}" for i in range(n_genes)]
        for i in range(10):
            arrow_scores[i] = 10.0 - i

        result = pathway_enrichment(arrow_scores, pathways, hvg_names, top_n=20)
        assert "GO_0001" in result
        assert result["GO_0001"]["overlap"] == 10


class TestValidateAgainstMarkers:
    def test_full_pipeline(self):
        from afe import ArrowFieldEmbedding
        rng = np.random.RandomState(42)
        X = rng.randn(50, 20)
        afe = ArrowFieldEmbedding(
            n_arrows=2, encoding_mode="pca", backend="tsne", random_state=42
        )
        afe.fit_transform(X)

        pca_loadings = np.eye(20)  # Simple identity
        hvg_names = [f"gene_{i}" for i in range(20)]
        marker_genes = {
            "type_A": ["gene_0", "gene_1", "gene_2"],
            "type_B": ["gene_3", "gene_4", "gene_5"],
        }

        reports = validate_against_markers(
            afe, pca_loadings, hvg_names, marker_genes, top_n_genes=10
        )
        assert len(reports) == 2
        for rep in reports:
            assert "arrow_index" in rep
            assert "enrichment" in rep


class TestArrowGeneProgramSummary:
    def test_basic_summary(self):
        reports = [
            {
                "arrow_index": 0,
                "enrichment": [
                    {"cell_type": "A", "p_value": 0.001},
                    {"cell_type": "B", "p_value": 0.5},
                ],
            },
            {
                "arrow_index": 1,
                "enrichment": [
                    {"cell_type": "C", "p_value": 0.01},
                ],
            },
            {
                "arrow_index": 2,
                "enrichment": [],
            },
        ]
        summary = arrow_gene_program_summary(reports)
        assert summary["n_arrows"] == 3
        assert summary["arrows_with_significant_enrichment"] == 2
        assert summary["fraction_significant"] == pytest.approx(2 / 3)
        assert summary["best_overall_p_value"] == pytest.approx(0.001)

    def test_empty_reports(self):
        summary = arrow_gene_program_summary([])
        assert summary == {}
