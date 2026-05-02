"""Biological validation helpers for scRNA-seq AFE analysis.

Maps arrows back through PCA loadings to gene-level programs, then validates
against known marker genes or pathway databases.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import hypergeom

_logger = logging.getLogger(__name__)

__all__ = [
    "map_arrows_to_genes",
    "marker_gene_enrichment",
    "pathway_enrichment",
    "validate_against_markers",
    "arrow_gene_program_summary",
]


def map_arrows_to_genes(
    afe,
    pca_loadings: np.ndarray,
    hvg_names: List[str],
    top_n_genes: int = 20,
) -> List[Dict]:
    """Map each arrow to its strongest gene-level contributors.

    Traces the attribution chain::

        arrow -> residual dimension -> PCA component -> gene loading

    Parameters
    ----------
    afe
        Fitted ``ArrowFieldEmbedding`` with ``encoding_mode="pca"`` or
        ``"direct"``.  PCA mode is strongly preferred for interpretability.
    pca_loadings : ndarray (n_hvgs, n_pcs)
        PCA loadings matrix mapping genes to principal components.
    hvg_names : list of str
        Gene names corresponding to rows of ``pca_loadings``.
    top_n_genes : int
        Number of top-loading genes to report per arrow.

    Returns
    -------
    arrow_genes : list of dict
        One dict per arrow with keys ``arrow_index``, ``top_genes``,
        ``gene_scores``.
    """
    if hasattr(afe, "_check_fitted"):
        afe._check_fitted()

    from .attribution import get_arrow_attributions

    gap_report = afe.get_gap_report()
    residual_dims = list(gap_report.get("residual_dims", []))
    attributions = get_arrow_attributions(afe, feature_names=None, top_n=top_n_genes)

    pca_loadings = np.asarray(pca_loadings)
    n_hvgs, n_pcs = pca_loadings.shape

    results = []
    for attr in attributions:
        arrow_idx = attr["arrow_index"]
        gene_scores = np.zeros(n_hvgs, dtype=np.float64)

        if attr["kind"] == "pca_component":
            # Arrow corresponds to a single PCA component
            comp_idx = attr.get("component_index", arrow_idx)
            if comp_idx < n_pcs:
                loadings = np.abs(pca_loadings[:, comp_idx])
                gene_scores = loadings
        elif attr["kind"] == "direct":
            # Arrow stores raw residual channels
            # Map each channel to the residual dimension, then to PCA
            for ch in attr.get("channels", []):
                res_idx = ch.get("residual_index")
                if res_idx is not None and res_idx < len(residual_dims):
                    orig_dim = residual_dims[res_idx]
                    if orig_dim < n_pcs:
                        gene_scores += np.abs(pca_loadings[:, orig_dim])
        elif attr["kind"] == "adaptive_group":
            # Sum loadings across all PCA components in the group
            for dim_info in attr.get("dimensions", []):
                res_idx = dim_info.get("residual_index")
                if res_idx is not None and res_idx < len(residual_dims):
                    orig_dim = residual_dims[res_idx]
                    if orig_dim < n_pcs:
                        gene_scores += np.abs(pca_loadings[:, orig_dim])

        # Rank genes by score
        order = np.argsort(gene_scores)[::-1][:top_n_genes]
        top_genes = [
            {
                "gene": str(hvg_names[i]) if i < len(hvg_names) else f"gene_{i}",
                "score": float(gene_scores[i]),
                "rank": int(rank + 1),
            }
            for rank, i in enumerate(order)
            if gene_scores[i] > 1e-12
        ]

        results.append({
            "arrow_index": int(arrow_idx),
            "kind": attr["kind"],
            "top_genes": top_genes,
            "gene_scores": gene_scores,
        })

    return results


def marker_gene_enrichment(
    arrow_gene_scores: np.ndarray,
    marker_genes: Dict[str, List[str]],
    hvg_names: List[str],
    top_n: int = 100,
) -> Dict[str, Dict]:
    """Test whether top-scoring genes for an arrow enrich known marker sets.

    Uses a one-tailed hypergeometric test for each cell-type marker set.

    Parameters
    ----------
    arrow_gene_scores : ndarray (n_hvgs,)
        Per-gene scores for a single arrow (from ``map_arrows_to_genes``).
    marker_genes : dict
        Maps cell type name -> list of marker gene symbols.
    hvg_names : list of str
        Full list of HVG names (the background population).
    top_n : int
        Number of top-scoring genes to treat as the "selected" set.

    Returns
    -------
    enrichment : dict
        Maps cell type -> dict with ``p_value``, ``odds_ratio``, ``overlap``,
        ``selected``, ``markers_in_background``.
    """
    arrow_gene_scores = np.asarray(arrow_gene_scores)
    n_background = len(hvg_names)
    hvg_set = set(hvg_names)

    # Top-scoring genes for this arrow
    order = np.argsort(arrow_gene_scores)[::-1][:top_n]
    selected_genes = {hvg_names[i] for i in order if i < len(hvg_names)}
    k = len(selected_genes)

    results = {}
    for cell_type, markers in marker_genes.items():
        # Only markers present in the background
        markers_bg = [m for m in markers if m in hvg_set]
        K = len(markers_bg)
        if K == 0:
            continue

        overlap = len(selected_genes & set(markers_bg))

        # Hypergeometric test: population N, success states K, draws k, successes overlap
        # Survival function = P(X >= overlap)
        p_value = hypergeom.sf(overlap - 1, n_background, K, k)

        # Odds ratio
        a = overlap
        b = k - overlap
        c = K - overlap
        d = n_background - K - k + overlap
        if b > 0 and c > 0:
            odds_ratio = (a * d) / (b * c)
        else:
            odds_ratio = float("inf") if a > 0 else 0.0

        results[cell_type] = {
            "p_value": float(p_value),
            "odds_ratio": float(odds_ratio),
            "overlap": int(overlap),
            "selected": int(k),
            "markers_in_background": int(K),
            "marker_genes": markers_bg,
        }

    return results


def pathway_enrichment(
    arrow_gene_scores: np.ndarray,
    pathways: Dict[str, List[str]],
    hvg_names: List[str],
    top_n: int = 100,
) -> Dict[str, Dict]:
    """Over-representation analysis of top arrow genes against pathways.

    Identical statistical framework to ``marker_gene_enrichment`` but
    operates on generic pathway / gene-set databases (e.g. GO, KEGG,
    MSigDB hallmarks).

    Parameters
    ----------
    arrow_gene_scores : ndarray (n_hvgs,)
    pathways : dict
        Maps pathway name -> list of gene symbols.
    hvg_names : list of str
    top_n : int

    Returns
    -------
    enrichment : dict
        Maps pathway name -> dict with ``p_value``, ``odds_ratio``,
        ``overlap``, ``selected``, ``pathway_genes_in_background``.
    """
    # Same math, different labels
    return marker_gene_enrichment(arrow_gene_scores, pathways, hvg_names, top_n)


def validate_against_markers(
    afe,
    pca_loadings: np.ndarray,
    hvg_names: List[str],
    marker_genes: Dict[str, List[str]],
    top_n_genes: int = 100,
) -> List[Dict]:
    """Full pipeline: map arrows to genes, then test marker enrichment.

    Returns a structured report suitable for tables or JSON export.
    """
    arrow_maps = map_arrows_to_genes(afe, pca_loadings, hvg_names, top_n_genes=top_n_genes)

    reports = []
    for am in arrow_maps:
        enrichment = marker_gene_enrichment(
            am["gene_scores"],
            marker_genes,
            hvg_names,
            top_n=top_n_genes,
        )

        # Sort by p-value
        sorted_enrichment = sorted(
            enrichment.items(),
            key=lambda x: x[1]["p_value"],
        )

        top_hits = [
            {
                "cell_type": cell_type,
                "p_value": info["p_value"],
                "odds_ratio": info["odds_ratio"],
                "overlap": info["overlap"],
            }
            for cell_type, info in sorted_enrichment[:5]
        ]

        reports.append({
            "arrow_index": am["arrow_index"],
            "kind": am["kind"],
            "top_genes": [g["gene"] for g in am["top_genes"][:10]],
            "enrichment": top_hits,
        })

    return reports


def arrow_gene_program_summary(
    validation_reports: List[Dict],
) -> Dict:
    """Aggregate arrow-to-marker validation across all arrows.

    Returns summary statistics: how many arrows significantly enriched
    at least one marker set, median best p-value per arrow, etc.
    """
    n_arrows = len(validation_reports)
    if n_arrows == 0:
        return {}

    sig_count = 0
    best_pvals = []
    for rep in validation_reports:
        hits = rep.get("enrichment", [])
        if not hits:
            best_pvals.append(1.0)
            continue
        best_p = min(h["p_value"] for h in hits)
        best_pvals.append(best_p)
        if best_p < 0.05:
            sig_count += 1

    return {
        "n_arrows": n_arrows,
        "arrows_with_significant_enrichment": sig_count,
        "fraction_significant": sig_count / n_arrows if n_arrows > 0 else 0.0,
        "median_best_p_value": float(np.median(best_pvals)),
        "mean_best_p_value": float(np.mean(best_pvals)),
        "best_overall_p_value": float(min(best_pvals)) if best_pvals else 1.0,
    }
