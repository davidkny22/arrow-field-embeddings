"""Arrow attribution helpers.

The helpers in this module expose how arrows map back to residual dimensions,
PCA components, or adaptive groups. They are intentionally post-hoc: they
describe a fitted ``ArrowFieldEmbedding`` without changing its representation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


CHANNEL_NAMES = ("azimuth", "elevation", "magnitude")


def _feature_name(feature_names: Optional[List[str]], dim_idx: int) -> str:
    if feature_names is not None and dim_idx < len(feature_names):
        return str(feature_names[dim_idx])
    return f"dim_{dim_idx}"


def _dimension_record(feature_names: Optional[List[str]], dim_idx: int) -> Dict:
    return {
        "original_dim": int(dim_idx),
        "feature_name": _feature_name(feature_names, int(dim_idx)),
    }


def _top_loading_features(
    loading: np.ndarray,
    residual_dims: List[int],
    feature_names: Optional[List[str]],
    top_n: int,
) -> List[Dict]:
    if len(loading) == 0:
        return []
    order = np.argsort(np.abs(loading))[::-1][:top_n]
    records = []
    for residual_pos in order:
        if residual_pos >= len(residual_dims):
            continue
        original_dim = int(residual_dims[residual_pos])
        records.append({
            "residual_index": int(residual_pos),
            "original_dim": original_dim,
            "feature_name": _feature_name(feature_names, original_dim),
            "loading": float(loading[residual_pos]),
            "abs_loading": float(abs(loading[residual_pos])),
        })
    return records


def get_arrow_attributions(
    afe,
    feature_names: Optional[List[str]] = None,
    top_n: int = 8,
) -> List[Dict]:
    """Return structured attribution metadata for each fitted arrow.

    Parameters
    ----------
    afe
        Fitted ``ArrowFieldEmbedding`` instance.
    feature_names
        Optional names for original input dimensions.
    top_n
        Maximum number of loading-ranked features to include for PCA/adaptive
        summaries.
    """
    if hasattr(afe, "_check_fitted"):
        afe._check_fitted()

    gap_report = afe.get_gap_report()
    residual_dims = list(gap_report.get("residual_dims", []))
    encoder = getattr(afe, "_encoder", None)
    mode = getattr(afe, "encoding_mode", "unknown")
    n_arrows = int(getattr(afe, "n_arrows", 0))

    if mode == "direct":
        return _direct_attributions(n_arrows, residual_dims, feature_names)
    if mode == "pca":
        return _pca_attributions(n_arrows, residual_dims, encoder, feature_names, top_n)
    if mode == "adaptive":
        return _adaptive_attributions(
            n_arrows, residual_dims, encoder, feature_names, top_n
        )
    return [_empty_arrow(a, mode) for a in range(n_arrows)]


def arrow_dim_labels(afe, feature_names: Optional[List[str]] = None) -> List[List[str]]:
    """Return compact per-arrow labels for viewer/export display."""
    labels = []
    for attr in get_arrow_attributions(afe, feature_names=feature_names):
        if attr["kind"] == "direct":
            names = [
                ch["feature_name"]
                for ch in attr["channels"]
                if ch.get("original_dim") is not None
            ]
            labels.append(names or [attr["label"]])
        elif attr["kind"] == "pca_component":
            labels.append([attr["label"]])
        elif attr["kind"] == "adaptive_group":
            names = [d["feature_name"] for d in attr.get("dimensions", [])]
            labels.append(names or [attr["label"]])
        else:
            labels.append([attr["label"]])
    return labels


def _empty_arrow(arrow_index: int, mode: str) -> Dict:
    return {
        "arrow_index": int(arrow_index),
        "kind": "empty",
        "encoding_mode": mode,
        "label": f"arrow_{arrow_index}",
        "channels": [],
        "dimensions": [],
    }


def _direct_attributions(
    n_arrows: int,
    residual_dims: List[int],
    feature_names: Optional[List[str]],
) -> List[Dict]:
    attributions = []
    for arrow_index in range(n_arrows):
        channels = []
        dimensions = []
        for channel_index, channel_name in enumerate(CHANNEL_NAMES):
            residual_index = arrow_index * 3 + channel_index
            if residual_index < len(residual_dims):
                original_dim = int(residual_dims[residual_index])
                dim_record = _dimension_record(feature_names, original_dim)
                dimensions.append(dim_record)
                channels.append({
                    "channel_index": int(channel_index),
                    "channel": channel_name,
                    "residual_index": int(residual_index),
                    **dim_record,
                })
            else:
                channels.append({
                    "channel_index": int(channel_index),
                    "channel": channel_name,
                    "residual_index": None,
                    "original_dim": None,
                    "feature_name": None,
                })
        attributions.append({
            "arrow_index": int(arrow_index),
            "kind": "direct",
            "encoding_mode": "direct",
            "label": f"direct_{arrow_index}",
            "channels": channels,
            "dimensions": dimensions,
        })
    return attributions


def _pca_attributions(
    n_arrows: int,
    residual_dims: List[int],
    encoder,
    feature_names: Optional[List[str]],
    top_n: int,
) -> List[Dict]:
    pca = getattr(encoder, "_pca", None)
    attributions = []
    for arrow_index in range(n_arrows):
        if pca is None or arrow_index >= getattr(pca, "n_components_", 0):
            attributions.append(_empty_arrow(arrow_index, "pca"))
            continue
        explained = float(pca.explained_variance_ratio_[arrow_index])
        loading = pca.components_[arrow_index]
        attributions.append({
            "arrow_index": int(arrow_index),
            "kind": "pca_component",
            "encoding_mode": "pca",
            "component_index": int(arrow_index),
            "label": f"PC{arrow_index + 1} ({explained:.1%} var)",
            "variance_explained": explained,
            "channels": [
                {"channel_index": 0, "channel": "azimuth", "source": "component_loading_angle"},
                {"channel_index": 1, "channel": "elevation", "source": "component_loading_angle"},
                {"channel_index": 2, "channel": "magnitude", "source": "component_score"},
            ],
            "top_features": _top_loading_features(
                loading, residual_dims, feature_names, top_n
            ),
        })
    return attributions


def _adaptive_attributions(
    n_arrows: int,
    residual_dims: List[int],
    encoder,
    feature_names: Optional[List[str]],
    top_n: int,
) -> List[Dict]:
    groups = getattr(encoder, "_groups", None) or []
    group_pcas = getattr(encoder, "_group_pcas", None) or []
    attributions = []
    for arrow_index in range(n_arrows):
        if arrow_index >= len(groups):
            attributions.append(_empty_arrow(arrow_index, "adaptive"))
            continue

        group = list(groups[arrow_index])
        dimensions = []
        for residual_index in group:
            if residual_index < len(residual_dims):
                original_dim = int(residual_dims[residual_index])
                dimensions.append({
                    "residual_index": int(residual_index),
                    **_dimension_record(feature_names, original_dim),
                })

        pca = group_pcas[arrow_index] if arrow_index < len(group_pcas) else None
        attr = {
            "arrow_index": int(arrow_index),
            "kind": "adaptive_group",
            "encoding_mode": "adaptive",
            "group_index": int(arrow_index),
            "label": f"group_{arrow_index}",
            "channels": [
                {"channel_index": 0, "channel": "azimuth", "source": "group_loading_angle"},
                {"channel_index": 1, "channel": "elevation", "source": "group_score_or_angle"},
                {"channel_index": 2, "channel": "magnitude", "source": "group_score_or_value"},
            ],
            "dimensions": dimensions,
        }
        if pca is not None and getattr(pca, "components_", None) is not None:
            local_residual_dims = [
                int(residual_dims[i]) for i in group if i < len(residual_dims)
            ]
            attr["variance_explained"] = [
                float(v) for v in pca.explained_variance_ratio_
            ]
            attr["top_features"] = _top_loading_features(
                pca.components_[0], local_residual_dims, feature_names, top_n
            )
        attributions.append(attr)
    return attributions


__all__ = ["arrow_dim_labels", "get_arrow_attributions"]
