"""3D interactive visualization for Arrow Field Embeddings."""

import numpy as np
from typing import Optional, List, Union

try:
    import plotly.graph_objects as go

    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

# Default arrow color palette (qualitative, colorblind-friendly)
DEFAULT_ARROW_COLORS = [
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#ff7f00",  # orange
    "#a65628",  # brown
    "#f781bf",  # pink
    "#999999",  # grey
]


def _arrows_to_cartesian(
    spatial: np.ndarray, arrows: np.ndarray, scale: float
) -> np.ndarray:
    """Convert (azimuth, elevation, magnitude) to 3D direction vectors.

    Parameters
    ----------
    spatial : ndarray (n, 3) — point positions
    arrows : ndarray (n, k, 3) — (azimuth, elevation, magnitude) per arrow
    scale : float — global arrow length scaling

    Returns
    -------
    endpoints : ndarray (n, k, 3) — Cartesian direction offsets
    """
    theta = arrows[:, :, 0]  # azimuth
    phi = arrows[:, :, 1]    # elevation
    mag = arrows[:, :, 2]    # magnitude

    # Spherical to Cartesian
    cos_phi = np.cos(phi)
    dx = mag * cos_phi * np.cos(theta) * scale
    dy = mag * cos_phi * np.sin(theta) * scale
    dz = mag * np.sin(phi) * scale

    return np.stack([dx, dy, dz], axis=-1).astype(np.float32)


def plot_afe(
    spatial: np.ndarray,
    arrows: np.ndarray,
    labels: Optional[np.ndarray] = None,
    arrow_scale: float = 1.0,
    arrow_colors: Optional[List[str]] = None,
    point_size: float = 3.0,
    colormap: str = "Spectral",
    title: str = "Arrow Field Embedding",
    opacity: float = 0.8,
    show_arrows: Union[bool, List[int]] = True,
) -> "go.Figure":
    """Render an AFE result as an interactive 3D plotly figure.

    Parameters
    ----------
    spatial : ndarray (n, 3)
    arrows : ndarray (n, k, 3) — (azimuth, elevation, magnitude)
    labels : ndarray (n,), optional — point labels for coloring
    arrow_scale : float — global arrow length scaling
    arrow_colors : list of str, optional — per-arrow-index colors
    point_size : float
    colormap : str — plotly colorscale name
    title : str
    opacity : float
    show_arrows : bool or list of int — True=all, list=specified indices

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    if not _HAS_PLOTLY:
        raise ImportError("plotly is required for visualization. pip install plotly")

    n, k, _ = arrows.shape
    if arrow_colors is None:
        arrow_colors = DEFAULT_ARROW_COLORS

    # Auto-scale arrows relative to spatial extent
    spatial_range = np.ptp(spatial, axis=0).mean()  # average axis spread
    auto_sizeref = spatial_range * 0.03 * arrow_scale

    fig = go.Figure()

    # Points
    marker_kwargs = dict(size=point_size, opacity=opacity)
    if labels is not None:
        marker_kwargs["color"] = labels
        marker_kwargs["colorscale"] = colormap

    fig.add_trace(
        go.Scatter3d(
            x=spatial[:, 0],
            y=spatial[:, 1],
            z=spatial[:, 2],
            mode="markers",
            marker=marker_kwargs,
            name="Points",
        )
    )

    # Arrows
    if show_arrows is False:
        pass
    else:
        arrow_indices = list(range(k)) if show_arrows is True else show_arrows
        directions = _arrows_to_cartesian(spatial, arrows, arrow_scale)

        for ai in arrow_indices:
            if ai >= k:
                continue
            color = arrow_colors[ai % len(arrow_colors)]

            # Use Cone trace for arrows
            fig.add_trace(
                go.Cone(
                    x=spatial[:, 0],
                    y=spatial[:, 1],
                    z=spatial[:, 2],
                    u=directions[:, ai, 0],
                    v=directions[:, ai, 1],
                    w=directions[:, ai, 2],
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    sizemode="absolute",
                    sizeref=auto_sizeref,
                    name=f"Arrow {ai}",
                    opacity=0.7,
                )
            )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        showlegend=True,
    )

    return fig


def plot_spatial_information_gap(gap_report: dict) -> "go.Figure":
    """Visualize spatial information gap analysis as a correlation heatmap."""
    if not _HAS_PLOTLY:
        raise ImportError("plotly is required for visualization. pip install plotly")

    corr = gap_report["correlation_matrix"]
    fig = go.Figure(
        data=go.Heatmap(
            z=np.abs(corr).T,
            x=[f"Dim {i}" for i in range(corr.shape[0])],
            y=["X", "Y", "Z"],
            colorscale="Viridis",
            colorbar=dict(title="|Correlation|"),
        )
    )
    fig.update_layout(
        title="Spatial Information Gap: |Correlation| of HD Dims with Spatial Coords",
        xaxis_title="Original Dimension",
        yaxis_title="Spatial Coordinate",
    )
    return fig


__all__ = ["plot_afe", "plot_spatial_information_gap"]
