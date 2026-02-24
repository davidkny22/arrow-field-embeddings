"""API routes for the AFE Viewer server."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .engine import engine

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class RunAFERequest(BaseModel):
    dataset: str = Field(..., description="Built-in dataset name (swiss_roll, mnist, tabula_muris)")
    n_arrows: int = Field(default=2, ge=1, le=100, description="Number of arrows per point")
    encoding_mode: str = Field(default="direct", description="Encoding mode: direct, pca, or adaptive")


class NeighborsRequest(BaseModel):
    point_index: int = Field(..., ge=0, description="Index of the query point")
    k: int = Field(default=10, ge=1, le=200, description="Number of neighbours")


class NeighborsResponse(BaseModel):
    indices: list[int]
    distances: list[float]


class ReconstructRequest(BaseModel):
    point_index: int = Field(..., ge=0, description="Index of the point to reconstruct")


class ReconstructResponse(BaseModel):
    original: list[float]
    reconstructed: list[float]
    spatial_only: list[float]
    per_dim_error: list[float]


class HealthResponse(BaseModel):
    status: str
    datasets: list[str]


# ---------------------------------------------------------------------------
# Dataset loaders -- lazily import from benchmarks
# ---------------------------------------------------------------------------

_DATASET_LOADERS: dict[str, tuple] | None = None


def _get_loaders() -> dict:
    """Return mapping of dataset name -> (load_fn, kwargs)."""
    global _DATASET_LOADERS
    if _DATASET_LOADERS is not None:
        return _DATASET_LOADERS

    from benchmarks.datasets import load_swiss_roll, load_mnist, load_tabula_muris

    _DATASET_LOADERS = {
        "swiss_roll": (load_swiss_roll, {"n_samples": 3000}),
        "mnist": (load_mnist, {"n_samples": 10000}),
        "tabula_muris": (load_tabula_muris, {}),
    }
    return _DATASET_LOADERS


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter()


def _get_preset_registry() -> list[dict]:
    """Lazily import the preset registry from main to avoid circular imports."""
    from .main import preset_registry
    return preset_registry


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check -- lists available preset IDs."""
    registry = _get_preset_registry()
    return HealthResponse(
        status="ok",
        datasets=[p["id"] for p in registry],
    )


@router.get("/presets")
async def list_presets():
    """Return the list of presets discovered from public/presets/index.json."""
    return _get_preset_registry()


@router.post("/run_afe")
async def run_afe(body: RunAFERequest):
    """Run Arrow Field Embedding on a built-in dataset.

    Returns a full AFEDataset JSON blob (same format as the viewer presets).
    """
    loaders = _get_loaders()
    if body.dataset not in loaders:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown dataset '{body.dataset}'. "
                   f"Available: {list(loaders.keys())}",
        )

    if body.encoding_mode not in ("direct", "pca", "adaptive"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid encoding_mode '{body.encoding_mode}'. "
                   f"Must be one of: direct, pca, adaptive",
        )

    load_fn, kwargs = loaders[body.dataset]
    try:
        X, labels = load_fn(**kwargs)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load dataset '{body.dataset}': {exc}",
        )

    try:
        result = engine.run_afe(
            X=X,
            labels=labels,
            n_arrows=body.n_arrows,
            encoding_mode=body.encoding_mode,
            dataset_name=body.dataset,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"AFE computation failed: {exc}",
        )

    return result


@router.post("/neighbors", response_model=NeighborsResponse)
async def neighbors(body: NeighborsRequest):
    """Return k-NN indices and distances for a point in the current dataset."""
    if engine.last_result is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded. Call /run_afe first.",
        )

    n_points = engine.last_result["n_points"]
    if body.point_index >= n_points:
        raise HTTPException(
            status_code=400,
            detail=f"point_index {body.point_index} out of range "
                   f"(dataset has {n_points} points)",
        )

    positions = np.array(
        engine.last_result["positions"], dtype=np.float32
    ).reshape(-1, 3)

    try:
        result = engine.get_neighbors(
            positions=positions,
            index=body.point_index,
            k=body.k,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return NeighborsResponse(**result)


@router.post("/reconstruct", response_model=ReconstructResponse)
async def reconstruct(body: ReconstructRequest):
    """Per-dimension reconstruction detail for a single point."""
    if engine.afe is None or engine.X is None:
        raise HTTPException(
            status_code=400,
            detail="No AFE result available. Call /run_afe first.",
        )

    if body.point_index >= len(engine.X):
        raise HTTPException(
            status_code=400,
            detail=f"point_index {body.point_index} out of range "
                   f"(dataset has {len(engine.X)} points)",
        )

    try:
        result = engine.get_reconstruction(index=body.point_index)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ReconstructResponse(**result)
