"""Reproducibility helpers for AFE benchmark records."""

from __future__ import annotations

import json
import hashlib
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np


RESULT_SCHEMA_VERSION = "afe-benchmark-v3"
DEFAULT_PACKAGES = (
    "afe",
    "numpy",
    "scipy",
    "scikit-learn",
    "pacmap",
    "umap-learn",
    "trimap",
    "openTSNE",
)


def collect_package_versions(packages: Iterable[str] = DEFAULT_PACKAGES) -> Dict[str, Optional[str]]:
    """Collect installed package versions, using None for missing packages."""
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def collect_machine_info() -> Dict[str, str]:
    """Return lightweight, JSON-serializable machine/runtime metadata."""
    return {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def build_run_metadata(
    *,
    dataset: str,
    preprocessing_version: str,
    backend: str,
    backend_params: Optional[Dict] = None,
    seed: Optional[int] = None,
    encoding_mode: Optional[str] = None,
    n_arrows: Optional[int] = None,
    arrow_capacity: Optional[Dict] = None,
    package_versions: Optional[Dict[str, Optional[str]]] = None,
    machine_info: Optional[Dict[str, str]] = None,
) -> Dict:
    """Build the common metadata block required for paper benchmark rows."""
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "dataset": dataset,
        "preprocessing_version": preprocessing_version,
        "backend": backend,
        "backend_params": backend_params or {},
        "seed": seed,
        "encoding_mode": encoding_mode,
        "n_arrows": n_arrows,
        "arrow_capacity": arrow_capacity or {},
        "package_versions": package_versions or collect_package_versions(),
        "machine_info": machine_info or collect_machine_info(),
    }


def add_run_metadata(record: Dict, **metadata_kwargs) -> Dict:
    """Return a copy of a benchmark row with common benchmark metadata attached."""
    enriched = dict(record)
    enriched.update(build_run_metadata(**metadata_kwargs))
    return enriched


def _identity_fingerprint(identity: Dict) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def spatial_embedding_filename(
    dataset: str,
    backend: str,
    seed: int,
    backend_params: Optional[Dict] = None,
    preprocessing_version: Optional[str] = None,
    result_schema_version: str = RESULT_SCHEMA_VERSION,
) -> str:
    """Stable filename for cached spatial coordinates."""
    safe_dataset = str(dataset).replace("/", "_").replace("\\", "_")
    safe_backend = str(backend).replace("/", "_").replace("\\", "_")
    identity = {
        "backend": backend,
        "backend_params": backend_params or {},
        "preprocessing_version": preprocessing_version,
        "result_schema_version": result_schema_version,
        "seed": seed,
    }
    fingerprint = _identity_fingerprint(identity)
    return f"{safe_dataset}__{safe_backend}__seed{seed}__{fingerprint}__spatial.npz"


def save_spatial_embedding(path: str | Path, spatial: np.ndarray, metadata_dict: Optional[Dict] = None) -> Path:
    """Save fixed spatial coordinates plus JSON metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(metadata_dict or {}, sort_keys=True)
    np.savez_compressed(path, spatial=np.asarray(spatial, dtype=np.float32), metadata=metadata_json)
    return path


def load_spatial_embedding(path: str | Path) -> tuple[np.ndarray, Dict]:
    """Load fixed spatial coordinates and metadata."""
    with np.load(Path(path), allow_pickle=False) as data:
        spatial = np.asarray(data["spatial"], dtype=np.float32)
        metadata_json = str(data["metadata"]) if "metadata" in data.files else "{}"
    return spatial, json.loads(metadata_json)


def get_or_compute_spatial_embedding(
    *,
    X: np.ndarray,
    dataset: str,
    backend: str,
    seed: int,
    backend_params: Optional[Dict],
    preprocessing_version: str,
    cache_dir: Optional[str | Path],
    compute_fn,
) -> tuple[np.ndarray, Optional[Path], bool]:
    """Load a cached spatial embedding or compute and cache it.

    Returns ``(spatial, cache_path, cache_hit)``.
    """
    if cache_dir is None:
        return np.asarray(compute_fn(), dtype=np.float32), None, False

    expected_metadata = {
        "dataset": dataset,
        "backend": backend,
        "backend_params": backend_params or {},
        "preprocessing_version": preprocessing_version,
        "seed": seed,
        "result_schema_version": RESULT_SCHEMA_VERSION,
    }
    cache_path = Path(cache_dir) / spatial_embedding_filename(
        dataset,
        backend,
        seed,
        backend_params=backend_params,
        preprocessing_version=preprocessing_version,
    )
    if cache_path.exists():
        spatial, metadata_dict = load_spatial_embedding(cache_path)
        if metadata_dict == expected_metadata:
            return spatial, cache_path, True

    spatial = np.asarray(compute_fn(), dtype=np.float32)
    save_spatial_embedding(cache_path, spatial, expected_metadata)
    return spatial, cache_path, False


__all__ = [
    "DEFAULT_PACKAGES",
    "RESULT_SCHEMA_VERSION",
    "add_run_metadata",
    "build_run_metadata",
    "collect_machine_info",
    "collect_package_versions",
    "get_or_compute_spatial_embedding",
    "load_spatial_embedding",
    "save_spatial_embedding",
    "spatial_embedding_filename",
]
