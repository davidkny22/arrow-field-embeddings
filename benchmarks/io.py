"""JSONL I/O, embedding persistence, and metadata building for benchmarks."""

import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, Optional

from afe.reproducibility import (
    RESULT_SCHEMA_VERSION,
    build_run_metadata,
)

EMBEDDINGS_DIR = Path(__file__).parent / "embeddings"


def _json_convert(obj):
    """Handle numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def load_completed(output_path):
    """Load completed run keys from JSONL file."""
    completed = set()
    path = Path(output_path)
    if path.exists():
        with open(path) as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if record.get("type") == "significance":
                        continue
                    if record.get("result_schema_version") != RESULT_SCHEMA_VERSION:
                        raise ValueError(
                            f"{path}:{line_no} is not a {RESULT_SCHEMA_VERSION} "
                            "benchmark row. Start a new output file for paper benchmark runs."
                        )
                    key = (record["dataset"], record["method"], record["seed"])
                    completed.add(key)
                except (json.JSONDecodeError, KeyError):
                    continue  # Skip corrupt lines from crashes
    return completed


def load_all_results(output_path):
    """Load all benchmark results from JSONL (excludes significance records)."""
    results = []
    path = Path(output_path)
    if path.exists():
        with open(path) as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if record.get("type") != "significance":
                        if record.get("result_schema_version") != RESULT_SCHEMA_VERSION:
                            raise ValueError(
                                f"{path}:{line_no} is not a {RESULT_SCHEMA_VERSION} "
                                "benchmark row. Start a new output file for paper benchmark runs."
                            )
                        results.append(record)
                except json.JSONDecodeError:
                    continue
    return results


def append_result(output_path, record):
    """Append a single result record to JSONL, flushed to disk immediately."""
    with open(output_path, "a") as f:
        f.write(json.dumps(record, default=_json_convert) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _save_embedding(dataset_name, backend_name, encoding_mode, n_arrows, seed,
                     X_high, Y_spatial, arrows=None):
    """Save embeddings to disk for post-hoc metric computation."""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    if encoding_mode is None:
        fname = f"{dataset_name}__{backend_name}__baseline__seed{seed}.npz"
    else:
        fname = f"{dataset_name}__{backend_name}__{encoding_mode}_{n_arrows}arr__seed{seed}.npz"
    path = EMBEDDINGS_DIR / fname
    save_dict = {"X_high": X_high, "Y_spatial": Y_spatial}
    if arrows is not None:
        save_dict["arrows"] = arrows
    np.savez_compressed(path, **save_dict)


def _preprocessing_version(dataset_name):
    return f"{dataset_name}:loader-v1"


def _metadata_for_row(
    dataset_name,
    backend_name,
    backend_params,
    seed,
    encoding_mode,
    n_arrows,
    n_features,
    package_versions,
    machine_info,
):
    return build_run_metadata(
        dataset=dataset_name,
        preprocessing_version=_preprocessing_version(dataset_name),
        backend=backend_name,
        backend_params=backend_params,
        seed=seed,
        encoding_mode=encoding_mode,
        n_arrows=n_arrows,
        arrow_capacity={
            "spatial_dims": 3,
            "dims_per_arrow": 3,
            "arrow_channel_capacity": int((n_arrows or 0) * 3),
            "total_representation_dims": int(3 + (n_arrows or 0) * 3),
            "n_features": int(n_features),
        },
        package_versions=package_versions,
        machine_info=machine_info,
    )
