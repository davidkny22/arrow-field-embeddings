"""Tests for benchmark reproducibility helpers."""

from pathlib import Path

import numpy as np

from afe.reproducibility import get_or_compute_spatial_embedding


def _cache_dir(name: str) -> Path:
    path = Path(".benchmarks") / "test_tmp" / name
    path.mkdir(parents=True, exist_ok=True)
    for child in path.glob("*.npz"):
        child.unlink()
    return path


def test_spatial_cache_reuses_identical_run_identity():
    cache_dir = _cache_dir("same_identity")
    calls = []

    def compute():
        calls.append("compute")
        return np.ones((4, 3), dtype=np.float32)

    first, first_path, first_hit = get_or_compute_spatial_embedding(
        X=np.zeros((4, 2), dtype=np.float32),
        dataset="dataset",
        backend="tsne",
        seed=0,
        backend_params={"perplexity": 30},
        preprocessing_version="dataset:loader-v1",
        cache_dir=cache_dir,
        compute_fn=compute,
    )
    second, second_path, second_hit = get_or_compute_spatial_embedding(
        X=np.zeros((4, 2), dtype=np.float32),
        dataset="dataset",
        backend="tsne",
        seed=0,
        backend_params={"perplexity": 30},
        preprocessing_version="dataset:loader-v1",
        cache_dir=cache_dir,
        compute_fn=compute,
    )

    assert calls == ["compute"]
    assert first_path == second_path
    assert first_hit is False
    assert second_hit is True
    np.testing.assert_array_equal(first, second)


def test_spatial_cache_separates_backend_params_and_preprocessing():
    cache_dir = _cache_dir("identity_changes")
    calls = []

    def compute_one():
        calls.append("one")
        return np.ones((4, 3), dtype=np.float32)

    def compute_two():
        calls.append("two")
        return np.full((4, 3), 2.0, dtype=np.float32)

    _, first_path, _ = get_or_compute_spatial_embedding(
        X=np.zeros((4, 2), dtype=np.float32),
        dataset="dataset",
        backend="tsne",
        seed=0,
        backend_params={"perplexity": 30},
        preprocessing_version="dataset:loader-v1",
        cache_dir=cache_dir,
        compute_fn=compute_one,
    )
    changed_params, second_path, second_hit = get_or_compute_spatial_embedding(
        X=np.zeros((4, 2), dtype=np.float32),
        dataset="dataset",
        backend="tsne",
        seed=0,
        backend_params={"perplexity": 50},
        preprocessing_version="dataset:loader-v1",
        cache_dir=cache_dir,
        compute_fn=compute_two,
    )
    changed_preproc, third_path, third_hit = get_or_compute_spatial_embedding(
        X=np.zeros((4, 2), dtype=np.float32),
        dataset="dataset",
        backend="tsne",
        seed=0,
        backend_params={"perplexity": 30},
        preprocessing_version="dataset:loader-v2",
        cache_dir=cache_dir,
        compute_fn=compute_two,
    )

    assert calls == ["one", "two", "two"]
    assert first_path != second_path
    assert first_path != third_path
    assert second_hit is False
    assert third_hit is False
    assert changed_params[0, 0] == 2.0
    assert changed_preproc[0, 0] == 2.0
