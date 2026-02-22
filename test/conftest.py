"""Shared test fixtures for AFE."""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def synthetic_data():
    """Simple 10-dimensional synthetic dataset with known structure."""
    rng = np.random.RandomState(42)
    n = 200
    d = 10

    # Create data with correlated groups:
    # Dims 0-2: one cluster of correlated features
    # Dims 3-5: another cluster
    # Dims 6-9: independent features
    base1 = rng.randn(n, 1)
    base2 = rng.randn(n, 1)

    X = np.empty((n, d), dtype=np.float32)
    X[:, 0] = (base1 + rng.randn(n, 1) * 0.1).ravel()
    X[:, 1] = (base1 + rng.randn(n, 1) * 0.2).ravel()
    X[:, 2] = (base1 + rng.randn(n, 1) * 0.15).ravel()
    X[:, 3] = (base2 + rng.randn(n, 1) * 0.1).ravel()
    X[:, 4] = (base2 + rng.randn(n, 1) * 0.2).ravel()
    X[:, 5] = (base2 + rng.randn(n, 1) * 0.15).ravel()
    for i in range(6, d):
        X[:, i] = rng.randn(n).astype(np.float32)

    labels = (X[:, 0] > 0).astype(int)
    return X, labels


@pytest.fixture(scope="session")
def swiss_roll_data():
    """Swiss Roll dataset."""
    from sklearn.datasets import make_swiss_roll
    X, t = make_swiss_roll(n_samples=500, noise=0.1, random_state=42)
    return X.astype(np.float32), t


@pytest.fixture
def manual_embedding(synthetic_data):
    """Pre-computed 3D embedding for manual backend tests."""
    X, _ = synthetic_data
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3, random_state=42)
    return pca.fit_transform(X).astype(np.float32)
