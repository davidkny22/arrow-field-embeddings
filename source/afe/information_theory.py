"""Information-theoretic analysis for Arrow Field Embeddings.

Provides mutual information estimates and rate-distortion curves to quantify
how much high-dimensional information the spatial layout captures versus how
much the arrow field recovers.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
from scipy.linalg import eigh
from scipy.special import digamma

_logger = logging.getLogger(__name__)

__all__ = [
    "gaussian_mutual_information",
    "knn_mutual_information",
    "information_gain",
    "rate_distortion_curve",
    "negative_control_shuffled",
    "negative_control_random_spatial",
    "negative_control_gaussian_arrows",
]


def gaussian_mutual_information(
    X: np.ndarray,
    Y: np.ndarray,
) -> float:
    """Gaussian MI proxy: I(X; Y) under joint Gaussian assumption.

    Uses the standard formula::

        I(X; Y) = 0.5 * log( det(Sigma_X) * det(Sigma_Y) / det(Sigma_XY) )

    Data is standardized internally (zero mean, unit variance per dimension)
    to make MI comparable across datasets.  Small eigenvalues are clipped
    to avoid numerical issues.

    Parameters
    ----------
    X : ndarray (n, d_x)
    Y : ndarray (n, d_y)

    Returns
    -------
    mi : float
        Estimated mutual information in nats.  Returns 0.0 if either
        variable is constant.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    # Standardize for comparability
    X = _standardize(X)
    Y = _standardize(Y)

    n = X.shape[0]
    if n == 0:
        return 0.0

    # Covariances
    cov_x = np.cov(X, rowvar=False, bias=True)
    cov_y = np.cov(Y, rowvar=False, bias=True)
    xy = np.hstack([X, Y])
    cov_xy = np.cov(xy, rowvar=False, bias=True)

    # Handle 1D case
    if cov_x.ndim == 0:
        cov_x = np.array([[float(cov_x)]])
    if cov_y.ndim == 0:
        cov_y = np.array([[float(cov_y)]])

    # Clip tiny eigenvalues for numerical stability
    eps = 1e-10
    det_x = _posdet(cov_x, eps)
    det_y = _posdet(cov_y, eps)
    det_xy = _posdet(cov_xy, eps)

    if det_x <= eps or det_y <= eps or det_xy <= eps:
        return 0.0

    mi = 0.5 * (np.log(det_x) + np.log(det_y) - np.log(det_xy))
    return max(0.0, float(mi))


def _standardize(Z: np.ndarray) -> np.ndarray:
    """Standardize columns to zero mean, unit variance."""
    mean = np.mean(Z, axis=0, keepdims=True)
    std = np.std(Z, axis=0, keepdims=True)
    std[std < 1e-10] = 1.0
    return (Z - mean) / std


def _posdet(mat: np.ndarray, eps: float) -> float:
    """Positive definite determinant via eigenvalue clipping."""
    eigvals = eigh(mat, eigvals_only=True)
    eigvals = np.maximum(eigvals, eps)
    return float(np.prod(eigvals))


def knn_mutual_information(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 5,
    subsample: Optional[int] = None,
    random_state: Optional[int] = None,
) -> float:
    """Kraskov-Stögbauer-Grassberger kNN MI estimator.

    This is a non-parametric MI estimate based on the average distance to
    the k-th nearest neighbor in the joint (X, Y) space and marginal spaces.
    It is more robust than the Gaussian proxy for non-Gaussian data but
    scales as O(n^2) and is therefore limited to moderate sample sizes.

    Parameters
    ----------
    X : ndarray (n, d_x)
    Y : ndarray (n, d_y)
    k : int
        Number of nearest neighbors (default 5).
    subsample : int or None
        If given, randomly subsample to this many points before estimation.
        Recommended for n > 5,000.
    random_state : int or None
        RNG seed for subsampling.

    Returns
    -------
    mi : float
        Estimated mutual information in nats.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    # Subsample if requested
    n = X.shape[0]
    if subsample is not None and subsample < n:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=subsample, replace=False)
        X = X[idx]
        Y = Y[idx]
        n = subsample

    if n < k + 1:
        return 0.0

    # Standardize each space
    X = _standardize(X)
    Y = _standardize(Y)

    # Joint space
    XY = np.hstack([X, Y])

    # Compute distances (using Chebyshev / L-infinity norm per KSG)
    dx = _knn_distances(X, k)
    dy = _knn_distances(Y, k)
    dxy = _knn_distances(XY, k)

    # KSG formula
    # I(X;Y) = psi(k) + psi(N) - mean(psi(nx + 1) + psi(ny + 1))
    # where nx, ny are counts within epsilon = 2 * dxy
    eps = 2.0 * dxy + 1e-12

    nx = _count_within_radius(X, eps)
    ny = _count_within_radius(Y, eps)

    mi = (
        digamma(k)
        + digamma(n)
        - np.mean(digamma(nx + 1) + digamma(ny + 1))
    )
    return max(0.0, float(mi))


def _knn_distances(Z: np.ndarray, k: int) -> np.ndarray:
    """Distance to k-th nearest neighbor for each point (L-infinity norm)."""
    n = Z.shape[0]
    dists = np.zeros(n, dtype=np.float64)
    for i in range(n):
        # L-infinity distance to all other points
        d = np.max(np.abs(Z - Z[i]), axis=1)
        d[i] = np.inf  # exclude self
        dists[i] = np.partition(d, k - 1)[k - 1]
    return dists


def _count_within_radius(Z: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Count points within L-infinity radius for each point (including self)."""
    n = Z.shape[0]
    counts = np.zeros(n, dtype=np.int64)
    for i in range(n):
        d = np.max(np.abs(Z - Z[i]), axis=1)
        counts[i] = np.sum(d <= radius[i])
    return counts


def information_gain(
    X: np.ndarray,
    spatial: np.ndarray,
    arrows: np.ndarray,
    estimator: str = "gaussian",
    **estimator_kwargs,
) -> Dict[str, float]:
    """Quantify how much arrows add beyond the spatial layout.

    Computes MI(X; spatial) and MI(X; [spatial, arrows]) and returns the
    difference as "information_gain".  Also reports the fractional gain
    relative to the total MI(X; X) (which is the differential entropy for
    the Gaussian estimator).

    Parameters
    ----------
    X : ndarray (n, d)
        Original high-dimensional data.
    spatial : ndarray (n, 3)
        3D spatial embedding.
    arrows : ndarray (n, k, 3)
        Arrow field in spherical coordinates.  Flattened internally to
        (n, k*3).
    estimator : {"gaussian", "knn"}
        Which MI estimator to use.
    **estimator_kwargs
        Passed to the estimator (e.g. ``k=5`` for knn).

    Returns
    -------
    result : dict with keys:
        mi_spatial : MI(X; spatial)
        mi_full : MI(X; [spatial, arrows_flat])
        information_gain : mi_full - mi_spatial
        fractional_gain : information_gain / H(X)  (Gaussian only)
        estimator : name of estimator used
    """
    X = np.asarray(X, dtype=np.float64)
    spatial = np.asarray(spatial, dtype=np.float64)
    arrows = np.asarray(arrows, dtype=np.float64)

    # Flatten arrows
    arrows_flat = arrows.reshape(arrows.shape[0], -1)

    # Concatenate spatial + arrows
    full = np.hstack([spatial, arrows_flat])

    mi_fn = gaussian_mutual_information if estimator == "gaussian" else knn_mutual_information

    mi_spatial = mi_fn(X, spatial, **estimator_kwargs)
    mi_full = mi_fn(X, full, **estimator_kwargs)

    gain = max(0.0, mi_full - mi_spatial)

    result = {
        "mi_spatial": float(mi_spatial),
        "mi_full": float(mi_full),
        "information_gain": float(gain),
        "estimator": estimator,
    }

    if estimator == "gaussian":
        h_x = gaussian_mutual_information(X, X)  # H(X) for standardized Gaussian
        result["entropy_x"] = float(h_x)
        result["fractional_gain"] = float(gain / h_x) if h_x > 0 else 0.0
        result["spatial_fraction"] = float(mi_spatial / h_x) if h_x > 0 else 0.0

    return result


def rate_distortion_curve(
    X: np.ndarray,
    representations: List[Tuple[str, np.ndarray]],
    metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    metric_name: str = "mse",
) -> List[Dict]:
    """Empirical rate-distortion curve for a set of representations.

    Each representation is assigned a "rate" equal to its dimensionality
    and a "distortion" equal to the reconstruction error against X.  The
    curve can be compared against PCA truncation at the same dimensional
    budget to show whether AFE provides better rate-distortion efficiency.

    Parameters
    ----------
    X : ndarray (n, d)
        Original data (used as target for distortion).
    representations : list of (name, Z)
        Z is ndarray (n, m).  Name is a label like "spatial_3d",
        "afe_3d_9arr", "pca_50d".
    metric_fn : callable or None
        Function ``metric_fn(X, Z_recon) -> float``.  Default is MSE
        after linear least-squares reconstruction from Z to X.
    metric_name : str
        Label for the distortion metric.

    Returns
    -------
    curve : list of dicts, each with keys:
        name, rate, distortion, metric_name
    """
    if metric_fn is None:
        metric_fn = _reconstruction_mse

    curve = []
    for name, Z in representations:
        Z = np.asarray(Z, dtype=np.float64)
        rate = float(Z.shape[1])
        distortion = float(metric_fn(X, Z))
        curve.append({
            "name": name,
            "rate": rate,
            "distortion": distortion,
            "metric_name": metric_name,
        })

    return curve


def _reconstruction_mse(X: np.ndarray, Z: np.ndarray) -> float:
    """Linear least-squares reconstruction MSE from Z to X."""
    # Fit linear regression X ~ Z
    # coeffs = (Z^T Z)^-1 Z^T X
    Z = np.asarray(Z, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    # Add bias column
    Zb = np.hstack([np.ones((Z.shape[0], 1)), Z])
    coeffs, _, _, _ = np.linalg.lstsq(Zb, X, rcond=None)
    X_pred = Zb @ coeffs
    mse = np.mean((X - X_pred) ** 2)
    return float(mse)


def negative_control_shuffled(
    X: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Return a copy of X with features independently shuffled per column.

    Destroys all joint structure while preserving marginal distributions.
    """
    rng = np.random.default_rng(random_state)
    X_shuf = np.array(X, dtype=np.float64)
    for j in range(X_shuf.shape[1]):
        rng.shuffle(X_shuf[:, j])
    return X_shuf


def negative_control_random_spatial(
    n_points: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Return a random uniform 3D spatial embedding.

    Used as a negative-control spatial backend with no structure.
    """
    rng = np.random.default_rng(random_state)
    return rng.standard_normal((n_points, 3)).astype(np.float64)


def negative_control_gaussian_arrows(
    n_points: int,
    n_arrows: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Return random Gaussian arrows with no structure.

    Shape (n_points, n_arrows, 3) with spherical coordinates drawn from
    standard normal and then converted to proper ranges.
    """
    rng = np.random.default_rng(random_state)
    arrows = rng.standard_normal((n_points, n_arrows, 3)).astype(np.float64)
    # Normalize spherical coords: theta in [0, 2pi), phi in [-pi/2, pi/2]
    arrows[:, :, 0] = np.mod(arrows[:, :, 0], 2 * np.pi)
    arrows[:, :, 1] = np.arctan(arrows[:, :, 1])
    arrows[:, :, 2] = np.abs(arrows[:, :, 2])
    return arrows
