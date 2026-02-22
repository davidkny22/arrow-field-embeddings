"""Dataset loading for AFE benchmarks.

Adapted from continuous-pacmap-public/benchmarks/datasets.py.
"""

import numpy as np
from sklearn.datasets import fetch_openml, make_swiss_roll, make_s_curve
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"


def _download_file(url, dest):
    """Download a file from url to dest."""
    import urllib.request
    print(f"  Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")


def load_mnist(n_samples=10000):
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True,
                        parser='pandas', as_frame=False)
    if n_samples and n_samples < len(X):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), n_samples, replace=False)
        X, y = X[idx], y[idx]
    return X.astype(np.float32), y.astype(int)


def load_fashion_mnist(n_samples=10000):
    X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True,
                        parser='pandas', as_frame=False)
    if n_samples and n_samples < len(X):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), n_samples, replace=False)
        X, y = X[idx], y[idx]
    return X.astype(np.float32), y.astype(int)


def load_swiss_roll(n_samples=3000):
    X, t = make_swiss_roll(n_samples=n_samples, noise=0.1, random_state=42)
    return X.astype(np.float32), t


def load_hierarchical_gaussians(n_clusters=9, n_per_cluster=500, dim=50):
    rng = np.random.RandomState(42)
    n_macro = 3
    clusters_per_macro = n_clusters // n_macro

    X_list, y_list = [], []
    label = 0
    for macro in range(n_macro):
        macro_center = rng.randn(dim) * 20
        for sub in range(clusters_per_macro):
            sub_center = macro_center + rng.randn(dim) * 3
            points = sub_center + rng.randn(n_per_cluster, dim) * 0.5
            X_list.append(points)
            y_list.append(np.full(n_per_cluster, label))
            label += 1

    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list).astype(int)
    return X, y


def load_s_curve_hole(n_target=10000):
    rng = np.random.RandomState(42)
    X, t = make_s_curve(n_samples=n_target * 3, noise=0.1, random_state=42)
    mask = ~((np.abs(X[:, 0]) < 0.5) & (np.abs(X[:, 2]) < 0.5))
    X, t = X[mask], t[mask]
    if len(X) > n_target:
        idx = rng.choice(len(X), n_target, replace=False)
        X, t = X[idx], t[idx]
    return X.astype(np.float32), t


def load_coil20():
    """Load COIL-20 dataset, downloading if needed.

    Uses allow_pickle=True because the upstream PaCMAP repo .npy files
    were saved with pickle-backed object arrays.
    """
    data_path = DATA_DIR / "coil_20.npy"
    label_path = DATA_DIR / "coil_20_labels.npy"

    if not data_path.exists() or not label_path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        base_url = "https://github.com/YingfanWang/PaCMAP/raw/master/data"
        if not data_path.exists():
            _download_file(f"{base_url}/coil_20.npy", data_path)
        if not label_path.exists():
            _download_file(f"{base_url}/coil_20_labels.npy", label_path)

    # allow_pickle needed for upstream PaCMAP .npy format
    X = np.load(data_path, allow_pickle=True)  # noqa: S301
    X = X.reshape(X.shape[0], -1).astype(np.float32)
    y = np.load(label_path, allow_pickle=True).astype(int)  # noqa: S301
    return X, y


def load_usps():
    X, y = fetch_openml('usps', version=2, return_X_y=True,
                        parser='pandas', as_frame=False)
    return X.astype(np.float32), y.astype(int)


def load_20newsgroups(n_samples=10000, n_components=500):
    from sklearn.datasets import fetch_20newsgroups_vectorized
    from sklearn.decomposition import TruncatedSVD

    bunch = fetch_20newsgroups_vectorized(subset='all')
    X_sparse = bunch.data
    y = bunch.target

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X = svd.fit_transform(X_sparse)

    if n_samples and n_samples < len(X):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), n_samples, replace=False)
        X, y = X[idx], y[idx]

    return X.astype(np.float32), y.astype(int)


def load_pbmc3k():
    """Load PBMC3k single-cell RNA-seq. Requires scanpy."""
    import scanpy as sc
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = sc.datasets.pbmc3k_processed()
    X = adata.obsm['X_pca']
    y = adata.obs['louvain'].cat.codes.values
    return X.astype(np.float32), y.astype(int)


LABELED_DATASETS = {'mnist', 'fashion_mnist', 'hierarchical_gaussians',
                    'coil20', 'usps', '20newsgroups', 'pbmc3k'}

DATASETS = {
    'swiss_roll': load_swiss_roll,
    'hierarchical_gaussians': load_hierarchical_gaussians,
    'mnist': load_mnist,
    'fashion_mnist': load_fashion_mnist,
    's_curve_hole': load_s_curve_hole,
    'coil20': load_coil20,
    'usps': load_usps,
    '20newsgroups': load_20newsgroups,
    'pbmc3k': load_pbmc3k,
}
