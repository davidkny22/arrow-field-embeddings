"""Dataset loading for AFE benchmarks.

Imported from triplet-aware-pacmap benchmarking infrastructure.

Categories:
  general    — standard ML/manifold benchmarks (swiss_roll, MNIST, etc.)
  scrna      — single-cell RNA-seq datasets (24 datasets)

Subcategories within general:
  manifold   — low-dimensional continuous manifolds (swiss_roll, mammoth, s_curve_hole)
  image      — image classification datasets (MNIST, Fashion-MNIST, COIL-20, USPS)
  text       — text/NLP datasets (20Newsgroups)
  synthetic  — synthetic cluster datasets (hierarchical_gaussians)

Subcategories within scrna:
  Discrete/Clustered (8): PBMC3k, PBMC68k, Baron (H), Baron (M), Zeisel,
                           Lawlor, Tabula Muris, Macosko Retina
  Continuous/Trajectory (8): Planaria, Dentate Gyrus, Paul15, La Manno,
                              C. elegans, Hydra, Campbell, Tasic
  Complex/Noisy (4): Velmeshev, Usoskin, Chen, Marques
  Additional Pancreas (4): Muraro, Segerstolpe, Xin, Romanov
"""

import numpy as np
from sklearn.datasets import fetch_openml, make_swiss_roll, make_s_curve
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"
SCRNA_CACHE_DIR = DATA_DIR / "scrna_cache"


def _download_file(url, dest):
    """Download a file from url to dest, with progress."""
    import urllib.request
    print(f"  Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")


# ---------------------------------------------------------------------------
# Shared scRNA-seq preprocessing
# ---------------------------------------------------------------------------

def _preprocess_sce(sce, label_col='cell.type', n_hvgs=2000, n_pcs=50):
    """Convert a SingleCellExperiment to (X_pca, y_labels).

    Pipeline: raw counts -> filter -> normalize -> log1p -> HVG -> scale -> PCA.
    Caches the result as .npz to avoid reprocessing.
    """
    import scanpy as sc
    import warnings

    # Try to find the counts matrix
    assay_names = list(sce.assay_names)
    X_raw = None
    for name in ['counts', 'logcounts', assay_names[0]]:
        if name in assay_names:
            X_raw = sce.assay(name)
            break
    if X_raw is None:
        raise ValueError(f"No usable assay found. Available: {assay_names}")

    # Build AnnData (genes x cells -> cells x genes)
    X_dense = np.array(X_raw.T, dtype=np.float32)
    adata = sc.AnnData(X_dense)

    # Find label column
    col_data = sce.col_data
    col_names = list(col_data.column_names)
    y = None
    for candidate in [label_col, 'cell.type', 'cell_type', 'Cell_type',
                      'celltype', 'CellType', 'label', 'Label', 'cluster',
                      'inferred cell type']:
        if candidate in col_names:
            labels = list(col_data[candidate])
            adata.obs['celltype'] = labels
            adata.obs['celltype'] = adata.obs['celltype'].astype('category')
            y = adata.obs['celltype'].cat.codes.values
            break
    if y is None:
        raise ValueError(f"No label column found. Available: {col_names}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        n_hvg = min(n_hvgs, adata.n_vars)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg)
        adata = adata[:, adata.var.highly_variable].copy()
        sc.pp.scale(adata, max_value=10)
        n_pc = min(n_pcs, adata.n_vars - 1, adata.n_obs - 1)
        sc.tl.pca(adata, n_comps=n_pc)

    X = adata.obsm['X_pca'].astype(np.float32)
    y = adata.obs['celltype'].cat.codes.values.astype(int)
    return X, y


def _load_scrnaseq_dataset(name, version, label_col='cell.type', path=None):
    """Load a dataset from the scrnaseq package with caching."""
    SCRNA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{path}" if path else ""
    cache_path = SCRNA_CACHE_DIR / f"{name}_{version}{suffix}.npz"

    if cache_path.exists():
        data = np.load(cache_path)
        return data['X'], data['y']

    import scrnaseq
    path_str = f" path={path}" if path else ""
    print(f"  Fetching {name} (version {version}{path_str}) from scrnaseq...")
    kwargs = {}
    if path is not None:
        kwargs['path'] = path
    sce = scrnaseq.fetch_dataset(name, version, **kwargs)
    X, y = _preprocess_sce(sce, label_col=label_col)
    np.savez_compressed(cache_path, X=X, y=y)
    print(f"  Cached to {cache_path} ({X.shape[0]} cells, {X.shape[1]} PCs)")
    return X, y


# ---------------------------------------------------------------------------
# General benchmark datasets
# ---------------------------------------------------------------------------

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


def load_mammoth():
    data_path = DATA_DIR / "mammoth_3d.json"
    if data_path.exists():
        import json
        with open(data_path) as f:
            data = np.array(json.load(f), dtype=np.float32)
        return data, np.zeros(len(data), dtype=int)
    raise FileNotFoundError(
        f"Mammoth data not found at {data_path}. "
        "This file should be included in the repository under data/."
    )


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
    """Load COIL-20 dataset, downloading from upstream PaCMAP repo if needed."""
    data_path = DATA_DIR / "coil_20.npy"
    label_path = DATA_DIR / "coil_20_labels.npy"

    if not data_path.exists() or not label_path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        base_url = "https://github.com/YingfanWang/PaCMAP/raw/master/data"
        print("COIL-20 data not found locally. Downloading from upstream PaCMAP repo...")
        if not data_path.exists():
            _download_file(f"{base_url}/coil_20.npy", data_path)
        if not label_path.exists():
            _download_file(f"{base_url}/coil_20_labels.npy", label_path)

    X = np.load(data_path, allow_pickle=True)
    X = X.reshape(X.shape[0], -1).astype(np.float32)
    y = np.load(label_path, allow_pickle=True).astype(int)
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


def load_ag_news(n_samples=10000):
    """AG News: sentence embeddings via all-MiniLM-L6-v2 (384d).

    4 classes: World, Sports, Business, Sci/Tech.
    Uses cached embeddings from data/ag_news_minilm.npz if available,
    otherwise downloads AG News and embeds on the fly.
    """
    cache_path = DATA_DIR / "ag_news_minilm.npz"
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        X, y = data['X'], data['y']
    else:
        from datasets import load_dataset
        from sentence_transformers import SentenceTransformer

        ds = load_dataset("ag_news", split="train")
        rng = np.random.default_rng(42)
        indices = rng.choice(len(ds), size=10000, replace=False)
        indices.sort()
        subset = ds.select(indices.tolist())

        texts = subset["text"]
        y = np.array(subset["label"])

        model = SentenceTransformer("all-MiniLM-L6-v2")
        X = np.array(
            model.encode(texts, show_progress_bar=True, batch_size=256),
            dtype=np.float32,
        )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path, X=X, y=y,
            texts=np.array(texts, dtype=object),
            label_names=np.array(["World", "Sports", "Business", "Sci/Tech"]),
        )

    if n_samples and n_samples < len(X):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), n_samples, replace=False)
        X, y = X[idx], y[idx]

    return X.astype(np.float32), y.astype(int)


# ---------------------------------------------------------------------------
# scRNA-seq datasets
# ---------------------------------------------------------------------------

def load_pbmc3k():
    """PBMC3k (scanpy built-in). Blood, human, ~2638 cells."""
    import scanpy as sc
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = sc.datasets.pbmc3k_processed()
    X = adata.obsm['X_pca']
    y = adata.obs['louvain'].cat.codes.values
    return X.astype(np.float32), y.astype(int)


def load_pbmc68k_reduced():
    """PBMC68k reduced (scanpy built-in). Blood, human, ~700 cells."""
    import scanpy as sc
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = sc.datasets.pbmc68k_reduced()
    X = adata.obsm['X_pca']
    y = adata.obs['louvain'].cat.codes.values
    return X.astype(np.float32), y.astype(int)


def load_paul15():
    """Paul15 HSC/Myeloid (scanpy built-in). Mouse, ~2730 cells."""
    import scanpy as sc
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = sc.datasets.paul15()
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        n_hvg = min(2000, adata.n_vars)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg)
        adata = adata[:, adata.var.highly_variable].copy()
        sc.pp.scale(adata, max_value=10)
        n_pc = min(50, adata.n_vars - 1, adata.n_obs - 1)
        sc.tl.pca(adata, n_comps=n_pc)
    X = adata.obsm['X_pca'].astype(np.float32)
    y = adata.obs['paul15_clusters'].cat.codes.values.astype(int)
    return X, y


def load_baron_human():
    """Baron Pancreas (human). ~8569 cells."""
    return _load_scrnaseq_dataset(
        'baron-pancreas-2016', '2023-12-14', label_col='cell.type',
        path='human')


def load_baron_mouse():
    """Baron Pancreas (mouse). ~1886 cells."""
    return _load_scrnaseq_dataset(
        'baron-pancreas-2016', '2023-12-14', label_col='cell.type',
        path='mouse')


def load_muraro():
    """Muraro Pancreas (human). ~2640 cells."""
    return _load_scrnaseq_dataset(
        'muraro-pancreas-2016', '2023-12-19', label_col='cell.type')


def load_segerstolpe():
    """Segerstolpe Pancreas (human). ~3354 cells."""
    return _load_scrnaseq_dataset(
        'segerstolpe-pancreas-2016', '2023-12-19',
        label_col='inferred cell type')


def load_xin():
    """Xin Pancreas (human). ~1600 cells."""
    return _load_scrnaseq_dataset(
        'xin-pancreas-2016', '2023-12-19', label_col='cell.type')


def load_zeisel():
    """Zeisel Brain 2015. Mouse cortex, ~3005 cells."""
    return _load_scrnaseq_dataset(
        'zeisel-brain-2015', '2023-12-14', label_col='level1class')


def load_tasic():
    """Tasic Brain 2016. Mouse cortex, ~1809 cells."""
    return _load_scrnaseq_dataset(
        'tasic-brain-2016', '2024-04-18', label_col='broad_type')


def load_chen():
    """Chen Brain 2017. Mouse hypothalamus, ~14419 cells."""
    return _load_scrnaseq_dataset(
        'chen-brain-2017', '2023-12-14', label_col='SVM_clusterID')


def load_campbell():
    """Campbell Brain 2017. Mouse hypothalamus, ~21086 cells."""
    return _load_scrnaseq_dataset(
        'campbell-brain-2017', '2023-12-14', label_col='clust_all')


def load_marques():
    """Marques Brain 2016. Mouse oligodendrocytes, ~5069 cells."""
    return _load_scrnaseq_dataset(
        'marques-brain-2016', '2023-12-19', label_col='cell.type')


def load_romanov():
    """Romanov Brain 2017. Mouse hypothalamus, ~2881 cells."""
    return _load_scrnaseq_dataset(
        'romanov-brain-2017', '2023-12-19', label_col='level1 class')


def load_lawlor():
    """Lawlor Pancreas 2017. Human, ~638 cells."""
    return _load_scrnaseq_dataset(
        'lawlor-pancreas-2017', '2023-12-17', label_col='cell type')


def load_lamanno():
    """La Manno Brain 2016. Human midbrain, ~1977 cells."""
    return _load_scrnaseq_dataset(
        'lamanno-brain-2016', '2023-12-17', label_col='cell.type',
        path='human-embryo')


def load_usoskin():
    """Usoskin Brain 2015. Mouse sensory neurons, ~852 cells."""
    return _load_scrnaseq_dataset(
        'usoskin-brain-2015', '2023-12-19', label_col='Level 1')


# ---------------------------------------------------------------------------
# New scRNA-seq datasets (trajectory / atlas / challenge)
# ---------------------------------------------------------------------------

def _preprocess_anndata(adata, label_col, max_cells=None, n_hvgs=2000, n_pcs=50):
    """Preprocess an AnnData object: filter, normalize, HVG, scale, PCA.

    Optionally subsamples to max_cells first.
    Returns (X_pca, y_labels).
    """
    import scanpy as sc
    import warnings

    if max_cells and adata.n_obs > max_cells:
        rng = np.random.RandomState(42)
        idx = rng.choice(adata.n_obs, max_cells, replace=False)
        adata = adata[idx].copy()

    y = adata.obs[label_col].astype('category').cat.codes.values.astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        n_hvg = min(n_hvgs, adata.n_vars)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg)
        adata = adata[:, adata.var.highly_variable].copy()
        sc.pp.scale(adata, max_value=10)
        n_pc = min(n_pcs, adata.n_vars - 1, adata.n_obs - 1)
        sc.tl.pca(adata, n_comps=n_pc)

    X = adata.obsm['X_pca'].astype(np.float32)
    y = adata.obs[label_col].astype('category').cat.codes.values.astype(int)
    return X, y


def _download_h5ad(url, dest):
    """Download an h5ad file with progress."""
    import urllib.request
    print(f"  Downloading {dest.name} from {url[:80]}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp, open(dest, 'wb') as f:
        total = int(resp.headers.get('Content-Length', 0))
        downloaded = 0
        while True:
            chunk = resp.read(1024 * 1024)  # 1 MB chunks
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0 and downloaded % (10 * 1024 * 1024) == 0:
                print(f"    {downloaded / (1024*1024):.0f}/{total/(1024*1024):.0f} MB")
    print(f"  Saved ({dest.stat().st_size / (1024*1024):.1f} MB)")


def load_dentate_gyrus():
    """Dentate Gyrus (Hochgerner 2018). Mouse hippocampus, ~2930 cells.

    Single-path neurogenesis trajectory: radial glia -> neuroblasts -> granule cells.
    Source: scvelo built-in dataset.
    """
    cache_path = SCRNA_CACHE_DIR / "dentate_gyrus.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        return data['X'], data['y']

    SCRNA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    import scvelo as scv
    print("  Loading Dentate Gyrus from scvelo...")
    adata = scv.datasets.dentategyrus()
    X, y = _preprocess_anndata(adata, label_col='clusters')
    np.savez_compressed(cache_path, X=X, y=y)
    print(f"  Cached to {cache_path} ({X.shape[0]} cells, {X.shape[1]} PCs)")
    return X, y


def load_macosko():
    """Macosko Retina (Macosko 2015). Mouse retina, ~44k cells.

    High-noise challenge dataset with 39 transcriptionally distinct clusters.
    Source: scrnaseq package.
    """
    cache_path = SCRNA_CACHE_DIR / "macosko_retina.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        return data['X'], data['y']

    SCRNA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    import scrnaseq
    print("  Fetching Macosko Retina from scrnaseq...")
    sce = scrnaseq.fetch_dataset('macosko-retina-2015', '2023-12-19')

    import scanpy as sc
    import warnings

    assay_names = list(sce.assay_names)
    X_raw = sce.assay(assay_names[0])
    X_dense = np.array(X_raw.T, dtype=np.float32)
    adata = sc.AnnData(X_dense)

    clusters = list(sce.col_data['cluster'])
    labels = [str(c) for c in clusters]
    adata.obs['cluster'] = labels

    valid_mask = np.array([l not in ('--', 'nan', 'None') for l in labels])
    adata = adata[valid_mask].copy()
    print(f"  Kept {adata.n_obs} cells with valid cluster labels")

    if adata.n_obs > 20000:
        rng = np.random.RandomState(42)
        idx = rng.choice(adata.n_obs, 20000, replace=False)
        adata = adata[idx].copy()
        print(f"  Subsampled to {adata.n_obs} cells")

    X, y = _preprocess_anndata(adata, label_col='cluster')
    np.savez_compressed(cache_path, X=X, y=y)
    print(f"  Cached to {cache_path} ({X.shape[0]} cells, {X.shape[1]} PCs)")
    return X, y


def load_celegans():
    """C. elegans embryo (Packer 2019). ~89k cells, subsampled to 20k.

    Deterministic hierarchical lineage tree with discrete branches.
    Source: h5ad from wormcells GitHub releases.
    """
    cache_path = SCRNA_CACHE_DIR / "celegans.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        return data['X'], data['y']

    SCRNA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h5ad_path = SCRNA_CACHE_DIR / "packer2019.h5ad"

    if not h5ad_path.exists():
        url = ("https://github.com/Munfred/wormcells-site/releases/"
               "download/packer2019/packer2019.h5ad")
        _download_h5ad(url, h5ad_path)

    import scanpy as sc
    print("  Reading C. elegans h5ad...")
    adata = sc.read_h5ad(h5ad_path)

    label_col = 'cell_type'
    if label_col not in adata.obs.columns:
        for candidate in ['cell_subtype', 'plot_cell_type', 'Cell_type']:
            if candidate in adata.obs.columns:
                label_col = candidate
                break

    mask = adata.obs[label_col].notna() & (adata.obs[label_col] != '')
    adata = adata[mask].copy()

    X, y = _preprocess_anndata(adata, label_col=label_col, max_cells=20000)
    np.savez_compressed(cache_path, X=X, y=y)
    print(f"  Cached to {cache_path} ({X.shape[0]} cells, {X.shape[1]} PCs)")
    return X, y


def load_tabula_muris():
    """Tabula Muris (TM Consortium 2018). Mouse multi-tissue atlas, subsampled to 20k.

    Massive multi-tissue atlas (20 organs). Tests global tissue separation.
    Source: figshare (10x droplet data + annotations CSV).
    """
    cache_path = SCRNA_CACHE_DIR / "tabula_muris.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        return data['X'], data['y']

    SCRNA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    import urllib.request
    import zipfile
    import scanpy as sc
    import pandas as pd

    ann_path = SCRNA_CACHE_DIR / "annotations_droplet.csv"
    if not ann_path.exists():
        print("  Downloading Tabula Muris annotations (7 MB)...")
        urllib.request.urlretrieve(
            "https://ndownloader.figshare.com/files/13088039", ann_path)

    zip_path = SCRNA_CACHE_DIR / "droplet.zip"
    droplet_dir = SCRNA_CACHE_DIR / "droplet"
    if not droplet_dir.exists():
        if not zip_path.exists():
            print("  Downloading Tabula Muris droplet data (375 MB)...")
            urllib.request.urlretrieve(
                "https://ndownloader.figshare.com/files/10700167", zip_path)
        print("  Extracting droplet.zip...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(SCRNA_CACHE_DIR)

    print("  Reading Tabula Muris annotations...")
    ann = pd.read_csv(ann_path)

    print("  Reading 10x matrices per tissue...")
    tissue_dirs = sorted([d for d in droplet_dir.iterdir() if d.is_dir()])
    adatas = []
    for td in tissue_dirs:
        try:
            adata_t = sc.read_10x_mtx(td, var_names='gene_symbols',
                                       make_unique=True)
            dir_name = td.name
            idx = dir_name.find('-10X_')
            channel = dir_name[idx + 1:] if idx >= 0 else dir_name

            new_names = []
            for bc in adata_t.obs_names:
                barcode = bc.rstrip('-1').replace('-1', '')
                new_names.append(f"{channel}_{barcode}")
            adata_t.obs_names = new_names
            adata_t.obs['channel'] = channel
            adatas.append(adata_t)
        except Exception as e:
            print(f"    Skipping {td.name}: {e}")

    import anndata
    adata = anndata.concat(adatas, join='outer')
    adata.obs_names_make_unique()
    print(f"  Combined: {adata.n_obs} cells, {adata.n_vars} genes")

    cell_to_label = dict(zip(ann['cell'].astype(str), ann['cell_ontology_class']))
    labels = [cell_to_label.get(obs_name, '') for obs_name in adata.obs_names]
    adata.obs['cell_ontology_class'] = labels

    mask = np.array([l != '' for l in labels])
    adata = adata[mask].copy()
    print(f"  Cells with labels: {adata.n_obs}")

    X, y = _preprocess_anndata(adata, label_col='cell_ontology_class',
                                max_cells=20000)
    np.savez_compressed(cache_path, X=X, y=y)
    print(f"  Cached to {cache_path} ({X.shape[0]} cells, {X.shape[1]} PCs)")
    return X, y


def load_planaria():
    """Planaria (Plass 2018). Flatworm whole-organism atlas, ~21k cells.

    Gold standard continuous trajectory: stem-cell-to-tissue differentiation tree.
    Source: DGE matrix + annotations from Planaria Single Cell Atlas (PSCA).
    """
    cache_path = SCRNA_CACHE_DIR / "planaria.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        return data['X'], data['y']

    SCRNA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    import urllib.request
    import gzip
    import scanpy as sc
    import warnings

    base_url = "http://bimsbstatic.mdc-berlin.de/rajewsky/PSCA"

    dge_path = SCRNA_CACHE_DIR / "planaria_dge.txt.gz"
    if not dge_path.exists():
        print("  Downloading Planaria DGE matrix (18 MB)...")
        urllib.request.urlretrieve(f"{base_url}/dge.txt.gz", dge_path)

    ann_path = SCRNA_CACHE_DIR / "planaria_R_annotation.txt"
    if not ann_path.exists():
        print("  Downloading Planaria annotations...")
        urllib.request.urlretrieve(f"{base_url}/R_annotation.txt", ann_path)

    print("  Reading Planaria annotations...")
    with open(ann_path) as f:
        annotations = [line.strip() for line in f]

    print("  Reading Planaria DGE matrix (this may take a minute)...")
    import pandas as pd
    df = pd.read_csv(dge_path, sep='\t', index_col=0, compression='gzip')
    X_raw = df.values.T.astype(np.float32)
    n_cells = X_raw.shape[0]

    print(f"  DGE shape: {X_raw.shape} (cells x genes)")
    print(f"  Annotations: {len(annotations)} cell types")

    assert len(annotations) == n_cells, \
        f"Annotation count ({len(annotations)}) != cell count ({n_cells})"

    adata = sc.AnnData(X_raw)
    adata.obs['celltype'] = annotations
    adata.obs['celltype'] = adata.obs['celltype'].astype('category')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        n_hvg = min(2000, adata.n_vars)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg)
        adata = adata[:, adata.var.highly_variable].copy()
        sc.pp.scale(adata, max_value=10)
        n_pc = min(50, adata.n_vars - 1, adata.n_obs - 1)
        sc.tl.pca(adata, n_comps=n_pc)

    X = adata.obsm['X_pca'].astype(np.float32)
    y = adata.obs['celltype'].cat.codes.values.astype(int)
    np.savez_compressed(cache_path, X=X, y=y)
    print(f"  Cached to {cache_path} ({X.shape[0]} cells, {X.shape[1]} PCs)")
    return X, y


def load_velmeshev():
    """Velmeshev Brain/Autism (Velmeshev 2019). Human PFC, subsampled to 20k.

    High-noise disease context challenge dataset (ASD vs control).
    Source: UCSC Cell Browser (autism.cells.ucsc.edu).
    """
    cache_path = SCRNA_CACHE_DIR / "velmeshev.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        return data['X'], data['y']

    SCRNA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    import urllib.request
    import scanpy as sc
    import warnings

    base_url = "https://autism.cells.ucsc.edu/autism"

    meta_path = SCRNA_CACHE_DIR / "velmeshev_meta.tsv"
    if not meta_path.exists():
        print("  Downloading Velmeshev metadata (11 MB)...")
        urllib.request.urlretrieve(f"{base_url}/meta.tsv", meta_path)

    expr_path = SCRNA_CACHE_DIR / "velmeshev_exprMatrix.tsv.gz"
    if not expr_path.exists():
        print("  Downloading Velmeshev expression matrix (1.8 GB)...")
        print("  This will take several minutes...")
        urllib.request.urlretrieve(f"{base_url}/exprMatrix.tsv.gz", expr_path)

    print("  Reading Velmeshev metadata...")
    import pandas as pd
    meta = pd.read_csv(meta_path, sep='\t', index_col=0)

    label_col = None
    for candidate in ['cluster', 'Cluster', 'cell_type', 'CellType',
                       'celltype', 'cell.type', 'Type']:
        if candidate in meta.columns:
            label_col = candidate
            break
    if label_col is None:
        for col in meta.columns:
            n_unique = meta[col].nunique()
            if 3 <= n_unique <= 200:
                label_col = col
                break

    print(f"  Using label column: '{label_col}' ({meta[label_col].nunique()} types)")

    valid_meta = meta[meta[label_col].notna()].copy()
    if len(valid_meta) > 20000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(valid_meta), 20000, replace=False)
        valid_meta = valid_meta.iloc[idx]
    selected_cells = set(valid_meta.index)
    print(f"  Selected {len(selected_cells)} cells")

    print("  Reading expression matrix (streaming)...")
    import gzip

    with gzip.open(expr_path, 'rt') as f:
        header = f.readline().strip().split('\t')
        all_cells = header

        cell_indices = []
        cell_names = []
        for i, cell in enumerate(all_cells):
            if cell in selected_cells:
                cell_indices.append(i)
                cell_names.append(cell)

        if not cell_indices:
            raise ValueError("No matching cells found between metadata and expression matrix")

        print(f"  Reading {len(cell_indices)} cells from {len(all_cells)} total...")

        rows = []
        gene_names = []
        for line_num, line in enumerate(f):
            parts = line.strip().split('\t')
            gene_names.append(parts[0])
            row = [float(parts[i]) if i < len(parts) else 0.0 for i in cell_indices]
            rows.append(row)
            if line_num % 5000 == 0 and line_num > 0:
                print(f"    ... {line_num} genes read")

    X_raw = np.array(rows, dtype=np.float32).T
    print(f"  Expression matrix: {X_raw.shape}")

    adata = sc.AnnData(X_raw)
    adata.obs_names = cell_names
    labels = valid_meta.loc[cell_names, label_col].values
    adata.obs['celltype'] = list(labels)
    adata.obs['celltype'] = adata.obs['celltype'].astype('category')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        n_hvg = min(2000, adata.n_vars)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg)
        adata = adata[:, adata.var.highly_variable].copy()
        sc.pp.scale(adata, max_value=10)
        n_pc = min(50, adata.n_vars - 1, adata.n_obs - 1)
        sc.tl.pca(adata, n_comps=n_pc)

    X = adata.obsm['X_pca'].astype(np.float32)
    y = adata.obs['celltype'].cat.codes.values.astype(int)
    np.savez_compressed(cache_path, X=X, y=y)
    print(f"  Cached to {cache_path} ({X.shape[0]} cells, {X.shape[1]} PCs)")
    return X, y


def load_hydra():
    """Hydra (Siebert 2019). Cnidarian whole-organism, ~25k cells.

    Multi-branch differentiation tree (epithelial vs interstitial lineages).
    Source: GEO GSE121617 UMI count matrix.

    NOTE: Cell type annotations are only available in Seurat RDS files on Dryad.
    This loader uses Louvain clustering on the count matrix as labels.
    """
    cache_path = SCRNA_CACHE_DIR / "hydra.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        return data['X'], data['y']

    SCRNA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    import urllib.request
    import gzip
    import scanpy as sc
    import warnings

    counts_path = SCRNA_CACHE_DIR / "hydra_UMICounts.txt.gz"
    if not counts_path.exists():
        url = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE121nnn/GSE121617/"
               "suppl/GSE121617_Hydra_DS_transcriptome_UMICounts.txt.gz")
        print("  Downloading Hydra UMI counts (66 MB)...")
        urllib.request.urlretrieve(url, counts_path)

    print("  Reading Hydra count matrix...")

    with gzip.open(counts_path, 'rt') as f:
        header = f.readline().strip().split('\t')
        all_cells = [c.strip('"') for c in header]
        n_cells = len(all_cells)

        rng = np.random.RandomState(42)
        max_cells = min(20000, n_cells)
        sel_idx = sorted(rng.choice(n_cells, max_cells, replace=False))
        sel_set = set(sel_idx)
        cell_names = [all_cells[i] for i in sel_idx]
        print(f"  Selecting {max_cells}/{n_cells} cells")

        rows = []
        for line_num, line in enumerate(f):
            parts = line.strip().split('\t')
            row = [int(parts[i + 1]) for i in sel_idx]
            rows.append(row)
            if (line_num + 1) % 10000 == 0:
                print(f"    {line_num + 1} genes read...", flush=True)

    X_raw = np.array(rows, dtype=np.float32).T
    print(f"  Shape: {X_raw.shape} (cells x genes)")

    adata = sc.AnnData(X_raw)
    adata.obs_names = cell_names

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        n_hvg = min(2000, adata.n_vars)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg)
        adata = adata[:, adata.var.highly_variable].copy()
        sc.pp.scale(adata, max_value=10)
        n_pc = min(50, adata.n_vars - 1, adata.n_obs - 1)
        sc.tl.pca(adata, n_comps=n_pc)

        sc.pp.neighbors(adata, n_pcs=n_pc)
        sc.tl.leiden(adata, resolution=1.0)

    X = adata.obsm['X_pca'].astype(np.float32)
    y = adata.obs['leiden'].astype('category').cat.codes.values.astype(int)
    np.savez_compressed(cache_path, X=X, y=y)
    print(f"  Cached to {cache_path} ({X.shape[0]} cells, {X.shape[1]} PCs)")
    return X, y


# ---------------------------------------------------------------------------
# Dataset registries
# ---------------------------------------------------------------------------

LABELED_DATASETS = {
    # General
    'mnist', 'fashion_mnist', 'hierarchical_gaussians',
    'coil20', 'usps', '20newsgroups', 'ag_news',
    # scRNA (all have cell type labels)
    'pbmc3k', 'pbmc68k_reduced', 'baron_human', 'baron_mouse',
    'zeisel', 'lawlor', 'tabula_muris', 'macosko',
    'planaria', 'dentate_gyrus', 'paul15', 'lamanno',
    'celegans', 'hydra', 'campbell', 'tasic',
    'velmeshev', 'usoskin', 'chen', 'marques',
    'muraro', 'segerstolpe', 'xin', 'romanov',
}

DATASETS_GENERAL = {
    'swiss_roll': load_swiss_roll,
    'hierarchical_gaussians': load_hierarchical_gaussians,
    'mnist': load_mnist,
    'fashion_mnist': load_fashion_mnist,
    'mammoth': load_mammoth,
    's_curve_hole': load_s_curve_hole,
    'coil20': load_coil20,
    'usps': load_usps,
    '20newsgroups': load_20newsgroups,
    'ag_news': load_ag_news,
}

DATASETS_SCRNA = {
    # Discrete/Clustered
    'pbmc3k': load_pbmc3k,
    'pbmc68k_reduced': load_pbmc68k_reduced,
    'baron_human': load_baron_human,
    'baron_mouse': load_baron_mouse,
    'zeisel': load_zeisel,
    'lawlor': load_lawlor,
    'tabula_muris': load_tabula_muris,
    'macosko': load_macosko,
    # Continuous/Trajectory
    'planaria': load_planaria,
    'dentate_gyrus': load_dentate_gyrus,
    'paul15': load_paul15,
    'lamanno': load_lamanno,
    'celegans': load_celegans,
    'hydra': load_hydra,
    'campbell': load_campbell,
    'tasic': load_tasic,
    # Complex/Noisy
    'velmeshev': load_velmeshev,
    'usoskin': load_usoskin,
    'chen': load_chen,
    'marques': load_marques,
    # Additional Pancreas
    'muraro': load_muraro,
    'segerstolpe': load_segerstolpe,
    'xin': load_xin,
    'romanov': load_romanov,
}

DATASET_CATEGORIES = {
    'general': DATASETS_GENERAL,
    'scrna': DATASETS_SCRNA,
}

# Flat dict of all datasets (for backwards compat)
DATASETS = {**DATASETS_GENERAL, **DATASETS_SCRNA}
