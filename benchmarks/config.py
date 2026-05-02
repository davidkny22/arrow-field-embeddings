"""Benchmark configuration: backends, modes, arrow counts, dataset costs."""

import math

# ── Arrow count selection ──────────────────────────────────────────────

ARROW_COUNTS_SWEEP = {
    "low":    [1, 2, 3, 5],       # d < 20
    "medium": [2, 5, 10, 25],     # 20 <= d < 100
    "high":   [5, 10, 25, 50],    # d >= 100
}

ALL_BACKENDS = ["pacmap", "umap", "trimap", "tsne"]
ALL_MODES = ["direct", "pca", "adaptive"]

# Approximate cost for each dataset: n_samples * n_arrows.
# Used to sort datasets fastest-first so results come in quickly.
DATASET_COST = {
    # 3D datasets (1 arrow)
    'swiss_roll': 3000,             # 3000 × 1
    's_curve_hole': 10000,          # ~10000 × 1
    'mammoth': 10000,               # ~10000 × 1
    # Small 50D (16 arrows)
    'lawlor': 10208,                # 638 × 16
    'pbmc68k_reduced': 11200,       # 700 × 16
    'usoskin': 13632,               # 852 × 16
    'xin': 25600,                   # 1600 × 16
    'tasic': 28944,                 # 1809 × 16
    'baron_mouse': 30176,           # 1886 × 16
    'lamanno': 31632,               # 1977 × 16
    'paul15': 43680,                # 2730 × 16
    'pbmc3k': 42208,                # 2638 × 16
    'muraro': 42240,                # 2640 × 16
    'romanov': 46096,               # 2881 × 16
    'dentate_gyrus': 46880,         # 2930 × 16
    'gaussian_noise': 48000,        # 3000 × 16
    'zeisel': 48080,                # 3005 × 16
    'segerstolpe': 53664,           # 3354 × 16
    'hierarchical_gaussians': 72000, # 4500 × 16
    'marques': 81104,               # 5069 × 16
    'baron_human': 137104,          # 8569 × 16
    # Large 50D
    'chen': 230704,                 # 14419 × 16
    'velmeshev': 320000,            # 20000 × 16
    'tabula_muris': 320000,         # 20000 × 16
    'macosko': 320000,              # 20000 × 16
    'celegans': 320000,             # 20000 × 16
    'hydra': 320000,                # 20000 × 16
    'planaria': 336000,             # 21000 × 16
    'campbell': 337376,             # 21086 × 16
    # High-D (many arrows)
    'usps': 790330,                 # 9298 × 85
    'ag_news': 1270000,             # 10000 × 127
    '20newsgroups': 1660000,        # 10000 × 166
    'mnist': 2610000,               # 10000 × 261
    'fashion_mnist': 2610000,       # 10000 × 261
}


def sort_datasets_by_cost(dataset_names):
    """Sort datasets fastest-first based on estimated cost."""
    return sorted(dataset_names, key=lambda d: DATASET_COST.get(d, 999999))


def get_optimal_arrow_count(n_features):
    """The correct arrow count: one arrow per 3 residual dimensions.

    The spatial embedding captures 3 dimensions. Each arrow encodes
    3 more dimensions (azimuth, elevation, magnitude), so the number
    of arrows needed is ceil((d - 3) / 3).

    For d <= 3, uses 1 arrow — the gap analysis still finds residual
    information even when spatial dims == data dims.
    """
    n_residual = n_features - 3
    if n_residual <= 0:
        return 1
    return math.ceil(n_residual / 3)


def get_arrow_counts(n_features):
    """Select arrow count sweep based on dimensionality (for exploratory runs)."""
    if n_features < 20:
        return ARROW_COUNTS_SWEEP["low"]
    elif n_features < 100:
        return ARROW_COUNTS_SWEEP["medium"]
    else:
        return ARROW_COUNTS_SWEEP["high"]
