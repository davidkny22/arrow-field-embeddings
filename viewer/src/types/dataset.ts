/** AFE dataset types. */

export interface ClusterData {
  id: number;
  label: string;
  size: number;
  centroid: [number, number, number];
}

export interface AFEDataset {
  version: number;
  dataset: string;
  encoding_mode: string;
  n_points: number;
  n_arrows: number;
  embedding_dim: number;

  // Core data (flat arrays for GPU efficiency)
  positions: number[];       // [x0,y0,z0, x1,y1,z1, ...] (n*3)
  arrows: number[];          // [θ,φ,r, ...] (n*k*3, row-major by point then arrow)

  // Labels
  label_indices: number[];   // per-point label index into label_names
  label_names: string[];     // unique label strings

  // Clusters (derived from labels)
  clusters: ClusterData[];

  // Metadata
  gap_report: {
    information_gap_score: number;
    n_residual_dims: number;
    n_captured_dims: number;
  };
  metrics: {
    knn_recall_k10: number;
    arrow_knn_recall_k10: number;
    reconstruction_mse: number;
    arrow_info_gain: number;
    spearman_dist_corr: number;
  };

  // Per-point reconstruction error (optional, for color mode)
  recon_error?: number[];          // (n,) per-point MSE with full AFE
  recon_error_spatial?: number[];  // (n,) spatial-only MSE
}

export type ColorMode =
  | 'cluster'
  | 'highlight'
  | 'neighborhood'
  | 'arrow_magnitude'
  | 'arrow_direction'
  | 'recon_error';

export interface DatasetEntry {
  id: string;
  label: string;
  url: string;
}
