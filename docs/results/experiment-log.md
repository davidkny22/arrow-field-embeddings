# Arrow Field Embeddings: Experiment Log

## Origin and Question

The idea started with a simple question: what if, instead of crushing high-dimensional data down to 3 dimensions and losing everything else, you gave every point arrows that encode what the spatial layout can't?

Standard dimensionality reduction (t-SNE, UMAP, PaCMAP) places points in 3D space. The spatial positions capture some of the original structure, but for a 50-dimensional dataset, most of it is gone. There is no way to recover it from the 3D coordinates alone, and no way to compare two spatially close points to understand whether they are genuinely similar in the original space or only appear similar because the projection collapsed their differences.

The core idea: attach arrows to every point. Each arrow carries 3 geometric channels (azimuth, elevation, magnitude), adding 3 dimensions of information per arrow. A point with k arrows preserves 3 + 3k total dimensions. Arrow k means the same thing on every point, so arrows at different points can be directly compared. Two points with matching arrow configurations are similar in the original high-dimensional space. Two points with divergent arrows are different in specific, identifiable ways, even if the spatial layout places them near each other.

The residual definition emerged from the design process: rather than computing a formal inverse mapping from 3D back to high-D (complex, backend-dependent), measure what the spatial layout already captures by correlating each original dimension with the 3 spatial coordinates. Dimensions well-captured by the spatial layout are deprioritized. Dimensions poorly captured get assigned to arrows. This became the spatial information gap: 1 - mean(max absolute correlation per dimension). Arrows focus on the information the spatial layout misses.

Three encoding modes were designed for different data characteristics. Direct mapping assigns raw residual dimension values to arrow channels (dimension i*3+c goes to arrow i, channel c). PCA on residuals projects the residual space through PCA and encodes PC loadings as angular directions with PC scores as magnitudes. Adaptive grouping uses eigenvalue-gap detection to determine how many natural groups exist in the residual covariance, hierarchical clustering to assign dimensions to groups, and correlation validation to split weak groups.

The spatial invariance guarantee is structural: AFE never modifies the spatial coordinates from the backend. It is an additive layer.

---

## Experiment 1: Initial Benchmark

**Goal:** Validate that AFE recovers high-dimensional structure across multiple datasets, backends, and encoding modes, and that spatial metrics are preserved.

**Setup:** 15 datasets spanning synthetic (swiss_roll, s_curve_hole, gaussian_noise, hierarchical_gaussians), scRNA-seq (pbmc3k, pbmc68k_reduced, celegans, dentate_gyrus, planaria, paul15), and image (mnist, fashion_mnist, usps, coil20) domains. 3 backends (PaCMAP, TriMAP, UMAP). 3 encoding modes (direct, PCA, adaptive). 10 seeds per configuration. 1,770 benchmark records, 393 significance tests.

Metrics computed: knn_recall_k10, knn_recall_k50, spearman_dist_corr, trustworthiness_k10, continuity_k10, normalized_stress (spatial quality), reconstruction_mse, reconstruction_cosine (reconstruction quality), arrow_knn_recall_k10, arrow_knn_recall_k50, arrow_consistency, arrow_info_gain, spatial_info_gap (AFE-specific).

No recon_knn_recall_k10 metric was computed. t-SNE was not included as a backend.

**Results:**

Spatial preservation: baseline and AFE spatial metrics (knn_k10, trustworthiness, spearman, stress) were identical within floating-point precision across all configurations. Significance tests on spatial metrics showed no significant differences (all p > 0.05 after Benjamini-Hochberg correction).

Reconstruction error reduction (direct encoding, mean across seeds):

| Domain | Datasets | Recon MSE reduction vs baseline |
|--------|----------|---------------------------------|
| Synthetic (3D) | swiss_roll, s_curve_hole | 100% (exact reconstruction, MSE = 0.0000) |
| Synthetic (50D) | gaussian_noise | 96% (0.97 to 0.038) |
| Synthetic (50D) | hierarchical_gaussians | 93-97% depending on backend |
| scRNA-seq (50D) | pbmc3k, celegans, dentate_gyrus, paul15, planaria, pbmc68k_reduced | 83-95% |
| Image (256-784D) | mnist, fashion_mnist, usps | 99.6-99.9% |
| coil20 (50D) | coil20 | Reconstruction MSE values in 100K-1M range |

All reconstruction improvements were statistically significant (Wilcoxon signed-rank, p = 0.002, significant after BH correction) across all datasets and backends.

Reconstruction error was consistent across PaCMAP, TriMAP, and UMAP. Backend choice affected spatial quality but not the magnitude of AFE's reconstruction improvement.

**Limitations found:**

- t-SNE was not tested.
- No ReconKNN metric. Reconstruction MSE and cosine measure vector-level fidelity but do not directly test whether the reconstruction recovers the correct neighborhood structure.
- planaria/trimap had only baseline records (10 records). AFE runs for this combination did not complete.
- coil20 had reconstruction MSE values in the 100K-1M range, indicating unnormalized features. MSE comparisons with other datasets are not meaningful for this dataset.

### Decision Log

**Decision:** Rerun with t-SNE added and ReconKNN metric computed.

**Rationale:** The spatial preservation and reconstruction error reduction were clear across all tested configurations. Without t-SNE and without a neighborhood-level reconstruction metric, the evidence was incomplete. Reconstruction MSE measures vector fidelity but cannot distinguish between "arrows recover the right neighborhoods" and "arrows recover some structure that happens to reduce MSE."

---

## Experiment 2: Expanded Benchmark with ReconKNN

**Goal:** Add t-SNE as a backend, compute ReconKNN@10, expand dataset coverage, fill gaps from the first run.

**Setup:** 17 datasets (added ag_news [text, MiniLM 384D], tabula_muris and velmeshev [scRNA-seq 50D]; removed coil20). 4 backends (PaCMAP, TriMAP, t-SNE, UMAP). 3 encoding modes. 10 seeds. 2,720 benchmark records, 882 significance tests.

New metric: recon_knn_recall_k10. Reconstruct approximate HD vectors from the full AFE representation (spatial + arrows), compute KNN@10 against the original data. Directly comparable to baseline spatial KNN@10 since both use the same function, same k, same reference set.

**Results, direct encoding ReconKNN@10 (mean across seeds, per backend):**

| Dataset | Dims | PaCMAP | TriMAP | t-SNE | UMAP | Baseline range |
|---------|-----:|-------:|-------:|------:|-----:|---------------|
| swiss_roll | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.78-0.88 |
| s_curve_hole | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.56-0.76 |
| gaussian_noise | 50 | 0.7628 | 0.7629 | 0.7629 | 0.7613 | 0.03-0.18 |
| hierarchical_gaussians | 50 | 0.7820 | 0.8024 | 0.4620 | 0.7955 | 0.06-0.26 |
| pbmc3k | 50 | 0.7909 | 0.7923 | 0.7104 | 0.7850 | 0.09-0.24 |
| pbmc68k_reduced | 50 | 0.8591 | 0.8692 | 0.8604 | 0.8626 | 0.36-0.50 |
| celegans | 50 | 0.8756 | 0.8732 | 0.8513 | 0.8809 | 0.26-0.45 |
| dentate_gyrus | 50 | 0.8364 | 0.8740 | 0.7352 | 0.9146 | 0.24-0.41 |
| paul15 | 50 | 0.8441 | 0.7755 | 0.8205 | 0.8064 | 0.21-0.35 |
| planaria | 50 | 0.8075 | 0.8260 | 0.7376 | 0.8222 | 0.13-0.33 |
| tabula_muris | 50 | 0.8905 | 0.9016 | 0.8761 | 0.8855 | 0.30-0.51 |
| velmeshev | 50 | 0.7701 | 0.7758 | 0.7348 | 0.7725 | 0.15-0.33 |
| mnist | 784 | 0.9889 | 0.9887 | 0.9860 | 0.9881 | 0.28-0.48 |
| fashion_mnist | 784 | 0.9952 | 0.9957 | 0.9958 | 0.9945 | 0.28-0.46 |
| usps | 256 | 0.9895 | 0.9894 | 0.9902 | 0.9893 | 0.32-0.51 |
| 20newsgroups | 500 | 0.6901 | 0.6443 | 0.5930 | 0.5965 | 0.06-0.25 |
| ag_news | 384 | 0.9662 | 0.9658 | 0.9682 | 0.9686 | 0.26-0.48 |

**Encoding mode comparison (ReconKNN@10, averaged across 4 backends):**

| Dataset | Direct | PCA | Adaptive |
|---------|-------:|----:|---------:|
| swiss_roll | 1.0000 | 0.0651 | 0.3726 |
| s_curve_hole | 1.0000 | 0.0273 | 0.1113 |
| gaussian_noise | 0.7625 | 0.1126 | 0.2854 |
| hierarchical_gaussians | 0.7105 | 0.2121 | 0.0985 |
| pbmc3k | 0.7696 | 0.2261 | 0.4288 |
| pbmc68k_reduced | 0.8628 | 0.5002 | 0.6848 |
| celegans | 0.8702 | 0.2972 | 0.6359 |
| dentate_gyrus | 0.8400 | 0.2924 | 0.5568 |
| paul15 | 0.8116 | 0.3814 | 0.5220 |
| planaria | 0.7983 | 0.1835 | 0.4968 |
| tabula_muris | 0.8884 | 0.3134 | 0.6385 |
| velmeshev | 0.7633 | 0.2464 | 0.4676 |
| mnist | 0.9879 | 0.9589 | 0.8259 |
| fashion_mnist | 0.9953 | 0.9086 | 0.3474 |
| usps | 0.9896 | 0.9822 | 0.1657 |
| 20newsgroups | 0.6309 | 0.5152 | 0.4186 |
| ag_news | 0.9672 | 0.7694 | 0.7757 |

**Findings:**

1. Direct encoding achieved the highest ReconKNN@10 on every dataset. The range was 0.59-1.00 for direct, 0.03-0.98 for PCA, 0.10-0.83 for adaptive.

2. PCA encoding achieved 0.96+ on image datasets (MNIST, USPS) where intrinsic dimensionality is lower than ambient dimensionality. On low-dimensional data (swiss_roll 0.07, s_curve_hole 0.03), PCA performed below baseline spatial KNN.

3. Adaptive encoding was inconsistent across datasets. It achieved 0.68 on pbmc68k_reduced but 0.17 on USPS. On hierarchical_gaussians it achieved 0.10, below the baseline spatial KNN range.

4. Spatial metrics were identical between baseline and AFE across all 17 datasets, all 4 backends, and all 3 encoding modes. Significance tests confirmed no spatial degradation.

5. ReconKNN improvement was consistent across t-SNE, UMAP, PaCMAP, and TriMAP. Backend choice affected spatial quality. AFE's reconstruction quality was independent of backend choice.

6. t-SNE results had zero standard deviation across seeds for all metrics except runtime. The t-SNE implementation produces a deterministic embedding for a given dataset, so 10 seeds produced identical measurements.

7. Datasets with higher spatial information gap scores showed larger absolute ReconKNN values. scRNA-seq and text datasets had gap scores of 0.85-0.97. Synthetic 3D datasets had gap scores of 0.02-0.60.

**Anomalies:**

- hierarchical_gaussians/tsne: direct ReconKNN was 0.4620, compared to 0.78-0.80 on the other three backends.
- 20newsgroups showed the lowest direct ReconKNN across all datasets (0.59-0.69). Arrow capacity was 166 arrows x 3 channels = 498 dimensions for a 500D feature space (498 residual dims).

### Decision Log

**Decision:** Rerun the full benchmark with a shared-coordinate protocol and expanded metric coverage.

**Rationale:** Spatial metrics are identical between baseline and AFE in all tested configurations, confirming the invariance guarantee through metric equality. The shared-coordinate protocol (compute each spatial embedding once, save it, reload for both baseline and AFE evaluation) makes the invariance verifiable by construction rather than by metric comparison. The expanded run also adds metrics not yet computed: ReconKNN@50, mutual information estimates, downstream kNN classification, and negative controls.

---

## Observations Across Both Experiments

### Confirmed

- Spatial metrics are identical between baseline and AFE across all tested configurations (17 datasets, 4 backends, 3 encoding modes, 10 seeds).
- Direct encoding achieves the highest ReconKNN@10 on all 17 datasets tested.
- Reconstruction improvement is consistent across all four backends tested (t-SNE, UMAP, PaCMAP, TriMAP).
- ReconKNN standard deviation across seeds is < 0.015 for direct encoding on all datasets except hierarchical_gaussians/UMAP (0.0141) and dentate_gyrus/adaptive (0.0153).
- Image datasets (MNIST, Fashion-MNIST, USPS) achieve ReconKNN > 0.98 with direct encoding.
- 3D synthetic datasets (swiss_roll, s_curve_hole) achieve ReconKNN = 1.0000 with direct encoding (exact reconstruction).

### Observed

- PCA encoding achieves 0.96+ on image datasets but performs below baseline spatial KNN on 3D synthetic data. The failure on 3D data occurs because PCA rotates the single residual dimension away from the arrow channel when there is only one arrow.
- Adaptive encoding fails on specific datasets (USPS: 0.17, hierarchical_gaussians: 0.10) while performing reasonably on others (pbmc68k_reduced: 0.68, celegans: 0.64). The failure does not correlate with dimensionality or domain in an obvious way.
- t-SNE produces identical results across all 10 seeds. The spatial embedding is deterministic for a given dataset in this implementation.
- Adding t-SNE and ReconKNN in the second run did not change the pattern from the first run for reconstruction MSE/cosine. The same datasets that showed strong MSE reduction also showed strong ReconKNN.
- Runtime overhead from AFE is small relative to the spatial backend computation on all datasets tested.

### Unexplained

- hierarchical_gaussians/tsne shows direct ReconKNN of 0.46 compared to 0.78-0.80 on the other three backends. The baseline spatial KNN on tsne for this dataset is 0.26, compared to 0.06-0.11 on the other backends. Whether the different spatial structure captured by tsne interacts with the residual selection has not been tested.
- Adaptive encoding achieves 0.17 on USPS but 0.83 on MNIST. Both are handwritten digit image datasets. The eigenvalue gap structure of their residual covariances has not been compared.

---

## Where This Stands

**What has been tested:**

- 17 datasets across synthetic, image, text (TF-IDF and dense embeddings), and scRNA-seq domains.
- 4 DR backends (t-SNE, UMAP, PaCMAP, TriMAP) with default hyperparameters.
- 3 encoding modes (direct, PCA, adaptive) at maximum arrow capacity.
- 10 random seeds per configuration.
- Spatial invariance verified via metric equality and paired significance tests (Wilcoxon signed-rank, BH-corrected).
- Reconstruction quality measured via ReconKNN@10, reconstruction MSE, reconstruction cosine, arrow_knn_recall, arrow_info_gain, and spatial_info_gap.

**What has not been tested:**

- Shared-coordinate protocol (saving spatial embeddings to disk and reloading for both baseline and AFE evaluation).
- Mutual information estimates (Gaussian proxy, kNN/KSG) to validate whether arrows recover information beyond what correlation-based metrics capture.
- Downstream kNN classification accuracy from original HD, 3D spatial, and reconstructed AFE vectors.
- Negative controls (shuffled features, shuffled arrows, random spatial embeddings) to establish a reconstruction metric floor.
- Arrow-count ablations (sweeping arrow count from 1 to maximum on representative datasets).
- ReconKNN@50 as a support metric alongside ReconKNN@10.
- Non-default backend hyperparameters.
- Datasets with raw dimensionality above 784.

**What the next experiments should address:**

1. Shared-coordinate protocol with saved embeddings for byte-level invariance verification.
2. Expanded dataset set (the benchmark runner supports 34 dataset loaders).
3. MI estimates to validate information recovery beyond correlation.
4. Downstream classification to test whether neighborhood recovery translates to task performance.
5. Negative controls to establish metric baselines.
6. Arrow-count ablations to characterize the capacity-recovery relationship.
7. ReconKNN@50 to test whether larger neighborhood sizes change the recovery picture.

---

## Open Questions

1. **What causes adaptive encoding to fail on specific datasets?** Adaptive achieves 0.17 ReconKNN on USPS but 0.83 on MNIST. Both are image datasets with similar structure. The failure mode has not been diagnosed.

2. **Why does 20newsgroups show substantially lower ReconKNN (0.59-0.69) than ag_news (0.97) despite both being text datasets with similar dimensionality?** 20newsgroups uses TF-IDF (500D, sparse), ag_news uses MiniLM embeddings (384D, dense). Whether this is a sparsity effect or a representation effect has not been isolated.

3. **What determines the per-dataset ceiling for direct encoding ReconKNN?** scRNA-seq datasets range from 0.73 (velmeshev) to 0.91 (dentate_gyrus/UMAP) despite identical dimensionality (50D) and arrow count (16). The source of this variation has not been characterized.

4. **Does AFE's reconstruction improve downstream kNN classification accuracy?** Neither run computed classification metrics. Whether neighborhood recovery translates to preserved class separability has not been tested.

5. **Do the arrows recover mutual information lost in the spatial projection?** The spatial information gap measures correlation, not information. MI estimates (Gaussian proxy, kNN/KSG) applied to the spatial embedding alone vs. spatial + arrows would directly quantify how much information the arrows add. This has not been measured.

6. **How do negative controls compare against AFE's reconstruction?** Neither run included shuffled features, shuffled arrows, Gaussian noise inputs, or random spatial embeddings. There is no established floor for the reconstruction metrics.

7. **How does reconstruction degrade as arrow count decreases?** All experiments used maximum arrow capacity. The relationship between representation budget (3 + 3k dimensions) and ReconKNN has not been measured.

8. **Does ReconKNN@50 tell a different story than ReconKNN@10?** Neither run computed ReconKNN@50. Whether larger neighborhood sizes change the recovery picture is unknown.
