# Shared diagnostics and frozen PCA references

Verified against source `748ee444` (Methods 1.3.8, ABI 2.20) plus the local
wide-matrix T²/Q correction. A source capability and an installed binary's
availability are separate facts. No release is implied by this local audit.

## Fit on calibration, project everything else

The existing `n4m_decomposition_flexible_pca_*` C interface provides PCA without
requesting full-spectrum moments. Its fit state owns the calibration means and
loadings; `transform` is const and does not fit on the evaluated batch.

```c
n4m_pp_flex_pca_handle_t *pca = NULL;
n4m_status_t status = n4m_decomposition_flexible_pca_create(&pca, 5.0);
/* Check status after every call. All matrix views contain caller-owned f64
   buffers; training_scores is n_train x 5, test_scores is n_test x 5. */
status = n4m_decomposition_flexible_pca_fit(pca, X_train);
status = n4m_decomposition_flexible_pca_transform(pca, X_train, training_scores);
status = n4m_decomposition_flexible_pca_transform(pca, X_test, test_scores);
n4m_decomposition_flexible_pca_destroy(pca);
```

The same recipe is available to Rust through the public C ABI and to Python/JS
through `FlexiblePCA.fit/transform`. Keep sample/group identities and physical
axis units in the owning protocol. Missing values must be resolved using the
declared calibration-only preprocessing policy before fitting: this interface
does not silently impute them. Scaling is a separate, training-fitted operator;
PCA itself centres features and does not standardize them implicitly.

`n4m_preflight_pca_compute` (ABI 2.19) additionally supplies calibration scores,
loadings, retained eigenvalues and exact full-spectrum moments. Its ABI 2.20
`_resolved` counterpart publishes the variance resolution and unresolved flags.
These are one-shot calibration diagnostics, not a reusable fitted handle.
Do not sum retained eigenvalues and present them as exact full-spectrum moments.

The 250-million-product refusal reported in issue #45 predates current 1.3.8.
Current `preflight_pca_budget.hpp` bounds the whole computation (2.4 billion
long-double multiply-add equivalents, 1 GiB working memory), including the
eigensolver ceiling. This change is already upstream; the local fix does not
raise or remove that guard. The regression witness uses the exact 390 x 2,151
shape, verifies exact moments, and proves that held-out projection leaves the
training scores bit-identical. On this Linux release build its synthetic
rank-two matrix and frozen projection took 0.333 seconds. This is one local
measurement, not a Windows, browser or representative production benchmark.

## API reuse matrix

| Requested operation | Existing owner and interface | Scientific contract / limit |
| --- | --- | --- |
| Training PCA and frozen test projection | Methods `decomposition.h`: `n4m_decomposition_flexible_pca_create/fit/transform` | Existing stateful API, including older ABI 2.17 consumers; no full moments required. Test data never enter fit. |
| Exact covariance moments, eigenvalues and accuracy | Methods `metrics.h`: `n4m_preflight_pca_compute`, `_resolved` | ABI 2.19 / 2.20 respectively. Numerical and work budgets are explicit. |
| Calibration T² and Q plus limits | Methods `outlier_detection.h`: `n4m_outlier_detection_hotelling_t2{,_resolved}`, `n4m_outlier_detection_q_residuals` | T² uses resolved rank; Q uses Jackson–Mudholkar approximation. Local correction supports tall and wide matrices using the existing compact SVD. These batch procedures refit the supplied batch: they are not frozen test-domain scores. |
| Full PCA-based domain monitor | Methods `metrics.h`: `n4m_stats_pca_monitor` | Use its documented reference/monitor contract; preserve the population, retained components, numerical domain and uncertainty assumptions. |
| Physical-axis alignment | Methods `transform/resampling.h`: `n4m_transform_resampler_*` | Supply physical axis coordinates and units; feature indices are not wavelengths. |
| SNV | Methods `metrics.h`: `n4m_preflight_snv_compute`; `transform/scatter.h` | Row-local transformation; never a replacement for sampling/provenance validation. |
| Offset/gain correction to a fitted mean reference | Methods `transform/scatter.h`: `n4m_transform_msc_fit/transform/get_reference` | MSC transformation exists. A separately named RMS/SAM/offset/gain report must declare its reference, normalization and masks. |
| Local polynomial noise proxy | Methods `n4m_preflight_noise_proxy_compute` | Mask-aware SG residual RMS corrected by smoothing noise gain; edges are not padded. A proxy under a locally polynomial signal, not an SNR or a physical-cause probability. |
| Distribution summaries | Methods `n4m_preflight_summary_compute` | Use declared units/validity masks. This does not automatically equal RMS to a supplied spectrum. |
| Nearest neighbours | Methods `n4m_preflight_nearest_neighbors` | Native bounded kernel. A ratio of neighbour distances is not automatically the LOF algorithm. |
| HDBSCAN, partition comparison | Methods `n4m_cluster_hdbscan`, `n4m_preflight_adjusted_rand_index`, `n4m_cluster_jaccard_matrix` | Keep representation, distance protocol, seed and populations in Preflight/Domains. |
| Caller-reference spectral SAM/RMS, spike/jump/clip ratios | No equivalent dedicated summary entry point verified in this public surface | Preserve as explicitly unsupported until a protocol and native kernel/parity fixture are qualified; do not quietly substitute a different reference or an index-axis formula. |
| Physical root-cause hypotheses and operational actions | Preflight / Domains / consuming product | Keep heuristic strength, evidence coverage and causal validation separate. These interpretations do not belong in generic Methods kernels. |

The shared C ABI is the portable boundary for Linux and Windows; the JS package
builds that same core to WASM. Local native and WASM tests verify the wide T²/Q
correction against constant-band padding, which preserves covariance geometry,
including a constant, unresolved subspace. Existing tall, rank-deficient,
nonfinite, scale, numerical-domain and exact-moment tests remain in the suite.
No new exported symbol, struct layout or ABI minor version is introduced.
