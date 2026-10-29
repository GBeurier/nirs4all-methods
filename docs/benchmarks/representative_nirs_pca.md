# Representative wide NIRS PCA qualification

Four real canonical public matrices were read from Datasets 0.3.14. Their
X/M/Y/raw-manifest hashes were checked before and after; no files were modified.
The executable benchmark is `benchmarks/diagnostics/benchmark_real_pca.py`, taking
an explicit source/license/identity manifest. It performs three fitted-PCA runs,
frozen query projection checks and an independent full NumPy SVD comparison of
the exact diagnostic reference. Input reading and the NumPy oracle are excluded
from native timing. Timings below are one Linux host, one numerical thread.

| Matrix | Shape | Declared license | Median fast fit | Exact diagnostic reference |
| --- | --- | --- | --- | --- |
| Figshare UVA illicit drugs, article 21252300 | 390 × 2151 | CC-BY-4.0 | 0.017 s | 0.098 s |
| EcoSIS NGEE Arctic Seward reflectance | 511 × 2151 | CC-BY-4.0 | 0.020 s | 0.221 s |
| EcoSIS Cucurbita two stresses reflectance | 541 × 2151 | CC-BY-4.0 | 0.020 s | 0.263 s |
| EcoSIS freeze-dried leaf reflectance | 492 × 2151 | ODbL-1.0 | 0.019 s | 0.200 s |

All have 2151 declared ascending bands from 350 to 2500 nm and finite values.
The exact 390×2151 witness is Figshare, not EcoSIS; the historical issue's stated
shape is not evidence of that source identity. In the freeze-dried matrix,
492 rows have 82 declared IDs with multiplicity six. The nature of repetition is
not established: these are not labelled technical replicates and do not imply
train/test independence. These matrices qualify numerical behavior/performance,
not predictive generalization.

Against independent SVD, the exact diagnostic reference's largest absolute
projector discrepancy was 1.8e-15 and its largest retained eigenvalue discrepancy
7.9e-10 (the NGEE eigenvalues are about 1.3e5–4.8e5). Changing query contents never
changed calibration scores, means, loadings or eigenvalues. No full moments were
requested or represented by a truncated sum.

The existing fast FlexiblePCA mode uses the native deterministic randomized SVD
for these shapes. Its approximation is distinct: on NGEE, its projector differed
by 0.00573 and the eigenvalue absolute discrepancy reached 116, about 0.09% of the
smallest retained eigenvalue. This observation motivated exposing the existing
exact Gram solver in the new diagnostic profile. The fast estimator and its
serialized-state behavior were not silently changed.

RSS was sampled every 2 ms during the fast fit/projections; sampled incremental
peaks ranged from roughly 4 KiB (allocator reuse) to 8.4 MiB, and are not rigorous
allocation bounds. The JSON receipt retains baseline/peak RSS, repetitions,
source hashes, library hash, exact timings and oracle errors. The native budget
counts all live buffers and the eigensolver iteration ceiling; this benchmark
supports these representative shapes without removing that bound. Windows and
browser execution have separate qualification receipts; Linux timings are not
transferred to those environments.
