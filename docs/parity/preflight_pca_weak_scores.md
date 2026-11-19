# Preflight PCA weak-score portability

## Scope and reproduction

The Domains F4.3 report in [PR 16](https://github.com/dcornet/nirs4all-domains/pull/16)
isolates a 24 by 25 synthetic matrix after SNV. Its fifth component is resolved:
variance approximately 1.36535e-10, resolution approximately 2.04083e-12, and
lambda1/lambda5 approximately 1.68e11. The exact post-SNV binary64 matrix is
frozen in `parity/fixtures/preflight_pca_weak_scores_v1.json`. The fixture records
the upstream commit, original fixture hash, synthetic provenance and licence.
Its little-endian input SHA-256 is
`ac3b5248cc741c2c4e6b27b2b7d40035f089444f2e5eb624077ae3c0ac98bf39`.

On 2026-11-19 the public Windows wheels 1.3.13 (ABI 2.20) and 1.4.1
(ABI 2.24) reproduced the original Native/WASM discrepancy exactly. The relevant
PCA implementation was numerically unchanged across these releases. This was
not already fixed in 1.4.1. The Windows replay used Windows 10.0.26200 and actual
MSVC binaries, not Linux or emulation.

## Attribution and correction

In the wide branch (`n <= p`), the smaller Gram eigenvectors gave provisional
loadings `Xc^T u / sigma`. After two-pass modified Gram-Schmidt, the published
loadings were finalised, but the scores still came from `u * sigma`. Gram
eigenvector error in weak/null directions contaminated those scores. A small
backward eigenpair residual therefore did not imply accurate weak scores.

Both branches now use the same compensated projection of the centred data onto
the final loading vectors. The solver, eigenvalues, moments, operational
resolution, unresolved flags and zero unresolved scores are unchanged. The
extra wide-branch projection is included in the guaranteed work budget; no
extra matrix buffer is allocated. Requests near the work limit can be refused.

This corrects the demonstrated weak-score defect. It does not promise relative
accuracy for arbitrarily small eigenvalues, individually portable eigenvectors
inside a nearly degenerate eigenspace, or a different solver. The generic
fitted PCA implementation is outside this change.

## Independent reference and measured results

`parity/scripts/preflight_pca_weak_scores_reference.py` checks the frozen goldens
with mpmath 1.3.0 at 100 decimal digits. It centres the exact binary64 input at
high precision and takes its direct SVD, without loading Methods. Independent
SciPy LAPACK `gesvd`/`gesdd` probes also agreed on the weak score direction.
PC5 comparisons align its sign to the reference; the production sign convention
is unchanged. Pearson/eta comparisons retain atol=rtol=1e-10.

| Exact original input, PC5 | Before (published binaries) | Patched source |
| --- | ---: | ---: |
| Windows/WASM Pearson absolute difference | 5.262218247703565e-6 | 3.1030927827302435e-11 |
| Windows/WASM eta-squared absolute difference | 3.082494986032802e-7 | 4.455116831003636e-14 |
| Windows covariance eigenpair relative residual | 3.534503096736108e-17 | 3.534503096736108e-17 |
| Linux/WASM Pearson absolute difference | 5.551115123125783e-17 | 3.936850845320805e-13 |
| Linux/WASM eta-squared absolute difference | 1.734723475976807e-18 | 5.762404442499758e-14 |

The Windows before measurements were identical with 1.3.13 and 1.4.1; the Linux
before probe used 1.3.19. Published WASM scores were identical across all three
versions on the four original raw/SNV controls.

The original Linux/WASM agreement concealed the common score error: both were
approximately 9.56e-6 away from the direct-SVD score in relative L2 norm.
Across 36 combinations of row/column reversal, power-of-two scaling
(`2^-30`, `1`, `2^30`) and requested components (`5`, `20`, `24`):

| Patched backend | Maximum PC5 relative L2 error | Maximum Pearson absolute error | Maximum eta-squared absolute error |
| --- | ---: | ---: | ---: |
| Windows/MSVC | 1.785e-10 | 1.457e-11 | 1.200e-12 |
| Linux/GCC | 1.809e-10 | 2.043e-11 | 1.090e-12 |
| WASM/Node 24 | 1.785e-10 | 2.344e-11 | 1.200e-12 |

Eigenpair residuals are normalised by the covariance spectral norm. The score
reference tolerance (relative L2 2e-9) is specific to this conditioned fixture;
it does not replace the unchanged factor-statistics tolerances. Score variances
remain within the existing absolute eigenvalue accuracy contract.

## Verification and release status

The following checks passed (exit 0):

- Linux: `cmake --preset dev-release -DN4M_BUILD_STATIC=OFF -DN4M_BUILD_CLI=OFF`,
  build targets `n4m_preflight_tests n4m_preflight_pca_scores_tests`, then
  `ctest --preset dev-release -R '^n4m_preflight(_pca_scores)?_tests$'` (2/2).
- Windows: the corresponding `ci-windows-msvc-release` configure/build/test
  presets, using CMake 4.1.3 and Visual Studio 2022 MSVC (2/2). The test process
  DLL search path explicitly selects the candidate build's `cpp/src/Release`.
- WASM: `cmake --preset emscripten`, build target `n4m_wasm`, then
  `npm ci --ignore-scripts`, `npm run build`, `npm run stage:wasm`,
  `node test/run_preflight_diagnostics.mjs` and
  `node test/run_preflight_pca_weak_scores.mjs` in `bindings/js`.
- The new WASM test fails with the published 1.4.1 binary at relative score
  error 9.55956e-6 (expected exit 1), and passes against the candidate.
- Python: `pytest bindings/python/tests/test_intrinsic_diagnostics.py -k pca -q`
  against the candidate library (5 passed, 28 deselected).
- Independent oracle script: all frozen binary64 goldens reproduced at 100 digits.
- A separate 25 by 24 tall-input probe retained bit-identical complete PCA outputs
  against published WASM 1.4.1 after sharing the projection loop.
- Independent Opus plan and implementation reviews found no blocking issues.

The C ABI regression exercises both public Preflight entrypoints. Tests retain
PC5 as resolved, check prefix invariance with different k, verify score variance
against the absolute contract and keep unresolved score columns exactly zero.
Candidate Windows DLL SHA-256:
`97cb6a18950fa6e489f9e765b2ae1078fbf2bc326249ef7d527fc61908319870`.

No release, version bump or downstream tolerance adjustment is included. This
numeric change requires a Methods minor release, to be combined with the
maintainer's concurrent AOM fix. The full Domains qualification, opaque hashes
and PDF gates were not rerun and are not declared passed by these owner checks.
