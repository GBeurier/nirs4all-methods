# Phase 31i — Batch 9: §5 PLS heads (GLM, QDA, Cox)

**Status:** shipped — local tag `phase-31i-batch-9-pls-heads`
(`0.79.0+abi.1.9.0`).

## Methods (all paper-only)

| Method | Internal kernel | Public C ABI entry | Paper |
|--------|-----------------|--------------------|-------|
| PLS-GLM (softmax/Poisson) | `fit_pls_glm` | `p4a_pls_glm_fit` | Bastien et al. 2005 |
| PLS-QDA | `fit_pls_qda` | `p4a_pls_qda_fit` | Barker & Rayens 2003 |
| PLS-Cox | `fit_pls_cox` | `p4a_pls_cox_fit` | Bastien 2008 |

## Why paper-only

Both `plsRglm` (Bastien 2005 IRLS) and `plsRcox` (Bastien 2008 deviance-
residual variant) are on CRAN, but installing them on this host fails
because of their `car` / `bipartite` / `pbkrtest` transitive deps which
themselves cascade-fail on a minimal R env. Even with the packages
installed, pls4all's `fit_pls_glm` is a simplified PLS-then-link
variant (not full IRLS), so numerical agreement would be qualitative
at best — closer to "test that the C ABI is callable" than to a true
parity gate.

Python `lifelines` ships Cox PH but no PLS-reduced variant; sklearn
ships `QuadraticDiscriminantAnalysis` but not the PLS+QDA pipeline.

## ABI delta

- 3 new public symbols, total 151 (sorted, all `p4a_*` prefixed).
- C ABI minor 8 → 9.

## Local gate

- 256 internal C++ tests (dev-release, local-asan-ubsan-gcc, local-ubsan-gcc).
- ABI symbol diff: clean.
- `git diff --check`: clean.
- Parity gate: 13 external PASS, 16 paper-only smoke PASS, 0 numpy.
