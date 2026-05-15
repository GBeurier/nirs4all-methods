# Phase 31d — Batch 4: O2PLS + approximate-PRESS

**Status:** shipped — local tag `phase-31d-batch-4-o2pls-press`
(`0.74.0+abi.1.5.0`).

## Methods

| Method | Internal kernel | Public C ABI entry | Reference (Python) | Reference (R) |
|--------|-----------------|--------------------|---------------------|---------------|
| O2PLS (Trygg & Wold 2003) | `fit_o2pls` | `p4a_o2pls_fit` | numpy-mirror | `OmicsPLS::o2m` 2.1.0 (qualitative) |
| Approximate-PRESS (§29) | `approximate_press` | `p4a_approximate_press_compute` | numpy-mirror | none — ≠ LOO-PRESS |

## R reference: qualitative vs strict

The O2PLS algorithm has two widely cited formulations:

1. **Peel-then-PLS** (Trygg & Wold 2003 §3.2): remove `n_x_orthogonal`
   X-orthogonal components, then `n_y_orthogonal` Y-orthogonal components,
   then run a `n_predictive`-component PLS on the residual. This is what
   pls4all implements (`fit_o2pls` in `multiblock_extensions.cpp`).
2. **Joint iterative SVD** (Bouhaddani 2018 / OmicsPLS): alternate between
   X-orthogonal removal, Y-orthogonal removal and predictive component
   extraction in a single iterative loop driven by SVD.

Both are valid O2PLS realizations but produce different predictions
because they decompose variance in different orders. On the parity-gate
cell, RMSE-rel between pls4all and `OmicsPLS::o2m` is **0.45**. The
parity tolerance for the R comparison is intentionally widened to **1.0**:
the column documents that OmicsPLS is the canonical R O2PLS and that
predictions are within 2× scale, while the NumPy mirror — which exactly
reproduces pls4all's algorithm step-for-step — provides the strict
parity check (rmse_rel = 7.97e-02 at tol 0.2).

## Runner extensions

- `MethodSpec.prediction_key` lets methods like approximate-PRESS, whose
  primary output is not `predictions`, route a different result matrix
  to the parity comparator (`press_per_component` here).
- `_make_dataset` accepts `n_targets > 1` so methods that need
  multivariate Y (O2PLS with both X- and Y-orthogonal components needs
  q ≥ max(nx, ny) + n_predictive + 1) can request it.

## ABI delta

- 2 new public symbols, total 140 (sorted, all `p4a_*` prefixed).
- C ABI minor 4 → 5.
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.

## Local gate (this checkpoint)

- 256 internal C++ tests (dev-release, local-asan-ubsan-gcc, local-ubsan-gcc).
- ABI symbol diff vs expected list: clean.
- `git diff --check`: clean.
- Parity gate: 15 PASS, 9 `no_r_reference` (documented per-method).
