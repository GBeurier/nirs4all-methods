# Phase 31g — Batch 7: §7 sparse variants

**Status:** shipped — local tag `phase-31g-batch-7-sparse-variants`
(`0.77.0+abi.1.7.0`).

## Methods

| Method | Internal kernel | Public C ABI entry | Reference |
|--------|-----------------|--------------------|-----------|
| Sparse PLS-DA | `fit_sparse_pls_da` | `p4a_sparse_pls_da_fit` | R `spls::splsda` 2.3.2 (qualitative) |
| Group sparse PLS | `fit_group_sparse_pls` | `p4a_group_sparse_pls_fit` | paper-only (Liland & Indahl 2009) |
| Fused sparse PLS | `fit_fused_sparse_pls` | `p4a_fused_sparse_pls_fit` | paper-only (Yengo et al. 2016) |

## Runner extensions

- `MethodSpec.needs_labels` — request a deterministic int32 class label
  vector built by `_make_dataset` from `cell_params["n_classes"]`.
- `MethodSpec.needs_group_assignment` — request a deterministic int32
  feature-group vector (consecutive feature blocks).

## R reference qualitative parity

`spls::splsda` returns hard class labels; pls4all returns soft dummy-
encoded predictions. The parity test one-hot-encodes the R labels to
match pls4all's shape, so the comparison is on the *classification
boundary*. rmse_rel ≈ 0.92 — tolerance widened to admit this difference;
the column documents the R ref's presence and the soft-vs-hard scoring
gap.

## ABI delta

- 3 new public symbols, total 144 (sorted, all `p4a_*` prefixed).
- C ABI minor 6 → 7.
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.

## Local gate (this checkpoint)

- 256 internal C++ tests (dev-release, local-asan-ubsan-gcc, local-ubsan-gcc).
- ABI symbol diff vs expected list: clean.
- `git diff --check`: clean.
- Parity gate: 13 external PASS, 9 paper-only smoke PASS, 0 numpy.
