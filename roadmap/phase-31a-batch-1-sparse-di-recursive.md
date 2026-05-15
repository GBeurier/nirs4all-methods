# Phase 31a — Batch 1: Sparse SIMPLS + DI-PLS + Recursive PLS (public C ABI)

**Status:** shipped — local tag `phase-31a-batch-1-sparse-di-recursive`
(`0.71.0+abi.1.2.0`).

This phase opens the multi-batch tranche that promotes the internal
`pls4all::core::` kernels from Phases 8 – 30 to the public C ABI, with an
external-reference parity gate per method (Python + R when available).

## Scope

Three methods promoted in this batch, chosen because they all need a
result-buffer pattern compatible with downstream batches:

| Method | Internal kernel | Public C ABI entry | Parity reference (Python) | Parity reference (R) |
|--------|-----------------|--------------------|----------------------------|----------------------|
| Sparse SIMPLS (Chun & Keles 2010) | `fit_pls_sparse_simpls` | `p4a_sparse_simpls_fit` | `numpy-mirror` 1.26.4 | `spls` 2.3.2 |
| Domain-Invariant PLS | `fit_di_pls` | `p4a_di_pls_fit` | `numpy-mirror` 1.26.4 | none (documented gap) |
| Recursive / moving-window PLS | `run_recursive_pls` | `p4a_recursive_pls_run` | `scikit-learn` 1.4.2 | `pls` 2.8.5 |

## ABI delta

- New universal handle `p4a_method_result_t` with accessors
  `p4a_method_result_get_double_matrix`, `..._get_int_vector`,
  `..._get_scalar`, `p4a_method_result_destroy`.
- Three new `*_fit / *_run` entry points listed above.
- Total exported symbols: 124 → 131. C ABI minor 1 → 2.

## Parity gate

Cell sizes are fixed (small) per `benchmarks/parity_timing/runner.py`. RMSE
ratios are predictions vs each reference's predictions on a deterministic
dataset:

| Method | Py reference | R reference | RMSE rel (Py) | RMSE rel (R) | Status |
|--------|--------------|-------------|---------------|--------------|--------|
| `sparse_simpls` | numpy-mirror | `spls` 2.3.2 | 5.67e-03 | 2.51e-05 | both PASS |
| `di_pls` | numpy-mirror | — | 4.82e-03 | — | Py PASS, R none |
| `recursive_pls` | scikit-learn 1.4.2 | `pls` 2.8.5 | 1.23e-02 | 1.23e-02 | both PASS |

`no_r_reference` is flagged for DI-PLS because the nirs4all DiPLS is
*Dynamic Inner* PLS, not Domain-Invariant, and there is no widely
installable R port of the source-target projection formulation.

## Local gate (this checkpoint)

- 256 internal C++ tests (dev-release, local-asan-ubsan-gcc, local-ubsan-gcc).
- ABI symbol diff vs expected list: clean (131 symbols, all `p4a_*`).
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.
- `git diff --check`: clean.
- Parity gate: 5 PASS, 1 `no_r_reference` (documented).
