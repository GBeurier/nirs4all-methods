# Phase 6f - Public AOM/POP C ABI

Status: shipped locally as `phase-6f-public-aom-pop-abi`.

Goal: expose the internal `select_aom_global` and `select_aom_per_component`
kernels through the stable C ABI, without leaking C++ containers. Add a
Python ctypes smoke that drives the new surface end-to-end against the
`nirs4all/bench/AOM_v0/aompls` parity oracle.

Delivered:

- New opaque handles:
  - `p4a_validation_plan_t` with create / destroy / set_n_samples /
    add_fold / get_n_samples / get_n_folds.
  - `p4a_aom_global_result_t` with create-by-`p4a_aom_global_select`,
    destroy, plus typed getters for `n_operators`, `max_components`,
    `selected_operator_index`, `selected_n_components`, `best_score`,
    `operator_kinds` (as `p4a_operator_kind_t*`), `operator_scores`,
    `rmse_curves` (row-major, `n_operators x max_components`) and
    `predictions` (row-major, `X.rows x Y.cols`).
  - `p4a_aom_per_component_result_t` with create-by-
    `p4a_aom_per_component_select`, destroy, plus typed getters for
    `n_operators`, `max_components`, `selected_n_components`,
    `best_score`, `operator_kinds` (as `p4a_operator_kind_t*`),
    `selected_operator_indices` (as `int32_t*`),
    `component_scores` (row-major, `max_components x n_operators`),
    `prefix_scores` and `predictions`.
- Public entry points `p4a_aom_global_select` and
  `p4a_aom_per_component_select` that consume the existing
  `p4a_context_t`, `p4a_config_t`, `p4a_operator_bank_t`,
  `p4a_matrix_view_t` and `p4a_validation_plan_t` handles, set
  `*out_result = NULL` on entry, and produce a result handle on success.
- ABI symbol list extended (additive only) and SOVERSION bumped to
  `abi.1.1.0`; project version bumped to `0.66.0`.
- Public ABI tests in `cpp/tests/abi/test_validation_plan_abi.cpp`,
  `cpp/tests/abi/test_aom_global_public_abi.cpp` and
  `cpp/tests/abi/test_aom_pop_public_abi.cpp` that exercise every getter,
  every NULL-pointer path, every invalid-input path (empty bank, empty
  plan, plan/X row mismatch, duplicate/overlap fold indices, out-of-range
  fold indices, unsupported solver, non-strict operator), and assert
  result pointer is nulled on every failure.
- Hardened internal `validate_plan` in `cpp/src/core/aom_selection.cpp`
  to reject duplicate or train/test-overlapping fold indices.
- Python ctypes wrappers in `bindings/python/src/pls4all/_aom.py`:
  `OperatorBank`, `OperatorKind`, `ValidationPlan`, `AomGlobalResult`,
  `AomPerComponentResult`, `aom_global_select`,
  `aom_per_component_select`. Wrappers are non-copyable and clean up via
  context manager / destructor. Result wrappers copy returned buffers
  into Python `list`s so lifetime is decoupled from the underlying
  result handle.
- `Config` Python wrapper extended with `center_x` / `scale_x` /
  `center_y` / `scale_y` / `store_scores` properties (already exposed by
  the C ABI).
- New driver script `bindings/python/smoke_aom_pop.py` that exercises
  every shipped AOM/POP fixture in `parity/fixtures/` through the public
  C ABI and compares scalars + buffers against the bench oracle.

Gate (local, all green):

- 209 C++ ABI/core tests (+8 new vs 201).
- ABI symbol diff against `cpp/abi/expected_symbols_linux.txt`: clean,
  additive only.
- CLI selfcheck.
- `ldd build/dev-release/cpp/src/libp4a.so`: only
  libc/libstdc++/libgcc/libm/loader.
- `git diff --check`: clean.
- UBSAN debug build: 209 tests pass.
- ASAN+UBSAN debug build: 209 tests pass, no leaks.
- Python smoke: 4 fixtures (3 global, 1 POP) pass parity vs.
  `nirs4all/bench/AOM_v0/aompls` through the new ctypes surface.

Deferred:

- POP holdout, approximate-PRESS and one-SE variants (Phase 6g).
- POP-NIPALS and AOM-NIPALS materialized variants.
- Soft / sparse / superblock AOM selection fixtures.
- Per-block and per-target AOM plans.
- R / MATLAB / JS / Julia / Java AOM/POP bindings (track in the binding
  roadmap once Phase 2 Python NumPy zero-copy lands).
