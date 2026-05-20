# Phase 3 — Stateful scatter preprocessings + `_fit/_transform` contract

## Goal

1. **Establish the stateful-operator ABI contract**: every operator with learnable parameters follows `_create → _fit → _transform → _inverse_transform (optional) → _destroy`. Calling `_transform` before `_fit` returns `N4M_ERR_NOT_FITTED`. Calling `_fit` twice overwrites (sklearn semantics).
2. **Implement 4 stateful preprocessings**: MSC, EMSC, Baseline (column centering), Derivate.
3. **Clean up the public header**: delete the inherited pls4all orphan declarations (Codex R3 + R4) — config, pipeline, model, operator_bank, gating_strategy, AOM, validation_plan, method_result, and `n4m_array_t` accessors.
4. **Adopt new process artefacts** (Codex P1-P3): `docs/reviews/PHASES.md` aggregate verdict index, `docs/reviews/DEFERRALS.md` cross-phase tracker.

## Out of scope

- Polynomial baseline (Phase 5 baseline correction family).
- Re-implementing `n4m_array_t` (caller-allocates pattern stays; revisit at Phase 9 if CARS/MCUVE genuinely needs variable-shape output).
- `_state_to_bytes / _from_bytes` per operator (deferred to Phase 11 pipeline-level serialization).

## Operators

| Operator | What `_fit` does | What `_transform` does | Inverse? | Stateful fields |
|----------|------------------|-----------------------|----------|-----------------|
| **MSC** (MultiplicativeScatterCorrection) | computes `ref = mean(X_train, axis=0)` (length p) | for each row: regress row on `[1, ref]` to get `(a, b)`; apply `(row - a) / b` | yes (`row * b + a`) | `ref[p]` (double array) |
| **EMSC** (ExtendedMSC) | computes `ref = mean(X_train, axis=0)` | regress on `[1, λ, λ², …, λ^d, ref]` to get coefficients; subtract poly + bg, then divide by `ref` weight | no (poly subtraction not invertible without storing all coefs per-sample) | `ref[p]`, `degree`, normalised wavelength axis `wl[p]` |
| **Baseline** (CenteringTransformer) | computes `mean = mean(X_train, axis=0)` | `X - mean` (per column) | yes (`X + mean`) | `mean[p]` (column means) |
| **Derivate** | (none — stateless, but lifecycle still `_create / _fit (no-op) / _transform`) | first/second finite-difference derivative along axis=1 | no | `order`, `delta` |

Notes on nirs4all parity:
- nirs4all's `MultiplicativeScatterCorrection` (`operators/transforms/nirs.py`) uses `np.polyfit(ref, row, 1)` to get `(slope, intercept)`, then `(row - intercept) / slope`. We replicate with `numpy.linalg.lstsq` equivalents (least-squares on `[1, ref]`).
- nirs4all's `ExtendedMultiplicativeScatterCorrection` regresses on `[1, λ, λ², …, λ^d, ref]` then subtracts the poly terms and divides by the ref coefficient.
- nirs4all's `Baseline` in `operators/transforms/signal.py` is column-wise mean centering (matches `sklearn.preprocessing.StandardScaler(with_std=False)`).
- nirs4all's `Derivate` in `operators/transforms/scalers.py` is `np.diff(X, n=order, axis=1) / delta**order`. Output shape is `(rows, cols - order)`.

The Derivate output shape change (cols → cols - order) is a problem for the caller-allocates pattern when the caller doesn't know `order` ahead of time. **Decision**: Derivate takes the output buffer pre-sized to the correct shape, OR we expose a `n4m_pp_derivate_output_cols(order, input_cols)` helper. Use the helper — small additive ABI surface.

## C ABI surface added

Per-operator (using MSC as example):
```c
typedef struct n4m_pp_msc_handle_t n4m_pp_msc_handle_t;
N4M_API n4m_status_t n4m_pp_msc_create(n4m_pp_msc_handle_t** out);
N4M_API void         n4m_pp_msc_destroy(n4m_pp_msc_handle_t* h);
N4M_API n4m_status_t n4m_pp_msc_fit(n4m_pp_msc_handle_t* h, n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_pp_msc_transform(const n4m_pp_msc_handle_t* h,
                                          n4m_matrix_view_t X,
                                          n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_msc_inverse_transform(const n4m_pp_msc_handle_t* h,
                                                  n4m_matrix_view_t X,
                                                  n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_msc_is_fitted(const n4m_pp_msc_handle_t* h, int* out);
```
6 symbols per operator (vs 3 for stateless).

For Derivate (stateless-but-lifecycle-shaped):
```c
typedef struct n4m_pp_derivate_handle_t n4m_pp_derivate_handle_t;
N4M_API n4m_status_t n4m_pp_derivate_create(n4m_pp_derivate_handle_t** out,
                                            int32_t order, double delta);
N4M_API void         n4m_pp_derivate_destroy(n4m_pp_derivate_handle_t* h);
N4M_API n4m_status_t n4m_pp_derivate_fit(n4m_pp_derivate_handle_t* h,
                                         n4m_matrix_view_t X);  /* no-op, validates dims */
N4M_API n4m_status_t n4m_pp_derivate_transform(const n4m_pp_derivate_handle_t* h,
                                               n4m_matrix_view_t X,
                                               n4m_matrix_view_t out);
N4M_API int64_t n4m_pp_derivate_output_cols(int32_t order, int64_t input_cols);
```

EMSC + Baseline follow MSC's shape (Baseline has `inverse_transform` since centering is reversible; EMSC does not).

ABI additions: **MSC (6) + EMSC (5, no inverse) + Baseline (6) + Derivate (5) = 22 new symbols**. Total: 57 → **79**.

## n4m.h cleanup (Codex R3 + R4)

Delete from `cpp/include/n4m/n4m.h`:
- §7-§15 entirely: `n4m_algorithm_t / solver_t / deflation_t / operator_kind_t / gating_mode_t / model_array_t / n4m_config_*`, `n4m_operator_bank_*`, `n4m_gating_strategy_*`, `n4m_pipeline_*`, `n4m_model_*`, `n4m_aom_*`, `n4m_validation_plan_*`, `n4m_method_result_*`, `n4m_component_coefficients_*`.
- `n4m_array_t` opaque type + 4 accessor declarations (`_dtype, _shape, _view, _free`). All currently orphan (no implementations in `cpp/src/`).
- The static_asserts at the end of §16 that reference removed enums.

Keep:
- §1 status codes
- §2 backend + dtype enums (used by context/matrix_view)
- §3 matrix view
- §4 context lifecycle
- §5 (new) "ABI naming convention" comment block listing reserved prefixes (`n4m_pp_*`, `n4m_split_*`, `n4m_filter_*`, `n4m_aug_*`, `n4m_util_*`, `n4m_rng_*`, `n4m_metric_*`).
- §6 RNG (Phase 1)
- §7 Stateless preprocessings (Phase 2)
- §8 (new) Stateful preprocessings (Phase 3)
- §16 ABI guard rails — trimmed to remove dead static_asserts

After cleanup, `n4m.h` should be **~600 lines** (down from 2193).

## Files to create

- `cpp/src/core/preprocessing/scatter/msc.{c,h}`
- `cpp/src/core/preprocessing/scatter/emsc.{c,h}`
- `cpp/src/core/preprocessing/scaling/baseline.{c,h}` — column centering
- `cpp/src/core/preprocessing/derivatives/derivate.{c,h}` (new subdirectory)
- `cpp/src/c_api/c_api_preprocessing.cpp` — append 4 operators (22 new wrappers)
- `cpp/tests/test_preprocessing_stateful.cpp` — 8 tests (smoke + parity per operator) + 4 `N4M_ERR_NOT_FITTED` tests
- `parity/python_generator/scripts/generate_phase3_fixtures.py` — extends the Phase 2 generator
- `parity/fixtures/{msc,emsc,baseline_center,derivate}_v1.json`
- `docs/algorithms/{msc,emsc,baseline_center,derivate}.md`
- `docs/reviews/PHASES.md` — aggregate Phase 0-3 verdict table
- `docs/reviews/DEFERRALS.md` — cross-phase deferral tracker
- `docs/reviews/phase-3/{codex-pre.md, opus-post.md}` — pre + post review transcripts

## Parity references

- nirs4all 0.8.5: `MultiplicativeScatterCorrection`, `ExtendedMultiplicativeScatterCorrection` (`operators/transforms/nirs.py`); `Baseline` (`operators/transforms/signal.py`); `Derivate` (`operators/transforms/scalers.py`).
- numpy: `np.polyfit`, `np.linalg.lstsq`, `np.diff`, `np.mean`.
- Tolerance: 1e-12 abs / 1e-13 rel for closed-form ops (Baseline, Derivate); 1e-10 / 1e-11 for least-squares ops (MSC, EMSC — LAPACK gelsd is the gold standard but its tiny pivoting differences cost a couple of ULPs).

## Acceptance criteria

- ✅ Build clean: `cmake --build --preset dev-debug` → 0 warnings.
- ✅ All 4 operators implement `_fit/_transform` contract; calling `_transform` before `_fit` returns `N4M_ERR_NOT_FITTED`.
- ✅ Tests pass: 34 → 34 + (4 × 2 parity + 4 × 1 not-fitted) = 34 + 12 = **46/46**.
- ✅ n4m.h shrunk from 2193 → ~600 lines (orphan PLS declarations removed).
- ✅ ABI: 57 → **79 symbols** exported. ABI minor bump 1.2.0 → 1.3.0.
- ✅ `docs/reviews/PHASES.md` and `docs/reviews/DEFERRALS.md` created.
- ✅ Opus post-review transcript at `docs/reviews/phase-3/opus-post.md`.
- ✅ Commit + push, CI green.

## Verification

```bash
cd /home/delete/nirs4all/nirs4all-methods
cmake --build --preset dev-debug
./build/dev-debug/cpp/tests/n4m_tests   # 46/46 pass
./build/dev-debug/cpp/cli/n4m_cli --selfcheck
nm -D --defined-only build/dev-debug/cpp/src/libn4m.so.1.3.0 | awk '$2=="T" {print $3}' | sort -u | wc -l  # 79
wc -l cpp/include/n4m/n4m.h  # ~600
```

## Next phase

[Phase 4 — Derivatives & smoothing](phase-4-derivatives.md): SavitzkyGolay, FirstDerivative, SecondDerivative, NorrisWilliams, Gaussian.
