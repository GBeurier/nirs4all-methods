# `sweep` — n4m.model_selection.sweep.sweep

_Namespace_: **`n4m.model_selection.sweep`** · _Fully-qualified_: `n4m.model_selection.sweep.sweep` · _Catalog id_: `utilities.sweep`

## API surface

**C ABI (ABI 2):** [`n4m_model_selection_sweep_run`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/ensemble.h#L19). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.model_selection.sweep import sweep_run`

**Signature:** [`sweep_run(X, y, *, cv: int = 5, fold_ids = None, ridge_lambdas = (0.01, 0.1, 1.0, 10.0), pls_components = None, heads = ('ridge',), center_x: bool | None = None, scale_x: bool | None = None, center_y: bool | None = None, scale_y: bool | None = None, score_only: bool = False, cuda_pls_parallel_folds: bool | None = None, cuda_pls_min_device_features: int | None = None, cuda_pls_many_batched: bool | None = None)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L6150)

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `X` | `—` | `required` |
| `y` | `—` | `required` |
| `cv` | `int` | `5` |
| `fold_ids` | `—` | `None` |
| `ridge_lambdas` | `—` | `(0.01, 0.1, 1.0, 10.0)` |
| `pls_components` | `—` | `None` |
| `heads` | `—` | `('ridge',)` |
| `center_x` | `bool \| None` | `None` |
| `scale_x` | `bool \| None` | `None` |
| `center_y` | `bool \| None` | `None` |
| `scale_y` | `bool \| None` | `None` |
| `score_only` | `bool` | `False` |
| `cuda_pls_parallel_folds` | `bool \| None` | `None` |
| `cuda_pls_min_device_features` | `int \| None` | `None` |
| `cuda_pls_many_batched` | `bool \| None` | `None` |

## Explanations

### Bibliographic source

No canonical scientific paper defines this orchestration utility; it is a reproducible enumeration and scoring surface.

### Mathematical principle

Evaluate every configuration in a declared finite grid under the supplied validation plan, collect scores and diagnostics, then expose a deterministic ranked table rather than silently choosing a configuration.

### Appropriate uses

Small, explicit hyperparameter or method grids where the complete candidate record is needed for audit and comparison.

### Limits and validation

A sweep increases selection-induced optimism; use nested or held-out evaluation. It cannot make an under-specified grid represent an untested method family and should not be confused with Bayesian optimization.

### Implementation

`n4m.model_selection.sweep`; C ABI `n4m_model_selection_sweep_run`.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/model_selection/sweep.py; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h

## Catalog note

Native Ridge/PLS candidate screen. Ridge is exact over row-additive moments when efficient and uses native dual kernels in wide regimes; compatible single-target NIPALS/regression PLS1 grids score from train/held-out moments, with a scalar host loop for medium-width CUDA builds and a device-resident cuBLAS component loop plus reused exact-CV fold workspace for very wide p>=1024 CUDA moment screens; materialized prefix scoring remains for other PLS regimes. PLS fit-cost counters are exposed as n_pls_moment_cv_fits, n_pls_moment_host_cv_fits, n_pls_moment_cuda_device_cv_fits, n_pls_materialized_cv_fits, n_pls_moment_final_fits, n_pls_moment_host_final_fits, n_pls_moment_cuda_device_final_fits and n_pls_materialized_final_fits so timing screens can distinguish moment-prefix fits, host/device PLS1 moment execution, and materialized fallback fits. Python exposes score_only=True for ranking passes that keep scores and selected ids while skipping selected-model buffers; score-only materialized fallback cells compute held-out SSE directly from linear fits when possible, and wide dual cross-kernel Ridge cells score directly from K_cross, to avoid held-out prediction-buffer allocation. This is not stream-parallel or fused batched IKPLS yet.

_Timing benchmark_: `benchmarks/cross_binding/bench_moment_sweep_timing.py`


_See also_: [methods index](index.md).