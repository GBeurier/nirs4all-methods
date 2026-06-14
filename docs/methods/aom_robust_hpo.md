# `aom_robust_hpo` - native AOM robust-HPO preprocessing screen

_Group_: **Diagnostic / AOM** · _ABI_: `n4m_model_selection_robust_hpo_fit`

## Description

`aom_robust_hpo` screens a fixed bank of strict-linear spectral preprocessing
chains and selects the best Ridge or PLS head by contiguous K-fold CV RMSE.
It is intended for fast, reproducible preprocessing-candidate campaigns where
the user wants the candidate-score table, the selected chain/head/parameter,
and a reusable linear prediction surface in the original input feature space.

Native v1 deliberately excludes stateful or sample-fitted preprocessings
(`SNV`, `MSC`, `EMSC`, `ASLS`, etc.). Those remain available in the Python
sklearn estimator `AOMRobustHPORegressor`, which fits each chain fold-locally.

## Backend Status

The public method is a native C ABI method and builds in both the regular CPU
and CUDA-enabled `libn4m` configurations. The preprocessing bank and Ridge head
are strict CPU kernels. The PLS head goes through the existing native PLS model
path, so a CUDA build can use the library's configured accelerated linear
algebra path where available.

This is not yet the lab-scale batched 200k-chain GPU grinder. It is the
catalogued product method: deterministic, source-free, ABI-stable, and suitable
for compact/wide preprocessing selection from Python or C.

## Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `profile` | `int` | `0` | `0=compact`, `1=wide` |
| `cv` | `int` | `5` | Contiguous folds, clipped to `n_samples` |
| `heads_mask` | `int` | `3` | Bitmask: `1=Ridge`, `2=PLS`, `3=both` |

## Result

The C ABI returns `n4m_method_result_t` with:

| Key | Shape | Meaning |
|-----|-------|---------|
| `predictions` | `n_samples x 1` | In-sample predictions after refitting the selected candidate |
| `coefficients_transformed` | `n_features x 1` | Linear coefficients in the selected transformed feature space |
| `input_coefficients` | `n_input_features x 1` | Selected transformed-space coefficients folded back into the original input feature space |
| `intercept` | `1 x 1` | Fitted intercept |
| `candidate_scores` | `n_candidates x 4` | `chain_id`, `head_id`, `param`, `mean_cv_rmse` |

Scalar diagnostics: `selected_chain_id`, `selected_head_id`,
`selected_param`, `selected_cv_rmse`, `n_chains`, `n_candidates`, `profile`,
`cv`, `n_samples`, `n_features`, `n_features_transformed`, `n_targets`.

The selected model can be replayed on any compatible input matrix as:

```python
y_hat = X @ res["input_coefficients"] + res["intercept"]
```

## Python Usage

```python
import numpy as np
import n4m

rng = np.random.default_rng(7)
X = rng.standard_normal((64, 256))
y = X[:, 8] - 0.4 * X[:, 19] + 0.05 * rng.standard_normal(64)

res = n4m.aom_robust_hpo(X, y, profile="compact", cv=5, heads=("ridge", "pls"))
print(res["selected_chain_id"], res["selected_head_id"], res["selected_cv_rmse"])
print(res["candidate_scores"][:5])

np.testing.assert_allclose(
    X @ res["input_coefficients"] + res["intercept"],
    res["predictions"],
)
```

The native sklearn wrapper uses the same folded coefficients:

```python
model = n4m.NativeAOMRobustHPORegressor(profile="compact", cv=5).fit(X, y)
pred = model.predict(X_new)
diag = model.get_diagnostics()
```

## C ABI Usage

```c
n4m_context_t* ctx = NULL;
n4m_config_t* cfg = NULL;
n4m_method_result_t* res = NULL;
n4m_context_create(&ctx);
n4m_config_create(&cfg);

n4m_model_selection_robust_hpo_fit(ctx, cfg, &x_view, &y_view,
                       /*profile=*/0, /*cv=*/5, /*heads_mask=*/3, &res);

const double* scores = NULL;
int64_t rows = 0, cols = 0;
n4m_method_result_get_double_matrix(res, "candidate_scores",
                                    &scores, &rows, &cols);

n4m_method_result_destroy(res);
n4m_config_destroy(cfg);
n4m_context_destroy(ctx);
```

## Native Profiles

`compact` includes raw, detrend degree 1/2, six Savitzky-Golay-style
smooth/derivative variants, Norris-Williams, finite difference, and a few
strict-linear compositions.

Compact `chain_id` mapping:

| ID | Chain |
|----|-------|
| 0 | `raw` |
| 1 | `detrend1` |
| 2 | `detrend2` |
| 3 | `savgol_w5_p2_d0` |
| 4 | `savgol_w7_p2_d0` |
| 5 | `savgol_w7_p2_d1` |
| 6 | `savgol_w11_p2_d2` |
| 7 | `nw_s5_g5_d1` |
| 8 | `finite_diff1` |
| 9 | `detrend1_savgol_w7_p2_d1` |
| 10 | `detrend1_nw_s5_g5_d1` |
| 11 | `savgol_w5_p2_d0_finite_diff1` |

`wide` has 31 chains. It adds larger Savitzky-Golay windows, more
Norris-Williams variants, second finite difference, Gaussian/FCK variants,
Whittaker smoothing, and additional strict-linear compositions.

## Benchmarks

Timing script: `benchmarks/cross_binding/bench_aom_robust_hpo_timing.py`.
Latest checked-in CSV: `benchmarks/cross_binding/aom_robust_hpo_timing.csv`.
CUDA-build native smoke timing can be regenerated with:

```bash
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
python benchmarks/cross_binding/bench_aom_robust_hpo_timing.py \
  --native-only \
  --output benchmarks/cross_binding/aom_robust_hpo_timing_cuda_smoke.csv
```

The checked-in compact smoke timing on ABI `1.16.0` shows the native ABI path
and the Python sklearn preset selecting from the same 84 compact candidates.
CPU medians were 3.14 ms for 32 x 64, 16.33 ms for 64 x 128, and 60.47 ms for
96 x 256. The Python sklearn reference wrapper took 29.03 ms, 49.74 ms, and
263.59 ms on the same cells.

The CUDA-build native smoke medians were 506.16 ms, 313.89 ms, and 245.42 ms
on those cells. This validates the CUDA-enabled build path; it is not evidence
of fused GPU acceleration for compact AOM robust-HPO.
