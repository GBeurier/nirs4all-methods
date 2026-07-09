# nirs4all-methods - gaps for the in-browser Web/WASM app

`nirs4all-web/studio-lite` is a full-WASM browser app that drives **libn4m** through the
emscripten convenience surface (`bindings/js/src/wasm_entry.c`, the `_n4m_wasm_*` helpers)
plus a curated catalog (`nirs4all-web/studio-lite/src/catalog/nodes.ts`, CI-gated against
`cpp/abi/expected_symbols_*.txt`). The items below are operators/capabilities that **exist
in libn4m but are not yet usable from the browser**, so the Web app either omits them or
falls back. None require changing the public ABI snapshot - the wasm helpers are
emscripten-only `_n4m_wasm_*` exports appended in `bindings/js/CMakeLists.txt`.

This is a forward-development backlog, not a bug list.

## 0. Held commits to reconcile with the in-flight AOM work
Two additive `bindings/js` commits are held on `release-readiness-fixes` (NOT pushed) while
the AOM core is being developed, to avoid colliding:
- `8d2371e` — MIR-PLS / MB-PLS / missing-aware NIPALS added to the generic model dispatcher.
- `4fc0a7a` — split-operator wasm dispatcher (Kennard-Stone / SPXY / KMeans / KBins-stratified).
The deployed demo runs the **staged** wasm built from them. When AOM settles: reconcile +
push (or rebuild + re-stage the wasm against the finished AOM).

## 1. Non-coefficient model predict surface  *(blocks ~7 models in the editor)*
The generic dispatcher `n4m_wasm_model_fit` only supports models that expose an
**input-space coefficient triple** (`coefficients` + `x_mean` + `y_mean` [+ real
`intercept`]) so the browser predicts via the centred/affine form
(`n4m_wasm_model_predict_from_coeffs`). These **exported** models don't fit that contract,
so they're absent from the editor:
| model | fit symbol | why it doesn't fit | needs |
|---|---|---|---|
| PLS-LDA / PLS-QDA / PLS-logistic | `n4m_pls_lda_fit` / `_qda_fit` / `_logistic_fit` | classification heads expose decision scores / score-space, not input-space coeffs | a wasm `decision_function` / `predict_proba`-style predict |
| NPLS / ONPLS | `n4m_n_pls_fit` / `n4m_on_pls_fit` | multi-way (tensor) / loadings only | tensor input contract + predict |
| GPR-PLS | `n4m_gpr_pls_fit` | returns posterior/variance | predict returning mean (+ optional variance) |
| LW-PLS | `n4m_lw_pls_fit` | locally weighted — per-query, no global coeffs | "fit → portable state + predict(X_new)" surface |
| Kernel-PLS | `n4m_kernel_pls_fit` | dual coefficients | kernel-predict surface (store support + kernel params) |
| DI-PLS | `n4m_di_pls_fit` | needs a target domain | target-domain input |

**Proposed fix:** a generic `n4m_wasm_model_predict(model_token, fitted_state_blob, X_new) → Y`
so the browser no longer needs the coeff-triple shortcut; each model serializes a portable
fitted state the predict path consumes.

## 2. Preprocessing dispatcher — shape-changing + stateful gaps
The generic `n4m_wasm_pp_*` dispatcher only handles **shape-preserving** (n×p → n×p)
transforms, and only stateless ops + MSC (which got internal `*_get_state`/`*_set_state`).
- **Shape-changing** ops (crop, resample, edge-trimming derivatives, PCA/SVD feature
  projection, wavelet features, interval selection) need an output-shape query
  (`n4m_wasm_pp_out_cols(...)`) + a non-shape-preserving transform path. The catalog
  excludes them today.
- **Stateful** ops beyond MSC (EMSC, EPO, OSC, direct standardization,
  baseline-with-fitted-reference, COW/DTW alignment) need MSC-style internal state
  accessors (`*_state_n` / `*_get_state` / `*_set_state`) so the fitted state round-trips
  for predict-later (.n4a). Without them they can't be exposed safely.

## 3. AOM / POP — richer inputs
- The operator-bank is exposed, but `WeightedPLS` (needs a length-n weights vector) and
  multi-group `GroupSparsePLS` (needs feature group-ids) need an **auxiliary array param**
  the scalar NodeDef can't carry — a wasm fit variant taking the extra array.
- AOM branch/selection modes beyond the duplication-style screen (`by_source`) need
  multi-source X (see also dag-ml's multi-source gap).

## 4. Moments engine (infra — not an editor operator)
`n4m_moments_*` (μ, XᵀY, XᵀX) is infrastructure used inside AOM/PLS sweeps; it maps to no
user-facing editor operator. Not needed for the editor. A `n4m_wasm_moments_*` surface would
only matter if the in-browser "moment engine" tiers become a goal (cf. the gpu-moment-engine
plan).

---
*Source: the nirs4all-web/studio-lite build (2026-06). The browser/WASM binding is the layer to grow;
the C++ core already implements the numerics.*
