# Generic estimator roles (ABI 2.13) — design

Status: proposal revised after one adversarial Codex review (2026-09-26; findings folded into §2 and §7). Target: the next Methods release batch (1.0.22), which also carries the unreleased ABI 2.11/2.12 work.

## 1. Problem

Every `n4m` method must be usable from Python, R and JS/WASM through the same
life cycle — create with parameters, fit, transform or predict on new data,
export the fitted state, import it in another language — so that bindings and
the nirs4all / DAG-ML controllers consume a few standard interfaces instead of
one adapter per method. This is the sklearn `BaseEstimator` / `TransformerMixin`
/ `RegressorMixin` / `ClassifierMixin` contract, but owned by the C++ core and
exposed through the C ABI.

Audit at ABI 2.12 (212 catalog entries):

| Reachable through | Methods |
|---|---|
| Generic role with portable fitted state (N4MM model, N4MP pipeline, affine `n4m_model_from_method_result`) | 28 |
| Generic role without state or without serialization (2.11 splitters, 2.11 augmentations, 2.12 filters) | 38 |
| Per-method C functions only | the rest |

Consequences today:

- There is no native method manifest. Parameter names, defaults, required
  inputs and state formats are re-encoded by hand in each binding (R vectors
  and `strcmp` dispatch, Python `_C_PREFIX` tables and ~165 classes, JS
  `strcmp` chain). Defaults already drifted between bindings.
- Numerics leaked into bindings: Python computes classifier decisions, DS/PDS
  application and several numpy Ridge/PLS fits; R has an affine fallback.
- Many estimators are train-only (no out-of-sample predict), transformer state
  lives only in native handles (not picklable, not `saveRDS`-able), selectors
  have no apply step, classifiers have no native `predict_proba`.
- N4MP re-implements MSC/EMSC/ASLS/OSC/EPO/detrend/Norris-Williams/wavelet
  instead of calling the per-method kernels; only SNV and Savitzky-Golay are
  checked equal.

## 2. Decisions

### D0 — Every catalog entry is an estimator or a procedure

Each catalog entry declares `kind: estimator` or `kind: procedure`.

- *Estimators* have a fitted state and a life cycle (D1–D6): transformers,
  regressors, classifiers, selectors, sample filters.
- *Procedures* are one-shot computations with no reusable state: splitters,
  train-only augmentations, diagnostics, sweeps and HPO campaigns, metrics.
  They are called through one generic entry point,
  `n4m_procedure_run(ctx, method_index, params, inputs, out_result)`, which
  validates named parameters and inputs against the manifest and returns a
  `n4m_method_result_t` (splitters and augmenters additionally keep typed
  outputs, D5).

"Every method usable in Python, R and WASM" therefore means: every estimator
through the estimator life cycle and every procedure through the procedure
call, both driven by the same manifest. No entry is exempt.

### D0b — Typed role interfaces, not one catch-all interface

The surface is a set of **typed interfaces**, in the scikit-learn / torch
spirit: each role has its own operations and an input/output signature, and
that signature is what types the nirs4all / DAG-ML controllers and therefore
the Studio and Web nodes. A method implements one or more roles (PLS is both a
regressor and a transformer); it never exposes operations of a role it does not
declare.

| Role | Analogy | Operations | Signature (ports) | DAG-ML node |
|---|---|---|---|---|
| transformer | `TransformerMixin` / `nn.Module` X→X | fit, transform, export | Data[n,p] (+Target if supervised) → Data[n,k] | `transform` |
| selector | `SelectorMixin` | fit, selected_indices, transform (column subset), export | Data[n,p] (+Target) → Data[n,k⊆p] | `transform` (shape changing) |
| regressor | `RegressorMixin` | fit, predict, export | Data[n,p] + Target[n,q] → Prediction[n,q] | `model` |
| classifier | `ClassifierMixin` | fit, predict_labels, decision_function, predict_proba (when defined), classes, export | Data + Labels[n] → Prediction labels[n] / scores[n,c] | `model` |
| sample filter | outlier detector (`fit_predict` mask) | fit, apply_mask, export | Data (+Target) → keep mask[n], train only | `exclude` |
| splitter (procedure) | `BaseCrossValidator` | split | Data (+Target, groups) → folds | `split` |
| augmenter (procedure) | train-only sampler | augment | Data (+Target) → Data', train only | `augmentation` |
| generic procedure | function | run | inputs → MethodResult | none (diagnostics, sweeps) |

In C the stateful roles share one opaque `n4m_estimator_t`, because ownership,
parameters and N4ME serialization are identical; the role is part of the
method identity (`roles` mask), each operation checks it, and the manifest
publishes the role, its signature and the DAG-ML node kind. Typing lives in
the contract, not in the handle: bindings expose one base class per role
(`NativeTransformer`, `NativeSelector`, `NativeRegressor`,
`NativeClassifier`, `NativeSampleFilter`, `NativeSplitter`,
`NativeAugmenter`), multi-role methods inherit several, and controller
manifests are derived from the same role → node-kind mapping.

### D1 — One estimator handle over a closed, compiled adapter table

Add one opaque handle, `n4m_estimator_t`, created from a `method_id`. Internally
each supported method registers a C++ adapter implementing a small interface:

```cpp
namespace n4m::estimator {
struct FitInputs;            // see D4
class Writer; class Reader;  // N4ME state blocks, see D6

class Adapter {
 public:
  virtual ~Adapter() = default;
  virtual n4m_status_t fit(n4m_context_t*, const FitInputs&) = 0;
  virtual n4m_status_t transform(n4m_context_t*, const View& X, MutView out) const;   // TRANSFORMER, SELECTOR
  virtual n4m_status_t predict(n4m_context_t*, const View& X, MutView out) const;     // REGRESSOR, CLASSIFIER scores
  virtual n4m_status_t predict_proba(n4m_context_t*, const View& X, MutView out) const;
  virtual n4m_status_t predict_labels(n4m_context_t*, const View& X, int64_t* out) const;
  virtual n4m_status_t selected_indices(Span<int64_t> out, int64_t* count) const;     // SELECTOR
  virtual n4m_status_t apply_mask(n4m_context_t*, const View& X, const View* Y, uint8_t* mask) const; // SAMPLE_FILTER
  virtual n4m_status_t output_shape(int32_t op, int64_t rows, int64_t* r, int64_t* c) const = 0;
  virtual n4m_status_t save_state(Writer&) const = 0;
  virtual n4m_status_t load_state(Reader&) = 0;
  virtual const n4m_method_result_t* fit_result() const;  // diagnostics, borrowed
};
}
```

Unimplemented operations return `N4M_ERR_UNSUPPORTED`. Adapters call the
existing kernels (the `*_state_*` C engines, `n4m_estimators_*_fit`,
`n4m_model_fit`); they never re-implement them.

Rejected alternatives:

- *Extend `n4m_model_t` (N4MM) to every method.* N4MM is a PLS-shaped record
  (algorithm/solver/deflation, latent arrays). Forcing transformers, selectors,
  filters and splitters into it breaks its meaning and its inspection contract.
- *Extend `n4m_pipeline_t` (N4MP) to every method.* N4MP is a transformer
  sequence with positional `double` parameters; it cannot express predictors,
  selectors, integer/array parameters or extra fit inputs.
- *Open plugin registration.* Not needed: every method is compiled in, and a
  closed table keeps the ABI and the serialized formats verifiable.

N4MM and N4MP stay. They become state encodings embedded by the adapters
that need them (D6), and `n4m_model_*` / `n4m_pipeline_*` keep working.

### D2 — The catalog is the manifest; the library exposes it

The catalog entry (authored in `catalog/methods.yaml`, split into
`catalog/methods/<id>.yaml`, schema `catalog/schema/method.json`) gains an
`estimator` block. It is the authoring source; the **compiled table is the only
runtime authority**: bindings generate their classes from a JSON dump produced
by the built library (`n4m_cli --manifest-json`), never from the YAML directly.

```yaml
estimator:
  kind: estimator                    # estimator|procedure (D0)
  roles: [transformer]               # estimators: transformer|regressor|classifier|selector|sample_filter
                                     # procedures: splitter|augmenter|generic
  params:
    - {name: n_components, type: int, default: 1, min: 1}
    - {name: scale, type: bool, default: true}
  inputs: {y: required}              # y: none|optional|required; groups, feature_groups,
                                     # sample_weight, blocks, axis, target_domain: none|optional|required
  state:                             # every field needed to transform/predict new rows
    format: n4me.osc.v1
    fields: [x_mean: f64[p], weights: f64[p,k], loadings: f64[p,k]]
    retains_training_rows: false
```

Capabilities are not declared by hand: they are derived from the adapter's
implemented operations and, for imported states, from the validated N4ME bytes
(the N4MM inspection rule).

A generator (`catalog/scripts/generate_estimator_manifest.py`) emits a
compiled table (`cpp/src/estimator/generated/manifest.inc`) with, for each
method, its id, roles, parameter descriptors, input requirements, flags and
the adapter factory. CI fails if the generated file is out of date, and
`validate_catalog.py` checks that every `estimator` block names an adapter
that exists.

Introspection ABI (library-owned static strings, never freed):

```c
typedef struct n4m_method_info_v1_t {
    uint32_t struct_size;
    const char* method_id;      /* "preprocessing.orthogonalization.osc" */
    const char* fq_name;        /* "n4m.transform.orthogonalization.osc" */
    uint32_t roles;             /* N4M_ROLE_* mask */
    uint64_t capabilities;      /* N4M_CAP_* mask, see D5 */
    int32_t n_params;
    const char* state_format;   /* "n4me.osc.v1" */
} n4m_method_info_v1_t;

typedef struct n4m_param_info_v1_t {
    uint32_t struct_size;
    const char* name;
    int32_t type;               /* N4M_METHOD_PARAM_INT|DOUBLE|BOOL|ENUM|INT_ARRAY|DOUBLE_ARRAY */
    int32_t has_default;        /* 0 = required parameter */
    int64_t default_length;     /* 1 for scalars, n for array defaults */
    double min_value, max_value;/* NaN when unbounded */
    int32_t n_choices;          /* ENUM */
    const char* const* choices;
} n4m_param_info_v1_t;

N4M_API n4m_status_t n4m_method_count(int32_t* out);
N4M_API n4m_status_t n4m_method_find(const char* method_id, int32_t* out_index);
N4M_API n4m_status_t n4m_method_info_v1(int32_t index, n4m_method_info_v1_t* out);
N4M_API n4m_status_t n4m_method_param_info_v1(int32_t index, int32_t param, n4m_param_info_v1_t* out);
N4M_API n4m_status_t n4m_method_param_default_double(int32_t index, int32_t param, double* out, int64_t cap, int64_t* count);
N4M_API n4m_status_t n4m_method_param_default_int(int32_t index, int32_t param, int64_t* out, int64_t cap, int64_t* count);
```

Descriptor rules follow the existing inspection contract: the caller sets
`struct_size` to the size it was compiled with; the library writes
`min(struct_size, sizeof(current))` bytes, zeroes that prefix on every error,
and rejects a `struct_size` smaller than the v1 layout with
`N4M_ERR_INVALID_ARGUMENT`.

Bindings generate their classes, R constructors and JS factories from this
introspection (at build time, from the same generated JSON exported by the
generator). They no longer carry method lists.

### D3 — Named, typed parameters; defaults owned by the core

```c
N4M_API n4m_status_t n4m_params_create(n4m_context_t*, int32_t method_index, n4m_params_t** out);
N4M_API n4m_status_t n4m_params_set_int(n4m_params_t*, const char* name, int64_t v);
N4M_API n4m_status_t n4m_params_set_double(n4m_params_t*, const char* name, double v);
N4M_API n4m_status_t n4m_params_set_bool(n4m_params_t*, const char* name, int v);
N4M_API n4m_status_t n4m_params_set_enum(n4m_params_t*, const char* name, const char* choice);
N4M_API n4m_status_t n4m_params_set_int_array(n4m_params_t*, const char* name, const int64_t* v, int64_t n);
N4M_API n4m_status_t n4m_params_set_double_array(n4m_params_t*, const char* name, const double* v, int64_t n);
N4M_API n4m_status_t n4m_params_get_*(...);   /* resolved value, default when unset */
N4M_API void n4m_params_destroy(n4m_params_t*);
```

Setters take no context; they return a precise status
(`N4M_ERR_INVALID_ARGUMENT` for unknown name, type or bound) and the named
diagnostic is reported by `n4m_params_validate(ctx, params)`, which
`n4m_estimator_create` and `n4m_procedure_run` also call, through the context
error buffer. Unknown names, wrong types and out-of-bounds values therefore
fail at `set`, with the message available at validation. Unset
parameters take the manifest default, so a Level 1 recipe
`{method_id, params}` yields the same estimator in every language.

### D4 — One versioned fit-input struct

```c
typedef struct n4m_fit_inputs_v1_t {
    uint32_t struct_size;
    const n4m_matrix_view_t* X;             /* required */
    const n4m_matrix_view_t* Y;             /* continuous targets */
    const int64_t* labels; int64_t n_labels; /* class IDs for classifiers */
    const double* sample_weight; int64_t n_sample_weight;
    const int64_t* groups; int64_t n_groups;              /* sample groups */
    const int32_t* feature_groups; int64_t n_feature_groups;
    const int32_t* block_sizes; int32_t n_blocks;        /* multiblock column partition */
    const double* axis; int64_t n_axis;                   /* wavelengths */
    const n4m_matrix_view_t* X_target;      /* target-domain / slave spectra */
    const n4m_matrix_view_t* Y_target;
    const n4m_validation_plan_t* plan;      /* folds for internal CV (selectors, AOM) */
    uint64_t seed;
} n4m_fit_inputs_v1_t;
```

Classifiers receive integer class IDs; the state stores the sorted unique IDs
(`n4m_estimator_classes`) and hosts map IDs to names in their envelope.
Kernels that take `int32_t` labels and a class count get them from the adapter
(remapped to 0..k-1). Wavelength-dependent methods receive `axis` here and
build their kernel at fit, instead of at construction.

`fit` checks the manifest input requirements and returns
`N4M_ERR_INVALID_ARGUMENT` with a context message naming the missing input.
Views may be strided; adapters whose kernel needs contiguous row-major data
get a private copy from one shared helper. This also lifts the contiguous-only
restriction of the 2.11/2.12 roles.

### D5 — Estimator life cycle and capabilities

```c
typedef enum { N4M_ROLE_TRANSFORMER = 1, N4M_ROLE_REGRESSOR = 2, N4M_ROLE_CLASSIFIER = 4,
               N4M_ROLE_SELECTOR = 8, N4M_ROLE_SAMPLE_FILTER = 16 } n4m_role_t;
/* procedures: kind = N4M_PROCEDURE_{GENERIC, SPLITTER, AUGMENTER} in n4m_method_info_v1_t */

/* capabilities */
#define N4M_CAP_OOS_TRANSFORM       (1ull << 0)  /* transform new rows */
#define N4M_CAP_OOS_PREDICT         (1ull << 1)
#define N4M_CAP_PREDICT_PROBA       (1ull << 2)
#define N4M_CAP_SERIALIZABLE        (1ull << 3)
#define N4M_CAP_SUPERVISED          (1ull << 4)
#define N4M_CAP_SEEDED              (1ull << 5)
#define N4M_CAP_TRAIN_ONLY          (1ull << 6)  /* augmenters: never applied at predict */
#define N4M_CAP_RETAINS_TRAINING_ROWS (1ull << 7)/* state embeds training data */
#define N4M_CAP_AFFINE              (1ull << 8)  /* predict == X @ B + b */
#define N4M_CAP_DECISION_FUNCTION   (1ull << 9)

N4M_API n4m_status_t n4m_estimator_create(n4m_context_t*, const char* method_id,
                                          const n4m_params_t* params /* NULL = defaults */,
                                          n4m_estimator_t** out);
N4M_API n4m_status_t n4m_estimator_fit(n4m_context_t*, n4m_estimator_t*, const n4m_fit_inputs_v1_t*);
N4M_API n4m_status_t n4m_estimator_is_fitted(const n4m_estimator_t*, int* out);
N4M_API n4m_status_t n4m_estimator_output_shape(const n4m_estimator_t*, int32_t op,
                                                int64_t n_rows, int64_t* rows, int64_t* cols);
N4M_API n4m_status_t n4m_estimator_transform(n4m_context_t*, const n4m_estimator_t*,
                                             const n4m_matrix_view_t* X, n4m_matrix_view_t* out);
N4M_API n4m_status_t n4m_estimator_predict(...same shape as transform...);
N4M_API n4m_status_t n4m_estimator_decision_function(...); /* method-defined scores, e.g. QDA log-likelihoods */
N4M_API n4m_status_t n4m_estimator_predict_proba(...);     /* only when the method defines probabilities */
N4M_API n4m_status_t n4m_estimator_predict_labels(n4m_context_t*, const n4m_estimator_t*,
                                                  const n4m_matrix_view_t* X, int64_t* out, int64_t n);
N4M_API n4m_status_t n4m_estimator_classes(const n4m_estimator_t*, int64_t* out, int64_t cap, int64_t* count);
N4M_API n4m_status_t n4m_estimator_selected_indices(const n4m_estimator_t*, int64_t* out, int64_t cap, int64_t* count);
N4M_API n4m_status_t n4m_estimator_apply_mask(n4m_context_t*, const n4m_estimator_t*,
                                              const n4m_matrix_view_t* X, const n4m_matrix_view_t* Y,
                                              uint8_t* mask, n4m_filter_stats_t* stats);
N4M_API n4m_status_t n4m_estimator_fit_result(const n4m_estimator_t*, const n4m_method_result_t** borrowed);
N4M_API n4m_status_t n4m_estimator_info(const n4m_estimator_t*, int32_t* method_index, uint64_t* capabilities);
N4M_API n4m_status_t n4m_estimator_get_params(const n4m_estimator_t*, n4m_params_t** out_copy);
N4M_API void n4m_estimator_destroy(n4m_estimator_t*);
```

Classifier labels, decision scores and probabilities are computed natively;
each classifier documents its score semantics in the manifest, and
`PREDICT_PROBA` is set only where probabilities are defined.

`fit_result` is borrowed: valid until the next `fit`, `load_state` or
`destroy` of the same estimator. All `_v1` input structs carry `struct_size`
set by the caller; unknown trailing fields are rejected, shorter known
layouts accepted.

Procedures with typed outputs keep dedicated dispatch calls, now driven by
method index and named parameters (no handle, no no-op `fit`):

```c
N4M_API n4m_status_t n4m_procedure_run(n4m_context_t*, int32_t method_index, const n4m_params_t*,
                                       const n4m_fit_inputs_v1_t*, n4m_method_result_t** out);
N4M_API n4m_status_t n4m_split_run(n4m_context_t*, int32_t method_index, const n4m_params_t*,
                                   const n4m_fit_inputs_v1_t*, int32_t fold, n4m_split_result_t* out);
N4M_API n4m_status_t n4m_augment_run(n4m_context_t*, int32_t method_index, const n4m_params_t*,
                                     const n4m_fit_inputs_v1_t*, n4m_matrix_view_t* out_X,
                                     n4m_matrix_view_t* out_Y /* paired-Y kinds only */);
```

Paired-Y augmentation (mixup, local mixup) requires the kernels to expose the
sampled partners and weights so Y is mixed with the same draw; that kernel
change is part of slice S4.

### D6 — N4ME fitted-state format

```
"N4ME" | format u32 = 1 | writer ABI (3 × u32)
method_id (u32 length + UTF-8) | params (canonical, sorted by name, typed)
capabilities u64 | n_features_in u64 | n_outputs u64
state block: u32 tag + u64 length + adapter bytes   (repeated)
FNV-1a-64 over all preceding bytes
```

- Adapters that already have a format embed it unchanged as a state block:
  affine and PLS regressors embed N4MM, N4MP-kind transformers embed their
  N4MP single-step payload. Every other adapter defines its own block, versioned
  by `state_format` in the manifest.
- Import validates magic, versions, checksum, the method id against the
  table, the parameters against the manifest, every state field's declared
  shape, and bounds, then calls `load_state`. Section lengths are u64; the
  default payload cap is 256 MiB, raised by the caller through
  `n4m_context_set_max_state_bytes` for large retained-row models (a GPR-PLS
  Cholesky factor alone is n² doubles: 64 MiB ≈ 2,900 training rows). The ABI mirrors N4MM/N4MP:
  `n4m_estimator_export_size`, `_export_to_buffer`, `_import_from_buffer`,
  and an allocation-free `n4m_serialization_inspect_estimator_v1`.
- States that embed training rows (`RETAINS_TRAINING_ROWS`: kernel PLS,
  LW-PLS, GPR-PLS; LW-PLS also needs a new out-of-sample predict kernel, since
  today it only predicts its training rows) export only when the caller passes
  `N4M_EXPORT_ALLOW_TRAINING_ROWS`; otherwise export fails with a message.
  This keeps data retention an explicit decision.

N4ME is the Level 2 unit of portability: the same bytes predict identically in
Python, R and WASM.

### D7 — Existing symbols

- ABI policy: minor releases are additive. Per-method C symbols
  (`n4m_transform_*`, `n4m_estimators_*_fit`, …) stay as the kernel-level API.
  Bindings stop using them for the generic path.
- The ABI 2.11 `n4m_splitter_run` / `n4m_augmentation_run` (on `main`, in the
  R `n4m` 1.0.21.9006 source) stay, implemented over the same code as
  `n4m_split_run` / `n4m_augment_run`.
- The ABI 2.12 filter-role symbols (`n4m_sample_filter_*`,
  `n4m_feature_filter_*`) were never distributed (verified 2026-09-26: they
  exist only on the unmerged branch of PR #37 and a local R library). They
  are replaced by the `SAMPLE_FILTER` / `SELECTOR` estimator roles; headers,
  the three symbol snapshots and the change log are rewritten together so the
  released 2.13 describes 2.12 as a superseded, unreleased draft. The
  composite filter is not carried over: composing filters is DAG-ML's job
  (D8).
- N4MP operator kinds are re-pointed at the per-method kernels, with an
  equivalence test per kind, so one implementation remains per method.

### D8 — What stays out of n4m

Composition (pipelines, branches, stacking, CV/OOF, selection) belongs to
DAG-ML. n4m provides estimators; DAG-ML controllers compose them. Model
families that need a host framework (torch, sklearn RF, ranger) are not n4m
estimators.

## 3. Bindings and controllers

- **Python**: one `NativeEstimator` base in `n4m` (`BaseEstimator` plus the
  mixin selected by role), with `get_params`/`set_params` from the manifest,
  `fit/transform/predict/predict_proba`, and `__getstate__`/`__setstate__`
  through N4ME. Named classes (`n4m.PLS`, `n4m.OSC`, …) are generated thin
  subclasses; existing hand-written classes are replaced when their method is
  covered, and the numpy fallbacks listed in §1 are deleted.
- **R**: one S3 class `n4m_estimator` (`n4m_estimator("…", …)`, `fit`,
  `predict`, `transform`, `n4m_export`, `n4m_import`); N4ME bytes kept in the
  object so `saveRDS`/`readRDS` rehydrate without refitting. Named
  constructors generated from the manifest. The `strcmp` dispatch and the
  hard-coded kind vectors are deleted.
- **JS/WASM**: one `NativeEstimator` class; `EXPORTED_FUNCTIONS` already come
  from the ABI snapshot. The `fitModel` name chain and coefficient-based
  prediction are deleted in favour of N4ME.
- **Controllers**: DAG-ML controller manifests for n4m nodes are generated
  from the n4m manifest (roles → node kinds, params → schema), so the nirs4all
  Python, nirs4all R and Core/WASM controllers accept every method with no
  per-method code. Level 1 recipes are `{method_id, params}`; Level 2 artifacts
  carry N4ME bytes. The trained envelopes v1–v7 converge on one N4ME-based
  envelope.

## 4. Coverage plan

Slices, each merged to `main` without release, in the order of what they
unlock:

| Slice | Content | Methods |
|---|---|---|
| S0 | Core: manifest generator + introspection, params, estimator handle, procedure call, N4ME, conformance harness; adapters for the 16 affine regressors and the N4MM PLS families | ~24 |
| S1 | Promote `weighted_pls`, `o2pls`; AOM/sweep results through `input_coefficients`; 26 selectors (apply = column subset) | ~37 |
| S2 | 58 transformers (34 stateless, 24 fitted, each with a full state spec: several kernels expose only opaque state today and need accessors); N4MP re-pointed at the kernels | 58 |
| S3 | Classifiers with native labels/decision/proba where defined; sample filters; splitters and 22 augmenters through the procedure dispatch | ~45 |
| S4 | Non-affine regressors (GPR-PLS, PLS-GLM, kernel PLS; LW-PLS with a new predict kernel), transfer (DS, PDS, robust DS, slope/bias, DI-PLS), augmenters with paired Y (kernel change) or axis | ~25 |
| S5 | Remaining procedures: diagnostics, sweeps, AOM/HPO campaigns, metrics through `n4m_procedure_run` | rest |

The final claim is per entry: estimator or procedure, languages, operations,
oracle result — never a bare 212/212.

## 5. Tests

- **Conformance suite (C++), generated from the manifest**: for every method,
  fit on a train split; transform/predict/proba on held-out rows; export →
  import → identical outputs (bitwise); parameter round trip; refusal of each
  missing required input; refusal of unsupported operations; seeded
  determinism.
- **Kernel equivalence**: estimator output equals the per-method C function
  output (and each N4MP kind equals its kernel).
- **Cross-binding matrix**: one fixture set produced by the C++ suite
  (inputs, params, N4ME bytes, outputs); Python, R and JS each load the same
  bytes, predict, and must match within the documented tolerance; each binding
  also exports and the other two import. The matrix (method × language ×
  operation) is published in `docs/parity/` and is the only basis for coverage
  claims.

## 6. Decisions on the review questions

1. Class labels: integer IDs in the core and in N4ME; hosts own the ID→name
   mapping in their envelope.
2. No JSON parser in the core; bindings generate typed setters from the
   manifest dump.
3. The 2.12 filter symbols are removed as one unreleased ABI revision (headers,
   snapshots and change log together), after the non-distribution check above.

## 7. Review record

One adversarial Codex review (2026-09-26) raised: LW-PLS has no out-of-sample
path; classifier inputs/probabilities were underspecified; procedures were
exempted instead of covered; state serialization and size limits were
underestimated; descriptor/lifetime rules were missing; array defaults, nested
configs and paired-Y augmentation were not expressible; and 2.12 removal needed
a release-boundary rule. All are addressed above (D0, D2–D7, §4).
