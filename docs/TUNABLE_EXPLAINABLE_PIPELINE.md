# Tunable & explainable pipelines — cross-language synthesis

**Status:** decision + planning synthesis (spans `dag-ml`, `nirs4all-methods`, `nirs4all`).
**Companions:** [`NATIVE_FINETUNING.md`](NATIVE_FINETUNING.md) (why native HPO),
[`FINETUNING_ROADMAP.md`](FINETUNING_ROADMAP.md) (the libn4m optimizer roadmap),
[`PIPELINE_FINETUNING.md`](PIPELINE_FINETUNING.md) (the earlier `object__attribute` plan).
This document supersedes the "how do we make it cross-language" parts of those.

> **Codex independent review:** done (high reasoning effort, all three repos).
> It **converges** with this recommendation, with three material corrections now folded
> throughout and summarised in §7. Headline: *"Adopt the estimator-facade approach.
> Do not build a tuner or SHAP engine inside dag-ml, and do not link dag-ml directly to
> libn4m. Compose them in the bindings."* — but *"the 'no dag-ml change at all' part is
> too optimistic; a small cross-language execution-contract addition is unavoidable."*

---

## 1. Objectives (owner's exact ask)

1. **`dag-ml` exposes a finetuning interface** (as it does for Optuna today).
2. **`nirs4all-methods` provides a cross-language, reproducible set of finetuning
   methods** for `nirs4all-core` in every language.
3. **The pipeline itself is tunable** — by Optuna *or* the n4m methods — exactly the
   way an `sklearn.Pipeline` is tunable.
4. **A `dag-ml` pipeline is usable as a `BaseEstimator`-equivalent in multiple
   languages**, so you can:
   - **finetune at (sub)pipeline scale**, and
   - **run an explainer (SHAP) at pipeline scale** — because today it is *impossible*
     to SHAP e.g. a Torch model buried in a complex pipeline.

**Hard constraint:** achieve this **without changing the repos' responsibilities**
and **without launching a big new `dag-ml` build if it can be avoided.**

---

## 2. The core insight — reify, don't build a tuner

`sklearn` works because a `Pipeline` **is** a `BaseEstimator`: `fit` / `predict` /
`set_params` / `get_params`. `GridSearchCV`, Optuna and SHAP all operate on it
*uniformly* — none of them looks inside. They only need that interface.

**So the whole feature reduces to one thing: give a `dag-ml` (sub-)pipeline the
same `fit/predict/set_params` interface.** Everything else is then *reuse*:

| Goal | Met by | Needs a native dag-ml HPO loop? |
|---|---|---|
| tune the pipeline (Optuna) | Optuna drives the estimator as a black box | ❌ |
| tune the pipeline (n4m, any language) | the libn4m ask/tell optimizer drives the estimator | ❌ |
| finetune a **sub**-pipeline | reify the sub-DAG as an estimator, tune it | ❌ |
| **SHAP at pipeline scale** | a model-agnostic explainer wraps `estimator.predict` | ❌ |

The last row is the key unlock: a model-agnostic explainer (SHAP
`KernelExplainer` / `PermutationExplainer`) needs **only `predict(X)`**. Wrap the
pipeline's `predict` and SHAP explains the *whole* pipeline — including a Torch
model inside it — because it never sees the Torch model, only the pipeline's
outputs. That is exactly why Torch-in-a-complex-pipeline can't be explained today,
and reification fixes it for free.

**dag-ml is the thing being *tuned/explained*, not the tuner/explainer.**

---

## 3. Current status (verified in code)

Legend: ✅ exists · 🟡 partial · ❌ missing.

### 3a. `nirs4all-methods` (libn4m) — the finetuning *kernels*
- ✅ Native **ask/tell optimizer**: 9 samplers (random, sobol[Tier-A vs scipy],
  lhs, ternary, ga, pso, cmaes, tpe, gp_ei) + 5 pruners (none, median, asha,
  hyperband, racing). Stable C-ABI (`n4m_optimizer_*`, `n4m_search_space_*`,
  `n4m_trial_*`), enum-value-only, cross-binding parity-tested (Track-Q golden
  traces + Sobol-vs-scipy + pruner-vs-reference). **Objective 2 is essentially met.**
- ✅ Conditional/nested search space (`CONDITION_IN/NOT_IN`, active-aware cascade — E2).
- ✅ Python binding (`n4m.model_selection.optimizer`).
- 🟡 **Delivery gaps** (Codex): the optimizer is **not yet in the shipping PyPI package**
  (only the dev tree); the host-side space compiler follows **map insertion order**
  (`n4m_engine.py:194`) rather than an explicit ordered schema; `save/load` returns
  `NOT_IMPLEMENTED`; some options are advertised but inert. → backlog M1/M2/M4.
- ❌ R / MATLAB-Octave / WASM bindings of the optimizer (thin marshallers; not yet).

### 3b. `dag-ml` — the coordinator / DAG
- ✅ **Host operator vtable** `RuntimeController::invoke(NodeTask) -> NodeResult`
  (host fits/transforms/predicts operators; phases FitCv/Refit/Predict/Explain).
  A JSONL process adapter + a one-shot Python `op_callback` transport the same
  contract. **This is the substrate a driver sits on.**
- ✅ Execution **bundles** + replay/predict; candidate **SELECT** + scoring C-ABI;
  nested (inner) CV per node (`inner_cv: NestedCvSpec`).
- ✅ Per-node **flat** param override `GenerationParamOverride{node_id, params}`.
- 🟡 `NodeKind::Tuner` DSL node — **passthrough only**: `sampler`/`n_trials` are inert;
  the only native "search" is Cartesian/Zip variant enumeration + CV-SELECT (grid,
  not adaptive). The tuner is compiled to an *external host controller* node.
- ❌ **No native ask/tell tuner** (no sampler/trial loop) and **no `dagml_tune`/`fit`/
  `predict`/`explain` execute C-ABI symbol** (68 symbols are control-plane: compile,
  plan, schedule, select, score, replay, contracts).
- ❌ **No link to nirs4all-methods** (zero `n4m_*` references; roadmap "largely-blocked").
- ❌ **No native SHAP/attribution** — only an `ExplanationBlock` payload contract +
  an `Explain` phase gate; the reference CLI emits a hardcoded mock.
- ❌ **No general sub-DAG fit** (`NodeKind::Subgraph` is a bare enum variant).
- ❌ **No nested `object__attribute` addressing** at the graph level (overrides are flat).

### 3c. `nirs4all` (Python)
- ✅ Optuna finetuning (mature) + a new **`N4MFinetuneManager`** (`engine:"n4m"`) that
  drives the libn4m optimizer, incl. `when`/`when_not` conditional attributes and
  `options` **operator choice** (operators in the search space) for sklearn-Pipeline
  model steps. Matches Optuna's optimum; 26 unit tests. **But Python-only orchestration.**
- 🟡 The `dag-ml` backend (`engine:"dag-ml"`, ADR-17) runs *plain* pipelines natively,
  but **force-routes any pipeline with `finetune_params`/`train_params` to the legacy
  Python engine** — so finetuning never touches dag-ml today.
- ❌ No `DagMlEstimator` (fit/predict/set_params) facade; no pipeline-scale SHAP over it.

### 3d. Verdict against the objectives
| # | Objective | Status |
|---|---|---|
| 1 | dag-ml exposes a finetuning interface | 🟡 grid-only + external-controller; **not** an adaptive/drivable estimator yet |
| 2 | nirs4all-methods cross-language finetuning methods | ✅ (optimizer done; non-Python bindings pending) |
| 3 | pipeline tunable by Optuna or n4m | 🟡 **Python only** (sklearn-Pipeline model step); not cross-language, not the dag-ml pipeline |
| 4 | dag-ml pipeline as BaseEstimator + (sub)pipeline finetune + pipeline-scale SHAP | ❌ not implemented |

---

## 4. Target architecture (reification — no responsibility change)

```
                 ┌─────────────────────────────────────────────┐
   any binding   │  DagMlEstimator  (fit / predict /            │  ← NEW, thin, per-binding
   (py/R/wasm)   │                   set_params / get_params)   │     facade over existing
                 └───────┬───────────────────────────┬─────────┘     dag-ml execution
                         │ set_params(node__p=…)      │ predict(X)
                         ▼                            ▼
      ┌──────────── dag-ml (UNCHANGED responsibility) ────────────┐
      │ compile→plan→refit(fit_cv, OOF)→bundle ; predict on bundle │
      │ per-node override · host operator vtable · SELECT/score    │
      └────────────────────────────────────────────────────────────┘
                         ▲                            ▲
        ask config       │ tell(OOF score)            │ predict(X_perturbed)
                 ┌───────┴────────┐          ┌────────┴──────────┐
                 │ optimizer      │          │ SHAP / explainer  │  ← reuse; only needs predict
                 │ Optuna (py) OR │          │ (host-side, any   │
                 │ libn4m (any)   │          │  language)        │
                 └────────────────┘          └───────────────────┘
```

- **Finetuning at (sub)pipeline scale** = *propose config → `set_params` → `fit`(CV/OOF)
  → score → `tell`*. The estimator is a black box; the optimizer drives it from
  outside. The **HPO algorithm stays in n4m**, the **execution stays in dag-ml**;
  only ~thin glue wires them.
- **Explainability at pipeline scale** = a model-agnostic explainer wraps
  `estimator.predict`. No SHAP engine in dag-ml; works for Torch-in-pipeline.

**Responsibilities are preserved:**
- `dag-ml` → still the coordinator/executor. The `DagMlEstimator` *packages* its
  existing execute; it is not a new responsibility (and can live in the bindings).
- `nirs4all-methods` → still the numeric + optimizer kernels. Unchanged.
- `nirs4all` (host controllers) → owns the estimator facade + finetune loop + SHAP
  wrapper. This is exactly the "Python-exclusive controllers" layer of the North Star.

---

## 5. Backlog (minimal-first; converged with Codex)

The one correction vs. the first draft: dag-ml **does** need a *small* execution-
contract slice (D-items) — it cannot yet return **reusable fitted state** or **real
predictions from replay**. This is an *extraction/export* job (the machinery exists —
`RuntimeController::invoke`, `NodeTask`/`NodeResult` carrying preds+artifacts+lineage,
the scheduler's fit/refit, `ExecutionBundle`), **not** a new subsystem, and explicitly
**not** a tuner or SHAP engine.

### dag-ml — small, unavoidable (`D`)
| ID | Item | Size |
|---|---|---|
| **D1** | **Promote one shared training operation into core** (compile→plan→FIT_CV→score/select→optional REFIT→capture artifacts/caches→bundle) returning a `TrainingOutcome` {effective plan, bundle, scores + terminal prediction blocks, lineage, prediction-cache payloads, artifact records}. The CLI (`main.rs:~2951`) and Python (`in_process.rs:~520`) each reproduce this privately today. Include a `refit=false` eval mode so HPO trials don't needlessly refit. | M |
| **D2** | **Expose training via the C ABI** — stateless JSON-in/JSON-out over the existing vtables; **no long-lived Rust estimator handle**. | S |
| **D3** | **Real replay output**: `dagml_replay_execute_json` currently returns only counts (`ReplayExecutionSummary`, `lib.rs:~4559`); add a `ReplayOutcome` with terminal + aggregated predictions, explanations, lineage. | S |
| **D4** | **Estimator output binding**: `{node_id, port, target/class, aggregation}` (or v1: exactly one unambiguous terminal prediction sink). | S |
| **D5** | **Canonical parameter addressing** — structured `{node_id, namespace∈{operator,fit,structural}, path:[…]}` (NOT `node__subparam` as the wire format). Structural change = clone template → apply patches → **recompile → invalidate fitted state**. | S |

### nirs4all-methods — finish delivery (`M`)
| ID | Item | Size |
|---|---|---|
| **M1** | **Ship the optimizer in the actual PyPI package** (today it's only importable from the dev tree / a hacked venv). | S |
| **M2** | **Ordered search-space schema**: canonicalize ordered axes / categorical values / conditions with an **explicit order** — the current Python compiler follows *mapping insertion order* (`n4m_engine.py:194`), which is not a sound cross-language contract. Validate duplicate names, unknown condition parents, cycles. Implement-or-reject currently-inert options. | S |
| **M3** | **≥1 non-Python optimizer wrapper** (R or WASM) + **cross-binding golden proposal traces** driven by the same score tape (extends Track-Q). | M |
| **M4** *(defer)* | `save/load` (currently `NOT_IMPLEMENTED`, `c_api_optimization.cpp:480`) — needed for resumable Studio jobs, not for a first release. Do **not** grow `n4m_finetune_estimator` into a host-callback runner. | S |

### nirs4all (host) — the estimator + one shared objective (`N`)
| ID | Item | Size |
|---|---|---|
| **N1** | **New `DagMLPipelineEstimator`** (`nirs4all/sklearn/dagml_pipeline.py`) — sklearn-cloneable; stores template config pre-fit; `get_params(deep=True)` exposes explicit graph params; `set_params` invalidates `plan_`/`bundle_`/artifacts; `fit`→D1/D2, `predict`→D3; explicit scorer + `predict_proba`; accepts identities/groups/metadata/multi-source, not just ndarray `X`. **Do NOT retrofit `NIRSPipeline`** (predict-only; `fit()` raises, `pipeline.py:226`). | M |
| **N2** | **Generic `PipelineObjective`** used by **both** Optuna and n4m: *ask → clone_unfitted → with_params → leakage-safe CV → tell → refit winner*. Reuse `N4MFinetuneManager`'s space compiler, but **separate optimizer-driving from terminal-model training** (it currently calls private controller build/train/evaluate). | M |
| **N3** | **Pipeline-scale SHAP** over the fitted facade's `predict` (model-agnostic Kernel/Permutation). Extending `nirs4all.explain()` to accept the estimator is nice UX, not required. | M |
| **N4** | **Thin cross-language finetune glue** in R / WASM / MATLAB — the tiny *ask→with_params→cv→tell→refit* loop wiring the libn4m optimizer to the local `DagMLPipelineEstimator`. Acceptable per §6 **iff** all bindings consume the same canonical `SearchSpaceSpec`, dag-ml eval op, score direction/aggregation, failure/pruning rules, and conformance tape. | M each |

### Deferred (explicitly NOT in the first feature)
- Arbitrary **embedded** sub-DAG extraction (needs a small `GraphSliceSpec` compiling to a standalone `GraphSpec`; `NodeKind::Subgraph` is a stub). — see §6.
- **Structural** search (operator replacement / branch activation / variable-length) — that's template *materialization + recompile*, a separate feature from value tuning.
- Parallel adaptive HPO, wall-clock timeouts, native study persistence.

**Smallest credible milestone** (Codex's, matches this backlog):
1. dag-ml: D1+D2+D3 (shared train op + C-ABI training + real replay output).
2. nirs4all: N1 + N2 (trainable estimator + one CV objective for Optuna *and* n4m).
3. nirs4all-methods: M1 + M3 (optimizer in shipping Python + one 2nd language + score-tape conformance).
4. Demonstrate: complex branch/stacking `fit→predict`; sklearn clone/CV; sequential n4m HPO; Optuna HPO through the **same** objective; Kernel SHAP on raw features with a **buried Torch model**.
5. Defer embedded slicing, structural search, parallel HPO, study persistence.

---

## 6. The one genuine decision — where the finetune loop lives

- **Option A (recommended, small):** host-side thin glue (B5). The loop is *wiring*
  two already-cross-language C-ABI objects (optimizer + estimator). The HPO algorithm
  (n4m) and the execution (dag-ml) are shared; only the ~30-line loop is per-binding,
  and the *search space* is declared once (B2). This honours "no responsibility
  change + avoid big dag-ml work."
- **Option B (bigger, later):** the native dag-ml `Tuner` coordinator (B7). Removes
  even the per-language glue, but is the large dag-ml build the owner wants to avoid,
  and it nudges a responsibility (HPO search) into dag-ml that its current migration
  split deliberately keeps host-side.

**Recommendation:** ship Option A; keep Option B as an explicit, deferred escape
hatch justified only by measured per-language duplication.

**Codex reinforces Option A, strongly:** *"A native callback runner would create
Python GIL, R unwinding, MATLAB lifetime and WASM async/reentrancy problems while
moving host orchestration into the numerical-methods repository. Ask/tell is already
the shared native helper."* The loop is tiny; the hard semantics (fold isolation, OOF,
scoring, refit, artifact lifecycle) live behind the **shared dag-ml evaluator
contract**, not in the loop. Per-language glue is acceptable **iff** all bindings
consume the same canonical `SearchSpaceSpec`, dag-ml eval op, score direction/
aggregation rules, trial failure/pruning rules, and conformance score tape.

---

## 7. Codex independent review — verdict (folded in)

**Verdict: adopt the estimator-facade approach.** Converges with §2–§6. Do **not**
build `dagml_tune`, a SHAP algorithm in dag-ml, a direct dag-ml→libn4m dependency, or
a long-lived callback-owning Rust estimator. Compose in the bindings.

Codex's per-goal scorecard:

| Goal | Codex verdict |
|---|---|
| dag-ml exposes finetuning | Yes — as a parameterizable, **CV-evaluable estimator interface**, not an optimizer |
| n4m provides reproducible methods | Yes — the ask/tell ABI is the right shared state machine |
| Optuna **or** n4m tunes a pipeline | Yes — both drive the **same** estimator objective |
| Pipeline-scale SHAP | Yes — for model-agnostic SHAP over a **deterministic, row-preserving** `predict` |
| Arbitrary embedded sub-DAG estimator | **Not yet** — requires explicit graph slicing (`GraphSliceSpec`) |
| Structural pipeline search | **Not** solved by flat overrides — needs materialization/recompile |

**The one correction to this doc's first draft** (now applied in §4/§5): the
"no dag-ml change at all" claim was too optimistic. dag-ml's common ABI cannot yet
(1) create reusable fitted pipeline state, or (2) return real predictions from replay
(it returns counts). Both are small extraction/export jobs (D1–D3), because the Rust
core already has `RuntimeController::invoke`, `NodeTask`/`NodeResult` (carrying preds/
explanations/artifacts/lineage), the scheduler's fit/refit, and `ExecutionBundle`.
It confirmed everything **else** is bindings/host work, and that the fitted state
should be a serializable `(template, effective ExecutionPlan, ExecutionBundle, output
binding, host-owned artifacts)` package — matching dag-ml's existing ownership boundary.

Codex also flagged the exact place the current Python path falls short of goal 4:
`nirs4all/pipeline/explainer.py:113,199` captures and explains **one inner model**,
not the complete pipeline — which pipeline-scale SHAP over `estimator.predict` fixes.

---

## 8. Risks & hard rules (sharpened by Codex)

**Leakage / OOF — the load-bearing rule.** **Never** `fit(X,y) → score(X,y) → tell`.
Use `fit(outer_train) → score(outer_validation)`, **or** a dag-ml CV evaluation that
returns a leakage-safe OOF score. Final refit of the winner happens **only after
selection**. The outer optimizer owns outer validation; any dag-ml internal CV /
stacking / early-stopping must operate strictly inside outer-train. Preserve groups,
repetitions, sample identities, source layout and metadata when subsetting. Trial
caches + artifact namespaces must include (param fingerprint, data fingerprint, fold
id, trial id, seed). **Never reuse a fitted handle between candidates.** A trial
failure → `tell_result(FAILED)`, not an artificial `inf` (unless that policy is
explicitly standardized).

**Parameter addressing.** Separate graph/control params, operator constructor params,
and fit-only kwargs. Topology/operator replacement = **structural materialization +
recompile**, not a value patch. Never mutate an already-fitted plan or bundle. Do
**not** parse raw node ids with `__` — keep a **reversible alias table**. Expose only
explicitly-declared graph/controller params; dag-ml cannot introspect hidden defaults
inside opaque host objects. (A fixed graph with nested *value* params is cheap;
searching over operator replacement / branch activation / variable length is the
separate materialization feature — deferred.)

**SHAP correctness & cost.** Pipeline-scale Kernel/Permutation SHAP works only when
`predict` is **deterministic, row-preserving, target-free, fixed-output,
side-effect-free**. For a buried Torch model: eval mode, disable dropout &
prediction-time augmentation, no-grad, fixed seeds / deterministic backend, and
explain **logits/probabilities, not hard labels**. For spectra, independent-wavelength
masking is off-manifold and brutally expensive — use **grouped adjacent bands**, a
proper background distribution, batched predictions, and a bounded budget. Multi-output
pipelines must explicitly select the target/class explained. **Perturbed samples need
fresh content-derived identities** so prediction caches can't return unperturbed rows.

**Reproducibility — three distinct claims.** (1) *optimizer trajectory* = same ordered
space + seed + ask/tell event tape; (2) *objective* = additionally identical folds,
metrics, data, pipeline behaviour; (3) *artifact/result* = additionally identical
portable operators + numeric env. Start with **sequential ask/tell, one coordinator
thread, no wall-clock timeout** — adaptive samplers diverge if tell-order or even one
score changes; do not claim parallel HPO reproducible yet. **Use an ordered vector
schema with explicit categorical order — not language map iteration** (today's Python
compiler follows insertion order, `n4m_engine.py:194`). **A facade does not make Python
artifacts portable:** a Torch/joblib pipeline is pipeline-SHAP-able in Python; it crosses
languages only when its operators use portable n4m artifacts (or another agreed portable
format) — otherwise the *contract* is cross-language while the fitted host artifact stays
language-owned, which is consistent with the existing responsibility split.

**The pivot (D1–D3):** the whole approach rests on dag-ml exposing reusable fitted
state + real replay predictions. Confirmed small (extraction/export), not a subsystem.
