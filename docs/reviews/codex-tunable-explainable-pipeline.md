## Verdict

Adopt the estimator-facade approach.

Do **not** build a tuner or SHAP engine inside dag-ml, and do **not** link dag-ml directly to libn4m. Compose them in the bindings.

But the “no dag-ml change at all” part is too optimistic. A small cross-language execution-contract addition is unavoidable: training must return reusable fitted pipeline state, and replay must return actual predictions. That is a modest extraction/export job, not a new subsystem.

| Goal | Verdict |
|---|---|
| dag-ml exposes finetuning | Yes—as a parameterizable, CV-evaluable estimator interface, not as an optimizer |
| n4m provides reproducible methods | Yes—the ask/tell ABI is the right shared state machine |
| Optuna or n4m tunes a pipeline | Yes—both drive the same estimator objective |
| Pipeline-scale SHAP | Yes—for model-agnostic SHAP over deterministic, row-preserving `predict` |
| Arbitrary embedded sub-DAG estimator | Not yet; requires explicit graph slicing/boundaries |
| Structural pipeline search | Not solved by flat overrides; materialization/recompile is required |

## A. Is the approach sound?

Yes. It is the smallest clean architecture.

The language-neutral conceptual interface should be:

```text
clone_unfitted()
get_params()
with_params()/set_params()
fit(training_data, fit_context)
predict(new_data, predict_context)
score(held_out_data)
save/load fitted state
```

Python makes that a sklearn estimator. R, MATLAB and JS expose their idiomatic equivalents. The actual fitted state should be:

```text
template graph/DSL
+ effective ExecutionPlan
+ ExecutionBundle
+ explicit output binding
+ host-owned artifact provider/store
```

The optimizer is completely outside this object:

```text
Optuna or libn4m
    ask/propose
    → clone estimator
    → apply params
    → leakage-safe CV evaluation
    → tell
    → final fit of winner
```

Likewise, pipeline-scale SHAP becomes:

```python
shap.KernelExplainer(fitted_pipeline.predict, background)
```

Perturbations enter at raw `X`, traverse preprocessing, branching and the buried Torch model, and produce terminal predictions. That fixes the current Python path, which captures and explains one inner model rather than the complete pipeline: [explainer.py](/home/delete/nirs4all/nirs4all/nirs4all/pipeline/explainer.py:113) and [explainer.py](/home/delete/nirs4all/nirs4all/nirs4all/pipeline/explainer.py:199).

Where it breaks:

1. **Current common dag-ml ABI cannot create reusable fitted pipeline state.**
2. **Current replay discards predictions from its return value.**
3. **Flat overrides do not support arbitrary nested paths or topology changes.**
4. **An arbitrary embedded sub-DAG has no defined input/output boundary.**
5. **A Python-trained Torch/joblib artifact is not automatically loadable in R or JS.**
6. **Not every graph is a valid SHAP function**—SHAP needs a stable, deterministic, row-preserving mapping from inputs to a fixed output.

So: sound and sufficient for whole fixed-topology pipelines, declared standalone fragments, HPO and model-agnostic SHAP. Not sufficient by itself for arbitrary graph extraction, topology search, or cross-language portability of host-native model binaries.

## B/C. Minimal build, by repository

### dag-ml: small but unavoidable execution slice

The underlying Rust core already has the required machinery:

- Host invocation is correctly represented by `RuntimeController::invoke(NodeTask) -> NodeResult`: [dataview.rs](/home/delete/nirs4all/dag-ml/crates/dag-ml-core/src/runtime/dataview.rs:309).
- Tasks carry phase, fold, effective params, inputs and seed: [task.rs](/home/delete/nirs4all/dag-ml/crates/dag-ml-core/src/runtime/task.rs:60).
- Results already carry predictions, explanations, artifacts and lineage: [task.rs](/home/delete/nirs4all/dag-ml/crates/dag-ml-core/src/runtime/task.rs:378).
- Scheduler methods already execute campaign phases and capture refit artifacts: [scheduler.rs](/home/delete/nirs4all/dag-ml/crates/dag-ml-core/src/runtime/scheduler.rs:167).
- `ExecutionBundle` is already the correct persistence abstraction: [bundle.rs](/home/delete/nirs4all/dag-ml/crates/dag-ml-core/src/bundle.rs:915).

What is missing is a common public transaction around those pieces.

Build these:

1. **Promote one shared training operation into core.**

   Move the duplicated CLI/Python sequence into a core function:

   ```text
   compile/plan
   → FIT_CV
   → score/select
   → optional REFIT
   → capture artifacts/caches
   → build bundle
   → return TrainingOutcome
   ```

   The private CLI version already exists around [main.rs](/home/delete/nirs4all/dag-ml/crates/dag-ml-cli/src/main.rs:2951), while Python independently reproduces it in [in_process.rs](/home/delete/nirs4all/dag-ml/crates/dag-ml-py/src/in_process.rs:520).

   `TrainingOutcome` should contain at least:

   ```text
   effective ExecutionPlan
   ExecutionBundle
   scores and terminal prediction blocks
   lineage
   prediction-cache records/payloads required by replay
   artifact records
   ```

   Include a `refit=false` evaluation mode so HPO trials do not needlessly refit; refit only the winner.

2. **Expose training through the C ABI using the existing vtables.**

   It can remain a stateless JSON-in/JSON-out call. No long-lived Rust estimator handle is necessary.

3. **Return useful replay output.**

   `dagml_replay_execute_json` exists at [dag_ml.h](/home/delete/nirs4all/dag-ml/crates/dag-ml-capi/include/dag_ml.h:380), but currently returns only counts through `ReplayExecutionSummary`: [lib.rs](/home/delete/nirs4all/dag-ml/crates/dag-ml-capi/src/lib.rs:4559).

   Add a `ReplayOutcome` containing terminal predictions, aggregated predictions, explanations and lineage. Do not require bindings to capture these through a side channel.

4. **Define the estimator output.**

   A graph may have multiple prediction-producing nodes. Require either:

   ```text
   output_binding = {node_id, port_name, target/class index, aggregation}
   ```

   or, in minimal v1, exactly one unambiguous terminal prediction sink.

5. **Define canonical parameter addressing.**

   Do not make `node__subparam` the wire format. Use something structured:

   ```json
   {
     "node_id": "model:pls",
     "namespace": "operator",
     "path": ["optimizer", "learning_rate"]
   }
   ```

   Suggested namespaces are `operator`, `fit`, and `structural`. Each binding can render sklearn-style names from a registered reversible alias map.

   Current overrides are only a top-level map merge: [generation.rs](/home/delete/nirs4all/dag-ml/crates/dag-ml-core/src/generation.rs:84) and [generation.rs](/home/delete/nirs4all/dag-ml/crates/dag-ml-core/src/generation.rs:471).

   For the first version, the simplest correct implementation is: clone template, apply patches, recompile, invalidate fitted state. Compilation cost is negligible beside model fitting.

Do **not** build:

- `dagml_tune`
- a SHAP algorithm
- a direct dag-ml → libn4m dependency
- a callback-owning, long-lived Rust estimator object

A serializable `(plan, bundle, output binding)` package plus host-owned artifacts matches dag-ml’s existing ownership boundary much better.

### Subpipeline scope

There are three different meanings:

1. **Tune only parameters belonging to a subgraph, while evaluating the whole graph.**

   Already compatible with the facade. Restrict the search-space addresses; no sub-DAG execution needed.

2. **Compile a declared fragment as a standalone estimator.**

   Also compatible. Represent the fragment as its own valid `GraphSpec`.

3. **Extract and fit/predict an arbitrary embedded region of an existing graph.**

   Not currently supported. `NodeKind::Subgraph` is only an enum member: [graph.rs](/home/delete/nirs4all/dag-ml/crates/dag-ml-core/src/graph.rs:15).

   If this exact capability is mandatory, add a small `GraphSliceSpec`:

   ```text
   selected node IDs
   boundary input bindings
   boundary output bindings
   artifact namespace
   ```

   It should compile into an ordinary standalone `GraphSpec`. A recursive `Subgraph` runtime is unnecessary.

### nirs4all-methods: retain ask/tell; finish delivery

The native optimizer ABI already has the right surface:

- typed spaces and constraints;
- all sampler/pruner options;
- `ask`, `ask_batch`, `tell`, `tell_result`, `tell_intermediate`, `best`.

See [optimization.h](/home/delete/nirs4all/nirs4all-methods/cpp/include/n4m/optimization.h:107) and [optimization.h](/home/delete/nirs4all/nirs4all-methods/cpp/include/n4m/optimization.h:155).

Do not generalize `n4m_finetune_estimator` into a host-callback pipeline runner. That function is appropriately limited to fully native n4m estimators and currently only handles the internal `n_components` regression path: [c_api_optimization.cpp](/home/delete/nirs4all/nirs4all-methods/cpp/src/c_api/c_api_optimization.cpp:573).

The genuine minimal work here is:

- expose the optimizer in the actual shipping Python package;
- add at least one non-Python wrapper;
- canonicalize ordered axes, categorical values and conditions;
- add cross-binding golden proposal traces driven by the same score tape;
- validate duplicate names, unknown condition parents and cycles;
- either implement or reject currently inert non-default options rather than advertising them.

Also, `save/load` is declared but returns `NOT_IMPLEMENTED`: [c_api_optimization.cpp](/home/delete/nirs4all/nirs4all-methods/cpp/src/c_api/c_api_optimization.cpp:480). Defer it for a local first release, but it becomes necessary for resumable Studio jobs.

### nirs4all: new estimator, common objective adapter

Do not retrofit the existing `NIRSPipeline`. It is deliberately prediction-only—`fit()` raises—and its parameter surface only exposes `fold`: [pipeline.py](/home/delete/nirs4all/nirs4all/nirs4all/sklearn/pipeline.py:226) and [pipeline.py](/home/delete/nirs4all/nirs4all/nirs4all/sklearn/pipeline.py:495).

Add a separate class, for example:

```text
nirs4all/sklearn/dagml_pipeline.py
    DagMLPipelineEstimator
```

It should:

- be sklearn-cloneable;
- store only constructor/template configuration before fitting;
- expose explicit graph parameters through `get_params(deep=True)`;
- invalidate `plan_`, `bundle_` and artifacts on `set_params`;
- call the new dag-ml training outcome API from `fit`;
- call replay from `predict`;
- support an explicit scorer and `predict_proba` where appropriate;
- accept identities, groups, metadata and multi-source envelopes—not merely ndarray `X`.

Then add a generic `PipelineObjective` used by both Optuna and n4m. The current `N4MFinetuneManager` is model-controller-specific and calls private controller build/train/evaluate methods: [n4m_engine.py](/home/delete/nirs4all/nirs4all/nirs4all/optimization/n4m_engine.py:460). Reuse its space compiler, but separate optimizer driving from terminal-model training.

External SHAP can use the fitted facade directly. Extending `nirs4all.explain()` to accept an estimator is useful UX, but not architecturally required.

## D. Where should the finetune loop live?

Host-side, in each binding.

That is the clean answer.

```text
study = create_optimizer(canonical_space)

for each trial, sequentially:
    params = ask()
    candidate = template.clone_unfitted().with_params(params)
    score = shared_dagml_evaluator.cross_validate(candidate, split_plan)
    tell(score or FAILED/PRUNED)

winner = template.with_params(best_params).fit(all_designated_training_data)
```

The loop itself is tiny. The difficult semantics—fold isolation, OOF handling, scoring, final refit and artifact lifecycles—must live behind the shared dag-ml evaluator contract.

A native callback runner would create Python GIL, R unwinding, MATLAB lifetime and WASM async/reentrancy problems while moving host orchestration into the numerical-methods repository. Ask/tell is already the shared native helper.

One critical correction to your shorthand: never do this:

```text
fit(X, y) → score(X, y) → tell
```

Use either:

```text
fit(outer_train) → score(outer_validation)
```

or a dag-ml CV evaluation returning leakage-safe OOF score. Final refit happens only after selection.

Per-language glue is acceptable if all bindings consume the same:

- canonical ordered `SearchSpaceSpec`;
- dag-ml evaluation operation;
- score direction and aggregation rules;
- trial failure/pruning status rules;
- conformance score tape.

## E. Sharp risks

### Parameter addressing

- Separate graph/control parameters, operator constructor parameters and fit-only kwargs.
- Treat topology/operator replacement as structural materialization followed by recompile.
- Never mutate an already-fitted plan or bundle.
- Do not parse arbitrary raw node IDs using `__`; keep a reversible alias table.
- Expose only explicitly declared graph/controller parameters. dag-ml cannot introspect hidden defaults inside opaque host objects.

A fixed graph with nested value parameters is cheap. Searching over operator replacement, passthrough, branch activation or variable-length pipelines is a separate template-materialization feature.

### Leakage and OOF

- The outer optimizer owns outer validation.
- Any dag-ml internal CV, stacking or early stopping must operate strictly inside outer-train.
- Preserve groups, repetitions, sample identities, source layout and metadata when subsetting.
- Trial caches and artifact namespaces must include parameter fingerprint, data fingerprint, fold identity, trial ID and seed.
- Never reuse fitted handles between candidates.
- A failure must become `tell_result(FAILED)`, not an artificial `inf` score unless that policy is explicitly standardized.

### SHAP correctness and cost

Pipeline-scale Kernel/Permutation SHAP works only when `predict` is:

- deterministic;
- row-preserving;
- free of target access;
- fixed-output;
- side-effect-free.

For buried Torch models:

- set evaluation mode;
- disable dropout and prediction-time augmentation;
- use no-grad;
- fix seeds/deterministic backend settings;
- explain logits or probabilities, not hard labels.

For spectra, independent wavelength masking is often off-manifold and computationally brutal. Use grouped adjacent bands, an appropriate background distribution, batched predictions and bounded evaluation budgets. A multi-output pipeline must explicitly select the target/class being explained.

Perturbed samples also need fresh content-derived identities so prediction caches cannot return results for unperturbed rows.

### Reproducibility

Distinguish three claims:

1. **Optimizer trajectory reproducibility:** same ordered space, seed and ask/tell event tape.
2. **Objective reproducibility:** additionally requires identical folds, metrics, data and pipeline behavior.
3. **Artifact/result reproducibility:** additionally requires identical portable operators and numeric environments.

Start with sequential ask/tell, one coordinator thread and no wall-clock timeout. Adaptive samplers diverge if tell order or even one score changes. Parallel scheduling should not be claimed reproducible yet.

The current Python n4m compiler follows mapping insertion order: [n4m_engine.py](/home/delete/nirs4all/nirs4all/nirs4all/optimization/n4m_engine.py:194). Use an ordered vector schema with explicit categorical order, not language map iteration.

Finally, a facade does not make Python artifacts portable. A Python Torch/joblib pipeline can be pipeline-SHAPed in Python. The same fitted pipeline can cross languages only when its operators use portable n4m artifacts or another agreed portable format. Otherwise the contract is cross-language while the fitted host artifact remains language-owned—which is consistent with the existing repository responsibilities.

## Smallest credible milestone

1. In dag-ml, promote shared CV/refit/bundle execution, add C ABI training, and return real replay outputs.
2. In nirs4all, add a separate trainable dag-ml estimator plus one common CV objective adapter for Optuna and n4m.
3. In nirs4all-methods, ship the optimizer wrapper in Python and one second language; add shared score-tape conformance.
4. Demonstrate:
   - complex branch/stacking `fit → predict`;
   - sklearn clone/CV;
   - sequential n4m HPO;
   - Optuna HPO through the same objective;
   - Kernel SHAP on raw features with a buried Torch model.
5. Defer arbitrary embedded graph slicing, structural variable-length search, parallel adaptive HPO and native study persistence.

That meets the four goals without changing responsibilities and without a large dag-ml build.