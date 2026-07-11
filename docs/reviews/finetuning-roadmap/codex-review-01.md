# Codex review — FINETUNING_ROADMAP.md (round 1)

**Date:** 2026-07-10 · **Reviewer:** `codex exec` (codex-cli 0.144.1, reasoning_effort=medium, sandbox=read-only) over the full repo (+ sibling `../nirs4all`).
**Target:** `docs/FINETUNING_ROADMAP.md` + `docs/NATIVE_FINETUNING.md`.
**Disposition:** all 14 findings + 5 mandatory items ACCEPTED and applied (per the repo's "Codex wins" convention). See the roadmap's `[codex]` markers.

---

**Blocker**

1. **F0 does not actually reserve the result/status surface the goal requires.**  
Claim: “complete symbol set” with `ask/tell/tell_intermediate/best/get_trials/enqueue` and `n4m_finetune_estimator(...)` in F0. Evidence: [roadmap:38](/home/delete/nirs4all/nirs4all-methods/docs/FINETUNING_ROADMAP.md:38), [roadmap:123](/home/delete/nirs4all/nirs4all-methods/docs/FINETUNING_ROADMAP.md:123), [roadmap:125](/home/delete/nirs4all/nirs4all-methods/docs/FINETUNING_ROADMAP.md:125). Current Optuna results carry `COMPLETE/PRUNED/FAIL`, failed/pruned counts, duration, metric, direction: [optuna.py:281](/home/delete/nirs4all/nirs4all/optimization/optuna.py:281), [optuna.py:445](/home/delete/nirs4all/nirs4all/optimization/optuna.py:445).  
Why risky: `tell(score)` cannot represent failed trials, pruned terminal state, cancellation/timeout, error text, or wall-clock progress without later ABI additions.  
Fix: F0 needs `n4m_trial_status_t`, terminal `tell_result(...)` or `tell_completed/tell_pruned/tell_failed`, timeout/progress fields, explicit error propagation, trial duration, and an options struct with `size`/reserved fields.

2. **The PLS fidelity claim is overbroad and partly false.**  
Claim: “CV-RMSE at 1..K components is a free by-product of a single fit” for NIPALS/SIMPLS. Evidence: [roadmap:79](/home/delete/nirs4all/nirs4all-methods/docs/FINETUNING_ROADMAP.md:79), [roadmap:143](/home/delete/nirs4all/nirs4all-methods/docs/FINETUNING_ROADMAP.md:143). Generic component CV currently loops `k=1..K` and calls CV each time: [model_selection.cpp:69](/home/delete/nirs4all/nirs4all-methods/cpp/src/core/model_selection.cpp:69). The optimized sweep path is only exact PLS1 with `q == 1`, NIPALS, regression deflation: [sweep.cpp:2052](/home/delete/nirs4all/nirs4all-methods/cpp/src/core/sweep.cpp:2052), [sweep.hpp:3](/home/delete/nirs4all/nirs4all-methods/cpp/src/core/sweep.hpp:3). Even the optimized test shows one max-prefix fit per fold, not one global fit: [test_sweep.cpp:525](/home/delete/nirs4all/nirs4all-methods/cpp/tests/test_sweep.cpp:525).  
Why risky: F2’s pruning performance/correctness depends on a capability the repo only has for a narrow PLS1 route.  
Fix: rewrite as “one max-K fit per CV fold for eligible PLS1/NIPALS paths”; either scope F2 to that or add a real prefix-CV engine for SIMPLS/multi-target.

3. **The companion doc remains inconsistent outside the stated superseded section.**  
Claim: only `NATIVE_FINETUNING.md §4.2` is superseded. Evidence: [roadmap:273](/home/delete/nirs4all/nirs4all-methods/docs/FINETUNING_ROADMAP.md:273). But native doc still says SH/Hyperband fidelity is CV-fold fraction: [NATIVE_FINETUNING.md:192](/home/delete/nirs4all/nirs4all-methods/docs/NATIVE_FINETUNING.md:192), says “whenever intermediate scores exist → Hyperband”: [NATIVE_FINETUNING.md:199](/home/delete/nirs4all/nirs4all-methods/docs/NATIVE_FINETUNING.md:199), omits `pruner`/`eval_mode` from the dag-ml controls: [NATIVE_FINETUNING.md:209](/home/delete/nirs4all/nirs4all-methods/docs/NATIVE_FINETUNING.md:209), and still advertises `n4m.sklearn.N4MSearchCV`: [NATIVE_FINETUNING.md:225](/home/delete/nirs4all/nirs4all-methods/docs/NATIVE_FINETUNING.md:225).  
Why risky: implementers can follow the wrong fidelity axis/API from the “why + architecture” doc.  
Fix: update or mark stale all affected sections, not just §4.2.

**Major**

4. **“Later enum values are not ABI surface” is too loose.**  
Claim: after F0, samplers/pruners are just enum values and “export no symbol.” Evidence: [roadmap:15](/home/delete/nirs4all/nirs4all-methods/docs/FINETUNING_ROADMAP.md:15), [roadmap:119](/home/delete/nirs4all/nirs4all-methods/docs/FINETUNING_ROADMAP.md:119). But repo ABI rules classify a new enum value as ABI MINOR: [n4m_version.h:11](/home/delete/nirs4all/nirs4all-methods/cpp/include/n4m/n4m_version.h:11).  
Why risky: adding enum values later still requires headers, bindings, docs, versioning, and language enum updates.  
Fix: reserve all numeric enum values in F0 and mark unimplemented values as `N4M_ERR_UNSUPPORTED`; otherwise plan ABI-minor follow-ups.

5. **SearchSpace constraints are not rich enough for dag-ml parity.**  
Claim: `set_condition(child,parent,parent_equals)` is 1:1 with the DSL and must co-freeze `mutex/requires/exclude`. Evidence: [roadmap:122](/home/delete/nirs4all/nirs4all-methods/docs/FINETUNING_ROADMAP.md:122), [NATIVE_FINETUNING.md:137](/home/delete/nirs4all/nirs4all-methods/docs/NATIVE_FINETUNING.md:137), dag-ml constraints include `pick/arrange/count/_mutex_/_requires_/_exclude_`: [detect.py:263](/home/delete/nirs4all/nirs4all/nirs4all/pipeline/dagml/detect.py:263).  
Why risky: Flavor A grid and Flavor B native HPO will silently evaluate different spaces.  
Fix: define a generic constraint ABI now: requires/excludes/mutex groups, multi-parent conditions, `in/not-in`, and explicit unsupported-case errors.

6. **The current parity/registry machinery cannot support the proposed HPO parity tiers.**  
Claim: existing `MethodSpec` fields “fit a sampler directly.” Evidence: [roadmap:17](/home/delete/nirs4all/nirs4all-methods/docs/FINETUNING_ROADMAP.md:17). Actual `MethodSpec` is prediction/reference oriented: [registry.py:8668](/home/delete/nirs4all/nirs4all-methods/benchmarks/parity_timing/registry.py:8668). The per-method parity script compares prediction arrays only: [per_method_parity.py:179](/home/delete/nirs4all/nirs4all-methods/parity/scripts/per_method_parity.py:179), [per_method_parity.py:265](/home/delete/nirs4all/nirs4all-methods/parity/scripts/per_method_parity.py:265). Current parity CI says cross-binding runner is manual, not a gate: [parity-gate.yml:157](/home/delete/nirs4all/nirs4all-methods/.github/workflows/parity-gate.yml:157).  
Why risky: pruner decision fixtures, TPE/CMA state fixtures, and ask/tell traces have nowhere precise to live.  
Fix: add an `HpoSpec` or equivalent registry schema, dedicated comparators, fixtures, and an explicit CI job.

7. **F5/F6 sequencing understates the sibling-repo work.**  
Claim: F5 can start after F1; F6 then adds controller dispatch. Evidence: [roadmap:235](/home/delete/nirs4all/nirs4all-methods/docs/FINETUNING_ROADMAP.md:235), [roadmap:176](/home/delete/nirs4all/nirs4all-methods/docs/FINETUNING_ROADMAP.md:176). Actual nirs4all dispatch calls `_execute_finetune` for `finetune_params`: [base_model.py:718](/home/delete/nirs4all/nirs4all/nirs4all/controllers/models/base_model.py:718), which delegates to `OptunaManager`: [base_model.py:851](/home/delete/nirs4all/nirs4all/nirs4all/controllers/models/base_model.py:851). dag-ml currently rejects `finetune_params`: [run_backend.py:624](/home/delete/nirs4all/nirs4all/nirs4all/pipeline/dagml/run_backend.py:624), and forces them to Python: [detect.py:170](/home/delete/nirs4all/nirs4all/nirs4all/pipeline/dagml/detect.py:170).  
Why risky: pipeline acceptance cannot be proven in this repo alone, and F5 depends on F0 constraint semantics plus controller/refit integration.  
Fix: split F5 into contract + dag-ml execution + nirs4all dispatch checkpoints, each with cross-repo gates.

8. **`n4m_finetune_estimator` risks creating a second control loop.**  
Claim: pure-native one-call entry is F0, while dag-ml owns Tuner execution/refit later. Evidence: [roadmap:125](/home/delete/nirs4all/nirs4all-methods/docs/FINETUNING_ROADMAP.md:125), [roadmap:168](/home/delete/nirs4all/nirs4all-methods/docs/FINETUNING_ROADMAP.md:168), architecture says dag-ml owns search invariants: [NATIVE_FINETUNING.md:71](/home/delete/nirs4all/nirs4all-methods/docs/NATIVE_FINETUNING.md:71).  
Why risky: two loops can diverge on nested CV, selection, leakage rules, best-param shape, and refit behavior.  
Fix: define `n4m_finetune_estimator` as a thin evaluator/driver with identical `TrialResult` and best-param schema, or defer it until the dag-ml contract is frozen.

9. **Categorical values and string lifetime are underspecified.**  
Claim: categorical add/get APIs return labels. Evidence: [NATIVE_FINETUNING.md:133](/home/delete/nirs4all/nirs4all-methods/docs/NATIVE_FINETUNING.md:133), [NATIVE_FINETUNING.md:161](/home/delete/nirs4all/nirs4all-methods/docs/NATIVE_FINETUNING.md:161). C ABI rules say returned strings are library-owned and not freed by callers: [n4m.h:20](/home/delete/nirs4all/nirs4all-methods/cpp/include/n4m/n4m.h:20), with symmetric ownership: [n4m.h:25](/home/delete/nirs4all/nirs4all-methods/cpp/include/n4m/n4m.h:25).  
Why risky: borrowed host choice strings can dangle; string-only categories may not round-trip Python bool/int categories.  
Fix: require core to copy UTF-8 labels, define pointer lifetime, expose index as canonical, and either restrict categories to strings or add typed categorical payloads.

10. **The “same DSL” promise is broader than the native space.**  
Claim: all routes accept the same DSL. Evidence: [roadmap:38](/home/delete/nirs4all/nirs4all-methods/docs/FINETUNING_ROADMAP.md:38). Current Optuna path supports nested params, train params, scalar pass-through, and `sorted_tuple`: [optuna.py:1146](/home/delete/nirs4all/nirs4all/optimization/optuna.py:1146), [optuna.py:1299](/home/delete/nirs4all/nirs4all/optimization/optuna.py:1299). F0 only names int/float/log/categorical/ordinal: [roadmap:122](/home/delete/nirs4all/nirs4all-methods/docs/FINETUNING_ROADMAP.md:122).  
Why risky: `sampler:"tpe"` native and `sampler:"optuna"` can optimize different spaces under identical `finetune_params`.  
Fix: either add missing param kinds/static pass-through explicitly, or define the native subset and reject all unsupported keys before optimization.

11. **Metric/task coverage is regression-heavy despite pipeline-wide goal.**  
Claim: native finetuning should be reachable from pipelines in every language. Evidence: [roadmap:4](/home/delete/nirs4all/nirs4all-methods/docs/FINETUNING_ROADMAP.md:4), F0 metric is just an `int32_t`: [NATIVE_FINETUNING.md:180](/home/delete/nirs4all/nirs4all-methods/docs/NATIVE_FINETUNING.md:180). Current nirs4all metric logic handles regression and classification and resolves direction from metric/task type: [metrics.py:8](/home/delete/nirs4all/nirs4all/nirs4all/core/metrics.py:8), [metrics.py:119](/home/delete/nirs4all/nirs4all/nirs4all/core/metrics.py:119), [optuna.py:578](/home/delete/nirs4all/nirs4all/nirs4all/optimization/optuna.py:578).  
Why risky: classification, categorical targets, multi-target aggregation, and multi-objective behavior are undefined.  
Fix: add an objective/metric ABI with direction defaults, target-type rules, aggregation, and explicit “multi-objective unsupported” or reserved vector-objective shape.

12. **Parallel ask / constant-liar is only a risk note, not an ABI design.**  
Claim: parallel/batch ask is handled by documenting constant-liar/sequential guarantees. Evidence: [roadmap:106](/home/delete/nirs4all/nirs4all-methods/docs/FINETUNING_ROADMAP.md:106), [roadmap:257](/home/delete/nirs4all/nirs4all-methods/docs/FINETUNING_ROADMAP.md:257). F0 only lists scalar `ask`: [roadmap:123](/home/delete/nirs4all/nirs4all-methods/docs/FINETUNING_ROADMAP.md:123).  
Why risky: Studio requires async/streamable behavior: [NATIVE_FINETUNING.md:53](/home/delete/nirs4all/nirs4all-methods/docs/NATIVE_FINETUNING.md:53); scalar ask without a liar policy can duplicate active trials.  
Fix: add `ask_batch` or define scalar `ask` as active-trial aware with explicit liar strategy in F0 options.

**Minor**

13. **Catalog wording is too narrow.**  
Claim: `optimization.<name>` is a “dotted two-segment id” matching `aom_pop.aom_pls`. Evidence: [roadmap:18](/home/delete/nirs4all/nirs4all-methods/docs/FINETUNING_ROADMAP.md:18). The catalog has many three-segment IDs, e.g. [methods.yaml:675](/home/delete/nirs4all/nirs4all-methods/catalog/methods.yaml:675), and the splitter simply maps arbitrary `method_id` to files: [split_legacy_methods.py:19](/home/delete/nirs4all/nirs4all-methods/catalog/scripts/split_legacy_methods.py:19).  
Fix: say “dotted method id” and specify `namespace`, `leaf`, `c_surface`, and reference coverage for the new `optimization` category.

14. **Reference coverage for `optimization` is not in the gate.**  
Claim: `validate.py --strict-abi` green is enough for catalog. Evidence: [roadmap:18](/home/delete/nirs4all/nirs4all-methods/docs/FINETUNING_ROADMAP.md:18). Reference coverage is a separate `--check-references` option: [validate.py:438](/home/delete/nirs4all/nirs4all-methods/catalog/scripts/validate.py:438), and current donor categories do not include `optimization`: [validate.py:43](/home/delete/nirs4all/nirs4all-methods/catalog/scripts/validate.py:43).  
Fix: add `optimization` reference policy and include `validate.py --check-references` in sampler/pruner gates.

**Mandatory Before Implementation**

1. Freeze a complete F0 ABI options/result/status model, including failure, pruning, timeout, streaming, categorical ownership, and parallel ask semantics.  
2. Rewrite the PLS learning-curve fidelity scope to match the actual PLS1/NIPALS implementation, or add the missing prefix-CV implementation first.  
3. Reconcile `NATIVE_FINETUNING.md` with the roadmap everywhere, not just §4.2.  
4. Add a concrete HPO parity/registry/CI design instead of reusing prediction-only `MethodSpec`.  
5. Split the cross-repo sequencing so dag-ml, nirs4all controller dispatch, and best-param/refit normalization each have explicit gates.