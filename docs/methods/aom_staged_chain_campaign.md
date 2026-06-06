# `aom_staged_chain_campaign` - staged strict-chain cartesian screen/refit

_Group_: **AOM / moment campaign** · _Catalog_: `aom_pop.aom_staged_chain_campaign`
· _Backed by_: `aom_chain_screen_refit_campaign` (pure Python orchestration over
the single `libn4m` runtime)

## Description

`aom_staged_chain_campaign` is the first-class staged-cartesian workflow for AOM /
moment preprocessing selection. It runs several score-only strict-linear
preprocessing screens in sequence — the `compact`, `wide` and `lab` profiles,
focused family plans such as `savgol_focus` / `strict_family_focus`, or an
explicit `stages` list mixing profiles and Ridge / PLS / mixed head plans —
then:

1. merges the per-stage retained candidates (deduplicated by chain/head/param,
   keeping the best screen score and recording every stage a candidate came
   from);
2. keeps the **top global** and **top per-head** rows across all stages;
3. exact-CV **refits** that retained union exactly once; and
4. attaches **preprocessing-impact** and **screen-vs-refit rank** diagnostics, and
   an optional **offline holdout audit**.

It does not add any new numerical kernel: every fit flows through the existing
`aom_chain_score_campaign` (screen), `aom_refit_candidates` (exact-CV refit),
`aom_candidate_preprocessing_impact`, and `aom_candidate_rank_diagnostics`
helpers, which all call the single `libn4m` C/CUDA runtime. The orchestrator is
the staged equivalent of the lab proto `cartesian.py` / `impact.py`, expressed
over the catalogued n4m moment routes.

## Constraints (load-bearing)

* **Strict-linear only.** Stages screen the strict-linear AOM chain grids
  (`build_aom_strict_chain_grid`). No hors-moment nonlinear or supervised lifts
  are introduced.
* **No identity selection.** The campaign consumes only `X` / `y` arrays. Stage
  `name` values are cosmetic labels; no dataset, source, id or name is ever read
  or used to pick chains, heads or the winner. Renaming stages cannot change the
  selected model.
* **Train-only production selection.** The production winner is the minimum
  exact-CV refit row (`selection_metric="refit_cv_rmse"`,
  `selection_uses_test_set=False`). A held-out set, when supplied, is scored only
  for the offline audit and is never used for selection.
* **Single infrastructure.** This is Python orchestration; `libn4m` stays the one
  C/binding engine.

## Parameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `X`, `y` | arrays | — | Training spectra and target(s); `y` 1-D or `(n, 1)` |
| `stages` | list \| None | `None` | Explicit stage list (profile name or override dict). `None` → use `plan` |
| `plan` | str | `"compact_wide_lab"` | One of `compact`, `wide`, `lab`, `compact_wide`, `compact_lab`, `wide_lab`, `compact_wide_lab`, `savgol_focus`, `strict_family_focus` |
| `cv` | int | `5` | Exact CV folds (screen + refit) |
| `ridge_lambdas` / `pls_components` | seq | small grids | Default head grids (per-stage overridable) |
| `heads` | seq | `("ridge", "pls")` | Default heads (per-stage overridable) |
| `top_k` | int | `50` | Rows each stage keeps in its own screen |
| `refit_top_k` | int \| None | `None`→`top_k` | Global retained rows refit with exact CV |
| `refit_per_head_top_k` | int \| None | `10` | Extra per-head retained rows (`None` disables) |
| `checkpoint_dir` | path \| None | `None` | Directory for one resumable score checkpoint per stage |
| `resume` | bool | `True` | Resume matching stage checkpoints when present |
| `max_chunks_per_run` | int \| None | `None` | Limit new chunks processed per stage in this call |
| `scale_x_values` | seq \| None | `None` | Optional grid such as `[False, True]`; each value runs the same staged campaign and the model config is selected by train exact-CV refit |
| `pls_score_mode` | str | `"cv"` | `"gcv_proxy"` enables the fast PLS screen proxy |
| `moment_policy` / `refit_moment_policy` | str | `"auto"` | Screen / refit moment routing |
| `impact` / `rank_diagnostics` | bool | `True` | Toggle the post-hoc audit reports |
| `X_audit` / `y_audit` | arrays \| None | `None` | Optional **offline** held-out audit |
| `return_stage_screens` | bool | `False` | Include the raw per-stage screen reports |

Stage override dict keys: `name`, `profile`, `heads`, `ridge_lambdas`,
`pls_components`, `top_k`, `max_chains`, `families`, `templates`,
`pls_score_mode`, `moment_policy`, `chain_ordering`, `split_head_scoring`.
Missing keys fall back to the campaign defaults. Unknown keys raise.

## Returned report

A JSON-friendly dict keyed by `report_schema = "n4m.aom_staged_chain_campaign.v1"`:

| Key | Meaning |
|-----|---------|
| `stages` | Per-stage summaries (`name`, `profile`, `heads`, `n_chains`, `n_top_candidates`, `screen_best`, …) |
| `rows` | Exact-CV refit rows (the retained union), each with `chain`, `head`, `param`, `refit_cv_rmse`, `screen_cv_rmse`, and cross-stage `campaign_stage` / `campaign_stages` |
| `best` / `best_cv` / `best_refit` | Production winner = minimum `refit_cv_rmse` row |
| `best_by_head` | Per-head best refit row |
| `merged_top_candidates` | Cross-stage deduplicated global screen pool |
| `retention` | `refit_top_k`, `refit_per_head_top_k`, and global/per-head union counts |
| `checkpoint_dir` / `max_chunks_per_run` / `n_remaining_stage_chunks_total` | Staged resume state; partial screens remain exact-refit-able over currently retained rows |
| `impact` | `aom_candidate_preprocessing_impact` over the refit rows (`refit_cv_rmse`) |
| `rank_diagnostics` | `aom_candidate_rank_diagnostics` (screen `screen_cv_rmse` vs exact `refit_cv_rmse` rank drift / recall) |
| `audit` | Offline holdout report (`audit_only=True`) or `None` |
| `refit` | The full `aom_refit_candidates` report |
| `selection_metric` / `selection_policy` / `selection_uses_test_set` | Selection provenance (`refit_cv_rmse` / `exact_cv_refit_train_only` / `False`) |
| `model_config_grid` / `model_config_summaries` / `selected_model_config` | Present when `scale_x_values` is used; records the train-CV-selected model config and per-config best refit scores |

`rows` is directly consumable by `NativeAOMFixedCandidateRegressor.from_refit_report`.

## Python usage

```python
import numpy as np
import n4m

rng = np.random.default_rng(7)
X = rng.standard_normal((64, 256))
y = X[:, 8] - 0.4 * X[:, 19] + 0.05 * rng.standard_normal(64)

# Staged compact -> wide -> lab screen over mixed Ridge/PLS heads.
report = n4m.aom_staged_chain_campaign(
    X, y,
    plan="compact_wide_lab",
    cv=5,
    refit_top_k=20,           # global retained rows
    refit_per_head_top_k=5,   # extra per-head rows
    checkpoint_dir="artifacts/aom_staged_checkpoints",
)

best = report["best"]                       # production winner, exact-CV on train
print(best["head"], best["param"], best["refit_cv_rmse"])
print(report["retention"])                  # how many rows were exact-refit
print(report["impact"]["by_operator"][:3])  # which preprocessing families paid off
print(report["rank_diagnostics"]["spearman_rank_correlation"])
print(report["screen_complete"], report["n_remaining_stage_chunks_total"])

# Materialize the winner with the existing reusable estimator.
model = n4m.NativeAOMFixedCandidateRegressor.from_refit_report(report).fit(X, y)
y_hat = model.predict(X)
```

Model-configuration grid selection, still train-only:

```python
report = n4m.aom_staged_chain_campaign(
    X, y,
    plan="compact",
    scale_x_values=[False, True],
    refit_top_k=12,
    refit_per_head_top_k=2,
)

assert report["selection_uses_test_set"] is False
print(report["selected_model_config"])  # {"scale_x": True/False, ...}
print(report["model_config_summaries"])
```

When a model-config grid is used, `rows` and `best` come from the selected
config, while route/candidate counters such as `n_screen_pls_moment_cv_fits`
sum the work paid across every config.

Focused preprocessing-family plans, useful when `max_chains` is intentionally
small. Start with `savgol_focus` for the fast incremental campaign; use
`strict_family_focus` as a heavier family-audit profile because Gaussian/FCK /
Whittaker stages can dominate wall time on some datasets.

```python
# Prioritize SavGol diversity instead of waiting for late lab-profile entries.
report = n4m.aom_staged_chain_campaign(
    X, y,
    plan="savgol_focus",
    max_chains=8,       # applied per focused stage
    refit_top_k=12,
    scale_x_values=[False, True],
)

# Also force strict Gaussian/FCK/Whittaker stages to be screened early.
report = n4m.aom_staged_chain_campaign(
    X, y,
    plan="strict_family_focus",
    max_chains=8,
    refit_top_k=12,
)
```

These plans are fixed source-free stage recipes over the existing strict-linear
lab families. They do not read dataset/source/name/id metadata and they still
select the final row only by train exact-CV refit.

Sklearn estimator form (same train-CV selection, no held-out audit inputs):

```python
model = n4m.NativeAOMStagedChainCampaignRegressor(
    plan="compact_wide_lab",
    cv=5,
    refit_top_k=20,
    refit_per_head_top_k=5,
    checkpoint_dir="artifacts/aom_staged_checkpoints",
).fit(X, y)

y_hat = model.predict(X)
diag = model.get_diagnostics()
assert diag["selection_uses_test_set"] is False
print(model.selected_head_, model.selected_param_, model.selected_cv_rmse_)
```

Fast SavGol-focused reusable preset:

```python
model = n4m.NativeAOMSavgolFocusRegressor(
    cv=5,
    checkpoint_dir="artifacts/aom_savgol_focus_checkpoints",
).fit(X, y)

diag = model.get_diagnostics()
assert diag["plan"] == "savgol_focus"
assert diag["selection_uses_test_set"] is False
print(diag["selected_stage"], diag["selected_model_config"])
```

The preset delegates to the same staged campaign engine with
`plan="savgol_focus"`, `max_chains=6`, `top_k=10`, `refit_top_k=8`,
`refit_per_head_top_k=2`, `scale_x_values=[False, True]` and
`split_head_scoring="auto"` by default. On a
CUDA build it also defaults to the one-GPU PLS route knobs used in the local
benchmark (`cuda_pls_parallel_folds=True`, `cuda_pls_min_device_features=1`,
`backend_min_cuda_product=1`). Use the generic staged estimator when you need
custom plans or family templates.

Cost-safe strict-family audit preset:

```python
model = n4m.NativeAOMStrictFamilyLiteRegressor(
    cv=5,
    checkpoint_dir="artifacts/aom_strict_family_lite_checkpoints",
).fit(X, y)

diag = model.get_diagnostics()
assert diag["plan"] == "strict_family_focus"
assert diag["selection_uses_test_set"] is False
print(diag["selected_stage"], diag["n_refit_candidates"])
```

`NativeAOMStrictFamilyLiteRegressor` exercises the broader
`strict_family_focus` stage recipe, but defaults to a small audit budget:
`max_chains=2`, `top_k=6`, `refit_top_k=4`, `refit_per_head_top_k=1`,
`scale_x=False`, no `scale_x_values` grid and `split_head_scoring="auto"`.
It is meant to sample SavGol,
Norris-Williams, finite-difference, Gaussian, FCK and Whittaker behavior without
paying for the heavier strict-family benchmark profile.

Custom stages (e.g. a Ridge-only compact pass then a PLS-only wide pass):

```python
report = n4m.aom_staged_chain_campaign(
    X, y,
    stages=[
        {"name": "ridge_compact", "profile": "compact", "heads": ("ridge",)},
        {"name": "pls_wide", "profile": "wide", "heads": ("pls",),
         "pls_score_mode": "gcv_proxy"},
    ],
    refit_per_head_top_k=5,
)
```

Offline audit (test ranking only — never used for selection):

```python
report = n4m.aom_staged_chain_campaign(
    X_train, y_train,
    plan="compact_wide",
    X_audit=X_test, y_audit=y_test,   # offline audit only
)
assert report["audit"]["audit_only"] is True
assert report["selection_uses_test_set"] is False
# report["best"] is identical whether or not the audit set is supplied.
```

The same callable is exported from `n4m`, `n4m.aom` and `n4m.moment`.
The reusable sklearn estimator is exported as
`NativeAOMStagedChainCampaignRegressor` from `n4m`, `n4m.sklearn`, `n4m.aom`
and `n4m.moment`.

Resume is stage-local: each stage delegates to
`aom_chain_score_campaign(..., checkpoint_path=...)`. With
`max_chunks_per_run`, the report may have `screen_complete=False`; the retained
rows available so far are still exact-CV refit on train and can be reused as a
partial audit. A later call with the same data/configuration and
`checkpoint_dir` resumes the remaining chunks.

## Reused building blocks

| Step | Helper |
|------|--------|
| Strict-chain grids | `build_aom_strict_chain_grid` / `iter_aom_strict_chain_grid` |
| Per-stage screen | `aom_chain_score_campaign` (chunked, streaming) |
| Global + per-head retention | `aom_screen_refit_candidate_pool` semantics |
| Exact-CV refit | `aom_refit_candidates` |
| Preprocessing impact | `aom_candidate_preprocessing_impact` |
| Screen-vs-refit rank | `aom_candidate_rank_diagnostics` |
| Offline audit | `aom_evaluate_candidates` |
| Report save/load | `aom_save_candidate_report` / `aom_load_candidate_report` |

## Workflow / benchmark note

This is the staged runner the AOM benchmark campaign calls for: run
compact/wide/lab strict-chain screens with Ridge, PLS and mixed heads, retain the
top global and per-head candidates, exact-refit the retained rows, and read
`impact` / `rank_diagnostics` to decide whether a wider cartesian budget is
justified before comparing against the robust AOM / TabPFN baselines. Because
each stage screens in chunks of `chain_chunk_size` and only keeps its own
`top_k`, large `lab` cartesians stream rather than materialize every scored
candidate.

### Timing smoke

`benchmarks/cross_binding/bench_aom_staged_chain_campaign_timing.py` records the
end-to-end wall-clock cost of the staged campaign on synthetic data, one CSV row
per `--plans` entry, with the retention / selection / impact / rank-diagnostics
and refit route / CUDA counters pulled from the report. It accepts the small
screen controls (`--plan`s, `--max-chains`, `--top-k`, `--refit-top-k`,
`--refit-per-head-top-k`, `--chain-chunk-size`, `--heads`, `--components`,
`--ridge-lambdas`, `--repeats`) and the campaign's CPU/GPU knobs
(`--cuda-pls-parallel-folds`, `--cuda-pls-min-device-features`,
`--cuda-pls-many-batched`, `--backend-min-cuda-product`, `--moment-policy`).

This measures **orchestration and exact-refit timing only** — the staged screen,
cross-stage retention, the single exact-CV refit of the retained union, and the
post-hoc diagnostics. It is **not** a benchmark of the future fused IKPLS
grinder; the per-candidate exact CV it times is the target that grinder must
beat, not the grinder itself.

```bash
PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so \
  python benchmarks/cross_binding/bench_aom_staged_chain_campaign_timing.py \
  --plans compact,compact_wide --max-chains 16 --top-k 24 --refit-top-k 8 \
  --output /tmp/aom_staged_chain_campaign_timing.csv
```

Tiny one-GPU CUDA smoke used for release readiness:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/bench_aom_staged_chain_campaign_timing.py \
  --output benchmarks/cross_binding/aom_staged_chain_campaign_timing_cuda_smoke.csv \
  --repeats 1 --plans compact --n-samples 96 --n-features 128 --cv 3 \
  --heads pls --components 1 --ridge-lambdas 0.1 --max-chains 4 \
  --chain-chunk-size 2 --top-k 4 --refit-top-k 3 \
  --refit-per-head-top-k 1 --moment-policy auto \
  --cuda-pls-min-device-features 1 --cuda-pls-parallel-folds
```

The smoke is expected to keep `selection_uses_test_set=False` and show nonzero
`n_pls_moment_cuda_device_cv_fits` with zero host PLS CV fits.
