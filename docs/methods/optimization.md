# Optimization role — native hyperparameter finetuning

The `optimization` role (C ABI header [`n4m/optimization.h`](../../cpp/include/n4m/optimization.h), ABI 2.1) is a **portable ask/tell hyperparameter optimizer**: the search algorithm lives once in `libn4m` and is reused by every binding, so finetuning is bit-reproducible across Python / R / MATLAB-Octave / WASM. Design rationale and the full plan are in [`FINETUNING_ROADMAP.md`](../FINETUNING_ROADMAP.md) and [`NATIVE_FINETUNING.md`](../NATIVE_FINETUNING.md); the ABI freeze is detailed in [`FINETUNING_F0_PR.md`](../FINETUNING_F0_PR.md).

## Model

Three objects, all opaque C-ABI handles:

- **`n4m_search_space_t`** — a typed set of parameters (`int`, `float`, `log_int`, `log_float`, `categorical` with typed values, `ordinal`, `sorted_tuple`) plus declarative constraints (`mutex_group`, `requires`, `exclude`, `condition_in`/`_not_in`). Built with `n4m_search_space_add_*`.
- **`n4m_optimizer_t`** — the stateful search. `ask()` proposes a trial; the host evaluates it however it likes; `tell()` / `tell_result()` reports the outcome (`completed` / `pruned` / `failed`, with a score). `tell_intermediate()` reports a fidelity-rung score for pruning. `best()` returns the incumbent; `get_trials()` streams the trace; `enqueue()` warm-starts a known configuration.
- **`n4m_trial_t`** — one proposed configuration (borrowed; owned by the optimizer). Read parameters with `n4m_trial_get_int/float/category`, activation with `n4m_trial_is_active`.

The **host drives the loop** (`ask → evaluate → tell`); there is no C→host objective callback, so the pattern is safe under R's non-reentrant evaluator, WASM's synchronous runtime, and Python's GIL. For a native model the loop is wrapped in one call: **`n4m_finetune_estimator`** runs the ask/tell loop with an internal cross-validation objective (F0: PLS `n_components`).

## Samplers and pruners

Selected via `n4m_optimizer_options_t.sampler` / `.pruner`. Algorithms sit behind reserved enum values; a value not yet implemented returns `N4M_ERR_NOT_IMPLEMENTED` at `n4m_optimizer_create`, so activating one in a later phase adds **no** new ABI symbol.

| Kind | Status | Phase | Page |
|---|---|---|---|
| sampler `random` | ✅ implemented | F0 | [random.md](random.md) |
| sampler `ternary` | ✅ implemented | F1 | [ternary.md](ternary.md) |
| sampler `lhs` | ✅ implemented | F1 | [lhs.md](lhs.md) |
| sampler `ga` | ✅ implemented | F3 | [ga_search.md](ga_search.md) |
| sampler `pso` | ✅ implemented | F3 | [pso_search.md](pso_search.md) |
| sampler `cmaes` | ✅ implemented | F4 | [cmaes.md](cmaes.md) |
| sampler `sobol` | reserved | F1 | — |
| sampler `tpe`, `gp_ei` | reserved | F4 | — |
| pruner `none` | ✅ implemented | F0 | — |
| pruner `median` | ✅ implemented | F2 | [median_pruner.md](median_pruner.md) |
| pruner `asha` | ✅ implemented | F2 | [asha.md](asha.md) |
| pruner `racing` | ✅ implemented | F2 | [racing.md](racing.md) |
| pruner `hyperband` | reserved | F2/F5 | — |

## Reproducibility

The optimizer owns a seeded `n4m_rng`; the same seed and the same sequential tell-order reproduce the exact ask sequence, identically in every binding (the cross-binding parity guarantee). Parity tiers per sampler are described in [`FINETUNING_ROADMAP.md` §3](../FINETUNING_ROADMAP.md).
