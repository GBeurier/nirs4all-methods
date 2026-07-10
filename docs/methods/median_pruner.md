# `median` — median stopping rule (pruner)

**Role:** `optimization` · **kind:** `n4m_pruner_kind_t = N4M_PRUNER_MEDIAN` · **since:** ABI 2.1 (F2)

Median early-stopping rule (Google Vizier). A trial is pruned when its latest intermediate score — reported through `n4m_optimizer_tell_intermediate(opt, id, step, score, &prune)` — is worse than the **median** of the peer trials' scores at the **same step**. It never prunes before `min_peers` peers have reported at that step, nor during the warm-up steps.

The pruner is **orthogonal to the sampler**: `n4m_optimizer_options_t.pruner` is set independently of `.sampler`, so any sampler can be paired with `median` (`sampler=tpe` × `pruner=median`, etc.). The decision is a pure function of the recorded intermediate histories, which makes it **decision-level testable against a canned history** — the parity tier for pruners.

The intermediate-score axis (the meaning of `step`) is supplied by the caller: a fidelity rung (PLS `n_components` learning curve, a subsample fraction) or DL epochs. See `FINETUNING_ROADMAP.md` §2c for the fidelity-axis discussion.

Config: `min_peers` defaults to `n4m_optimizer_options_t.n_startup_trials` (≥1); warm-up steps default to 0.

## Usage (C ABI)

```c
n4m_optimizer_options_t opts;
n4m_optimizer_options_init(&opts);
opts.pruner = N4M_PRUNER_MEDIAN;
opts.n_startup_trials = 5;   // don't prune until 5 peers have reported at a step
/* per trial, per rung: */
int32_t prune = 0;
n4m_optimizer_tell_intermediate(opt, trial_id, step, rung_score, &prune);
if (prune) { /* stop early; report tell_result(..., N4M_TRIAL_PRUNED, ...) */ }
```

## Parity

- **Tier B (decision-level):** given a fixed `(step, score)` history the keep/prune verdict is RNG-free and matches the Optuna `MedianPruner` semantics; verified against canned histories in the C++ tests (a dedicated cross-reference fixture lands with the Track-Q parity machinery).

## References

- Golovin, Solnik, Moitra, Kochanski, Karro & Sculley, *Google Vizier: A Service for Black-Box Optimization*, KDD (2017). [`golovin2017vizier`](_finetuning_bibliography.bib)
