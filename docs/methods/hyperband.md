# `hyperband` — Hyperband bracketed successive halving (pruner)

**Role:** `optimization` · **kind:** `n4m_pruner_kind_t = N4M_PRUNER_HYPERBAND` · **since:** ABI 2.1 (F5)

Hyperband run as an early-stopping pruner: several **brackets** of successive
halving that hedge different early-stopping aggressiveness, so you do not have to
guess the right stopping rate up front (Hyperband's advantage over plain ASHA).

- **Rungs** sit at geometric resource levels `eta^k` (resource = `step + 1`); the
  pruner only decides on a rung boundary, never at an intermediate step.
- Each trial is assigned a **bracket** `s ∈ [0, s_max]` round-robin by its stable
  ask order, where `s_max = floor(log_eta(R))`. Bracket `s` has a **grace period**:
  it is exempt from pruning below rung `s`, so higher-`s` brackets run more configs
  to a smaller resource while `s = 0` behaves like near-pure random search.
- At each rung a trial survives only if it ranks in the **top 1/eta** of the
  **same-bracket** peers that reached that rung (ASHA-style asynchronous promotion;
  ties survive — only strictly-better peers count).

Configured entirely from the options struct (no new ABI symbols):

- `opts.reduction_factor` — `eta` (default 3 when left 0).
- `opts.max_resource` — the top rung `R`. When left 0 it is **derived** from the
  largest `step` reported so far, so callers that don't know the budget up front
  still get sensible brackets.

Like all pruners, `hyperband` is **orthogonal to the sampler** and consumes
whatever intermediate-score axis the caller supplies.

> **Fidelity-axis caveat (roadmap §2c):** Hyperband/ASHA assume rank-preservation
> across rungs. The intended native fidelity is the PLS `n_components` learning
> curve / subsample fraction / epochs — **not** a CV-fold fraction (folds are
> exchangeable, so use `racing` there). Hyperband only consumes the rung stream;
> the fidelity *engine* that produces those scores is a separate deliverable.

## Usage (C ABI)

```c
n4m_optimizer_options_t opts;
n4m_optimizer_options_init(&opts);
opts.pruner           = N4M_PRUNER_HYPERBAND;
opts.reduction_factor = 3;   /* eta */
opts.max_resource     = 27;  /* top rung; 0 = derive from reports */
/* per trial, per rung: */
int32_t prune = 0;
n4m_optimizer_tell_intermediate(opt, trial_id, rung, rung_score, &prune);
```

## Parity

- **Tier B (decision-level):** the promote/prune/grace verdict is an RNG-free
  function of the rung history, bracket assignment and `eta`; verified against a
  canned history in the C++ tests (bracket-0 successive halving prunes the worst
  of three; a bracket-1 trial with a far worse score survives rung 0 via its grace
  period; a non-rung step never prunes). A cross-reference fixture vs Optuna's
  `HyperbandPruner` lands with the Track-Q parity machinery.

## References

- Li, Jamieson, DeSalvo, Rostamizadeh & Talwalkar, *Hyperband: A Novel
  Bandit-Based Approach to Hyperparameter Optimization*, JMLR 18 (2018), 1–52.
  [`li2018hyperband`](_finetuning_bibliography.bib)
- Li et al., *A System for Massively Parallel Hyperparameter Tuning* (ASHA),
  MLSys (2020). See `_finetuning_bibliography.bib`.
