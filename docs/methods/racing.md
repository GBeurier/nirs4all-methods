# `racing` — Hoeffding racing (pruner)

**Role:** `optimization` · **kind:** `n4m_pruner_kind_t = N4M_PRUNER_RACING` · **since:** ABI 2.1 (F2)

Statistical racing. Each trial's intermediate scores (reported via `n4m_optimizer_tell_intermediate`) are treated as repeated **observations** of its performance. A trial is pruned when its confidence interval no longer overlaps the best trial's — i.e. we are statistically confident it is worse. The interval is a Hoeffding bound `ε = R·√(ln(2/δ) / (2n))`, where `n` is the number of observations, `R` the observed score range, and `δ` the confidence (fixed at 0.05 in F2). A trial needs at least 2 observations before it can be pruned.

**Why racing and not ASHA for CV folds (roadmap §2c):** successive-halving assumes rank-preservation across fidelity rungs. CV folds are *exchangeable* — a fold subset is a higher-variance estimate of the *same* target, not a low-fidelity proxy — so promoting by rung rank across folds is statistically unjustified. The correct fold-based early-stop is confidence-bound elimination (racing). Use `racing` when the intermediate-score axis is CV folds; use `asha`/`median` when it is a genuine fidelity ladder (learning curve, subsample, epochs).

Racing is conservative by design: it only prunes once the observations make it confident, so it never eliminates a trial that a bit more evidence would have vindicated.

## Usage (C ABI)

```c
n4m_optimizer_options_t opts;
n4m_optimizer_options_init(&opts);
opts.pruner = N4M_PRUNER_RACING;
/* report one fold score per step: */
int32_t prune = 0;
n4m_optimizer_tell_intermediate(opt, trial_id, fold_index, fold_score, &prune);
```

## Parity

- **Tier B (decision-level):** the keep/prune verdict is an RNG-free function of the observation history + `δ`; verified against a canned history in the C++ tests.

## References

- Maron & Moore, *Hoeffding Races*, NeurIPS (1993); Mnih, Szepesvári & Audibert, *Empirical Bernstein Stopping*, ICML (2008). See `_finetuning_bibliography.bib`.
