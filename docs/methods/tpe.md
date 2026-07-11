# `tpe` — Tree-structured Parzen Estimator (sampler)

**Role:** `optimization` · **kind:** `n4m_sampler_kind_t = N4M_SAMPLER_TPE` · **since:** ABI 2.1 (F4)

Tree-structured Parzen Estimator — the Optuna-default sampler — in its **univariate** form. After `n_startup_trials` random trials, each parameter is modelled independently: the completed history is sorted by score and split into a **good** set (top `γ = 0.25`) and the rest. Two Parzen (kernel-density) models are built — `l(x)` over the good values and `g(x)` over the bad values — and the next value is chosen to maximise the density ratio `l(x)/g(x)`, concentrating sampling where good trials cluster while avoiding the bad region:

- **numeric axes:** history is mapped to unit space (`unit_from_numeric`, log-aware), KDE with a `1/√n`-scaled Gaussian bandwidth; `n_ei = 24` candidates are drawn from `l` and the best `l/g` is decoded back.
- **categorical / ordinal axes:** Laplace-smoothed category frequencies in the good vs bad sets; the category maximising `l/g` is chosen.

TPE handles **mixed and conditional** spaces naturally (each active parameter is modelled on the trials in which it was active), which is where it beats CMA-ES (continuous-only) and the population samplers. It plugs into the base sampler's per-parameter hooks, so constraints, conditions, forced values, sorted tuples, `ask_batch`, and warm-start all work as for the base sampler. The categorical proposal is *sampled* proportional to `l/g` (not argmax), so the base constraint-retry loop escapes an infeasible categorical combination. TPE activates only once ≥ `max(n_startup_trials, 2)` trials exist for an axis, and falls back to a uniform draw before that.

> Stepped / integer axes are modelled in continuous unit space and snapped by `numeric_from_unit` at decode time (a minor `l/g` approximation on the grid); the decoded value is always on-grid.

## Usage (C ABI)

```c
n4m_optimizer_options_t opts;
n4m_optimizer_options_init(&opts);
opts.sampler = N4M_SAMPLER_TPE;
opts.n_startup_trials = 15;   // random exploration before TPE kicks in
```

## Parity

- **Tier B-state:** the RNG-free sub-decisions — the γ good/bad split and the `l/g` ranking over an injected candidate set — are compared against Optuna's `TPESampler`; the public `ask` point is RNG-entangled (it draws `n_ei` candidates) and is *not* bit-matched. An `optuna-compat` flag + a decision-level fixture land with Track-Q. Convergence on a mixed continuous+categorical objective is verified in the C++ tests.
- **Not** a bit-for-bit clone of Optuna's TPE (its behaviour depends on undocumented, version-drifting constants); this is a clean-room univariate TPE.

## References

- Bergstra, Bardenet, Bengio & Kégl, *Algorithms for Hyper-Parameter Optimization*, NeurIPS (2011); Bergstra, Yamins & Cox, *Making a Science of Model Search*, ICML (2013). See `_finetuning_bibliography.bib`.
