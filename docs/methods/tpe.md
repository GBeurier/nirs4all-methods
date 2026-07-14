# `tpe` — Tree-structured Parzen Estimator (sampler)

**Role:** `optimization` · **kind:** `n4m_sampler_kind_t = N4M_SAMPLER_TPE` · **since:** ABI 2.1 (F4)

Clean-room Tree-structured Parzen Estimator in **univariate** form. Once an axis has enough usable completed history, its observations are sorted by score and split into a **good** set (top `γ = 0.25`) and the rest. Two Parzen (kernel-density) models are built — `l(x)` over the good values and `g(x)` over the bad values — and the next value is chosen to maximise the density ratio `l(x)/g(x)`, concentrating sampling where good trials cluster while avoiding the bad region:

- **numeric axes:** history is mapped to unit space (`unit_from_numeric`, log-aware), KDE with a `1/√n`-scaled Gaussian bandwidth; `n_ei = 24` candidates are drawn from `l` and the best `l/g` is decoded back.
- **categorical / ordinal axes:** Laplace-smoothed category frequencies in the good vs bad sets; a category is sampled proportionally to its `l/g` ratio.

TPE handles **mixed and conditional** spaces per axis: only completed, scored trials in which an axis was active enter that axis's history. Conditional activation, scalar forced values, `ask_batch` and scalar warm-start use the base optimizer behavior. TPE activates only once at least `max(n_startup_trials, 2)` usable trials exist for an axis and uses a uniform draw before that.

Hard `mutex_group`, `requires` and `exclude` constraints are supported through the base retry loop (at most 200 complete candidate draws); failure to find a feasible candidate returns an error rather than a constraint-violating trial. A `sorted_tuple` is sampled independently by the base RNG and is **not** modelled by TPE or forceable through one scalar `enqueue()` value. Hard constraints that reference the tuple root are rejected for every sampler. Constant-liar batching is not implemented: any `liar` other than `none` returns `N4M_ERR_NOT_IMPLEMENTED` at creation, so a batch uses only the completed history available at each `ask()`.

> Stepped / integer axes are modelled in continuous unit space and snapped by `numeric_from_unit` at decode time (a minor `l/g` approximation on the grid); the decoded value is always on-grid.

## Usage (C ABI)

```c
n4m_optimizer_options_t opts;
n4m_optimizer_options_init(&opts);
opts.sampler = N4M_SAMPLER_TPE;
opts.n_startup_trials = 15;   // explicit override; options_init() defaults to 10
```

## Parity

- **Tier B-state target:** Track-Q commits the deterministic native `tpe_mixed` trace for a fixed seed and score tape. It does not currently compare internal decisions with Optuna or exercise TPE with every pruner/constraint combination. Convergence on a mixed continuous+categorical objective is verified in the C++ tests.
- **Not** a bit-for-bit clone of Optuna's TPE; this is a clean-room univariate TPE.

## References

- Bergstra, Bardenet, Bengio & Kégl, *Algorithms for Hyper-Parameter Optimization*, NeurIPS (2011); Bergstra, Yamins & Cox, *Making a Science of Model Search*, ICML (2013). See `_finetuning_bibliography.bib`.
