# `gp_ei` — Gaussian-process Bayesian optimization (Expected Improvement)

**Role:** `optimization` · **kind:** `n4m_sampler_kind_t = N4M_SAMPLER_GP_EI` · **since:** ABI 2.1 (F4)

Bayesian optimization with a Gaussian-process surrogate and the Expected
Improvement acquisition. Once at least `max(n_startup_trials, 2)` completed,
scored trials are available, every ask fits an RBF GP over the numeric axes (int /
float / log-int / log-float, in unit space), then returns the candidate that
maximises EI over a random acquisition batch. It is intended for smooth,
low-dimensional, expensive objectives where spending compute to pick the next
point can be worthwhile.

The surrogate is deliberately simple and dependency-free:

- **Kernel:** squared-exponential (RBF), unit signal variance on standardised
  targets, lengthscale from the **median pairwise distance** heuristic (no
  marginal-likelihood inner loop — robust and cheap).
- **Fit:** `K + 1e-6·I`, dense Cholesky, `α = K⁻¹y` by forward/back substitution
  (from-scratch, in `gp.cpp`; fine for the trial counts NIRS finetuning uses).
- **Acquisition:** EI with a small exploration margin `ξ = 0.01`, maximised by
  random search over 64 candidates per ask (direction-symmetric — MAXIMIZE is
  handled by negating the posterior mean).

The GP models every numeric axis (`int`, `float`, `log_int`, `log_float`) in unit
space; stepped and integer observations are recorded after snapping. Categorical
and ordinal axes are drawn independently from the native RNG. A `sorted_tuple`
is also generated independently and is not represented in the surrogate.
Conditional activation is applied to the decoded trial. In a purely categorical
space there is no GP subspace, so all proposals are uniform native draws (use
`tpe` when categories should be learned).

Warm-start (`enqueue`) is unsupported (`N4M_ERR_UNSUPPORTED`): a forced candidate
is not a model proposal. Search spaces containing hard `mutex_group`, `requires`
or `exclude` constraints are rejected at `n4m_optimizer_create` with
`N4M_ERR_UNSUPPORTED`; there is no fitness fallback. Hard constraints that
reference a tuple root are rejected for every sampler. A non-`none` liar is also
rejected (`N4M_ERR_NOT_IMPLEMENTED`), so batch asks do not model pending trials.

## Usage (C ABI)

```c
n4m_optimizer_options_t opts;
n4m_optimizer_options_init(&opts);
opts.sampler = N4M_SAMPLER_GP_EI;
opts.n_startup_trials = 8;   // explicit override; options_init() defaults to 10
```

## Parity

- **Tier C** (self-consistency + convergence): GP-fitting details (lengthscale
  policy, acquisition optimizer) differ across libraries, so there is no
  bit-exact external reference. The C++ tests check minimization, maximization,
  categorical-only and degenerate-history behavior. Track-Q commits the native
  `gp_ei_sphere2` trace as an acceptance target for future bindings; it does not
  yet cover every pruner or mixed-space combination.

## References

- Jones, Schonlau & Welch, *Efficient Global Optimization of Expensive Black-Box
  Functions*, J. Global Optimization 13 (1998), 455–492.
  [`jones1998ego`](_finetuning_bibliography.bib)
- Rasmussen & Williams, *Gaussian Processes for Machine Learning*, MIT Press
  (2006). [`rasmussen2006gp`](_finetuning_bibliography.bib)
