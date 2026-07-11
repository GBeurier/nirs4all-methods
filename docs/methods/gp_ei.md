# `gp_ei` — Gaussian-process Bayesian optimization (Expected Improvement)

**Role:** `optimization` · **kind:** `n4m_sampler_kind_t = N4M_SAMPLER_GP_EI` · **since:** ABI 2.1 (F4)

Bayesian optimization with a Gaussian-process surrogate and the Expected
Improvement acquisition. After `n_startup_trials` random trials, every ask fits an
RBF GP on the completed, scored history over the **continuous axes** (int / float /
log-int / log-float, in unit space), then returns the candidate that maximises EI
over a random acquisition batch. This is the sample-efficient sampler for
smooth, low-dimensional, expensive objectives — the regime where a single PLS/DL
fit dominates the trial cost, so spending compute to pick the *next* point pays off.

The surrogate is deliberately simple and dependency-free:

- **Kernel:** squared-exponential (RBF), unit signal variance on standardised
  targets, lengthscale from the **median pairwise distance** heuristic (no
  marginal-likelihood inner loop — robust and cheap).
- **Fit:** `K + 1e-6·I`, dense Cholesky, `α = K⁻¹y` by forward/back substitution
  (from-scratch, in `gp.cpp`; fine for the trial counts NIRS finetuning uses).
- **Acquisition:** EI with a small exploration margin `ξ = 0.01`, maximised by
  random search over 64 candidates per ask (direction-symmetric — MAXIMIZE is
  handled by negating the posterior mean).

Non-continuous axes (categorical / ordinal / sorted-tuple / conditionally
inactive) are drawn independently by the shared decode — the GP models only the
continuous subspace (Optuna's independent-fallback convention). Warm-start
(`enqueue`) is unsupported (`N4M_ERR_UNSUPPORTED`): a forced candidate is not a
model proposal. Purely-categorical spaces degrade to random (use `tpe` there).

## Usage (C ABI)

```c
n4m_optimizer_options_t opts;
n4m_optimizer_options_init(&opts);
opts.sampler = N4M_SAMPLER_GP_EI;
opts.n_startup_trials = 8;   // random exploration before the GP takes over
```

## Parity

- **Tier C** (self-consistency + convergence): GP-fitting details (lengthscale
  policy, acquisition optimiser) differ across libraries, so there is no bit-exact
  external reference. The C++ test asserts convergence on a smooth 2-D objective
  in ~60 evaluations (`best < 0.5`; empirically `< 0.03` across ten seeds — far
  more sample-efficient than random/CMA-ES on the same objective). Cross-binding
  identical at a fixed seed via the shared `n4m_rng` and native kernel/Cholesky.

## References

- Jones, Schonlau & Welch, *Efficient Global Optimization of Expensive Black-Box
  Functions*, J. Global Optimization 13 (1998), 455–492.
  [`jones1998ego`](_finetuning_bibliography.bib)
- Rasmussen & Williams, *Gaussian Processes for Machine Learning*, MIT Press
  (2006). [`rasmussen2006gp`](_finetuning_bibliography.bib)
