# `cmaes` — separable CMA-ES (sampler)

**Role:** `optimization` · **kind:** `n4m_sampler_kind_t = N4M_SAMPLER_CMAES` · **since:** ABI 2.1 (F4)

Covariance Matrix Adaptation Evolution Strategy, **separable (diagonal) variant** (Ros & Hansen 2008), over the unit hypercube. A generation of `λ = 4 + ⌊3·ln P⌋` candidates is sampled from `N(m, σ²·diag(C))` and clamped to `[0,1)`; once scored, the `μ = λ/2` best update the mean `m`, the diagonal covariance `C`, the global step-size `σ`, and the two evolution paths, following the canonical CMA-ES equations. The diagonal covariance drops the eigendecomposition of full CMA-ES, so the sampler stays cheap and robust for the modest continuous dimensionality typical of NIRS finetuning (Ridge `alpha`, learning rates, continuous preprocessing parameters).

CMA-ES is the most **sample-efficient** sampler here for smooth continuous objectives. Non-continuous axes (int / categorical / ordinal) are handled by the shared `decode_candidate` (bucketed) — the independent-fallback behaviour Optuna's `CmaEsSampler` also uses for mixed spaces; for heavily categorical spaces prefer `tpe` or `ga`.

**Synchronous update (F4):** the distribution advances only once its whole generation is scored (`liar = none`), so `ask_batch` returns a partial batch at a generation boundary, and warm-start (`n4m_optimizer_enqueue`) is unsupported (`N4M_ERR_UNSUPPORTED`).

## Usage (C ABI)

```c
n4m_optimizer_options_t opts;
n4m_optimizer_options_init(&opts);
opts.sampler = N4M_SAMPLER_CMAES;
opts.seed = 42;
```

## Parity

- **Tier B-state:** the distribution state (mean, diagonal covariance, step-size, evolution paths) after N ranked tells is a deterministic function of the seed + the tells — the RNG-free surface to compare against `pycma`'s diagonal mode (a state-level fixture lands with Track-Q). Convergence on a smooth objective is verified in the C++ tests.

## References

- Hansen & Ostermeier, *Completely Derandomized Self-Adaptation in Evolution Strategies*, Evol. Comput. 9 (2001), 159–195; Ros & Hansen, *A Simple Modification in CMA-ES Achieving Linear Time and Space Complexity*, PPSN (2008); Hansen, *The CMA Evolution Strategy: A Tutorial* (2016). See `_finetuning_bibliography.bib`.
