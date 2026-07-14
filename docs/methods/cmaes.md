# `cmaes` — separable CMA-ES (sampler)

**Role:** `optimization` · **kind:** `n4m_sampler_kind_t = N4M_SAMPLER_CMAES` · **since:** ABI 2.1 (F4)

Covariance Matrix Adaptation Evolution Strategy, **separable (diagonal) variant** (Ros & Hansen 2008), over the unit hypercube. A generation of `λ = 4 + ⌊3·ln P⌋` candidates is sampled from `N(m, σ²·diag(C))` and clamped to `[0,1)`; once resolved, the best completed and scored members (up to `μ = λ/2`) update the mean `m`, diagonal covariance `C`, global step-size `σ`, and two evolution paths. The diagonal covariance drops the eigendecomposition of full CMA-ES, so the sampler stays cheap for modest numeric dimensionality.

CMA-ES adapts every numeric axis (`int`, `float`, `log_int`, `log_float`) in unit space; stepped and integer proposals are snapped during decoding. Categorical and ordinal axes are drawn independently from the native RNG and do not update the CMA distribution. A `sorted_tuple` axis is likewise generated independently and is not modelled. This is a native mixed-space behavior, not an external-library compatibility guarantee; for heavily categorical spaces prefer `tpe` or `ga`.

**Synchronous update (F4):** the distribution advances only once its whole generation is terminal (`liar = none`), so `ask_batch` returns a partial batch at a generation boundary, and warm-start (`n4m_optimizer_enqueue`) is unsupported (`N4M_ERR_UNSUPPORTED`). The distribution updates from **completed, scored** members only — pruned and failed trials never enter the mean/covariance. Any non-`none` liar value is rejected at optimizer creation with `N4M_ERR_NOT_IMPLEMENTED`.

Conditional activation is honored by the decode. Search spaces containing hard `mutex_group`, `requires` or `exclude` constraints are rejected by `n4m_optimizer_create` with `N4M_ERR_UNSUPPORTED`; CMA-ES never treats an infeasible configuration as a host-supplied poor fitness. Hard constraints that reference a `sorted_tuple` root are rejected for every sampler.

## Usage (C ABI)

```c
n4m_optimizer_options_t opts;
n4m_optimizer_options_init(&opts);
opts.sampler = N4M_SAMPLER_CMAES;
opts.seed = 42;
```

## Parity

- **Tier B-state target:** the distribution state after ranked tells is deterministic for a fixed seed, ask/tell order and score tape. Track-Q currently commits the native `cmaes_sphere2` proposal trace; it has no state-level `pycma` fixture and does not cover every pruner or mixed-space combination. Convergence on a smooth objective is verified in the C++ tests.

## References

- Hansen & Ostermeier, *Completely Derandomized Self-Adaptation in Evolution Strategies*, Evol. Comput. 9 (2001), 159–195; Ros & Hansen, *A Simple Modification in CMA-ES Achieving Linear Time and Space Complexity*, PPSN (2008); Hansen, *The CMA Evolution Strategy: A Tutorial* (2016). See `_finetuning_bibliography.bib`.
