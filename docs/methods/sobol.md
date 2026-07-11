# `sobol` — Sobol low-discrepancy sequence

**Role:** `optimization` · **kind:** `n4m_sampler_kind_t = N4M_SAMPLER_SOBOL` · **since:** ABI 2.1 (F1)

Sobol quasi-random sampling over the search space. Each parameter is assigned one
Sobol dimension (in space order); the unscrambled Gray-code sequence uses the
Joe–Kuo `new-joe-kuo-6.21201` direction numbers (embedded in
`sobol_direction.hpp`, extracted from `scipy.stats.qmc.Sobol._sv`). Numeric axes
map the unit coordinate through `numeric_from_unit` (log / step / int aware);
categorical axes bucket the coordinate. Like all quasi-random sequences, Sobol
fills the unit cube far more evenly than i.i.d. random at small budgets — which is
where NIRS finetuning usually lives — so it is a strong space-filling startup
sampler and the default seed for the `auto` policy.

The direction table covers the first `kSobolMaxDim = 52` parameters; any parameter
beyond that (and every conditional / sorted-tuple axis, which the base sampler
draws) falls back to the base uniform sampler. Because the sequence is
deterministic per ask, Sobol is intended for unconstrained or lightly-constrained
numeric spaces; hard MUTEX/EXCLUDE constraints that reject the proposed point
cannot be escaped by resampling the same point — use `random`/`tpe` there.

This is the **unscrambled** variant. The scrambled (Owen / digital-shift) variant
is a Tier-B randomised sequence and a later addition.

## Usage (C ABI)

```c
n4m_optimizer_options_t opts;
n4m_optimizer_options_init(&opts);
opts.sampler = N4M_SAMPLER_SOBOL;
```

## Parity

- **Tier A** (bit-exact): the unscrambled sequence is bit-identical to
  `scipy.stats.qmc.Sobol(scramble=False)`. Verified in C++
  (`test_sobol_sequence_parity`, the first five points of a 3-D space against the
  known dyadic reference) and end-to-end through the Python binding
  (`test_sobol_parity.py`, `d ∈ {1,3,6,10}`, `N` up to 32, `np.array_equal`).
  Cross-binding identical by construction — the direction table and Gray-code
  recursion are shared native code.

## References

- Sobol, *On the distribution of points in a cube and the approximate evaluation
  of integrals*, USSR Comp. Math. and Math. Phys. 7 (1967), 86–112.
  [`sobol1967distribution`](_finetuning_bibliography.bib)
- Joe & Kuo, *Constructing Sobol sequences with better two-dimensional
  projections*, SIAM J. Sci. Comput. 30 (2008), 2635–2654.
  [`joe2008sobol`](_finetuning_bibliography.bib)
