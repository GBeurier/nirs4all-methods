# `sobol` — Sobol low-discrepancy sequence

**Role:** `optimization` · **kind:** `n4m_sampler_kind_t = N4M_SAMPLER_SOBOL` · **since:** ABI 2.1 (F1)

Sobol quasi-random sampling over the search space. Each parameter is assigned one
Sobol dimension (in space order); the unscrambled Gray-code sequence uses the
Joe–Kuo `new-joe-kuo-6.21201` direction numbers (embedded in
`sobol_direction.hpp`, extracted from `scipy.stats.qmc.Sobol._sv`). Numeric axes
map the unit coordinate through `numeric_from_unit` (log / step / int aware);
categorical axes bucket the coordinate. Like all quasi-random sequences, Sobol
fills the unit cube more evenly than i.i.d. random at many small budgets, making
it useful for explicit space-filling studies. It is selected only with
`opts.sampler = N4M_SAMPLER_SOBOL`: the C ABI has no implicit `auto` sampler
policy, and `n4m_optimizer_options_init()` defaults to `random`.

The direction table covers the first `kSobolMaxDim = 52` parameters. Numeric,
categorical and ordinal axes in that prefix consume their ordered Sobol
coordinate, including axes that are later deactivated by a condition. Axes after
the 52nd use the base uniform RNG. A `sorted_tuple` occupies its ordered axis
position but its components are generated independently by the base RNG; they
are not Sobol coordinates. After `2^30` Sobol points, all axes use the base RNG.

Conditional activation is supported. Search spaces containing a hard
`mutex_group`, `requires` or `exclude` constraint are rejected at
`n4m_optimizer_create` with `N4M_ERR_UNSUPPORTED`; there is no deterministic
retry or fitness fallback. A hard constraint that references a `sorted_tuple`
root is rejected for every sampler.

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
  Track-Q also compares its 3-D/32-point native cell with SciPy and commits the
  resulting trace. That fixture is the acceptance target for future bindings;
  the R, MATLAB-Octave and WASM optimizer wrappers are not yet covered.

## References

- Sobol, *On the distribution of points in a cube and the approximate evaluation
  of integrals*, USSR Comp. Math. and Math. Phys. 7 (1967), 86–112.
  [`sobol1967distribution`](_finetuning_bibliography.bib)
- Joe & Kuo, *Constructing Sobol sequences with better two-dimensional
  projections*, SIAM J. Sci. Comput. 30 (2008), 2635–2654.
  [`joe2008sobol`](_finetuning_bibliography.bib)
