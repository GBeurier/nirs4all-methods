# `lhs` — Latin Hypercube sampling

**Role:** `optimization` · **kind:** `n4m_sampler_kind_t = N4M_SAMPLER_LHS` · **since:** ABI 2.1 (F1)

Latin Hypercube sampling over the numeric axes for the first `n_startup_trials` asks. Each numeric dimension is stratified into `N = n_startup_trials` equal bins and visited exactly once (an independent permutation per dimension, plus per-cell jitter), so every 1-D projection of the startup batch is evenly covered — better space-filling than i.i.d. random at small budgets, which is where NIRS finetuning usually lives. Beyond the startup batch, and for categorical / ordinal / sorted-tuple axes, `lhs` falls back to the base uniform sampler.

The design is precomputed from the seed, so it is deterministic and identical across bindings. As a startup strategy it pairs naturally with an adaptive sampler for the tail (the `auto` policy uses `sobol`/`random`/`lhs` seeding before the adaptive phase).

## Usage (C ABI)

```c
n4m_optimizer_options_t opts;
n4m_optimizer_options_init(&opts);
opts.sampler = N4M_SAMPLER_LHS;
opts.n_startup_trials = 20;   // the stratified batch size
```

## Parity

- **Tier C** (self-consistency + coverage): the startup batch replays bit-identically for a fixed seed, and each numeric axis's `n_startup` values fall one-per-stratum (verified in the C++ tests). Cross-binding identical at a fixed seed via the shared `n4m_rng`.

## References

- McKay, Beckman & Conover, *A Comparison of Three Methods for Selecting Values of Input Variables…*, Technometrics 21 (1979), 239–245. [`mckay1979lhs`](_finetuning_bibliography.bib)
