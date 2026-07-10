# `ga` — genetic algorithm (sampler)

**Role:** `optimization` · **kind:** `n4m_sampler_kind_t = N4M_SAMPLER_GA` · **since:** ABI 2.1 (F3)

Real-coded genetic algorithm over the unit hypercube. Every candidate is a unit vector `u ∈ [0,1)^P` decoded per parameter (`numeric_from_unit` for numeric axes; bucketed for categorical / ordinal), so mixed continuous / discrete / categorical spaces are handled uniformly. A **generation** of `pop_size` (= 16 in F3) candidates is handed out via `ask()`; once those trials report scores, the next generation is produced by **tournament selection + uniform crossover + Gaussian mutation, with elitism** (the best candidate is carried over unchanged, so the incumbent never regresses).

**Synchronous evolution (F3):** the population evolves only once its whole generation is scored (`liar = none`). `ask_batch` therefore returns a *partial* batch at a generation boundary — score the current generation before asking further. Warm-start (`n4m_optimizer_enqueue`) is **not supported** for population samplers (a forced candidate cannot be inverse-encoded into the genome) and returns `N4M_ERR_UNSUPPORTED`.

GA suits **combinatorial / rugged** search surfaces — preprocessing-chain choices, mixed categorical+numeric spaces — where gradient-free population search beats independent sampling. Sorted-tuple axes are sampled by the base sampler (not encoded in the genome). Hard constraints are handled through fitness (the host scores an infeasible candidate poorly), not by rejection.

> Note: this is the HPO-sampler GA over the typed search space — distinct from the feature-selection `n4m_feature_selection_ga_select` (a GA over feature masks). A later F3 refinement may share the RNG-consolidated population loops between them (see `FINETUNING_ROADMAP.md`); for now they are independent.

## Usage (C ABI)

```c
n4m_optimizer_options_t opts;
n4m_optimizer_options_init(&opts);
opts.sampler = N4M_SAMPLER_GA;
opts.seed = 42;
/* standard ask/tell loop; run enough trials for several generations */
```

## Parity

- **Tier B-state:** the population trajectory is a deterministic function of the seed + the reported fitnesses; identical across bindings at a fixed seed via the shared `n4m_rng`. Convergence on a continuous test objective is verified in the C++ tests.

## References

- Leardi & Lupiáñez González, *Genetic algorithms applied to feature selection in PLS regression*, Chemom. Intell. Lab. Syst. 41 (1998), 195–207. See `_finetuning_bibliography.bib`.
