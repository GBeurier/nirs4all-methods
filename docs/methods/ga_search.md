# `ga` — genetic algorithm (sampler)

**Role:** `optimization` · **kind:** `n4m_sampler_kind_t = N4M_SAMPLER_GA` · **since:** ABI 2.1 (F3)

Real-coded genetic algorithm over the unit hypercube. Every candidate is a unit vector `u ∈ [0,1)^P` decoded per parameter (`numeric_from_unit` for numeric axes; bucketed for categorical / ordinal), so mixed continuous / discrete / categorical spaces are handled uniformly. A **generation** of `pop_size` (= 16 in F3) candidates is handed out via `ask()`; once every trial is terminal, the next generation is produced by **tournament selection + uniform crossover + Gaussian mutation, with elitism**. Completed trials contribute their scores; pruned or failed members are represented by worst sentinel fitness.

**Synchronous evolution (F3):** the population evolves only once its whole generation is resolved (`liar = none`). `ask_batch` therefore returns a *partial* batch at a generation boundary — finish the current generation before asking further. Any non-`none` liar value is rejected at optimizer creation with `N4M_ERR_NOT_IMPLEMENTED`. Warm-start (`n4m_optimizer_enqueue`) is **not supported** for population samplers (a forced candidate cannot be inverse-encoded into the genome) and returns `N4M_ERR_UNSUPPORTED`.

GA is intended for **combinatorial / rugged** search surfaces such as mixed categorical+numeric spaces. Conditional activation is applied after candidate decoding. The genome contains one unit coordinate per declared axis, but the coordinate for a `sorted_tuple` is unused; tuple components are sampled independently by the base RNG. Search spaces containing hard `mutex_group`, `requires` or `exclude` constraints are rejected by `n4m_optimizer_create` with `N4M_ERR_UNSUPPORTED`; GA never delegates infeasibility to a host fitness penalty. Hard constraints that reference a tuple root are rejected for every sampler.

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

- **Tier B-state target:** the population trajectory is deterministic for a fixed seed, ask/tell order and score tape. Track-Q commits the native `ga_sphere2` trace as a target for future bindings; it does not yet exercise GA with every pruner or search-space feature. Convergence on a continuous objective is verified in the C++ tests.

## References

- Leardi & Lupiáñez González, *Genetic algorithms applied to feature selection in PLS regression*, Chemom. Intell. Lab. Syst. 41 (1998), 195–207. See `_finetuning_bibliography.bib`.
