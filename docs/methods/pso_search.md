# `pso` — particle swarm optimization (sampler)

**Role:** `optimization` · **kind:** `n4m_sampler_kind_t = N4M_SAMPLER_PSO` · **since:** ABI 2.1 (F3)

Particle Swarm Optimization over the unit hypercube (Kennedy & Eberhart). A swarm of `swarm_size` (= 16 in F3) particles — each with a position `u ∈ [0,1)^P`, a velocity, and a remembered personal best — is asked out per iteration. Once the iteration's trials are terminal, completed scores update personal and global bests, then velocities and positions advance:

```
v ← w·v + c1·r1·(pbest − x) + c2·r2·(gbest − x)      x ← clamp01(x + v)
```

with the Clerc & Kennedy (2002) convergence constants `w = 0.729`, `c1 = c2 = 1.494`. Candidates are decoded with the shared `Optimizer::decode_candidate`, so mixed continuous / discrete / categorical spaces work. Conditional activation is honored. Particle position and velocity contain one coordinate per declared axis, but the coordinate for a `sorted_tuple` is unused by decoding; tuple components are drawn independently by the base RNG.

**Synchronous update (F3):** the swarm advances only once its whole iteration is terminal (`liar = none`), so `ask_batch` returns a *partial* batch at an iteration boundary. Only completed scores can update personal/global bests; pruned and failed particles receive worst fitness for that iteration. Any non-`none` liar value is rejected at optimizer creation with `N4M_ERR_NOT_IMPLEMENTED`. Velocities are capped at `vmax = 0.5` of the unit range. Warm-start (`n4m_optimizer_enqueue`) is **not supported** for population samplers (returns `N4M_ERR_UNSUPPORTED`).

Search spaces containing hard `mutex_group`, `requires` or `exclude` constraints are rejected by `n4m_optimizer_create` with `N4M_ERR_UNSUPPORTED`; PSO does not turn infeasibility into a host fitness penalty. Hard constraints that reference a `sorted_tuple` root are rejected for every sampler.

PSO is intended for smooth-ish continuous and mixed surfaces; it complements the more disruptive `ga`. It uses the same terminal-state generation guard as GA, keyed on trial-id ranges.

> HPO-sampler PSO over the typed space — distinct from the feature-selection `n4m_feature_selection_pso_select` (binary PSO over feature masks).

## Usage (C ABI)

```c
n4m_optimizer_options_t opts;
n4m_optimizer_options_init(&opts);
opts.sampler = N4M_SAMPLER_PSO;
opts.seed = 42;
```

## Parity

- **Tier B-state target:** the swarm trajectory is deterministic for a fixed seed, ask/tell order and score tape. Track-Q commits the native `pso_sphere2` trace as a target for future bindings; it does not yet exercise PSO with every pruner or search-space feature. Convergence on a continuous objective is verified in the C++ tests.

## References

- Kennedy & Eberhart, *A discrete binary version of the particle swarm algorithm*, IEEE SMC (1997); Clerc & Kennedy (2002) convergence constants. See `_finetuning_bibliography.bib`.
