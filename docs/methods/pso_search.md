# `pso` — particle swarm optimization (sampler)

**Role:** `optimization` · **kind:** `n4m_sampler_kind_t = N4M_SAMPLER_PSO` · **since:** ABI 2.1 (F3)

Particle Swarm Optimization over the unit hypercube (Kennedy & Eberhart). A swarm of `swarm_size` (= 16 in F3) particles — each with a position `u ∈ [0,1)^P`, a velocity, and a remembered personal best — is asked out per iteration. Once the iteration's trials are scored, each particle's personal best and the swarm's global best update, velocities move toward them, and positions advance:

```
v ← w·v + c1·r1·(pbest − x) + c2·r2·(gbest − x)      x ← clamp01(x + v)
```

with the Clerc & Kennedy (2002) convergence constants `w = 0.729`, `c1 = c2 = 1.494`. Candidates are decoded with the shared `Optimizer::decode_candidate`, so mixed continuous / discrete / categorical spaces work (sorted-tuple axes fall back to the base sampler; hard constraints are handled via fitness).

PSO is a good default for smooth-ish continuous and mixed surfaces; it complements the more disruptive `ga`. It reuses the same async-population lifecycle as `ga` (keyed on trial-id ranges).

> HPO-sampler PSO over the typed space — distinct from the feature-selection `n4m_feature_selection_pso_select` (binary PSO over feature masks).

## Usage (C ABI)

```c
n4m_optimizer_options_t opts;
n4m_optimizer_options_init(&opts);
opts.sampler = N4M_SAMPLER_PSO;
opts.seed = 42;
```

## Parity

- **Tier B-state:** the swarm trajectory is a deterministic function of the seed + reported fitnesses; identical across bindings at a fixed seed via the shared `n4m_rng`. Convergence on a continuous test objective is verified in the C++ tests.

## References

- Kennedy & Eberhart, *A discrete binary version of the particle swarm algorithm*, IEEE SMC (1997); Clerc & Kennedy (2002) convergence constants. See `_finetuning_bibliography.bib`.
