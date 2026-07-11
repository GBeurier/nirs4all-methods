# `ternary` — unimodal integer search

**Role:** `optimization` · **kind:** `n4m_sampler_kind_t = N4M_SAMPLER_TERNARY` · **since:** ABI 2.1 (F1)

Ternary search over a single **unimodal integer** axis — a port of the nirs4all `BinarySearchSampler`. It anchors the search by probing the low bound, the high bound, and the midpoint, then repeatedly bisects the larger interval adjacent to the current best value, converging in O(log n) evaluations instead of the O(n) of a linear sweep. The canonical use is PLS `n_components`, whose CV-RMSE is unimodal in the component count.

The proposal is a **pure function of the completed-trial history** (it recomputes the search state each `ask`), so it is deterministic and idempotent within a single ask — safe under constraint retries and reproducible across bindings.

The sampler tunes the **first integer parameter** in the search space; any other parameters are drawn uniformly at random by the base sampler. If the space has no integer parameter, it degrades to pure random search.

## Usage (C ABI)

```c
n4m_search_space_t* space = NULL;
n4m_search_space_create(&space);
n4m_search_space_add_int(space, "n_components", 1, 30, 1, /*log=*/0);

n4m_optimizer_options_t opts;
n4m_optimizer_options_init(&opts);
opts.sampler = N4M_SAMPLER_TERNARY;
/* ... ask/tell loop as for `random` ... */
```

## Parity

- **Tier A** (deterministic): the proposal sequence is a pure function of the completed history and the fixed bounds; it replays exactly and is bit-identical across bindings.
- **Behavioural:** matches the nirs4all `BinarySearchSampler` intent (triplet anchor → bisect-toward-best). A tighter decision-level parity fixture against the Python sampler is scheduled with the Track-Q parity machinery.

## References

- Ports the nirs4all `BinarySearchSampler` (unimodal ternary search). See `docs/methods/_finetuning_bibliography.bib`.
