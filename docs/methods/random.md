# `random` — uniform random search

**Role:** `optimization` · **kind:** `n4m_sampler_kind_t = N4M_SAMPLER_RANDOM` · **since:** ABI 2.1 (F0)

Uniform random sampling over the typed search space. Each parameter is drawn independently from its declared distribution: uniform for `int`/`float`, log-uniform for `log_int`/`log_float`, uniform-over-choices for `categorical`/`ordinal`, and (for `sorted_tuple`) a sorted vector of uniform draws. Conditional parameters are activated per their `condition_in`/`_not_in` constraint; `mutex`/`requires`/`exclude` constraints are enforced by bounded rejection sampling.

Random search is the default non-adaptive baseline and the startup-seeding strategy for the adaptive samplers. It is competitive with grid search at a fraction of the cost when only a few hyperparameters matter, per Bergstra & Bengio (2012).

## Usage (C ABI)

```c
n4m_search_space_t* space = NULL;
n4m_search_space_create(&space);
n4m_search_space_add_int(space, "n_components", 1, 30, 1, /*log=*/0);
n4m_search_space_add_float(space, "alpha", 1e-4, 1e0, 0.0, /*log=*/1);

n4m_optimizer_options_t opts;
n4m_optimizer_options_init(&opts);          /* sampler=random, pruner=none */
opts.seed = 42;

n4m_optimizer_t* opt = NULL;
n4m_optimizer_create(ctx, space, &opts, &opt);
for (int i = 0; i < n_trials; ++i) {
    n4m_trial_t* t = NULL;
    n4m_optimizer_ask(opt, &t);
    int64_t nc; n4m_trial_get_int(t, "n_components", &nc);
    /* ... host evaluates, obtains `score` ... */
    int64_t id; n4m_trial_get_id(t, &id);
    n4m_optimizer_tell(opt, id, score);
}
n4m_trial_t* best; double best_score;
n4m_optimizer_best(opt, &best, &best_score);
```

## Parity

- **Tier C** (self-consistency + regret): a random study replays bit-identically for a fixed seed, and reaches a known optimum within the expected budget on benchmark spaces.
- **Cross-binding:** identical seed → identical trial sequence across all bindings (1e-12), via the shared `n4m_rng` (`splitmix64`).

## References

- Bergstra & Bengio, *Random Search for Hyper-Parameter Optimization*, JMLR 13 (2012), 281–305. [`bergstra2012random`](_finetuning_bibliography.bib)
