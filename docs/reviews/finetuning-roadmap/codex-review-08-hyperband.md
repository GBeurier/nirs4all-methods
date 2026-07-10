**Findings**

- Major: [cpp/src/core/optimization/pruners.cpp:157](/home/delete/nirs4all/_worktrees/native-hpo-phase1/cpp/src/core/optimization/pruners.cpp:157) recomputes `R` from the current largest reported step when `max_resource == 0`, so `n_brackets` and `idx % n_brackets` are not stable. Trigger: Hyperband, `max_resource=0`, `eta=3`; ask trials 0,1,2; report step 0 scores `1,2,9`. At that moment `R=1`, `n_brackets=1`, so idx 2 can be pruned as bracket 0. If a later report reaches step 8, `R=9`, `n_brackets=3`, and idx 2 would be bracket 2 with grace at rung 0. Minimal fix: do not derive bracket count from a moving high-water mark; use a fixed/lazily latched `R` for the optimizer lifetime, or avoid Hyperband pruning until a stable `R` is known.

- Major: [cpp/src/core/optimization/pruners.cpp:153](/home/delete/nirs4all/_worktrees/native-hpo-phase1/cpp/src/core/optimization/pruners.cpp:153) and [cpp/src/core/optimization/pruners.cpp:163](/home/delete/nirs4all/_worktrees/native-hpo-phase1/cpp/src/core/optimization/pruners.cpp:163) can signed-overflow on `step + 1` / `maxstep + 1`. Trigger: public `tell_intermediate(..., step=INT32_MAX, ...)` with Hyperband. Minimal fix: compute resource/R with `int64_t` or reject `step < 0 || step == INT32_MAX` before calling the pruner.

- Major: [cpp/src/core/optimization/pruners.cpp:154](/home/delete/nirs4all/_worktrees/native-hpo-phase1/cpp/src/core/optimization/pruners.cpp:154) allows pruning at rungs above fixed `max_resource`. Trigger: `max_resource=9`, `eta=3`, report `step=26` (`resource=27`, `k=3`) for same-bracket trials; `k_max=2`, but the code still ranks/prunes at rung 3. Minimal fix: after computing `k_max`, return false or reject when `k > k_max`.

- Minor: [cpp/tests/test_optimization.cpp:587](/home/delete/nirs4all/_worktrees/native-hpo-phase1/cpp/tests/test_optimization.cpp:587) covers the explicit `max_resource=9` happy path only. It does distinguish Hyperband from ASHA via bracket-1 grace, but it would not catch the `max_resource=0` moving-bracket bug or pruning above `max_resource`. Minimal fix: add regression cases for default `max_resource=0` stability and `k > k_max`.

**Correct**

`rung_index()` handles exact powers as intended for normal-range inputs: step 0 maps to resource 1/rung 0, non-powers do not prune, and `eta` is clamped before use. `n_brackets` does not become zero with the current constructor clamp.

`index_of()` is deterministic under the base append-only ask/tell path; a live trial passed by `Optimizer::tell_intermediate()` should be found.

The same-bracket/same-step peer filter is correct for fixed `n_brackets`, and maximize/minimize strict-better comparisons are symmetric.

ASHA now receives `opts.reduction_factor`, with default `0 -> 3`, so the existing ASHA behavior is preserved.

The C/Python option layouts mirror correctly: Python size is 120, `max_resource` offset 56, `reduction_factor` offset 60, `reserved` offset 64 with length 56.
---

## Resolution (all findings applied)

- **Major — moving high-water-mark `R` (max_resource=0) → unstable brackets:** the
  derive-`R`-from-largest-reported-step path is removed. `max_resource` is now
  **required (> 0)**; `make_pruner` returns `N4M_ERR_INVALID_ARGUMENT` for a
  hyperband pruner with `max_resource == 0`. A fixed `R` makes the bracket count
  stable for the study's lifetime, so a trial's bracket cannot change under it.

- **Major — signed overflow on `step+1` / `maxstep+1`:** resource and `R` are now
  computed in `int64_t`; `rung_index` takes `int64_t`; `step < 0` returns false
  early. (The `maxstep+1` derivation is gone with the fix above.)

- **Major — pruning at rungs above `max_resource`:** added a `k > k_max` guard —
  a report at a rung above the top rung `R` never prunes.

- **Minor — test coverage:** added `test_hyperband_edges` — (a) `max_resource == 0`
  is rejected at create; (b) three same-bracket trials reporting at a rung above `R`
  all survive (proving the `k > k_max` guard; without it the worst would be halved).

The pure-Python reference (`parity/hpo/references.py`) mirrors the same rule
(`R` required, `k > k_max` guard), and the Track-Q gate confirms native hyperband
decisions == the reference on the `hyperband_prune` spec. 385 C++ tests pass; the
`hyperband_prune` golden trace is unchanged (its steps stay within `R`).
