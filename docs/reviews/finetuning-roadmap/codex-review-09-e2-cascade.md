Found 1 Major and 2 Minor issues.

1. **Major — `#` cascade remains declaration-order dependent.** [optimizer.cpp:303](/home/delete/nirs4all/_worktrees/native-hpo-phase1/cpp/src/core/optimization/optimizer.cpp:303), [test_optimization.cpp:377](/home/delete/nirs4all/_worktrees/native-hpo-phase1/cpp/tests/test_optimization.cpp:377)

   Trigger: declare conditional `model#pls#n_components` before unconditional `model`; sample `model="ridge"`. `resolve(child)` returns false, but the later `model` iteration matches `model#*` and overwrites the child to active. Minimal fix: conjunct cascaded states:

   ```cpp
   kv.second.active = kv.second.active && active;
   ```

   Add a separate reversed-declaration-order `#` case; the current test uses unrelated names `a/b/c`, so it misses this.

2. **Minor — missing parents fail open for `NOT_IN`.** [optimizer.cpp:296](/home/delete/nirs4all/_worktrees/native-hpo-phase1/cpp/src/core/optimization/optimizer.cpp:296)

   Trigger: child has `CONDITION_NOT_IN(missing, "x")`. Both lookups miss, making `cond_ok=true` and `parent_active=true`, so the child activates although its parent cannot be active. Minimal scoped fix:

   ```cpp
   const bool parent_active = parent_spec != nullptr && resolve(*parent_spec);
   ```

3. **Minor — avoidable tight-loop allocation/work.** [optimizer.cpp:281](/home/delete/nirs4all/_worktrees/native-hpo-phase1/cpp/src/core/optimization/optimizer.cpp:281)

   Flat spaces now populate a node-allocating map and scan every trial parameter for every specification. With 500 parameters and 200 retries, that is roughly 50 million prefix checks. Minimal fix: skip unconditional specifications in the application loop; additionally reserve/replace the map and consider a generic recursive lambda. Normal objective-heavy HPO is unlikely to notice this.

Otherwise correct: the temporary memo `0` is observable only through a cycle, not legitimate shared-parent fan-out; inactive parents suppress children for both `IN` and `NOT_IN`; recursive resolution itself is declaration-order independent. The test’s `a="q", b="x"` path directly fails the old implementation, and `saw_q_with_b_x` is a sound coverage guard.
---

## Resolution (all findings applied)

- **Major — `#` cascade declaration-order dependent:** `apply_conditions` now only
  ever **deactivates** (never re-activates). The application loop skips
  unconditional params (resolve→active, a no-op) and skips active conditional
  params, applying `active = false` only for conditional params that resolve
  inactive. A structural ancestor's `#` cascade can therefore no longer overwrite a
  deactivated child back to active regardless of declaration order. New test
  `test_conditional_cascade_order` declares `model#pls#nc` *before* `model` and
  asserts it stays inactive when `model=="ridge"`.
- **Minor — missing parent fails open for NOT_IN:** `parent_active = parent_spec
  != nullptr && resolve(*parent_spec)` — a missing parent cannot be active, so its
  child is deactivated (was previously fail-open).
- **Minor — tight-loop allocation:** the deactivate-only restructure also removes
  the per-spec full-trial scan for flat/active spaces (the common case), so a
  no-condition space does no prefix scanning at all.

387 C++ tests pass; HPO parity gate + nirs4all engine tests green. No ABI change.
