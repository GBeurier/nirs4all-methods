# Native HPO — Phase 1 implementation progress log

**Branch:** `feat/native-hpo-phase1` · **Worktree:** `_worktrees/native-hpo-phase1` · **Scope:** `nirs4all-methods` only (stop before dag-ml / nirs4all-core / any other repo).
**Plan:** [`FINETUNING_ROADMAP.md`](FINETUNING_ROADMAP.md) §6 (two-phase + parallel tracks) · [`FINETUNING_F0_PR.md`](FINETUNING_F0_PR.md) (F0 detail).
**Rule:** each big block is Codex-reviewed before moving on. Everything stays green (`ctest dev-release`) + ABI-gated.

## Status board

| Block | State | Codex review | Notes |
|---|---|---|---|
| Setup (worktree, baseline build) | ✅ done | — | isolated worktree; docs committed; toolchain OK |
| **F0** — ABI surface + `random`/`none` slice + scaffolding | 🟢 done+reviewed | ✅ Codex | 365/365 tests; ABI 2.1; **Codex-reviewed (14 findings all applied)**. Catalog `--strict-abi` reconcile + HPO parity CI (Track Q) deferred. |
| **F1** — samplers: sobol, lhs, ternary | 🟡 in progress | 🟢 ternary+lhs | `ternary` + `lhs` ✅ **Codex-reviewed (7 findings applied)**, 369 tests; `sobol` needs the Joe–Kuo direction-number table (deliberate follow-up) |
| **F2** — pruners: fidelity engine → median, asha, hyperband, racing | 🟡 in progress | 🟢 median+asha | `median`+`asha`+`racing` ✅ (374 tests; median/asha Codex-reviewed); `hyperband` (needs bracket scheduler → F5) + `n_components` fidelity engine remain |
| **F3** — RNG consolidation → ga, pso | ✅ done (samplers) | 🟢 Codex | `ga` + `pso` samplers ✅ (376 tests, shared `decode_candidate`); RNG-consolidation of the *feature-selection* loops deferred (HPO samplers are clean fresh impls) |
| **F4** — cmaes, tpe, (gp_ei?) | ✅ done (cmaes+tpe) | ⬜ | `cmaes` + `tpe` ✅ (379 tests); `gp_ei` optional/cuttable (roadmap) |
| **Q** — HpoSpec + comparators + parity CI | ⬜ todo | ⬜ | Track Q (start early) |
| **B** — bindings python→R/MATLAB/WASM + cross-binding gate | ⬜ todo | ⬜ | Track B |
| **Phase 1 done → WAIT for green light** | ⬜ | — | do NOT touch dag-ml/core |

Legend: ⬜ todo · 🟡 in progress · ✅ done · 🟢 done+reviewed

## Log

### 2026-07-10 — F4: TPE sampler
- Added `N4M_SAMPLER_TPE` (`cpp/src/core/optimization/tpe.cpp`): univariate Tree-structured Parzen Estimator (Optuna default). Per param: split the completed history into good (top γ=0.25) / bad, build Parzen `l(x)`/`g(x)` (KDE in unit space for numeric via a new `unit_from_numeric` inverse; Laplace-smoothed category frequencies for categorical), draw n_ei=24 candidates from `l` and keep argmax `l/g`. Added a base `override_categorical()` hook (parallel to `override_numeric`) so TPE plugs into the base sampler's constraint/condition/forced machinery. Handles **mixed/conditional** spaces. Convergence test on a continuous+categorical objective (finds x≈3, category 'a'). Also fixed a stale test (`reserved sampler` now uses `sobol`, since TPE is implemented). **379 passed, 0 failed.** ABI unchanged. Doc `docs/methods/tpe.md`.

### 2026-07-10 — F4: CMA-ES sampler
- Added `N4M_SAMPLER_CMAES` (`cpp/src/core/optimization/cma.cpp`): **separable (diagonal) CMA-ES** (Ros & Hansen 2008) over the unit hypercube — the canonical mean/covariance/step-size/evolution-path update with a diagonal covariance (no eigendecomposition). Reuses the async-population lifecycle + boundary guard + shared decode (non-continuous axes bucketed = Optuna's independent fallback). Convergence test asserts `best < 0.1` on a smooth 2D objective (a broken CMA-ES would not converge tightly). **378 passed, 0 failed.** ABI unchanged. Doc `docs/methods/cmaes.md`.

### 2026-07-10 — F3 samplers Codex review applied
- Codex review of ga+pso: **1 Blocker, 2 Major, 2 Minor — all applied** (`docs/reviews/finetuning-roadmap/codex-review-05-F3-samplers.md`). **Blocker:** population samplers refuse to cross a generation/iteration boundary until it is fully resolved (synchronous LIAR_NONE evolution) — `ask_batch` returns a *partial* batch at the boundary instead of evolving on unscored members (added `resolved_in_range`). **Major:** `enqueue`/warm-start rejected for population samplers (`N4M_ERR_UNSUPPORTED` via an `allow_enqueue()` hook) — a forced candidate can't be inverse-encoded into the population; constraint handling documented as fitness-only. **Minor:** PSO velocity clamp (vmax=0.5). +1 regression test (batch boundary + enqueue reject). **377 passed, 0 failed.**

### 2026-07-10 — F3: PSO sampler + shared decode
- Added `N4M_SAMPLER_PSO` (`cpp/src/core/optimization/pso.cpp`): particle-swarm optimization over the unit hypercube (Clerc–Kennedy w=0.729, c1=c2=1.494) — swarm of 16 particles with position/velocity/personal-best, global best = best personal best, positions clamped to [0,1). Factored the unit-vector→trial decode into `Optimizer::decode_candidate()` (shared by `ga` + `pso`; GA simplified to use it, still green). Convergence test on a 2D continuous quadratic. **376 passed, 0 failed.** ABI unchanged. Doc `docs/methods/pso_search.md`.

### 2026-07-10 — F3: GA sampler
- Added `N4M_SAMPLER_GA` (`cpp/src/core/optimization/ga.cpp`): real-coded genetic algorithm over the unit hypercube `[0,1)^P`, decoded per parameter (numeric via `numeric_from_unit`, categorical/ordinal bucketed) — handles mixed spaces uniformly. Generational: a population of 16 is asked out, then tournament selection + uniform crossover + Gaussian mutation + elitism produce the next generation once scores arrive (keyed on trial-id ranges; the async-population lifecycle will be reused by `pso`/`cmaes`). Convergence test on a 2D continuous quadratic. **375 passed, 0 failed.** ABI unchanged. Doc `docs/methods/ga_search.md`. (This is the HPO-sampler GA over the typed space — distinct from the feature-selection `ga_select`; sharing the RNG-consolidated loops is a later refinement.)

### 2026-07-10 — F2: racing pruner
- Added `N4M_PRUNER_RACING` (Hoeffding racing) into the reviewed pruner architecture: each trial's intermediate scores are repeated observations; a trial is pruned once its Hoeffding confidence interval (δ=0.05) no longer overlaps the best trial's. This is the **fold-safe** early-stop (roadmap §2c) — correct for exchangeable CV folds where successive-halving's rank-preservation assumption fails. Decision-level test (clearly-worse trial pruned once enough observations accumulate). **374 passed, 0 failed.** ABI unchanged. Doc `docs/methods/racing.md`. Only `hyperband` remains reserved (needs the bracket scheduler / total budget → F5).

### 2026-07-10 — F2 pruners Codex review applied
- Codex review of the pruner block: **1 Blocker, 3 Major, 3 Minor — all applied** (`docs/reviews/finetuning-roadmap/codex-review-04-F2-pruners.md`). **Blocker:** terminal state is now terminal — `tell_intermediate`/`tell_result` reject reports on a non-RUNNING trial (only an idempotent same-status re-report is accepted), so an auto-pruned trial can no longer be overwritten to COMPLETED and win `best()`. **Major:** true 50th-percentile median (mean of the two middle values for even n) — direction-symmetric; **finite-score validation** on both tell paths (rejects NaN/Inf before it corrupts `std::sort`); **centralized pruner-kind validation** in `make_optimizer` (out-of-range/unimplemented pruner is rejected, not silently degraded to none). **Minor:** one value per `(trial, step)` (update-in-place); ASHA `reduction_factor` documented as fixed-at-3 for F2 + ties-survive semantics. +2 regression tests (pruned-is-terminal, invalid-pruner + NaN). **373 passed, 0 failed.**

### 2026-07-10 — F2: ASHA pruner
- Added `N4M_PRUNER_ASHA` (asynchronous successive halving) into the same factory: at each rung a trial survives only if in the top 1/reduction_factor (=3) of the peers at that rung, decided asynchronously (sound for a per-`tell` verdict). Decision-level test on a canned history. **371 passed, 0 failed.** ABI unchanged. Doc `docs/methods/asha.md`. (Hyperband bracket-scheduler + racing + the `n_components` fidelity engine remain in F2.)

### 2026-07-10 — F2 opener: pruner architecture + median pruner
- Added the `Pruner` abstraction (`cpp/src/core/optimization/pruners.cpp`): `Optimizer` holds a `unique_ptr<Pruner>` set by a `make_pruner()` factory from `opts.pruner`; `tell_intermediate()` delegates the keep/prune verdict and marks pruned trials `N4M_TRIAL_PRUNED`. This makes pruners **orthogonal to samplers** (composed via the options struct), matching the roadmap's sampler ⟂ pruner split. `make_optimizer` now accepts `NONE`+`MEDIAN`; `asha`/`hyperband`/`racing` slot into the same factory later.
- Implemented `N4M_PRUNER_MEDIAN` (Vizier median stopping rule): prune when a trial's intermediate score is worse than the median of peer scores at the same step; never before `min_peers` (= `n_startup_trials`) peers. Decision-level test on a canned history. **370 passed, 0 failed.** ABI unchanged. Doc `docs/methods/median_pruner.md`.

### 2026-07-10 — F1 samplers Codex review applied
- Codex review of ternary+lhs: **0 Blocker, 5 Major, 2 Minor — all applied** (`docs/reviews/finetuning-roadmap/codex-review-03-F1-samplers.md`). Ternary reworked into **grid-index space**: honours `step` (proposes only on-grid values), **reserves RUNNING trials** so batched asks don't collide, skips inactive/off-domain history, and keeps arithmetic bounded (index space, guard against absurdly wide ranges). Base `enqueue()` now **validates numeric ranges + categorical indices** (rejects out-of-range warm-starts). LHS: seed **domain-separated** from the base RNG; `size_t` index comparison; clearer Fisher-Yates. +2 regression tests (stepped ternary + batch reservation distinctness, enqueue out-of-range rejection). **369 passed, 0 failed.**

### 2026-07-10 — F1: lhs sampler
- Added `N4M_SAMPLER_LHS` (`cpp/src/core/optimization/lhs.cpp`): Latin Hypercube over the numeric axes for the first `n_startup_trials` asks (one independent permutation per dimension + per-cell jitter), random beyond the batch and for categoricals. Refactored the unit→value mapping into `Optimizer::numeric_from_unit()` so `random`/`lhs`/(future) `sobol` share it. Test asserts each decile is hit exactly once across the startup batch. **367 passed, 0 failed.** ABI unchanged. Doc `docs/methods/lhs.md`.
- **`sobol` is intentionally deferred:** a *useful* Tier-A Sobol must bit-match `scipy.stats.qmc.Sobol(scramble=False)`, which requires the exact Joe–Kuo `new-joe-kuo-6.21201` direction-number table. Rather than ship an approximate/incorrect sequence, it will be done deliberately with the real table (fetch + embed a modest dimension subset) as a dedicated block.

### 2026-07-10 — F1: ternary sampler
- Added `N4M_SAMPLER_TERNARY` (`cpp/src/core/optimization/ternary.cpp`): unimodal-integer ternary search porting the nirs4all `BinarySearchSampler` (triplet anchor low/high/mid → bisect the larger gap toward the current best). Introduced a small base-class hook `override_numeric()` so adaptive samplers reuse the constraint/loop/conditions machinery without duplicating `sample()`. Proposal is a pure function of the completed history (idempotent within an ask). Tunes the first integer axis; others stay random.
- Test: converges to k∈{6,7,8} on a unimodal objective in ≤25 trials. **366 passed, 0 failed.** ABI snapshot unchanged (enum-value-only, no new symbol) — confirms the "later samplers add no ABI symbol" design. Doc `docs/methods/ternary.md`.

### 2026-07-10 — F0 Codex review applied
- Codex read-only review of F0: **14 findings (4 Blocker, 9 Major, 1 Minor), all applied** (transcript `docs/reviews/finetuning-roadmap/codex-review-02-F0.md`). Highlights: made `optimization.h` independently includable (moved the `N4M_STATIC_ASSERT` macro above the role-header includes and relocated the HPO enum asserts into `optimization.h`) + added C/C++ compile-only include guards; hardened the ABI boundary (try/catch on every name-based trial accessor, `std::string_view` lookups, `struct_size` default-preserving copy with a `< 8` guard); made constraints authoritative (condition constraints reject unsupported shapes with `N4M_ERR_UNSUPPORTED`, enqueue validates param names, sampling skips RNG for forced dims and re-checks constraints, `ask` returns an error on constraint-exhaustion instead of a silent invalid trial); validated numeric ranges (reject NaN/Inf, log needs positive bounds); default `direction = AUTO` (derive from metric); `n4m_finetune_estimator` now rejects unsupported params, returns the full trial trace, and returns `NOT_FITTED` when nothing completes; implemented `timeout`/`duration` via `steady_clock`; documented MUTEX_GROUP (nirs4all `_mutex_` issubset) semantics.
- 6 new regression tests lock the fixes (invalid ranges, struct_size guard, enqueue warm-start, conditional activation + conflicting-parent rejection, finetune unsupported-param rejection, AUTO+R2 maximization). **365 passed, 0 failed.** ABI snapshot unchanged (internal fixes), version-sync green.

### 2026-07-10 — F0 green
- **Build clean, all tests pass: 359 passed, 0 failed** (7 new optimization tests incl. a real `n4m_finetune_estimator` PLS-CV run; `abi_version_compatible_with_header` green after the bump).
- ABI snapshots regenerated (linux 734 + derived macos/windows 733) — drift is exactly the 31 new additive `n4m_*` symbols, nothing removed. ABI minor bumped **2.0 → 2.1**; `docs/abi/changes_log.md` entry added; CHANGELOG updated.
- Docs: `docs/methods/{optimization,random}.md` + `_finetuning_bibliography.bib` (14 refs). 
- **Deferred to Track Q (not blocking F0 landing):** catalog `optimization.{random,none}.yaml` + `validate.py --strict-abi` reconcile (non-strict passes; new symbols are warnings), and the `HpoSpec` + parity comparators + CI job. The C++ doctest is F0's correctness gate with a single sampler.
- Next: commit F0, Codex review, then F1 (sobol/lhs/ternary).

### 2026-07-10 — F0 code + env fix
- Wrote F0: `cpp/include/n4m/optimization.h` (frozen ABI), `cpp/src/core/optimization/optimizer.{hpp,cpp}` (SearchSpace + Optimizer: random sampler, none pruner, constraint rejection, conditional activation, ask/tell), `cpp/src/c_api/c_api_optimization.cpp` (all wrappers + `n4m_finetune_estimator` internal PLS-CV driver), `cpp/tests/test_optimization.cpp` (7 cases). Wired n4m.h include + 4-byte enum asserts, CMakeLists (core .cpp), tests CMake + main.cpp registration.
- **Toolchain/env fix (matters for every build here):** the conda-linked `~/.local/bin/gfortran` used by FITPACK pulls a `libm.so` linker script referencing the non-existent `/lib64/libm.so.6`, breaking the `libn4m.so` link. FITPACK (spline smoothing) has a non-Fortran fallback and is irrelevant to HPO, so configure with **`-DCMAKE_Fortran_COMPILER=NOTFOUND`** to disable it. C/C++ compiler is system `/usr/bin/{cc,c++}`; real libm is `/usr/lib/x86_64-linux-gnu`. Apply the same flag to the dev-release build used for ABI snapshots.

### 2026-07-10 — setup
- Created isolated worktree `_worktrees/native-hpo-phase1` on branch `feat/native-hpo-phase1` from `main` (f69a2ec7). Main checkout left clean/untouched.
- Moved + committed the 4 design docs (strategy, roadmap, F0 PR, Codex review) into the branch (`e39ac038`).
- Toolchain: cmake 4.3.2, ninja, g++ (no clang++; g++ path).
- Next: baseline build sanity, then F0.
