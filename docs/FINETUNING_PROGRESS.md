# Native HPO — Phase 1 implementation progress log

**Branch:** `feat/native-hpo-phase1` · **Worktree:** `_worktrees/native-hpo-phase1` · **Scope:** `nirs4all-methods` only (stop before dag-ml / nirs4all-core / any other repo).
**Plan:** [`FINETUNING_ROADMAP.md`](FINETUNING_ROADMAP.md) §6 (two-phase + parallel tracks) · [`FINETUNING_F0_PR.md`](FINETUNING_F0_PR.md) (F0 detail).
**Rule:** each big block is Codex-reviewed before moving on. Everything stays green (`ctest dev-release`) + ABI-gated.

## Status board

| Block | State | Codex review | Notes |
|---|---|---|---|
| Setup (worktree, baseline build) | ✅ done | — | isolated worktree; docs committed; toolchain OK |
| **F0** — ABI surface + `random`/`none` slice + scaffolding | 🟢 done+reviewed | ✅ Codex | 365/365 tests; ABI 2.1; **Codex-reviewed (14 findings all applied)**. Catalog `--strict-abi` reconcile + HPO parity CI (Track Q) deferred. |
| **F1** — samplers: sobol, lhs, ternary | ⬜ todo | ⬜ | Track S |
| **F2** — pruners: fidelity engine → median, asha, hyperband, racing | ⬜ todo | ⬜ | Track P |
| **F3** — RNG consolidation → ga, pso | ⬜ todo | ⬜ | Track G |
| **F4** — cmaes, tpe, (gp_ei?) | ⬜ todo | ⬜ | Track M |
| **Q** — HpoSpec + comparators + parity CI | ⬜ todo | ⬜ | Track Q (start early) |
| **B** — bindings python→R/MATLAB/WASM + cross-binding gate | ⬜ todo | ⬜ | Track B |
| **Phase 1 done → WAIT for green light** | ⬜ | — | do NOT touch dag-ml/core |

Legend: ⬜ todo · 🟡 in progress · ✅ done · 🟢 done+reviewed

## Log

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
