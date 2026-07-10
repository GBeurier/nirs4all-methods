# Native HPO — Phase 1 implementation progress log

**Branch:** `feat/native-hpo-phase1` · **Worktree:** `_worktrees/native-hpo-phase1` · **Scope:** `nirs4all-methods` only (stop before dag-ml / nirs4all-core / any other repo).
**Plan:** [`FINETUNING_ROADMAP.md`](FINETUNING_ROADMAP.md) §6 (two-phase + parallel tracks) · [`FINETUNING_F0_PR.md`](FINETUNING_F0_PR.md) (F0 detail).
**Rule:** each big block is Codex-reviewed before moving on. Everything stays green (`ctest dev-release`) + ABI-gated.

## Status board

| Block | State | Codex review | Notes |
|---|---|---|---|
| Setup (worktree, baseline build) | ✅ done | — | isolated worktree; docs committed; toolchain OK |
| **F0** — ABI surface + `random`/`none` slice + scaffolding | ✅ done | 🟡 pending | builds clean; 359/359 tests pass; ABI 2.1 + snapshots + changelog + docs. Catalog `--strict-abi` reconcile + HPO parity CI (Track Q) deferred. |
| **F1** — samplers: sobol, lhs, ternary | ⬜ todo | ⬜ | Track S |
| **F2** — pruners: fidelity engine → median, asha, hyperband, racing | ⬜ todo | ⬜ | Track P |
| **F3** — RNG consolidation → ga, pso | ⬜ todo | ⬜ | Track G |
| **F4** — cmaes, tpe, (gp_ei?) | ⬜ todo | ⬜ | Track M |
| **Q** — HpoSpec + comparators + parity CI | ⬜ todo | ⬜ | Track Q (start early) |
| **B** — bindings python→R/MATLAB/WASM + cross-binding gate | ⬜ todo | ⬜ | Track B |
| **Phase 1 done → WAIT for green light** | ⬜ | — | do NOT touch dag-ml/core |

Legend: ⬜ todo · 🟡 in progress · ✅ done · 🟢 done+reviewed

## Log

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
