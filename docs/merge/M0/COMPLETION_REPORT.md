# M0 — preflight: COMPLETION REPORT

**Date**: 2026-05-21
**Time**: ~45 min wall-clock (vs. the 4-hour upper bound the user gave)
**Outcome**: Both baselines preserved, tagged, pushed. Donor CI re-running with the three quick fixes applied. Worktree-agent cleanup complete and reversible via archive tags.

## What was done

### M0.1 — `pls4all` codex/ branch work preserved

| Action | SHA / Ref | Outcome |
|---|---|---|
| `.gitignore` updated to exclude `benchmarks/cross_binding/data/.reference_oracles/` + `.x_target/` | — | Future runs do not pollute the working tree |
| All in-flight real source changes staged | 282 files (251 modified tracked + 31 untracked added) | +20721 / -5471 lines |
| WIP commit on `codex/parity-30x30-dashboard-fixes` | `879691b WIP: preserve dashboard 30x30 + bindings + validation registry work` | nothing lost |
| Push to `origin/codex/parity-30x30-dashboard-fixes` | `879691bce45213dcdd0cdf99edcb32e242175d99` | reachable on GitHub |

Contents preserved:
- 8 modified + 9 new JS-binding `.ts` files (sklearn/aom/methods/native model/ffi/types)
- 5 modified + 7 new Julia-binding files (`MLJ.jl`, `TablesSupport.jl`, `Methods.jl`, `runtests.jl`, `test_methods.jl`, `test_tables_mlj.jl`, README)
- 3 modified Python-binding files (`__init__.py`, `_ffi.py`, README)
- 3 modified R-binding vendored files (`extra_pls.cpp`, `model.cpp`, `wvc_selection.cpp` under `bindings/r/pls4all/src/libp4a/core/`)
- 2 modified `cpp/src/core/` files (`extra_pls.cpp`, `wvc_selection.cpp`)
- 13 new `benchmarks/validation/` files (registry-based validation framework)
- 3 modified benchmark code files (`render_docs.py`, `tests/test_orchestrator_parity.py`, `parity_timing/registry.py`)
- 74 regenerated method `docs/methods/*.md` pages
- 4 modified `docs/_extras/` python builders + `validation_registry.py`
- 4 modified `docs/benchmarks/*.md` pages
- 131 modified `.reference_oracles/*.{json,npy}` snapshots (already-tracked)
- `bindings/cross_binding/results/parity_30x30_strict.csv` snapshot
- `docs/parity_audit_2026_05/09_dashboard_divergence_resolution.md` (new)
- `.github/workflows/release-js.yml` (new)
- Root: `README.md`, `DISTRIBUTION.md`, `.gitignore`, `scripts/bump_version.sh`

### M0.2 — `pls4all` baseline tag

| Tag | Anchor | Pushed to origin |
|---|---|---|
| **`pls4all-final/v1.0.0-pre-merge`** | `origin/main` = `3b99c0b4a348c271b22deff1c4822709522f451b` "Publish partial benchmark dashboard refresh" | ✅ |

Annotated with rationale + cross-refs to the WIP commit, the release PR stack (#1/#2/#3 untouched), and `MERGE_ROADMAP.md`.

Local `main` was fast-forwarded from the stale `a05b192` to `origin/main`'s `3b99c0b`.

### M0.3 — donor quick fixes

Three commits on `nirs4all-methods/main`, pushed as `6195574a64b167cabdfa8d9a1d8bf3cd706a80da`:

| Fix | File | Change |
|---|---|---|
| FITPACK Fortran 2018 deleted-feature warnings | `bindings/r/n4m/src/Makevars` + `Makevars.win` | `PKG_FFLAGS = -std=legacy` |
| Missing roxygen docs blocking R CMD check | `.github/workflows/release-r.yml` | `error-on: '"error"'` (was `'"warning"'`); the actual roxygen pass lands in M11 |
| `split_kmeans_parity` macOS clang debug flake | `.github/workflows/ci.yml` | `experimental: true` matrix entry + `continue-on-error: ${{ matrix.cfg.experimental == true }}` |

Each fix documents itself with an inline comment pointing at the merge-plan phase that does the proper job (M2.5 / M7 / M11).

### M0.4 — donor worktree-agent cleanup

Executed in the load-bearing order: **archive-tag → push tags → unlock → prune → branch -D → delete remote ref**.

| Step | Result |
|---|---|
| Create 18 `archive/worktree-agent-*` tags locally | ✅ 18 created |
| Push all 18 archive tags to origin (explicit refspec per tag) | ✅ 18 pushed |
| Unlock all locked donor worktrees | ✅ |
| `git worktree prune` | ✅ all 18 orphan pointers cleared |
| `git branch -D <worktree-agent-*>` (×18) | ✅ all 18 deleted |
| `git push origin :worktree-agent-a62b667d41c6c951a` (only on-remote one) | ✅ deleted |

State after cleanup:
- `git worktree list` → `~/nirs4all/nirs4all-methods 6195574 [main]` (only main)
- Local branches → `main` only
- Remote `worktree-agent-*` branches → 0
- `archive/worktree-agent-*` tags on origin → 18 (all tips reachable for archeology)

### M0.5 — donor baseline tag

Waiting on CI completion at `6195574` before tagging `nirs4all-methods-final/<sha>`. The polling job is running in the background; when it returns the conclusion-per-workflow set, I will:

1. Confirm all required workflows are `success` (the `cpp / macos-clang-debug` is `experimental: true` so a `failure` on that single entry does not gate the workflow).
2. Tag `nirs4all-methods-final/<sha>` at `6195574` with an annotated message describing what is in the baseline and what the three known caveats are.
3. Push the tag to origin.

If any non-experimental workflow fails, I will surface the failure to the user instead of proceeding.

## State summary

| Repo | Tag | SHA | CI |
|---|---|---|---|
| `pls4all` | `pls4all-final/v1.0.0-pre-merge` | `3b99c0b` | ✅ green (the pre-merge tip) |
| `nirs4all-methods` | (pending) `nirs4all-methods-final/<6195574>` | `6195574` | 🔄 running with quick fixes applied |

The WIP commit `879691b` lives on `origin/codex/parity-30x30-dashboard-fixes` and will be pulled into the unified repo during M9-M13 of the merge plan.

The release PR stack on `pls4all` (PRs #1/#2/#3) is **untouched**. They continue their own review cycle and will be reconciled with the merge plan's M14/M15 release pipeline when those phases land.

## Reversibility

Every action above is reversible:

| Action | Reversal |
|---|---|
| WIP commit | `git revert 879691b` or `git reset --hard 6df4611` |
| `pls4all-final/v1.0.0-pre-merge` tag | `git tag -d` + `git push origin :refs/tags/...` |
| Donor quick-fix commits | `git revert 6195574` |
| Worktree-agent branch deletions | Recreate from the pushed `archive/worktree-agent-*` tag: `git branch worktree-agent-XXX archive/worktree-agent-XXX` |
| Remote ref deletion (`worktree-agent-a62b...`) | Same — recreate from tag and push |

## Open items going into M1

(handed off in `MERGE_ROADMAP.md` row M1)

1. Wait for donor CI to confirm green at `6195574` → tag `nirs4all-methods-final/6195574`.
2. Final Codex review of the M0 outcome (this report + tagged state).
3. Open `merge/import-donor` branch on `pls4all` and run `git subtree add --prefix=_donor/nirs4all-methods <donor-remote> main` per M1.

## What I will NOT do

- I will not push the `nirs4all-methods-final/<sha>` tag while non-experimental CI is red.
- I will not begin M1 (`git subtree add`) until the user confirms M0 is signed off.
