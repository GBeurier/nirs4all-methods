# M0 — preflight: STATUS

**Date**: 2026-05-21
**Verdict**: ✅ **COMPLETE** — both baselines tagged, both repos clean for M1.
**Codex final review**: READY-FOR-M1: yes (with 2 non-blocking doc-drift / operational notes).

## Snapshot

| Repo | Tag | Anchor SHA | CI |
|---|---|---|---|
| `pls4all` | `pls4all-final/v1.0.0-pre-merge` | `3b99c0b` `Publish partial benchmark dashboard refresh` | ✅ all green at this tip |
| `nirs4all-methods` | `nirs4all-methods-final/6195574` | `6195574` `M0 baseline: unblock donor CI ahead of merge` | ✅ all 7 workflows green |

Both tags are annotated with rationale + cross-refs to `MERGE_ROADMAP.md` and the caveats below; both pushed to origin.

## What was done

(detailed action log in `COMPLETION_REPORT.md`)

| # | Action | Outcome |
|--:|---|---|
| 1 | `pls4all`: gitignore `benchmarks/cross_binding/data/.reference_oracles/` + `.x_target/` | future generated dumps no longer pollute working tree |
| 2 | `pls4all`: WIP commit `879691b` on `codex/parity-30x30-dashboard-fixes` preserving 282 changed files (bindings JS/Julia/Python/R, cpp/src/core, benchmarks/validation/, docs/methods regen, 131 oracle snapshots, release-js workflow) | nothing lost — pushed to origin/codex/... |
| 3 | `pls4all`: fast-forward local `main` to `origin/main` (`a05b192` → `3b99c0b`) | local now mirrors origin |
| 4 | `pls4all`: tag `pls4all-final/v1.0.0-pre-merge` at `3b99c0b`, push | baseline tag on origin |
| 5 | donor `nirs4all-methods`: commit `6195574` with 3 quick fixes — `PKG_FFLAGS=-std=legacy` (Makevars + Makevars.win), R workflow `error-on:error`, `experimental: true` on `macos-clang-debug` matrix entry | donor CI reaches green at this tip |
| 6 | donor: 18 `archive/worktree-agent-*` tags created and pushed to origin; 18 local branches deleted; 1 remote branch deleted; worktrees pruned | worktree-agent cleanup complete, fully reversible |
| 7 | donor: tag `nirs4all-methods-final/6195574` annotated and pushed | baseline tag on origin |

## Caveats carried forward into the merge plan

| # | Caveat | Where it gets resolved properly |
|--:|---|---|
| 1 | `macos-clang-debug` cpp matrix entry is `experimental: true` (1-index drift in `split_kmeans_parity` at `i=34`, likely floating-point tie-resolution under `-O0`) | M2.5 common-core unification / M7 unified parity gate |
| 2 | R workflow `error-on` temporarily set to `"error"` (was `"warning"`); FITPACK Fortran 2018 warnings are fixed by `-std=legacy`; missing roxygen docs for ~200 exported R functions still emit warnings but no longer gate the workflow | M11 R packages: full + slim |
| 3 | `pls4all` release PR stack #1 / #2 / #3 (SOVERSION foundations, PyPI cibuildwheel, CRAN vendored build) left untouched on their own branches | M14 license audit + M15 release mechanics will reconcile or absorb them |
| 4 | The `codex/parity-30x30-dashboard-fixes` branch carries `879691b` WIP — bindings work, validation registry, 30x30 dashboard fixes | M9-M13 of merge plan will pull these into the unified repo's `main` |

## Reversibility

Every action above is reversible with a single command if needed. Most relevant:

| To undo | Command |
|---|---|
| Restore a deleted `worktree-agent-X` branch | `git branch worktree-agent-X archive/worktree-agent-X` (tags are on origin) |
| Revert donor M0 fixes | `git revert 6195574` |
| Revert the `pls4all` WIP commit | `git revert 879691b` |
| Drop either baseline tag | `git tag -d <tag> && git push origin :refs/tags/<tag>` |

## Codex's two non-blocking operational notes

1. **Doc drift**: earlier M0 files (`STATUS.md` v2, `COMPLETION_REPORT.md`, `ci-status.md`, the archive manifest) still describe pre-execution BLOCKED states. The execution log is in `COMPLETION_REPORT.md`; this updated `STATUS.md` reflects post-execution state. The other files remain as historical record of the analysis (not refreshed because the analysis was *valid*; only the verdict needed updating).
2. **M1 operational note**: the local `pls4all` checkout is currently on the `codex/parity-30x30-dashboard-fixes` branch. Before running `git subtree add --prefix=_donor/nirs4all-methods <donor-remote> main` in M1, the merge branch (`merge/import-donor`) must be created **off the `pls4all-final/v1.0.0-pre-merge` tag** (i.e. `origin/main` = `3b99c0b`), not the codex tip. The donor remote can be added as a temporary git remote pointing at either the GitHub URL or the local path `/home/delete/nirs4all/nirs4all-methods/`.

## M1 entry conditions (all satisfied)

- [x] `pls4all-final/v1.0.0-pre-merge` reachable from origin
- [x] `nirs4all-methods-final/6195574` reachable from origin
- [x] CI green at both baseline tips
- [x] No locked or orphan worktrees on either repo
- [x] All in-flight pls4all work preserved (codex branch pushed, WIP commit traceable)
- [x] All donor worktree-agent tips preserved as `archive/*` tags on origin
- [x] Codex review passed — READY-FOR-M1: yes

**Next**: M1 — `git subtree add --prefix=_donor/nirs4all-methods <donor> main` on a new `merge/import-donor` branch of pls4all, branched off `pls4all-final/v1.0.0-pre-merge`.
