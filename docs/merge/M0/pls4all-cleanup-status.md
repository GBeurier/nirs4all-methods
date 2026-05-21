# `pls4all` working-tree cleanup status — **revision 2** (after Codex M0 review)

**Repo**: `https://github.com/GBeurier/pls4all`
**Inspected at**: 2026-05-21
**Local path**: `/home/delete/nirs4all/pls4all/`

## Snapshot — corrected baselines

The v1 of this report used the wrong baseline (local `main` at `a05b192`). Codex caught that local `main` is **4 commits behind** `origin/main`, and that the existing release tag `v0.97.3` already points at a different commit (`49dc750`), not at the wished-for "clean baseline". The corrected view:

| Anchor | Full SHA | Subject |
|---|---|---|
| `v0.97.3` (release tag, exists on origin) | `49dc7507446895b32997f85a6b59803e35d2f823` | Prepare adaptive benchmarks and R release |
| `origin/main` | `3b99c0b4a348c271b22deff1c4822709522f451b` | Publish partial benchmark dashboard refresh |
| Local `main` | `a05b1928c191b1c665a74526e7f56686f42db47b` | Make long benchmark shards resumable |
| `codex/parity-30x30-dashboard-fixes` HEAD (current checkout) | `6df4611a03522f0b0a9a85515e6507e8d26063c5` | Fill strict binding divergence scores |

### Distance map

```
           v0.97.3  ──┐
                      │
   (release branch, not on main)
                      │
                      ▼
       origin/main = 3b99c0b ── +2 ─→ bfd04cb "Fix benchmark dashboard verdict rendering"
                                       3b99c0b "Publish partial benchmark dashboard refresh"
                      │
      (local main is 4 behind origin/main)
                      │
                      ▼
       local  main  = a05b192    (4 commits behind origin/main)

  codex/parity tip = 6df4611  (4 commits ahead of origin/main, ALSO PUSHED TO ORIGIN as origin/codex/parity-30x30-dashboard-fixes)
                              │
                              ├─ 6df4611 Fill strict binding divergence scores
                              ├─ e79112b Fix strict dashboard source and PLS-QDA parity
                              ├─ 3869d02 Fix 30x30 parity gates and dashboard divergence
                              └─ 0429d61 Snapshot stopped benchmark dashboard timings
```

> Earlier v1 of this doc said the codex branch may not be pushed. It **is** pushed: `origin/codex/parity-30x30-dashboard-fixes` = `6df4611`.

### Working tree state (current checkout = codex branch)

| | |
|---|---|
| Modified (tracked) files | **251** |
| Untracked **status entries** | 616 |
| Untracked **files on disk** | **792** (Git porcelain counts `.x_target/` as one entry but it is 165 files on disk) |
| Total real generated files in `benchmarks/cross_binding/data/` | **979** (165 in `.x_target/` + 814 in `.reference_oracles/`) |
| `benchmarks/cross_binding/data/` size on disk | **3.6 GB** |

## Where the changes live

### Modified-tracked breakdown (251 files)

| Category | Count | Nature |
|---|--:|---|
| Benchmark oracle dumps (`benchmarks/cross_binding/data/.reference_oracles/*.{json,npy}`) | **131** | Generated artifacts |
| Docs (mostly `docs/methods/*.md`, 74 files) | **86** | Auto-generated method pages (5163 inserts / 2312 deletes — registry refresh) |
| JS binding (`bindings/js/src/*.ts`) | **8** | Real source (267 inserts / 141 deletes) |
| Julia binding (`bindings/julia/`) | **5** | Real source (130/87) |
| Python binding (`bindings/python/`) | **3** | Real source (107/58) |
| R binding (`bindings/r/pls4all/src/libp4a/core/`) | **3** | Real source (vendored under R src) |
| C++ core (`cpp/src/core/`) | **2** | Real source: `extra_pls.cpp`, `wvc_selection.cpp` (35/7) |
| Benchmark code (`benchmarks/cross_binding/{render_docs,tests}.py`, `parity_timing/registry.py`) | **3** | Real source (45/17) |
| Root docs (`README.md`, `DISTRIBUTION.md`, `.gitignore`) | **3** | Real text |
| Bench result CSV (`benchmarks/cross_binding/results/parity_30x30_strict.csv`) | **1** | Generated snapshot |
| Scripts (`scripts/bump_version.sh`) | **1** | Real source |
| Other docs (`docs/_static/{bench-data.json,custom.css}`, `docs/_extras/build_*.py`, `docs/conf.py`, `docs/dev/readthedocs.md`, `docs/parity/methodology.md`, `docs/benchmarks/*.md`) | **5** | Mix of generated + real |

### Untracked breakdown (792 files on disk)

| Category | File count | Nature |
|---|--:|---|
| `benchmarks/cross_binding/data/.reference_oracles/*.{json,npy}` (untracked subset) | **597** | Generated artifacts |
| `benchmarks/cross_binding/data/.x_target/` (single porcelain entry, real **165 files** + several GB) | **165** | Generated target arrays — bulk of the size budget |
| Real source — JS binding new files (`bindings/js/jsr.json`, `bindings/js/scripts/`, `bindings/js/src/{aom,core,methods,nativeModel,p4a.d}.ts`, `bindings/js/test/run_methods_smoke.mjs`) | **9** | New code |
| Real source — Julia binding new files (`bindings/julia/Pls4all.jl/{README.md,src/MLJ.jl,src/Methods.jl,src/TablesSupport.jl,test/runtests.jl,test/test_methods.jl,test/test_tables_mlj.jl}`) | **7** | New code |
| Real config — new GHA workflow (`.github/workflows/release-js.yml`) | **1** | New workflow |
| `benchmarks/validation/` (single porcelain entry; on disk it contains **13 files**: `export_registry.py`, `models.py`, `__init__.py`, `README.md`, `validate_registry.py`, `tests/test_export_registry.py`, plus `registry/{datasets,suites,references,comparators,tolerances,manifest,methods}.json`) | **13** | **Real code + data, not "1 dir"** — full tree must be reviewed and staged |
| Real source — `docs/_extras/validation_registry.py` | **1** | New code |
| Real text — `docs/parity_audit_2026_05/09_dashboard_divergence_resolution.md` | **1** | New doc |

> The v1 of this doc treated `benchmarks/validation/` as a single untracked dir; that was wrong — it contains active registry code and a 72 KB methods.json. Confirm content before discarding anything.

## Status of `origin/main`

CI: ✅ green (see `ci-status.md`).

The merge-base candidate that respects existing infrastructure is **`origin/main` (`3b99c0b`)**, not local `main`. Local main is stale.

## Open PRs

3 release-prep PRs stacked on `release/{m0-baseline,m1-pypi,m2-cran}`:

| # | Title | Head → Base | State |
|--:|---|---|---|
| 1 | M0 — Foundations: SOVERSION fix, version-sync, SPDX casing normalisation | `release/m0-baseline` → `main` | open |
| 2 | M1 — PyPI: cibuildwheel + auditwheel + retag + Trusted Publishing | `release/m1-pypi` → `release/m0-baseline` | open |
| 3 | M2 — CRAN: vendored static build + roxygen + tests + release-r.yml | `release/m2-cran` → `release/m1-pypi` | open |

**Decision still needed**: do we land these PR stacks before the merge starts (they touch the very same M14/M15 release pipeline), or close them in favour of the unified release pipeline that lands in M15?

## Recommended cleanup path — **revision 2**

### Option A (revised) — "Use codex tip as the merge baseline" — **only safe path**

Codex was correct that Options B/C in v1 used a stale local `main`. Both are unsafe and removed from this revision.

1. **Sync local `main`.**
   ```bash
   cd /home/delete/nirs4all/pls4all
   git fetch origin
   git checkout main
   git merge --ff-only origin/main          # advance local main to 3b99c0b
   ```

2. **Commit the in-flight real source work on `codex/parity-30x30-dashboard-fixes`** in 5 focused commits:
   - `bindings: JS sklearn + AOM + methods native bindings`
   - `bindings: Julia MLJ + TablesSupport + tests`
   - `bindings: Python ffi + R libp4a updates`
   - `cpp: extra_pls + wvc_selection diagnostic fixes`
   - `benchmarks: validation registry + render_docs + parity_timing registry + 30x30 strict CSV`

3. **Add `.gitignore` rules** for the 979 generated oracle/target artifacts (and the `~3.6 GB`) before any final tag. Probably:
   ```gitignore
   # benchmark oracle dumps
   benchmarks/cross_binding/data/.reference_oracles/
   benchmarks/cross_binding/data/.x_target/
   ```
   Stage the doc regen separately (`docs/methods/*.md` are auto-generated from the registry; either commit them with the registry change as one logical PR or ignore them and regenerate at release).

4. **Open a PR `codex/parity-30x30-dashboard-fixes` → `main`** and merge it. Squash if you want a clean linear history. Or rebase the codex commits on top of `origin/main` first and merge fast-forward.

5. **Run CI on the merged tip and wait for green** (the M0 gate per `MERGE_ROADMAP.md` §9 row M0).

6. **Tag the merged `main` tip as `pls4all-final/v1.0.0-pre-merge`** — this becomes the merge baseline.
   - **Note**: the existing `v0.97.3` tag (at `49dc750`) is preserved unchanged; the new tag is additive.

### What the release PR stack does — needs maintainer decision

PR #1 / #2 / #3 (the `release/m0-baseline` → `release/m1-pypi` → `release/m2-cran` chain) overlap heavily with the merge plan's M14 (license audit) and M15 (release mechanics). Three options:

- **A1.** Merge the release PR stack first (lands SOVERSION fix, cibuildwheel, CRAN vendored static build inside `pls4all`), then start the merge. Pro: each piece is reviewable in isolation; con: ~1 week of release-PR-review delay before M0 can ship.
- **A2.** Close the release PR stack and re-land the work inside M14/M15 of the merge plan. Pro: single integrated release pipeline; con: throws away review work already done.
- **A3.** Cherry-pick the SOVERSION fix only (PR #1's smallest piece) into the merge baseline; close PR #2 / #3 and let M15 redo them. Pro: keeps the load-bearing ABI bump; con: harder review traceability.

Recommendation: **A3** — SOVERSION fixes are tiny and naturally belong on the baseline; cibuildwheel + CRAN can wait for M15 where they get a unified treatment.

## What I will NOT do without your approval

- I will **not** run any state-changing git command on either repo (`commit`, `push`, `checkout`, `merge`, `rebase`, `stash --no-keep-index`, `reset --hard`, `tag`, `branch -D`, `worktree unlock/remove/prune`).
- I will **not** delete or move any uncommitted file from either working tree.
- I will **not** touch `.gitignore`.

Tell me whether to proceed with Option A (with sub-choice A1 / A2 / A3) and I'll execute the commands step by step, asking Codex to review after each step.
