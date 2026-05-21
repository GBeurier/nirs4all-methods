# M0.d — CI status of both repos — **revision 2** (after Codex M0 review)

## `pls4all` (base) — baselines (refer to `pls4all-cleanup-status.md` for the merge-baseline decision)

| Anchor | Full SHA | Subject |
|---|---|---|
| `v0.97.3` (release tag) | `49dc7507446895b32997f85a6b59803e35d2f823` | Prepare adaptive benchmarks and R release |
| `origin/main` | `3b99c0b4a348c271b22deff1c4822709522f451b` | Publish partial benchmark dashboard refresh |
| Local `main` (stale, 4 behind) | `a05b1928c191b1c665a74526e7f56686f42db47b` | Make long benchmark shards resumable |
| `codex/parity-30x30-dashboard-fixes` (HEAD, pushed) | `6df4611a03522f0b0a9a85515e6507e8d26063c5` | Fill strict binding divergence scores |

### CI verdict per anchor

| Anchor | Workflow result | Comment |
|---|---|---|
| `v0.97.3` | ✅ green at release time | The release was cut at `49dc750`; CI green at that SHA is the v0.97.3 contract. |
| `origin/main` (`3b99c0b`) | ✅ all green | All cpp matrix jobs + sanitizers + parity gate + version-sync + docs pass. |
| local `main` (`a05b192`) | ✅ green | Same as origin/main's prior tip from CI history; local just lags. |
| `codex/parity-...` (`6df4611`) | ⚠️ no check runs at this SHA on origin | The branch is pushed but workflows are gated and never ran at this tip. Before tagging the merge baseline at the post-merge tip, **CI must be re-run on the merged commit and confirmed green.** |

**Practical implication.** The merge baseline is **not** any existing anchor. Per `pls4all-cleanup-status.md` "Option A revised", the baseline becomes the merged tip after:
1. local `main` is fast-forwarded to `origin/main`,
2. the codex branch's in-flight work is committed and PR'd into `main`,
3. CI runs green on the merged tip,
4. that new tip gets the additive tag `pls4all-final/v1.0.0-pre-merge`.

The existing `v0.97.3` tag at `49dc750` is preserved unchanged.

The decision between A1 (land release PR stack first), A2 (close release PRs and redo in M15), and A3 (cherry-pick SOVERSION fix only) sits in `pls4all-cleanup-status.md`; CI implications: A1 adds ~1 week of release-PR CI cycles before M0 closes; A3 adds one CI cycle for the cherry-pick PR only.

---

## `nirs4all-methods` (donor)

### `main` tip `95906aae18ceb882835e25b7ccaa2887f80f02cd`

**CI verdict: ❌ RED.**

Most recent workflow run at the tip:

| Workflow | Result |
|---|---|
| `version-sync` | ✅ |
| `Sanitizers` | ✅ |
| `docs` | ✅ |
| `CI` (cpp matrix) | ❌ failing — see below |
| `release-r` (R CMD check on 4 platforms) | ❌ failing — see below |

### CI cpp matrix failure detail

One platform job fails:

| Job | Failure |
|---|---|
| `cpp / macos-clang-debug` | `n4m_tests: 265 passed, 1 failed`<br>`FAIL split_kmeans_parity: kmeans/seed42_ts_0_25/train: index mismatch at i=34 (got 45, want 46)` |

All other cpp matrix jobs (linux-gcc12/13, linux-clang16, macos-clang-release, macos-universal2, windows-msvc-release/debug, windows-mingw, debug variants) pass.

**Root-cause hypothesis**: platform-specific tie-breaking in k-means initialisation on macOS clang debug build. Fix is small (deterministic tie-resolution by lower-index-first) but must be verified.

### release-r (R CMD check) failure detail

All 4 R CMD check jobs (linux-release, linux-devel, windows-release, macos-arm64-release) fail with the same two warnings, treated as errors by the workflow's `error-on: "warning"`:

```
Status: 2 WARNINGs
❯ checking whether package 'n4m' can be installed ... WARNING
  Warning: Fortran 2018 deleted feature: Shared DO termination label 20 at (1)
  Warning: Fortran 2018 deleted feature: Shared DO termination label 80 at (1)
  Warning: Fortran 2018 deleted feature: Shared DO termination label 260 at (1)
❯ checking for missing documentation entries ... WARNING
```

**Root causes**:
1. Vendored FITPACK Fortran uses Fortran 77 shared DO termination labels (label 20 / 80 / 260) — modern `gfortran -std=f2018` flags as deleted feature. Fix: add `-std=legacy` to FFLAGS in `bindings/r/n4m/src/Makevars` and `Makevars.win`. (Alternative — modernise FITPACK — is far more work and out of M0 scope.)
2. Missing roxygen entries for some R functions in `bindings/r/n4m/man/`. Fix: add `\name{}` / `\title{}` blocks for the few exported R functions lacking docs (or reduce the public R surface to documented set if the exposures were accidental).

---

## M0.d verdict — **BLOCKED**

The roadmap's M0 gate (`MERGE_ROADMAP.md` §9 row M0) requires **both `gh pr checks` green** at the chosen baselines. The donor side is not green.

### Required fixes before tagging `nirs4all-methods-final/<sha>`

1. **`split_kmeans_parity` macOS debug.** Either tighten tie-resolution in `cpp/src/core/splitters/kmeans.c` so the index choice is platform-stable, or relax the fixture tolerance for this case in `parity/divergences.json` with a documented rationale and a tracking issue.
2. **FITPACK Fortran flag.** Add `-std=legacy` to `FFLAGS` in `bindings/r/n4m/src/Makevars` and `Makevars.win`.
3. **R documentation entries.** Restore the missing roxygen `\title{}` / `\param{}` / `\return{}` blocks so `R CMD check --as-cran` passes with zero WARNINGs.

After each push, wait for the workflow to go green. Only then tag `nirs4all-methods-final/<merged-tip-sha>` and proceed to M1.

These three fixes are estimated 2-3 hours combined. They are **not** deferrable to M11 of the merge plan — the M0 gate requires green CI here, and starting M1 with a red donor amplifies every later debugging session.

### Required action before tagging `pls4all-final/v1.0.0-pre-merge`

Per `pls4all-cleanup-status.md` Option A revised:
1. Sync local `main` to `origin/main` (fast-forward).
2. Commit the codex branch's in-flight real source changes (5 focused commits).
3. Decide A1/A2/A3 for the release PR stack.
4. Merge codex → main.
5. Wait for green CI on the merged tip.
6. Tag.

The earlier-revision "ready to be tagged at `a05b192`" claim was wrong (local main is stale; `v0.97.3` is at a different commit; baseline is a future merged tip).
