# Donor (`nirs4all-methods`) worktree-agent inventory

**Repo**: `https://github.com/GBeurier/nirs4all-methods`
**Inspected at**: 2026-05-21
**Inspected at commit (main tip)**: `95906aa Rename project to nirs4all-methods`
**Local path**: `/home/delete/nirs4all/nirs4all-methods/`

## Summary

| Layer | State |
|---|---|
| Total `worktree-agent-*` branches | 18 |
| Active worktree working dirs on disk | 0 (all parent dir `/home/delete/nirs4all/chemometrics4all/` no longer exists — was renamed) |
| `git worktree list` entries pointing to non-existent paths | 18, **all marked `locked`** — `git worktree prune` skips locked entries, so cleanup needs an explicit unlock first |
| Branches whose phase content is integrated into main | **18 / 18** (each phase was integrated by aggregate commits — see §"Integration evidence" below) |
| Branches pushed to `origin` | 1 (`worktree-agent-a62b667d41c6c951a`) — no PR refers to it |
| Author on every tip | `gbeurier` (the maintainer) |
| Tip date range | 2026-05-18 → 2026-05-19 |

## Per-branch table

| # | Branch | Tip | Date | Ahead | Behind | Files | Remote | Phase | Subject (tip commit) | Content-in-main? | Recommendation |
|--:|---|---|---|--:|--:|--:|:-:|---|---|:-:|:--|
| 1  | `worktree-agent-a62b667d41c6c951a` | `7476249` | 2026-05-19 | 1 | 37 | 21 | ✓ | phase-10 | phase-10: add resampling preprocessors | ✓ | **close + delete remote** |
| 2  | `worktree-agent-a4d683af3b4d9c636` | `c242757` | 2026-05-18 | 1 | 37 | 37 |  | phase-11 | phase-11: 9 sample-partitioning splitters (c4a_split_* category) | ✓ | **close** |
| 3  | `worktree-agent-a40029a940a5f7368` | `376b740` | 2026-05-18 | 1 | 37 | 17 |  | phase-12 | phase-12: YOutlierFilter (4 methods) | ✓ | **close** |
| 4  | `worktree-agent-a402c58f3cc0f19b4` | `077cf58` | 2026-05-18 | 1 | 34 | 26 |  | phase-13 | phase-13: XOutlierFilter (6 methods incl. Mahalanobis + IsolationForest + LOF) | ✓ | **close** |
| 5  | `worktree-agent-a62f804de4a887c03` | `67f0e9c` | 2026-05-18 | 1 | 37 | 25 |  | phase-14 | phase-14: HighLeverage + SpectralQuality + Composite filters | ✓ | **close** |
| 6  | `worktree-agent-a23aefc9f7105e3c3` | `e6b4e24` | 2026-05-18 | 1 | 34 | 49 |  | phase-15 | phase-15: 7 noise + drift augmenters (c4a_aug_* category) | ✓ | **close** |
| 7  | `worktree-agent-a9860ac0187876fda` | `ec435cd` | 2026-05-18 | 1 | 34 | 51 |  | phase-16 | phase-16: 10 wavelength + spectral augmenters | ✓ | **close** |
| 8  | `worktree-agent-a930356face899912` | `a9a3c28` | 2026-05-18 | 1 | 34 | 45 |  | phase-17 | phase-17: 10 mixup + physical + environmental augmenters | ✓ | **close** |
| 9  | `worktree-agent-ac746cf475ecf61cf` | `1d93ca1` | 2026-05-18 | 1 | 34 | 47 |  | phase-18 | phase-18: 12 edge artifacts + splines + random augmenters | ✓ | **close** |
| 10 | `worktree-agent-a737bd14b02708e7b` | `a9a45a2` | 2026-05-18 | 1 | 37 | 29 |  | phase-19 | phase-19: signal type detection + NIRS metrics + Hotelling T²/Q-residuals | ✓ | **close** |
| 11 | `worktree-agent-aeb108cd1920422ff` | `ee28761` | 2026-05-18 | 1 | 37 | 14 |  | phase-20 | phase-20: transfer metrics (9-metric vector for dataset alignment) | ✓ | **close** |
| 12 | `worktree-agent-ae0957170d00f1e1b` | `103a708` | 2026-05-18 | 1 | 37 | 14 |  | phase-21 | phase-21: FCK static transformer (fractional convolutional kernels) | ✓ | **close** |
| 13 | `worktree-agent-ac966dcc666177491` | `bb6619c` | 2026-05-18 | 1 | 29 | 26 |  | phase-22 | phase-22: Python binding (ctypes + sklearn tier-2, 20 ops covered by Gate 1) | ✓ | **close** |
| 14 | `worktree-agent-aeca18f184655fdec` | `7089dab` | 2026-05-18 | 1 | 29 | 18 |  | phase-23 | phase-23: R binding (.Call + idiomatic R API, 15 ops covered by Gate 1) | ✓ | **close** |
| 15 | `worktree-agent-a5893245d6c36424d` | `2ae5319` | 2026-05-18 | 1 | 34 | 44 |  | phase-6  | phase-6: wavelets & denoising (6 operators + filter banks) | ✓ | **close** |
| 16 | `worktree-agent-a2f069496633f1cb9` | `7737f26` | 2026-05-18 | 1 | 37 | 24 |  | phase-7  | phase-7: signal type conversion preprocessings (5 operators) | ✓ | **close** |
| 17 | `worktree-agent-acf7d86e13d2d30ec` | `1bbe493` | 2026-05-18 | 1 | 37 | 18 |  | phase-8  | phase-8: OSC + EPO orthogonalization preprocessings | ✓ | **close** |
| 18 | `worktree-agent-a31c030eb1c9da7da` | `55a27e8` | 2026-05-18 | 1 | 37 | 21 |  | phase-9  | phase-9: FlexiblePCA + FlexibleSVD (CARS/MCUVE deferred) | ✓ | **close** |

## Integration evidence — how the phase content reaches `main`

The phase content is on `main` today, but the relationship between each `worktree-agent-*` branch and `main` is **not** "file-identical sub-tree" or a clean cherry-pick. Two checks tell the real story:

1. **`git cherry main <branch>` returns `+` for every branch.** That means git does not recognise the branch's unique commit as a direct cherry-pick of any commit on `main`. Each branch is a divergent line of history.
2. **File-level diff `main..<branch>` shows large delta** (e.g. `main` is 1683 files / 33344 lines forward of `worktree-agent-a23aefc9f7105e3c3` and 204216 lines backward — the *deletion* count means whole subtrees were restructured on `main` after the phase was integrated).

So each branch was **integrated by later aggregate commits on `main`** (restructuring, renames, follow-on phases), not by a clean squash that left tree-equivalent content. The integration story is: the phase landed, then subsequent main commits revised the same files heavily.

**Conclusion**: the branches are semantically obsolete (their work is on main, but the exact tree they hold is not). Keeping them is purely archival of "what the work looked like before later refactors" — useful for archeology, not for active development. The "close all" recommendation stands; the archive tag (step 2 in cleanup) preserves the obsolete tree if archeology is ever needed.

## Recommendation

**Close all 18 branches** — the content is preserved on `main`, the worktree working directories are already gone, and no PR references them.

For `worktree-agent-a62b667d41c6c951a` (the only one pushed to GitHub), also delete the remote ref.

### Cleanup commands (one-shot — needs maintainer approval before running)

**Order is load-bearing.** Each `worktree-agent-*` branch is **checked out by a (locked) linked worktree** — the `+` prefix in `git branch` output. While that worktree pointer exists, `git branch -D <name>` fails with `error: Cannot delete branch '<name>' checked out at '<path>'`. The unlock and prune steps must happen **before** branch deletion.

```bash
cd /home/delete/nirs4all/nirs4all-methods

# 1) Archive-tag every branch tip FIRST so nothing is lost.
for br in $(git branch | grep -E '^[ +*] worktree-agent' | sed 's/^[+ *] //'); do
  git tag "archive/$br" "$br"
done

# 2) Push the archive tags to origin EXPLICITLY (refspec per tag, not --tags
#    which could push unrelated tags). This makes the obsolete tree state
#    discoverable for archeology after the donor repo is archived in M16.
for br in $(git branch | grep -E '^[ +*] worktree-agent' | sed 's/^[+ *] //'); do
  git push origin "refs/tags/archive/$br"
done

# 3) Unlock every locked worktree.
#    `git worktree prune` and `git worktree remove` both refuse locked entries.
for path in $(git worktree list --porcelain | awk '/^worktree / {print $2}' | grep chemometrics4all); do
  git worktree unlock "$path" 2>/dev/null || true
done

# 4) Prune the orphan worktree pointers. The working dirs are already gone
#    from disk; this clears git's metadata and frees each branch to be deleted.
git worktree prune --verbose

# 5) Now safe to delete local branches (they are no longer "checked out").
for br in $(git branch | grep -E '^[ +*] worktree-agent' | sed 's/^[+ *] //'); do
  git branch -D "$br"
done

# 6) Delete the only remote ref (`worktree-agent-a62b667d41c6c951a`).
#    Its tip is preserved by `archive/worktree-agent-a62b667d41c6c951a`
#    which was pushed in step 2.
git push origin :worktree-agent-a62b667d41c6c951a
```

### Why archive-tag AND push-tags BEFORE deleting (load-bearing)

`git branch -D` (force-delete) on a branch whose tip is not reachable from any ref will eventually be garbage-collected. The archive tags pin those tips so `git fsck --no-reflogs` cannot prune them; they survive the donor repo archive flag in M16 and any future `git gc --aggressive`.

Pushing the archive tags **before** deleting the remote `worktree-agent-a62b667d41c6c951a` ref means the only on-remote tip is mirrored to a permanent ref **first**. If the tag push fails for any reason, the remote branch is still there — nothing is lost. The reverse order (delete remote ref first, then push tags) creates a window where a tag-push failure leaves the tip orphaned.

### Decision policy applied (per resolved §11.3 of MERGE_ROADMAP.md)

- The user's resolution was *"inventaire + revue"*. This file is the inventory; the recommendation column is the proposal.
- No branches deleted yet. Maintainer signoff is the gate before the cleanup commands above run.
- If any single branch is contested by another contributor, escalate before deletion (the inventory column "Remote" surfaces such risk).
