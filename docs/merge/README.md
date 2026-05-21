# Merge roadmap status — `pls4all` ⤳ `nirs4all-methods` unification

This directory ships the status reports and procedures produced during
the M0–M17 merge sprint. Each subdirectory carries the work-product of
its phase (analysis, reconciliation, gate evidence, follow-up plan).

| Phase | Status | Doc |
|------:|:------|-----|
| M0 | ✅ done | [`M0/STATUS.md`](M0/STATUS.md) + [`M0/COMPLETION_REPORT.md`](M0/COMPLETION_REPORT.md) + [`M0/worktree-inventory.md`](M0/worktree-inventory.md) + [`M0/donor-archive-manifest.json`](M0/donor-archive-manifest.json) + [`M0/ci-status.md`](M0/ci-status.md) + [`M0/pls4all-cleanup-status.md`](M0/pls4all-cleanup-status.md) |
| M1 | ✅ done | [`M1/STATUS.md`](M1/STATUS.md) |
| M2 | ✅ done | (in commit `ad43f6a`; no separate STATUS doc) |
| M2.5 | ✅ done | [`M2.5/RECONCILIATION.md`](M2.5/RECONCILIATION.md) |
| M3 | ✅ done | [`M3/STATUS.md`](M3/STATUS.md) |
| M4 | 🟡 scaffold + recipe | [`M4/SPLIT_PLAN.md`](M4/SPLIT_PLAN.md) + [`M4/EXTRACTION_RECIPE.md`](M4/EXTRACTION_RECIPE.md) |
| M5 | 🟡 script ready, deferred | [`M5/STATUS.md`](M5/STATUS.md) |
| M6 | 🟡 stubs only | (in commit `b0424c4`) |
| M7+M8 | 🟡 partial | [`M7/STATUS.md`](M7/STATUS.md) |
| M9–M11 | 🟡 scaffolds | (in commit `8b027ef`) |
| M12 | ⚪ deferred | — |
| M13 | 🟡 donor bindings lifted; secondary refresh deferred | (in commit `8b027ef`) |
| M14 | ✅ done | (in commit `855e426`; see `NOTICE.md` + `THIRD_PARTY_LICENSES.md` at repo root) |
| M15 | 🟡 scaffold | (in commit `855e426`; `.github/workflows/release-wheels.yml` gated `if: false`) |
| M16 | 📄 procedure documented | [`M16/M16_REPO_RENAME.md`](M16/M16_REPO_RENAME.md) |
| M17 | 📄 procedure documented | [`M17/M17_RELEASE_V1.md`](M17/M17_RELEASE_V1.md) |

For the verification report (what actually works end-to-end), see [`VERIFICATION_REPORT.md`](VERIFICATION_REPORT.md).

For the master roadmap (rationale, target tree, parity strategy, subset
machinery), see [`MERGE_ROADMAP.md`](MERGE_ROADMAP.md).

## How the M-numbers map to phases

The numbering follows `MERGE_ROADMAP.md` §9 (the 17-phase work
breakdown). `M2.5` is the common-core unification that the v0 draft
missed — added during Codex review.

## Convention

Each phase committed at least one git commit on
`merge/import-donor`. Status docs above are post-execution reports.
Commits referenced in the table land the actual deliverables; status
docs explain the deliverable, what gate it satisfies, and what is
deferred.

## Open items not blocking v1.0.0-rc1

(see [`M17/M17_RELEASE_V1.md`](M17/M17_RELEASE_V1.md) preflight checklist)

- M4 full extraction (stubs + recipe ready; execution gated by M5+M6)
- M5+M6 ABI rename + header reorganisation (paired focused session)
- M7+M8 unified runner + dashboard wiring
- M9+M10+M11 package activation (vendored-source build via
  catalog/scripts/render_subset.py)
- M11 CRAN cleanup pass (existing 1 WARNING + 4 NOTEs)
- M13 secondary binding refresh after M5 lands
- M15 cibuildwheel matrix activation (currently gated `if: false`)

These ship in a v1.1.0 follow-up release pass per the roadmap budget.
