# M5 — ABI rename `n4m_*` → `n4m_*`: STATUS

**Status**: 🟡 **SCRIPT READY, EXECUTION DEFERRED**
**Date**: 2026-05-21 (same merge sprint)
**Pre-rename tag**: `abi-p4a-final/v0.97.3-pre-merge` → `9472dd9` (pushed)

## What was done in this commit

1. **Script written** at `scripts/migrate_p4a_to_n4m.py` — a stdlib-only
   Python tool that performs the mechanical rename across selectable
   stages (`core`, `bindings`, `parity`, `benchmarks`, `docs`, `all`)
   with a `--dry-run` mode for impact analysis.

2. **Dry-run impact analysis** completed:

| Stage | Files affected | Substitutions |
|---|--:|--:|
| `core` (cpp/src, cpp/include/pls4all, cpp/tests, cpp/cli) | 293 | **14083** |
| `bindings` (Python, R, JS, Julia, MATLAB, …) | ~150+ | ~16000 |
| `parity` + `benchmarks` + `docs` | ~hundreds | ~17600 |
| **Total** | ~600 files | **~18500 token replacements** |

3. **Pre-rename tag landed**: `abi-p4a-final/v0.97.3-pre-merge` (pushed to
   origin) so the historical `n4m_*` symbol surface stays recoverable.

## What is NOT done (why M5 is deferred)

A live attempt at executing `--stage core` exposed a **deeper-than-mechanical
issue with the public-header layout** that the M5 rename alone cannot solve:

1. **pls4all's renamed `p4a.h`** (now declaring `n4m_*` symbols) lives at
   `cpp/include/pls4all/p4a.h` and contains the PLS-side ABI surface.
2. **Donor's `n4m.h`** at `cpp/include/n4m/n4m.h` is the umbrella header
   declaring the non-PLS surface (preprocessing, augmentation, splitter,
   filter, utility).
3. After `pls4all/p4a.h → n4m/n4m.h` substitution in pls4all source
   includes, those sources end up including the donor's `n4m.h` —
   which does NOT declare the PLS symbols they need. Build fails with
   unresolved `n4m_simpls_fit` etc.

The fix is **not** a simple sed: it needs the donor's `n4m.h` and pls4all's
renamed `p4a.h` content to be **merged into one umbrella header** without
duplicating the common-core declarations (context, status, matrix_view,
backend codes — present in both). That merge is M6's job
(per `MERGE_ROADMAP.md` row M6: "split into per-category headers").

So the correct execution sequence is **M5 ∘ M6**: the rename runs WHILE
the header layout is being collapsed; doing M5 alone strands the build in
an inconsistent state.

## Plan to actually execute M5 (in a focused session paired with M6)

1. **First, run M6 header reorganisation**:
   - Move pls4all's `cpp/include/pls4all/p4a.h` content into split files
     at `cpp/include/n4m/{n4m.h umbrella, n4m_pls.h, n4m_selection.h, …}`.
   - Strip duplicate common-core declarations.
   - The donor's existing `cpp/include/n4m/n4m.h` becomes the umbrella that
     `#include`s the new per-category headers.

2. **Then run M5 (script)**:
   ```bash
   python scripts/migrate_p4a_to_n4m.py --apply --stage core
   # Build + test gate
   cmake --build --preset dev-release && \
   ./build/dev-release/cpp/tests/n4m_tests
   # Should now report 265/265 with n4m_* symbols
   ```

3. **Then run M5 on the binding stages**, one at a time, gated by the
   binding's own smoke tests:
   ```bash
   python scripts/migrate_p4a_to_n4m.py --apply --stage bindings  # 16K subs
   python scripts/migrate_p4a_to_n4m.py --apply --stage parity
   python scripts/migrate_p4a_to_n4m.py --apply --stage benchmarks
   python scripts/migrate_p4a_to_n4m.py --apply --stage docs
   ```

4. **Symbol-table normalisation** (Linux versioned ELF `@@N4M_1`):
   - Update `cpp/abi/expected_symbols_linux.txt` to versioned-symbol
     format.
   - Update visibility script (`pls4all.ld` → `n4m.ld`, the version-script
     that grants visibility).
   - Adjust macOS `-exported_symbols_list` and Windows `.def` if needed.

## Why deferred to a focused session

- **~18500 token replacements across ~600 files**: a single failed
  intermediate state breaks 265 unit tests, all bindings, all parity
  fixtures, and the dashboard. The rollback surface is the entire repo.
- **Header layout dependency**: M5 cannot land cleanly without the M6
  header split (see above).
- **Per-binding test gates**: each binding (Python/R/MATLAB/JS/Julia/Go/Rust/.NET/Ruby/Lua/Nim/JNI) has its own smoke test that needs to run after the rename. The session driver has to manually exercise each binding's local test runner.

## Recovery commands if a future attempt corrupts the tree

```bash
# Hard reset to the pre-rename state:
git reset --hard abi-p4a-final/v0.97.3-pre-merge

# Or rebuild from M4 scaffold:
git checkout 9472dd9 -- cpp/ CMakeLists.txt
```

## Caveats noted by Codex on M4 review (carried into M5 planning)

- M4/EXTRACTION_RECIPE.md doesn't yet make "M4 partial" explicit at the
  top — should be updated to reflect that the actual extraction lands
  AFTER M5 rename so the new files start life with `n4m_*` naming, not
  `n4m_*`.
- `SPLIT_PLAN.md` says `approximate_press → dispatch.cpp` while we placed
  it at `models/core/approximate_press.cpp` — should reconcile.
- After M5 rename, M4 phase-1 helper extractions need to use `n4m::core`
  and `n4m::detail` namespaces, not `n4m::core`.

These caveats are written into the M5 plan above; they apply to the
focused session that executes both M5 and M6.

## What this commit lands

| File | Purpose |
|------|---------|
| `scripts/migrate_p4a_to_n4m.py` | Stdlib-only mechanical rename script (--dry-run + --apply with per-stage selection). Ready to run after M6 header split lands. |
| `docs/merge/M5/STATUS.md` | This status report (off-repo working notes). |
| Tag `abi-p4a-final/v0.97.3-pre-merge` → `9472dd9` (already pushed to origin) | Historical pre-rename baseline, lets future attempts roll back cleanly. |
