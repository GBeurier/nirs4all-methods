# Status — refactor state on `main`

> Snapshot: 2026-05-23 · after Phase A landing.
> Founding document: [`docs/REFACTOR_PLAN.md`](docs/REFACTOR_PLAN.md).

Use this file as the **first read** when picking up the refactor. It
summarises what's done, what's in-flight, and what the next concrete
work item is.

---

## TL;DR

- **Phase A — Rename cleanup**: ✅ **landed on `main`** (5 commits).
  See [`roadmap/phase-A-rename-cleanup.md`](roadmap/phase-A-rename-cleanup.md).
- **Build**: ✅ `cmake --preset dev-debug && cmake --build` green; `ctest` green
  (`libn4m.so.1.9.0`, `libn4m_static.a`, `n4m_cli`, `n4m_tests` produced;
  266 doctest cases pass).
- **Phases B–F**: not started.

## Where Phase A landed

```
75df87a Baseline: stage pre-Phase-A work (n4m rename in-flight + REFACTOR_PLAN)
252248f A1: token-level p4a_*/pls4all_* → n4m_*/n4m_* sweep
f993b91 Phase A2-A13: structural renames, single c_api surface, unified CMake
ba05aac Phase A14-A26: docs, templates, dev shell, doctor + bootstrap
4762df8 A27: final audit — restore PLS umbrella, build green, snapshot refreshed
```

Pushed to `origin/main`.

## Repo shape now

```
cpp/
  include/n4m/                  # Public C ABI: n4m.h (umbrella) +
                                # pls.h (PLS-domain — restored in A27)
                                # + 13 per-category headers
  src/core/                     # C++17 numerical core, pls + donor sides
  src/c_api/                    # Single c_api/ surface (32 .cpp files,
                                # ex-c_api_n4m/ merged in)
  cli/n4m_cli.cpp               # ex-cli_n4m/
  tests/                        # ex-tests_n4m/
  abi/expected_symbols_*.txt    # ex-abi_n4m/, regenerated post-A27

bindings/
  python/src/n4m/               # full n4m Python (ex-python_n4m/)
  python/src/pls4all/           # mature slim subset (kept)
  r/n4m/                        # ex-r/pls4all/, fully renamed
  matlab/  octave/  js/  julia/  jni/  android/
  _archive/{go,rust,dotnet,lua,nim,ruby}/   # frozen PoCs

.devcontainer/                  # Phase A20 dev shell (Dockerfile +
                                # compose + post-create)
.github/ISSUE_TEMPLATE/         # 8 issue forms + config.yml
.github/PULL_REQUEST_TEMPLATE/  # 7 PR templates + generic fallback
.github/codespaces/             # Codespaces mirror

docs/REFACTOR_PLAN.md           # founding doc
roadmap/phase-A-rename-cleanup.md  # this phase's record
```

## Verifying current state

```bash
# Fresh build
rm -rf build/dev-debug && cmake --preset dev-debug
cmake --build --preset dev-debug --parallel
ctest --preset dev-debug --output-on-failure

# CLI smoke
./build/dev-debug/cpp/cli/n4m_cli --version
./build/dev-debug/cpp/cli/n4m_cli --selfcheck

# Tool resolution
make doctor

# No legacy refs in active code
grep -rE '\b(p4a_|P4A_)\b' cpp/ catalog/ parity/ bindings/ \
  --include='*.h' --include='*.hpp' --include='*.cpp' --include='*.c' \
  --include='*.py' --include='*.R' --include='*.r' --include='*.m' \
  --include='*.yaml' 2>/dev/null \
  | grep -vE "_archive/|merge/|reviews/|migrate_p4a_to_n4m" | head
# → empty
```

## Known structural debts (Phase A judgment calls, follow up in B/F)

1. **`cpp/include/n4m/pls.h` recovered as a single file** (1844 LOC).
   The donor-side `n4m.h` umbrella had pruned the PLS-domain surface
   ("Phase 3 trim") but `cpp/src/core/model.cpp` still depends on it.
   Phase A27 restored the deleted `p4a.h` (post-rename) under
   `cpp/include/n4m/pls.h` and auto-includes it from `n4m.h`. Phase B
   per `REFACTOR_PLAN.md` §2.1 should re-split into per-category
   headers: `n4m_pls.h`, `n4m_preprocessing.h`, `n4m_selection.h`, …
   ⇒ track when splitting the catalog.

2. **R binding still under `bindings/r/n4m/`** with the full mature
   surface (300+ symbols). Phase F-bootstrap will rebuild the R
   binding from per-language Jinja templates (§2.11). Until then,
   the renamed-as-is form is the source of truth.

3. **Two `pls4all` carve-outs intentionally preserved**:
   - `bindings/python/src/pls4all/` — slim PLS-only subset package
     (Phase E will formalise as `bindings/python/packaging/pls4all/`)
   - `github.com/GBeurier/pls4all` URLs — until the GitHub repo
     itself is renamed, all URL references stay
   - `catalog/subsets/pls4all.yaml` — the manifest for that subset
   - `bindings/_archive/*` — frozen PoC bindings (not in scope)

4. **Stale catalog**: `benchmarks/parity_timing/registry.py` (~10k
   LOC) is the legacy method catalog; `catalog/methods.yaml` +
   `methods.discovered.yaml` are duplicates. Phase B replaces all
   of these with `catalog/methods/<id>.yaml` (one YAML per method).

5. **Snapshot version-tag handling**: `cpp/abi/expected_symbols_*.txt`
   stores bare symbol names; the linker version script (adopted in
   A3 from `c_api_n4m/`) now attaches `@@N4M_1` to every export.
   `abi-check.yml` was updated to strip the suffix before diffing.
   Watch for this when regenerating the snapshot — use:
   ```bash
   nm -D --defined-only libn4m.so.* \
     | awk '{print $3}' | grep "^n4m_\|^N4M_" \
     | sed 's/@@.*//' | sort -u
   ```

## Next concrete work item

**Phase B — Unified catalog + auto-generation**
(see `docs/REFACTOR_PLAN.md` §3 "Phase B" — 10 tasks B1–B10).

| Step | Action |
|------|--------|
| B1 | Write the YAML schema for a method file (`catalog/schema/method.json`); document in `catalog/README.md` |
| B2 | Implement `catalog/scripts/split_legacy_methods.py` — one-shot script that splits `catalog/methods.yaml` into `catalog/methods/<id>.yaml` (one file per method) |
| B3 | Migrate `benchmarks/parity_timing/registry.py` data into the per-method YAMLs (data stays in YAML, logic stays in Python but reads the YAMLs) |
| B4 | Migrate `parity/tolerances.md` → `parity.tolerances` field in each YAML |
| B5 | Implement `catalog/scripts/validate.py` — all the structural invariants listed in §3 Phase B B5 |
| B6 | Implement `catalog/scripts/render_api.py` with per-language Jinja templates from day one |
| B7 | Add CI workflow `catalog-validate.yml` — hard blocker on PR merge |
| B8 | Implement `catalog/scripts/migrate_schema.py` (catalog_version bump tool) |
| B9 | Delete `catalog/methods.yaml`, `methods.discovered.yaml`, `benchmarks/parity_timing/registry.py` once B2–B5 are stable |
| B10 | Delete `roadmap/phase-*.md` files that describe implemented methods (replaced by the catalog YAMLs) |

**Phase B exit criterion**: `python catalog/scripts/validate.py`
returns 0 errors, `render_api.py` reproduces existing APIs identically,
`registry.py` is gone.

## Subsequent phases (not started)

- **Phase C — Parity infra rebuild** (scenario-based) — 20 tasks
- **Phase D — Benchmark + SPA dashboard** — 15 tasks
- **Phase E — Sub-packages** (packaging-only) — 11 tasks
- **Phase F — Binding scaling** — F-prep (7) + F-bootstrap (12) + F-rollout (8)

All detailed under `docs/REFACTOR_PLAN.md` §3.

## How to pick up later

1. **Read this file** (Status.md) and
   [`roadmap/phase-A-rename-cleanup.md`](roadmap/phase-A-rename-cleanup.md).
2. Verify the build still works: `cmake --preset dev-debug && cmake --build`.
3. Open [`docs/REFACTOR_PLAN.md`](docs/REFACTOR_PLAN.md) § "Phase B"
   for the next batch.
4. Optionally enter the dev shell: `make bootstrap` (or
   `make shell` if Docker is already running).

## Workflow conventions

- **One PR per Phase sub-batch** (we did Phase A as 5 commits on `main`;
  for B+ either continue on `main` while in dev, or split into PRs).
- **Conventional Commits** with scope (`feat(catalog):`, `fix(parity):`).
- **Codex review** of each diff (see `CONTRIBUTING.md` for the loop).
- **Issue → PR templates** under `.github/` are the gating UX for
  external contributors (closed-lib model — see `CONTRIBUTING.md`).
