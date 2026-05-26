# Status — refactor state on `codex/phase-b-catalog`

> Snapshot: 2026-05-25 · after Phase B catalog foundation + threading audit.
> Founding document: [`docs/REFACTOR_PLAN.md`](docs/REFACTOR_PLAN.md).

Use this file as the **first read** when picking up the refactor. It
summarises what's done, what's in-flight, and what the next concrete
work item is.

---

## TL;DR

- **Phase A — Rename cleanup**: ✅ **landed on `main`** (5 commits).
  See [`roadmap/phase-A-rename-cleanup.md`](roadmap/phase-A-rename-cleanup.md).
- **Greg audit — GIL / multithreading / sklearn jobs**: ✅ Python bindings use
  `ctypes.CDLL` (GIL released during native calls), contexts are per-thread,
  and benchmark sklearn-style references now honor `BENCH_SKLEARN_N_JOBS`
  with `BENCH_THREADS` fallback.
- **Phase B — Catalog foundation**: ✅ on branch `codex/phase-b-catalog`.
  Split method files, schema, validator, render/migration scaffolding, and
  catalog CI are in place. ABI-symbol reconciliation is still outstanding.
- **Build**: ✅ `cmake --preset dev-debug && cmake --build` green; `ctest` green
  (`libn4m.so.1.9.0`, `libn4m_static.a`, `n4m_cli`, `n4m_tests` produced;
  266 doctest cases pass).
- **Phases C–F**: not started.

## Where Phase A landed

```
75df87a Baseline: stage pre-Phase-A work (n4m rename in-flight + REFACTOR_PLAN)
252248f A1: token-level p4a_*/pls4all_* → n4m_*/n4m_* sweep
f993b91 Phase A2-A13: structural renames, single c_api surface, unified CMake
ba05aac Phase A14-A26: docs, templates, dev shell, doctor + bootstrap
4762df8 A27: final audit — restore PLS umbrella, build green, snapshot refreshed
```

Pushed to `origin/main`.

## Current branch commits

```
70271b1 fix(benchmarks): honor sklearn thread budget
a8bd3c9 feat(catalog): split method catalog foundation
```

Pushed to `origin/codex/phase-b-catalog`.

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

# Catalog foundation checks
python catalog/scripts/selftest.py
python catalog/scripts/split_legacy_methods.py --check
python catalog/scripts/validate.py
python catalog/scripts/validate_catalog.py

# Full registry parity sweep, sequential cells, 8 native threads per cell
BENCH_SKLEARN_N_JOBS=8 python benchmarks/cross_binding/orchestrator.py \
  --algorithms all --registry-cells --threads 8 --workers 1 \
  --libn4m-build blas-omp --n-runs 2 \
  --canonical-pls4all-only --reference-backends registry \
  --timeout 180 --out-csv /tmp/n4m_full_parity.csv --force \
  --flush-each-cell

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

4. **Catalog transition**: `catalog/methods/<id>.yaml` now exists for
   all 160 legacy method rows and new scripts prefer the split files.
   During the transition, `catalog/methods.yaml` remains the editable
   source for split regeneration and `benchmarks/parity_timing/registry.py`
   still drives full parity/timing.

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

**Finish Phase B — Unified catalog + auto-generation**
(see `docs/REFACTOR_PLAN.md` §3 "Phase B" — 10 tasks B1–B10).

| Step | State | Action |
|------|-------|--------|
| B1 | ✅ | `catalog/schema/method.json` + `catalog/README.md` added |
| B2 | ✅ | `split_legacy_methods.py` generates/prunes 160 split files |
| B3 | ⏳ | Migrate registry data (canonical references, alternates, params) into YAML while keeping Python logic as a reader |
| B4 | ⏳ | Migrate `parity/tolerances.md` rows into `parity.tolerances` per method |
| B5 | ⚠️ | `validate.py` exists; strict ABI is advisory until symbol guesses are reconciled |
| B6 | ⚠️ | `render_api.py` has per-language Jinja templates but currently renders metadata only under `build/catalog/rendered_api/` |
| B7 | ✅ | `.github/workflows/catalog-validate.yml` added |
| B8 | ✅ | `migrate_schema.py` no-op v1 tool added |
| B9 | ⏳ | Delete legacy YAMLs and `registry.py` after B3–B5 are stable |
| B10 | ⏳ | Delete implemented-method roadmap files after YAML owns those records |

**Phase B exit criterion**: `python catalog/scripts/validate.py`
returns 0 errors, `render_api.py` reproduces existing APIs identically,
`registry.py` is gone.

Current validation status: `validate.py` returns 0 with 419 ABI warnings from
auto-discovered symbol guesses. Do not enable `--strict-abi` in CI until those
symbols are reconciled against `cpp/abi/expected_symbols_*.txt`.

## Full parity run — 2026-05-25

Command run:

```bash
BENCH_SKLEARN_N_JOBS=8 python benchmarks/cross_binding/orchestrator.py \
  --algorithms all --registry-cells --threads 8 --workers 1 \
  --libn4m-build blas-omp --n-runs 2 \
  --canonical-pls4all-only --reference-backends registry \
  --timeout 180 --out-csv /tmp/n4m_full_parity.csv --force \
  --flush-each-cell
```

Result CSV: `/tmp/n4m_full_parity.csv`.

- 153 rows planned/run.
- 151 subprocess successes; 2 subprocess failures:
  `pls/ref_r_mixomics`, `opls/ref_r_ropls`.
- 0 binding-parity failures.
- 145 reference-parity passes; 3 reference-parity failures:
  `pcr/ref_r_pls`, `opls/registry_pls4all` (missing oracle because
  `ref_r_ropls` failed), `continuum_regression/ref_r_jico`.

## Clean parity matrix + harness fixes — 2026-05-25 (uncommitted)

Re-ran the registry parity matrix at `--n-runs 40` (robust timing) and
root-caused/fixed the failures above. Result written to the canonical
`benchmarks/cross_binding/results/full_matrix.csv` (459 rows = 73 methods
× threads {1,3,10}, blas-omp). Each thread count was run to its own CSV
with non-oversubscribing workers (12/8/2) and concatenated.

**Outcome (all three thread counts identical):**
- 153/153 cells run, **0 subprocess failures** (iriv no longer times out).
- **73/73 binding-parity pass.**
- 150/152 reference-parity pass. The only two failures are external-library
  convention differences, not pls4all/harness bugs (the pls4all binding
  matches its canonical oracle in both):
  - `pcr/ref_r_pls` — R `pls` PCR centering/scaling convention vs sklearn.
  - `continuum_regression/ref_r_jico` — `JICO` convention vs the
    `stone_brooks_1990` oracle.

**Fixes applied (working tree, not committed — for Codex review):**
1. `orchestrator.py` resolves `Rscript`/`octave` against the subprocess
   `PATH` (which prepends `N4M_R_ENV`), so the configured R env wins over
   whatever `Rscript` sits on the orchestrator's own PATH. Previously the
   `pls4all_r` env was never used and `mixOmics`/`ropls`/`JICO`/… cells
   failed with "no package called …".
2. **Parity prediction snapshotted at `seed_base`** instead of the last
   adaptive timed run. The timing loop regenerates the dataset per run
   (`seed_base + i`), so the old "last run" snapshot keyed the prediction
   (and oracle) by a backend-dependent run count — fast bindings and slow
   references then compared predictions on *different* datasets, producing
   "reference oracle missing" and spurious binding-parity failures at
   `--n-runs 40`. Now `prediction_seed == seed_base` for every backend.
   Touches `scripts/_common.py`, `scripts/_npy.R`,
   `scripts/bench_registry_reference.py`, and
   `orchestrator.py::record_prediction_seed`. This is why the maintainer's
   `--n-runs 2` run mostly worked (uniform run counts) while `--n-runs 40`
   exposed it. Regenerated reference oracles accompany the change.
3. `benchmarks/parity_timing/registry.py::_iriv_indices_numpy` —
   `_generate_binary_matrix` used unbounded rejection sampling
   (`while True`) requiring every sub-model row to select >1 variable;
   `P(all rows)` → 0 once a round drops to very few variables, so iriv hung
   (>300s) on datasets that reached that regime. Now bounded (10000
   attempts) + deterministic row repair; `seed_base` behaviour unchanged.

**Also added (timing):** `dashboard_refresh_2026_05_25_builds.csv` — the
pls4all binding-tier timing across the four CPU builds
(`dev-release`/`blas-on`/`omp-on`/`blas-omp`) × threads {1,3,10}.
`registry_pls4all` is 73/73 clean on every build; the only failures are
pre-existing binding-tier surface gaps (`on_pls` R tiers, `python_tier2`
for a few methods), unrelated to the build dimension.

Dashboards re-rendered: `docs/_static/bench-data.json`,
`docs/benchmarks/cross_binding{,_threads}.md`, and the standalone
`build/cross_binding_dashboard/index.html`.

### Binding-tier gap fix (Codex) — 2026-05-25

The `--only-pls4all` build sweep surfaced hard failures in two methods'
secondary tiers (they pass in the canonical `registry_pls4all` binding).
Fixed via Codex:
- `cars_select` × python_tier2: the sklearn `CARSSelector` legitimately
  lacks `top_k`/enpls ranking semantics → now emits a clean `not_available`
  skip (only for the exact `top_k`-only unsupported case).
- `on_pls` (multi-block `block_reconstruction_0`): wired the export into
  `pls4all.sklearn.on_pls` so python_tier2 now passes (`exact`); the 4 R
  tiers emit clean `not_available` (no idiomatic block-reconstruction).
- `docs/_extras/build_landing.py::parity_code` now treats the canonical
  `not_available` reason prefix as not-available (not `error`), and
  classifies timing-only cells (ran, no parity verdict) as `not_run`
  instead of a false `drift`.
Verified: 0 cells misclassified as error; canonical path unchanged.
Files: `scripts/bench_python_tier2.py`, `scripts/bench_r_{tier1,tier2,pls_compat,mdatools_compat}.R`,
`bindings/python/src/pls4all/sklearn/_diagnostics.py`, `docs/_extras/build_landing.py`.
Delta: `dashboard_refresh_2026_05_25_tiergap.csv`.

### Size-scaling timing sweep — 2026-05-26 (complete, Codex-validated)

`dashboard_refresh_2026_05_25_sizes.csv`: the full 11-size matrix —
**73/73 methods × 11 sizes (100–10000 samples × 50–2500 features) × 4 CPU
builds × {registry_pls4all, cpp} = 6424 cells**, threads=1, `--n-runs 40`,
`--timeout 900`. Ran to completion including the final parity pass (~14.5 h
on 12 workers).

- **6068 cells completed; binding parity 6068/6068 (100%) within `1e-6`**
  (Codex independently recomputed the `.npy` predictions: max diff 4.6e-8,
  5330 zero-diff + 738 within-tolerance, 0 mismatches). **Scope of this
  parity:** `registry_pls4all` and `cpp` both route through
  `MethodSpec.pls4all_fn`, so this validates **build/backend consistency**
  across the 4 libn4m builds — NOT an independent C-ABI-vs-Python path.
- **356 failures** (binding parity empty): **340 timeouts** = legitimate
  "method doesn't scale to this size" (all `subprocess_s=900`; the same
  methods complete at smaller sizes; p-driven), concentrated in interval/
  wrapper selectors (iriv 56, emcuve 48, rep 40, uve 32, bipls/stability/
  bve 24 each) + kernel/GP/PRESS/PCR at the largest sizes. **16 non-timeout
  errors** = a real isolated bug (see below).

**Two follow-up findings (Codex, not yet fixed):**
1. `vip_spa_select` fails at p=50 with many samples (2500×50, 10000×50):
   `ValueError: n_features_to_select has to be either an int in {1,…,n}`.
   The registry calls `auswahl.VIP_SPA(n_features_to_select=top_k=6)`; VIP
   pre-filtering leaves <6 candidate features at those cells, making 6
   invalid. Param-derivation bug, isolated to `vip_spa_select`.
2. Asymmetry: `gpr_pls 10000×500` completed on all 4 `cpp` builds but timed
   out on all 4 `registry_pls4all` builds — the only partial-OK group.

The earlier disk constraint (shared 1 TB disk was ~99% full) is resolved —
≈700 GB free now — so the full sweep with `--n-runs 40` ran with no pruning
and the parity pass completed. The `--timeout 900` (vs 300) was the key
change: it lets ~200 s/run p=2500 cells finish (whole-subprocess timeout =
warmup + probe + reps) instead of being lost as false timeouts.

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
   and continue at B3/B4/B5 strict-ABI reconciliation.
4. Optionally enter the dev shell: `make bootstrap` (or
   `make shell` if Docker is already running).

## Workflow conventions

- **One PR per Phase sub-batch** (we did Phase A as 5 commits on `main`;
  for B+ either continue on `main` while in dev, or split into PRs).
- **Conventional Commits** with scope (`feat(catalog):`, `fix(parity):`).
- **Codex review** of each diff (see `CONTRIBUTING.md` for the loop).
- **Issue → PR templates** under `.github/` are the gating UX for
  external contributors (closed-lib model — see `CONTRIBUTING.md`).
