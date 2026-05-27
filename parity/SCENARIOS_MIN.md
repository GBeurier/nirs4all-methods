# Parity scenarios — minimal model (Phase C, current)

The full scenario/snapshot rebuild in the refactor plan (Phase C, C1-C20:
per-scenario YAML, Docker-per-external-library environments, GHCR images,
`env_lock_hash` provenance, snapshot migration) is **deferred**. The parity
*signal* it targets already exists, produced by infra that runs the external
libraries in local environments. Phase C (minimal) puts a unified **gate +
summary** around that existing infra rather than rebuilding it.

## What a "scenario" is today

A parity scenario is the tuple

```
(method, n, p, seed, backend)
```

- **method** — a dashboard method id (PLS family or donor op).
- **n × p** — sample/feature size (the dashboard cell sizes, e.g. 250×50, 50×250).
- **seed** — deterministic input seed (`1_234_567_890` base; splitmix64 for
  donor inputs so the C++ and binding tiers see identical bits).
- **backend** — the implementation: a C++ build tier (`cpp` @ native/blas/omp/
  blas+omp), a binding tier (`python_tier1/2`, R, MATLAB, …), or an external
  reference (`ref_python_scikit_learn`, `ref_python_nirs4all`, `ref_r_pls`, …).

Scenarios are materialised as rows in the cross-binding result CSVs
(`benchmarks/cross_binding/results/`):

- `full_matrix.csv` + `dashboard_refresh_*.csv` (PLS family, via
  `benchmarks/cross_binding/orchestrator.py` + `benchmarks/parity_timing/registry.py`).
- the donor refreshes (`dashboard_refresh_*_n4m_fixtures.csv`,
  `*_donor_timing.csv`, `*_donor_binding.csv`, `*_donor_reference.csv`) via
  `donor_ops.py` + `bench_donor_{timing,binding_timing,reference_timing}.py`.

## The two parity relations

- **Reference parity** — n4m output vs the method's canonical **external
  library** reference (C6/C10 in the plan). External libs run in local venvs /
  Rscript / Octave. For donors, nirs4all is the universal reference.
- **Binding parity** — every n4m binding tier vs the **C++ core** (C11
  inter-binding parity). Donor raw↔idiomatic + raw≡C++ are gated bit-exact by
  `benchmarks/cross_binding/tests/test_donor_binding_specs.py`.

## The three gates (Phase C-min)

1. **Binding/reference summary** — `make parity [METHOD=<id>]` runs
   `parity/comparator/run.py`, which reuses the dashboard's verdict
   classification and writes `parity/results/latest.json` (per-method
   reference/binding verdict histogram + divergence δ + totals). Exits non-zero
   on any `error` verdict; `divergent` verdicts are recorded (often expected,
   allowlisted per method) and reported, not failed.

2. **Self-consistency** — `make parity-paper-only [METHOD=<id>]` runs
   `parity/comparator/self_consistency.py` for methods with **no external
   reference** (AOM-PLS, POP-PLS). Their correctness gate is determinism (same
   seed → bit-identical output) + structural invariants (finite, non-empty,
   expected element count). The spec set is hand-curated and catalog-free — a
   temporary bridge until the catalog gains real `self_consistency` blocks.

3. **Golden snapshots** — `make snapshot [METHOD=<id>]` (`--write`) regenerates
   `parity/snapshots/current/<method>/<scenario>.json`, the scenario layout that
   full Phase C (C4) will migrate the 202 legacy fixtures into. These are
   `source_kind: n4m_self` (a regression gate on n4m's OWN output, with seed /
   dims / ABI / lib-sha256 provenance), NOT external validation.
   `parity/generators/run.py --check` is the read-only golden gate.

Gates 2 and 3 also run in CI (`cross-binding-parity.yml`, which builds libn4m).

## Reference environments (Phase C-min)

`parity/environments/` ships the **recipe schema** (`README.md`) + one **real
pinned recipe** (`env-py-sklearn-1.4/`, the sklearn pins the fixture generator
already uses) + a local-convenience conda env (`dev/environment.yml`). The
per-(framework, version) matrix and the GHCR build CI are deferred.

## Deferred (full Phase C)

`parity-env-build.yml` (GHCR image build/publish), the full per-framework
environment matrix, `parity/scenarios/<method>.yaml` (catalog-coupled), the
one-shot migration of the 202 `parity/fixtures/*.json`, the in-container
external reference generators (`parity/generators/{python,r,octave}/`), the
numeric comparator against migrated external snapshots, and the
`workflow_dispatch` parity CI (`parity-snapshots.yml` / `parity-run.yml` /
`parity-paper-only.yml`). Tracked in `docs/REFACTOR_PLAN.md` Phase C.
