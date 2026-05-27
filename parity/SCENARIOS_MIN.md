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

## The gate

`make parity [METHOD=<id>]` runs `parity/comparator/run.py`, which reuses the
dashboard's verdict classification and writes `parity/results/latest.json` (a
per-method reference/binding verdict histogram + divergence δ + totals). It
exits non-zero on any `error` verdict; `divergent` verdicts are recorded (often
expected, allowlisted per method) and reported, not failed.

## Deferred (full Phase C)

`parity/environments/<env_id>/` Docker recipes, `parity-env-build.yml`,
`parity/scenarios/<method>.yaml`, `parity/snapshots/current/`, the standalone
`parity/comparator/` numeric comparator against migrated snapshots,
`self_consistency.py` for paper-only methods, and the `workflow_dispatch`
parity CI. Tracked in `docs/REFACTOR_PLAN.md` Phase C.
