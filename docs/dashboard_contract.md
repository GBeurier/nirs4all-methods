# Dashboard data contract (D-min)

The benchmark/parity dashboard is driven by **one canonical JSON payload**
emitted by [`docs/_extras/build_landing.py`](_extras/build_landing.py)
(`build_payload`) and written to `docs/_static/bench-data.json`. The shape of
that payload is the **contract** — validated by
[`docs/dashboard.schema.json`](dashboard.schema.json) and the
`test_dashboard_contract` test. The current Sphinx/template dashboard and the
future Svelte SPA (Phase D-SPA) both consume this same contract.

## Why a contract (D-min vs D-SPA)

Phase D in the refactor plan proposed a full Svelte/Vite single-page app. Per
the review, that is split:

- **D-min (this):** a stable, schema-validated `dashboard.json` contract with
  **automated per-method score cards** (reference parity, binding parity,
  divergence). This is the load-bearing part — it makes the scientific signal
  (does n4m match its external reference? do the bindings match the C++ core?
  by how much?) visible and maintainable in the existing flow.
- **D-SPA (deferred):** the interactive Svelte app. Optional polish; consumes
  this contract unchanged. Full scope below.

## Payload surface (stable keys)

| Key | Meaning |
|-----|---------|
| `columns` | The implementation columns (C++ tiers, bindings, external references), with `id`/`label`/`group`/`lang`/`kind`. |
| `rows` | One per `(algo, n, p, threads)`. Each has `cells` keyed by column id; a cell carries `ms` (timing), `ok`, and the parity verdicts (`parity`, `reference_parity`, `binding_parity`) + `divergence` (numeric δ) + `divergence_basis`. |
| `method_scores` | **The D-min score card.** Per method, aggregated across its cells. |
| `stats` | Global counts (algos, backends, rows, cells, exact). |

## `method_scores[<method>]`

```jsonc
{
  "reference":  { "exact": 12, "divergent": 4, "not_available": 8, ... },  // n4m vs canonical external reference (C6/C10/C11 verdicts)
  "binding":    { "exact": 20, ... },                                       // each binding tier vs the C++ core
  "divergence": {
    "reference": { "max": 20.65, "median": 0.0, "n": 16 },                  // |δ| of reference-gate cells
    "binding":   { "max": 0.0,   "median": 0.0, "n": 40 }                   // |δ| of binding-gate cells
  }
}
```

- **reference / binding** are verdict histograms (counts by
  `exact / divergent / not_available / not_run / drift / error`).
- **divergence** is split by gate **basis**: `reference` (n4m vs external
  library, e.g. live nirs4all) vs `binding` (binding tier vs the C++ core).

## Source of the verdicts

The verdicts come from the parity gate (**Phase C**): the cross-binding
orchestrator (`benchmarks/cross_binding/orchestrator.py`) for the PLS family
and the donor pipelines (`donor_ops.py` + `bench_donor_{binding,reference}_timing.py`)
for the donor methods, plus `parity/comparator/run.py`'s
`parity/results/latest.json` summary. `method_scores` is a presentation-side
aggregation of those per-cell verdicts — it never invents data.

## Regenerating

```bash
python docs/_extras/build_landing.py \
  --results benchmarks/cross_binding/results \
  --out docs/_static/bench-data.json
```

Then `test_dashboard_contract` validates the result against
`docs/dashboard.schema.json`.

## D-SPA (deferred): scope

D-SPA is the interactive single-page app from Phase D of the refactor plan
(`docs/REFACTOR_PLAN.md`, D1–D15). It is **deferred, not cancelled** — D-min
deliberately delivered the load-bearing half (the schema-validated contract +
automated score cards) so the SPA becomes pure front-end work that consumes
`bench-data.json`/`dashboard.json` **unchanged**.

**Planned views** (each reads only the contract):

- **Matrix** — methods × implementation columns, filterable, with parity badges
  (green/yellow/red + a distinct `paper_only` badge from the self-consistency
  gate).
- **Method drill-down** — per method: timing curves vs size (linear + log),
  the multi-reference parity table, snapshot provenance.
- **Drift** — parity verdicts over n versions, lazy-loaded from archived
  `snapshots-YYYY-MM.tar.zst` assets (sparse; only manually-archived points).
- **Catalog** — browse by category / status, fuzzy search on name/symbol.
- **Host** — surfaces the active host card; cross-host comparison warning-gated.
- A **stale-data badge** when `generated_at` lags the latest method-touching
  commit.

**Currently stubbed:** the `dashboard/` Svelte+Vite app does not exist; the
`make dashboard-data` / `dashboard-serve` / `dashboard-build` targets print a
"not yet bootstrapped (Phase D)" message and exit 0. Publication
(`dashboard-publish.yml` → `gh-pages`) and the self-hosted timing workflow
(`benchmarks.yml`) are likewise unbuilt.

**Trigger to build:** when the static Sphinx/landing dashboard becomes the
limiting factor for consuming the parity/timing signal (filtering, history,
multi-host). Until then the static landing page + the `method_scores` cards
cover the need. Because the contract is fixed and tested, building D-SPA later
is additive and does not touch the data pipeline.

> Indexed in [`DEFERRALS.md`](../DEFERRALS.md).
