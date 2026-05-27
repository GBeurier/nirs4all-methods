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
- **D-SPA (deferred):** the interactive app (rich filtering, drift view,
  host panels, lazy-loaded history). Optional polish; consumes this contract
  unchanged.

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
