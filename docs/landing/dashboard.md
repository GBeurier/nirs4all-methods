# GitHub Pages dashboard (`pls4all.github.io`)

The project's public landing page —
[`https://gbeurier.github.io/pls4all/`](https://gbeurier.github.io/pls4all/)
— is a fully interactive dashboard built from the same canonical
benchmark CSV that drives this documentation's
[per-method tables](../methods/index.md).

This page is the user manual for that dashboard: what each column
means, how to filter, and how to relate a dashboard cell to the
documentation page for that algorithm.

## What you see at first load

A table with one row per `(algorithm, n, p, threads)` cell and one
column per backend. Each cell should show:

- **binding parity** for pls4all rows, compared with the native C++
  baseline;
- **reference parity** for every successful row, compared with the
  registry-declared method oracle; and
- a **median time** (ms or s), formatted compactly.

Hovering a cell reveals: the binding diff when relevant, the reference
RMSE-relative diff when available, the library version that produced it,
and the failure reason if any.

Above the table you'll find:

- **stats summary** — how many algorithms / backends / sizes / threads
  are represented; how many cells are exact.
- **host card** — the OS, CPU model, RAM, BLAS used to produce the
  CSV. This is captured at build time and embedded in the JSON
  payload.
- **generated-at timestamp** — the mtime of the source CSV in UTC.

## Columns

Columns are grouped into language bands and tier groups:

| Group key | Display label | What's in it |
|---|---|---|
| `cpp`     | pls4all · C++ (libp4a)        | `pls4all.cpp.ref`, `pls4all.cpp.blas`, `pls4all.cpp.omp`, `pls4all.cpp.blas+omp`, `pls4all.cpp.cuda` |
| `python`  | pls4all · Python               | `pls4all.python` (tier-1 ctypes), `pls4all.sklearn` (tier-2 BaseEstimator), `pls4all.registry` (canonical entry point) |
| `r`       | pls4all · R                    | `pls4all.R` (tier-1 dispatcher), `pls4all.R.formula` (tier-2 formula + S3) |
| `matlab`  | pls4all · MATLAB/Octave        | `pls4all.matlab` (tier-1 MEX dispatcher), `pls4all.matlab.classdef` (tier-2 classdef) |
| `reference` | canonical reference          | Synthetic provenance column naming the oracle for each method. |
| `ext-py`  | external · Python              | `sklearn`, `ikpls`, registry-declared `ref.python_*` libs |
| `ext-r`   | external · R                   | `pls`, `mixOmics`, `ropls`, registry-declared `ref.r_*` libs |
| `ext-ml`  | external · MATLAB/Octave       | `plsregress`, registry-declared `ref.matlab_*` libs |

The C++ tier suffix (`ref`, `blas`, `omp`, `blas+omp`, `cuda`) maps
1:1 to the libp4a build flags. Current binding-parity calculations use
the one-thread native baseline selected by the orchestrator, currently
`cpp` from the `blas-omp` build when present. `cuda` only appears when
CUDA was available at build time.

## Filters and presets

The toolbar exposes the following filters:

- **Algorithm group** — Core, Sparse, Ensemble, Robust / weighted,
  Nonlinear / local, Multi-block, Calibration transfer, Classification,
  Missing data, Regularised, Selectors. Driven by the
  `ALGO_GROUP` map in `docs/_extras/build_landing.py`.
- **Language** — restrict to columns of one host language (C++,
  Python, R, MATLAB/Octave, …).
- **Threads** — pick a thread count (typically 1, 3, or 10).
- **Column preset** — one of:
  - `headline` — the four canonical pls4all columns plus the
    primary external in each language.
  - `api-parity` — the C++ native baseline plus the pls4all *mimicking*
    bindings (sklearn-style, R formula + S3, MATLAB classdef) lined up
    against the canonical externals users would otherwise reach for
    (`sklearn`, `ikpls`, `pls`, `mixOmics`, `ropls`, `plsregress`).
    Raw low-level bindings (ctypes, R dispatcher, MEX) are excluded —
    this is the apples-to-apples view.
  - `pls4all` — all pls4all bindings, no externals.
  - `cpp-tiers` — the five libp4a builds side-by-side.
  - `externals` — only the reference libraries.
  - `thread-sweep` — a tight subset suitable for comparing 1 vs 3 vs
    10 thread scaling.
  - `all` — every column present in the current CSV.

Presets resolve at build time against the columns actually present in
the data, so an empty preset is dropped automatically.

## Cell tooltips

Each cell exposes:

- `parity` — legacy alias kept for older renderers.
- `binding_parity` — pls4all binding/core consistency verdict, or not
  applicable for external libraries.
- `reference_parity` — method-oracle verdict.
- `ms` — float median wall-clock in milliseconds.
- `fmt` — pre-formatted time string (`1.97 ms`, `4.2 s`).
- `reason` — for non-`exact` cells, the captured failure / drift
  reason, trimmed to 200 chars. Examples:
  - `ModuleNotFoundError: No module named 'sgPLS'`
  - `R `pls::cppls 2.8.5` is Liland 2009 Canonical Powered PLS, NOT
    Indahl 2005 Powered-PLS (pls4all's algorithm)`
  - `parity diff 5.4e-02 vs tolerance 1e-6 — scaling convention drift`

The dashboard renders `reason` in a tooltip so the table itself stays
scannable.

## Known data issues from the May 2026 audit

- Some payload builders still merge `ref_*` rows into legacy external
  columns without propagating `ok`, `reason`, `parity` and
  `reference_parity` from the canonical reference row. This can show a
  timed reference cell as failed.
- Some filters still use the legacy `parity` alias, which represents
  binding parity in old runs. External libraries should be filtered by
  `reference_parity` instead.
- `--only-pls4all` CSVs do not schedule external oracle rows. Any
  "canonical reference predictions missing" note in that mode means
  reference parity was not run.
- The dashboard HTML embeds the payload at Sphinx build time. Running
  `build_landing.py` alone updates `docs/_static/bench-data.json`, but
  it does not update an already-built `docs/_build/html/index.html`.

## Relationship to this documentation

Each algorithm row in the dashboard links to its dedicated page in
[the methods catalogue](../methods/index.md). The per-method page
shows the same numbers plus:

- the algorithm's bibliographic source,
- its math principle,
- its parameters (with C ABI / Python / R / MATLAB defaults),
- per-binding code examples,
- a filtered benchmark table covering only that one method.

The dashboard is best for cross-method comparison; the per-method
page is best for understanding one algorithm in depth.

## Source layout

| File | Role |
|---|---|
| `docs/_templates/landing.html`      | The HTML / CSS / JS shell. Jinja2 substitutes `{{ bench_data_json }}` at sphinx-build time. |
| `docs/_extras/build_landing.py`     | Reads `benchmarks/cross_binding/results/full_matrix.csv`, applies the column-id and verdict conventions, and emits the compact JSON payload. |
| `docs/_static/bench-data.json`      | The committed fallback payload — used by CI / Read the Docs builds where the raw CSVs are not present. |
| `docs/conf.py:setup()`              | Picks between live CSV regeneration and the committed fallback, then injects the JSON into `html_context`. |

## Regenerating locally

```bash
# Build with the live CSV (regenerates docs/_static/bench-data.json):
sphinx-build -b html docs docs/_build/html

# Or just write the JSON without a full Sphinx build:
python docs/_extras/build_landing.py \
  --results benchmarks/cross_binding/results \
  --out docs/_static/bench-data.json
```

## Deployment paths

The same HTML output is published from **two** locations:

- **GitHub Pages** — `https://gbeurier.github.io/pls4all/` — built by
  `.github/workflows/docs.yml` on every push to `main`. This is the
  canonical landing page.
- **Read the Docs** — see [`readthedocs.md`](../dev/readthedocs.md)
  for the RTD project configuration. The RTD build uses the committed
  `docs/_static/bench-data.json` fallback so it does not need to
  install the benchmark dependencies.
