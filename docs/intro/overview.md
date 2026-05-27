# What is pls4all?

**pls4all** is a portable PLS / NIRS engine written in C++17, exposed
through a stable C ABI, and packaged behind thin first-class bindings
for the current target languages — Python, R, MATLAB / Octave, and
JavaScript / WebAssembly. Additional language bindings (Go, Rust, Julia,
Ruby, .NET, Lua, Nim, JNI, Android) exist as frozen proofs-of-concept
under `bindings/_archive/` and are revived on request.

It is built around a single claim: **the same numerical PLS result,
in every language**, with timings that match or beat each language's
established library.

## Why this project exists

PLS (Partial Least Squares) and the wider chemometrics catalogue —
sparse PLS, OPLS, CPPLS, MB-PLS, kernel PLS, AOM/POP, calibration
transfer, variable selection (VIP / CARS / GA / SPA / …) — are
*scattered* across an ecosystem:

| Language | Where the algorithms live |
|---|---|
| Python   | `sklearn.cross_decomposition`, `ikpls`, `diPLSlib`, `hoggorm`, `tensorly`, `pybaselines`, in-tree implementations of papers |
| R        | `pls`, `spls`, `OmicsPLS`, `prospectr`, `mdatools`, `multiway`, `kernlab`, `plsVarSel`, `enpls`, `mixOmics`, `chemometrics`, `ropls`, `sgPLS`, `multiblock`, `plsRglm`, `plsRcox`, `softImpute`, `mboost`, … |
| MATLAB   | `plsregress`, `libPLS` |

Each library has its own numerical conventions (NIPALS vs SIMPLS,
unit-variance vs centring-only, deflation policy, intercept handling).
Comparing two methods across two languages quickly becomes a
multi-month integration project. pls4all collapses that surface to a
**single C++ kernel** with a single set of conventions, then exposes
each language's idiomatic API on top.

The benefits stack:

- **Determinism across languages.** Same kernel and same generated
  datasets, with numerical parity checked by explicit gates instead of
  claiming byte-identical outputs.
- **Performance** — BLAS / OpenMP / CUDA accelerated tiers, with a
  scalar reference tier kept around as the parity anchor.
- **Reproducibility** — every binding ships a `.n4a` bundle format
  that round-trips through the C ABI; a model trained in Python can be
  loaded and parity-checked in R or MATLAB.
- **Auditability** — the parity gate compares pls4all predictions to
  the external reference library that "owns" each algorithm
  (sklearn for PLS, ropls for OPLS, spls for sparse PLS, …) and
  publishes a verdict for every cell.

## The three layers

```
┌──────────────────────────────────────────────────────────────┐
│  Tier-2 idiomatic API                                       │
│    pls4all.sklearn.PLSRegression(...)      (Python)         │
│    pls(y ~ ., data, ncomp=)                (R)              │
│    pls4all.fitrpls(X, y, "NumComponents", k)  (MATLAB)      │
│    new pls4all.PLS({nComponents: k})       (JS)             │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│  Tier-1 raw / canonical API                                  │
│    pls4all._methods.pls_fit(ctx, cfg, X, y, k)  (Python)    │
│    pls4all_method("pls", X, y, n_components=k)  (R)         │
│    pls4all.pls_fit(X, y, k)                     (MATLAB)    │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│  Tier-0 — C ABI (libn4m)                                     │
│    n4m_*  symbols  (96 of them, frozen at ABI 1.x)           │
└──────────────────────────────────────────────────────────────┘
```

The C ABI is the only place numerical algorithms live. Every
binding above is a *reformatter* — no PLS math is duplicated in
Python or R.

## What's in the box

- **~70 algorithms** — every PLS variant in mainstream use plus the
  full chemometrics variable-selection catalogue. The complete
  catalogue is the [methods index](../methods/index.md).
- **A cross-binding parity gate** — for each `(algorithm, n, p, threads)`
  cell, every binding's predictions are compared element-wise to the
  reference library for that algorithm. See the
  [benchmark overview](../benchmarks/overview.md) and the
  [interactive dashboard](../landing/dashboard.md).
- **A stable C ABI** — frozen at 1.x; semantic versioning enforced by
  a per-PR ABI symbol gate. See [abi/reference](../abi/reference.md).
- **A `.n4a` bundle format** — content-addressed serialisation of a
  fitted model, portable across languages.
- **Acceleration matrix** — five libn4m builds (`ref`, `blas`, `omp`,
  `blas+omp`, `cuda`) so every cell can also serve as a benchmark of
  the acceleration stack itself.

## What pls4all is *not*

- **Not a data-loading framework.** pls4all assumes you arrive with
  `(X, y)` already in memory. Spectroscopy file formats, signal-type
  detection, dataset versioning live in upstream tooling
  (e.g. [`nirs4all`](https://github.com/GBeurier/nirs4all)).
- **Not a pipeline DSL.** Pipelines are composed in the host language
  (sklearn `Pipeline`, R `caret` / `mlr3`, MATLAB function chains).
- **Not a deep-learning library.** pls4all is strictly the PLS family
  plus the chemometrics adjuncts (variable selection, calibration
  transfer, diagnostics).

## Where to go next

| If you want to… | Read |
|---|---|
| Run your first fit in your language | [Getting started](getting_started.md) |
| Understand the data model and tiers | [Core concepts](concepts.md) |
| See what's measured and how | [Benchmark overview](../benchmarks/overview.md) |
| Browse the algorithm catalogue | [Methods index](../methods/index.md) |
| Compare bindings in a live UI | [GitHub Pages dashboard](../landing/dashboard.md) |
| Read pls4all in your language | [Python](../bindings/python.md) · [R](../bindings/r.md) · [MATLAB / Octave](../bindings/matlab.md) · [JS](../bindings/js.md) |
