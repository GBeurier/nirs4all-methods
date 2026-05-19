# Core concepts

chemometrics4all focuses on portable chemometrics / NIRS operators, not on
shipping PLS model variants. The PLS model family lives in `pls4all`.

## Layers

The intended architecture has three layers:

| Layer | Role |
|---|---|
| C++ core | Numerical kernels and deterministic utilities. |
| C ABI | Stable `c4a_*` entry points, opaque handles, and error reporting. |
| Bindings | Thin host-language wrappers around the C ABI. |

## Determinism

Future benchmarks must use deterministic datasets, pinned truth sources, and
recorded result metadata. No result is authoritative unless it can be traced
back to:

- the method entry in `benchmarks/benchmark_registry.json`;
- locked references in `benchmarks/truth_sources.lock.json`;
- the exact package versions or project commits used to produce the result.

## Benchmark Registry

The initial registry groups non-model operators by family:

| Group | Representative methods |
|---|---|
| Preprocessing | `snv`, `msc`, `savitzky_golay` |
| Baseline | `asls`, `airpls`, `arpls` |
| Splitter | `kennard_stone` |
| Diagnostic | `hotelling_t2` |

The landing page is a benchmark-status scaffold until real generated results
exist.
