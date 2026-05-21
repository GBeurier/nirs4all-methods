# What is nirs4all-methods?

nirs4all-methods is a portable chemometrics / NIRS operator library written in
C++17 and exposed through a stable C ABI target plus thin language bindings.

The repository does not currently publish a real cross-binding benchmark matrix.
The landing page is an explicit scaffold so inherited pls4all benchmark numbers
cannot be mistaken for nirs4all-methods results.

## What's in Scope

- preprocessing and normalization operators;
- baseline correction;
- sample splitters;
- filters and diagnostics;
- augmentation and transfer-metric utilities;
- benchmark registry and truth-source tracking for future result publication.

PLS regression and classification models belong in `pls4all`.

## Where To Go Next

| If you want to... | Read |
|---|---|
| Build the project | [Development build](../dev/build.md) |
| Understand benchmark status | [Benchmark overview](../benchmarks/overview.md) |
| Read benchmark policy | [Benchmark methodology](../benchmarks/methodology.md) |
| Check the landing payload | [Landing dashboard](../landing/dashboard.md) |
| Read binding notes | [Python](../bindings/python.md) · [R](../bindings/r.md) · [MATLAB](../bindings/matlab.md) |
