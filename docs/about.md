# chemometrics4all documentation

chemometrics4all is a portable chemometrics / NIRS operator library with a C++17
core, a stable C ABI target, and thin language bindings.

The project does not publish cross-binding benchmark numbers yet. The landing
page is a benchmark-status placeholder backed by the initial registry under
`benchmarks/`.

## Quick links

- **Architecture** — [overview](architecture/overview.md) · [memory model](architecture/memory_model.md) · [error model](architecture/error_model.md) · [threading](architecture/threading.md) · [serialization](architecture/serialization.md)
- **ABI** — [reference](abi/reference.md) · [stability policy](abi/stability_policy.md) · [changes log](abi/changes_log.md)
- **Bindings** — [Python](bindings/python.md) · [R](bindings/r.md) · [MATLAB](bindings/matlab.md) · [JavaScript / WebAssembly](bindings/js.md) · [Android](bindings/android.md)
- **Parity** — [methodology](parity/methodology.md) · {doc}`tolerances <parity/tolerances>`
- **Benchmarks** — [index](benchmarks/index.md) · [overview](benchmarks/overview.md) · [methodology](benchmarks/methodology.md)
- **Development** — [workflow](dev/workflow.md) · [build](dev/build.md) · [testing](dev/testing.md) · [style](dev/style.md) · [release process](dev/release_process.md)

```{toctree}
:hidden:
:maxdepth: 2
:glob:

intro/*
abi/reference
abi/stability_policy
abi/changes_log
architecture/overview
architecture/memory_model
architecture/error_model
architecture/threading
architecture/serialization
benchmarks/index
benchmarks/overview
benchmarks/methodology
algorithms/*
bindings/python
bindings/r
bindings/matlab
bindings/js
bindings/android
parity/methodology
parity/tolerances
dev/workflow
dev/build
dev/testing
dev/style
dev/release_process
dev/readthedocs
landing/dashboard
handoff/HANDOFF
```

## Benchmark status

The benchmark registry currently covers a starter set of non-model operators:
SNV, MSC, Savitzky-Golay, AsLS, AirPLS, ArPLS, Kennard-Stone, and Hotelling T2.
It is a reference contract, not a result set.

Run this local consistency check after editing the benchmark registry:

```bash
python benchmarks/check_truth_sources.py
```
