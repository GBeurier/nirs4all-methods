# chemometrics4all documentation

chemometrics4all is a portable chemometrics / NIRS operator library with a C++17
core, a stable C ABI target, and thin language bindings.

The project does not publish cross-binding benchmark numbers yet. The landing
page is a benchmark-status placeholder backed by the initial registry under
`benchmarks/`.

## Quick links

- **Architecture** — [overview](architecture/overview.md) · [memory model](architecture/memory_model.md) · [error model](architecture/error_model.md) · [threading](architecture/threading.md) · [serialization](architecture/serialization.md)
- **ABI** — [reference](abi/reference.md) · [stability policy](abi/stability_policy.md) · [changes log](abi/changes_log.md)
- **Methods** — [catalogue](methods/index.md) · generated pages with signatures, bibliography, math, implementations, parity and timings
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
benchmarks/external_reference_audit
methods/index
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

The benchmark dashboard is backed by committed cross-binding results and
reference snapshots. Each method name links to a generated method page with
the public signatures, mathematical contract, reference libraries, parity
status, and timing rows.

Run this local consistency check after editing the benchmark registry:

```bash
python benchmarks/check_truth_sources.py
```
