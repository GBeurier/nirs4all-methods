# Archived: Julia binding (frozen)

The Julia binding (`Pls4all.jl`) is **on hold** — Julia is not among the current
target languages (R, Python, Octave/MATLAB, JS; see `bindings/SPEC.md`). It was
moved here from `bindings/julia/` during the cross-binding/parity consolidation
so the active `bindings/` tree contains only the four target languages.

It is a working PoC (raw FFI + a small Sklearn-style wrapper) and is left intact.
Because it now lives under `_archive/`, it is **frozen**: it is excluded from
`scripts/bump_version.sh` version sync and from the cross-binding parity gate, so
its `Project.toml` version no longer tracks the canonical header.

To reactivate (when Julia is picked up via a `binding-request`): move it back
under `bindings/julia/`, re-add its `Project.toml` to `scripts/bump_version.sh`,
re-pin it to the current version, and wire it into the conformance/parity gate
(`benchmarks/cross_binding/`).
