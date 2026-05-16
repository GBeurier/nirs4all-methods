# Phase 34 — Octave / MATLAB MEX Binding

**Status**: shipped 2026-05-16 · tag `phase-34-octave-matlab-binding` ·
release `0.86.0+abi.1.13.0`.

## Goal

Ship a MATLAB / GNU Octave binding for libp4a backed by the public
`p4a_pls_fit_simple` C ABI helper introduced in Phase 33, gated by the
same cross-binding parity fixture used by WASM / R / Julia. The
binding must be byte-equivalent (rmse_rel ≤ 1e-12) to the native
Python reference.

## Why MEX

* MATLAB / Octave is the dominant chemometrics teaching environment.
  Several major NIRS users — including the original PLS_Toolbox /
  Eigenvector workflows — sit on top of MATLAB column-major double
  matrices.
* MEX gives a stable, vendor-neutral entry point: the same C source
  compiles via `mkoctfile --mex` (Octave) and `mex` (MATLAB) with no
  preprocessor branching.
* By routing through `p4a_pls_fit_simple` (raw-pointer ABI), we avoid
  the matrix-view-pointer interop quirks that bit WASM and Julia.

## Architecture

```
bindings/matlab/
├── mex/
│   ├── p4a_pls_fit_mex.c    — column-major ↔ row-major adapter +
│   │                          call into p4a_pls_fit_simple
│   └── p4a_version_mex.c    — thin wrapper around p4a_get_version_string
├── +pls4all/
│   ├── pls_fit.m            — public function: pls4all.pls_fit(X, Y, k)
│   ├── version.m            — pls4all.version() → "0.86.0+abi.1.13.0"
│   ├── p4a_pls_fit_mex.mex  — compiled artifact (built by build_mex.m)
│   └── p4a_version_mex.mex
├── build_mex.m              — Octave + MATLAB build script
└── test/
    └── test_parity.m        — parity gate vs bindings/js/test/parity_fixture.json
```

The package directory `+pls4all/` exposes everything under the
`pls4all.*` namespace so users can call `pls4all.pls_fit(...)` and
`pls4all.version()` from any MATLAB / Octave session after a single
`addpath('bindings/matlab')`.

## Parity gate

```
$ OCTAVE_HOME=$CONDA_PREFIX \
    LD_LIBRARY_PATH=$(pwd)/build/dev-release/cpp/src:... \
    octave --quiet --eval "addpath('bindings/matlab/test'); test_parity"

pls4all.version()        = 0.86.0+abi.1.13.0
fixture pls4all_version  = 0.85.0+abi.1.13.0
rmse_rel coefficients    = 0.000e+00
rmse_rel x_mean          = 2.708e-17
rmse_rel y_mean          = 0.000e+00
rmse_rel predictions     = 4.732e-17
PARITY GATE PASS
```

`rmse_rel ≤ 1e-12` was the gate; we are several orders of magnitude
inside it (≤ 5e-17, i.e. the test is exact to double-precision
machine epsilon).

## Implementation notes

* **Column-major ↔ row-major**: MATLAB / Octave doubles are
  column-major; the libp4a ABI is row-major. `colmajor_to_rowmajor`
  / `rowmajor_to_colmajor` allocate fresh buffers and transpose
  explicitly. For the parity input (50 × 5) the cost is negligible
  (~2 µs); for production-size matrices users should fit larger
  blocks rather than calling per-row.
* **Package-qualified internal calls**: Octave's `+pkg/` packages do
  NOT share an unqualified namespace internally. Inside `pls_fit.m`
  the MEX must be called as `pls4all.p4a_pls_fit_mex(...)`, not
  `p4a_pls_fit_mex(...)`. This caught us during the parity run.
* **Conda Octave OCTAVE_HOME**: the conda-forge Octave 10.3 package
  ships with placeholder paths embedded; the harness sets
  `OCTAVE_HOME=$CONDA_PREFIX` before invoking `octave` so the
  standard-library m-files are findable.
* **MATLAB path**: same source, same build script. `mex` instead of
  `mkoctfile --mex`, plus `-output` instead of `-o`. The build
  script in `build_mex.m` autodetects which is available.

## ABI surface

No new symbols. The binding uses `p4a_pls_fit_simple`
(ABI 1.13+, added in Phase 33). ABI version remains 1.13.0.

## Release

* Library version: `0.85.0` → `0.86.0` (additive: new binding).
* ABI version: unchanged at `1.13.0`.
* Tag: `phase-34-octave-matlab-binding`.

## Backlog / follow-ups

* MEX wrapper for `p4a_pls_predict` against an already-fitted model
  (currently the binding only exposes the one-shot fit + in-sample
  prediction path).
* Document the OCTAVE_HOME requirement in `bindings/matlab/README.md`
  once we publish prebuilt binaries; users installing via apt /
  Homebrew / native Octave won't hit the conda-placeholder issue.
