# pls4all — MATLAB / Octave binding

MEX shims that expose libp4a's `p4a_pls_fit_simple` C ABI to MATLAB
(R2018a+) and GNU Octave (>= 6.0). Parity-gated against the native
Python reference at `rmse_rel <= 5e-17` (i.e. exact to machine
epsilon) on the shared fixture `bindings/js/test/parity_fixture.json`.

## Layout

```
bindings/matlab/
├── mex/                      C sources for the MEX shims
├── +pls4all/                 MATLAB / Octave package (functions + .mex artifacts)
├── build_mex.m               Build script (Octave + MATLAB)
└── test/test_parity.m        Cross-binding parity gate
```

## Build

### Octave

```bash
# From the repo root, after the CMake "dev-release" preset is built.
# OCTAVE_HOME is required only for the conda-forge Octave package
# whose internal paths are placeholders.

OCTAVE_HOME=$CONDA_PREFIX \
mkoctfile --mex \
  -o bindings/matlab/+pls4all/p4a_pls_fit_mex \
  bindings/matlab/mex/p4a_pls_fit_mex.c \
  -Icpp/include -Ibuild/dev-release/generated \
  -Lbuild/dev-release/cpp/src -lp4a \
  -Wl,-rpath,$(pwd)/build/dev-release/cpp/src

OCTAVE_HOME=$CONDA_PREFIX \
mkoctfile --mex \
  -o bindings/matlab/+pls4all/p4a_version_mex \
  bindings/matlab/mex/p4a_version_mex.c \
  -Icpp/include -Ibuild/dev-release/generated \
  -Lbuild/dev-release/cpp/src -lp4a \
  -Wl,-rpath,$(pwd)/build/dev-release/cpp/src
```

### MATLAB

From inside MATLAB:

```matlab
cd bindings/matlab
build_mex
```

The script honours `PLS4ALL_INCLUDE_DIR`, `PLS4ALL_GENERATED_DIR`, and
`PLS4ALL_LIB_DIR` environment variables; if unset, it falls back to
the CMake `dev-release` preset under the repo.

## Usage

```matlab
addpath('bindings/matlab');     % registers the +pls4all package
v = pls4all.version();          % "0.89.0+abi.1.13.0"

X = randn(50, 5);
Y = X(:, 1) + 0.5 * X(:, 2) - 0.3 * X(:, 3);

[coefs, x_mean, y_mean, preds] = pls4all.pls_fit(X, Y, 3);
%   coefs:  5 x 1 regression coefficients
%   x_mean: 1 x 5 per-feature mean (centring)
%   y_mean: 1 x 1 per-target mean
%   preds:  50 x 1 in-sample predictions
```

## Parity gate

```bash
OCTAVE_HOME=$CONDA_PREFIX \
LD_LIBRARY_PATH=$(pwd)/build/dev-release/cpp/src:... \
octave --quiet --eval "addpath('bindings/matlab/test'); test_parity"
```

Expected output:

```
pls4all.version()        = 0.89.0+abi.1.13.0
fixture pls4all_version  = 0.89.0+abi.1.13.0
rmse_rel coefficients    = 0.000e+00
rmse_rel x_mean          = 2.708e-17
rmse_rel y_mean          = 0.000e+00
rmse_rel predictions     = 4.732e-17
PARITY GATE PASS
```

## Limitations

* Currently only the one-shot `pls_fit` entry point is wrapped.
  `p4a_pls_predict` against a saved model is on the backlog.
* Column-major <-> row-major conversion happens on every call. For
  large matrices, fit on the full block rather than per-row.
