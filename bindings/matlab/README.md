# n4m MATLAB / Octave binding

MEX shims that expose the public `libn4m` C ABI to MATLAB and GNU Octave. The
binding exposes the V1 user-facing namespace as `+n4m`; compiled entry
points and exported C symbols use the `n4m_*` prefix.

## Surface

- `n4m.version()`
- `n4m.pls_fit(X, Y, n_components)`
- `n4m.snv_transform(X, ...)`
- `n4m.savgol_transform(X, ...)`
- `n4m.kennard_stone_split(X, ...)`
- generated method/model wrappers backed by `n4m_method_fit_mex` and
  `n4m_model_fit_mex`

The preprocessing and splitter wrappers are the upstream execution surface used
by `nirs4all-core` MATLAB/Octave parity tests for the portable
Kennard-Stone/SNV/Savitzky-Golay/PLS subset.

## Layout

```text
bindings/matlab/
├── mex/                      C sources for MEX shims
├── +n4m/                     V1 namespace functions, classes, and MEX artifacts
├── build_mex.m               Build script for Octave and MATLAB
└── test/test_parity.m        Cross-binding parity gate
```

## Build

Build `libn4m` first:

```bash
cmake --preset dev-release
cmake --build --preset dev-release --target n4m_c --parallel
```

Then compile the MEX shims. The script uses the repo's `dev-release` build by
default and honors `N4M_INCLUDE_DIR`, `N4M_GENERATED_DIR`, and
`N4M_LIB_DIR` when the library lives elsewhere.

```bash
octave --no-gui --no-history --eval "cd bindings/matlab; build_mex"
```

From MATLAB:

```matlab
cd bindings/matlab
build_mex
```

`build_mex.m` installs these compiled shims into `bindings/matlab/+n4m`:

- `n4m_preprocess_mex`
- `n4m_split_mex`
- `n4m_method_fit_mex`
- `n4m_model_fit_mex`
- `n4m_pls_fit_mex`
- `n4m_version_mex`

## Usage

```matlab
addpath('bindings/matlab')

v = n4m.version();

X = randn(50, 20);
Y = X(:, 1) + 0.5 * X(:, 2);

Xsnv = n4m.snv_transform(X);
Xsavgol = n4m.savgol_transform(Xsnv, ...
    'window_length', 11, 'polyorder', 3, 'deriv', 0, 'mode', 'interp');
split = n4m.kennard_stone_split(Xsavgol, 'test_size', 0.25, 'zero_based', true);

[coefs, x_mean, y_mean, preds] = n4m.pls_fit(Xsavgol, Y, 3);
```

## Parity gate

```bash
LD_LIBRARY_PATH=$(pwd)/build/dev-release/cpp/src \
octave --no-gui --no-history --eval \
  "addpath('bindings/matlab'); cd bindings/matlab/test; test_parity"
```

`cross-binding-parity.yml` runs the Octave MEX build and parity gate in CI.
MATLAB uses the same source package but remains a manual release/runtime check
because GitHub-hosted runners do not provide a MATLAB license.

## Limitations

- Matrix layout conversion happens on each MEX call. For large matrices, batch
  work into full-block calls rather than per-row loops.
- The shared public subset targets the MATLAB/Octave intersection. Any runtime
  divergence must be documented in `COMPAT.md`.
