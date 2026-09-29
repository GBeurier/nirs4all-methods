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
- `n4m.Optimizer(space, options, constraints)` for native HPO ask/tell,
  pruning, rich trace and N4MOPT checkpoint/restart
- generated method/model wrappers backed by `n4m_method_fit_mex` and
  `n4m_model_fit_mex`

The preprocessing and splitter wrappers are the upstream execution surface used
by `nirs4all-core` MATLAB/Octave parity tests for the portable
Kennard-Stone/SNV/Savitzky-Golay/PLS subset.

## Layout

```text
bindings/matlab/
├── mex/                      C/C++ sources for MEX shims
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
- `n4m_optimizer_mex`
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

axis = struct('name', 'components', 'kind', 'int', 'low', 1, ...
              'high', 12, 'step', 1);
study = n4m.Optimizer(axis, struct('sampler', 'tpe', ...
                                 'n_startup_trials', 4, 'seed', 7));
for k = 1:12
    trial = study.ask();
    % Replace this demonstration score with a held-out host-model metric.
    score = (double(trial.parameters.components) - 4)^2;
    study.tell(trial, score);
end
best = study.best();
checkpoint = study.save();  % uint8 N4MOPT, portable across bindings
study.close();
study = n4m.Optimizer.load(checkpoint, axis);
trace = study.trials();      % owning native trace v1 fields
study.close();
```

The `space` argument is an ordered struct array with `name` and `kind` on each
axis. Numeric axes use `low`, `high` and optional `step`; categorical and
ordinal axes use `choices`; sorted tuples use `length`, `low` and `high`.
Unused fields may be empty. Categorical choices may be cell strings, doubles,
int64 values or logical values. `constraints` is an optional struct array with
`kind`, cell-string `refs` and optional cell-string `labels`. `askBatch(n)`
returns a cell array of committed trials plus native status, including partial
batches; `intermediate(trial, step, score)` returns the native pruning decision.
`enqueue` accepts a struct of numeric values, with zero-based category indices.
For names that are not valid MATLAB struct fields, such as DAG paths containing
points, each trial also carries exact ordered `parameter_names` and
`parameter_values` cell arrays; use `n4m.Optimizer.getParameter(trial, name)`
to retrieve them. `enqueue(names, values)` accepts a cell array of such names.
Call `close` or let the handle destructor release native resources.

## Parity gate

```bash
LD_LIBRARY_PATH=$(pwd)/build/dev-release/cpp/src \
octave --no-gui --no-history --eval \
  "addpath('bindings/matlab'); cd bindings/matlab/test; test_parity"
```

`cross-binding-parity.yml` builds the Octave MEX package and runs
the optimizer lifecycle and 14 selected Python/native HPO golden traces
through this binding with a fixed score tape. The canonical Python specs and
JSON goldens are rendered as a temporary `.m` fixture by
`parity/hpo/export_matlab_fixtures.py` because the CI's Octave 6.4 has no
`jsondecode`. The workflow also continues a Python N4MOPT checkpoint in
Octave and the reciprocal Octave checkpoint in Python, checking the next
proposal exactly in both directions. All four cross-binding jobs, including
the Octave optimizer checks, passed on `38c2cc84` ([CI run](https://github.com/GBeurier/nirs4all-methods/actions/runs/36546137355)).
The [subsequent cross-binding CI](https://github.com/GBeurier/nirs4all-methods/actions/runs/36548060107)
also reproduces all 45 sampler-pruner compositions under Octave.
MATLAB uses the same source; its runtime check is deferred until a license is
available and does not gate the current Octave qualification.

## Limitations

- Matrix layout conversion happens on each MEX call. For large matrices, batch
  work into full-block calls rather than per-row loops.
- The shared public subset targets the MATLAB/Octave intersection. Any runtime
  divergence must be documented in `COMPAT.md`.
