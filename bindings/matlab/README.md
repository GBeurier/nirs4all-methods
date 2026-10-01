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
- `n4m.RolePipeline(steps, featureNames, options)` for native fitted recipes,
  feature identity and portable per-step N4ME states
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
└── test/                     Cross-binding parity and native lifecycle gates
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
- `n4m_role_pipeline_mex`
- `n4m_version_mex`

To build only the RolePipeline and runtime-version shims:

```matlab
build_mex({'n4m_role_pipeline_mex', 'n4m_version_mex'})
```

RolePipeline uses the published ABI 2.14 surface. A build against a released
library must use that release's headers, including generated `n4m/n4m_export.h`.
For example, when the release header pack includes that generated file:

```bash
N4M_INCLUDE_DIR=/path/to/released/include \
N4M_GENERATED_DIR=/path/to/released/include \
N4M_LIB_DIR=/path/to/released/lib \
octave --no-gui --no-history --eval \
  "addpath('bindings/matlab'); build_mex({'n4m_role_pipeline_mex','n4m_version_mex'})"
```

Do not combine checkout headers for a newer ABI with an older public library.
`which('n4m.n4m_role_pipeline_mex')` identifies the loaded shim, and
`n4m.version()` reports the loaded native library version.

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

## Native RolePipeline

```matlab
addpath('bindings/matlab')
steps = {struct('method_id', 'models.regularized.ridge', ...
                'params', struct('alpha', 0.05))};
featureNames = {'source:nir/first.column', 'source:nir/second.column'};
model = n4m.RolePipeline(steps, featureNames, struct('num_threads', 1));
X = [1 0; 0 1; 1 1; 2 1];
Y = [2; 3; 5; 7];
model.fit(X, Y);
prediction = model.predict(X, featureNames);
states = model.exportStates();  % ordered cell array of uint8 N4ME vectors
info = model.stepsInfo();
model.close();

replay = n4m.RolePipeline(steps, featureNames);
replay.importStates(states);     % native import, no FIT
prediction = replay.predict(X, featureNames);
replay.close();
```

`steps` is a cell vector of scalar structs or a struct vector. Each step has
`method_id`, the exact native catalog ID, and optional `params`, a scalar
struct of named values. The native catalog supplies defaults and validates
types, bounds, required inputs and role order. Integer parameters accept
integer classes or exactly represented integral doubles; use `int64` for
larger values. Continuous parameters and arrays use `double`; booleans accept
logical scalars or zero/one; enums use character row strings. Recipes contain
optional sample filters, then transformers/selectors, and one terminal
regressor or classifier. A PLS step inside a recipe acts as a transformer.

`fit(X, Y, aux)` borrows full real double matrices through native column-major
views. `Y` may have several target columns, or be empty for a classifier.
The optional scalar `aux` struct accepts `labels`, `sample_weight`, `groups`,
`feature_groups`, `block_sizes`, `axis`, `X_target` and `fold_ids`. Integer
channels use exact int64-compatible vectors; weights and axis use double
vectors; `X_target` uses a full real double matrix. Native validation checks
every row/column count and rejects inputs no recipe step uses. All routing,
filtering, fitting and numerical operations execute in `libn4m`.

`predict`, `transform`, `decisionFunction`, `predictProba` and `predictLabels`
take `X` and optional ordered feature names. The native terminal role and
capabilities determine available operations; unsupported operations raise an
error. Classifier `predict` and `predictLabels` return an int64 column vector,
and `classes()` returns ascending native int64 class IDs. This facade accepts
integer class IDs in `aux.labels`; applications that use named labels keep
their own ordered label table. `nFeaturesIn()`, `nOutputs()` and
`transformCols()` report fitted widths, and `isFitted()` reports native state.

`featureNames()` returns stored names. `setFeatureNames(names)` changes or
clears them before fit/import. Stored and supplied names must have the same
order and spelling; names containing `:` or `.` are passed as cell strings.
Inputs without supplied names are positional. Width mismatches are always
refused. `stepsInfo()` returns native `method_id`, `role`, zero-based
`state_index`, `contains_training_rows`, `n_features_in` and `n_features_out`.
Sample filters have state index -1 and contribute no exported state.

`exportStates()` refuses states retaining training rows. Such an export
requires the explicit call `exportStates(true)`; imported states retain that
same policy. `importStates(states)` checks native method/parameter identity,
state count, chained widths, payload integrity and the context byte limit
(256 MiB by default, configurable through constructor `max_state_bytes`).
The binding commits a fit or import only after it succeeds, preserving any
previous fitted model when a replacement fails. Recipe and feature metadata
must travel with the state bytes; DAG-ML owns its RAW/archive wrapper.

Call `close()` or `delete()` to release native resources. Closing is
idempotent. The MEX registry uses tokens rather than user pointers, locks the
loaded shim while models exist, and releases remaining entries at process
exit. Saving or loading the native handle through MATLAB object serialization
is refused; persist N4ME bytes and the recipe instead.

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

The RolePipeline gate checks matrix/target layout, typed parameters, native
auxiliary inputs, feature identity, classifier IDs, failed replacement,
training-row export policy, handle lifetime and N4ME refusals. It also consumes
the canonical `parity/fixtures/role_pipeline_negative.json` witness for
Python-produced regression/classification states, predictions, transforms,
decisions and negative cases. The fixture ABI must exactly equal the loaded
runtime ABI. Select a release fixture explicitly when using a released
runtime; payloads from a newer checkout must not be substituted.

```bash
python3 bindings/matlab/test/export_role_pipeline_fixture.py \
  /path/to/matching/role_pipeline_negative.json /tmp/n4m_role_fixture.m
N4M_ROLE_PIPELINE_FIXTURE=/tmp/n4m_role_fixture.m \
octave --no-gui --no-history --eval \
  "addpath('bindings/matlab'); addpath('bindings/matlab/test'); test_role_pipeline"
```

The renderer changes only test-data representation, retaining exact integers,
booleans, cell ordering and N4ME bytes. On a runtime with `jsondecode`,
`test_role_pipeline('/path/to/matching/role_pipeline_negative.json')` reads the
JSON directly. The generated `.m` path supports Octave versions without
`jsondecode`; it is trusted test data and is never accepted by product import.
RolePipeline qualification requires real compiled MEX execution. Its runtime
qualification is pending until that gate is run; existing optimizer parity
results do not qualify this new surface.

## Limitations

- RolePipeline borrows column-major input matrices; some lower-level MEX shims
  convert layout. Batch work into full-block calls rather than per-row loops.
- The shared public subset targets the MATLAB/Octave intersection. Any runtime
  divergence must be documented in `COMPAT.md`.
