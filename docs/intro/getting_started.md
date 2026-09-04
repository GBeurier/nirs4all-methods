# Getting started

A 5-minute path to a fitted PLS model in your language of choice.

The examples on this page target `libn4m` ABI `2.5.0`. The Methods project and
each binding have independent package versions; runtime hosts must validate the
C ABI before creating a context.

The shared assumption: `X` is an `(n × p)` matrix of predictors and
`y` is an `(n,)` (or `(n × q)`) response. Centring is on, scaling
defaults to off — the spectroscopy convention.

## Python

### Install (Python)

```bash
# 1. build the C library (or install the wheel when published)
cmake --preset dev-release
cmake --build --preset dev-release --parallel

# 2. install the Python binding
pip install ./bindings/python
```

### Fit a model — sklearn-style

```python
from pls4all.sklearn import PLSRegression

mdl = PLSRegression(n_components=5).fit(X, y)
yhat = mdl.predict(X_test)
score = mdl.score(X_test, y_test)        # R²
```

The estimator is a real sklearn `BaseEstimator` — drop it in a
`Pipeline`, `GridSearchCV`, or any cross-validator.

### Fit a model — Python raw

```python
import pls4all

with pls4all.Context() as ctx, pls4all.Config() as cfg:
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver    = pls4all.Solver.SIMPLS
    cfg.n_components = 5
    model = pls4all.Model.fit(ctx, cfg, X, y)
    yhat  = model.predict(ctx, X_test)
    payload = model.to_bytes()  # raw N4MM bytes; no canonical file extension
    restored = pls4all.Model.from_bytes(ctx, payload)
    yhat_restored = restored.predict(ctx, X_test)
    restored.close()
    model.close()
```

## R

### Install (R)

```r
# from the repo's bindings/r/pls4all directory:
install.packages(".", repos = NULL, type = "source")
```

### Fit a model — formula + S3

```r
library(pls4all)
fit  <- pls(y ~ ., data = train_df, ncomp = 5)
yhat <- predict(fit, newdata = test_df)
summary(fit)
```

### Fit a model — R raw dispatcher

```r
res  <- pls4all_method("pls", X, y, n_components = 5)
yhat <- n4m_predict(res, X_test)
```

The dispatcher covers all 71 methods; switch by changing the first
arg (e.g. `"sparse_simpls"`, `"cppls"`, `"opls"`, `"cars_select"`).

## MATLAB / Octave

### Install (MATLAB / Octave)

```matlab
% from bindings/matlab/
build_mex                          % produces +n4m/n4m_*_mex.mex
addpath(pwd);                      % make +n4m visible
```

### Fit a model — Statistics-toolbox style

```matlab
mdl  = n4m.fitrpls(X, y, "NumComponents", 5);
yhat = predict(mdl, Xtest);
mse  = loss (mdl, Xtest, ytest);
```

### Fit a model — MATLAB raw dispatcher

```matlab
res = n4m.fit("pls", X, y, "NumComponents", 5);
yhat = predict(res, Xtest);
```

## JavaScript / WebAssembly

The WebAssembly binding ships the same surface; see
[bindings/js](../bindings/js.md). The library is loaded as an ES
module:

```javascript
import * as n4m from '@nirs4all/methods';

await n4m.loadModule();   // loads n4m.wasm
// Raw row-major Float64Array matrices in, typed arrays out (see
// bindings/js/INPUT_CONTRACT.md).
const model = n4m.fitPls({ data: X, rows, cols }, { data: y, rows, cols: 1 }, 5);
const yhat = n4m.predictPls(model, { data: Xtest, rows: rowsTest, cols });
```

## Rust

The `n4m` crate is the official thin Rust binding. It requires an
exact prebuilt `libn4m`; it does not contain a second numerical engine.

```rust
n4m::configure_library("/absolute/path/to/libn4m.so.2")?;
let context = n4m::Context::new()?;
```

The dynamic selection is process-wide and one-shot. A missing library, an ABI
mismatch or an attempt to switch libraries after initialization is refused
before a native handle is created. See the
[Rust binding guide](https://github.com/GBeurier/nirs4all-methods/tree/codex/r4-doc002-methods/bindings/rust/n4m)
for linked and dynamic
loading, model serialization and HPO examples.

## Serialize raw fitted state

```python
# Python slim binding
payload = mdl.to_bytes()
info = pls4all.inspect_n4mm(payload)  # validates all bytes before import
assert info.format_version in (1, 2)
loaded = pls4all.Model.from_bytes(ctx, payload)
assert payload[:4] == b"N4MM"
loaded.close()
```

N4MM is a raw fitted-model payload behind the C ABI, not a full pipeline
bundle. It has no canonical filename extension. Corruption and unsupported
N4MM format versions fail; the writer ABI triple is provenance and an ABI
difference currently produces a context warning rather than a load failure.

R, MATLAB and JS do not yet expose N4MM model import/export wrappers. The
`.n4a` extension remains reserved for the nirs4all full-pipeline bundle, which
has a separate lifecycle and contract.

For a Core Archive V2 Methods member, use the
[R2 artifact preflight](../native-default-r2.md) before import. It maps each
wire limitation to the Python, JavaScript, or C inspector and links the
scientific [SNV](../methods/pp_snv.md),
[Savitzky-Golay](../methods/pp_savgol.md), and
[PLS](../methods/pls.md) contracts used by format 2.

## Next steps

- Browse the [methods index](../methods/index.md) — every algorithm
  with parameters, math, and a per-binding example.
- Read the [benchmark overview](../benchmarks/overview.md) to learn
  how parity verdicts and timings are produced.
- See the live [GitHub Pages dashboard](../landing/dashboard.md) to compare bindings
  interactively.
