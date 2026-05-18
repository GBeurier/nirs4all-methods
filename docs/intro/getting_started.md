# Getting started

A 5-minute path to a fitted PLS model in your language of choice.

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
    model.export("my_model.n4a")
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
yhat <- pls4all_predict(res, X_test)
```

The dispatcher covers all 71 methods; switch by changing the first
arg (e.g. `"sparse_simpls"`, `"cppls"`, `"opls"`, `"cars_select"`).

## MATLAB / Octave

### Install (MATLAB / Octave)

```matlab
% from bindings/matlab/
build_mex                          % produces +pls4all/p4a_*_mex.mex
addpath(pwd);                      % make +pls4all visible
```

### Fit a model — Statistics-toolbox style

```matlab
mdl  = pls4all.fitrpls(X, y, "NumComponents", 5);
yhat = predict(mdl, Xtest);
mse  = loss (mdl, Xtest, ytest);
```

### Fit a model — MATLAB raw dispatcher

```matlab
res = pls4all.fit("pls", X, y, "NumComponents", 5);
yhat = predict(res, Xtest);
```

## JavaScript / WebAssembly

The WebAssembly binding ships the same surface; see
[bindings/js](../bindings/js.md). The library is loaded as an ES
module:

```javascript
import init, { Context, Config, fitPls } from '@pls4all/wasm';

await init();   // loads libp4a.wasm
const ctx = new Context();
const cfg = new Config({ algorithm: 'pls_regression', nComponents: 5 });
const model = fitPls(ctx, cfg, X, y);
const yhat = model.predict(Xtest);
```

## Save / load across languages

```python
# Python
mdl.export("model.n4a")
```

```r
# R loads the same file
loaded <- pls4all_load("model.n4a")
yhat   <- pls4all_predict(loaded, X_test)
```

```matlab
% MATLAB loads the same file
loaded = pls4all.load("model.n4a");
yhat   = predict(loaded, Xtest);
```

Bytes round-trip exactly. The bundle carries pls4all version + ABI
version so a stale file fails loudly rather than producing wrong
predictions.

## Next steps

- Browse the [methods index](../methods/index.md) — every algorithm
  with parameters, math, and a per-binding example.
- Read the [benchmark overview](../benchmarks/overview.md) to learn
  how parity verdicts and timings are produced.
- See the live [GitHub Pages dashboard](../landing/dashboard.md) to compare bindings
  interactively.
