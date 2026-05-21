# Python Binding

The Python package is a ctypes binding over the `pls4all` C ABI. It exposes:

- native lifecycle objects: `Context`, `Config`, `Model`, `MethodResult`;
- the full public method surface: diagnostics, monitoring, AOM/POP,
  multi-block methods, selectors, ECR/IRIV/IRF/VIP-SPA and related
  `p4a_method_result_t` APIs;
- a scikit-learn-compatible layer under `pls4all.sklearn`.

## Build The Native Library

```bash
cmake --preset dev-release
cmake --build --preset dev-release --parallel
```

This produces `build/dev-release/cpp/src/libp4a.so` on Linux, or the
equivalent dynamic library on macOS/Windows.

## Loader Rules

In order:

1. `$PLS4ALL_LIB_PATH` points directly to `libp4a`.
2. `pls4all/lib/libp4a*` next to the installed package, for wheel layout.
3. `<repo-root>/build/dev-release/cpp/src/libp4a*`, for editable checkout.
4. The standard system search path.

## Low-Level API

```python
import numpy as np
import pls4all

X = np.random.randn(50, 5)
y = X[:, 0] + 0.5 * X[:, 1]
Y = y.reshape(-1, 1)

with pls4all.Context() as ctx, pls4all.Config() as cfg:
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.SIMPLS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = 3

    model = pls4all.Model.fit(ctx, cfg, X, Y)
    try:
        yhat = model.predict(ctx, X)
        coefs = model.coefficients
    finally:
        model.close()
```

Method-result functions follow the C ABI names:

```python
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    cfg.n_components = 3
    result = pls4all.sparse_simpls_fit(ctx, cfg, X, Y, sparsity_lambda=0.01)
    try:
        yhat = result.matrix("predictions")
    finally:
        result.close()
```

## Scikit-Learn API

The `pls4all.sklearn` submodule provides estimators compatible with the
scikit-learn `fit`/`predict`/`score`, `get_params` and `set_params`
contracts.

```python
from pls4all.sklearn import PLSRegression, PCR, OPLSRegression

reg = PLSRegression(n_components=3)
reg.fit(X, y)
yhat = reg.predict(X)
r2 = reg.score(X, y)

pcr = PCR(n_components=3).fit(X, y)
opls = OPLSRegression(n_components=1).fit(X, y)
```

Selectors expose the usual scikit-learn feature-selection shape:

```python
from pls4all.sklearn import VIPSelector, CARSSelector

sel = VIPSelector(n_components=3, top_k=10).fit(X, y)
mask = sel.get_support()
X_selected = sel.transform(X)
```

Some kernels only expose in-sample predictions because the C result does
not contain enough state for `predict(Xnew)`. Those estimators document the
restriction in their class docstrings and raise a clear error for new
inputs.

## Julia Compatibility Mapping

The Julia binding now mirrors the Python ergonomics:

- Python `pls4all.sklearn.PLSRegression` maps to Julia
  `Pls4all.Sklearn.PLSRegression`.
- Python scikit-learn estimators map to Julia MLJModelInterface types
  `Pls4all.PLSRegressor`, `Pls4all.PCRRegressor`, and
  `Pls4all.OPLSRegressor`.
- Python NumPy/pandas-style tabular inputs map to Julia Tables.jl inputs
  accepted by `Pls4all.table_matrix`, `Pls4all.pls_fit`, and
  `Pls4all.Sklearn.fit!`.

