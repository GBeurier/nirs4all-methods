# `ridge` - direct closed-form Ridge regression

_Group_: **Regularised** · _ABI_: `n4m_ridge_fit`

## Description

`ridge` fits the native closed-form multi-output Ridge model:

```text
beta = (Xc'Xc + alpha I)^-1 Xc'Yc
intercept = y_mean - x_mean beta
```

The solver is selected by shape: primal augmented-QR for `p <= n`, dual
Gram-on-samples for `p > n`. This is distinct from `ridge_pls`, which is
ridge-augmented SIMPLS.

## Python Usage

```python
import n4m

res = n4m.ridge(X, y, alpha=0.1)
y_hat = res["predictions"]
coef = res["coefficients"]
intercept = res["intercept"]
```

Returned keys include `coefficients`, `intercept`, `x_mean`, `x_scale`,
`y_mean`, `predictions`, scalar `rmse`, and scalar `lambda`.

Reusable sklearn-style wrapper:

```python
from n4m.sklearn import NativeRidgeRegressor

model = NativeRidgeRegressor(alpha=0.1).fit(X, y)
y_hat = model.predict(X_test)
```

## C ABI

```c
n4m_ridge_fit(ctx, cfg, &x_view, &y_view,
              &alpha, 1, &result);
```
