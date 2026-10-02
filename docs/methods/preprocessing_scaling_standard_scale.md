# StandardScale

`preprocessing.scaling.standard_scale` is a native transformer exposed as
`n4m.roles.StandardScale`. The optional scikit-learn dependency is required for
the `n4m.roles` Python facade.

```python
from n4m.roles import StandardScale

scaler = StandardScale(with_mean=True, with_std=True).fit(X_train)
X_scaled = scaler.transform(X_test)
```

Both parameters are booleans and default to `True`; `y` is optional. Fitting
learns column means and population standard deviations (`ddof=0`) from the
training data, with a scale of one for zero-variance columns. Transform uses
those learned values. The fitted component state can be saved with `to_n4me()`
and loaded with `StandardScale.from_n4me()`.

The method uses the shared `n4m_estimator_*` C ABI and exports no dedicated C
symbols. Its typed native contract is maintained in
[the method catalog](../../catalog/methods/preprocessing.scaling.standard_scale.yaml).
