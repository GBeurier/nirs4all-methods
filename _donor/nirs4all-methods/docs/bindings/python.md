# Python binding

The current package ships a ctypes-only wrapper that exposes ABI introspection and the
context / config lifecycles. See `bindings/python/README.md` for installation and loader rules.

## Hello-version

```python
import n4m

print(n4m.version())          # "0.58.0+abi.1.0.0"
print(n4m.abi_version())      # (1, 0, 0)
print(n4m.build_info())       # ""

with n4m.Context() as ctx:
    ctx.seed = 42
    assert ctx.seed == 42
    try:
        ctx.backend = n4m.Backend.CUDA
    except n4m.N4MError as exc:
        print(exc.status_name)    # "backend unavailable"
        print(exc.last_error)     # "backend 5 is not compiled into this build of libn4m"

with n4m.Config() as cfg:
    cfg.algorithm = n4m.Algorithm.PCR
    cfg.solver = n4m.Solver.SVD
    cfg.deflation = n4m.Deflation.REGRESSION
    assert cfg.algorithm == n4m.Algorithm.PCR
    assert cfg.solver == n4m.Solver.SVD
    assert cfg.deflation == n4m.Deflation.REGRESSION
    cfg.algorithm = n4m.Algorithm.PLS_SVD
    cfg.deflation = n4m.Deflation.CANONICAL
    assert cfg.algorithm == n4m.Algorithm.PLS_SVD
    assert cfg.deflation == n4m.Deflation.CANONICAL
```

Phase 2 expands the surface to a full sklearn-compatible estimator with
zero-copy NumPy `n4m_matrix_view_t` round-trips.

## EPO compatibility note

`EPO.fit_transform(X, d)` applies the training-time EPO projection expected by
nirs4all/sklearn-style callers via the public
`n4m_pp_epo_transform_with_d(...)` ABI entry point. `EPO.transform(X)` remains
available for the no-`d` contract and is therefore a pass-through at
`d = d_mean`. Use `EPO.transform_with_d(X, d)` when the external parameter is
known for the transform batch.
