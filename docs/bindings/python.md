# Python binding

The current package ships a ctypes-only wrapper that exposes ABI introspection and the
context / config lifecycles. See `bindings/python/README.md` for installation and loader rules.

## Hello-version

```python
import chemometrics4all

print(chemometrics4all.version())          # "0.58.0+abi.1.0.0"
print(chemometrics4all.abi_version())      # (1, 0, 0)
print(chemometrics4all.build_info())       # ""

with chemometrics4all.Context() as ctx:
    ctx.seed = 42
    assert ctx.seed == 42
    try:
        ctx.backend = chemometrics4all.Backend.CUDA
    except chemometrics4all.Chemometrics4allError as exc:
        print(exc.status_name)    # "backend unavailable"
        print(exc.last_error)     # "backend 5 is not compiled into this build of libc4a"

with chemometrics4all.Config() as cfg:
    cfg.algorithm = chemometrics4all.Algorithm.PCR
    cfg.solver = chemometrics4all.Solver.SVD
    cfg.deflation = chemometrics4all.Deflation.REGRESSION
    assert cfg.algorithm == chemometrics4all.Algorithm.PCR
    assert cfg.solver == chemometrics4all.Solver.SVD
    assert cfg.deflation == chemometrics4all.Deflation.REGRESSION
    cfg.algorithm = chemometrics4all.Algorithm.PLS_SVD
    cfg.deflation = chemometrics4all.Deflation.CANONICAL
    assert cfg.algorithm == chemometrics4all.Algorithm.PLS_SVD
    assert cfg.deflation == chemometrics4all.Deflation.CANONICAL
```

Phase 2 expands the surface to a full sklearn-compatible estimator with
zero-copy NumPy `c4a_matrix_view_t` round-trips.
