# Python binding

The current package ships a ctypes-only wrapper that exposes ABI introspection and the
context / config lifecycles. See [`bindings/python/README.md`](../../bindings/python/README.md) for installation and loader rules.

## Hello-version

```python
import pls4all

print(pls4all.version())          # "0.28.0+abi.1.0.0"
print(pls4all.abi_version())      # (1, 0, 0)
print(pls4all.build_info())       # ""

with pls4all.Context() as ctx:
    ctx.seed = 42
    assert ctx.seed == 42
    try:
        ctx.backend = pls4all.Backend.CUDA
    except pls4all.Pls4allError as exc:
        print(exc.status_name)    # "backend unavailable"
        print(exc.last_error)     # "backend 5 is not compiled into this build of libp4a"

with pls4all.Config() as cfg:
    cfg.algorithm = pls4all.Algorithm.PCR
    cfg.solver = pls4all.Solver.SVD
    cfg.deflation = pls4all.Deflation.REGRESSION
    assert cfg.algorithm == pls4all.Algorithm.PCR
    assert cfg.solver == pls4all.Solver.SVD
    assert cfg.deflation == pls4all.Deflation.REGRESSION
    cfg.algorithm = pls4all.Algorithm.PLS_SVD
    cfg.deflation = pls4all.Deflation.CANONICAL
    assert cfg.algorithm == pls4all.Algorithm.PLS_SVD
    assert cfg.deflation == pls4all.Deflation.CANONICAL
```

Phase 2 expands the surface to a full sklearn-compatible estimator with
zero-copy NumPy `p4a_matrix_view_t` round-trips.
