# Python binding

The current Python package ships a minimal **ctypes-only** binding that:

- loads `libp4a.{so,dll,dylib}` via `ctypes.CDLL`,
- exposes the version / status / dtype / backend introspection queries,
- exposes a Pythonic `Context` and `Config` wrapper for the lifecycle calls,
- raises a typed exception (`Pls4allError`) on any non-OK status, capturing the per-context `last_error` message.

A real wheel with NumPy zero-copy `p4a_matrix_view_t` conversion + a
sklearn-compatible estimator lands in Phase 2 on top of the live C core.

## Build the underlying library

```bash
cmake --preset dev-release
cmake --build --preset dev-release --parallel
```

This produces `build/dev-release/cpp/src/libp4a.so` (or `.dylib` / `.dll`).
The binding looks for it on disk via the loader rules below.

## Loader rules

In order:

1. `$PLS4ALL_LIB_PATH` — explicit path to `libp4a` (most direct).
2. `pls4all/lib/libp4a*` next to the installed package (wheel layout).
3. `<repo-root>/build/dev-release/cpp/src/libp4a*` (developer convenience).
4. The standard system search path (`LD_LIBRARY_PATH`, macOS rpath, Windows `PATH`).

## Smoke test

```python
import pls4all
print(pls4all.version())       # "0.87.0+abi.1.13.0"
print(pls4all.abi_version())   # (1, 13, 0)

with pls4all.Context() as ctx:
    ctx.seed = 42
    print(ctx.last_error)      # ""
    try:
        ctx.backend = pls4all.Backend.CUDA
    except pls4all.Pls4allError as e:
        print(e)               # 'backend 5 is not compiled into this build of libp4a'

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
