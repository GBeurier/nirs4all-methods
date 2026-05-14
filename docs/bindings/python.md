# Python binding

Phase 0 ships a ctypes-only wrapper that exposes ABI introspection and the
context / config lifecycles. See [`bindings/python/README.md`](../../bindings/python/README.md) for installation and loader rules.

## Hello-version

```python
import pls4all

print(pls4all.version())          # "0.1.0+abi.1.0.0"
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
```

Phase 2 expands the surface to a full sklearn-compatible estimator with
zero-copy NumPy `p4a_matrix_view_t` round-trips.
