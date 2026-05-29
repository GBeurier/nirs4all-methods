# pls4all

`pls4all` is the slim, PLS-only subset of **nirs4all-methods** — a thin Python
binding over the portable `libn4m` C ABI (a C++17 PLS / NIRS engine). The wheel
bundles the `libn4m` shared library, so `pip install pls4all` is self-contained;
no separate native build is required. For the full method surface (preprocessing,
selectors, diagnostics, augmenters, …) install the `nirs4all-methods` package
instead — both load the same `libn4m`.

The binding loads `libn4m` with `ctypes.CDLL` (so the GIL is released during
native calls) and exposes:

- `version()` / `abi_version()` introspection,
- a Pythonic `Context` and `Config` (RAII lifecycle wrappers),
- the PLS fit/predict surface and a scikit-learn-compatible
  `pls4all.sklearn.PLSRegression` (and the other PLS-family estimators),
- a typed `Pls4allError` raised on any non-OK status, carrying the
  context's `last_error` message.

## Quick start

```python
import numpy as np
import pls4all
from pls4all.sklearn import PLSRegression

print(pls4all.version())      # e.g. "0.98.0+abi.1.9.0"
print(pls4all.abi_version())  # (1, 9, 0)

rng = np.random.default_rng(0)
X = rng.standard_normal((40, 12))
y = X @ rng.standard_normal(12)

model = PLSRegression(n_components=5).fit(X, y)
print(model.predict(X).shape)  # (40,)
```

Low-level lifecycle, if you need it:

```python
with pls4all.Context() as ctx, pls4all.Config() as cfg:
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.SIMPLS
    cfg.n_components = 5
    # ... drive a fit through the C ABI ...
```

`scikit-learn` is an optional dependency (only `pls4all.sklearn` needs it); the
core `import pls4all` works with NumPy alone.

## Loading `libn4m`

The bundled wheel ships `libn4m` inside `pls4all/lib/`, found automatically. For
development against a local build the loader searches, in order:

1. `$PLS4ALL_LIB_PATH` — explicit path to `libn4m` (most direct),
2. `pls4all/lib/libn4m*` next to the installed package (wheel layout),
3. `<repo-root>/build/dev-release/cpp/src/libn4m*` (developer convenience),
4. the standard system search path (`LD_LIBRARY_PATH`, macOS rpath, Windows `PATH`).

## Building `libn4m` from source (developers)

```bash
cmake --preset dev-release
cmake --build --preset dev-release --parallel
```

This produces `build/dev-release/cpp/src/libn4m.so` (`.dylib` / `.dll` on
macOS / Windows), which the loader rules above pick up.

See <https://github.com/GBeurier/nirs4all-methods> for the full project.
