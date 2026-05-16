# pls4all — Nim binding

Nim wrapper around libp4a's `p4a_pls_fit_simple` C ABI helper. Uses
Nim's built-in `{.importc, dynlib.}` pragmas — no nimble packages,
no extra deps. Parity-gated against the shared cross-binding fixture
(`bindings/js/test/parity_fixture.json`) at machine epsilon
(`rmse_rel = 0.0` — bit-exact).

## Layout

```
bindings/nim/
├── pls4all.nim            Public API (dynlib pragmas)
└── test/test_parity.nim   Parity gate vs shared fixture
```

## Build & run

```bash
# Compile and run the parity gate in one step.
LD_LIBRARY_PATH=$(pwd)/build/dev-release/cpp/src \
REPO_ROOT=$(pwd) \
nim c -r \
  -d:libp4aName=$(pwd)/build/dev-release/cpp/src/libp4a.so \
  --path:bindings/nim \
  --outdir:bindings/nim/test \
  bindings/nim/test/test_parity.nim
```

For consumers, importing the binding is a single line:

```nim
import pls4all

echo pls4all.version()                 # "0.93.0+abi.1.13.0"

let n = 50; let p = 5; let q = 1
var x = newSeq[float64](n * p)
var y = newSeq[float64](n * q)
# fill x, y ...

let fit = pls4all.plsFit(x, y, n, p, q, 3)
echo fit.coefficients
```

## Parity gate output

```
pls4all.version           = 0.93.0+abi.1.13.0
pls4all.abiVersion        = 1.13.0
fixture pls4all_version   = 0.93.0+abi.1.13.0
rmse_rel coefficients    = 0.000e+00
rmse_rel x_mean          = 0.000e+00
rmse_rel y_mean          = 0.000e+00
rmse_rel predictions     = 0.000e+00
PARITY GATE PASS
```

## Library lookup

The `{.dynlib.}` pragma is resolved at runtime by the platform loader
(dlopen on Linux/macOS, LoadLibrary on Windows). The default name is
`libp4a.so`; override at compile time with
`-d:libp4aName=/abs/path/libp4a.so.0.93.0` if you need a specific
artifact. `LD_LIBRARY_PATH` works the standard way.
