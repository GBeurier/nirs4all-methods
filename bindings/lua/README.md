# pls4all — Lua / LuaJIT binding

Lua wrapper around libp4a's `p4a_pls_fit_simple` C ABI helper. Uses
**LuaJIT's built-in FFI** library — no LuaRocks required. Parity-
gated against the shared cross-binding fixture
(`bindings/js/test/parity_fixture.json`) at machine epsilon
(`rmse_rel = 0.0` — bit-exact).

LuaJIT only. Plain Lua 5.x does not ship with FFI; ports to plain
Lua would use the `alien` or `ffi` rock, but we ship the canonical
zero-dep LuaJIT path here.

## Layout

```
bindings/lua/
├── pls4all.lua            Public API (LuaJIT FFI)
└── test/test_parity.lua   Parity gate vs shared fixture
```

## Usage

```lua
package.path = "bindings/lua/?.lua;" .. package.path
local pls4all = require("pls4all")

print(pls4all.version())  -- "0.92.0+abi.1.13.0"

local n, p, q = 50, 5, 1
local x = {}
local y = {}
-- fill x (length n*p, row-major), y (length n*q, row-major) ...

local fit = pls4all.pls_fit(x, y, n, p, q, 3)
print("coefficients[1]:", fit.coefficients[1])
```

## Parity gate

```bash
PLS4ALL_LIB=$(pwd)/build/dev-release/cpp/src/libp4a.so \
LD_LIBRARY_PATH=$(pwd)/build/dev-release/cpp/src \
REPO_ROOT=$(pwd) \
luajit bindings/lua/test/test_parity.lua
```

Expected output:

```
pls4all.version           = 0.94.0+abi.1.13.0
pls4all.abi_version       = 1.13.0
fixture pls4all_version   = 0.94.0+abi.1.13.0
rmse_rel coefficients    = 0.000e+00
rmse_rel x_mean          = 0.000e+00
rmse_rel y_mean          = 0.000e+00
rmse_rel predictions     = 0.000e+00
PARITY GATE PASS
```

## Library lookup

`pls4all.lua` searches for libp4a in this order:
1. `PLS4ALL_LIB` env var (absolute path).
2. Standard loader names: `p4a`, `libp4a.so`, `libp4a.dylib`.
3. Walk-up from the source file to the repo-local CMake `dev-release`
   artifact.

Set `PLS4ALL_LIB` if your libp4a lives outside the standard search
path (e.g. in an embedded or custom install).
