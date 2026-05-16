# Phase 41 — Lua / LuaJIT Binding

**Status**: shipped 2026-05-16 · tag `phase-41-lua-binding` ·
release `0.92.0+abi.1.13.0`.

## Goal

Ship a Lua binding via LuaJIT's built-in FFI. Gated by the shared
cross-binding parity fixture. Bit-exact agreement with the native
Python reference is the target.

## Why LuaJIT FFI

* LuaJIT's `ffi` module is the most performant Lua FFI available and
  ships with the JIT — no LuaRocks, no `alien`, no `lua-ffi-libffi`.
  A LuaJIT binary that's 400 KB compressed gets us a fully working
  zero-rock binding.
* LuaJIT is the substrate behind OpenResty, Nginx Lua, neovim's
  built-in Lua, Torch's tensor layer, and many high-throughput
  data-plane services. Shipping a LuaJIT binding makes libp4a
  callable from each of these without adapter layers.

## Architecture

```
bindings/lua/
├── pls4all.lua            Public API (LuaJIT FFI)
└── test/test_parity.lua   Parity gate
```

Public API (`local pls4all = require("pls4all")`):
* `pls4all.version() -> string` — `"0.92.0+abi.1.13.0"`.
* `pls4all.abi_version() -> { number, number, number }` — `{1, 13, 0}`.
* `pls4all.pls_fit(x, y, n, p, q, n_components) -> table` —
  one-shot SIMPLS regression returning a table with `coefficients`,
  `x_mean`, `y_mean`, `predictions` as Lua sequences.

LuaJIT's `ffi.new("double[?]", N, src_table)` populates a C array
directly from a Lua sequence, and `buf[i]` reads back doubles. No
intermediate JSON, no packed-string trickery: this is the most
direct FFI hand-off in the binding set.

## Parity gate

```
$ luajit bindings/lua/test/test_parity.lua
pls4all.version           = 0.92.0+abi.1.13.0
pls4all.abi_version       = 1.13.0
fixture pls4all_version   = 0.92.0+abi.1.13.0
rmse_rel coefficients    = 0.000e+00
rmse_rel x_mean          = 0.000e+00
rmse_rel y_mean          = 0.000e+00
rmse_rel predictions     = 0.000e+00
PARITY GATE PASS
```

Same bit-exact result as JNI / Rust / .NET / Ruby. LuaJIT's `double`
maps directly onto the IEEE 754 C `double` the libp4a ABI expects.

## Implementation notes

* **Library lookup precedence**: `PLS4ALL_LIB` env, then standard
  loader names, then a walk-up from `debug.getinfo(1, "S").source`
  to the CMake `dev-release` artifact. Mirrors the Ruby / Python /
  Rust bindings.
* **LuaJIT requirement**: plain Lua 5.1 / 5.2 / 5.3 / 5.4 do not
  ship FFI. A port to plain Lua would use `lua-ffi-libffi` or
  `alien`. We document the LuaJIT-only path here and leave the
  plain-Lua port to the backlog.

## ABI surface

No new symbols. Uses `p4a_pls_fit_simple` from ABI 1.13.0.

## Release

* Library version: `0.91.0` → `0.92.0` (additive: new binding).
* ABI version: unchanged at `1.13.0`.
* Tag: `phase-41-lua-binding`.

## Backlog

* Plain-Lua port via `lua-ffi-libffi` (no FFI in Lua core, but the
  third-party rock works without LuaJIT).
* `pls4all.predict(model, x)` against a saved model.
* LuaRocks publishing once we ship prebuilt libp4a libraries.
