# Phase 42 — Nim Binding

**Status**: shipped 2026-05-16 · tag `phase-42-nim-binding` ·
release `0.93.0+abi.1.13.0`.

## Goal

Ship a Nim binding via the language's built-in `{.importc, dynlib.}`
pragmas. Gated by the shared cross-binding parity fixture. Bit-exact
agreement with the native Python reference is the target.

## Why Nim

* Nim has the cleanest C ABI interop story of any modern compiled
  language — no codegen tool, no FFI runtime, just two pragmas on
  the proc declarations. The compiler emits dlopen / dlsym wiring
  automatically.
* Nim's `openArray[float64]` is a contiguous, length-aware slice
  type — handing its address to a C ABI is type-safe and zero-copy.
  Same property as Rust slices / Go slices / Span<double>.

## Architecture

```
bindings/nim/
├── pls4all.nim            Public API (dynlib pragmas)
└── test/test_parity.nim   Parity gate
```

Public API:
* `pls4all.version(): string` — `"0.93.0+abi.1.13.0"`.
* `pls4all.abiVersion(): tuple[major, minor, patch: int]` — `(1, 13, 0)`.
* `pls4all.plsFit(x, y: openArray[float64]; n, p, q, nComponents: int): FitResult` —
  one-shot SIMPLS regression returning a `FitResult` object with the
  four output `seq[float64]` arrays.

## Parity gate

```
$ nim c -r -d:libp4aName=$(pwd)/build/dev-release/cpp/src/libp4a.so \
        --path:bindings/nim \
        bindings/nim/test/test_parity.nim

pls4all.version           = 0.93.0+abi.1.13.0
pls4all.abiVersion        = 1.13.0
fixture pls4all_version   = 0.93.0+abi.1.13.0
rmse_rel coefficients    = 0.000e+00
rmse_rel x_mean          = 0.000e+00
rmse_rel y_mean          = 0.000e+00
rmse_rel predictions     = 0.000e+00
PARITY GATE PASS
```

Bit-exact, like every other slice-pointer binding (JNI / Rust / .NET /
Ruby / Lua).

## Implementation notes

* **Compile-time libp4a name**: Nim's `{.dynlib.}` pragma is resolved
  at runtime by the platform loader but the *name string* is a
  compile-time const. We expose `libp4aName` as a `{.strdefine.}` so
  consumers can override at compile time with
  `-d:libp4aName=/abs/path/libp4a.so.0.93.0`. The default
  `"libp4a.so"` defers to the dynamic-linker search path.
* **`openArray[float64]` → `ptr cdouble`**: `unsafeAddr x[0]` is the
  canonical Nim pattern; the slice is guaranteed contiguous. Same
  trick the rest of the bindings use.
* **`{.push.}` / `{.pop.}`**: groups the four FFI proc declarations
  under the same `importc, dynlib` pragmas without repeating them.

## ABI surface

No new symbols. Uses `p4a_pls_fit_simple` from ABI 1.13.0.

## Release

* Library version: `0.92.0` → `0.93.0` (additive: new binding).
* ABI version: unchanged at `1.13.0`.
* Tag: `phase-42-nim-binding`.

## Backlog

* `pls4all.predict(model, x)` against a saved model.
* Nimble package publishing once we ship prebuilt libp4a binaries.
