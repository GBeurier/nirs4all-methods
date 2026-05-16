# Phase 40 — Ruby (Fiddle) Binding

**Status**: shipped 2026-05-16 · tag `phase-40-ruby-binding` ·
release `0.91.0+abi.1.13.0`.

## Goal

Ship a Ruby binding using **only the standard library** (Fiddle).
Gated by the shared cross-binding parity fixture. Bit-exact agreement
with the native Python reference is the target.

## Why Fiddle (not the `ffi` gem)

* Fiddle has been in Ruby core since 1.9. No `gem install`, no
  `bundler`, no native-extension build, no `ruby-dev` headers.
  Anyone with a system Ruby (Debian, RHEL, macOS Apple Ruby) can
  consume the binding out of the box.
* The `ffi` gem (libffi-based) is more ergonomic but compiles a
  C extension at install time, which makes packaging fragile and
  excludes minimal-image deployments.

## Architecture

```
bindings/ruby/
├── lib/pls4all.rb         Public API (Fiddle DLL imports)
└── test/test_parity.rb    Parity gate
```

Public API (`module Pls4all`):
* `Pls4all.version -> String` — `"0.91.0+abi.1.13.0"`.
* `Pls4all.abi_version -> [Integer, Integer, Integer]` — `[1, 13, 0]`.
* `Pls4all.pls_fit(x, y, n, p, q, n_components) -> FitResult` — one-
  shot SIMPLS regression returning a Struct with the four output
  arrays (coefficients, x_mean, y_mean, predictions).

`Array#pack("E*")` / `String#unpack("E*")` marshal Ruby Float arrays
to/from packed IEEE 754 little-endian double buffers. This is the
canonical Ruby FFI pattern and matches the libp4a row-major C ABI
exactly.

## Parity gate

```
$ ruby bindings/ruby/test/test_parity.rb
Pls4all.version           = 0.91.0+abi.1.13.0
Pls4all.abi_version       = 1.13.0
fixture pls4all_version   = 0.91.0+abi.1.13.0
rmse_rel coefficients    = 0.000e+00
rmse_rel x_mean          = 0.000e+00
rmse_rel y_mean          = 0.000e+00
rmse_rel predictions     = 0.000e+00
PARITY GATE PASS
```

Same bit-exact result as JNI / Rust / .NET — Ruby `pack("E*")` writes
the exact same IEEE 754 byte sequence the C ABI consumes.

## Implementation notes

* **Library lookup precedence**: `PLS4ALL_LIB` env var, then standard
  loader names (libp4a.so / libp4a.dylib / p4a.dll), then a repo-local
  walk-up to find the CMake `dev-release` artifact. Mirrors the
  pattern used by the Python and Rust bindings.
* **`private_constant :Lib`**: the Fiddle::Importer module is internal.
  Public callers go through the module-level methods so we can change
  the underlying FFI mechanism (e.g. swap Fiddle for an ffi-gem path)
  without breaking consumers.

## ABI surface

No new symbols. Uses `p4a_pls_fit_simple` from ABI 1.13.0.

## Release

* Library version: `0.90.0` → `0.91.0` (additive: new binding).
* ABI version: unchanged at `1.13.0`.
* Tag: `phase-40-ruby-binding`.

## Backlog

* `Pls4all.predict(model, x)` against a saved model.
* RubyGems publishing — gated on us shipping prebuilt libp4a shared
  libraries for the three major platforms.
