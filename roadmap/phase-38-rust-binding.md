# Phase 38 — Rust Binding

**Status**: shipped 2026-05-16 · tag `phase-38-rust-binding` ·
release `0.89.0+abi.1.13.0`.

## Goal

Ship a Rust crate for libp4a backed by `p4a_pls_fit_simple`, gated
by the shared cross-binding parity fixture. Bit-exact agreement with
the native Python reference is the target.

## Why Rust

* Rust is the modern systems-programming default. A Rust binding
  unlocks libp4a inside services using Tokio / Axum / Actix-Web,
  WASM compile targets (via `wasm-bindgen`, distinct from Phase 32's
  Emscripten WASM), and Rust-based ML tooling (Burn, Candle, tch-rs).
* `extern "C"` is built into the language. We hand-roll the FFI
  block to avoid pulling in `bindgen` as a build-time dependency.

## Architecture

```
bindings/rust/pls4all/
├── Cargo.toml          rlib crate, no deps
├── build.rs            Locates libp4a (auto-walks repo, env override)
├── src/lib.rs          extern "C" + safe wrappers + FitError
└── examples/test_parity.rs    Parity gate
```

Public API:
* `pls4all::version() -> String` — `"0.89.0+abi.1.13.0"`.
* `pls4all::abi_version() -> (u32, u32, u32)` — `(1, 13, 0)`.
* `pls4all::pls_fit(&[f64], &[f64], n, p, q, k) -> Result<FitResult, FitError>` —
  one-shot SIMPLS regression. `FitResult` owns four `Vec<f64>`
  outputs (coefficients, x_mean, y_mean, predictions).

The wrappers use `#![deny(unsafe_op_in_unsafe_fn)]` to force each
unsafe block to document its invariants in-line. The two unsafe
blocks (CStr conversion in `version()` and the FFI call in
`pls_fit`) each carry a SAFETY comment explaining why the
contract holds.

## Parity gate

```
$ cargo run --release --example test_parity
pls4all::version()       = 0.89.0+abi.1.13.0
pls4all::abi_version()   = 1.13.0
fixture pls4all_version  = 0.89.0+abi.1.13.0
rmse_rel coefficients    = 0.000e0
rmse_rel x_mean          = 0.000e0
rmse_rel y_mean          = 0.000e0
rmse_rel predictions     = 0.000e0
PARITY GATE PASS
```

All four arrays are **bit-exact** vs the native Python reference,
matching JNI's result. Both Rust `Vec<f64>` and Java `double[]` are
row-major IEEE 754 doubles with the same layout as the C ABI;
nothing is copied or transposed.

## Implementation notes

* **Hand-rolled FFI vs bindgen**: with only four exported symbols
  in use, hand-rolling the `extern "C"` block is faster to read,
  free of build-time codegen, and keeps the crate dependency-free.
  If we expose more of libp4a later we'll revisit.
* **`#[link(name = "p4a")]` + `build.rs`**: the build script adds
  the `-L` search path; the attribute provides the link name. This
  avoids hard-coding the lib directory in `Cargo.toml`.
* **rpath baking**: `build.rs` emits a `-Wl,-rpath,...` cargo
  `rustc-link-arg` so the example binary loads `libp4a.so` without
  `LD_LIBRARY_PATH`. Downstream packagers override with
  `PLS4ALL_RUNTIME_RPATH`.
* **Zero-dep error type**: the `FitError` enum implements `Display`
  and `std::error::Error` by hand rather than via `thiserror`,
  keeping the crate dependency tree empty.

## ABI surface

No new symbols. Uses `p4a_pls_fit_simple` from ABI 1.13.0.

## Release

* Library version: `0.88.0` → `0.89.0` (additive: new binding).
* ABI version: unchanged at `1.13.0`.
* Tag: `phase-38-rust-binding`.

## Backlog

* `pls4all::predict(model, x)` against a saved model.
* WASM via `wasm-bindgen` for browser-side Rust → libp4a (distinct
  from Phase 32's Emscripten path which compiles libp4a directly).
* Publish to crates.io once the API stabilises.
