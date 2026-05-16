# pls4all — Rust binding

Safe Rust wrappers around libp4a's `p4a_pls_fit_simple` C ABI helper.
Parity-gated against the shared cross-binding fixture
(`bindings/js/test/parity_fixture.json`) at machine epsilon
(`rmse_rel = 0.0` — bit-exact). Zero non-stdlib dependencies.

## Layout

```
bindings/rust/pls4all/
├── Cargo.toml
├── build.rs                       Tells cargo where libp4a.so lives
├── src/lib.rs                     extern "C" + safe wrappers
└── examples/test_parity.rs        Parity gate vs shared fixture
```

## Build

```bash
# From the repo root, after the CMake "dev-release" preset is built.
PLS4ALL_LIB_DIR=$(pwd)/build/dev-release/cpp/src \
LD_LIBRARY_PATH=$(pwd)/build/dev-release/cpp/src \
cargo build --manifest-path bindings/rust/pls4all/Cargo.toml --release
```

The `build.rs` script walks up from the manifest dir to find
`build/dev-release/cpp/src` automatically; setting
`PLS4ALL_LIB_DIR` is only required for out-of-tree builds.

## Parity gate

```bash
REPO_ROOT=$(pwd) \
PLS4ALL_LIB_DIR=$(pwd)/build/dev-release/cpp/src \
LD_LIBRARY_PATH=$(pwd)/build/dev-release/cpp/src \
cargo run --manifest-path bindings/rust/pls4all/Cargo.toml \
          --example test_parity --release
```

Expected output:

```
pls4all::version()       = 0.89.0+abi.1.13.0
pls4all::abi_version()   = 1.13.0
fixture pls4all_version  = 0.89.0+abi.1.13.0
rmse_rel coefficients    = 0.000e0
rmse_rel x_mean          = 0.000e0
rmse_rel y_mean          = 0.000e0
rmse_rel predictions     = 0.000e0
PARITY GATE PASS
```

## Usage

```rust
use pls4all::pls_fit;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 50;
    let p = 5;
    let q = 1;

    let mut x = vec![0.0; n * p];
    let mut y = vec![0.0; n * q];
    // ... fill x, y ...

    let fit = pls_fit(&x, &y, n, p, q, 3)?;
    println!("coefficients: {:?}", fit.coefficients);
    println!("predictions length = {}", fit.predictions.len());
    Ok(())
}
```

## Limitations

* Only the one-shot fit + in-sample prediction path is wrapped. A
  `predict(model, x)` against a saved model is on the backlog.
* The build depends on libp4a being present at link time (`-lp4a`).
  Cross-compilation needs the matching libp4a built for the target.
