# Rust binding

`bindings/rust/n4m` is the official local pre-release Rust binding for the
stable `libn4m` C ABI. It is a thin ownership and serialization layer: all
numerical fitting, prediction, and optimizer work remains in the native
library.

The crate supports C-ABI model fit/predict, N4MM/N4MOPT serialization, native
optimizer HPO, and selection-only `n4m_finetune_estimator`. It requires a
prebuilt shared `libn4m`; `N4M_LIB_DIR` identifies its directory. Build and
test it with:

```sh
cmake --preset dev-debug && cmake --build --preset dev-debug --parallel
N4M_LIB_DIR="$PWD/build/dev-debug/cpp/src" \
N4M_RUNTIME_RPATH="$PWD/build/dev-debug/cpp/src" \
cargo test -p n4m
```

The build script compiles a C11 ABI probe against the public headers, including
an explicit `/std:c11` mode on MSVC. CI exercises the binding on Linux, macOS,
and Windows, with additional Linux sanitizer and declared-MSRV checks.

This is not a crates.io release: publication is intentionally deferred, no
crates.io token is needed, and no cross-language Rust prediction fixture is
committed. Therefore the binding makes no prediction-parity claim. See the
[crate README](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/rust/n4m/README.md)
and the
[distribution policy](https://github.com/GBeurier/nirs4all-methods/blob/main/docs/dev/DISTRIBUTION.md)
for operational details.
