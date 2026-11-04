#!/usr/bin/env bash
# Build-time toolchain only; the resulting wheel needs no Rust installation.
set -euo pipefail
if ! command -v rustup >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/n4m-rustup-init.sh
    sh /tmp/n4m-rustup-init.sh -y --profile minimal --default-toolchain 1.96.0
fi
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
rustup toolchain install 1.96.0 --profile minimal

if [[ "$(uname -s)" == Darwin* ]]; then
    rustup target add --toolchain 1.96.0 x86_64-apple-darwin aarch64-apple-darwin
fi
