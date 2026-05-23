#!/usr/bin/env bash
#
# devcontainer post-create hook.
#
# Installs the optional toolchains (Rust, Julia, .NET) that aren't baked
# into the base image to keep it small for contributors who don't need
# them. Skip silently when the source is unreachable so a flaky network
# doesn't break the container.

set -euo pipefail

log() { printf '\033[1;36m[post-create]\033[0m %s\n' "$*"; }

# rustup — quiet installer
if ! command -v rustc >/dev/null 2>&1; then
    log "Installing Rust (rustup)"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal \
        || log "rustup install failed (network?); skipping"
fi

# juliaup
if ! command -v julia >/dev/null 2>&1; then
    log "Installing Julia (juliaup)"
    curl -fsSL https://install.julialang.org \
        | sh -s -- -y --default-channel release \
        || log "juliaup install failed (network?); skipping"
fi

# Ensure R has the system libs the in-tree binding needs at install time.
log "Pre-warming R site library with common deps"
R --quiet --no-save <<'EOF' || true
opts <- options(install.packages.compile.from.source = "never")
installed <- rownames(installed.packages())
need <- setdiff(
    c("pls", "mdatools"),
    installed
)
if (length(need)) {
    install.packages(need, repos = "https://cloud.r-project.org")
}
options(opts)
EOF

log "post-create complete"
