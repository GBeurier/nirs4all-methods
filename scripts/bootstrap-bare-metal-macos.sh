#!/usr/bin/env bash
#
# scripts/bootstrap-bare-metal-macos.sh — best-effort installer for the
# n4m dev toolchain on macOS via Homebrew.

set -euo pipefail

log() { printf '\033[1;36m[bootstrap]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[bootstrap]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[bootstrap]\033[0m %s\n' "$*"; exit 1; }

if [[ "$(uname -s)" != "Darwin" ]]; then
    die "this script is for macOS only"
fi
if ! command -v brew >/dev/null 2>&1; then
    log "Installing Homebrew"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

log "Updating brew and installing base toolchain"
brew update
# Apple Clang ships with Xcode; the brewed llvm is optional. CMake / Ninja /
# OpenBLAS / Python / R / Octave / Node are all formulae.
brew install \
    cmake ninja pkg-config \
    openblas lapack \
    python@3.12 uv \
    r octave \
    node@22 \
    docker docker-compose \
    coreutils ripgrep fd jq tree

# Symlink the brewed openblas headers somewhere find_package(BLAS) will see them.
BREW_PREFIX="$(brew --prefix)"
export OPENBLAS_DIR="${BREW_PREFIX}/opt/openblas"
log "OpenBLAS at ${OPENBLAS_DIR}"
log "Add to your shell rc:  export OPENBLAS_DIR=${OPENBLAS_DIR}"

# Emscripten
if ! command -v emcc >/dev/null 2>&1; then
    log "Installing Emscripten (best effort)"
    brew install emscripten || warn "brew emscripten failed; you can install manually if you touch bindings/js"
fi

log "Done. Run \`make doctor\` to verify everything resolves."
