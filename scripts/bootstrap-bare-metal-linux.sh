#!/usr/bin/env bash
#
# scripts/bootstrap-bare-metal-linux.sh — best-effort installer for the
# n4m dev toolchain on Ubuntu 22.04 / 24.04. Use the devcontainer instead
# unless you specifically need a bare-metal setup.
#
# Installs: C++17 toolchain (gcc-12, clang-16, lld-16), CMake, Ninja,
# OpenBLAS, Python 3.12 + uv, R 4.4 + system deps, Octave 9, Node 22,
# and (best-effort) Emscripten 3.x. Rust / Julia / .NET are skipped —
# install them yourself if you need them.
#
# Idempotent: re-runnable. Defers to apt for the dpkg parts.

set -euo pipefail

log() { printf '\033[1;36m[bootstrap]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[bootstrap]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[bootstrap]\033[0m %s\n' "$*"; exit 1; }

if [[ "$(uname -s)" != "Linux" ]]; then
    die "this script is for Linux only — see bootstrap-bare-metal-{macos.sh,windows.ps1}"
fi
if ! command -v apt-get >/dev/null 2>&1; then
    die "apt-get not found — this script supports Debian/Ubuntu only. On Fedora/RHEL/Arch, install the same packages manually then run \`make doctor\`."
fi

if [[ "$(id -u)" -ne 0 ]] && ! command -v sudo >/dev/null 2>&1; then
    die "run as root or install sudo"
fi
SUDO=""
if [[ "$(id -u)" -ne 0 ]]; then SUDO="sudo"; fi

UBUNTU_VER="$(lsb_release -rs 2>/dev/null || echo unknown)"
log "Ubuntu version: ${UBUNTU_VER}"

# ---------------------------------------------------------------------------
# Base apt packages (matches .devcontainer/Dockerfile)
# ---------------------------------------------------------------------------
log "Updating apt and installing base toolchain"
${SUDO} apt-get update -qq
${SUDO} env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates curl wget gnupg lsb-release locales sudo \
    git jq tree less ripgrep fd-find \
    build-essential gcc-12 g++-12 \
    cmake ninja-build make pkg-config \
    libopenblas-dev liblapacke-dev gfortran \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    r-base r-base-dev libxml2-dev libssl-dev libcurl4-openssl-dev \
    libfontconfig1-dev libharfbuzz-dev libfribidi-dev \
    libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev \
    octave octave-statistics liboctave-dev \
    nodejs npm \
    docker.io

# clang-16 is in the 22.04 universe repo and 24.04 main — install if available
if apt-cache show clang-16 >/dev/null 2>&1; then
    ${SUDO} env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends clang-16 lld-16
else
    warn "clang-16 not in apt — falling back to gcc-12 only"
fi

${SUDO} update-alternatives --install /usr/bin/python python /usr/bin/python3.12 100 || true

# ---------------------------------------------------------------------------
# uv — Python package manager
# ---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    if [[ -f "${HOME}/.local/bin/uv" ]]; then
        ${SUDO} cp "${HOME}/.local/bin/uv" /usr/local/bin/uv
        ${SUDO} cp "${HOME}/.local/bin/uvx" /usr/local/bin/uvx 2>/dev/null || true
    fi
fi
uv --version

# ---------------------------------------------------------------------------
# Emscripten (optional — best effort)
# ---------------------------------------------------------------------------
EMSDK_DIR="${EMSDK:-/opt/emsdk}"
EMSDK_VER="${EMSDK_VERSION:-3.1.74}"
if [[ ! -d "${EMSDK_DIR}" ]]; then
    log "Installing Emscripten ${EMSDK_VER} into ${EMSDK_DIR} (best effort)"
    if ${SUDO} git clone --depth=1 https://github.com/emscripten-core/emsdk.git "${EMSDK_DIR}" 2>/dev/null; then
        (cd "${EMSDK_DIR}" && ${SUDO} ./emsdk install "${EMSDK_VER}" && ${SUDO} ./emsdk activate "${EMSDK_VER}") || warn "emsdk activation failed; check network"
    else
        warn "could not clone emsdk; skip — you can install later if you touch bindings/js"
    fi
fi

# ---------------------------------------------------------------------------
# Final check
# ---------------------------------------------------------------------------
log "Done. Run \`make doctor\` to verify everything resolves."
