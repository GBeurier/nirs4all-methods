#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-2.1
#
# Build libn4m + stage it inside the Python source tree so the wheel ships
# it as package data. Invoked by cibuildwheel via:
#
#   CIBW_BEFORE_ALL_LINUX:   bash bindings/python/scripts/build_libn4m_in_wheel.sh
#   CIBW_BEFORE_ALL_MACOS:   bash bindings/python/scripts/build_libn4m_in_wheel.sh
#   CIBW_BEFORE_ALL_WINDOWS: bash bindings/python/scripts/build_libn4m_in_wheel.sh
#
# (cibuildwheel runs CIBW_BEFORE_ALL once per matrix entry, before any
# wheel build for that target. The CWD is the project root.)
#
# Outputs:
#   bindings/python/src/pls4all/lib/libn4m*  — the shared library + SONAME
#                                              chain produced by CMake.
# A pre-existing lib/ directory is wiped first to avoid leaking stale
# artifacts from a previous matrix entry.

set -euo pipefail

# Always operate from the project root regardless of how this script is
# invoked. cibuildwheel sets CWD to the project source, but a manual call
# from elsewhere should still work.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

PRESET="${PLS4ALL_BUILD_PRESET:-dev-release}"
BUILD_DIR="build/${PRESET}"
SRC_LIB_DIR="bindings/python/src/pls4all/lib"
# The bindings/python tree ships BOTH the slim `pls4all` module and the full
# `n4m` module; each loads libn4m from its own `<pkg>/lib/` (see n4m/_ffi.py and
# pls4all/_ffi.py). Stage the library into both so the full nirs4all-methods
# surface works in the wheel, not just pls4all.
N4M_LIB_DIR="bindings/python/src/n4m/lib"

echo "::group::ensure cmake + ninja are present"

# cibuildwheel runs the Linux build inside the manylinux container, which
# ships only Python + pip — no system cmake/ninja in the base image. macOS
# and Windows runners normally have them via brew / MSVC, but installing
# via pip is a no-op when the binaries already exist and gives us one
# deterministic source. The pip wheels for cmake/ninja install entry-point
# scripts onto PATH that `cmake --preset` picks up regardless of host.
# Windows runners (setup-python) provide `python`, not `python3`; Linux/macOS
# provide `python3`. Resolve whichever exists.
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
"${PYTHON_BIN}" -m pip install --quiet --upgrade pip
"${PYTHON_BIN}" -m pip install --quiet "cmake>=3.21,<5" "ninja>=1.11"
echo "  cmake : $(command -v cmake) — $(cmake --version | head -n1)"
echo "  ninja : $(command -v ninja) — $(ninja --version)"
echo "::endgroup::"

echo "::group::pls4all libn4m build (preset=${PRESET})"

# Fresh build to avoid stale-cache surprises when cibuildwheel runs the
# script in a manylinux container that may already have a build directory
# mounted from an earlier matrix entry.
rm -rf "${BUILD_DIR}"

cmake --preset "${PRESET}"
cmake --build --preset "${PRESET}" --parallel --target n4m_c

echo "::endgroup::"

echo "::group::stage libn4m into ${SRC_LIB_DIR}"
rm -rf "${SRC_LIB_DIR}"
mkdir -p "${SRC_LIB_DIR}"

# Copy every libn4m* file produced by CMake — on Linux that's the SONAME
# symlinks (libn4m.so, libn4m.so.1) plus the real file
# (libn4m.so.1.9.0); on macOS it's libn4m.dylib + libn4m.1.dylib; on
# Windows it's n4m.dll (+ n4m.lib import lib, which we deliberately
# leave out of the wheel — Python loaders don't need the import lib).
case "$(uname -s)" in
    Linux*)
        # Preserve symlinks so the loader sees the conventional
        # libn4m.so -> libn4m.so.1 -> libn4m.so.1.9.0 chain.
        cp -a "${BUILD_DIR}/cpp/src/"libn4m.so*       "${SRC_LIB_DIR}/"
        ;;
    Darwin*)
        cp -a "${BUILD_DIR}/cpp/src/"libn4m*.dylib    "${SRC_LIB_DIR}/"
        ;;
    MINGW*|CYGWIN*|MSYS*)
        # MSVC multi-config generators (Visual Studio 17 2022 for our CI
        # preset) drop binaries under cpp/src/Release/n4m.dll. Ninja
        # single-config builds drop them directly under cpp/src/. Check both.
        dll_paths=()
        for candidate in \
            "${BUILD_DIR}/cpp/src/Release/n4m.dll" \
            "${BUILD_DIR}/cpp/src/n4m.dll"; do
            [[ -f "${candidate}" ]] && dll_paths+=("${candidate}")
        done
        if [[ ${#dll_paths[@]} -eq 0 ]]; then
            echo "ERROR: n4m.dll not found in ${BUILD_DIR}/cpp/src or Release/" >&2
            find "${BUILD_DIR}/cpp/src" \( -name 'n4m*' -o -name 'libn4m*' \) -ls || true
            exit 1
        fi
        cp -a "${dll_paths[@]}" "${SRC_LIB_DIR}/"
        ;;
    *)
        echo "ERROR: unrecognised platform $(uname -s)" >&2
        exit 1
        ;;
esac

# Mirror the staged library into the full-package module's lib/ too.
rm -rf "${N4M_LIB_DIR}"
mkdir -p "${N4M_LIB_DIR}"
cp -a "${SRC_LIB_DIR}/"* "${N4M_LIB_DIR}/"

ls -la "${SRC_LIB_DIR}" "${N4M_LIB_DIR}"
echo "::endgroup::"
