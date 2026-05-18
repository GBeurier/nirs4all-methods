#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-2.1
#
# Build libp4a + stage it inside the Python source tree so the wheel ships
# it as package data. Invoked by cibuildwheel via:
#
#   CIBW_BEFORE_ALL_LINUX:   bash bindings/python/scripts/build_libp4a_in_wheel.sh
#   CIBW_BEFORE_ALL_MACOS:   bash bindings/python/scripts/build_libp4a_in_wheel.sh
#   CIBW_BEFORE_ALL_WINDOWS: bash bindings/python/scripts/build_libp4a_in_wheel.sh
#
# (cibuildwheel runs CIBW_BEFORE_ALL once per matrix entry, before any
# wheel build for that target. The CWD is the project root.)
#
# Outputs:
#   bindings/python/src/pls4all/lib/libp4a*  — the shared library + SONAME
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

echo "::group::pls4all libp4a build (preset=${PRESET})"

# Fresh build to avoid stale-cache surprises when cibuildwheel runs the
# script in a manylinux container that may already have a build directory
# mounted from an earlier matrix entry.
rm -rf "${BUILD_DIR}"

cmake --preset "${PRESET}"
cmake --build --preset "${PRESET}" --parallel --target pls4all_c

echo "::endgroup::"

echo "::group::stage libp4a into ${SRC_LIB_DIR}"
rm -rf "${SRC_LIB_DIR}"
mkdir -p "${SRC_LIB_DIR}"

# Copy every libp4a* file produced by CMake — on Linux that's the SONAME
# symlinks (libp4a.so, libp4a.so.1) plus the real file
# (libp4a.so.1.16.0); on macOS it's libp4a.dylib + libp4a.1.dylib; on
# Windows it's p4a.dll (+ p4a.lib import lib, which we deliberately
# leave out of the wheel — Python loaders don't need the import lib).
case "$(uname -s)" in
    Linux*)
        # Preserve symlinks so the loader sees the conventional
        # libp4a.so -> libp4a.so.1 -> libp4a.so.1.16.0 chain.
        cp -a "${BUILD_DIR}/cpp/src/"libp4a.so*       "${SRC_LIB_DIR}/"
        ;;
    Darwin*)
        cp -a "${BUILD_DIR}/cpp/src/"libp4a*.dylib    "${SRC_LIB_DIR}/"
        ;;
    MINGW*|CYGWIN*|MSYS*)
        cp -a "${BUILD_DIR}/cpp/src/p4a.dll"          "${SRC_LIB_DIR}/"
        ;;
    *)
        echo "ERROR: unrecognised platform $(uname -s)" >&2
        exit 1
        ;;
esac

ls -la "${SRC_LIB_DIR}"
echo "::endgroup::"
