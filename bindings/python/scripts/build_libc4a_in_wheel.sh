#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-2.1
#
# cibuildwheel BEFORE_ALL hook: build libc4a once per wheel platform and copy
# the shared library into the bindings/python/src/chemometrics4all/lib/
# directory so the wheel bundles its own ABI dependency.
#
# Usage: pass the repo root as $1, or rely on CIBW_PROJECT_ROOT.
set -euo pipefail

ROOT="${1:-${CIBW_PROJECT_ROOT:-$(pwd)}}"
BUILD_DIR="${ROOT}/build/wheel"
LIB_DIR="${ROOT}/bindings/python/src/chemometrics4all/lib"

mkdir -p "${LIB_DIR}"

cmake -S "${ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCHEMOMETRICS4ALL_BUILD_SHARED=ON \
    -DCHEMOMETRICS4ALL_BUILD_STATIC=OFF \
    -DCHEMOMETRICS4ALL_BUILD_CLI=OFF \
    -DCHEMOMETRICS4ALL_BUILD_TESTS=OFF
cmake --build "${BUILD_DIR}" --target chemometrics4all_c --parallel

case "$(uname -s)" in
    Linux*)
        cp "${BUILD_DIR}/cpp/src/libc4a.so."*.* "${LIB_DIR}/"
        ;;
    Darwin*)
        cp "${BUILD_DIR}/cpp/src/libc4a."*.dylib "${LIB_DIR}/"
        ;;
    *)
        cp "${BUILD_DIR}/cpp/src/c4a.dll" "${LIB_DIR}/"
        ;;
esac
echo "Built and copied libc4a into ${LIB_DIR}"
