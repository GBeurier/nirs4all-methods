#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-2.1
#
# cibuildwheel BEFORE_ALL hook: build libn4m once per wheel platform and stage
# the shared library into bindings/python/src/n4m/lib/ so the
# wheel bundles its own ABI dependency.
#
# Uses N4M_BUILD_PRESET when set, defaulting to dev-release.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

PRESET="${N4M_BUILD_PRESET:-dev-release}"
BUILD_DIR="build/${PRESET}"
LIB_DIR="bindings/python/src/n4m/lib"

echo "::group::ensure cmake + ninja are present"
PYTHON_BIN="${PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=python3
    else
        PYTHON_BIN=python
    fi
fi
"${PYTHON_BIN}" -m pip install --quiet --upgrade pip
"${PYTHON_BIN}" -m pip install --quiet "cmake>=3.21,<5" "ninja>=1.11"
echo "  cmake : $(command -v cmake) - $(cmake --version | head -n1)"
echo "  ninja : $(command -v ninja) - $(ninja --version)"
echo "::endgroup::"

echo "::group::nirs4all-methods libn4m build (preset=${PRESET})"
rm -rf "${BUILD_DIR}"
cmake --preset "${PRESET}" \
    -DN4M_BUILD_SHARED=ON \
    -DN4M_BUILD_STATIC=OFF \
    -DN4M_BUILD_CLI=OFF \
    -DN4M_BUILD_TESTS=OFF
cmake --build --preset "${PRESET}" --parallel --target n4m_c
echo "::endgroup::"

echo "::group::stage libn4m into ${LIB_DIR}"
rm -rf "${LIB_DIR}"
mkdir -p "${LIB_DIR}"

case "$(uname -s)" in
    Linux*)
        cp -a "${BUILD_DIR}/cpp/src/"libn4m.so* "${LIB_DIR}/"
        ;;
    Darwin*)
        cp -a "${BUILD_DIR}/cpp/src/"libn4m*.dylib "${LIB_DIR}/"
        ;;
    MINGW*|CYGWIN*|MSYS*)
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
        cp -a "${dll_paths[@]}" "${LIB_DIR}/"
        ;;
    *)
        echo "ERROR: unrecognised platform $(uname -s)" >&2
        exit 1
        ;;
esac

ls -la "${LIB_DIR}"
echo "::endgroup::"
