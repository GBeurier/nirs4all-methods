#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-2.1
#
# cibuildwheel BEFORE_ALL hook: build libc4a once per wheel platform and stage
# the shared library into bindings/python/src/chemometrics4all/lib/ so the
# wheel bundles its own ABI dependency.
#
# Uses CHEMOMETRICS4ALL_BUILD_PRESET when set, defaulting to dev-release.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

PRESET="${CHEMOMETRICS4ALL_BUILD_PRESET:-dev-release}"
BUILD_DIR="build/${PRESET}"
LIB_DIR="bindings/python/src/chemometrics4all/lib"

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

echo "::group::chemometrics4all libc4a build (preset=${PRESET})"
rm -rf "${BUILD_DIR}"
cmake --preset "${PRESET}" \
    -DCHEMOMETRICS4ALL_BUILD_SHARED=ON \
    -DCHEMOMETRICS4ALL_BUILD_STATIC=OFF \
    -DCHEMOMETRICS4ALL_BUILD_CLI=OFF \
    -DCHEMOMETRICS4ALL_BUILD_TESTS=OFF
cmake --build --preset "${PRESET}" --parallel --target chemometrics4all_c
echo "::endgroup::"

echo "::group::stage libc4a into ${LIB_DIR}"
rm -rf "${LIB_DIR}"
mkdir -p "${LIB_DIR}"

case "$(uname -s)" in
    Linux*)
        cp -a "${BUILD_DIR}/cpp/src/"libc4a.so* "${LIB_DIR}/"
        ;;
    Darwin*)
        cp -a "${BUILD_DIR}/cpp/src/"libc4a*.dylib "${LIB_DIR}/"
        ;;
    MINGW*|CYGWIN*|MSYS*)
        dll_paths=()
        for candidate in \
            "${BUILD_DIR}/cpp/src/Release/c4a.dll" \
            "${BUILD_DIR}/cpp/src/c4a.dll"; do
            [[ -f "${candidate}" ]] && dll_paths+=("${candidate}")
        done
        if [[ ${#dll_paths[@]} -eq 0 ]]; then
            echo "ERROR: c4a.dll not found in ${BUILD_DIR}/cpp/src or Release/" >&2
            find "${BUILD_DIR}/cpp/src" \( -name 'c4a*' -o -name 'libc4a*' \) -ls || true
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
