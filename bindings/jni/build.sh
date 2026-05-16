#!/usr/bin/env bash
# SPDX-License-Identifier: CeCILL-2.1
#
# Build script for the pls4all JNI desktop binding. Compiles
# Pls4all.java, generates the JNI header, and builds libp4a_jni.so
# against the libp4a artifact produced by the CMake "dev-release"
# preset. Used by the parity test and by downstream JVM consumers.
#
# Environment overrides (all optional):
#   JAVA_HOME            JDK root (must contain bin/javac, include/jni.h)
#   PLS4ALL_INCLUDE_DIR  cpp/include (defaults to repo cpp/include)
#   PLS4ALL_GEN_DIR      generated headers (defaults to build/dev-release/generated)
#   PLS4ALL_LIB_DIR      libp4a search path (defaults to build/dev-release/cpp/src)
#   CC                   C compiler (defaults to /usr/bin/gcc)
#
# Outputs to bindings/jni/build/ — gitignored.

set -euo pipefail

cd "$(dirname "$0")/../.."     # repo root
REPO_ROOT="$(pwd)"

JAVA_HOME="${JAVA_HOME:-}"
if [[ -z "${JAVA_HOME}" ]]; then
    if command -v javac >/dev/null 2>&1; then
        JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v javac)")")")"
    else
        echo "error: javac not found; set JAVA_HOME or install a JDK" >&2
        exit 1
    fi
fi

# conda-forge openjdk places the JDK under <env>/lib/jvm rather than at the
# env root. If $JAVA_HOME/include/jni.h is missing but $JAVA_HOME/lib/jvm
# contains a JDK, prefer that one.
if [[ ! -f "${JAVA_HOME}/include/jni.h" && \
       -f "${JAVA_HOME}/lib/jvm/include/jni.h" ]]; then
    JAVA_HOME="${JAVA_HOME}/lib/jvm"
fi

PLS4ALL_INCLUDE_DIR="${PLS4ALL_INCLUDE_DIR:-${REPO_ROOT}/cpp/include}"
PLS4ALL_GEN_DIR="${PLS4ALL_GEN_DIR:-${REPO_ROOT}/build/dev-release/generated}"
PLS4ALL_LIB_DIR="${PLS4ALL_LIB_DIR:-${REPO_ROOT}/build/dev-release/cpp/src}"
CC="${CC:-/usr/bin/gcc}"

OUT_DIR="${REPO_ROOT}/bindings/jni/build"
mkdir -p "${OUT_DIR}/classes" "${REPO_ROOT}/bindings/jni/c"

echo "[1/3] javac Pls4all.java + JNI header"
"${JAVA_HOME}/bin/javac" -h "${REPO_ROOT}/bindings/jni/c" \
    -d "${OUT_DIR}/classes" \
    "${REPO_ROOT}/bindings/jni/java/io/github/pls4all/Pls4all.java"

echo "[2/3] gcc -shared libp4a_jni.so"
JNI_INCLUDE_LINUX="${JAVA_HOME}/include/linux"
if [[ "$(uname)" == "Darwin" ]]; then
    JNI_INCLUDE_LINUX="${JAVA_HOME}/include/darwin"
fi
"${CC}" -O2 -fPIC -shared -Wall -Wextra \
    -I"${JAVA_HOME}/include" -I"${JNI_INCLUDE_LINUX}" \
    -I"${PLS4ALL_INCLUDE_DIR}" -I"${PLS4ALL_GEN_DIR}" \
    -I"${REPO_ROOT}/bindings/jni/c" \
    "${REPO_ROOT}/bindings/jni/c/p4a_jni.c" \
    -L"${PLS4ALL_LIB_DIR}" -lp4a \
    -Wl,-rpath,"${PLS4ALL_LIB_DIR}" \
    -o "${OUT_DIR}/libp4a_jni.so"

echo "[3/3] javac TestParity.java"
"${JAVA_HOME}/bin/javac" -d "${OUT_DIR}/classes" \
    -cp "${OUT_DIR}/classes" \
    "${REPO_ROOT}/bindings/jni/test/TestParity.java"

echo
echo "Built ${OUT_DIR}/libp4a_jni.so and JVM classes."
echo "Run parity test:"
echo "  LD_LIBRARY_PATH=${PLS4ALL_LIB_DIR} \\"
echo "  PLS4ALL_JNI_LIB=${OUT_DIR}/libp4a_jni.so \\"
echo "  ${JAVA_HOME}/bin/java -Drepo.root=${REPO_ROOT} \\"
echo "    -cp ${OUT_DIR}/classes TestParity"
