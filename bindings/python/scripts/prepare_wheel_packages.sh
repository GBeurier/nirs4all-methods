#!/usr/bin/env bash
# Prepare the two PyPI package dirs for cibuildwheel (release-wheels.yml,
# CIBW_BEFORE_ALL). Orchestrates the three verified steps:
#   1. build libn4m + stage it into bindings/python/src/{pls4all,n4m}/lib
#      (scripts/build_libn4m_in_wheel.sh);
#   2. generate the self-contained per-package dirs
#      (scripts/make_python_package.py --all -> bindings/python_{nirs4all_methods,pls4all});
#   3. mirror the staged libn4m into each generated package's module lib/.
# After this, `CIBW_PACKAGE_DIR=bindings/python_<pkg>` builds a complete wheel.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

# Windows runners (setup-python) provide `python`, not `python3`; Linux/macOS
# provide `python3`. Resolve whichever exists.
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"

bash bindings/python/scripts/build_libn4m_in_wheel.sh
"${PYTHON_BIN}" bindings/python/scripts/make_python_package.py --all

stage() {  # <module> <generated-dir>
    mkdir -p "bindings/$2/src/$1/lib"
    cp -a "bindings/python/src/$1/lib/." "bindings/$2/src/$1/lib/"
}
stage n4m     python_nirs4all_methods
stage pls4all python_pls4all

echo "prepared dual packages:"
ls -la bindings/python_nirs4all_methods/src/n4m/lib bindings/python_pls4all/src/pls4all/lib
