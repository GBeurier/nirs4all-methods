#!/usr/bin/env bash
# pls4all.sklearn parity gate
# ===========================
#
# The sklearn tier-2 wrappers are validated via bit-exact comparison
# against the tier-1 ctypes path: every wrapper's `.fit(X, y).predict(X)`
# must equal the raw tier-1 fit+predict to within `np.array_equal`.
#
# This script runs the wrapper-vs-tier1 tests in the bindings/python
# test suite and prints a one-line pass/fail summary.
#
# Usage from the repo root:
#
#     bindings/python/scripts/sklearn_parity_gate.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

LIB="${N4M_LIB_PATH:-$ROOT/build/dev-release/cpp/src/libn4m.so}"
PY="${PY:-$ROOT/parity/python_generator/.venv/bin/python}"
OCT_HOME="${OCTAVE_HOME:-/home/delete/miniconda3/envs/pls4all_r}"

if [ ! -e "$LIB" ]; then
    echo "ERROR: libn4m not found at $LIB" >&2
    echo "Hint: build with 'cmake --build build/dev-release' first." >&2
    exit 2
fi
if [ ! -x "$PY" ]; then
    echo "ERROR: Python interpreter not found at $PY" >&2
    exit 2
fi

N4M_LIB_PATH="$LIB" \
OCTAVE_HOME="$OCT_HOME" \
LD_LIBRARY_PATH="$OCT_HOME/lib:${LD_LIBRARY_PATH:-}" \
OCTAVE_EXECUTABLE="$OCT_HOME/bin/octave-cli" \
PATH="$OCT_HOME/bin:$PATH" \
PYTHONPATH="$ROOT/bindings/python/src" \
    "$PY" -m pytest bindings/python/tests/ \
        -k "bit_exact or pickle_bitexact or wrapper_bitexact or matches_in_sample or preserves_state" \
        --tb=line -q
