#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-2.1
#
# Local release gate for the native libn4m ABI surface. This complements
# regen_abi_snapshots.sh --check with loadability, symbol-prefix, SONAME/RPATH,
# and forbidden-runtime-dependency checks.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB=""
PRESET=""
SELF_TEST_SELECTION=0
SELF_TEST_ROOT=""

usage() {
  cat <<'EOF'
Usage: scripts/check_native_abi_freshness.sh [--lib PATH] [--preset PRESET]
       scripts/check_native_abi_freshness.sh --self-test-selection

Checks that the built native libn4m matches the committed ABI snapshot and is
redistributable:
  - exported symbols equal cpp/abi/expected_symbols_<platform>.txt
  - no exported symbol escapes the n4m_ / N4M_<major> namespace
  - n4m_check_abi_compatibility accepts the current headers
  - Linux SONAME equals libn4m.so.<ABI major>
  - no RPATH/RUNPATH and no forbidden runtime dependencies

Without --lib, discovery selects the full ABI version declared by the current
headers and refuses ambiguous matches. Use --preset to constrain the build root.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lib) LIB="$2"; shift 2 ;;
    --preset) PRESET="$2"; shift 2 ;;
    --self-test-selection) SELF_TEST_SELECTION=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

abi_major="$(
  awk '/^#define N4M_ABI_VERSION_MAJOR / {print $3}' \
    "$REPO_ROOT/cpp/include/n4m/n4m_version.h"
)"
abi_minor="$(
  awk '/^#define N4M_ABI_VERSION_MINOR / {print $3}' \
    "$REPO_ROOT/cpp/include/n4m/n4m_version.h"
)"
abi_patch="$(
  awk '/^#define N4M_ABI_VERSION_PATCH / {print $3}' \
    "$REPO_ROOT/cpp/include/n4m/n4m_version.h"
)"

if [[ -z "$abi_major" || -z "$abi_minor" || -z "$abi_patch" ]]; then
  echo "ERROR: could not parse ABI version from n4m_version.h" >&2
  exit 1
fi

find_current_lib() {
  local root="$1"
  local matches=""
  local match_count="0"
  local expected=""

  [[ -d "$root" ]] || return 0

  case "$(uname -s)" in
    Linux)
      expected="libn4m.so.$abi_major.$abi_minor.$abi_patch"
      matches="$(find "$root" -type f -name "$expected" 2>/dev/null | LC_ALL=C sort)"
      ;;
    Darwin)
      expected="libn4m.$abi_major.$abi_minor.$abi_patch.dylib"
      matches="$(find "$root" -type f -name "$expected" 2>/dev/null | LC_ALL=C sort)"
      ;;
    *)
      expected="n4m.dll or libn4m.dll"
      matches="$(
        find "$root" -type f \( -name 'n4m.dll' -o -name 'libn4m.dll' \) \
          2>/dev/null | LC_ALL=C sort
      )"
      ;;
  esac

  match_count="$(printf '%s\n' "$matches" | sed '/^$/d' | wc -l | tr -d '[:space:]')"
  if [[ "$match_count" -gt 1 ]]; then
    echo "ERROR: multiple current libn4m candidates found under $root:" >&2
    printf '%s\n' "$matches" | sed 's/^/  /' >&2
    echo "Pass --preset or --lib to select one explicitly." >&2
    return 3
  fi
  if [[ "$match_count" -eq 1 ]]; then
    printf '%s\n' "$matches"
    return 0
  fi

  echo "ERROR: expected current ABI library '$expected' under $root, but none was found." >&2
  return 2
}

test_current_library_selection() {
  local current_name=""
  local stale_name=""
  local selected=""

  case "$(uname -s)" in
    Linux)
      current_name="libn4m.so.$abi_major.$abi_minor.$abi_patch"
      stale_name="libn4m.so.0.0.0"
      ;;
    Darwin)
      current_name="libn4m.$abi_major.$abi_minor.$abi_patch.dylib"
      stale_name="libn4m.0.0.0.dylib"
      ;;
    *)
      echo "Selection self-test skipped: native freshness gate supports Linux/macOS."
      return 0
      ;;
  esac

  SELF_TEST_ROOT="$(mktemp -d)"
  trap 'rm -rf -- "$SELF_TEST_ROOT"' EXIT
  mkdir -p "$SELF_TEST_ROOT/primary" "$SELF_TEST_ROOT/duplicate"
  : > "$SELF_TEST_ROOT/primary/$stale_name"
  : > "$SELF_TEST_ROOT/primary/$current_name"

  selected="$(find_current_lib "$SELF_TEST_ROOT/primary")"
  if [[ "$selected" != "$SELF_TEST_ROOT/primary/$current_name" ]]; then
    echo "ERROR: selection self-test chose '$selected' instead of the current ABI." >&2
    return 1
  fi

  : > "$SELF_TEST_ROOT/duplicate/$current_name"
  if find_current_lib "$SELF_TEST_ROOT" >/dev/null 2>&1; then
    echo "ERROR: selection self-test accepted ambiguous current libraries." >&2
    return 1
  fi

  rm -rf -- "$SELF_TEST_ROOT"
  SELF_TEST_ROOT=""
  trap - EXIT
  echo "Native ABI library selection self-test OK."
}

if [[ "$SELF_TEST_SELECTION" -eq 1 ]]; then
  test_current_library_selection
  exit 0
fi

if [[ -z "$LIB" ]]; then
  if [[ -n "$PRESET" ]]; then
    LIB="$(find_current_lib "$REPO_ROOT/build/$PRESET/cpp/src")"
  else
    LIB="$(find_current_lib "$REPO_ROOT/build")"
  fi
fi

if [[ -z "$LIB" || ! -f "$LIB" ]]; then
  echo "ERROR: no built libn4m found. Build n4m_c first or pass --lib." >&2
  exit 2
fi

LIB="$(cd "$(dirname "$LIB")" && pwd)/$(basename "$LIB")"
echo "Inspecting $LIB"

"$REPO_ROOT/scripts/regen_abi_snapshots.sh" --check --lib "$LIB"

tmp_all="$(mktemp)"
trap 'rm -f "$tmp_all"' EXIT

case "$(uname -s)" in
  Linux)
    nm -D --defined-only "$LIB" \
      | awk '{print $3}' \
      | sed 's/@@.*//' \
      | LC_ALL=C sort -u > "$tmp_all"
    ;;
  Darwin)
    nm -gU "$LIB" \
      | awk '{print $3}' \
      | sed 's/^_//' \
      | LC_ALL=C sort -u > "$tmp_all"
    ;;
  *)
    echo "ERROR: native ABI freshness script currently supports Linux/macOS; use abi-check.yml for Windows." >&2
    exit 2
    ;;
esac

if grep -vE '^n4m_|^N4M_[0-9]+$' "$tmp_all"; then
  echo "ERROR: forbidden non-n4m_ symbol exported" >&2
  exit 1
fi

python3 - "$LIB" "$abi_major" "$abi_minor" <<'PY'
import ctypes
import sys

lib_path, major, minor = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
lib = ctypes.CDLL(lib_path)
check = lib.n4m_check_abi_compatibility
check.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
check.restype = ctypes.c_int
status = int(check(major, minor))
if status != 0:
    raise SystemExit(
        f"ERROR: n4m_check_abi_compatibility rejected header ABI {major}.{minor}.x "
        f"with status {status}"
    )
print(f"ABI compatibility OK for header {major}.{minor}.x")
PY

forbidden='libopenblas|libmkl|libpython|libR\.|librcpp|libboost|libeigen|libpybind11|libembind|libnlohmann|libyaml-cpp'

case "$(uname -s)" in
  Linux)
    dyn="$(readelf -d "$LIB")"
    echo "$dyn" | grep -E 'SONAME|NEEDED|RPATH|RUNPATH' || true
    soname="$(echo "$dyn" | sed -n 's/.*SONAME.*\[\(.*\)\]/\1/p')"
    expected_soname="libn4m.so.$abi_major"
    if [[ "$soname" != "$expected_soname" ]]; then
      echo "ERROR: unexpected SONAME '$soname' (expected $expected_soname)" >&2
      exit 1
    fi
    if echo "$dyn" | grep -qE 'RPATH|RUNPATH'; then
      echo "ERROR: libn4m carries an RPATH/RUNPATH" >&2
      exit 1
    fi
    deps="$(ldd "$LIB" 2>&1)"
    echo "$deps"
    if echo "$deps" | grep -iE "$forbidden"; then
      echo "ERROR: libn4m links a forbidden runtime dependency" >&2
      exit 1
    fi
    ;;
  Darwin)
    deps="$(otool -L "$LIB")"
    echo "$deps"
    if otool -l "$LIB" | grep -q 'LC_RPATH'; then
      echo "ERROR: libn4m carries an LC_RPATH" >&2
      exit 1
    fi
    if echo "$deps" | grep -iE "$forbidden"; then
      echo "ERROR: libn4m links a forbidden runtime dependency" >&2
      exit 1
    fi
    ;;
esac

echo "Native ABI freshness OK."
