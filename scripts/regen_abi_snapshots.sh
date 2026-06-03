#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-2.1
#
# Canonical regenerator for the per-platform ABI symbol snapshots in
# cpp/abi/expected_symbols_{linux,macos,windows}.txt — the surface the
# abi-check.yml workflow diffs against (fail-closed on every platform).
#
# Why one script: the three platform extractions are subtly different (Linux
# nm -D + version-script @@N4M_<n> tag; macOS nm -gU + leading-underscore strip;
# Windows dumpbin /EXPORTS). Hand-editing them caused the macOS/Windows files to
# silently drift to a wrong, truncated, @@-tagged format. Always regenerate via
# this script (or its Windows analogue) so the format CI diffs against is exact.
#
# The exported *symbol-name set* is identical across platforms by design (only
# N4M_API-decorated symbols are exported). The only legitimate cross-platform
# difference is the Linux-only `N4M_1` version-node line, which the macOS/Windows
# linkers do not emit. So macOS/Windows snapshots == the Linux n4m_* names with
# the `N4M_1` line removed; `--derive` produces them from the Linux file when you
# only have a Linux box.
#
# Usage:
#   scripts/regen_abi_snapshots.sh [--lib PATH] [--check] [--derive]
#
#   (no args)   Regenerate THIS platform's snapshot from a built libn4m.
#   --lib PATH  Use this library instead of auto-discovering under build/.
#   --check     Do not write; regenerate to a temp file and diff against the
#               committed snapshot. Exit 1 on any drift (CI/pre-commit use).
#   --derive    (Linux only) Also rewrite macos+windows snapshots from the
#               freshly regenerated linux snapshot (strip the N4M_1 line).
#
# All sorting is pinned to LC_ALL=C so the order is reproducible across hosts.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ABI_DIR="$REPO_ROOT/cpp/abi"
LIB=""
CHECK=0
DERIVE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lib) LIB="$2"; shift 2 ;;
    --check) CHECK=1; shift ;;
    --derive) DERIVE=1; shift ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

uname_s="$(uname -s)"
case "$uname_s" in
  Linux)  PLATFORM=linux;  PATTERN='libn4m.so.*' ;;
  Darwin) PLATFORM=macos;  PATTERN='libn4m*.dylib' ;;
  *) echo "Unsupported platform '$uname_s' (Windows uses the pwsh path in abi-check.yml)." >&2; exit 2 ;;
esac

if [[ -z "$LIB" ]]; then
  LIB="$(find "$REPO_ROOT/build" -name "$PATTERN" 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "$LIB" || ! -f "$LIB" ]]; then
  echo "No libn4m found (looked for '$PATTERN' under build/). Build n4m_c first or pass --lib." >&2
  exit 2
fi

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

if [[ "$PLATFORM" == linux ]]; then
  # Keep the N4M_1 version-node symbol; strip the per-symbol @@N4M_<n> tag.
  nm -D --defined-only "$LIB" | awk '{print $3}' \
    | grep '^n4m_\|^N4M_' | sed 's/@@.*//' | LC_ALL=C sort -u > "$TMP"
else  # macos
  nm -gU "$LIB" | awk '{print $3}' | sed 's/^_//' | LC_ALL=C sort -u > "$TMP"
fi

SNAP="$ABI_DIR/expected_symbols_${PLATFORM}.txt"

if [[ "$CHECK" == 1 ]]; then
  if diff -u "$SNAP" "$TMP"; then
    echo "ABI snapshot ($PLATFORM) is up to date."
  else
    echo "ERROR: ABI snapshot drift on $PLATFORM. Regenerate with scripts/regen_abi_snapshots.sh and review." >&2
    exit 1
  fi
  exit 0
fi

cp "$TMP" "$SNAP"
echo "Wrote $SNAP ($(wc -l < "$SNAP") symbols)."

if [[ "$DERIVE" == 1 ]]; then
  if [[ "$PLATFORM" != linux ]]; then
    echo "--derive is Linux-only (derives macos/windows from the linux snapshot)." >&2
    exit 2
  fi
  for p in macos windows; do
    grep -v '^N4M_1$' "$ABI_DIR/expected_symbols_linux.txt" > "$ABI_DIR/expected_symbols_${p}.txt"
    echo "Derived $ABI_DIR/expected_symbols_${p}.txt ($(wc -l < "$ABI_DIR/expected_symbols_${p}.txt") symbols)."
  done
  echo "NOTE: the macOS/Windows snapshots are also diffed live on their own runners in abi-check.yml; the derive is a convenience, not a substitute."
fi
