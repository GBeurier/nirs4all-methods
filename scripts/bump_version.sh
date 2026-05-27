#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-2.1
#
# bump_version.sh — n4m version source-of-truth syncer.
#
# Reads the canonical project version from cpp/include/n4m/n4m_version.h
# and propagates it to every downstream manifest.
#
# Usage:
#   scripts/bump_version.sh                # sync manifests to header (idempotent)
#   scripts/bump_version.sh --check        # exit 1 if any manifest drifts from header
#   scripts/bump_version.sh --bump X.Y.Z   # rewrite header to X.Y.Z then sync
#   scripts/bump_version.sh --help
#
# The script targets the **project semver** only (N4M_PROJECT_VERSION_*). The
# ABI semver (N4M_ABI_VERSION_*) is bumped manually with a dedicated commit
# whenever the C ABI surface changes — those two semvers are intentionally
# independent (see CMakeLists.txt comments).
#
# Requires: bash >= 4, GNU sed, python3 (only for JSON edits in package.json /
# package-lock.json to avoid brittle regex over JSON).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HEADER="${ROOT}/cpp/include/n4m/n4m_version.h"

if [[ ! -f "${HEADER}" ]]; then
    echo "error: canonical version header not found at ${HEADER}" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# 1. Parse current canonical version from the header
# ---------------------------------------------------------------------------
read_header_macro() {
    local macro="$1"
    local value
    # The '|| true' guards against set -e exiting when grep finds no match,
    # so the custom error message below can fire instead of a silent abort.
    value=$(grep -E "^[[:space:]]*#[[:space:]]*define[[:space:]]+${macro}[[:space:]]+" "${HEADER}" \
            | head -n1 \
            | sed -E "s|^[[:space:]]*#[[:space:]]*define[[:space:]]+${macro}[[:space:]]+([0-9]+).*$|\1|" \
            || true)
    if [[ -z "${value}" ]]; then
        echo "error: could not parse ${macro} from ${HEADER}" >&2
        exit 2
    fi
    printf '%s' "${value}"
}

read_header_string() {
    local macro="$1"
    local value
    value=$(grep -E "^[[:space:]]*#[[:space:]]*define[[:space:]]+${macro}[[:space:]]+\"" "${HEADER}" \
            | head -n1 \
            | sed -E "s|^[[:space:]]*#[[:space:]]*define[[:space:]]+${macro}[[:space:]]+\"([^\"]+)\".*$|\1|" \
            || true)
    if [[ -z "${value}" ]]; then
        echo "error: could not parse string ${macro} from ${HEADER}" >&2
        exit 2
    fi
    printf '%s' "${value}"
}

# ---------------------------------------------------------------------------
# 2. CLI handling
# ---------------------------------------------------------------------------
MODE="sync"
NEW_VERSION=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check)  MODE="check"; shift ;;
        --bump)
            if [[ $# -lt 2 ]]; then
                echo "error: --bump requires MAJOR.MINOR.PATCH" >&2
                exit 2
            fi
            MODE="bump"
            NEW_VERSION="$2"
            shift 2
            ;;
        -h|--help)
            sed -nE '2,/^$/{ s/^# ?//; /^!/!p }' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *) echo "error: unknown argument: $1" >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# 3. --bump: rewrite the header first
# ---------------------------------------------------------------------------
if [[ "${MODE}" == "bump" ]]; then
    if [[ ! "${NEW_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "error: --bump requires MAJOR.MINOR.PATCH (got: '${NEW_VERSION}')" >&2
        exit 2
    fi
    IFS='.' read -r NV_MAJOR NV_MINOR NV_PATCH <<<"${NEW_VERSION}"

    sed -i -E \
        -e "s/^(#define[[:space:]]+N4M_PROJECT_VERSION_MAJOR[[:space:]]+)[0-9]+/\1${NV_MAJOR}/" \
        -e "s/^(#define[[:space:]]+N4M_PROJECT_VERSION_MINOR[[:space:]]+)[0-9]+/\1${NV_MINOR}/" \
        -e "s/^(#define[[:space:]]+N4M_PROJECT_VERSION_PATCH[[:space:]]+)[0-9]+/\1${NV_PATCH}/" \
        -e "s/^(#define[[:space:]]+N4M_PROJECT_VERSION_STRING[[:space:]]+\")[^\"]+(\".*)$/\1${NEW_VERSION}\2/" \
        "${HEADER}"
    echo "  bumped n4m_version.h to ${NEW_VERSION}"
    MODE="sync"
fi

# ---------------------------------------------------------------------------
# 4. Re-read header (post --bump if any) and cross-check internal consistency
# ---------------------------------------------------------------------------
PMAJOR=$(read_header_macro N4M_PROJECT_VERSION_MAJOR)
PMINOR=$(read_header_macro N4M_PROJECT_VERSION_MINOR)
PPATCH=$(read_header_macro N4M_PROJECT_VERSION_PATCH)
VERSION="${PMAJOR}.${PMINOR}.${PPATCH}"

VSTRING=$(read_header_string N4M_PROJECT_VERSION_STRING)
if [[ "${VSTRING}" != "${VERSION}" ]]; then
    echo "error: N4M_PROJECT_VERSION_STRING (\"${VSTRING}\") disagrees with" \
         "numeric macros (\"${VERSION}\") in ${HEADER}" >&2
    exit 2
fi

AMAJOR=$(read_header_macro N4M_ABI_VERSION_MAJOR)
AMINOR=$(read_header_macro N4M_ABI_VERSION_MINOR)
APATCH=$(read_header_macro N4M_ABI_VERSION_PATCH)
ABI_VERSION="${AMAJOR}.${AMINOR}.${APATCH}"

if [[ "${MODE}" == "check" ]]; then
    echo "  canonical project version: ${VERSION}"
    echo "  canonical ABI version    : ${ABI_VERSION}"
else
    echo "  syncing manifests to project version ${VERSION} (ABI ${ABI_VERSION})"
fi

# ---------------------------------------------------------------------------
# 5. Manifest update / check engine
# ---------------------------------------------------------------------------
# Each entry registers a file + a regex describing the line where the version
# lives + a replacement template. The placeholder ${V} expands to ${VERSION}.

EXIT_CODE=0
DRIFTED=()

# update_with_sed <relative_path> <sed_match_regex> <sed_replace_regex>
#   sed_match_regex: must capture the version with a group, used to extract the
#                    current value for --check mode.
#   sed_replace_regex: full sed -E expression (typically  s/MATCH/REPLACEMENT/).
update_with_sed() {
    local rel="$1" match="$2" replace="$3"
    local abs="${ROOT}/${rel}"

    if [[ ! -f "${abs}" ]]; then
        echo "  DRIFT: ${rel} missing (expected manifest)" >&2
        DRIFTED+=("${rel}")
        EXIT_CODE=1
        return 0
    fi

    local current
    current=$(grep -E "${match}" "${abs}" | head -n1 || true)
    if [[ -z "${current}" ]]; then
        echo "  DRIFT: ${rel} has no line matching /${match}/" >&2
        DRIFTED+=("${rel}")
        EXIT_CODE=1
        return 0
    fi

    # Extract the captured version string from the matched line. Use '|' as
    # the sed delimiter because some patterns (e.g. </Version>) contain '/'.
    local found
    found=$(printf '%s\n' "${current}" \
            | sed -E "s|.*${match}.*|\1|" \
            | head -n1)

    if [[ "${MODE}" == "check" ]]; then
        if [[ "${found}" != "${VERSION}" ]]; then
            echo "  DRIFT: ${rel} reports '${found}' (expected '${VERSION}')" >&2
            DRIFTED+=("${rel}")
            EXIT_CODE=1
        fi
    else
        if [[ "${found}" == "${VERSION}" ]]; then
            return 0  # already in sync
        fi
        sed -i -E "${replace}" "${abs}"
        echo "  updated ${rel}: ${found} → ${VERSION}"
    fi
}

# update_json_field <relative_path> <jq-style top-level field name>
#   uses python3 to safely rewrite a JSON top-level "version" field.
update_json_field() {
    local rel="$1" field="$2"
    local abs="${ROOT}/${rel}"

    if [[ ! -f "${abs}" ]]; then
        echo "  DRIFT: ${rel} missing (expected manifest)" >&2
        DRIFTED+=("${rel}")
        EXIT_CODE=1
        return 0
    fi

    local found
    found=$(python3 - "${abs}" "${field}" <<'PY'
import json, sys
path, field = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as f:
    doc = json.load(f)
val = doc.get(field, "")
sys.stdout.write(str(val))
PY
    )

    if [[ "${MODE}" == "check" ]]; then
        if [[ "${found}" != "${VERSION}" ]]; then
            echo "  DRIFT: ${rel} '${field}' reports '${found}' (expected '${VERSION}')" >&2
            DRIFTED+=("${rel}")
            EXIT_CODE=1
        fi
    else
        if [[ "${found}" == "${VERSION}" ]]; then
            return 0
        fi
        python3 - "${abs}" "${field}" "${VERSION}" <<'PY'
import json, sys
path, field, new = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, encoding="utf-8") as f:
    doc = json.load(f)
doc[field] = new
# Preserve trailing newline + 2-space indent (npm convention).
with open(path, "w", encoding="utf-8") as f:
    json.dump(doc, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY
        echo "  updated ${rel} (${field}): ${found} → ${VERSION}"
    fi
}

# update_package_lock <relative_path>
#   Updates BOTH the top-level "version" AND the nested "packages": {"":{...}}
#   root package version. npm reads both — they must agree.
update_package_lock() {
    local rel="$1"
    local abs="${ROOT}/${rel}"

    if [[ ! -f "${abs}" ]]; then
        echo "  DRIFT: ${rel} missing (expected manifest)" >&2
        DRIFTED+=("${rel}")
        EXIT_CODE=1
        return 0
    fi

    # Read both version fields; they may legitimately differ on a stale file
    # and must both be brought into agreement.
    local found_top found_root
    found_top=$(python3 - "${abs}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    doc = json.load(f)
sys.stdout.write(doc.get("version", ""))
PY
    )
    found_root=$(python3 - "${abs}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    doc = json.load(f)
sys.stdout.write(doc.get("packages", {}).get("", {}).get("version", ""))
PY
    )

    if [[ "${MODE}" == "check" ]]; then
        if [[ "${found_top}" != "${VERSION}" ]]; then
            echo "  DRIFT: ${rel} (top-level) reports '${found_top}' (expected '${VERSION}')" >&2
            DRIFTED+=("${rel}")
            EXIT_CODE=1
        fi
        # Always require the nested root entry to match — an empty/missing
        # value is itself drift (npm reads both fields and they MUST agree).
        if [[ "${found_root}" != "${VERSION}" ]]; then
            echo "  DRIFT: ${rel} (packages[''].version) reports '${found_root:-<missing>}' (expected '${VERSION}')" >&2
            DRIFTED+=("${rel}#root")
            EXIT_CODE=1
        fi
    else
        if [[ "${found_top}" == "${VERSION}" && "${found_root}" == "${VERSION}" ]]; then
            return 0  # both already in sync
        fi
        python3 - "${abs}" "${VERSION}" <<'PY'
import json, sys
path, new = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as f:
    doc = json.load(f)
doc["version"] = new
# Always materialise packages[""].version — npm requires it to agree with
# the top-level version. setdefault chain guarantees the slot exists.
doc.setdefault("packages", {}).setdefault("", {})["version"] = new
with open(path, "w", encoding="utf-8") as f:
    json.dump(doc, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY
        echo "  updated ${rel}: top=${found_top}, root=${found_root:-<missing>} → ${VERSION}"
    fi
}

# ---------------------------------------------------------------------------
# 6. The manifest table — every downstream version string lives here.
# ---------------------------------------------------------------------------

# Python (pyproject.toml: line  version = "X.Y.Z")
update_with_sed \
    "bindings/python/pyproject.toml" \
    "^version[[:space:]]*=[[:space:]]*\"([0-9]+\.[0-9]+\.[0-9]+)\"" \
    "s/^(version[[:space:]]*=[[:space:]]*\")[0-9]+\.[0-9]+\.[0-9]+(\")/\1${VERSION}\2/"

# Parity generator (own pyproject.toml — internal tool tracks n4m version)
update_with_sed \
    "parity/python_generator/pyproject.toml" \
    "^version[[:space:]]*=[[:space:]]*\"([0-9]+\.[0-9]+\.[0-9]+)\"" \
    "s/^(version[[:space:]]*=[[:space:]]*\")[0-9]+\.[0-9]+\.[0-9]+(\")/\1${VERSION}\2/"

# R (DESCRIPTION:  Version: X.Y.Z)
update_with_sed \
    "bindings/r/n4m/DESCRIPTION" \
    "^Version:[[:space:]]+([0-9]+\.[0-9]+\.[0-9]+)" \
    "s/^(Version:[[:space:]]+)[0-9]+\.[0-9]+\.[0-9]+/\1${VERSION}/"

# R CRAN comments (release summary line)
update_with_sed \
    "bindings/r/n4m/cran-comments.md" \
    '`n4m`[[:space:]]+([0-9]+\.[0-9]+\.[0-9]+)' \
    "s/(\`n4m\`[[:space:]]+)[0-9]+\.[0-9]+\.[0-9]+/\1${VERSION}/"

# Archived/frozen bindings under bindings/_archive/ (go, rust, dotnet, lua, nim,
# ruby, jni, julia, android) are intentionally excluded from active version sync
# — they are frozen at their last version and are not current target languages
# (R, Python, Octave/MATLAB, JS). See bindings/SPEC.md and DEFERRALS.md.

# JS (package.json: top-level "version")
update_json_field "bindings/js/package.json"      "version"
update_package_lock "bindings/js/package-lock.json"

# CITATION.cff (version: X.Y.Z   — note: existing value may have suffixes)
update_with_sed \
    "CITATION.cff" \
    "^version:[[:space:]]+([0-9]+\.[0-9]+\.[0-9]+)" \
    "s/^(version:[[:space:]]+)[0-9]+\.[0-9]+\.[0-9]+([-_a-zA-Z0-9.]*)$/\1${VERSION}/"

# Docs Sphinx (docs/conf.py: both 'release = "X.Y.Z"' and the short
# 'version = "X.Y"' that Sphinx uses for navigation. Both must agree with
# the canonical project semver.)
SHORT_VERSION="${PMAJOR}.${PMINOR}"

update_with_sed \
    "docs/conf.py" \
    "^release[[:space:]]*=[[:space:]]*\"([0-9]+\.[0-9]+\.[0-9]+)\"" \
    "s/^(release[[:space:]]*=[[:space:]]*\")[0-9]+\.[0-9]+\.[0-9]+(\")/\1${VERSION}\2/"

# docs/conf.py Sphinx 'short' version: must match MAJOR.MINOR.
update_sphinx_short_version() {
    local rel="docs/conf.py"
    local abs="${ROOT}/${rel}"
    if [[ ! -f "${abs}" ]]; then
        echo "  DRIFT: ${rel} missing (expected manifest)" >&2
        DRIFTED+=("${rel}")
        EXIT_CODE=1
        return 0
    fi
    local found
    found=$(grep -E '^version[[:space:]]*=[[:space:]]*"[0-9]+\.[0-9]+"' "${abs}" \
        | head -n1 \
        | sed -E 's|.*"([0-9]+\.[0-9]+)".*|\1|' \
        || true)
    if [[ -z "${found}" ]]; then
        echo "  DRIFT: ${rel} has no 'version = \"X.Y\"' line" >&2
        DRIFTED+=("${rel}#version")
        EXIT_CODE=1
        return 0
    fi
    if [[ "${MODE}" == "check" ]]; then
        if [[ "${found}" != "${SHORT_VERSION}" ]]; then
            echo "  DRIFT: ${rel} (short version) reports '${found}' (expected '${SHORT_VERSION}')" >&2
            DRIFTED+=("${rel}#version")
            EXIT_CODE=1
        fi
    else
        if [[ "${found}" == "${SHORT_VERSION}" ]]; then
            return 0
        fi
        sed -i -E "s/^(version[[:space:]]*=[[:space:]]*\")[0-9]+\.[0-9]+(\")/\1${SHORT_VERSION}\2/" "${abs}"
        echo "  updated ${rel} (short version): ${found} → ${SHORT_VERSION}"
    fi
}
update_sphinx_short_version

# README.md citation block (  version = {X.Y.Z}  )
update_with_sed \
    "README.md" \
    "version[[:space:]]*=[[:space:]]*\\{([0-9]+\.[0-9]+\.[0-9]+)\\}" \
    "s/(version[[:space:]]*=[[:space:]]*\\{)[0-9]+\.[0-9]+\.[0-9]+(\\})/\1${VERSION}\2/"

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
if [[ "${MODE}" == "check" ]]; then
    if [[ ${EXIT_CODE} -eq 0 ]]; then
        echo "  OK: every manifest is in sync with n4m_version.h (${VERSION})"
    else
        echo "" >&2
        echo "FAIL: ${#DRIFTED[@]} manifest(s) drifted from n4m_version.h." >&2
        echo "      Run scripts/bump_version.sh to re-sync." >&2
    fi
fi
exit ${EXIT_CODE}
