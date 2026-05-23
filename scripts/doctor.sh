#!/usr/bin/env bash
#
# scripts/doctor.sh — environment check for n4m contributors.
#
# Reports the resolved version + path for every tool n4m's build /
# test / parity / docs pipelines need. Exits non-zero if any required
# tool is missing (optional tools are reported but never fail).
#
# Designed to run identically on the devcontainer and on bare metal.

set -uo pipefail

# ---------------------------------------------------------------------------
# Pretty output helpers
# ---------------------------------------------------------------------------
NC=$'\e[0m'; BOLD=$'\e[1m'
GREEN=$'\e[1;32m'; YELLOW=$'\e[1;33m'; RED=$'\e[1;31m'; CYAN=$'\e[1;36m'

OK="${GREEN}OK${NC}"; WARN="${YELLOW}!${NC}"; MISS="${RED}MISS${NC}"

OS="$(uname -s)"
case "${OS}" in
    Linux*)   PKG_HINT="apt install <pkg>" ;;
    Darwin*)  PKG_HINT="brew install <pkg>" ;;
    MINGW*|MSYS*|CYGWIN*) PKG_HINT="choco install <pkg>" ;;
    *)        PKG_HINT="<your-os pkg manager> install <pkg>" ;;
esac

REQUIRED_MISSING=0
OPTIONAL_MISSING=0

# ---------------------------------------------------------------------------
# check <required|optional> <name> <version-cmd> [install-hint]
# ---------------------------------------------------------------------------
check() {
    local kind="$1" name="$2" version_cmd="$3" hint="${4:-}"
    local bin
    bin="$(command -v "${name}" 2>/dev/null || true)"
    if [[ -z "${bin}" ]]; then
        if [[ "${kind}" == "required" ]]; then
            printf "  %s  %-18s %s\n" "${MISS}" "${name}" "(required) — install: ${hint:-${PKG_HINT/<pkg>/${name}}}"
            REQUIRED_MISSING=$((REQUIRED_MISSING + 1))
        else
            printf "  %s     %-18s (optional)\n" "${WARN}" "${name}"
            OPTIONAL_MISSING=$((OPTIONAL_MISSING + 1))
        fi
        return
    fi
    local version
    version="$(eval "${version_cmd}" 2>&1 | head -n1 || echo '?')"
    printf "  %s    %-18s %s  ${CYAN}(%s)${NC}\n" "${OK}" "${name}" "${version}" "${bin}"
}

printf "${BOLD}n4m doctor${NC} — checking required tools on %s\n\n" "${OS}"

# ---------------------------------------------------------------------------
# Required toolchain
# ---------------------------------------------------------------------------
printf "${BOLD}C / C++ build${NC}\n"
check required cmake         'cmake --version'
check required ninja         'ninja --version'
check required cc            'cc --version'
check required c++           'c++ --version'
check required pkg-config    'pkg-config --version'

printf "\n${BOLD}Python pipeline${NC}\n"
check required python3       'python3 --version'
check optional uv            'uv --version'

printf "\n${BOLD}R binding${NC}\n"
check required R             'R --version | head -n1'
check optional Rscript       'Rscript --version'

printf "\n${BOLD}MATLAB / Octave${NC}\n"
check optional octave        'octave --version | head -n1' "$(case ${OS} in Linux*) echo 'apt install octave octave-statistics' ;; Darwin*) echo 'brew install octave' ;; *) echo 'see https://octave.org' ;; esac)"
check optional matlab        'matlab -batch "disp(version)"'

printf "\n${BOLD}JS / WASM${NC}\n"
check optional node          'node --version'
check optional npm           'npm --version'
check optional emcc          'emcc --version | head -n1'

printf "\n${BOLD}Container / infra${NC}\n"
check optional docker        'docker --version'

printf "\n${BOLD}Linalg backends${NC}\n"
if pkg-config --exists openblas 2>/dev/null; then
    printf "  ${OK}    openblas           %s  ${CYAN}(via pkg-config)${NC}\n" "$(pkg-config --modversion openblas)"
else
    printf "  ${WARN}     openblas           (optional) — required only for blas-omp preset\n"
fi

printf "\n${BOLD}Extra languages (optional, deferred via post-create.sh)${NC}\n"
check optional rustc         'rustc --version'
check optional cargo         'cargo --version'
check optional julia         'julia --version'
check optional dotnet        'dotnet --version'
check optional go            'go version'

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
if [[ ${REQUIRED_MISSING} -gt 0 ]]; then
    printf "${RED}%d required tool(s) missing.${NC}  Install hint: %s\n" "${REQUIRED_MISSING}" "${PKG_HINT}"
    printf "Run ${BOLD}scripts/bootstrap-bare-metal-${OS,,}.sh${NC} or ${BOLD}make bootstrap${NC} for a guided installer.\n"
    exit 1
fi
if [[ ${OPTIONAL_MISSING} -gt 0 ]]; then
    printf "${YELLOW}%d optional tool(s) missing.${NC}  CI exercises them; install only if you touch the relevant subsystem.\n" "${OPTIONAL_MISSING}"
fi
printf "${GREEN}All required tools resolved.${NC}\n"
exit 0
