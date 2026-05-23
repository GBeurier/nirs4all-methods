#!/usr/bin/env bash
#
# scripts/bootstrap.sh — interactive entry point invoked by `make bootstrap`.
#
# Decides between three setup paths:
#   1. VS Code + Docker available → print "Reopen in Container" hint and exit
#   2. Docker available, no VS Code → invite the user to `make shell`
#   3. Neither → run the bare-metal installer for the current OS
#
# Designed to be safe to re-run.

set -euo pipefail

log()  { printf '\033[1;36m[bootstrap]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[bootstrap]\033[0m %s\n' "$*"; }

OS="$(uname -s)"

have_docker() { command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; }
have_vscode() { command -v code >/dev/null 2>&1; }

if have_docker; then
    log "Docker detected."
    if have_vscode; then
        log "VS Code detected. Easiest path:"
        echo "  1) Open this folder in VS Code"
        echo "  2) Run command  → \"Dev Containers: Reopen in Container\""
        echo
        log "Alternatively, run \`make shell\` for a Compose-based shell."
    else
        log "VS Code not detected. Use:  make shell"
        log "(this runs \`docker compose -f .devcontainer/docker-compose.yml run --rm dev bash\`)"
    fi
    exit 0
fi

warn "Docker not available — falling back to the bare-metal installer."
case "${OS}" in
    Linux)
        exec scripts/bootstrap-bare-metal-linux.sh
        ;;
    Darwin)
        exec scripts/bootstrap-bare-metal-macos.sh
        ;;
    MINGW*|MSYS*|CYGWIN*)
        warn "Detected Windows. Please run the PowerShell installer from an elevated prompt:"
        echo "  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass"
        echo "  .\\scripts\\bootstrap-bare-metal-windows.ps1"
        exit 0
        ;;
    *)
        warn "Unrecognised OS: ${OS}. Install the toolchain manually then run \`make doctor\`."
        exit 1
        ;;
esac
