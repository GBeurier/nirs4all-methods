# scripts/bootstrap-bare-metal-windows.ps1
#
# Best-effort installer for the n4m dev toolchain on Windows using
# Chocolatey. Use the devcontainer (under WSL2 + Docker Desktop) unless
# you specifically need a native Windows setup.
#
# Run from an elevated PowerShell:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\scripts\bootstrap-bare-metal-windows.ps1

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

function Log($msg)  { Write-Host "[bootstrap] $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "[bootstrap] $msg" -ForegroundColor Yellow }
function Die($msg)  { Write-Host "[bootstrap] $msg" -ForegroundColor Red; exit 1 }

if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Die "run this script in an elevated PowerShell window (Run as administrator)"
}

if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Log "Installing Chocolatey"
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = `
        [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString(
        'https://community.chocolatey.org/install.ps1'))
}

# Core toolchain. Visual Studio Build Tools 2022 ships MSVC + Windows SDK
# (~6 GB). Skip if a more recent VS install is already on PATH.
$pkgs = @(
    'visualstudio2022buildtools',
    'visualstudio2022-workload-vctools',
    'cmake',
    'ninja',
    'python312',
    'r.project',
    'nodejs-lts',
    'docker-desktop',
    'git',
    'ripgrep',
    'jq'
)
foreach ($p in $pkgs) {
    Log "choco install $p"
    choco install -y --no-progress $p
}

# uv via the official installer (Windows path)
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Log "Installing uv"
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
}

# Octave — Chocolatey package is community-maintained; install if available
try {
    choco install -y --no-progress octave 2>$null
} catch {
    Warn "choco octave failed (community package); install from https://octave.org/download manually if you touch bindings/matlab"
}

Log "Done. Open a new PowerShell window (so PATH refreshes) and run: make doctor"
