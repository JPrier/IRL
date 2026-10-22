# Build the browser client and serve it locally.
#
#   ./scripts/web-dev.ps1 [-Release] [-Port 8080] [-Base /IRL/]
param(
    [switch]$Release,
    [int]$Port = 8080,
    [string]$Base = "/"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
if ($Release) { & ./scripts/web-build.ps1 } else { & ./scripts/web-build.ps1 -Dev }
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$py = if (Get-Command python3 -ErrorAction SilentlyContinue) { "python3" } else { "python" }
& $py scripts/web-serve.py --dir dist/web --port $Port --base $Base
