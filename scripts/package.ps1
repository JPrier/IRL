# Build a portable, self-contained IRL bundle for Windows.
#
#   .\scripts\package.ps1          # full bundle (includes ~950 MiB of assets)
#   .\scripts\package.ps1 -Slim    # exe + native libs + manifests only
#
# The executable resolves its assets relative to itself, so the produced folder
# can be zipped and run on another machine with no Cargo and no setup.
param([switch]$Slim)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$dist = "dist/IRL-windows-x64"

Write-Host "==> Building release binary (windows/x64)"
cargo build --release -p bevy-game

if (Test-Path $dist) { Remove-Item -Recurse -Force $dist }
New-Item -ItemType Directory -Force -Path "$dist/assets/speech", "$dist/assets/models" | Out-Null
Copy-Item "target/release/bevy-game.exe" "$dist/"

Write-Host "==> Copying native runtime libraries"
$dlls = @("sherpa-onnx-c-api.dll", "sherpa-onnx-cxx-api.dll", "onnxruntime.dll", "onnxruntime_providers_shared.dll")
foreach ($dll in $dlls) {
    $found = Get-ChildItem -Path "target/release" -Filter $dll -Recurse -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($found) {
        Copy-Item $found.FullName "$dist/"
        Write-Host "    + $dll"
    }
}

Copy-Item "assets/speech/manifest.json" "$dist/assets/speech/"
Copy-Item "assets/models/manifest.json" "$dist/assets/models/"

if ($Slim) {
    @"
Slim bundle: executable + native libraries + pinned manifests only.

To fetch and verify the speech assets and local model (~950 MiB), run once:

  .\scripts\setup.ps1

or set IRL_HOME to a directory that contains assets\ and .cache\.
"@ | Out-File -Encoding utf8 "$dist/FIRST-RUN.txt"
} else {
    Write-Host "==> Bundling assets (speech models + local LLM)"
    New-Item -ItemType Directory -Force -Path "$dist/.cache/speech", "$dist/.cache/models" | Out-Null
    if (Test-Path ".cache/speech/models") {
        Copy-Item -Recurse ".cache/speech/models" "$dist/.cache/speech/models"
    } else {
        Write-Warning ".cache/speech/models missing; run .\scripts\setup.ps1 first"
    }
    Get-ChildItem ".cache/models" -Filter *.gguf -ErrorAction SilentlyContinue |
        ForEach-Object { Copy-Item $_.FullName "$dist/.cache/models/" }
}

@"
@echo off
cd /d "%~dp0"
bevy-game.exe %*
"@ | Out-File -Encoding ascii "$dist/run.cmd"

Write-Host "==> Packaged: $dist"
$size = (Get-ChildItem $dist -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host ("    size: {0:N2} GiB" -f $size)
Write-Host "    Run it with: $dist\run.cmd"
