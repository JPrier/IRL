# Launch IRL: the graphical, voice-native Bevy client with the real local model
# and neural speech. Runs one-time setup automatically if assets are missing.
#
#   .\scripts\play.ps1 [extra bevy-game args]
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$model  = ".cache/models/Qwen3.5-0.8B-Q4_K_M.gguf"
$speech = ".cache/speech/models/kokoro-en-v0_19"

if (-not (Test-Path $model) -or -not (Test-Path $speech)) {
    Write-Host "==> First run: fetching assets (this happens once)"
    & "$PSScriptRoot/setup.ps1"
}

Write-Host "==> Building and launching IRL (graphical, voice-native)"
Write-Host "    WASD move   V talk   Y accept   N reject   E investigate   F3 diagnostics   Esc quit"
cargo run --release -p bevy-game -- --model $model @args
