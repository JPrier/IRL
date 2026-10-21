# One-command setup for IRL (Windows): fetch and verify every asset the game
# needs. Safe to re-run; already-verified assets are skipped.
#
#   .\scripts\setup.ps1
#
# Downloads:
#   * offline neural speech assets (Kokoro TTS, Whisper ASR, Silero VAD)
#   * the local language model (Qwen3.5-0.8B Q4_K_M, Apache-2.0)
#
# Nothing is fetched at gameplay time once this has run.
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$modelDir  = ".cache/models"
$modelFile = Join-Path $modelDir "Qwen3.5-0.8B-Q4_K_M.gguf"
$modelUrl  = "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf"
$modelSha  = "bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517"

Write-Host "==> IRL setup"

Write-Host "==> [1/2] Installing offline speech assets (Kokoro TTS, Whisper ASR, Silero VAD)"
cargo run --release -q -p voice-demo -- install

Write-Host "==> [2/2] Local language model: Qwen3.5-0.8B Q4_K_M (~508 MiB)"
New-Item -ItemType Directory -Force -Path $modelDir | Out-Null

$verified = $false
if (Test-Path $modelFile) {
    $hash = (Get-FileHash $modelFile -Algorithm SHA256).Hash.ToLower()
    if ($hash -eq $modelSha) {
        Write-Host "    already present and verified: $modelFile"
        $verified = $true
    }
}

if (-not $verified) {
    Write-Host "    downloading $modelUrl"
    $part = "$modelFile.part"
    if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
        & curl.exe -L --fail --retry 3 -C - -o $part $modelUrl
        if ($LASTEXITCODE -ne 0) { throw "download failed (curl exit $LASTEXITCODE)" }
    } else {
        Invoke-WebRequest -Uri $modelUrl -OutFile $part
    }
    Move-Item -Force $part $modelFile

    $hash = (Get-FileHash $modelFile -Algorithm SHA256).Hash.ToLower()
    if ($hash -ne $modelSha) { throw "SHA-256 mismatch for $modelFile" }
    Write-Host "    verified: $modelFile"
}

Write-Host ""
Write-Host "Setup complete. Launch the graphical, voice-native game with:"
Write-Host "    .\scripts\play.ps1"
Write-Host "Text-only client:"
Write-Host "    cargo run -p game"
