# Build IRL's browser client into a static site (default: dist/web).
#
#   ./scripts/web-build.ps1            # size-optimised release build
#   ./scripts/web-build.ps1 -Dev       # faster development build
#   ./scripts/web-build.ps1 -Out DIR
#
# Requires the wasm32-unknown-unknown target and wasm-bindgen-cli matching
# Cargo.lock (see docs/web.md). wasm-opt is used when installed.
param(
    [switch]$Dev,
    [string]$Out = "dist/web"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$BuildProfile = if ($Dev) { "web-dev" } else { "web-release" }
$TargetDir = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { "target" }
$Wasm = Join-Path $TargetDir "wasm32-unknown-unknown/$BuildProfile/web-game.wasm"

$lock = Get-Content Cargo.lock -Raw
$m = [regex]::Match($lock, 'name = "wasm-bindgen"\r?\nversion = "([^"]+)"')
$Want = $m.Groups[1].Value
$bindgen = Get-Command wasm-bindgen -ErrorAction SilentlyContinue
if (-not $bindgen) {
    Write-Error "wasm-bindgen CLI not found. Install it with: cargo install wasm-bindgen-cli --version $Want --locked"
}
$Have = (& wasm-bindgen --version).Split(" ")[1]
if ($Have -ne $Want) {
    Write-Error "wasm-bindgen CLI $Have does not match the library ($Want). Run: cargo install wasm-bindgen-cli --version $Want --locked --force"
}

Write-Host "==> cargo build -p web-game --target wasm32-unknown-unknown --profile $BuildProfile"
$extra = @()
if ($env:WEB_CARGO_ARGS) { $extra = $env:WEB_CARGO_ARGS.Split(" ") }
cargo build -p web-game --target wasm32-unknown-unknown --profile $BuildProfile @extra
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> wasm-bindgen $Have"
if (Test-Path $Out) { Remove-Item -Recurse -Force $Out }
New-Item -ItemType Directory -Force (Join-Path $Out "pkg") | Out-Null
wasm-bindgen --target web --no-typescript --out-dir (Join-Path $Out "pkg") --out-name web_game $Wasm
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$opt = Get-Command wasm-opt -ErrorAction SilentlyContinue
$pkgWasm = Join-Path $Out "pkg/web_game_bg.wasm"
if (-not $Dev -and $opt) {
    Write-Host "==> wasm-opt -Os"
    $tmp = Join-Path $Out "pkg/web_game_bg.opt.wasm"
    wasm-opt -Os --enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext `
        --enable-mutable-globals --enable-reference-types --enable-multivalue $pkgWasm -o $tmp
    if ($LASTEXITCODE -eq 0) { Move-Item -Force $tmp $pkgWasm } else { Write-Warning "wasm-opt failed; keeping the unoptimised module" }
}

foreach ($f in "index.html", "app.js", "host.js", "style.css") {
    Copy-Item (Join-Path "apps/web-game/web" $f) $Out
}
New-Item -ItemType File -Force (Join-Path $Out ".nojekyll") | Out-Null
$bytes = (Get-Item $pkgWasm).Length
$commit = (git rev-parse --short HEAD 2>$null)
@{ commit = $commit; profile = $BuildProfile; wasm_bytes = $bytes; wasm_bindgen = $Have } |
    ConvertTo-Json | Set-Content (Join-Path $Out "build-info.json")
Write-Host ("==> built {0}  (wasm {1:N1} MiB)" -f $Out, ($bytes / 1MB))
