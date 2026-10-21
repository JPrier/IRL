#!/usr/bin/env bash
# Build a portable, self-contained IRL bundle for this OS/arch.
#
#   ./scripts/package.sh          # full bundle (includes ~950 MiB of assets)
#   ./scripts/package.sh --slim   # exe + native libs + manifests only
#
# The executable resolves its assets relative to itself, so the produced
# folder can be zipped and run on another machine with no Cargo and no setup.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SLIM=0
[ "${1:-}" = "--slim" ] && SLIM=1

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) OS="windows"; BIN="bevy-game.exe" ;;
  Darwin)               OS="macos";   BIN="bevy-game" ;;
  Linux)                OS="linux";   BIN="bevy-game" ;;
  *)                    OS="unknown"; BIN="bevy-game" ;;
esac
ARCH="$(uname -m)"
DIST="dist/IRL-${OS}-${ARCH}"

echo "==> Building release binary ($OS/$ARCH)"
cargo build --release -p bevy-game

rm -rf "$DIST"
mkdir -p "$DIST/assets/speech" "$DIST/assets/models"
cp "target/release/$BIN" "$DIST/"

# Native runtime libraries must sit beside the executable.
copy_native() {
  local pattern="$1"
  local found
  found="$(find target/release -maxdepth 2 -name "$pattern" 2>/dev/null | head -1 || true)"
  if [ -n "$found" ]; then
    echo "    + $(basename "$found")"
    cp "$found" "$DIST/"
  fi
}
echo "==> Copying native runtime libraries"
if [ "$OS" = "windows" ]; then
  copy_native "sherpa-onnx-c-api.dll"
  copy_native "sherpa-onnx-cxx-api.dll"
  copy_native "onnxruntime.dll"
  copy_native "onnxruntime_providers_shared.dll"
else
  for lib in target/release/*.dylib target/release/*.so; do
    if [ -e "$lib" ]; then
      cp "$lib" "$DIST/"
      echo "    + $(basename "$lib")"
    fi
  done
fi

cp assets/speech/manifest.json "$DIST/assets/speech/"
cp assets/models/manifest.json "$DIST/assets/models/"

if [ "$SLIM" = "1" ]; then
  cat > "$DIST/FIRST-RUN.txt" <<'TXT'
Slim bundle: executable + native libraries + pinned manifests only.

To fetch and verify the speech assets and local model (~950 MiB), run once:

  Linux/macOS : ./scripts/setup.sh
  Windows     : .\scripts\setup.ps1

or set IRL_HOME to a directory that contains assets/ and .cache/.
TXT
else
  echo "==> Bundling assets (speech models + local LLM)"
  mkdir -p "$DIST/.cache/speech" "$DIST/.cache/models"
  if [ -d .cache/speech/models ]; then
    cp -R .cache/speech/models "$DIST/.cache/speech/models"
  else
    echo "    WARNING: .cache/speech/models missing; run scripts/setup.sh first" >&2
  fi
  shopt -s nullglob
  for gguf in .cache/models/*.gguf; do
    cp "$gguf" "$DIST/.cache/models/"
  done
  shopt -u nullglob
fi

cat > "$DIST/run.sh" <<'SH'
#!/usr/bin/env bash
cd "$(dirname "$0")"
exec ./bevy-game "$@"
SH
chmod +x "$DIST/run.sh"

cat > "$DIST/run.cmd" <<'CMD'
@echo off
cd /d "%~dp0"
bevy-game.exe %*
CMD

echo "==> Packaged: $DIST"
if command -v du >/dev/null 2>&1; then du -sh "$DIST" 2>/dev/null || true; fi
echo "    Run it with: $DIST/run.sh   (or run.cmd on Windows)"
