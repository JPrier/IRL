#!/usr/bin/env bash
# Build IRL's browser client into a static site (default: dist/web).
#
#   ./scripts/web-build.sh            # size-optimised release build
#   ./scripts/web-build.sh --dev      # faster build with debug assertions
#   ./scripts/web-build.sh --out DIR  # write the site somewhere else
#
# Requirements: the wasm32-unknown-unknown target (`rustup target add
# wasm32-unknown-unknown`) and wasm-bindgen-cli matching the wasm-bindgen
# version in Cargo.lock (`cargo install wasm-bindgen-cli --version <v> --locked`).
# wasm-opt (binaryen) is used when installed. No model weights, speech
# models, llama.cpp, ONNX or SQLite are part of the browser build.
#
# Extra cargo arguments can be passed in WEB_CARGO_ARGS (e.g. toolchain
# workarounds); RUST_LOG-style logging is set at runtime by the page.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROFILE="web-release"
OUT="dist/web"
while [ $# -gt 0 ]; do
  case "$1" in
    --dev) PROFILE="web-dev" ;;
    --release) PROFILE="web-release" ;;
    --out) shift; OUT="$1" ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

TARGET_DIR="${CARGO_TARGET_DIR:-target}"
WASM="$TARGET_DIR/wasm32-unknown-unknown/$PROFILE/web-game.wasm"
WANT_BINDGEN="$(awk '/^name = "wasm-bindgen"$/{getline; gsub(/"/,"",$3); print $3; exit}' Cargo.lock)"

if ! command -v wasm-bindgen >/dev/null 2>&1; then
  echo "error: wasm-bindgen CLI not found. Install it with:" >&2
  echo "  cargo install wasm-bindgen-cli --version $WANT_BINDGEN --locked" >&2
  exit 1
fi
HAVE_BINDGEN="$(wasm-bindgen --version | awk '{print $2}')"
if [ "$HAVE_BINDGEN" != "$WANT_BINDGEN" ]; then
  echo "error: wasm-bindgen CLI $HAVE_BINDGEN does not match the library ($WANT_BINDGEN in Cargo.lock)." >&2
  echo "  cargo install wasm-bindgen-cli --version $WANT_BINDGEN --locked --force" >&2
  exit 1
fi

echo "==> cargo build -p web-game --target wasm32-unknown-unknown --profile $PROFILE"
# shellcheck disable=SC2086
cargo build -p web-game --target wasm32-unknown-unknown --profile "$PROFILE" ${WEB_CARGO_ARGS:-}

echo "==> wasm-bindgen ($HAVE_BINDGEN)"
rm -rf "$OUT"
mkdir -p "$OUT/pkg"
wasm-bindgen --target web --no-typescript --out-dir "$OUT/pkg" --out-name web_game "$WASM"

if [ "$PROFILE" = "web-release" ] && command -v wasm-opt >/dev/null 2>&1; then
  echo "==> wasm-opt -Os"
  if ! wasm-opt -Os --enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext \
      --enable-mutable-globals --enable-reference-types --enable-multivalue \
      "$OUT/pkg/web_game_bg.wasm" -o "$OUT/pkg/web_game_bg.opt.wasm"; then
    echo "    warning: wasm-opt failed (old binaryen?); keeping the unoptimised module" >&2
  else
    mv "$OUT/pkg/web_game_bg.opt.wasm" "$OUT/pkg/web_game_bg.wasm"
  fi
elif [ "$PROFILE" = "web-release" ]; then
  echo "    (wasm-opt not found; skipping the optional size pass)"
fi

cp apps/web-game/web/index.html apps/web-game/web/app.js apps/web-game/web/host.js \
   apps/web-game/web/style.css "$OUT/"
# GitHub Pages: serve files as-is (no Jekyll processing of the pkg directory).
touch "$OUT/.nojekyll"

COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
WASM_BYTES="$(wc -c < "$OUT/pkg/web_game_bg.wasm" | tr -d ' ')"
GZIP_BYTES="$(gzip -9 -c "$OUT/pkg/web_game_bg.wasm" | wc -c | tr -d ' ')"
cat > "$OUT/build-info.json" <<JSON
{"commit": "$COMMIT", "profile": "$PROFILE", "wasm_bytes": $WASM_BYTES, "wasm_gzip_bytes": $GZIP_BYTES, "wasm_bindgen": "$HAVE_BINDGEN"}
JSON

echo "==> built $OUT"
echo "    wasm: $((WASM_BYTES / 1048576)) MiB ($((GZIP_BYTES / 1048576)) MiB gzip)"
