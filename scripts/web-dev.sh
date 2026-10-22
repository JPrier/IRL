#!/usr/bin/env bash
# Build the browser client and serve it locally.
#
#   ./scripts/web-dev.sh [--release] [--port 8080] [--base /IRL/]
#
# Defaults to the faster `web-dev` profile. Open the printed URL, paste a
# disposable OpenRouter key in the panel (or play with the deterministic
# policy), click the town and use WASD / Enter. No model weights are fetched.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUILD_ARGS=(--dev)
SERVE_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --release) BUILD_ARGS=(--release) ;;
    --port|--base) SERVE_ARGS+=("$1" "$2"); shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

./scripts/web-build.sh "${BUILD_ARGS[@]}"
exec python3 scripts/web-serve.py --dir dist/web "${SERVE_ARGS[@]}"
