#!/usr/bin/env bash
# Launch IRL: the graphical, voice-native Bevy client with the real local model
# and neural speech. Runs one-time setup automatically if assets are missing.
#
#   ./scripts/play.sh [extra bevy-game args]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL=".cache/models/Qwen3.5-0.8B-Q4_K_M.gguf"
SPEECH=".cache/speech/models/kokoro-en-v0_19"

if [ ! -f "$MODEL" ] || [ ! -d "$SPEECH" ]; then
  echo "==> First run: fetching assets (this happens once)"
  ./scripts/setup.sh
fi

echo "==> Building and launching IRL (graphical, voice-native)"
echo "    WASD move   V talk   Y accept   N reject   E investigate   F3 diagnostics   Esc quit"
cargo run --release -p bevy-game -- --model "$MODEL" "$@"
