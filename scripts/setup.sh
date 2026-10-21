#!/usr/bin/env bash
# One-command setup for IRL (Linux/macOS): fetch and verify every asset the
# game needs. Safe to re-run; already-verified assets are skipped.
#
#   ./scripts/setup.sh
#
# Downloads:
#   * offline neural speech assets (Kokoro TTS, Whisper ASR, Silero VAD)
#   * the local language model (Qwen3.5-0.8B Q4_K_M, Apache-2.0)
#
# Nothing is fetched at gameplay time once this has run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_DIR=".cache/models"
MODEL_FILE="$MODEL_DIR/Qwen3.5-0.8B-Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf"
MODEL_SHA="bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517"

sha_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

echo "==> IRL setup"

echo "==> [1/2] Installing offline speech assets (Kokoro TTS, Whisper ASR, Silero VAD)"
cargo run --release -q -p voice-demo -- install

echo "==> [2/2] Local language model: Qwen3.5-0.8B Q4_K_M (~508 MiB)"
mkdir -p "$MODEL_DIR"
if [ -f "$MODEL_FILE" ] && [ "$(sha_of "$MODEL_FILE")" = "$MODEL_SHA" ]; then
  echo "    already present and verified: $MODEL_FILE"
else
  echo "    downloading $MODEL_URL"
  curl -L --fail --retry 3 -C - -o "$MODEL_FILE.part" "$MODEL_URL"
  mv "$MODEL_FILE.part" "$MODEL_FILE"
  if [ "$(sha_of "$MODEL_FILE")" != "$MODEL_SHA" ]; then
    echo "ERROR: SHA-256 mismatch for $MODEL_FILE" >&2
    exit 1
  fi
  echo "    verified: $MODEL_FILE"
fi

echo
echo "Setup complete. Launch the graphical, voice-native game with:"
echo "    ./scripts/play.sh        # or: make play"
echo "Text-only client:"
echo "    make text"
