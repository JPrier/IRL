# Integrated voice pipeline benchmark — Milestone B evidence

This records **real, measured** end-to-end numbers for the complete offline
voice loop:

```text
WAV/mic ─► Whisper ASR ─► resident local LLM ─► authoritative validation
        ─► grounded NPC text ─► Kokoro neural TTS ─► reply WAV
```

Unlike [`voice-neural.md`](voice-neural.md) (which measures TTS and ASR in
isolation), the numbers here are taken while the resident LLM, the ASR and the
TTS engine are **all loaded and contending for the same CPU**. That is the
number that matters for turn-taking, and it is measurably worse than the
isolated figures.

## Host

| Item | Value |
|---|---|
| OS | Windows 11 (build 26200), Git Bash |
| CPU | AMD Ryzen 7 5800X3D (8 cores / 16 threads) |
| GPU | NVIDIA GeForce RTX 5080 (unused; all inference is `provider = "cpu"`) |
| RAM | 64 GiB |
| LLM | Qwen3.5-0.8B-Q4_K_M (`.cache/models/`, gitignored) |
| TTS | Kokoro 82M, voice `af_sarah` |
| ASR | Whisper tiny.en int8 |
| Date | October 2026 |

## Reproduce

```bash
# 1. One-time assets
cargo run --release -p voice-demo -- install

# 2. Full chain over a neural WAV fixture (ASR + decision + TTS in one process)
cargo run --release -p voice-demo -- tts --text "Who signed the ledger?" \
  --out .cache/speech/fixture-ledger.wav
cargo run --release -p voice-play -- once .cache/speech/fixture-ledger.wav \
  --model .cache/models/Qwen3.5-0.8B-Q4_K_M.gguf --out .cache/speech/reply.wav

# 3. Decision + TTS per turn, several natural phrasings
for p in "I need to see your ledger." "Who signed the ledger?" \
         "Please show me the ledger."; do
  cargo run --release -p voice-play -- say "$p" \
    --model .cache/models/Qwen3.5-0.8B-Q4_K_M.gguf --out .cache/speech/reply.wav
done
```

## Full chain (one turn, real model)

| Stage | Measured | Notes |
|---|---:|---|
| ASR (`Who signed the ledger?`, 1.31 s audio) | **82 ms** | transcript exact; ≈16× real time |
| LLM decision | **5857 ms** | 0.8B, CPU, grammar-constrained JSON |
| Neural TTS (2.67 s audio) | **1253 ms** | **2.13× real time** under contention |
| **Time from end of speech to first audio ready** | **7192 ms** | ASR + decision + TTS |

## Per-turn decision and TTS (resident LLM, `say` path)

| Player utterance | Grocer action | Decision | TTS audio | TTS time | TTS RTF |
|---|---|---:|---:|---:|---:|
| "I need to see your ledger." | `share_claim` (model) | 4572 ms | 2.85 s | 2295 ms | 1.24× |
| "What can you tell me about the ledger?" | `make_offer` (model) | 4450 ms | — | — | — |
| "I will bring medicine for your daughter." | `make_offer` (model) | 4374 ms | — | — | — |
| "Please show me the ledger." | `make_offer` (model) | 5240 ms | — | — | — |
| "Who signed the ledger?" | canonical obligation | 5845 ms | 2.67 s | 1252 ms | 2.13× |

## Findings

1. **Decision latency dominates.** At ~4.4–5.9 s per turn, the 0.8B model is far
   too slow for natural turn-taking. This is the single biggest responsiveness
   gap; see the target in the design brief (minimise P95 time-to-first-audio).
2. **TTS slows sharply under contention.** Isolated Kokoro measures ~3.0× real
   time ([`voice-neural.md`](voice-neural.md)); sharing the CPU with the resident
   LLM it drops to **1.24–2.13×**. The 1.5× audio-production target is met in
   some turns but **missed in others** (1.24×). This is why TTS-only speed is not
   reported as proof of pipeline speed.
3. **ASR is not the bottleneck.** 82 ms for a 1.31 s utterance.
4. **Disclosure is now audible.** `ShareClaim` — how an NPC reveals a fact —
   previously produced *no spoken line at all*, so the grocer's reveal was
   silent in the voice clients. It now speaks the grounded fact statement
   (`"The grocer's daughter is ill and needs medicine."`). Regression test:
   `rpg_eval::tests::share_claim_speaks_the_grounded_fact_statement`.
5. **Provenance is reported, not hidden.** The "Who signed the ledger?" turn
   shows `0 model decision(s), 1 canonical obligation(s)`: the model failed to
   honour the satisfied bargain and the canonical policy substituted the
   designed line. The voice client speaks the committed text either way and
   never attributes the canonical line to the model.

## NOT TESTED here

* P95/P99 latencies (only 5 samples; a distribution needs a longer run).
* Live microphone capture and speaker playback (unverified on this host).
* Streaming / barge-in (Kokoro is whole-utterance).
* Any model other than Qwen3.5-0.8B.
* Concurrent rendering (Bevy) + voice + LLM contention.
* Steam Deck or any constrained-hardware proxy.
