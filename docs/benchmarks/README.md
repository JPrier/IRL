# Benchmark evidence

Everything below was produced by the commands shown, on the host described.
**Read the caveats**: the speech numbers come from a diagnostic, non-neural
device. The LLM numbers *are* from a real local GGUF model (see §5), but only
the 0.8B model has been measured.

## Host

| Item | Value |
|---|---|
| OS | Windows 11, MINGW64/MSYS2 shell |
| Rust | 1.91.1, `x86_64-pc-windows-msvc` |
| CPU | 16 logical cores |
| GPU / VRAM | NOT RECORDED |
| Steam Deck | NOT TESTED |
| Date | October 2026 |

## 1. Test suite

```bash
cargo test --workspace
```

Result: **62 tests passed, 0 failed** (unit + integration + 4 property tests).

## 2. Simulated-player scenarios

```bash
cargo run -p scenario-runner -- --seed 2026
```

Result: **10/10 passed** (seed 2026).

| Scenario | Checks | Events | Canonical checksum |
|---|---:|---:|---|
| friendly_investigation | 3 | 8 | `4b4ec3b51215830f` |
| grocer_conditional_bargain | 7 | 11 | `113ad62938905fdb` |
| private_conversation_and_spy | 4 | 3 | `9945e405ab80f8d7` |
| blackmail_has_consequences | 4 | 2 | `2d6fb25d40d526a1` |
| ignore_plot_world_progresses | 4 | 9 | `c2ffd7ebc73b4bd5` |
| save_reload_continuity | 2 | 1 | `fc8ebeefbc24d206` |
| deterministic_replay | 2 | 4 | `d57a4620601abf45` |
| knowledge_boundaries | 3 | 2 | `d53a9166ad6e99ed` |
| three_independent_routes | 3 | 0 | `cbf29ce484222325` |
| innocent_grocer_stays_innocent | 2 | 12 | `fc192077aa931ee0` |

Different seeds change the checksums (the RNG is seeded), but replaying the same
seed produces the same checksum. That is asserted by `deterministic_replay`.

## 3. Integrated speech/buffer benchmark (DIAGNOSTIC)

```bash
cargo run -p model-bench -- --speech
```

Representative run (numbers vary by a few percent run-to-run):

| Utterance | Audio produced | Wall time | RTF | WAV bytes |
|---|---:|---:|---:|---:|
| ~5 s | 5.76 s | 0.006 s | ~987× | 276,480 |
| ~15 s | 17.28 s | 0.018 s | ~969× | 829,438 |
| ~40 s | 43.20 s | 0.045 s | ~964× | 2,073,600 |

Streaming aggregate: `chunks=3 audio=66.24s wall=0.068s speed_ratio≈967×
underruns=0 min_lead≈5.75s`.

> **These are not neural-TTS results.** `DiagnosticToneSynth` is a clearly
> labelled non-neural device that writes real PCM so the WAV path and buffer
> accounting can be tested. A neural backend must be measured separately against
> the ≥1.5× P05 target; no such numbers exist yet.

The neural backend correctly reports:

```
neural backend status: speech asset missing: assets/voices/<licensed-model>.onnx
```

## 4. Model matrix status

```bash
cargo run -p model-bench -- --list-models
```

| Profile | Status |
|---|---|
| heuristic (no-LLM baseline) | VERIFIED-DETERMINISTIC |
| scripted (test double) | VERIFIED-DETERMINISTIC |
| Qwen3.5-0.8B-Q4_K_M | **MEASURED** (real GGUF, CPU) |
| Qwen3.5-2B / 4B | NOT TESTED |
| Qwen3-1.7B | NOT TESTED |
| LFM2.5-1.2B-Instruct | NOT TESTED |
| Gemma 3 1B | NOT TESTED |

## 5. Real local LLM — Qwen3.5-0.8B-Q4_K_M (CPU)

```bash
cargo run -p model-bench -- --profile docs/benchmarks/qwen35-0.8b.json \
  --turns 3 --seed 2026 --json --out docs/benchmarks/qwen35-0.8b-report.json
```

Profile: `.cache/models/Qwen3.5-0.8B-Q4_K_M.gguf` (507 MiB), `n_ctx=4096`,
`n_threads=8`, CPU only. The model file is gitignored; the raw report is
committed at [`qwen35-0.8b-report.json`](qwen35-0.8b-report.json).

| Metric | Value |
|---|---:|
| Load time | 576 ms |
| Warm-up time | 257 ms |
| Avg prefill | 777 ms |
| Avg first token | 40 ms |
| Avg decode | 2630 ms |
| Decode throughput | 11.0 tok/s |
| Structured JSON validity | 3/3 (100%) |
| Invalid / rejected decisions | 0 |
| Peak RSS | not measurable on Windows (reported `null`) |

Dialogue probe (unconstrained, `no_think`): 26 tokens in 554 ms, non-empty,
**no reasoning block leaked** into the output.

Repeated gameplay (3 seeds): `gameplay_pass_rate = 1.0`; each run negotiated the
conditional bargain, satisfied the commitment and disclosed `f_grocer_ledger`
(`memory_consistent = true`), with **0 deterministic fallbacks**. When the
model's own action does not honour a satisfied deal, the policy substitutes the
canonical action and reports `canonical-obligation`/`canonical-agenda` — those
are counted separately and never presented as model output.

Real-model regression tests (weights required, `#[ignore]` by default):

```bash
RPG_TEST_MODEL=.cache/models/Qwen3.5-0.8B-Q4_K_M.gguf \
  cargo test -p rpg_inference --test model_integration -- --ignored
```

They cover 5 repeated schema-valid JSON decisions, malformed grammars failing
loudly, cancellation, and deadlines.

## NOT TESTED (explicit)

* Any model larger than 0.8B (2B/4B/other families).
* GPU/Vulkan/ROCm/CUDA paths, VRAM peaks, thermal throttling.
* Neural TTS real-time factor, naturalness, voice differentiation.
* ASR accuracy on game vocabulary, accents, noise, interruption.
* Steam Deck and low-end integrated GPU.
* Human blind playtests and inter-rater quality scoring.
* 90% line coverage instrumentation (tests exist; `tarpaulin`/`llvm-cov` is not
  wired into CI).

Publishing a "Steam Deck compatible" or "faster than speech" claim without
running the corresponding row is explicitly forbidden by the design brief.
