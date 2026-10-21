# Local model matrix

Status: **Qwen3.5-0.8B-Q4_K_M has been measured on the development host** (see
[`README.md`](README.md#5-real-local-llm--qwen3508b-q4_k_m-cpu) and
[`qwen35-0.8b-report.json`](qwen35-0.8b-report.json)). Every other profile is
still `NOT TESTED`; do not replace those cells with estimates.

## Profiles

| ID | Model | Params | Formats to test | Backend | Status |
|---|---|---|---:|---|---|
| <a id="qwen35-08b"></a>`qwen35-0.8b` | Qwen3.5-0.8B | 0.8B | 4-bit (Q4_K_M) | llama.cpp | **MEASURED** (CPU) |
| <a id="qwen35-2b"></a>`qwen35-2b` | Qwen3.5-2B | 2B | 4-bit, 5-bit, 8-bit | llama.cpp / mistral.rs | NOT TESTED |
| <a id="qwen35-4b"></a>`qwen35-4b` | Qwen3.5-4B | 4B | 4-bit | llama.cpp / mistral.rs | NOT TESTED |
| <a id="qwen3-17b"></a>`qwen3-1.7b` | Qwen3-1.7B | 1.7B | 4-bit, 5-bit, 8-bit | llama.cpp | NOT TESTED |
| <a id="lfm25"></a>`lfm25-1.2b` | LFM2.5-1.2B-Instruct | 1.2B | GGUF | llama.cpp | NOT TESTED (license review first) |
| <a id="gemma3-1b"></a>`gemma3-1b` | Gemma 3 1B | 1B | GGUF | llama.cpp | NOT TESTED |
| `heuristic` | no-LLM baseline | — | in-process | rpg_agents | VERIFIED-DETERMINISTIC |
| `scripted` | test double | — | in-process | rpg_agents | VERIFIED-DETERMINISTIC |

## How to measure a profile (once integrated)

1. Install a redistributable GGUF and record `sha256`.
2. Implement `rpg_domain::ports::InferenceEngine` in a resident backend
   (`crates/inference` already provides one over llama.cpp; **never** spawn per
   utterance).
3. Add a `model-profile.json` alongside the existing manifest fields:
   `model_id`, `artifact_sha256`, `quant`, `prompt_template`, `backend_commit`,
   `device`, `max_context`, `decoding`, `speech_model`, `voice`, `seed`.
4. Run the same recorded scenarios through it:

```bash
# Deterministic acceptance (recorded proposals):
cargo run -p scenario-runner -- --seed 2026

# Live-model behavioural run (non-bitwise, recorded output):
cargo run -p model-bench -- --profile model-profile.json --out runs/qwen35-2b.jsonl
```

5. Record, per profile:
   * first-token and first-audio P50/P95/P99;
   * integrated audio-speed ratio and underruns;
   * prompt/prefill/completion tokens and time;
   * peak RAM/VRAM and coexisting ASR/TTS footprint;
   * schema-valid action rate, rejected-action rate;
   * hallucinated-knowledge and narrative-contradiction counts;
   * blind quality ratings (separate sheet).

## Hard validity gates (a profile is disqualified if any fail)

* No canonical contradiction.
* No illicit item/fact disclosure.
* No unauthorized world commit.
* Neural voice backend present and licensed.
* Prompt never contains director-only / inner-circle facts for outsiders.

## Selection rule

Among profiles passing the hard gates, choose the **smallest / lowest-memory**
profile on the latency–quality Pareto frontier, keeping one documented
higher-quality option. Do not select by parameter count or a single tok/s figure.

## Licensing checklist

* [ ] Model license permits local redistribution or user download.
* [ ] Voice/model assets are ungated and redistributable, or optional.
* [ ] Piper GPL implications reviewed if used.
* [ ] Pocket TTS gated-download and per-voice licenses reviewed.
* [ ] Attribution captured in `assets/voices/MANIFEST.json`.
