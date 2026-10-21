# Agentic RPG — Rust vertical slice

An offline-first, deterministic, **agentic** RPG prototype: a small town where
NPCs have persistent beliefs, goals and relationships; conversations and actions
are validated by an authoritative Rust simulation; a director can place
flexible clues; and spies only learn what they actually perceive.

This repository is the **first production-quality increment** described in
[`docs/design/agentic_rpg_rust_prototype_agent_prompt.md`](docs/design/agentic_rpg_rust_prototype_agent_prompt.md).
It is deliberately honest about what is implemented and verified versus what
still requires model assets or hardware we did not have.

> **TL;DR — verified today**
> * `cargo test --workspace` → all unit + integration tests pass.
> * `cargo run -p scenario-runner` → **10/10** simulated-player scenarios pass.
> * `cargo run -p game` → playable text client with movement, conversation,
>   conditional bargains, threats, investigation and save/load.
> * `cargo run -p game -- --model .cache/models/Qwen3.5-0.8B-Q4_K_M.gguf` →
>   the playable client talks to a **real local GGUF model** (llama.cpp); NPC
>   decisions are grammar-constrained JSON and validated by the simulation.
> * `cargo run -p model-bench -- --profile docs/benchmarks/qwen35-0.8b.json` →
>   real load/warm/prefill/decode latency, tokens/s and structured-action rate.
> * `cargo run -p model-bench -- --speech` → real WAV output + streaming
>   buffer metrics for the **diagnostic (non-neural)** synthesizer.
> * `cargo run --release -p voice-demo -- install && cargo run --release -p voice-demo -- bench`
>   → **real offline neural voice**: Kokoro TTS (~3× real time) and Whisper ASR
>   (~27× real time) on CPU. See [docs/benchmarks/voice-neural.md](docs/benchmarks/voice-neural.md).
> * `cargo run --release -p voice-demo -- roundtrip <file.wav> --play` →
>   WAV → ASR transcript → neural TTS reply → audible playback.
>
> **Not yet verified / NOT TESTED**
> * **Streaming neural TTS / barge-in.** Kokoro is whole-utterance, so first
>   audio arrives only after synthesis completes.
> * **Bevy 0.20 desktop renderer** (the shipped client is a text console; the
>   simulation is renderer-agnostic by design).
> * Steam Deck / low-end hardware numbers.
> * Raw decision quality of the 0.8B model. It is usable but weak: the
>   authoritative policy enforces satisfied obligations and designed bargains,
>   and any fallback is always reported rather than hidden.

## Quick start

```bash
# Build everything
cargo build --workspace

# Run the full test suite (unit, integration, scenarios, replay, persistence)
cargo test --workspace

# Headless simulated-player acceptance suite
cargo run -p scenario-runner -- --seed 2026

# Play the text vertical slice (deterministic baseline)
cargo run -p game

# Play against a real local GGUF model (weights are gitignored; see docs/benchmarks)
cargo run -p game -- --model .cache/models/Qwen3.5-0.8B-Q4_K_M.gguf

# Measure a real model profile (load/warm/prefill/decode/structured validity)
cargo run -p model-bench -- --profile docs/benchmarks/qwen35-0.8b.json

# Integrated speech/buffer benchmark (diagnostic device)
cargo run -p model-bench -- --speech

# Real offline neural voice (requires the pinned assets; see docs/benchmarks/voice-neural.md)
cargo run --release -p voice-demo -- install
cargo run --release -p voice-demo -- list
cargo run --release -p voice-demo -- bench
cargo run --release -p voice-demo -- roundtrip path/to/speech.wav --play
cargo run --release -p voice-demo -- mic   # live microphone loop

# Candidate model matrix (status + evidence)
cargo run -p model-bench -- --list-models
```

## What works

### Canonical simulation (`crates/sim`)
* Fixed-step clock, A* navigation on a walled grid, movement over time.
* Typed actions (`speak`, `move_to`, `make_offer`, `promise`, `share_claim`,
  `investigate`, `threaten`, `report_crime`, `allocate_budget`, `repair`, …)
  that are **validated before any state changes**.
* Append-only event log with checksums; replaying the same seed + proposals
  reproduces the same canonical events.
* Offers and commitments are transactional: a trade only completes if both
  sides actually hold their side; promises are satisfied by world predicates.

### Perception (`crates/perception`)
* Line of sight with wall occlusion, field-of-view cones, distance-based
  hearing with wall attenuation, addressee inference, overhearers.
* Spies receive subjective perception events, never the global transcript.

### Social (`crates/social`)
* Multi-dimensional relationships (trust, affection, respect, fear, anger,
  sadness, temptation, suspicion, obligation, familiarity) — no single honor
  meter.
* Provenance-aware beliefs; a claim can be believed or doubted based on trust,
  suspicion and confidence.
* Rumor propagation with a tracked transmission chain.
* Threats raise fear *and* anger; comfort reduces sadness; gifts create
  obligation.

### Narrative (`crates/narrative`)
* Immutable anchors + flexible checkpoints.
* Clue placement is rejected in already-inspected locations and for facts whose
  prerequisites are not established.
* The "three clue rule" is represented structurally (parallel candidate facts
  and multiple eligible holders).
* A seeded, fixed mystery answer (the mafia leader) supports genuine deduction.

### Agents (`crates/agents`)
* Minimal, scoped NPC context (identity, traits, relationships, the NPC's own
  beliefs only, current turn).
* Deterministic fallback policy plus a JSON action contract for a real model.
* Priority scheduler (P0 player → P2 background).
* Director role-promotion proposals, validated by the narrative controller.
* Inner-circle facts are filtered out of non-inner-circle prompts.

### Inference (`crates/inference`)
* Resident `llama.cpp` backend behind `InferenceEngine`: weights load once and
  are reused for every NPC turn.
* GGUF chat template is used when available, with a ChatML fallback; a
  `no_think` normalisation works around the binding's missing template kwargs.
* Tool decisions are constrained with a JSON-schema→GBNF grammar, and
  `LlamaSampler::sample` is used correctly (it accepts internally — double
  accepting was the cause of a native grammar abort).
* Streaming tokens, deadlines, cooperative cancellation and honest errors.
* Malformed raw grammars fail loudly; constrained decoding stops at the first
  complete JSON value instead of running into the grammar's terminal state.

### Persistence (`crates/persistence`)
* Versioned JSON save/load.
* SQLite event store with migrations, transactional bulk append, idempotent
  event ids and world snapshots.

### Evaluation (`crates/eval`)
* 10 deterministic simulated-player scenarios, including:
  friendly investigation, conditional grocer bargain, private conversation with
  a spy present, blackmail consequences, world progress while ignoring the plot,
  save/reload continuity, deterministic replay, knowledge boundaries, three
  independent routes and "the innocent grocer stays innocent".
* `ScenarioResult` exposes per-check pass/fail detail, event counts and a
  canonical checksum for CI gating.

### Speech (`crates/speech`)
* Real WAV encode/decode and file output.
* Integrated streaming/buffer metric (`audio-seconds / wall-seconds`,
  underruns, minimum lead).
* `DiagnosticToneSynth` clearly labelled **non-neural** — for pipeline tests
  only.
* `MissingNeuralBackend` fails loudly with the expected asset path instead of
  fabricating audio.

## Workspace layout

```
crates/
  domain/        stable IDs, geometry, world schema, actions, ports, RNG
  perception/    line of sight, hearing, speech routing
  social/        relationships, beliefs, commitments, rumours
  narrative/     anchors, checkpoints, clue validation, director commits
  sim/           authoritative clock, events, action validation, economy
  agents/        NPC context/policy, inference scheduler, director proposals
  inference/     resident llama.cpp backend, model profiles, JSON grammar
  speech/        WAV I/O, diagnostic synth, streaming metrics, neural port
  persistence/   JSON snapshots + SQLite event store
  content/       the handcrafted Hidden Mafia town (8 locations, 14 NPCs)
  eval/          simulated players, 10 acceptance scenarios, metrics
apps/
  game/          playable text client
  scenario-runner/  headless CI gate
  model-bench/   model matrix + integrated speech benchmark
docs/
  design/        the authoritative game-design documents
  research/      technical decisions and candidate manifest
  benchmarks/    reproducible evidence and NOT-TESTED list
  architecture.md
  testing.md
  playtest.md
```

The domain and simulation crates do **not** depend on Bevy, a model runtime or
SQLite. A renderer or a real inference backend can be added behind the existing
ports without touching gameplay.

## Design principles enforced by tests

* **Model output is a proposal, never truth.** Every action goes through
  `Simulation::apply` → validation → typed event.
* **Knowledge is provenance-bound.** NPCs cannot disclose facts they do not
  believe; inner-circle secrets require inner-circle status.
* **Privacy is mechanical.** Whispering in a private room does not reach a spy
  in the shop.
* **Attention is not evidence.** Repeatedly investigating the grocer does not
  make them the leader; the fixed answer stays fixed.
* **Determinism is a feature.** Same seed + same proposals ⇒ same event log
  checksum.

## Honest limitations

1. **No neural voice yet.** The pipeline, WAV output and metrics are real; the
   neural backends are ports with clear failure modes. Installing a compliant
   local TTS/ASR model is the next integration step.
2. **The 0.8B model is the only measured LLM.** Real GGUF inference,
   grammar-constrained tool decisions and the playable loop work, but raw
   0.8B decision quality is limited; larger models (2B+) are NOT TESTED. The
   simulation enforces knowledge boundaries, satisfied obligations and the
   designed bargain regardless of what the model proposes.
3. **No Bevy renderer.** The text client proves the gameplay loop; the
   simulation is already renderer-agnostic.
4. **No hardware matrix.** All numbers in `docs/benchmarks/` are from the
   development host and are explicitly labelled; Deck/GPU tiers are NOT TESTED.

See [`docs/research/2026-10-technical-decisions.md`](docs/research/2026-10-technical-decisions.md)
for the full decision log and candidate manifest.

## License

Code: MIT OR Apache-2.0. The bundled design documents are project
documentation. No third-party model weights are included in this repository.
