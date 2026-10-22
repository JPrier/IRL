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
> * `cargo run --release -p voice-play -- once <file.wav> --model .cache/models/Qwen3.5-0.8B-Q4_K_M.gguf`
>   → **full offline voice loop**: WAV → Whisper ASR → resident local LLM →
>   authoritative validation → grounded NPC action.
> * `cargo run -p bevy-game -- --model .cache/models/Qwen3.5-0.8B-Q4_K_M.gguf`
>   → **graphical Bevy 0.20 client**: walk the town, press `V` to speak, watch
>   the authoritative simulation change.
> * `./scripts/web-dev.sh` → **the same game in the browser** (WebAssembly +
>   WebGL2): bring your own OpenRouter key, or play with the deterministic
>   policy. See [docs/web.md](docs/web.md).
> * `OPENROUTER_API_KEY=… cargo run -p bevy-game -- --provider openrouter` →
>   the native client with an **optional remote model**; local GGUF stays the
>   default and needs no key or network.
>
> **Not yet verified / NOT TESTED**
> * **Streaming neural TTS / barge-in.** Kokoro is whole-utterance, so first
>   audio arrives only after synthesis completes.
> * **Live microphone / speaker I/O on the current host.** The ASR/TTS paths are
>   exercised with WAV fixtures; live device I/O is unverified here.
> * Steam Deck / low-end hardware numbers.
> * Raw decision quality of the 0.8B model. It is usable but weak: the
>   authoritative policy enforces satisfied obligations and designed bargains,
>   and any fallback is always reported rather than hidden.

## Quick start (one command)

Everything the game needs is fetched, checksum-verified and cached locally by a
single setup step; after that the game runs fully offline. Nothing large is
committed to Git — `.cache/` is ignored and the setup scripts download it.

```bash
git clone https://github.com/JPrier/IRL && cd IRL

# Linux / macOS
./scripts/setup.sh      # fetch + verify neural speech assets and the local model
./scripts/play.sh       # build and launch the graphical, voice-native client
```

```powershell
# Windows (PowerShell)
.\scripts\setup.ps1
.\scripts\play.ps1
```

`make play` is the same as `./scripts/play.sh`, and `scripts/play.*` runs setup
automatically on first launch if anything is missing, so in practice:

```bash
./scripts/play.sh       # does setup if needed, then launches the game
```

### What setup downloads

| Asset | Size | License | Location |
|---|---|---|---|
| Kokoro 82M TTS (11 voices) | ~320 MiB | Apache-2.0 | `.cache/speech/` |
| Whisper tiny.en ASR (int8) | ~118 MiB | MIT | `.cache/speech/` |
| Silero VAD | 0.6 MiB | MIT | `.cache/speech/` |
| Qwen3.5-0.8B Q4_K_M (LLM) | ~508 MiB | Apache-2.0 | `.cache/models/` |

Speech assets are pinned by URL + SHA-256 in
[`assets/speech/manifest.json`](assets/speech/manifest.json); the model is pinned
the same way in [`assets/models/manifest.json`](assets/models/manifest.json).
Setup verifies every checksum and fails loudly on a mismatch.

### Ship it as a standalone bundle (no Rust needed)

A packaged build needs no Cargo and no setup step. `scripts/package.sh`
(Windows: `scripts/package.ps1`) builds a self-contained folder — the
executable, its native libraries (`onnxruntime.dll`, `sherpa-onnx-*.dll`,
`*.dylib`), and the speech + model assets:

```bash
make dist          # dist/IRL-<os>-<arch>/  — fully self-contained (~1.2 GiB)
make dist-slim     # executable + libraries only; fetches assets on first run
```

The executable resolves assets **relative to itself**, so the folder can be
zipped, moved to another machine, and run directly:

```bash
dist/IRL-linux-x86_64/run.sh          # or run.cmd on Windows
```

Check what a build resolved with `bevy-game --doctor`, and override the asset
location with `IRL_HOME=/some/dir`. Pushing a `v*` tag (or running the
**release** workflow) builds these bundles for Linux, Windows and macOS and
uploads them as artifacts.

### Controls (graphical client)

`WASD` walk (always — also while an NPC listens, thinks or speaks) ·
hold `V` to talk, release to send · `C` cancel the turn / interrupt the NPC ·
`Enter` type a line instead · `Y`/`N` accept/reject an offer · `Tab` next
offer · `E` investigate · `F5`/`F9` save/load · `F3` developer diagnostics ·
`F7` switch microphone · `F8` microphone self-test · `Esc` quit.

Microphone trouble? `bevy-game --mic-selftest` records raw device audio to
`mic_selftest.wav` (no VAD/ASR/LLM/TTS) and says whether the device delivers
audio at all. `bevy-game --voice-check [--mic-wav file.wav]` runs full voice
turns through the real runtime and reports simulation ticks per voice stage.
See [docs/voice-runtime.md](docs/voice-runtime.md).

### Browser build (WebAssembly)

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129 --locked
./scripts/web-dev.sh            # build + serve at http://127.0.0.1:8080/
./scripts/web-build.sh          # static site in dist/web (GitHub Pages ready)
```

The browser build shares the simulation, NPC policy, validation, saves and
the Bevy presentation with the native client. NPC decisions use OpenRouter
with a key you paste into the page (kept in memory only), or the
deterministic policy. Details, security model and limits: [docs/web.md](docs/web.md).

### Choosing the inference provider (native)

```bash
cargo run -p bevy-game -- --model model.gguf                     # local llama.cpp (default)
cargo run -p bevy-game -- --provider heuristic                   # deterministic policy
export OPENROUTER_API_KEY=sk-or-...                              # never on the command line
cargo run -p bevy-game -- --provider openrouter --openrouter-model openai/gpt-6-luna \
    --openrouter-max-requests 50 --openrouter-max-cost 0.25
```

OpenRouter runs on the inference worker like the local model and goes through
the same contract validation, canonical policy and simulation commit. Speech
stays the offline sherpa-onnx stack.

### Text-only client

```bash
cargo run -p game                                                    # deterministic baseline
cargo run -p game -- --model .cache/models/Qwen3.5-0.8B-Q4_K_M.gguf # real local model
```

### Developer and benchmark commands

```bash
# Full test suite, then the headless simulated-player acceptance gate
cargo test --workspace
cargo run -p scenario-runner -- --seed 2026

# Real-model latency and behaviour
cargo run -p model-bench -- --profile docs/benchmarks/qwen35-0.8b.json
cargo run -p model-bench -- --behavior --profile docs/benchmarks/qwen35-0.8b.json

# Neural speech harness (install/list/bench/roundtrip/mic)
cargo run --release -p voice-demo -- list
cargo run --release -p voice-demo -- bench
cargo run --release -p voice-demo -- roundtrip path/to/speech.wav --play
cargo run --release -p voice-demo -- mic

# Model matrix (status + evidence)
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
* **Real offline neural TTS and ASR** via `sherpa-onnx`: Kokoro 82M TTS
  (11 English voices, ~3× real time on CPU) and Whisper tiny.en int8 ASR
  (~27× real time). See [docs/benchmarks/voice-neural.md](docs/benchmarks/voice-neural.md).
* SHA-256-pinned, versioned asset manifest with explicit `voice-demo install`;
  no network access at gameplay time and no weights committed to Git.
* `cpal` microphone capture and speaker playback with a nonblocking callback
  and a resampler for arbitrary device rates.
* `DiagnosticToneSynth` remains a clearly labelled **non-neural** pipeline test
  device for CI, and `MissingNeuralBackend` fails loudly with the expected asset
  path instead of fabricating audio.

### OpenRouter (`crates/openrouter`)
* Optional remote provider behind `InferenceEngine`, shared by the native and
  browser clients: request bodies with the strict JSON-schema rendering of the
  shared action contract and `provider.require_parameters`, response/usage/
  cost parsing, SSE streaming, error classification, bounded retries, budgets
  and the model catalog.
* Native transport (`ureq`) on the inference worker with immediate
  cancellation; browser transport (`fetch` + `AbortController`).
* Mock-server tests; an opt-in billable integration test.

### Browser client (`apps/web-game`, `crates/web-runtime`, `crates/client`)
* The shared Bevy presentation (`irl_client`) over the shared `SimCore`,
  driven from the browser event loop; OpenRouter, speech and storage as
  nonblocking futures. Bring-your-own-key UI, budgets, IndexedDB saves with
  export/import, optional browser speech, diagnostics. See [docs/web.md](docs/web.md).

### Graphical client (`apps/bevy-game`)
* Bevy 0.20 2D renderer that consumes immutable simulation `Snapshot`s and
  submits validated `Command`s. The simulation never depends on Bevy.
* Town map, player/NPC sprites, proximity-aware talk targeting, offers,
  inventory, save/load (`F5`/`F9`) and a toggleable diagnostics overlay (`F3`).
* Push-to-talk (`V`) runs mic → VAD → ASR → local LLM → validation → neural
  TTS → playback as an explicit, cancellable state machine on dedicated
  workers ([`crates/runtime`](crates/runtime)); the fixed-tick simulation never
  waits for any of them. See [docs/voice-runtime.md](docs/voice-runtime.md).

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
  runtime/       authoritative SimCore (shared) + native fixed-tick thread and mic/ASR/LLM/TTS workers
  openrouter/    optional OpenRouter provider: shared protocol, native + browser transports
  client/        the shared Bevy presentation (native and browser)
  web-runtime/   browser driver for SimCore: event-loop ticks, fetch/speech/storage dispatch
  content/       the handcrafted Hidden Mafia town (8 locations, 14 NPCs)
  eval/          simulated players, 10 acceptance scenarios, metrics
apps/
  game/          playable text client
  bevy-game/     graphical Bevy 0.20 client (renderer + input only)
  web-game/      the same client for the browser (WASM + WebGL2), page and Playwright tests
  voice-play/    offline voice conversation loop (mic/WAV -> ASR -> LLM -> TTS)
  voice-demo/    neural speech harness (install/list/bench/roundtrip/mic)
  scenario-runner/  headless CI gate
  model-bench/   model matrix + integrated speech benchmark
docs/
  design/        the authoritative game-design documents
  research/      technical decisions and candidate manifest
  benchmarks/    reproducible evidence and NOT-TESTED list
  architecture.md
  testing.md
  web.md         browser build, OpenRouter, security model
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

1. **Whole-utterance neural TTS only.** Real Kokoro TTS and Whisper ASR are
   integrated and measured, but Kokoro exposes no incremental audio, so first
   audio arrives after synthesis completes (no streaming/barge-in yet). Live
   microphone/speaker I/O is implemented but unverified on hosts without audio
   devices; WAV fixtures cover CI.
2. **The 0.8B model is the only measured LLM.** Real GGUF inference,
   grammar-constrained tool decisions and the playable loop work, but raw
   0.8B decision quality is limited; larger models (2B+) are NOT TESTED. The
   simulation enforces knowledge boundaries, satisfied obligations and the
   designed bargain regardless of what the model proposes.
3. **Bevy client is a first pass.** `apps/bevy-game` renders the town and drives
   the authoritative simulation, but actors are flat coloured rectangles with no
   sprite art or walk animation, and the developer overlays are partial.
4. **No hardware matrix.** All numbers in `docs/benchmarks/` are from the
   development host and are explicitly labelled; Deck/GPU tiers are NOT TESTED.

See [`docs/research/2026-10-technical-decisions.md`](docs/research/2026-10-technical-decisions.md)
for the full decision log and candidate manifest.

## License

Code: MIT OR Apache-2.0. The bundled design documents are project
documentation. No third-party model weights are included in this repository.
