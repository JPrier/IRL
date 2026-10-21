# Technical decisions — Rust-first Agentic RPG (October 2026)

This is the decision log for the prototype. It separates **what was measured**
from **what is still a hypothesis**. No invented benchmark numbers.

## Host used for all measurements in this repository

| Item | Value |
|---|---|
| OS | Windows 11 (MINGW64 / MSYS2 shell) |
| CPU | 16 logical cores (development host) |
| Toolchain | `rustc 1.91.1`, host `x86_64-pc-windows-msvc` |
| GPU | **NOT RECORDED** |
| RAM | **NOT RECORDED** |
| Steam Deck | **NOT TESTED** |

Any number in `docs/benchmarks/` is from this host and must be re-measured
before it is used as a hardware requirement.

## Decisions made (with rationale)

### 1. Rust workspace, single authoritative simulation
**Decision:** `rpg_sim` owns the world and event log; all other crates are
clients or pure helpers.

**Why:** the design's core invariant is that model output cannot mutate truth.
Making this a crate boundary (not a convention) is enforceable in review and
testable.

**Rejected:** one crate where every module can mutate the world; a Python
simulation later rewritten in Rust.

### 2. Deterministic RNG in the domain
**Decision:** a small SplitMix64 implementation in `rpg_domain`.

**Why:** replay determinism is a product feature. Depending on a third-party RNG
whose algorithm can change across versions is a hidden reproducibility risk.

### 3. Typed actions + validator, not free-form tool JSON
**Decision:** a Rust `Action` enum is the capability surface. Models emit JSON
that is parsed into that enum and validated. Unknown actions are rejected.

**Why:** it is safer, faster and testable. It also lets the prompt enumerate a
small, context-appropriate shortlist.

**Trade-off:** novel actions must be decomposed into existing primitives. That is
the design's explicit stance.

### 4. Offers and commitments are separate from dialogue
**Decision:** `MakeOffer` creates a pending `Offer`; `AcceptOffer` transfers and
optionally creates a `Commitment` whose `QuestPredicate` is evaluated every tick.

**Why:** "the LLM said the quest is done" is not an outcome. World-state
predicates make completion auditable and replayable.

### 5. Perception is deterministic and subjective
**Decision:** Bresenham LoS, FOV cones, distance/wall-attenuated hearing, and
per-actor `observed_by` sets.

**Why:** spying and privacy are only meaningful if the observer set is computed,
not narrated. The scenario suite asserts that a spy in the shop does not observe
a whisper in the backroom.

### 6. Narrative director proposes; controller validates
**Decision:** `NarrativeChange` proposals are rejected in already-inspected
locations and when prerequisites are unmet. The mafia leader is seeded before
evidence is generated.

**Why:** prevents "rubber-band storytelling" and preserves genuine deduction.

### 7. No-LLM and scripted engines are first-class
**Decision:** `HeuristicEngine` and `ScriptedEngine` implement the same
`InferenceEngine` port as a future LLM.

**Why:** the whole game must be runnable and testable without downloading a
multi-GB model. These engines are also the latency baseline the research matrix
must beat.

### 8. Persistence: JSON save + SQLite event store
**Decision:** versioned JSON `SaveFile` for portability; SQLite for transactional
event storage and snapshots.

**Why:** JSON is easy to inspect and diff in tests; SQLite gives real
transactions and a migration path without a server.

### 9. Speech: real I/O and metrics, honest neural gap
**Decision:** implement WAV encode/decode, streaming buffer metrics, a clearly
labelled non-neural diagnostic synth, and a loud `MissingNeuralBackend`.

**Why:** the pipeline must be exercisable now, but a sine wave must never be
presented as evidence of neural TTS.

### 10. Text client first, renderer-agnostic simulation
**Decision:** ship a playable text client; keep Bevy out of the core crates.

**Why:** the gameplay loop and validation are the hard part and are fully
testable headlessly. Bevy 0.20 is available, but a renderer is a client concern
and can be added without touching `rpg_sim`.

**Deviation from the brief:** the brief asked for Bevy 0.20 as the initial
renderer. This increment does not include it (see "Not done" below). The
simulation deliberately does not depend on renderer types, so this is an
additive step rather than a rewrite.

## Candidate model manifest (NOT TESTED)

Machine-readable copy: `cargo run -p model-bench -- --list-models`.

| Model | Format | Params | Status |
|---|---|---|---|
| Qwen3.5-0.8B | GGUF 4-bit | 0.8B | NOT TESTED |
| Qwen3.5-2B | GGUF 4-bit | 2B | NOT TESTED |
| Qwen3.5-4B | GGUF 4-bit | 4B | NOT TESTED |
| Qwen3-1.7B | GGUF 4/5/8-bit | 1.7B | NOT TESTED |
| LFM2.5-1.2B-Instruct | GGUF | 1.2B | NOT TESTED |
| Gemma 3 1B | GGUF | 1B | NOT TESTED |
| heuristic (no-LLM) | in-process | — | VERIFIED-DETERMINISTIC |
| scripted (test double) | in-process | — | VERIFIED-DETERMINISTIC |

### Backend shortlist

| Backend | Why shortlisted | Status |
|---|---|---|
| `llama.cpp` via maintained Rust bindings | mature cross-platform, Vulkan, GGUF, structured output | NOT INTEGRATED |
| `mistral.rs` | Rust-native runtime, GGUF, structured outputs | NOT INTEGRATED |
| `sherpa-onnx` | offline streaming ASR + Kokoro/Piper TTS | NOT INTEGRATED |
| `whisper.cpp` | ASR alternative | NOT INTEGRATED |
| Pocket TTS | small native voices | NOT INTEGRATED |

### Licensing notes (must be re-verified before shipping)

* Piper development moved to a GPL-licensed repository; confirm the exact
  voice/model licenses before bundling.
* Pocket TTS model downloads may be gated and individual voices have different
  licenses.
* Keep a redistributable, ungated default voice pathway; gated models must be
  optional.
* Do not bundle model weights in this repository. The `assets/voices/` path is a
  documented install target, not a checked-in artifact.

## Not done / NOT TESTED (explicit)

1. Real LLM integration and the full model × hardware × workout matrix.
2. Real neural TTS/ASR installation, quality scoring and RTF on CPU/GPU.
3. Bevy 0.20 renderer, controller input, subtitles UI, mouse/keyboard.
4. Steam Deck / low-end GPU / AMD-Vulkan measurements.
5. Human blind playtests.
6. Distillation/fine-tuning (intentionally deferred until recurring errors are
   measured).
7. Full 90% line coverage instrumentation (tests exist; coverage tooling is not
   wired up).

## Next highest-value evidence-backed steps

1. Integrate `llama.cpp` behind `InferenceEngine`, then run the existing
   scenarios with a real 0.8B/2B/4B model on this host and record first-token and
   first-audio latency.
2. Integrate `sherpa-onnx` TTS with a redistributable voice, then re-run
   `model-bench --speech` and report true neural RTF and underruns.
3. Add a Bevy client that reads `Simulation` state; keep the text client as a
   headless diagnostic.
4. Measure on Steam Deck and a low-end integrated GPU before making any
   compatibility claim.
