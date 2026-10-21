# Implementation Agent Prompt — Rust-First, Voice-Native Agentic RPG Prototype

**Mission:** Build, run, benchmark, and validate a genuinely playable 2D vertical slice of an emergent quest-driven RPG. It must be the first production-quality increment of the eventual game, **not throwaway prototype code**. Implement the full local voice → interpretation → NPC decision → authoritative world action → voiced response loop, a constrained narrative director, and a repeatable model-evaluation harness. Favor a small, understandable, extensible system over a large mock game.

**Read first:** The provided design documents, `agentic_rpg_complete_design(1).md` and `emergent_quest_driven_rpg_design(1).md`, are the game-design specification. If present in the repository, read them in full; otherwise request/locate their supplied copies, and store them under `docs/design/` without losing details. This prompt determines *prototype execution priorities* where the earlier designs offer options. Preserve the original game concept and handwritten-notes coverage. Do not quietly omit requirements because they are difficult.

## 0. Definition of the deliverable

The final result is a **runnable desktop game**, a **headless simulation/test runner**, and a **local model and voice evaluation bench**, all sharing production Rust logic. A person must be able to install the documented assets, launch the game offline, move around a small town, speak to multiple NPCs through a microphone, hear their different generated voices, negotiate consequential actions, discover and resolve an investigation through different paths, and save/reload the results. The headless suite must be able to reproduce and inspect those experiences without a human.

The result is **not** acceptable if it is just an architecture document, a chatbot with a map, a text-only prototype with TTS promised later, a Python simulation due to be rewritten, an LLM demo that claims actions happened without game-state mutations, or a prerecorded voice/dialogue showcase. No reliance on paid/cloud inference at runtime. Local, offline operation is the default, including model assets once installed.

Work through the task in the repository, not merely in prose. Make concrete design decisions and document why. Implement and run code and tests; collect actual evidence. Where a physical Steam Deck or lower-tier GPU is unavailable, mark its results **NOT TESTED** and give reproducible commands for measuring it; do not invent numbers.

## 1. Non-negotiable constraints

1. **Rust owns the game.** Implement domain simulation, event log, NPC state/agents, story state, perception, quests, factions, scheduling, inference orchestration, audio pipeline, game UI and testing interfaces in Rust. Use a Rust-native 2D engine, initially **Bevy 0.20** unless a recorded, measured blocker justifies another Rust engine. Keep the renderer a client of engine-agnostic Rust world state. Avoid using Godot/GDScript, Python gameplay services, JS application logic, or disposable non-Rust implementations. FFI/native optimized libraries for ONNX/GGML/audio are allowed behind narrow Rust interfaces; explain their deployment and licensing.
2. **Voice always exists.** Working offline neural **STT/ASR and TTS** are required in the first integrated playable slice. It must include a usable default local speech backend and at least two distinguishable selectable NPC voices (or a documented alternate voice implementation if the chosen model supports only one). Speech must run through the same game systems as typed input. Typed input, subtitles, push-to-talk, mute, and audio-device fallback are accessibility/diagnostic options, **not excuses to ship without voice**. Test voice in the integrated demo and continuous regression suite.
3. **Smallest intelligent-enough model wins.** Model selection is a real research and experiment track. Do not declare a model optimal based on parameter count, generic leaderboard scores, or an isolated tokens/sec number. Compare tiny and somewhat larger models on the **exact same game inputs, hardware and semantic quality rubric**. The goal is the smallest model that meets quality, action reliability, latency, memory and coexistence requirements. Allow players/developers to swap candidates without recompiling or migrating NPC state.
4. **Generate speech faster than it is consumed.** Optimize the whole streaming path, not just raw token generation. Require the integrated LLM → text-chunk commit → TTS → audio buffer producer to remain ahead of human speech playback. Measure this under contention with ASR, renderer, memory retrieval and background agents. Define fast speech by audio-seconds-produced / wall-seconds elapsed, buffer underruns, first-audio latency, and sustained dialogue—not arbitrary tok/s claims.
5. **Simulation truth is authoritative.** LLMs may propose dialogue, actions, plans or flexible story details, but cannot change canonical state, mint resources, disclose inaccessible facts, skip validation, or overwrite established narrative truths. All state changes come from versioned typed events accepted by deterministic Rust controllers.
6. **Invest in tests immediately.** A fast, seeded, headless scenario harness, properties/invariants, replay, and CI gating must exist before large-scale content work. The benchmark/evaluation corpus is a reusable product component. No green CI based solely on mocked models or sample screenshots.
7. **No later rewrite plan.** All prototype subsystems must have final-tool-quality boundaries, serialized formats, testability, documentation, and migration strategy. Scope can be small; foundations should be real. If choosing an intentionally simplified algorithm, make it replaceable behind an explicit, tested contract instead of coupling it to everything.
8. **No reliance on every NPC continuously running an LLM.** Runtime inference is shared and sparse; deterministic policies run movement, schedules, and routine tasks. NPC identities, knowledge, and memories are persistent independent of active model sessions.
9. **Portable, affordable, local-first.** Support Linux x86_64 (including a SteamOS/Steam Deck target) and Windows x86_64; plan macOS as a portable backend target if feasible. CPU-only and GPU-accelerated paths must be investigated; do not silently require NVIDIA/CUDA. Restrict external services to development research, downloads and opt-in tooling, never required gameplay.

## 2. Research-first engineering decisions, with a running game as the goal

Before locking a backend, conduct short, reproducible capability spikes **in Rust**, then choose and integrate one default. Avoid spending the entire project on research. Write `docs/research/2026-10-technical-decisions.md` and a machine-readable candidate manifest including source URLs, licenses, exact versions, commits, hashes, model formats, quantization and host support. Keep rejected alternatives and rationale. Verify all source claims against current primary sources and actual APIs. Start with these leads, but expand the shortlist if newer verified models exist:

**Rust runtime/rendering:** Bevy 0.20 (released 2026-10-08), fixed-timestep world loop with deterministic game rules separated from ECS presentation. Benchmark a simple 800p scene and overlay, input, audio and event traffic. Do not tie canonical replay to Bevy entity IDs or render frame timing.

**LLM inference candidates:**

- Qwen3.5-0.8B — lower-capacity/latency extreme, not presumed adequate.
- Qwen3.5-2B — plausible middle tier.
- Qwen3.5-4B — richer-dialogue quality comparator.
- Qwen3 0.6B/1.7B and/or Gemma 3 1B as independent small-family controls if runtime support and licenses permit.
- LFM2.5-1.2B-Instruct as an additional speed/quality candidate, subject to its separate LFM licensing review.
- At least one stronger optional 4B–9B director or dialogue reference, if hardware permits; do not require it for the minimal tier.

Compare representative supported 4-bit, 5-bit and/or 8-bit weights as appropriate, and if useful low-bit variants, quantization-compatible KV/cache layouts, instruct/non-thinking settings, context sizes, constrained output, warm vs cold prefill and prefix caching. The fastest generic model may be the least suitable *NPC*. Qwen3.5 thinking and non-thinking selection must follow that family's actual supported chat template, not old Qwen3 toggles. Focus on concise decisions and naturally conversational short spoken responses; provide latency caps and output-length caps appropriate to each role.

**LLM backend candidates:** `llama.cpp` (through maintained Rust bindings or a packaged, persistent native backend) and `mistral.rs` (Rust-native embedding/runtime) are first choices. Compare actual GPU options (CUDA, Vulkan, HIP/ROCm, Metal where relevant) and CPU quantized execution, structured JSON/grammar enforcement, model support, first-token latency, resource usage, threading and deployment. `llama.cpp` is a mature cross-platform backend with Vulkan; `mistral.rs` advertises embedded Rust, GGUF and structured outputs. Use a persistent warmed engine; spawning an executable per utterance is forbidden. Do not hide the engine behind an untracked, manually installed developer server.

**Speech recognition / VAD candidates:** Sherpa-ONNX offline streaming recognizers (and Silero VAD where supported), plus Whisper.cpp with an appropriate tiny/base/small model as an alternative. Evaluate noise, interruptions, proper names, game vocabulary, false endpoints, accent tolerance, device throughput and input-to-stable-transcript latency. No web speech API. Consider endpointing and live partial transcripts, but never commit tentative transcripts as world actions.

**Neural TTS candidates:** Sherpa-ONNX running permitted Kokoro/Piper/VITS-compatible voice checkpoints; native-Rust Pocket TTS and its very small voices as a serious research track. Evaluate naturalness, character differentiation, pronunciation, short-chunk generation, startup, real-time factor, CPU load, licensing and exact asset availability. **Important:** the newer Piper development moved to a GPL-licensed repository; Pocket TTS model downloads can be gated and individual voice licenses differ. Do not accidentally make a proprietary/commercial-ready game depend on incompatible or inaccessible assets. Choose a redistributable or license-compliant **ungated default** voice pathway, document every required license/attribution, and make gated models strictly optional if permission is required.

**Other libraries:** Rust-friendly audio input/output and resampling (e.g., CPAL or Bevy audio), SQLite with transactional Rust access (`rusqlite` or `sqlx` if a concrete need warrants it), `serde` for versioned schemas, `tracing` for observability, `proptest` for invariants and a structured scenario/test runner. Use simple, maintained components and pin dependencies. Do not introduce a network microservice topology for a single-player local game unless a measured reason outweighs the overhead.

**Research outputs:** a working backend proof, integration experiment with speech synthesis, memory accounting, licensing table, and a reasoned initial default. A choice without a measured or source-backed rationale is not finished.

Primary sources to inspect (not conclusions in lieu of measurements):
- https://bevy.org/news/bevy-0-20/
- https://github.com/ggml-org/llama.cpp
- https://docs.mistralrs.dev/
- https://huggingface.co/Qwen/Qwen3.5-0.8B
- https://huggingface.co/Qwen/Qwen3.5-2B
- https://huggingface.co/Qwen/Qwen3.5-4B
- https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct
- https://github.com/k2-fsa/sherpa-onnx
- https://github.com/ggml-org/whisper.cpp
- https://github.com/kyutai-labs/pocket-tts
- https://huggingface.co/kyutai/pocket-tts
- https://github.com/OHF-Voice/piper1-gpl
- https://www.steamdeck.com/en/tech
- https://store.steampowered.com/hwsurvey/hw/

## 3. Required playable vertical slice

Build one **compact, handcrafted 2D town** with a readable map, walkable collision/navmesh or grid paths, approximately **6–8 distinct locations** and **12–15 persistent named NPCs** plus optional inexpensive background actors. Include a grocer and shop/backroom, warehouse, town guard office, gathering place, street or alley, residence, and one or two faction spaces. Provide a save/load menu, subtitles, inventory/notes interface, basic interactions, mouse/keyboard and controller controls, voice input and generated NPC voice playback. The prototype should feel like an intelligible game with minimal but consistent art and sound, not a debug terminal masquerading as one. Keep content data-driven, with placeholder art allowed if clearly intentional.

**Story:** Hidden Mafia. A criminal organization exists and attempts to control town shipments. Its existence, motives, and critical starting history are immutable anchors. For the first investigation, seed the identity of its leader *before* clues are created to make deduction testable. Keep the director's support for **flexible clue carriers, checkpoint selection and role promotion**. A separate configurable mystery mode may later allow unresolved candidates, but do not invalidate evidence or conflate the two modes. Include multiple independent paths to a major revelation—at least three, with failure recovery if an informant is alienated, leaves, or dies.

**Must be demonstrable during play, not merely described in a document:**

- Start a conversation by microphone and see transcription/target; receive an intelligent, context-sensitive, fully synthesized NPC reply. Repeat with different NPCs and distinguishable voices.
- Address an NPC based on position, facing and conversation context; allow explicit selection on ambiguity. Show who overheard according to hearing/occlusion, including a potential spy. NPCs know only what they actually perceived or were credibly told.
- Ask the grocer for a ledger. They may refuse and ask for a motivated favor (medicine, protection, debt relief, etc.); player can accept, renegotiate, pursue another route, steal or present leverage. These are **real conditional commitments and physical world predicates**, not flavor dialogue. The grocer may remain genuinely innocent despite repeated suspicion.
- Create a hidden quest by independently discovering damaging evidence and attempting blackmail. Respect actor knowledge, evidence possession, risk, fear/anger, and chance of retaliation; blackmail need not succeed.
- Ask for privacy. The NPC may lead the player to a nearby trusted room; test whether a listener can still hear. Privacy has real mechanical value; a spy can report information by a later plausible encounter, not instant global mind reading.
- Build trust/affection or fear through repeated behavior, possibly recruit a helper, create an obligation, and show a meaningful breach or betrayal condition. Distinguish trust, fear, affection, respect, anger, sadness, temptation and suspicion—not a single honor scalar.
- Observe NPCs follow their own schedules/goals when unattended. Show at least one background incident (theft/damage/business stress) and economically constrained faction response. Money, staff, inventory, action cost, delay and **cost of inaction** must affect whether repairs or investigations occur.
- Ignore the main plot while ordinary town life proceeds. The main conspiracy cannot complete or resolve itself offscreen by default. Advance major plot checkpoints based on explicit player discoveries/decisions.
- Ask about a secret the NPC could not know. They must not disclose it as fact; uncertainty, lies or mistaken rumors are allowed only when consistent with their beliefs and provenance.
- Let the director notice attention to an irrelevant NPC and **propose** a plausible new hook if an eligible uncommitted connection exists; the controller may accept or reject it. Do not force every suspicion to be correct or retroactively plant clues in thoroughly inspected places.
- Save, quit and reload; persistent consequences remain coherent across conversations and narrative branches.

Limit combat, animation fidelity, detailed national economics and huge maps for now. These are future extensions, not substitutes for the above systemic interactions.

## 4. Architecture and module boundaries

Create a Rust workspace with separable crates/modules analogous to:

```text
crates/
  domain/            # stable IDs, schemas, facts, beliefs, actions, policies
  sim/               # authoritative clock, events, inventories, resources, schedules
  perception/        # LoS, hearing, speech routing, occlusion, attention
  social/            # motives, relationships, commitments, rumor propagation
  narrative/         # anchors, checkpoints, clue carriers, director validation
  agents/            # NPC and director task construction, memory retrieval, context
  inference/         # shared local LLM provider trait, scheduler, decoding
  speech/            # capture, VAD, ASR, TTS, streaming, devices, interruption
  persistence/       # SQLite event store, snapshots, schema migrations
  game/              # Bevy scene, UI, interactions, debug overlays
  evaluation/        # seeded worlds, simulated players, quality scoring, bench harness
apps/
  desktop/           # playable executable
  scenario-runner/   # headless scenarios and replay
  model-bench/       # multi-model comparison and audio-performance harness
assets/             # authored map, character data, manifest, licensed sample assets
scenarios/          # deterministic acceptance stories and adversarial scripts
docs/               # architecture, research, evidence and results
```

Exact names may change, but separation of concerns may not. The pure world/domain crates should not depend on Bevy or model-runtime types. Keep a strictly validated flow: **transport/presentation → coordinator → authoritative controller → persistence/capabilities**. Model outputs are untrusted proposals. Do not let renderer, director, NPC prompt templates or audio listeners directly write world truth. Do not use one omniscient global prompt for all NPCs.

Expose interfaces such as:

```rust
trait InferenceEngine { /* model metadata, load/warm, constrained stream, cancel, metrics */ }
trait SpeechRecognizer { /* offline streaming capture/transcribe, finalization */ }
trait SpeechSynthesizer { /* warm voice, synthesize chunks, cancel, metrics */ }
trait MemoryRetriever { /* actor-scoped provenance-aware retrieval */ }
trait NarrativeProposer { /* bounded proposals, never direct commits */ }
trait ActionValidator { /* validate proposal against immutable & live constraints */ }
```

These are conceptual interfaces; use idiomatic Rust associated types or async/worker abstractions as profiling warrants. Avoid `Arc<Mutex<Everything>>`, per-NPC model loading, uncontrolled Tokio tasks, unbounded channels and per-frame database work. Prefer deterministic event queues and bounded specialized worker pools. Separate simulation clock time, wall time, audio sample timestamps and inference deadlines. No GPU inference or model-loading operation may block the render/simulation critical path.

**World state:** stable IDs, immutable truth, witnessed observations, actor-scoped beliefs (possibly mistaken), claims and provenance, relationships, current goals/plans, resources, faction budgets, inventories, knowledge, inspected locations, commitments, pending offers, plot anchors, uncommitted clue slots, and accepted story changes. NPCs must never get the full director's knowledge or hidden solution in their prompt.

**Tool/action API:** typed Rust enums + schema with capabilities such as `move_to`, `speak`, `ask_privately`, `make_offer`, `accept_offer`, `promise`, `transfer_item`, `share_claim`, `investigate`, `report_crime`, `hide_evidence`, `warn_contact`, `allocate_budget`, `repair`, `recruit`, `schedule_meeting`. Enumerate a compact contextual shortlist with actor, target, preconditions, estimated costs, relevant permissions and observability. Broader novel requests may be decomposed into sequences of *existing validated primitives*; no model-created new Rust tool implementations at runtime. Validate every hop and surface reasons for rejections. Distinguish proposed, pending/accepted and completed transactions. Spoken words cannot unilaterally complete a trade or promise for the player.

**Canonical log:** ordered append-only typed events, deterministic RNG seeds, schema migrations, idempotency keys, version/clock, recorded model proposal and validator result, memory references, causality links and observation recipients. Persist in SQLite with transaction boundaries and periodic snapshots; replay previously captured model decisions to reproduce canonical results **without rerunning stochastic inference**. A live-model *behavioral regression run* is a different, non-bitwise-deterministic evaluation mode. Log all external model/tool decisions sufficient for inspection and replay, without automatically retaining raw microphone audio by default.

**Perception:** 2D position, direction, field of view, walls/doors, acoustic zones or propagation, range, loudness/whisper, noise/attention, explicit current conversation membership. Deliver subjective events to listeners, not global transcripts. Expose perception cones/hearing heatmaps and why each actor was/was not a witness in a developer overlay. Addressee routing must handle unaddressed speech and nearby multiple candidates.

**Story:** deterministic story controller owns immutable anchors, inspected/committed clues, flexible checkpoints, eligibility rules, information packages and carrier placements, promotion restrictions and player-gated milestones. Director sees scoped summary and can propose but not assert novel canonical facts. Require independent discovery routes and validate that director proposals do not contradict player-observed facts or an NPC's realistic access. Allow NPCs to remain innocent; failure and negative outcomes are valid.

**Agent contexts:** build the shortest sufficient scoped prompt from identity, values, current motives and emotional state, nearby observable entities, relevant facts/beliefs with provenance, last few turns, active agreements and a small action shortlist. Retrieve memories by relevance/recency/salience and access rights; keep full event history outside the prompt. Compare prompt sizes and ablate retrieved fields. Use structured generation for action selection; generate the spoken line only after verifying that the proposed actions and knowledge are consistent. A combined action+dialogue model call is allowed **only if** its output can be held until authoritative validation and safely revised when rejected.

## 5. Full voice-native pipeline (not deferred)

Implement and demonstrate:

```text
mic / push-to-talk
 → optional VAD + acoustic endpointing
 → streaming local ASR (partial, then finalized transcript)
 → target identification + nearby acoustic observers
 → actor-scoped prompt + available action shortlist
 → shared, warmed local LLM (structured decision + candidate speech)
 → Rust authority validation / canonical event commit
 → safe incremental speech segments + local neural TTS
 → jitter/audio buffer → playback with positional volume + subtitles
 → audible speech perception events for other characters
```

Important causality: a tentative LLM fragment or ASR partial must **not** commit world state. If early TTS is desired, only play speech that is semantically safe given already validated decisions; alternatively stage a short completed turn before streaming voice. Compare split decision+speech, single constrained response and speculative synthesis. Measure which combination is faster *without* contradictions from rejected actions. NPCs hear actually emitted speech with position/time, not the internal candidate text. Cancel queued sound and superseded inference on interruption; support player barge-in, multiple speakers and simultaneous nearby listeners. Prevent NPC playback from immediately transcribing into the player's mic where practicable; include headphones/echo guidance, push-to-talk and non-feedback routing. Keep UI controls responsive throughout.

**Timing instrumentation, per utterance:** microphone onset/end, VAD endpoint, ASR first partial and final, routing, memory assembly, prompt tokens, prefill, first token, structured decision, validation, TTS first buffer, first audible sample, last emitted token, audio consumed, total response, queue wait, GPU/CPU/RAM/VRAM, KV cache, cancellation, underrun count and frame-time impact. Trace overlapping work with a common ID.

**Speed requirement:** Let `audio_seconds_generated(t)` be playable speech seconds produced by *the complete accepted-text-to-synthesis pipeline* while `wall_seconds(t)` is elapsed time since output production began. For sustained utterances, report `speed_ratio = audio_seconds_generated / wall_seconds` and the minimum buffered-audio lead throughout playback, including worst-case local model prefill, TTS phoneme segmentation and competing workloads. Target **≥1.5× real-time at the P05 (slow-tail) of sustained integrated production** with **≥2× as a stretch goal**, and **zero buffer underruns in acceptance dialogues**. Measure time to first audible speech separately; set provisional interactive goals such as P95 <2 seconds after speech endpoint on the selected reference desktop, and investigate Deck-specific limits independently. These are **targets, not claims of achieved behavior**. Establish empirical per-hardware thresholds and don't fake a passing result by shortening dialogue to a one-word greeting. Evaluate 5-, 15- and 40-second NPC replies, bursts of rapid turns, multiple voices and interrupted speech.

A separate pure TTS real-time factor is necessary but insufficient. Report tokens/sec and words/sec as supporting data, not the end-to-end acceptance test. The system must produce text quickly enough *and* synthesize enough corresponding audio quickly enough to stay ahead of playback. Test thermal throttling, low power, shared memory pressure, and CPU/GPU contention.

**Asset packaging:** reproducible voice/model manifest with license, provenance, expected checksum, size, local cache path and download workflow. After initial authorized download/installation, run air-gapped. Package or automate installation of the actual default voice assets—never a hidden internet request on first NPC response. Fail clearly if a required file is missing. Include a voice-test screen and automated speech-to-file smoke test; do not accept a fake sine wave/recorded WAV as evidence neural TTS works.

## 6. Research: model Pareto frontier and playtest system

Build the **model comparison tool as a permanent part of the product**, not a one-off spreadsheet. It must support a model-profile file per candidate (`model ID`, artifact revision/hash, quant, prompt template, inference backend/commit, device config, max context, decoding config, speech model/voice, run seed). The game must be able to run the *same saved world state and player utterance* through any profile. Record all output, decisions, validators, generated audio, timing and resources. Add an in-game developer selector and headless batch tool; changing a model may clear model KV caches but must not erase NPC memory or world truth.

**Benchmark matrix:**
- Hardware: CPU-only commodity 16GB PC; representative low-end integrated graphics; NVIDIA GPU at modest VRAM; AMD/Vulkan; actual Steam Deck LCD or OLED if available; faster reference PC as quality control. Use September 2026 Steam survey to justify tiers. Steam Deck has a 16GB shared-memory, 4-core/8-thread AMD APU and a tight 4–15W power envelope: do not extrapolate high-end PC results to it.
- Models: at least four distinct small/medium checkpoints across ≥2 families if feasible, with 0.8B / ~1–2B / ~4B size coverage. Choose comparable quantizations and repeat with at least one alternate quant for the strongest candidate. Also test no-LLM deterministic routine/intent paths as latency baselines; they are *not* a substitute for conversational intelligence.
- Workloads: (a) short benign speech, (b) tough or manipulative multi-turn discussion, (c) conditional quest bargain, (d) NPC knowledge restriction, (e) emotion/relationship reaction, (f) novel valid problem solving via primitive actions, (g) director clue proposal/consistency, (h) overhearing/rumor, (i) background event contention, (j) 10+ min continuity and save/reload. Run cold/warm, several seeds and repeated trials; randomize test order to avoid misleading thermal/cache effects.
- Metrics: first-token and first-audio P50/P95/P99; sustained integrated audio-speed ratio and underruns; output/prefill tokens and time; CPU/GPU utilization, peak RAM/VRAM, audio/ASR footprint and renderer FPS/frame-time; energy/battery if available; output schema validity; action proposal legality; intent accuracy; hallucinated knowledge, unsupported facts, narrative contradiction, commitment tracking, consistent personality, in-character dialogue, perceived naturalness, initiative, world-impact diversity and refusal quality.

**Quality scoring:** use an independent deterministic validator for knowable constraints; add blind, counterbalanced human playtests comparing voices and anonymized models for coherence, character believability, conversational variety, memory fidelity and emergent quest credibility. Optional stronger-model grading may help triage cases but cannot be sole ground truth and must not leak test answers into model prompts. Store transcripts, world diff, accepted/rejected actions, anonymized ratings and failure taxonomies. Use held-out paraphrases and adversarial cases; do not select a tiny model purely by overfitting a known prompt suite.

**Selection procedure:** first apply **hard validity gates** (no canonical contradiction, no illicit item/fact disclosure, no unauthorized commits, no missing neural voice support); then quality floors, integrated audio/smoothness and memory constraints. Among those passing, select the minimum-memory/smallest viable model on the latency-quality **Pareto frontier**, with a documented quality/high-performance option. Evaluate whether a tiny decision router and a larger dialogue generator, or a rare larger director model, beats one shared model once load/scheduling/VRAM overhead is counted. Don't add models solely because architectural diagrams look impressive. If no candidate meets low-end requirements, be candid and leave the gap open with data: do not misrepresent Steam Deck as supported. Expose user-adjustable quality/performance profiles for future expansion.

**Distillation/fine-tuning:** only after concrete recurring errors are measured. Capture opt-in, licensed training/evaluation traces, isolate holdout data, compare smaller fine-tunes to prompting/constrained-decoding improvements, and count runtime memory + development complexity. Do not prioritize training a custom model before the playable slice.

Deliver `model_bench` CLI, CSV/JSONL raw runs, an HTML/Markdown summary, concise leaderboard and model cards. Output must be easy to compare side by side and rerun via one command. Avoid reducing a whole NPC to a single opaque AI quality score.

## 7. Simulation, replay and adversarial test suite

Build tests alongside each module. Require `cargo fmt --check`, `cargo clippy -D warnings` (scoped with justified explicit lints if necessary), `cargo test --workspace`, Rust feature/backend smoke builds, deterministic headless scenarios, SQLite migration/roundtrip tests, lint/license checks and reproducible benchmark-report validation in CI. Run expensive full-model evaluations in a scheduled or self-hosted hardware job where normal hosted CI cannot download multi-GB models; make required regression profiles mandatory before releases. Provide a `make`/`just`/Cargo task for one-command developer verification and a documented model-asset cache.

**Use multiple complementary test layers:**

1. **Unit/property:** truth vs beliefs, event invariants, transaction conservation, impossible action rejection, offers/commitments, actor permissions, observed/committed clues, resource prioritization, stable IDs, schema migrations, navigation, propagation delay, hearing/LoS including occlusion. Property-based tests on random worlds and event sequences.
2. **Deterministic integration:** scripted actions and recorded model proposals, committed event log checksums, save/load at arbitrary points, same seed+same recorded proposals → identical canonical outcome, crash recovery, no double-spend, rollback of rejected action. Run many seeds.
3. **Model-contract tests:** structured decoding validity, bounded prompts, timeouts, cancellations, context scoping, malicious player utterances trying to command the model to reveal hidden facts, fake tool calls, speculative dialogue contradicted by rejected action, refusal or fallback behavior. Never mistake plausible prose for proof that game state changed.
4. **Real audio integration:** mic fixture → VAD/ASR → target → LLM → validated action → actual generated nonzero speech PCM → playback or a file sink. Test multiple voices, audio duration, speech speed, interrupted turns, ASR errors, audio device missing, echo, overlapping speech and CPU load. On machine without microphone, use real recorded fixture rather than mocking recognizer; separately smoke-test live microphone where available.
5. **Headless full-game scenarios:** scripted or model-based simulated players, branching choices, invariants and material outcome assertions. Include friendly investigation; grocer obsession with both valid tie-in and genuine innocence; favor/persuasion/payment/stealth/blackmail to a common discovery; manipulate someone after secretly harming them without premature blame; whisper and move to a private space while spy observes; spread a false rumor; create a crew and risky favor; promise then renege; break infrastructure and test triage, bankruptcy and inaction; lose/kill an informant and recover through alternate leads; challenge an NPC with inaccessible knowledge; try to make the director ratify an unsupported conspiracy; ignore plot for long periods while world changes; save during an in-flight conversation; replay same seed under aggressive/charitable/curious play styles and detect divergent **canonical events**, not merely varied dialogue.
6. **Performance/soak:** long saves and 1,000+ simulation ticks; burst player requests with queued P1/P2 tasks; 15 local named actors plus crowd; latency distributions; memory growth/leaks; inference starvation; save/reload; lower settings; thermal impact where measurable.
7. **Human exploratory review:** randomized blind playtests and bug triage for unnatural conversation, character inconsistency, forced quests, spurious suspicion validation and empty dialogue loops.

**Coverage policy:** instrument coverage for pure Rust simulation, narrative, perception and social modules. Aim for ≥90% line coverage overall for these modules, with explicit tests for 100% of defined canonical invariants and especially high branch coverage on validators and narrative commitments. Coverage numbers are not a substitute for mutation/property and scenario tests. Baseline and enforce thresholds once scaffolding settles; do not water down gates just to make a report green. Declare platform-dependent/audio hardware tests separately instead of reporting them as executed when they were skipped.

**Gating:** no PR may regress invariant tests, replay/saves, headless acceptance, voice smoke, critical knowledge boundaries, or model-interface schema compatibility. Any deliberate performance/quality baseline change requires checked-in benchmark evidence and a rationale. Differentiate fast PR CI, full-model nightly/release CI, platform-specific self-hosted CI and manual Deck evidence. Include reproducible seeds and test fixtures in the repo. Make failures diagnostic: replay ID, actor, observed state, proposed action, rejected constraint, prompt metadata, model hash and event diff.

## 8. Developer tools and observability

Provide a toggleable in-game debug view showing:

- NPC identity, current goal, planned next action, emotional/relationship state, faction membership, money/resources and routine.
- Sight cones, audibility/range/occlusion, addressed NPC and overhearers, spy observations and rumor hops.
- Each NPC's actual observations/beliefs vs world truth (omniscient view only in explicitly marked developer mode).
- Latest bounded prompt with token count, accessible actions, retrieved memories and reasons for retrieval.
- LLM queue priorities P0 player / P1 nearby NPC / P2 director/faction/memory, dropped/canceled tasks, prefill/generation timing and cache hits.
- ASR partial/final, active voice and TTS chunks/buffer fill, first audio and underrun timing; toggles for latency/quality profiles.
- Director proposals, accepted/rejected story changes, anchored facts, pending vs committed checkpoints, clue placement and route availability.
- Event chronology, commitments, inventory/budget transfers, time advance and explicit rejected action causes.

Export a whole session diagnostic bundle **without private microphone recordings by default**. Each test failure must reference enough structured state and event logs to reproduce it.

## 9. Execution plan, prioritization and proof

You are expected to independently execute these work packages, each finishing in a runnable, testable condition before starting the next. These are integration milestones, **not permission to stop before the voice-enabled complete slice**:

**A. Repository baseline:** inspect existing code/design, write a short gap matrix, identify target platforms and establish Rust workspace, tools/CI and a deterministic seedable headless world.

**B. Game loop / truth:** working Bevy town, movement, schedules, interaction, canonical log/SQLite snapshots, basic inventory, goals, people and faction budgets. Add visualization and property tests immediately.

**C. Agent and dialogue path:** actual local LLM runtime, scoped NPC contexts, action shortlist/schema, controller validation, committed outcomes, persistent memories and multi-turn exchange. Ensure model swapping and recorded replay.

**D. Voice vertical slice:** actual local microphone/VAD/ASR and neural TTS voices, streamed or safely chunked, with audio device management, subtitles, player input and meaningful measurement. This is a hard early milestone; do **not** leave it to the last phase.

**E. Game systems:** perception, spies, knowledge-provenance, rumor, private conversation, emotions/relationships, organic quests, conditional promises, multiple paths/blackmail, small recruitable crew, background economy/repair events and faction action.

**F. Director / investigation:** immutable Mafia anchors, seeded leader, flexible checkpoint/carrier proposals, valid promotion or innocence for grocer, independent leads, director controller and player-paced major story progression.

**G. Research and optimization:** run the real model + ASR/TTS matrix, instrument end-to-end throughput, find smallest acceptable option, quantify resource contention, generate reproducible comparisons, validate or candidly disqualify Steam Deck until measured.

**H. Harden/release:** full invariants and simulated-player suite, architecture and setup documentation, model asset bootstrap, clean offline install smoke, packaged Linux/Windows release artifacts or clearly reproducible release commands, known limitations and playtest guide. Run an adversarial review for contradictions and incomplete experiences.

Favor horizontal integration every day: a tiny functioning microphone-to-voiced-reply-to-real-action loop beats eight disconnected subsystems. No fake sample outputs, incomplete stubs hidden behind success logging or simulated benchmark numbers. Design the abstractions once, integrate real backends early, and iterate through repeatable tests.

## 10. Final definition of done / acceptance demonstration

**The deliverable is done only if all of the following can be honestly shown with code and evidence:**

- Fresh checkout → documented bootstrap/install → launch 2D game. Local model files can be provisioned legally; thereafter game runs without internet, subscriptions, server dependencies or Python.
- Move around town and converse using a real microphone; hear actual synthesized, distinguishable NPC voices, with typed and accessibility alternatives.
- A natural voiced negotiation produces an actual conditional offer and a lasting obligation or refused/alternative resolution. Save/reload preserves it.
- A spy realistically overhears some conversations but not an appropriately private one; the information arrives later by a plausible propagation chain.
- Different player strategies cause provably different resources, relationships, discoveries or faction responses; NPCs act independently; background world progresses while major story remains player-paced.
- Director can attach a valid previously uncommitted hook to an eligible ordinary NPC, and can reject an unsupported connection without changing observed truth.
- The mafia investigation has three independent viable routes; the game survives at least one lost/hostile clue-holder and enforces a fixed seeded identity for its first mystery mode.
- Actual small and medium LLMs have been compared, on the same recorded scenarios, with real integrated ASR/TTS latency, speech-speed, coherence and RAM/VRAM measurements. The chosen default is justified by evidence. If hardware is unavailable, publish the executable benchmark harness and **explicitly record which measurements are missing**.
- Replay and save/load are deterministic for recorded proposals, all relevant contract/invariant tests pass, full-game simulated-player tests are automated, and CI gates future regressions.
- No runtime component requires rewriting out of another programming language to become the final product. Remaining work is gameplay/content expansion or performance improvement behind established interfaces, not an architecture replacement.

**Final response format after implementation:** provide a concise playable-demo description; exact run/build/test commands; links to source changes and checked-in documentation; a table of platforms actually verified vs not tested; selected initial model/quantization and voice backends with licensing; model/latency/quality/VRAM evidence and known gaps; CI results; and the next highest-value evidence-backed improvement. Do not label anything 'working', 'Steam Deck compatible', 'high coverage', or 'faster than speech' without reproducible proof.

**Important execution rule:** Do the implementation. Research and architectural choices are inputs to the working game, not the end result. Keep working through failing builds/tests and make a best-effort playable integrated result using actual local models and neural voice; plainly document anything you cannot substantiate.
