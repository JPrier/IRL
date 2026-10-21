# Testing and CI

Testing is a product feature here, not an afterthought. Every layer is exercised,
and the integrated simulated-player suite is the acceptance gate.

## Layers

### 1. Unit and property tests
Run by `cargo test --workspace`.

* `rpg_domain`: RNG determinism, relationship clamping, geometry, predicate
  evaluation, belief/commitment defaults. Plus **property tests**
  (`crates/domain/tests/properties.rs`) for RNG range, bounded values,
  relationship clamping under arbitrary deltas, and distance symmetry.
* `rpg_perception`: LoS on open floor, wall occlusion, FOV cones, whisper vs
  shout, contiguous line cells.
* `rpg_social`: clamping, threat/comfort/gift reactions, commitment
  satisfaction, rumor propagation with no duplicate delivery.
* `rpg_narrative`: rejecting clues in inspected locations, committing fresh
  clues, role-promotion eligibility, checkpoint activation.
* `rpg_sim`: pathfinding around walls, action validation, speech routing,
  offers/commitments, threat consequences, knowledge boundaries, clue discovery,
  movement, deterministic replay, JSON round-trip.
* `rpg_agents`: scripted engine, JSON action parsing, grocer bargain fallback,
  prompt redaction of inner-circle facts, scheduler priority/FIFO.
* `rpg_persistence`: JSON round-trip, SQLite idempotent bulk append, snapshot
  round-trip, restore checksum.
* `rpg_speech`: WAV round-trip, non-silent diagnostic PCM, distinguishable
  voices, loud missing-neural failure, streaming underrun accounting.

### 2. Deterministic integration scenarios
`crates/eval` builds the real town and runs a simulated player through the full
loop. `cargo run -p scenario-runner` executes all of them and exits non-zero on
failure.

| Scenario | What it proves |
|---|---|
| `friendly_investigation` | Conversations + investigation produce learnable state. |
| `grocer_conditional_bargain` | A requested favor is a real commitment, not flavor; disclosure waits on a world predicate. |
| `private_conversation_and_spy` | Privacy is spatial: moving to the backroom keeps a spy from observing a whisper. |
| `blackmail_has_consequences` | Threats raise fear **and** anger and reduce trust; compliance is not free. |
| `ignore_plot_world_progresses` | Time and faction budgets move while the fixed leader and player-paced plot do not. |
| `save_reload_continuity` | Commitments and the event checksum survive save/load. |
| `deterministic_replay` | Same seed + script ⇒ same canonical checksum. |
| `knowledge_boundaries` | NPCs/players cannot disclose unknown facts; prompts do not leak the leader. |
| `three_independent_routes` | Parallel clue chains and multiple eligible holders exist. |
| `innocent_grocer_stays_innocent` | Player attention never retroactively rewrites identity. |

### 3. Replay and persistence
* `EventLog::checksum()` is compared across two runs of the same script.
* `SaveFile` round-trips through JSON and `Simulation::restore` preserves the
  checksum.
* SQLite append is idempotent (primary-key `INSERT OR REPLACE`) and transactional.

### 4. Speech pipeline
`cargo run -p model-bench -- --speech` synthesizes 5 s / 15 s / 40 s utterances,
writes real WAV files, and reports audio-seconds per wall-second, underruns and
minimum buffer lead. It is explicitly a **diagnostic, non-neural** device.

## Commands

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo run -p scenario-runner -- --seed 2026
cargo run -p model-bench -- --speech
```

`make verify` runs the first four.

## CI (`.github/workflows/ci.yml`)

* `fmt` job: `cargo fmt --all --check`.
* `clippy` job: `cargo clippy --workspace --all-targets -- -D warnings`.
* `test` job (Ubuntu + Windows): `cargo test --workspace`, then
  `cargo run -p scenario-runner`, then the diagnostic speech benchmark.

Real-model and neural-voice jobs are intentionally **not** in hosted CI: they
need multi-GB assets and representative hardware. They belong in a self-hosted
scheduled job and must publish raw JSONL evidence.

## What is deliberately not gated

* Real LLM quality/latency (no weights in repo).
* Real neural TTS/ASR quality (no assets in repo).
* Steam Deck / GPU tiers.
* Human playtests.

These are listed as NOT TESTED in `docs/benchmarks/README.md` rather than being
silently skipped.
