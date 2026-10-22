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
Synthesizes 5 s / 15 s / 40 s utterances, writes real WAV files, and reports
audio-seconds per wall-second, underruns and minimum buffer lead. It is
explicitly a **diagnostic, non-neural** device.

Two entry points run the identical measurement:

```bash
cargo run -p rpg_speech --example speech_bench   # CI: links only the speech stack
cargo run -p model-bench -- --speech             # dev host: same measurement
```

Hosted Linux CI uses the `rpg_speech` example because the combined `model-bench`
binary links llama.cpp and the static onnxruntime together, and their native
initialisers abort with `std::bad_alloc` on that runner before `main` runs. The
example links only the speech stack and covers the same WAV/buffer-accounting
path, so no coverage is lost. `model-bench --speech` remains the documented
developer command.

### 5. OpenRouter, browser runtime and browser build

* `rpg_openrouter` unit tests: strict schema rendering, request bodies,
  completion/usage/cost parsing, error classification and `Retry-After`,
  budgets, catalog checks, the SSE parser (every line ending, any chunking,
  split UTF-8, keep-alives, usage chunk, `[DONE]`, mid-stream errors,
  interrupted streams).
* `crates/openrouter/tests/native_mock.rs`: the native engine against a local
  mock HTTP server — streaming and non-streaming, 401/402/429/502, retries,
  dropped connections, prompt cancellation, deadlines, budgets, structured-
  output support checks, and remote decisions through the canonical commit
  path (including contract violations and impossible disclosures).
* `crates/runtime/tests/platform_neutral_core.rs`: save export/import,
  notices, streamed progress, inference accounting, delayed/duplicate/
  cancelled answers — the `SimCore` paths the browser relies on.
* `crates/web-runtime`: the browser scheduler and runtime with a fake
  dispatcher (fixed ticks vs frame rate, stall recovery, a 10 s outstanding
  request, cancel → stale, errors, saves, movement during inference).
* `apps/web-game/tests` (Playwright): the built site under `/IRL/` in
  Chromium (and Firefox in CI) — WASM init, WebGL2 canvas content, town and
  actors, ticking, walking, typed dialogue, the OpenRouter panel, mocked
  remote decisions committed canonically, invalid proposals, save/load/
  export/import, a 10 s delayed response, cancellation, API errors, budgets,
  and a key-leak sweep. `perf.spec.js` records performance numbers.
* Opt-in billable tests: `real_openrouter.rs` (`--ignored`) and
  `real-openrouter.spec.js` (`IRL_REAL_OPENROUTER=1`), run only by the manual
  `openrouter-real` workflow.

## Commands

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo run -p scenario-runner -- --seed 2026
cargo run -p rpg_speech --example speech_bench   # diagnostic WAV/buffer accounting
```

`make verify` runs the first four.

## CI (`.github/workflows/ci.yml`)

* `fmt` job: `cargo fmt --all --check`.
* `clippy` job: `cargo clippy --workspace --all-targets -- -D warnings`.
* `test` job (Ubuntu + Windows): `cargo test --workspace`, then
  `cargo run -p scenario-runner`, then the diagnostic speech benchmark via
  `cargo run -p rpg_speech --example speech_bench` (see §4).
* `web` job: browser dependency-isolation check, `clippy` for
  `wasm32-unknown-unknown`, `./scripts/web-build.sh`, then the Playwright
  smoke and performance specs against the built site; uploads the site
  (`irl-web`) and the Playwright report/traces.
* `web-pages.yml` deploys the site from `main`; `openrouter-real.yml` is
  manual and environment-gated.

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
