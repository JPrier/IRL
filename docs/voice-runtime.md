# Voice runtime: nonblocking voice and a continuously simulated world

The graphical client's runtime lives in [`crates/runtime`](../crates/runtime).
Its one non-negotiable invariant:

> No microphone operation, VAD call, ASR operation, LLM request, TTS
> generation or audio playback may block authoritative simulation
> progression. The world keeps living while the player speaks.

## What was wrong

Before this change, `apps/bevy-game/src/worker.rs` ran the authoritative
`Session` *and* the whole voice pipeline on one thread. `Command::VoiceTurn`
captured up to 6 s of microphone audio, ran Whisper, ran the model, synthesized
the full reply and then waited for the speaker queue to drain — all inside one
command handler. Ticks stopped for the entire interaction, NPCs froze, commands
piled up, and the renderer locked out all input through `snap.busy`.

Recording was broken independently of that:

| Defect | Effect |
| --- | --- |
| `audio.rs::push_mono` already downmixed the device stream to mono, then `capture_utterance` called `deinterleave(chunk, native_channels)` on it again. | On a stereo microphone (the default device on the development machine is 48 kHz × 2) every chunk lost half its samples: speech reached VAD/ASR at double speed and half duration, so it was rarely detected or transcribed. |
| The capture stream ran continuously but was only read during a turn; its 64-chunk queue filled with old audio and then **dropped new audio**. | The first part of every recording was stale room audio from before `V` was pressed, followed by lost speech. |
| The VAD and resampler were never reset between turns. | State from the previous utterance leaked into the next. |
| `V` started a fixed-length capture on a single key press although the UI said "hold V". | Release did nothing; the turn ran its full window. |
| Linear resampling without a low-pass filter. | 48/44.1 kHz input aliased energy above 8 kHz into the 16 kHz ASR band. |

## Architecture

```text
  renderer ──Command──►┐                              ┌──► irl-mic        (cpal + VAD)
                       │   irl-sim                    ├──► irl-asr        (Whisper)
  irl-mic  ──event──►  ├─► inbox ─► SimCore ─effects──┼──► irl-inference  (llama.cpp, resident)
  irl-asr  ──event──►  │   fixed tick schedule        └──► irl-tts        (Kokoro + speaker)
  irl-infer ─event──►  │        │
  irl-tts  ──event──►  ┘        └──Snapshot──► renderer (bounded queue)
```

* **`irl-sim`** owns the canonical `Session` inside `SimCore`. It waits only on
  its own inbox, with a timeout equal to the time left until the next tick.
  Every send it performs is on an unbounded channel, every telemetry read is
  an atomic, and the one cross-thread interrupt (cancelling the request that is
  inside the model) is an atomic store. `SimCore` is pure with respect to time
  and I/O — inputs are messages plus a timestamp, outputs are `Effect`s — so
  the deterministic tests drive it step by step.
* **Workers** each own one native backend, created on that thread by a factory
  so audio streams and ONNX/llama handles never cross threads. They may block
  inside their backend; nothing waits on them.
* **Snapshots** go to the renderer over a bounded queue with `try_send`, so a
  stalled or minimised window can neither block nor bloat the simulation.

### The conversation state machine

`Idle → Listening → Transcribing → Deciding → Validating → Synthesizing →
Speaking → Idle`. Every stage has a deadline; cancellation (`C`), barge-in
(pressing `V` while an NPC speaks), errors, timeouts and loading a save return
to `Idle` from any stage. A press of `V` while a turn is transcribing or
deciding is refused with a visible message; turns are never queued silently.

Typed dialogue (`Enter`) and transcribed speech enter the same
`SimCore::player_turn`, so both take the identical gameplay path.

### Handling a world that changes while an NPC thinks

A decision request is rendered from the world at *t₀* and answered at *t₁*.
Each request carries the conversation id, a request id and the world epoch
(bumped by every load). On completion `SimCore`:

1. drops anything whose ids, epoch or stage are no longer current (counted as
   *stale*): late answers after a cancel, timeout or load; duplicates;
   out-of-order answers from superseded conversations;
2. re-checks the situation in the **current** world — the NPC is still alive,
   present and within earshot (`invalidated` otherwise);
3. resolves the model output against the current world (canonical agenda and
   obligations included) and commits it through
   `Session::commit_npc_decision`, the same commit path every client uses, so
   provenance (model / canonical / fallback) is recorded identically;
4. if validation rejects a model proposal (an item changed hands, an offer
   expired…), regenerates once from the fresh world; otherwise the NPC says
   nothing;
5. voices only a committed action's own words, tagged with the current audio
   epoch.

A model that exceeds the decision deadline is interrupted and the
deterministic policy answers (provenance: `heuristic-fallback (inference
error: … timed out …)`).

### Audio contract

* Device callbacks downmix to mono **exactly once** (`rpg_speech::dsp`), with
  one allocation per device buffer and a non-blocking `try_send`.
* Everything after the callback is mono `f32` with its sample rate attached.
* `LinearResampler` positions samples with exact integer arithmetic (output is
  identical however the input is chunked; no drift) and low-pass filters before
  downsampling.
* Push-to-talk start clears the device queue, the resampler and the VAD. The
  capture queue is drained continuously, so it never fills with stale audio,
  and the level meter is live even when idle.
* Playback buffers carry an epoch. Moving the shared epoch (cancel, barge-in,
  load, quit) makes the device callback discard stale audio within one device
  period — including a half-played buffer. Completion is detected from the
  device's consumed-sample counter, never by sleeping.
* Kokoro streams one chunk per sentence through sherpa-onnx's progress
  callback; returning `false` aborts generation. On the development machine a
  3-sentence line reached the speaker after 734 ms instead of 2,235 ms.

## Diagnostics

`F3` shows, along the pipeline: simulation tps / delay / max lag / skipped and
renderer fps; capture device and format, samples/s, RMS, peak, dropped
buffers, recording length; VAD state; ASR latency and transcript; conversation
and request ids, the running request, queue wait and inference time; TTS
first-audio and synthesis time; playback state and underruns; worker queue
depths; and cancelled / stale / timeout / refused / regenerated / invalidated
counters plus recent state transitions.

`bevy-game --dump-utterance` also writes the exact 16 kHz audio sent to ASR to
`last_utterance.wav`.

## Tests

* `crates/runtime/tests/core_state_machine.rs` — deterministic, no threads:
  every stage in order while the world ticks; typed vs spoken parity; cancel
  then late answer; duplicates; out-of-order answers; NPC walking away (by
  injection and by its real schedule); regeneration after an item changes
  hands; timeouts at every stage; load and save during inference; refused
  push-to-talk; barge-in; failures at every stage; no-speech diagnoses.
* `crates/runtime/tests/nonblocking_runtime.rs` — the threaded production
  runtime with gated fake devices/models: ticks continue (at ~100 % of the
  target rate) while recording silence, recording speech, transcribing, with
  the model stalled (NPCs and the player keep moving), synthesizing, playing,
  after a cancel, on a model timeout and through a device error and recovery;
  a deliberately slow 10-second inference; mono/stereo at 16/44.1/48 kHz;
  repeated recordings; an empty microphone; stale pre-press audio; rapid `V`;
  self-test WAV; device switching; save/load mid-inference; clean shutdown
  while recording and while the model/playback are busy; sustained load; a
  renderer that stops reading snapshots.
* `apps/bevy-game/tests/real_voice.rs` (ignored; needs weights, assets and an
  output device): runs the built `bevy-game --voice-check`, i.e. the real
  Silero/Whisper/llama.cpp/Kokoro stack through the production runtime with a
  WAV-fixture microphone, and fails on a missing transcript or a slowed tick.

```bash
cargo test -p rpg_runtime                                   # CI, no devices or weights
cargo test --release -p bevy-game -- --ignored              # real models
```

## Manual hardware test procedure

Run these on the target machine; record the outputs in the PR or a playtest
note.

1. **Devices.** `bevy-game --list-devices`. Confirm the microphone you speak
   into is listed (and which is the default).
2. **Raw capture.** `bevy-game --mic-selftest 5` (add `--mic "<name>"` for a
   non-default device) and speak for the full five seconds. Expect
   `OK: the microphone records audio`, an effective rate within 20 % of the
   device rate, and a peak above ~0.05. Play `mic_selftest.wav` and listen.
   * `no audio arrived` → capture failure (driver/device).
   * `pure digital silence` → muted device, nothing plugged into the jack, or
     Windows *Privacy & security → Microphone → Let desktop apps access your
     microphone* is off.
   * `very quiet` → raise the input level or move closer.
3. **Pipeline with real speech, no microphone.** `bevy-game --voice-check
   --mic-wav .cache/speech/fixture-ledger.wav --near grocer --turns 2`. Expect a
   transcript, an NPC reply you can hear, `voice-check: PASS`, and every stage
   in the tick table near the target rate.
4. **Pipeline with your voice.** `bevy-game --voice-check --hold 4 --near
   grocer` and speak during each 4-second window.
5. **In game.** `make play`, walk next to an NPC, open `F3`.
   * Hold `V`: `● REC` appears and the meter moves with your voice; `speech`
     appears while you talk. Release: `heard: "…"` shows the transcript.
   * While the NPC is *thinking*, walk with WASD and watch other NPCs move; the
     tick counter keeps rising and `tps` stays at the target.
   * While the NPC speaks, press `C`: audio stops at once. Press `V` while it
     speaks: it stops and recording starts.
   * Ask something, press `F9` (load) before the answer: no answer and no audio
     arrive from the old timeline; `stale` increments.
   * Tap `V` quickly, hold `V` in silence, unplug the microphone while holding
     `V`: each gives a specific message and the game stays responsive; `F7`
     switches devices, `F8` runs the self-test in game.
   * `Enter`, type a line, `Enter`: the NPC answers through the same path.

## Measured on the development machine

Windows 11, release build, Qwen3.5-0.8B Q4_K_M, Whisper tiny.en, Kokoro,
target 8.3 tps (`--voice-check` with a WAV microphone):

| Stage | Wall s | Ticks | tps |
| --- | ---: | ---: | ---: |
| Listening | 4.86 | 39 | 8.0 |
| Transcribing | 0.04 | 1 | — |
| Deciding (≈6.2 s per model call) | 12.41 | 104 | 8.4 |
| Synthesizing | 1.15 | 10 | 8.7 |
| Speaking | 4.88 | 40 | 8.2 |

Max tick lag 29–32 ms (Windows timer granularity under full model load), no
skipped ticks.

The live microphone on that machine (`Microphone (High Definition Audio
Device)`, 48 kHz × 2) was opened and delivered audio at the correct rate, but
the signal was pure digital silence (endpoint unmuted at 88 %, privacy
access allowed) — consistent with nothing connected to that input. Live
speech through a physical microphone therefore remains to be verified with
steps 2, 4 and 5 above.
