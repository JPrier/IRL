//! # rpg_runtime
//!
//! The realtime runtime behind the graphical client.
//!
//! The authoritative simulation runs on its own thread at a fixed tick rate
//! and is the only owner of canonical world state. Everything slow or
//! device-bound lives on a dedicated worker:
//!
//! | thread          | owns                                  |
//! |-----------------|---------------------------------------|
//! | `irl-sim`       | [`core::SimCore`]: world, conversation state machine, validation, commits |
//! | `irl-mic`       | capture device, resampling, VAD, recording |
//! | `irl-asr`       | speech recognizer                     |
//! | `irl-inference` | the resident language model           |
//! | `irl-tts`       | neural voice + output device          |
//!
//! Workers communicate with the simulation only by message, and every
//! response carries correlation ids so the simulation can reject anything
//! stale. **No microphone, VAD, ASR, model, TTS or playback operation can
//! block a simulation tick** — the invariant the tests in `tests/` measure
//! directly, at every voice stage.
//!
//! Native backends are injected through [`backends`]; [`testing`] provides
//! controllable fakes so the production pipeline runs in ordinary CI without
//! devices or model weights.

pub mod backends;
pub mod core;
pub mod protocol;
pub mod runtime;
pub mod telemetry;
pub mod testing;
pub mod wav_mic;
pub mod workers;

pub use crate::core::{CoreConfig, Counters, SimCore, TALK_RADIUS};
pub use backends::Backends;
pub use protocol::*;
pub use runtime::{spawn, RuntimeConfig, RuntimeHandle, ShutdownReport};
pub use telemetry::Telemetry;
pub use workers::{MicConfig, ASR_RATE};
