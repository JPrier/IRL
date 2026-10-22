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

//!
//! ## Platforms
//!
//! [`core::SimCore`], [`protocol`], [`telemetry`] and [`backends`] are
//! platform-independent and compile for `wasm32-unknown-unknown`: the browser
//! client drives the very same `SimCore` from its event loop. The threaded
//! runtime ([`runtime`], [`workers`], [`wav_mic`], [`testing`]) uses native
//! threads and blocking channels and exists only on native targets.

pub mod backends;
pub mod core;
pub mod protocol;
#[cfg(not(target_arch = "wasm32"))]
pub mod runtime;
pub mod telemetry;
#[cfg(not(target_arch = "wasm32"))]
pub mod testing;
#[cfg(not(target_arch = "wasm32"))]
pub mod wav_mic;
#[cfg(not(target_arch = "wasm32"))]
pub mod workers;

pub use crate::core::{CoreConfig, Counters, InferenceStats, SimCore, TALK_RADIUS};
pub use backends::Backends;
pub use protocol::*;
#[cfg(not(target_arch = "wasm32"))]
pub use runtime::{spawn, RuntimeConfig, RuntimeHandle, ShutdownReport};
pub use telemetry::Telemetry;
#[cfg(not(target_arch = "wasm32"))]
pub use workers::{MicConfig, ASR_RATE};
