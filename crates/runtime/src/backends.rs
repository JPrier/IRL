//! The seams between the runtime and native libraries.
//!
//! The runtime only talks to these traits, so the same production code paths
//! run against real devices and models in the game and against controllable
//! fakes ([`crate::testing`]) in CI. Factories are `Send` closures that are
//! *invoked on the worker thread that will own the backend*: native handles
//! such as audio streams never cross threads.

use crate::protocol::MicInfo;
use rpg_domain::ports::{InferenceEngine, SpeechRecognizer};
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Microphone
// ---------------------------------------------------------------------------

/// An open capture device delivering mono `f32` chunks at
/// [`MicInfo::sample_rate`] (see the audio data contract in
/// `rpg_speech::dsp`). Implementations must never block.
pub trait MicSource {
    fn info(&self) -> MicInfo;
    /// Next queued mono chunk, if any.
    fn try_recv(&mut self) -> Option<Vec<f32>>;
    /// Drop everything queued; returns the number of samples discarded.
    fn clear(&mut self) -> usize;
    /// Device buffers lost because the consumer fell behind.
    fn dropped_chunks(&self) -> u64;
    /// An asynchronous stream error (device lost, permission revoked…).
    fn take_error(&mut self) -> Option<String>;
}

/// Opens capture devices. Called repeatedly (device selection, recovery).
pub trait MicBackend: Send {
    fn open(&mut self, device: Option<&str>) -> Result<Box<dyn MicSource>, String>;
    fn list_devices(&mut self) -> Vec<String>;
}

/// A voice-activity detector over 16 kHz mono PCM.
pub trait Vad {
    /// Feed samples; returns any speech segments that just finalized.
    fn accept(&mut self, samples: &[f32]) -> Vec<Vec<f32>>;
    /// Finalize trailing speech.
    fn flush(&mut self) -> Vec<Vec<f32>>;
    /// True while the detector currently hears speech.
    fn speaking(&self) -> bool;
    /// Forget all state and pending segments.
    fn reset(&mut self);
}

pub type VadFactory = Box<dyn FnOnce() -> Result<Box<dyn Vad>, String> + Send>;

// ---------------------------------------------------------------------------
// Speech recognition and inference
// ---------------------------------------------------------------------------

pub type AsrFactory = Box<dyn FnOnce() -> Result<Box<dyn SpeechRecognizer>, String> + Send>;

/// A loaded, resident inference engine.
pub struct LoadedEngine {
    pub engine: Box<dyn InferenceEngine>,
    /// Flag that aborts the in-flight completion when set, if the backend
    /// supports cooperative cancellation (llama.cpp checks it per token).
    pub interrupt: Option<Arc<AtomicBool>>,
}

pub type EngineFactory = Box<dyn FnOnce() -> Result<LoadedEngine, String> + Send>;

// ---------------------------------------------------------------------------
// Speech synthesis and playback
// ---------------------------------------------------------------------------

/// Timing of one synthesized line.
#[derive(Clone, Copy, Debug, Default)]
pub struct SynthStats {
    pub samples: usize,
    pub first_chunk_ms: f32,
    pub synth_ms: f32,
    /// The sink asked for generation to stop early.
    pub stopped: bool,
}

/// Receives synthesized audio chunks; returns `false` to stop generation.
/// Not `Send`: it is created and called on the TTS thread only.
pub type ChunkSink = Box<dyn FnMut(&[f32]) -> bool>;

/// A neural voice that streams audio as it is generated.
pub trait Synthesizer {
    fn backend_name(&self) -> String;
    /// Native output rate of the voice model.
    fn sample_rate(&self) -> u32;
    /// A stable voice for an actor id.
    fn voice_for(&self, speaker: &str) -> String;
    /// Generate `text`, handing mono chunks at [`Synthesizer::sample_rate`]
    /// to `sink` as soon as they exist. `sink` returns `false` to abort.
    fn synthesize(
        &mut self,
        text: &str,
        voice: &str,
        sink: ChunkSink,
    ) -> Result<SynthStats, String>;
}

/// An output device fed with epoch-tagged mono buffers. Never blocks.
pub trait AudioSink {
    fn device_name(&self) -> String;
    fn sample_rate(&self) -> u32;
    /// Queue `samples`; a full queue hands them back for a later retry.
    fn try_submit(&mut self, epoch: u64, samples: Vec<f32>) -> Result<(), Vec<f32>>;
    /// Samples ever queued.
    fn queued(&self) -> u64;
    /// Samples played or discarded (stale epoch).
    fn consumed(&self) -> u64;
    fn underruns(&self) -> u64;
    /// Whether an utterance is still being generated (for underrun stats).
    fn set_streaming(&mut self, streaming: bool);
    fn take_error(&mut self) -> Option<String>;
}

/// Builds the voice and its output device on the TTS thread. The shared
/// epoch is the playback epoch the device must honour.
pub type TtsFactory = Box<
    dyn FnOnce(Arc<AtomicU64>) -> Result<(Box<dyn Synthesizer>, Box<dyn AudioSink>), String> + Send,
>;

/// Everything the runtime needs from the outside world. `None` disables a
/// component (for example `--no-voice`); the world still runs.
pub struct Backends {
    pub engine: EngineFactory,
    pub mic: Option<(Box<dyn MicBackend>, VadFactory)>,
    pub asr: Option<AsrFactory>,
    pub tts: Option<TtsFactory>,
}
