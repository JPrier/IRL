//! Abstract ports (traits) for the replaceable external capabilities:
//! local LLM inference, speech recognition and speech synthesis.
//!
//! Keeping these traits in the domain crate means the simulation never
//! depends on a specific model runtime. Concrete backends live in `agents` and
//! `speech`; deterministic test doubles implement the same traits.

use crate::ids::ActorId;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

/// Logical role a completion is serving. Used for scheduling priority, prompt
/// selection and metrics — not for a separate model unless configured.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InferenceRole {
    /// P0: classify player intent / address.
    Intent,
    /// P0: a named NPC deciding how to respond.
    NpcDecision,
    /// P1: concise spoken line for an already-validated decision.
    Dialogue,
    /// P2: director narrative proposal.
    Director,
    /// P2: memory consolidation / summarisation.
    Memory,
}

impl InferenceRole {
    /// P0 player-facing, P1 important local, P2 background.
    pub fn priority(self) -> u8 {
        match self {
            InferenceRole::Intent | InferenceRole::NpcDecision => 0,
            InferenceRole::Dialogue => 1,
            InferenceRole::Director | InferenceRole::Memory => 2,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// Correlation id shared with the audio/trace pipeline.
    pub id: u64,
    pub role: InferenceRole,
    pub system: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub seed: u64,
    /// Optional JSON-schema-ish hint describing the required structured output.
    #[serde(default)]
    pub response_contract: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub text: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub first_token_ms: f32,
    pub prefill_ms: f32,
    pub decode_ms: f32,
    /// Whether the backend applied constrained/structured decoding.
    pub structured: bool,
}

impl InferenceResponse {
    pub fn total_ms(&self) -> f32 {
        self.prefill_ms + self.decode_ms
    }

    /// Rough words per second of generated text.
    pub fn words_per_sec(&self) -> f32 {
        let words = self.text.split_whitespace().count() as f32;
        let secs = self.decode_ms / 1000.0;
        if secs <= 0.0 {
            0.0
        } else {
            words / secs
        }
    }
}

#[derive(Debug, Error)]
pub enum InferenceError {
    #[error("model not loaded: {0}")]
    NotLoaded(String),
    #[error("model file missing: {0}")]
    MissingAsset(String),
    #[error("inference timed out after {0} ms")]
    Timeout(u64),
    #[error("backend error: {0}")]
    Backend(String),
    #[error("structured output failed schema validation: {0}")]
    Schema(String),
}

/// A resident, warmed inference backend. Implementations must **not** spawn a
/// fresh process per request.
pub trait InferenceEngine: Send {
    fn engine_name(&self) -> String;
    fn model_id(&self) -> String;
    /// Load weights and run a tiny warm-up pass. Idempotent.
    fn warm(&mut self) -> Result<(), InferenceError>;
    fn complete(&mut self, req: &InferenceRequest) -> Result<InferenceResponse, InferenceError>;
    /// Incremental completion. Backends that support token streaming override
    /// this; the default falls back to the buffered result.
    fn complete_streaming(
        &mut self,
        req: &InferenceRequest,
        sink: &mut dyn FnMut(&str),
    ) -> Result<InferenceResponse, InferenceError> {
        let response = self.complete(req)?;
        sink(&response.text);
        Ok(response)
    }
    /// Cancel an in-flight request if the backend supports it.
    fn cancel(&mut self, _id: u64) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Speech synthesis (TTS)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoiceInfo {
    pub id: String,
    pub display_name: String,
    pub language: String,
    pub license: String,
    /// Optional provenance/attribution string.
    #[serde(default)]
    pub attribution: String,
}

#[derive(Clone, Debug)]
pub struct SpeechRequest {
    pub text: String,
    pub voice_id: String,
    pub sample_rate: u32,
}

#[derive(Clone, Debug, Default)]
pub struct SpeechAudio {
    /// Mono f32 PCM in `sample_rate`.
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// Wall-clock milliseconds the synthesizer spent producing this audio.
    pub synth_ms: f32,
    /// Time to first audio buffer in milliseconds.
    pub first_audio_ms: f32,
}

impl SpeechAudio {
    pub fn audio_seconds(&self) -> f32 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.samples.len() as f32 / self.sample_rate as f32
        }
    }

    /// audio-seconds produced per wall-second spent synthesizing.
    pub fn real_time_factor(&self) -> f32 {
        let wall = self.synth_ms / 1000.0;
        if wall <= 0.0 {
            f32::INFINITY
        } else {
            self.audio_seconds() / wall
        }
    }

    pub fn is_silent(&self) -> bool {
        self.samples.iter().all(|s| s.abs() < 1e-6)
    }
}

#[derive(Debug, Error)]
pub enum SpeechError {
    #[error("voice not found: {0}")]
    UnknownVoice(String),
    #[error("speech asset missing: {0}")]
    MissingAsset(String),
    #[error("speech backend error: {0}")]
    Backend(String),
    #[error("audio device unavailable: {0}")]
    Device(String),
}

pub trait SpeechSynthesizer: Send {
    fn backend_name(&self) -> String;
    fn voices(&self) -> Vec<VoiceInfo>;
    fn warm(&mut self, voice_id: &str) -> Result<(), SpeechError>;
    fn synthesize(&mut self, req: &SpeechRequest) -> Result<SpeechAudio, SpeechError>;
    fn cancel(&mut self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Speech recognition (ASR)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Transcript {
    pub text: String,
    pub confidence: f32,
    pub is_final: bool,
    /// Wall-clock ms from end-of-speech to stable transcript.
    pub latency_ms: f32,
    /// Whether a non-neural fallback produced this transcript.
    pub fallback: bool,
}

#[derive(Debug, Error)]
pub enum RecognitionError {
    #[error("asr asset missing: {0}")]
    MissingAsset(String),
    #[error("asr backend error: {0}")]
    Backend(String),
    #[error("no speech detected")]
    NoSpeech,
}

pub trait SpeechRecognizer: Send {
    fn backend_name(&self) -> String;
    fn warm(&mut self) -> Result<(), RecognitionError>;
    /// Transcribe a complete utterance of mono f32 PCM.
    fn transcribe(&mut self, pcm: &[f32], sample_rate: u32)
        -> Result<Transcript, RecognitionError>;
}

// ---------------------------------------------------------------------------
// Memory retrieval
// ---------------------------------------------------------------------------

/// A single retrieved memory line with provenance, scoped to one actor.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetrievedMemory {
    pub text: String,
    /// 0..1 relevance/salience score used for ordering.
    pub score: f32,
    pub source_event: Option<u64>,
    pub about: Option<ActorId>,
}

pub trait MemoryRetriever: Send + Sync {
    /// Retrieve the most relevant memories for `actor` given a query.
    fn retrieve(&self, actor: &ActorId, query: &str, limit: usize) -> Vec<RetrievedMemory>;
}
