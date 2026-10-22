//! Messages between the renderer, the authoritative simulation and the
//! workers, plus the immutable [`Snapshot`] the renderer draws.
//!
//! Every message that answers an earlier request carries the correlation ids
//! of that request (conversation id, request id, world epoch, audio epoch), so
//! the simulation can recognise and discard a response that arrives after the
//! conversation it belongs to was cancelled, timed out, superseded or loaded
//! away.

use rpg_domain::action::InvestigateTarget;
use rpg_domain::geom::Vec2;
use rpg_domain::ids::OfferId;
use rpg_domain::ports::{InferenceError, InferenceRequest, InferenceResponse};

/// Identifies one player-initiated conversation turn.
pub type ConvId = u64;

// ---------------------------------------------------------------------------
// Renderer -> simulation
// ---------------------------------------------------------------------------

/// Which microphone to use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MicSelection {
    /// The host's default input device.
    Default,
    /// A device by exact name.
    Named(String),
    /// The next device in the host's list (wraps around).
    Next,
}

/// Commands the renderer sends to the simulation. None of them waits for a
/// model, microphone or speaker.
#[derive(Clone, Debug)]
pub enum Command {
    /// Walk to a world position (validated by the simulation).
    MoveTo(Vec2),
    /// Push-to-talk key went down (`true`) or up (`false`).
    PushToTalk(bool),
    /// Typed dialogue: the same gameplay path as a transcribed utterance.
    Say(String),
    /// Abandon the active conversation turn at whatever stage it is in.
    CancelVoice,
    /// Accept a pending offer from an NPC.
    AcceptOffer(OfferId),
    /// Reject a pending offer from an NPC.
    RejectOffer(OfferId),
    /// Investigate a location, actor, or item.
    Investigate(InvestigateTarget),
    /// Save the current canonical state to a JSON file.
    Save(String),
    /// Load canonical state from a JSON save file.
    Load(String),
    /// Serialize canonical state to the versioned save JSON and hand it to the
    /// platform as [`Effect::SaveData`] (browser storage, export).
    ExportSave {
        slot: String,
    },
    /// Load canonical state from versioned save JSON provided by the platform
    /// (browser storage, an imported file). `label` names the source in
    /// messages; the JSON is validated exactly like a native save file.
    ImportSave {
        label: String,
        json: String,
    },
    /// Show a system line in the dialogue log (platform notices such as
    /// "saved to browser storage" or a connection error).
    Notice(String),
    /// The platform reports voice components that are off or unsupported
    /// (browser voice toggles). Typed dialogue is unaffected.
    VoiceUnavailable {
        mic: bool,
        asr: bool,
        tts: bool,
        why: String,
    },
    /// Record raw microphone PCM to a WAV file, bypassing VAD/ASR/LLM/TTS.
    MicSelfTest {
        path: String,
        seconds: f32,
    },
    /// Switch the input device.
    SelectMic(MicSelection),
    Quit,
}

// ---------------------------------------------------------------------------
// Simulation -> workers
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum MicRequest {
    /// Discard anything captured so far and start recording `conv`.
    Start {
        conv: ConvId,
    },
    /// Finish recording `conv` and report what was captured.
    Stop {
        conv: ConvId,
    },
    /// Abandon recording `conv` without reporting audio.
    Cancel {
        conv: ConvId,
    },
    SelfTest {
        path: String,
        seconds: f32,
    },
    Select(MicSelection),
    Shutdown,
}

#[derive(Clone, Debug)]
pub enum AsrRequest {
    Transcribe {
        conv: ConvId,
        samples: Vec<f32>,
        sample_rate: u32,
    },
    Cancel {
        conv: ConvId,
    },
    Shutdown,
}

/// One NPC decision for the inference worker. The request contains only the
/// NPC's scoped, rendered context — the worker never sees the mutable world.
#[derive(Clone, Debug)]
pub struct InferJob {
    pub conv: ConvId,
    pub request_id: u64,
    /// World epoch the request was rendered in; a load bumps it.
    pub world_epoch: u64,
    pub request: InferenceRequest,
}

#[derive(Clone, Debug)]
pub enum InferRequest {
    Decide(InferJob),
    Cancel { request_id: u64 },
    Shutdown,
}

/// A validated, committed line to voice.
#[derive(Clone, Debug)]
pub struct SpeakJob {
    pub conv: ConvId,
    pub line: u64,
    /// Actor id of the speaker, used to pick a stable voice.
    pub speaker: String,
    pub text: String,
    /// Playback epoch this line belongs to. Once the shared epoch moves on,
    /// the line is aborted and any of its queued audio is discarded.
    pub audio_epoch: u64,
}

#[derive(Clone, Debug)]
pub enum TtsRequest {
    Speak(SpeakJob),
    Shutdown,
}

/// Side effects the simulation core asks the runtime to perform. Keeping
/// them as data makes the core deterministic and directly testable.
#[derive(Clone, Debug)]
pub enum Effect {
    Mic(MicRequest),
    Asr(AsrRequest),
    Infer(InferRequest),
    Tts(TtsRequest),
    /// Publish a new playback epoch: stale audio stops immediately.
    SetAudioEpoch(u64),
    /// Canonical save JSON produced by [`Command::ExportSave`], for the
    /// platform to store. Never contains credentials: it is exactly
    /// `rpg_persistence::SaveFile`.
    SaveData {
        slot: String,
        json: String,
    },
}

// ---------------------------------------------------------------------------
// Workers -> simulation
// ---------------------------------------------------------------------------

/// Format of the open capture device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MicInfo {
    pub name: String,
    /// Rate of the mono chunks the device delivers.
    pub sample_rate: u32,
    /// Channels of the device stream (downmixed before delivery).
    pub channels: u16,
    pub sample_format: String,
}

impl MicInfo {
    pub fn describe(&self) -> String {
        format!(
            "{} — {} Hz, {} ch ({}), delivered as mono",
            self.name, self.sample_rate, self.channels, self.sample_format
        )
    }
}

/// Result of a raw microphone self-test.
#[derive(Clone, Debug)]
pub struct SelfTestReport {
    pub path: String,
    pub device: MicInfo,
    pub requested_seconds: f32,
    pub samples: usize,
    pub wall_seconds: f32,
    pub rms: f32,
    pub peak: f32,
    pub dropped_chunks: u64,
}

impl SelfTestReport {
    /// Samples per wall second actually delivered by the device.
    pub fn effective_rate(&self) -> f32 {
        if self.wall_seconds <= 0.0 {
            0.0
        } else {
            self.samples as f32 / self.wall_seconds
        }
    }

    /// A one-line diagnosis separating capture failures from quiet input.
    pub fn verdict(&self) -> &'static str {
        if self.samples == 0 {
            "FAIL: no audio arrived from the device (capture failure)"
        } else if self.peak < 1e-4 {
            "FAIL: the device delivers pure digital silence (muted, or microphone permission denied)"
        } else if (self.effective_rate() - self.device.sample_rate as f32).abs()
            > 0.2 * self.device.sample_rate as f32
        {
            "WARN: the device delivers audio at an unexpected rate"
        } else if self.peak < 0.02 {
            "WARN: audio arrives but is very quiet — speak closer or raise the input gain"
        } else {
            "OK: the microphone records audio"
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "mic self-test: {} | {} samples in {:.1}s ({:.0} samples/s, device {} Hz) | rms {:.4} peak {:.4} | dropped {} | wrote {}",
            self.verdict(),
            self.samples,
            self.wall_seconds,
            self.effective_rate(),
            self.device.sample_rate,
            self.rms,
            self.peak,
            self.dropped_chunks,
            self.path
        )
    }
}

#[derive(Clone, Debug)]
pub enum MicEvent {
    DeviceReady(MicInfo),
    DeviceError(String),
    Devices(Vec<String>),
    Listening {
        conv: ConvId,
        /// Stale samples discarded from before the key press.
        discarded: usize,
    },
    /// Speech was captured; `samples` are 16 kHz mono, VAD-trimmed.
    Captured {
        conv: ConvId,
        samples: Vec<f32>,
        sample_rate: u32,
        recorded_seconds: f32,
        speech_seconds: f32,
    },
    /// Recording finished but the VAD found no speech in it.
    NoSpeech {
        conv: ConvId,
        /// Wall time push-to-talk was held.
        held_seconds: f32,
        /// Audio the device actually delivered in that time.
        recorded_seconds: f32,
        rms: f32,
        peak: f32,
    },
    SelfTest(Result<SelfTestReport, String>),
}

#[derive(Clone, Debug)]
pub enum AsrEvent {
    Ready {
        backend: String,
    },
    Unavailable(String),
    Transcript {
        conv: ConvId,
        text: String,
        latency_ms: f32,
    },
    Failed {
        conv: ConvId,
        error: String,
    },
}

/// A finished (or failed) decision request.
#[derive(Debug)]
pub struct InferDone {
    pub conv: ConvId,
    pub request_id: u64,
    pub world_epoch: u64,
    pub result: Result<InferenceResponse, InferenceError>,
    /// Time the job waited in the worker queue.
    pub queue_ms: f32,
    /// Time the model spent on it.
    pub infer_ms: f32,
}

#[derive(Debug)]
pub enum InferEvent {
    Ready {
        engine: String,
        model_id: String,
        is_real_model: bool,
    },
    LoadFailed(String),
    /// The backend is streaming an answer for this request. Carries only a
    /// count: partial, unvalidated output is never shown or acted on.
    Progress {
        conv: ConvId,
        request_id: u64,
        chunks: u32,
    },
    Done(InferDone),
}

#[derive(Clone, Debug)]
pub enum TtsEvent {
    Ready {
        backend: String,
        device: String,
        sample_rate: u32,
    },
    Unavailable(String),
    /// The first audio of the line reached the output queue.
    Started {
        conv: ConvId,
        line: u64,
        first_audio_ms: f32,
    },
    /// The line finished playing.
    Finished {
        conv: ConvId,
        line: u64,
        audio_seconds: f32,
        synth_ms: f32,
    },
    /// The line was aborted because its audio epoch was superseded.
    Cancelled {
        conv: ConvId,
        line: u64,
    },
    Failed {
        conv: ConvId,
        line: u64,
        error: String,
    },
    DeviceError(String),
}

/// Everything that can arrive in the simulation's inbox.
#[derive(Debug)]
pub enum Inbox {
    Ui(Command),
    Mic(MicEvent),
    Asr(AsrEvent),
    Infer(InferEvent),
    Tts(TtsEvent),
}

// ---------------------------------------------------------------------------
// Snapshot (plain data; no simulation types leak into the renderer)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KindView {
    Player,
    Npc,
    Background,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatusView {
    Alive,
    Dead,
    Departed,
}

#[derive(Clone, Debug)]
pub struct ActorView {
    pub id: String,
    pub name: String,
    pub pos: Vec2,
    pub kind: KindView,
    pub status: StatusView,
    pub location: String,
    pub nearby: bool,
}

#[derive(Clone, Debug)]
pub struct LocationView {
    pub id: String,
    pub name: String,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub private: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineKind {
    Player,
    Npc,
    System,
}

#[derive(Clone, Debug)]
pub struct DialogueLine {
    pub speaker: String,
    pub text: String,
    pub kind: LineKind,
}

/// One stack of items in the player's inventory, for display.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ItemView {
    pub name: String,
    pub qty: u32,
}

/// A pending or resolved offer between two actors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OfferView {
    pub id: u64,
    pub from: String,
    pub to: String,
    /// True when the player is the recipient, so the UI only lets the player
    /// accept/reject offers that were actually addressed to them.
    pub to_player: bool,
    pub give: String,
    pub want: String,
    pub price: i64,
    pub note: String,
    pub state: String,
}

/// A commitment (favour, debt, promise) tracked by the social layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitmentView {
    pub id: u64,
    pub kind: String,
    pub debtor: String,
    pub creditor: String,
    pub terms: String,
    pub state: String,
}

/// The explicit voice-interaction state machine:
/// `Idle → Listening → Transcribing → Deciding → Validating → Synthesizing →
/// Speaking → Idle`, with cancellation, errors and timeouts returning to
/// `Idle` from every stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VoiceStage {
    Idle,
    Listening,
    Transcribing,
    Deciding,
    Validating,
    Synthesizing,
    Speaking,
}

impl VoiceStage {
    pub fn label(self) -> &'static str {
        match self {
            VoiceStage::Idle => "idle",
            VoiceStage::Listening => "listening",
            VoiceStage::Transcribing => "transcribing",
            VoiceStage::Deciding => "thinking",
            VoiceStage::Validating => "validating",
            VoiceStage::Synthesizing => "preparing voice",
            VoiceStage::Speaking => "speaking",
        }
    }
}

/// Readiness of one voice component.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Readiness {
    Loading,
    Ready(String),
    Unavailable(String),
}

impl Readiness {
    pub fn is_ready(&self) -> bool {
        matches!(self, Readiness::Ready(_))
    }

    pub fn describe(&self) -> String {
        match self {
            Readiness::Loading => "loading".into(),
            Readiness::Ready(s) => s.clone(),
            Readiness::Unavailable(e) => format!("UNAVAILABLE: {e}"),
        }
    }
}

/// Player-facing voice state.
#[derive(Clone, Debug)]
pub struct VoiceView {
    pub stage: VoiceStage,
    pub conversation: Option<ConvId>,
    /// Actor id of the NPC being addressed, once known.
    pub npc: Option<String>,
    pub stage_seconds: f32,
    pub mic: Readiness,
    pub asr: Readiness,
    pub tts: Readiness,
    /// Live input level, updated continuously (also while idle).
    pub mic_rms: f32,
    pub mic_peak: f32,
    /// The VAD currently hears speech.
    pub speech_detected: bool,
    pub last_transcript: Option<String>,
}

/// Developer diagnostics (F3). Enough to tell at a glance whether a problem
/// is in capture, VAD, ASR, inference, the simulation or playback.
#[derive(Clone, Debug, Default)]
pub struct Diagnostics {
    /// `windows`, `linux`, `macos` or `browser`; filled in by the driver.
    pub platform: String,
    // Simulation
    pub ticks_per_second: f32,
    pub target_ticks_per_second: f32,
    pub tick_delay_ms: f32,
    pub max_tick_lag_ms: f32,
    pub skipped_ticks: u64,
    // Microphone
    pub mic_device: String,
    pub mic_samples_per_second: f32,
    pub mic_rms: f32,
    pub mic_peak: f32,
    pub mic_dropped_chunks: u64,
    pub vad_speaking: bool,
    pub recording_seconds: f32,
    // ASR
    pub last_transcript: Option<String>,
    pub asr_latency_ms: Option<f32>,
    // Inference
    pub conversation: Option<ConvId>,
    pub request_id: Option<u64>,
    pub active_inference: Option<u64>,
    pub llm_queue_ms: Option<f32>,
    pub llm_infer_ms: Option<f32>,
    /// Time to the first generated token/streamed byte of the last answer.
    pub llm_first_token_ms: Option<f32>,
    /// Streamed chunks received so far for the running request.
    pub llm_stream_chunks: Option<u32>,
    /// Decision requests waiting for an answer (0 or 1: one conversation at
    /// a time).
    pub pending_inference: usize,
    /// Session-wide inference accounting.
    pub inference: crate::core::InferenceStats,
    /// Provenance label of the most recent committed NPC decision.
    pub last_decision: Option<String>,
    /// NPC proposals the simulation rejected (before any regeneration).
    pub rejected_actions: u64,
    // TTS / playback
    pub tts_first_audio_ms: Option<f32>,
    pub tts_synth_ms: Option<f32>,
    pub playback: String,
    pub playback_underruns: u64,
    // Queues
    pub asr_queue: usize,
    pub infer_queue: usize,
    pub tts_queue: usize,
    // Counters
    pub cancelled: u64,
    pub stale_responses: u64,
    pub timeouts: u64,
    pub refused_ptt: u64,
    pub regenerations: u64,
    pub invalidated: u64,
    pub transitions: Vec<String>,
}

/// Everything the renderer needs for one frame.
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub tick: u64,
    pub day: u32,
    pub minute_of_day: u32,
    /// Bumped by every load, so the renderer can tell timelines apart.
    pub world_epoch: u64,
    pub locations: Vec<LocationView>,
    pub actors: Vec<ActorView>,
    pub player_pos: Vec2,
    pub player_location: String,
    pub nearby: Option<String>,
    pub dialogue: Vec<DialogueLine>,
    pub engine_name: String,
    pub model_id: String,
    pub model_decisions: usize,
    pub canonical_decisions: usize,
    pub fallback_decisions: usize,
    pub status: String,
    pub voice: VoiceView,
    pub diagnostics: Diagnostics,
    /// Developer-diagnostics data. Rendered only in the F3 overlay, never in
    /// the normal player UI (an NPC's beliefs are not player knowledge).
    pub recent_events: Vec<String>,
    pub nearby_beliefs: Vec<String>,
    pub nearby_relationship: Option<String>,
    pub perception: Vec<String>,
    pub offers: Vec<OfferView>,
    pub commitments: Vec<CommitmentView>,
    pub inventory: Vec<ItemView>,
    pub money: i64,
}

impl Snapshot {
    /// "HH:MM" from the simulation clock.
    pub fn clock(&self) -> String {
        format!(
            "{:02}:{:02}",
            self.minute_of_day / 60,
            self.minute_of_day % 60
        )
    }

    /// Position of an actor by id, if present.
    pub fn actor_pos(&self, id: &str) -> Option<Vec2> {
        self.actors.iter().find(|a| a.id == id).map(|a| a.pos)
    }
}
