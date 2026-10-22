//! Neural speech backends backed by `sherpa-onnx`.
//!
//! This module is compiled only with the `neural` feature. It provides:
//! * [`SherpaTts`] — offline multi-speaker neural TTS (Kokoro) implementing
//!   [`SpeechSynthesizer`];
//! * [`SherpaAsr`] — offline Whisper ASR implementing [`SpeechRecognizer`];
//! * [`VadSegmenter`] — Silero VAD that turns a continuous microphone stream
//!   into finalized utterances.
//!
//! Every constructor takes a [`SpeechAssets`] resolver, so the file layout and
//! checksums come from the pinned manifest rather than being hard-coded here.

use crate::assets::{AssetKind, SpeechAssets};
use rpg_domain::ports::SpeechRecognizer;
use rpg_domain::ports::{
    RecognitionError, SpeechAudio, SpeechError, SpeechRequest, SpeechSynthesizer, Transcript,
    VoiceInfo,
};
use sherpa_onnx::{
    GenerationConfig, OfflineRecognizer, OfflineRecognizerConfig, OfflineTts, OfflineTtsConfig,
    OfflineTtsKokoroModelConfig, OfflineTtsModelConfig, OfflineTtsVitsModelConfig,
    OfflineWhisperModelConfig, SileroVadModelConfig, VadModelConfig, VoiceActivityDetector,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// A cloneable cancellation handle shared with a synthesis worker.
///
/// The handle is intentionally cheap to clone and independent of the engine
/// borrow, so a UI or audio thread can request interruption while another
/// thread is mid-synthesis.
#[derive(Clone, Default)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
}

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }
    /// Request cancellation of the current operation.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }
    /// Clear the flag so the token can be reused.
    pub fn reset(&self) {
        self.flag.store(false, Ordering::SeqCst);
    }
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }
}

// ---------------------------------------------------------------------------
// Neural TTS
// ---------------------------------------------------------------------------

/// Kokoro-backed multi-speaker neural TTS.
pub struct SherpaTts {
    tts: OfflineTts,
    sample_rate: u32,
    voices: Vec<VoiceInfo>,
    sids: HashMap<String, i32>,
    cancel: CancelToken,
}

impl SherpaTts {
    /// Build a resident Kokoro engine from installed assets.
    pub fn from_assets(
        assets: &SpeechAssets,
        bundle_id: &str,
        threads: i32,
    ) -> Result<Self, SpeechError> {
        let resolved = assets
            .require_bundle(bundle_id)
            .map_err(|e| SpeechError::MissingAsset(e.to_string()))?;
        let model = resolved
            .files
            .model
            .clone()
            .ok_or_else(|| SpeechError::Backend("tts bundle has no model file".into()))?;
        let tokens = resolved
            .files
            .tokens
            .clone()
            .ok_or_else(|| SpeechError::Backend("tts bundle has no tokens file".into()))?;
        let voices_path = resolved.files.voices.clone();
        let data_dir = resolved.files.data_dir.clone();

        let mut model_config = OfflineTtsModelConfig {
            kokoro: OfflineTtsKokoroModelConfig {
                model: Some(model),
                voices: voices_path,
                tokens: Some(tokens),
                data_dir,
                ..Default::default()
            },
            num_threads: threads.max(1),
            provider: Some("cpu".into()),
            ..Default::default()
        };
        model_config.vits = OfflineTtsVitsModelConfig::default();

        let config = OfflineTtsConfig {
            model: model_config,
            ..Default::default()
        };
        let tts = OfflineTts::create(&config).ok_or_else(|| {
            SpeechError::Backend(format!("failed to create TTS engine for '{bundle_id}'"))
        })?;

        let reported = tts.num_speakers();
        let mut voices = Vec::new();
        let mut sids = HashMap::new();
        for v in &resolved.bundle.voices {
            // Guard against a manifest that points past the model's speaker set.
            if reported > 0 && v.sid >= reported {
                continue;
            }
            voices.push(VoiceInfo {
                id: v.id.clone(),
                display_name: v.display_name.clone(),
                language: v.language.clone(),
                license: resolved.bundle.license.clone(),
                attribution: resolved.bundle.attribution.clone(),
            });
            sids.insert(v.id.clone(), v.sid);
        }
        if voices.is_empty() {
            return Err(SpeechError::Backend(
                "tts bundle exposes no usable voices for this model".into(),
            ));
        }
        Ok(Self {
            sample_rate: tts.sample_rate().max(1) as u32,
            tts,
            voices,
            sids,
            cancel: CancelToken::new(),
        })
    }

    /// A cancellation handle for interrupting an in-flight synthesis.
    pub fn cancel_token(&self) -> CancelToken {
        self.cancel.clone()
    }

    /// Number of speakers reported by the loaded model.
    pub fn model_speakers(&self) -> i32 {
        self.tts.num_speakers()
    }

    /// Output sample rate of the loaded voice model, in Hz.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn sid_for(&self, voice_id: &str) -> Result<i32, SpeechError> {
        self.sids
            .get(voice_id)
            .copied()
            .ok_or_else(|| SpeechError::UnknownVoice(voice_id.to_string()))
    }

    /// Deterministically pick a voice for a named actor, so the same NPC keeps
    /// the same voice across sessions without any explicit assignment.
    pub fn voice_for_actor(&self, actor: &str) -> VoiceInfo {
        let mut hash: u64 = 1469598103934665603;
        for b in actor.as_bytes() {
            hash ^= *b as u64;
            hash = hash.wrapping_mul(1099511628211);
        }
        let idx = (hash as usize) % self.voices.len();
        self.voices[idx].clone()
    }
}

impl SpeechSynthesizer for SherpaTts {
    fn backend_name(&self) -> String {
        "sherpa-onnx/kokoro (neural)".into()
    }

    fn voices(&self) -> Vec<VoiceInfo> {
        self.voices.clone()
    }

    fn warm(&mut self, voice_id: &str) -> Result<(), SpeechError> {
        self.sid_for(voice_id).map(|_| ())
    }

    fn synthesize(&mut self, req: &SpeechRequest) -> Result<SpeechAudio, SpeechError> {
        let sid = self.sid_for(&req.voice_id)?;
        // A cancel aimed at a previous utterance must not poison this one.
        self.cancel.reset();
        let gen = GenerationConfig {
            sid,
            speed: 1.0,
            silence_scale: 0.2,
            ..Default::default()
        };
        let start = Instant::now();
        let generated = self
            .tts
            .generate_with_config(&req.text, &gen, None::<fn(&[f32], f32) -> bool>)
            .ok_or_else(|| SpeechError::Backend("tts generation returned no audio".into()))?;
        let synth_ms = start.elapsed().as_secs_f32() * 1000.0;
        let audio = SpeechAudio {
            samples: generated.samples().to_vec(),
            sample_rate: generated.sample_rate().max(1) as u32,
            synth_ms,
            // Whole-utterance offline generation: first audio arrives at the end.
            first_audio_ms: synth_ms,
        };
        if audio.is_silent() {
            return Err(SpeechError::Backend(
                "tts produced silent audio (check voice/model assets)".into(),
            ));
        }
        Ok(audio)
    }

    fn cancel(&mut self) -> bool {
        self.cancel.cancel();
        true
    }
}

/// Receives streamed TTS audio; returns `false` to stop generation.
pub type ChunkSink = Box<dyn FnMut(&[f32]) -> bool>;

/// Timing of one streamed synthesis.
#[derive(Clone, Copy, Debug, Default)]
pub struct StreamStats {
    /// Mono samples delivered to the sink, at [`SherpaTts::sample_rate`].
    pub samples: usize,
    /// Wall milliseconds until the first chunk reached the sink.
    pub first_chunk_ms: f32,
    /// Wall milliseconds for the whole generation.
    pub synth_ms: f32,
    /// Number of chunks delivered.
    pub chunks: usize,
    /// True when the sink asked generation to stop early.
    pub stopped: bool,
}

impl SherpaTts {
    /// Synthesize `text`, handing audio to `sink` as the model produces it
    /// (Kokoro emits one chunk per sentence), instead of waiting for the
    /// whole utterance. `sink` returns `false` to abort generation, which is
    /// how a cancelled conversation stops spending CPU on stale speech.
    ///
    /// The sink receives mono `f32` at [`SherpaTts::sample_rate`]. If the
    /// backend produces no progress callbacks, the complete audio is delivered
    /// once at the end, so callers never lose a line.
    pub fn synthesize_streaming(
        &mut self,
        text: &str,
        voice_id: &str,
        sink: ChunkSink,
    ) -> Result<StreamStats, SpeechError> {
        use std::cell::RefCell;
        use std::rc::Rc;
        let sid = self.sid_for(voice_id)?;
        self.cancel.reset();
        let gen = GenerationConfig {
            sid,
            speed: 1.0,
            silence_scale: 0.2,
            ..Default::default()
        };
        let start = Instant::now();
        // The callback runs synchronously on this thread, so the sink never
        // has to be `Send` (it may own thread-bound audio handles).
        let shared = Rc::new(RefCell::new((sink, StreamStats::default())));
        let cb_shared = shared.clone();
        let cancel = self.cancel.clone();
        let generated = self.tts.generate_with_config(
            text,
            &gen,
            Some(move |samples: &[f32], _progress: f32| -> bool {
                let mut guard = cb_shared.borrow_mut();
                let (sink, stats) = &mut *guard;
                if stats.chunks == 0 {
                    stats.first_chunk_ms = start.elapsed().as_secs_f32() * 1000.0;
                }
                stats.chunks += 1;
                stats.samples += samples.len();
                let keep_going = !cancel.is_cancelled() && sink(samples);
                stats.stopped |= !keep_going;
                keep_going
            }),
        );
        let mut guard = shared.borrow_mut();
        let (sink, stats) = &mut *guard;
        if stats.stopped {
            stats.synth_ms = start.elapsed().as_secs_f32() * 1000.0;
            return Ok(*stats);
        }
        let generated = generated
            .ok_or_else(|| SpeechError::Backend("tts generation returned no audio".into()))?;
        if stats.chunks == 0 {
            let all = generated.samples();
            if all.iter().all(|s| s.abs() < 1e-6) {
                return Err(SpeechError::Backend(
                    "tts produced silent audio (check voice/model assets)".into(),
                ));
            }
            stats.first_chunk_ms = start.elapsed().as_secs_f32() * 1000.0;
            stats.chunks = 1;
            stats.samples = all.len();
            stats.stopped = !sink(all);
        }
        stats.synth_ms = start.elapsed().as_secs_f32() * 1000.0;
        Ok(*stats)
    }
}

// ---------------------------------------------------------------------------
// Neural ASR
// ---------------------------------------------------------------------------

/// Whisper-backed offline speech recognizer.
pub struct SherpaAsr {
    recognizer: OfflineRecognizer,
    backend: String,
}

impl SherpaAsr {
    pub fn from_assets(
        assets: &SpeechAssets,
        bundle_id: &str,
        threads: i32,
    ) -> Result<Self, RecognitionError> {
        let resolved = assets
            .require_bundle(bundle_id)
            .map_err(|e| RecognitionError::MissingAsset(e.to_string()))?;
        let encoder = resolved
            .files
            .encoder
            .clone()
            .ok_or_else(|| RecognitionError::Backend("asr bundle has no encoder".into()))?;
        let decoder = resolved
            .files
            .decoder
            .clone()
            .ok_or_else(|| RecognitionError::Backend("asr bundle has no decoder".into()))?;
        let tokens = resolved
            .files
            .tokens
            .clone()
            .ok_or_else(|| RecognitionError::Backend("asr bundle has no tokens".into()))?;

        let mut config = OfflineRecognizerConfig::default();
        config.model_config.whisper = OfflineWhisperModelConfig {
            encoder: Some(encoder),
            decoder: Some(decoder),
            language: Some("en".into()),
            task: Some("transcribe".into()),
            tail_paddings: -1,
            enable_token_timestamps: false,
            enable_segment_timestamps: false,
        };
        config.model_config.tokens = Some(tokens);
        config.model_config.model_type = Some("whisper".into());
        config.model_config.num_threads = threads.max(1);
        config.model_config.provider = Some("cpu".into());

        let recognizer = OfflineRecognizer::create(&config).ok_or_else(|| {
            RecognitionError::Backend(format!("failed to create ASR engine for '{bundle_id}'"))
        })?;
        Ok(Self {
            recognizer,
            backend: format!("sherpa-onnx/whisper ({bundle_id})"),
        })
    }

    /// Create an ASR engine from an explicit set of file paths.
    ///
    /// Useful for tests and for pointing at an already-extracted model without
    /// going through the manifest.
    pub fn from_paths(
        encoder: impl Into<String>,
        decoder: impl Into<String>,
        tokens: impl Into<String>,
        threads: i32,
    ) -> Result<Self, RecognitionError> {
        let mut config = OfflineRecognizerConfig::default();
        config.model_config.whisper = OfflineWhisperModelConfig {
            encoder: Some(encoder.into()),
            decoder: Some(decoder.into()),
            language: Some("en".into()),
            task: Some("transcribe".into()),
            tail_paddings: -1,
            ..Default::default()
        };
        config.model_config.tokens = Some(tokens.into());
        config.model_config.model_type = Some("whisper".into());
        config.model_config.num_threads = threads.max(1);
        let recognizer = OfflineRecognizer::create(&config)
            .ok_or_else(|| RecognitionError::Backend("failed to create ASR engine".into()))?;
        Ok(Self {
            recognizer,
            backend: "sherpa-onnx/whisper".into(),
        })
    }
}

impl SpeechRecognizer for SherpaAsr {
    fn backend_name(&self) -> String {
        self.backend.clone()
    }

    fn warm(&mut self) -> Result<(), RecognitionError> {
        // The recognizer is created resident; a zero-length probe is cheap and
        // verifies the graph runs end to end.
        let stream = self.recognizer.create_stream();
        stream.accept_waveform(16_000, &[0.0f32; 1600]);
        self.recognizer.decode(&stream);
        Ok(())
    }

    fn transcribe(
        &mut self,
        pcm: &[f32],
        sample_rate: u32,
    ) -> Result<Transcript, RecognitionError> {
        if pcm.is_empty() {
            return Err(RecognitionError::NoSpeech);
        }
        let start = Instant::now();
        let stream = self.recognizer.create_stream();
        stream.accept_waveform(sample_rate as i32, pcm);
        self.recognizer.decode(&stream);
        let result = stream
            .get_result()
            .ok_or_else(|| RecognitionError::Backend("recognizer returned no result".into()))?;
        let latency_ms = start.elapsed().as_secs_f32() * 1000.0;
        let text = result.text.trim().to_string();
        if text.is_empty() {
            return Err(RecognitionError::NoSpeech);
        }
        Ok(Transcript {
            text,
            // The offline sherpa API does not expose per-token probabilities;
            // report a conservative 1.0 only when a transcript was produced and
            // make the absence of a real score explicit in the docs.
            confidence: 1.0,
            is_final: true,
            latency_ms,
            fallback: false,
        })
    }
}

// ---------------------------------------------------------------------------
// Voice activity detection
// ---------------------------------------------------------------------------

/// One finalized speech segment emitted by the VAD.
#[derive(Clone, Debug)]
pub struct SpeechSegmentAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl SpeechSegmentAudio {
    pub fn audio_seconds(&self) -> f32 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.samples.len() as f32 / self.sample_rate as f32
        }
    }
}

/// Silero VAD wrapper that yields finalized utterances from a sample stream.
pub struct VadSegmenter {
    vad: VoiceActivityDetector,
    sample_rate: u32,
    /// Number of segments produced so far (for diagnostics).
    pub segments_emitted: u64,
}

impl VadSegmenter {
    pub fn from_assets(
        assets: &SpeechAssets,
        bundle_id: &str,
        sample_rate: u32,
    ) -> Result<Self, SpeechError> {
        let resolved = assets
            .require_bundle(bundle_id)
            .map_err(|e| SpeechError::MissingAsset(e.to_string()))?;
        let model = resolved
            .files
            .vad_model
            .clone()
            .ok_or_else(|| SpeechError::Backend("vad bundle has no model".into()))?;
        Self::from_model_path(&model, sample_rate)
    }

    pub fn from_model_path(model: &str, sample_rate: u32) -> Result<Self, SpeechError> {
        let silero = SileroVadModelConfig {
            model: Some(model.to_string()),
            threshold: 0.5,
            min_silence_duration: 0.25,
            min_speech_duration: 0.25,
            window_size: 512,
            max_speech_duration: 20.0,
        };
        let config = VadModelConfig {
            silero_vad: silero,
            ten_vad: Default::default(),
            sample_rate: sample_rate as i32,
            num_threads: 1,
            provider: Some("cpu".into()),
            debug: false,
        };
        let vad = VoiceActivityDetector::create(&config, 30.0)
            .ok_or_else(|| SpeechError::Backend("failed to create Silero VAD".into()))?;
        Ok(Self {
            vad,
            sample_rate,
            segments_emitted: 0,
        })
    }

    /// Feed a chunk of mono PCM. Returns any segments that just finalized.
    pub fn accept(&mut self, samples: &[f32]) -> Vec<SpeechSegmentAudio> {
        self.vad.accept_waveform(samples);
        self.drain(false)
    }

    /// Flush trailing speech (call when the microphone stops).
    pub fn flush(&mut self) -> Vec<SpeechSegmentAudio> {
        self.vad.flush();
        self.drain(true)
    }

    /// True while the detector believes speech is present.
    pub fn speaking(&self) -> bool {
        self.vad.detected()
    }

    /// Forget all detector state and any finalized-but-unread segments, so
    /// the next utterance cannot inherit audio from the previous one.
    pub fn reset(&mut self) {
        self.vad.reset();
        while self.vad.front().is_some() {
            self.vad.pop();
        }
    }

    fn drain(&mut self, _flushing: bool) -> Vec<SpeechSegmentAudio> {
        let mut out = Vec::new();
        while let Some(segment) = self.vad.front() {
            let samples = segment.samples().to_vec();
            self.vad.pop();
            if !samples.is_empty() {
                out.push(SpeechSegmentAudio {
                    samples,
                    sample_rate: self.sample_rate,
                });
                self.segments_emitted += 1;
            }
        }
        out
    }
}

/// The default TTS bundle id used when none is configured.
pub const DEFAULT_TTS_BUNDLE: &str = "kokoro-en-v0_19";
/// The default ASR bundle id used when none is configured.
pub const DEFAULT_ASR_BUNDLE: &str = "whisper-tiny-en";
/// The default VAD bundle id used when none is configured.
pub const DEFAULT_VAD_BUNDLE: &str = "silero-vad";

/// Convenience: which bundle id to use for a kind, preferring an explicit id.
pub fn default_bundle_for(kind: AssetKind) -> &'static str {
    match kind {
        AssetKind::Tts => DEFAULT_TTS_BUNDLE,
        AssetKind::Asr => DEFAULT_ASR_BUNDLE,
        AssetKind::Vad => DEFAULT_VAD_BUNDLE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancel_token_is_shared_across_clones() {
        let a = CancelToken::new();
        let b = a.clone();
        assert!(!b.is_cancelled());
        a.cancel();
        assert!(b.is_cancelled());
        b.reset();
        assert!(!a.is_cancelled());
    }

    #[test]
    fn default_bundle_ids_match_manifest() {
        let m = crate::assets::SpeechManifest::parse(include_str!(
            "../../../assets/speech/manifest.json"
        ))
        .unwrap();
        assert!(m.bundle(DEFAULT_TTS_BUNDLE).is_some());
        assert!(m.bundle(DEFAULT_ASR_BUNDLE).is_some());
        assert!(m.bundle(DEFAULT_VAD_BUNDLE).is_some());
    }
}
