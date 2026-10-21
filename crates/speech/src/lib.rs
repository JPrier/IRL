//! # rpg_speech
//!
//! Speech I/O and measurement scaffolding.
//!
//! **Important honesty note:** a real local *neural* TTS/ASR backend requires
//! model assets that are not vendored in this repository. This crate provides:
//!
//! * the [`SpeechSynthesizer`]/[`SpeechRecognizer`] ports are defined in
//!   `rpg_domain::ports` and are the integration points for a real backend;
//! * a real WAV encoder/decoder and streaming buffer accounting;
//! * [`DiagnosticToneSynth`], explicitly a **non-neural diagnostic device** used
//!   to exercise the pipeline and measure buffering. It must never be presented
//!   as evidence that neural TTS works;
//! * [`MissingNeuralBackend`], which fails loudly with install instructions
//!   instead of silently faking audio.
//!
//! The end-to-end acceptance test for neural voice is `cargo run -p voice-demo`
//! once a compliant backend + assets are installed.
//!
//! ## Neural backends (feature `neural`, enabled by default)
//!
//! * [`assets`] resolves the pinned, checksummed speech-asset manifest.
//! * [`neural::SherpaTts`] is real offline multi-speaker neural TTS.
//! * [`neural::SherpaAsr`] is real offline neural speech recognition.
//! * [`neural::VadSegmenter`] segments a live microphone stream into utterances.
//! * [`audio`] provides real microphone capture and speaker playback via `cpal`.

pub mod assets;

#[cfg(feature = "neural")]
pub mod audio;
#[cfg(feature = "neural")]
pub mod neural;

#[cfg(feature = "neural")]
pub use audio::{
    AudioCapture, AudioDeviceInfo, AudioPlayback, CaptureConfig, PlaybackConfig,
};
#[cfg(feature = "neural")]
pub use neural::{
    default_bundle_for, CancelToken, SherpaAsr, SherpaTts, SpeechSegmentAudio, VadSegmenter,
    DEFAULT_ASR_BUNDLE, DEFAULT_TTS_BUNDLE, DEFAULT_VAD_BUNDLE,
};

use rpg_domain::ports::{SpeechAudio, SpeechError, SpeechRequest, SpeechSynthesizer, VoiceInfo};
use std::io::{Read, Write};
use std::path::Path;

// ---------------------------------------------------------------------------
// WAV I/O (16-bit PCM mono)
// ---------------------------------------------------------------------------

pub fn encode_wav(audio: &SpeechAudio) -> Vec<u8> {
    let channels: u16 = 1;
    let bits: u16 = 16;
    let rate = audio.sample_rate.max(1);
    let data_len = (audio.samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * channels as u32 * bits as u32 / 8).to_le_bytes());
    out.extend_from_slice(&(channels * bits / 8).to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for s in &audio.samples {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

pub fn write_wav(path: impl AsRef<Path>, audio: &SpeechAudio) -> std::io::Result<()> {
    let bytes = encode_wav(audio);
    let mut f = std::fs::File::create(path)?;
    f.write_all(&bytes)?;
    Ok(())
}

pub fn decode_wav(bytes: &[u8]) -> Result<SpeechAudio, SpeechError> {
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(SpeechError::Backend("not a WAV file".into()));
    }
    let mut sample_rate = 44_100u32;
    let mut data: &[u8] = &[];
    let mut i = 12;
    while i + 8 <= bytes.len() {
        let id = &bytes[i..i + 4];
        let size =
            u32::from_le_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]]) as usize;
        let body_start = i + 8;
        if id == b"fmt " && body_start + 8 <= bytes.len() {
            sample_rate = u32::from_le_bytes([
                bytes[body_start + 4],
                bytes[body_start + 5],
                bytes[body_start + 6],
                bytes[body_start + 7],
            ]);
        } else if id == b"data" {
            let end = (body_start + size).min(bytes.len());
            data = &bytes[body_start..end];
            break;
        }
        i = body_start + size + (size & 1);
    }
    let samples = data
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| i16::from_le_bytes(*c) as f32 / i16::MAX as f32)
        .collect();
    Ok(SpeechAudio {
        samples,
        sample_rate,
        synth_ms: 0.0,
        first_audio_ms: 0.0,
    })
}

pub fn read_wav(path: impl AsRef<Path>) -> Result<SpeechAudio, SpeechError> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .map_err(|e| SpeechError::Backend(e.to_string()))?
        .read_to_end(&mut bytes)
        .map_err(|e| SpeechError::Backend(e.to_string()))?;
    decode_wav(&bytes)
}

// ---------------------------------------------------------------------------
// Backends
// ---------------------------------------------------------------------------

/// A clearly-labelled, **non-neural** diagnostic synthesizer. It produces a
/// distinct pitched tone per voice so buffering, timing and file output can be
/// tested without model assets. It is not speech and must not be used as
/// evidence of TTS quality.
pub struct DiagnosticToneSynth {
    voices: Vec<(VoiceInfo, f32)>,
}

impl DiagnosticToneSynth {
    pub fn new() -> Self {
        Self {
            voices: vec![
                (
                    VoiceInfo {
                        id: "diag-low".into(),
                        display_name: "Diagnostic Low (non-neural)".into(),
                        language: "en".into(),
                        license: "diagnostic-only".into(),
                        attribution: "not speech".into(),
                    },
                    160.0,
                ),
                (
                    VoiceInfo {
                        id: "diag-high".into(),
                        display_name: "Diagnostic High (non-neural)".into(),
                        language: "en".into(),
                        license: "diagnostic-only".into(),
                        attribution: "not speech".into(),
                    },
                    320.0,
                ),
            ],
        }
    }
}

impl Default for DiagnosticToneSynth {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeechSynthesizer for DiagnosticToneSynth {
    fn backend_name(&self) -> String {
        "diagnostic-tone (NOT NEURAL)".into()
    }

    fn voices(&self) -> Vec<VoiceInfo> {
        self.voices.iter().map(|(v, _)| v.clone()).collect()
    }

    fn warm(&mut self, voice_id: &str) -> Result<(), SpeechError> {
        if self.voices.iter().any(|(v, _)| v.id == voice_id) {
            Ok(())
        } else {
            Err(SpeechError::UnknownVoice(voice_id.into()))
        }
    }

    fn synthesize(&mut self, req: &SpeechRequest) -> Result<SpeechAudio, SpeechError> {
        let freq = self
            .voices
            .iter()
            .find(|(v, _)| v.id == req.voice_id)
            .map(|(_, f)| *f)
            .ok_or_else(|| SpeechError::UnknownVoice(req.voice_id.clone()))?;
        let rate = req.sample_rate.max(8_000);
        // Deterministic duration: ~0.32s per word, minimum 0.4s.
        let words = req.text.split_whitespace().count().max(1);
        let secs = (words as f32 * 0.32).max(0.4);
        let n = (secs * rate as f32) as usize;
        let mut samples = Vec::with_capacity(n);
        for k in 0..n {
            let t = k as f32 / rate as f32;
            // Gentle envelope avoids clicks; two harmonics sound less harsh.
            let env = (t * 8.0).min(1.0) * (-((t - secs).max(0.0) * 8.0)).exp();
            let s = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.3
                + (2.0 * std::f32::consts::PI * freq * 2.0 * t).sin() * 0.1;
            samples.push(s * env);
        }
        Ok(SpeechAudio {
            samples,
            sample_rate: rate,
            synth_ms: 1.0,
            first_audio_ms: 1.0,
        })
    }
}

/// Placeholder for a real local neural backend. Fails loudly and tells the
/// operator exactly what is missing; it never fabricates audio.
pub struct MissingNeuralBackend {
    pub expected_asset: String,
}

impl SpeechSynthesizer for MissingNeuralBackend {
    fn backend_name(&self) -> String {
        "neural-not-installed".into()
    }
    fn voices(&self) -> Vec<VoiceInfo> {
        Vec::new()
    }
    fn warm(&mut self, _voice_id: &str) -> Result<(), SpeechError> {
        Err(SpeechError::MissingAsset(self.expected_asset.clone()))
    }
    fn synthesize(&mut self, _req: &SpeechRequest) -> Result<SpeechAudio, SpeechError> {
        Err(SpeechError::MissingAsset(self.expected_asset.clone()))
    }
}

// ---------------------------------------------------------------------------
// Integrated speech-performance measurement
// ---------------------------------------------------------------------------

/// Measures whether the *complete* text→audio pipeline stays ahead of playback.
#[derive(Clone, Debug, Default)]
pub struct SpeechRunMetrics {
    /// Total playable audio seconds produced.
    pub audio_seconds: f32,
    /// Total wall seconds spent producing it.
    pub wall_seconds: f32,
    /// Number of chunks produced.
    pub chunks: usize,
    /// Largest shortfall of buffered audio relative to playback, in seconds.
    pub min_lead_seconds: f32,
    /// Buffer underruns observed during playback.
    pub underruns: u32,
    /// First-audio latency (ms) for the last chunk.
    pub first_audio_ms: f32,
}

impl SpeechRunMetrics {
    /// audio-seconds generated per wall-second elapsed.
    pub fn speed_ratio(&self) -> f32 {
        if self.wall_seconds <= 0.0 {
            f32::INFINITY
        } else {
            self.audio_seconds / self.wall_seconds
        }
    }
}

/// Simulate streaming playback and count underruns. Each chunk is
/// `(audio_seconds, wall_seconds)`.
pub fn measure_streaming(chunks: &[(f32, f32)]) -> SpeechRunMetrics {
    let mut m = SpeechRunMetrics {
        min_lead_seconds: f32::INFINITY,
        ..Default::default()
    };
    let mut buffer = 0.0f32;
    let mut clock = 0.0f32;
    for (audio, wall) in chunks {
        // Production and playback are concurrent within a chunk: add the
        // produced audio, then let playback drain the buffer by the wall time.
        buffer += audio;
        buffer -= wall;
        if buffer < 0.0 {
            m.underruns += 1;
            buffer = 0.0;
        }
        clock += wall;
        m.audio_seconds += audio;
        m.wall_seconds += wall;
        m.chunks += 1;
        m.min_lead_seconds = m.min_lead_seconds.min(buffer);
    }
    if m.min_lead_seconds.is_infinite() {
        m.min_lead_seconds = 0.0;
    }
    m
}

/// Convenience for a single chunk measurement.
pub fn measure_chunk(audio: &SpeechAudio, wall_seconds: f32) -> SpeechRunMetrics {
    measure_streaming(&[(audio.audio_seconds(), wall_seconds)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_roundtrip_preserves_length_and_rate() {
        let audio = SpeechAudio {
            samples: (0..1000).map(|i| ((i as f32) * 0.01).sin() * 0.5).collect(),
            sample_rate: 22_050,
            synth_ms: 0.0,
            first_audio_ms: 0.0,
        };
        let bytes = encode_wav(&audio);
        let back = decode_wav(&bytes).unwrap();
        assert_eq!(back.sample_rate, 22_050);
        assert_eq!(back.samples.len(), audio.samples.len());
        assert!((back.samples[123] - audio.samples[123]).abs() < 0.001);
    }

    #[test]
    fn diagnostic_synth_produces_nonsilent_pcm() {
        let mut synth = DiagnosticToneSynth::new();
        synth.warm("diag-low").unwrap();
        let audio = synth
            .synthesize(&SpeechRequest {
                text: "The ledger is behind the flour.".into(),
                voice_id: "diag-low".into(),
                sample_rate: 24_000,
            })
            .unwrap();
        assert!(!audio.is_silent());
        assert!(audio.audio_seconds() > 0.5);
        assert!(audio.real_time_factor() > 0.0);
    }

    #[test]
    fn voices_are_distinguishable() {
        let mut synth = DiagnosticToneSynth::new();
        let a = synth
            .synthesize(&SpeechRequest {
                text: "hello".into(),
                voice_id: "diag-low".into(),
                sample_rate: 8_000,
            })
            .unwrap();
        let b = synth
            .synthesize(&SpeechRequest {
                text: "hello".into(),
                voice_id: "diag-high".into(),
                sample_rate: 8_000,
            })
            .unwrap();
        let diff: f32 = a
            .samples
            .iter()
            .zip(&b.samples)
            .map(|(x, y)| (x - y).abs())
            .sum();
        assert!(diff > 1.0, "voices should differ, diff={diff}");
    }

    #[test]
    fn missing_neural_backend_fails_loudly() {
        let mut b = MissingNeuralBackend {
            expected_asset: "voices/kokoro.onnx".into(),
        };
        assert!(matches!(
            b.synthesize(&SpeechRequest {
                text: "x".into(),
                voice_id: "v".into(),
                sample_rate: 22050,
            }),
            Err(SpeechError::MissingAsset(_))
        ));
    }

    #[test]
    fn streaming_metric_reports_underruns_and_ratio() {
        // Two chunks that cannot keep up with playback.
        let m = measure_streaming(&[(1.0, 2.0), (0.5, 2.0)]);
        assert_eq!(m.chunks, 2);
        assert!(m.underruns >= 1, "underruns={}", m.underruns);
        assert!(m.speed_ratio() > 0.0);
    }

    #[test]
    fn fast_producer_has_no_underruns() {
        let m = measure_streaming(&[(6.0, 0.01), (6.0, 0.02)]);
        assert_eq!(m.underruns, 0);
        assert!(m.speed_ratio() > 100.0);
    }
}
