//! Production backends for the runtime's seams: the cpal microphone and
//! speaker, Silero VAD, Whisper ASR, Kokoro TTS and the resident llama.cpp
//! model.
//!
//! Each factory runs on the worker thread that will own the backend, so the
//! native handles (audio streams, ONNX sessions, the model) never cross
//! threads, and none of them is ever touched by the simulation thread.

use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use rpg_domain::ports::{InferenceEngine, SpeechRecognizer, SpeechSynthesizer};
use rpg_inference::{LlamaEngine, ModelProfile};
use rpg_runtime::backends::*;
use rpg_runtime::wav_mic::WavMicBackend;
use rpg_runtime::MicInfo;
use rpg_speech::assets::SpeechAssets;
use rpg_speech::audio::{AudioCapture, AudioPlayback, CaptureConfig, PlaybackConfig};
use rpg_speech::{SherpaAsr, SherpaTts, VadSegmenter};

use crate::paths::{workspace_cache, workspace_manifest};

/// What to load, from the command line.
#[derive(Clone, Debug, Default)]
pub struct VoiceOptions {
    pub model_path: Option<String>,
    /// Fallback Kokoro voice for unnamed speakers.
    pub voice: String,
    pub no_voice: bool,
    /// Input device by name; `None` uses the host default.
    pub mic_device: Option<String>,
    /// Use a WAV file as the microphone (prerecorded-device testing).
    pub mic_wav: Option<PathBuf>,
}

pub fn backends(opts: &VoiceOptions) -> Backends {
    let engine = engine_factory(opts.model_path.clone());
    if opts.no_voice {
        return Backends {
            engine,
            mic: None,
            asr: None,
            tts: None,
        };
    }
    let mic: Box<dyn MicBackend> = match &opts.mic_wav {
        Some(path) => Box::new(WavMicBackend::new(path)),
        None => Box::new(CpalMic),
    };
    Backends {
        engine,
        mic: Some((mic, vad_factory())),
        asr: Some(asr_factory()),
        tts: Some(tts_factory(opts.voice.clone())),
    }
}

/// Threads for the native engines. One core is left for the simulation,
/// renderer and audio threads so heavy inference cannot starve them.
pub fn engine_threads() -> i32 {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4);
    (cores - 1).clamp(1, 16)
}

fn assets() -> Result<SpeechAssets, String> {
    SpeechAssets::load(workspace_cache(), workspace_manifest())
        .map_err(|e| format!("speech assets: {e} (run `voice-demo install`)"))
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

fn engine_factory(path: Option<String>) -> EngineFactory {
    Box::new(move || {
        let Some(path) = path else {
            return Ok(LoadedEngine {
                engine: Box::new(rpg_agents::HeuristicEngine::new()),
                interrupt: None,
            });
        };
        if !std::path::Path::new(&path).exists() {
            return Err(format!("model file not found: {path}"));
        }
        let path = path.replace('\\', "/");
        let json = format!(
            r#"{{"id":"bevy","model_path":"{path}","n_ctx":4096,"n_threads":{threads},"max_tokens":256,"temperature":0.4,"no_think":true,"deadline_ms":60000}}"#,
            threads = engine_threads()
        );
        let profile = ModelProfile::from_json(&json).map_err(|e| e.to_string())?;
        let mut engine = LlamaEngine::load(profile).map_err(|e| e.to_string())?;
        engine.warm().map_err(|e| e.to_string())?;
        let interrupt = Some(engine.cancel_handle());
        Ok(LoadedEngine {
            engine: Box::new(engine),
            interrupt,
        })
    })
}

// ---------------------------------------------------------------------------
// Microphone + VAD
// ---------------------------------------------------------------------------

struct CpalMic;

impl MicBackend for CpalMic {
    fn open(&mut self, device: Option<&str>) -> Result<Box<dyn MicSource>, String> {
        let capture = AudioCapture::open(&CaptureConfig {
            device_name: device.map(str::to_string),
            ..CaptureConfig::default()
        })
        .map_err(|e| e.to_string())?;
        Ok(Box::new(CpalMicSource(capture)))
    }

    fn list_devices(&mut self) -> Vec<String> {
        AudioCapture::list_devices()
            .map(|d| d.into_iter().map(|d| d.name).collect())
            .unwrap_or_default()
    }
}

struct CpalMicSource(AudioCapture);

impl MicSource for CpalMicSource {
    fn info(&self) -> MicInfo {
        MicInfo {
            name: self.0.device_name.clone(),
            sample_rate: self.0.native_sample_rate,
            channels: self.0.device_channels,
            sample_format: self.0.sample_format.clone(),
        }
    }

    fn try_recv(&mut self) -> Option<Vec<f32>> {
        self.0.try_recv()
    }

    fn clear(&mut self) -> usize {
        self.0.clear()
    }

    fn dropped_chunks(&self) -> u64 {
        self.0.dropped_chunks()
    }

    fn take_error(&mut self) -> Option<String> {
        self.0.take_error()
    }
}

struct SileroVad(VadSegmenter);

impl Vad for SileroVad {
    fn accept(&mut self, samples: &[f32]) -> Vec<Vec<f32>> {
        self.0
            .accept(samples)
            .into_iter()
            .map(|s| s.samples)
            .collect()
    }

    fn flush(&mut self) -> Vec<Vec<f32>> {
        self.0.flush().into_iter().map(|s| s.samples).collect()
    }

    fn speaking(&self) -> bool {
        self.0.speaking()
    }

    fn reset(&mut self) {
        self.0.reset();
    }
}

fn vad_factory() -> VadFactory {
    Box::new(|| {
        let assets = assets()?;
        let vad = VadSegmenter::from_assets(
            &assets,
            rpg_speech::DEFAULT_VAD_BUNDLE,
            rpg_runtime::ASR_RATE,
        )
        .map_err(|e| e.to_string())?;
        Ok(Box::new(SileroVad(vad)) as Box<dyn Vad>)
    })
}

// ---------------------------------------------------------------------------
// Speech recognition
// ---------------------------------------------------------------------------

fn asr_factory() -> AsrFactory {
    Box::new(|| {
        let assets = assets()?;
        let mut asr =
            SherpaAsr::from_assets(&assets, rpg_speech::DEFAULT_ASR_BUNDLE, engine_threads())
                .map_err(|e| e.to_string())?;
        asr.warm().map_err(|e| e.to_string())?;
        Ok(Box::new(asr) as Box<dyn SpeechRecognizer>)
    })
}

// ---------------------------------------------------------------------------
// Speech synthesis + playback
// ---------------------------------------------------------------------------

struct KokoroVoice {
    tts: SherpaTts,
    default_voice: String,
}

impl Synthesizer for KokoroVoice {
    fn backend_name(&self) -> String {
        self.tts.backend_name()
    }

    fn sample_rate(&self) -> u32 {
        self.tts.sample_rate()
    }

    /// Named NPCs receive a deterministic, distinct voice so the cast is
    /// audibly separable; the `--voice` choice is only the fallback.
    fn voice_for(&self, speaker: &str) -> String {
        if speaker == "you" || speaker == "system" {
            self.default_voice.clone()
        } else {
            self.tts.voice_for_actor(speaker).id
        }
    }

    fn synthesize(
        &mut self,
        text: &str,
        voice: &str,
        sink: ChunkSink,
    ) -> Result<SynthStats, String> {
        let s = self
            .tts
            .synthesize_streaming(text, voice, sink)
            .map_err(|e| e.to_string())?;
        Ok(SynthStats {
            samples: s.samples,
            first_chunk_ms: s.first_chunk_ms,
            synth_ms: s.synth_ms,
            stopped: s.stopped,
        })
    }
}

struct CpalSpeaker(AudioPlayback);

impl AudioSink for CpalSpeaker {
    fn device_name(&self) -> String {
        self.0.device_name.clone()
    }

    fn sample_rate(&self) -> u32 {
        self.0.sample_rate
    }

    fn try_submit(&mut self, epoch: u64, samples: Vec<f32>) -> Result<(), Vec<f32>> {
        self.0.try_submit(epoch, samples)
    }

    fn queued(&self) -> u64 {
        self.0.queued_samples()
    }

    fn consumed(&self) -> u64 {
        self.0.consumed_samples()
    }

    fn underruns(&self) -> u64 {
        self.0.underruns()
    }

    fn set_streaming(&mut self, streaming: bool) {
        self.0.set_streaming(streaming);
    }

    fn take_error(&mut self) -> Option<String> {
        self.0.take_error()
    }
}

fn tts_factory(default_voice: String) -> TtsFactory {
    Box::new(move |epoch: Arc<AtomicU64>| {
        let assets = assets()?;
        let mut tts =
            SherpaTts::from_assets(&assets, rpg_speech::DEFAULT_TTS_BUNDLE, engine_threads())
                .map_err(|e| e.to_string())?;
        tts.warm(&default_voice).map_err(|e| e.to_string())?;
        let playback = AudioPlayback::open(&PlaybackConfig {
            epoch: Some(epoch),
            ..PlaybackConfig::new(tts.sample_rate())
        })
        .map_err(|e| format!("speaker: {e}"))?;
        Ok((
            Box::new(KokoroVoice { tts, default_voice }) as Box<dyn Synthesizer>,
            Box::new(CpalSpeaker(playback)) as Box<dyn AudioSink>,
        ))
    })
}
