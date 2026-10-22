//! Controllable fakes for every backend seam.
//!
//! They let the deterministic tests (and anyone debugging the pipeline
//! without hardware) drive the *production* runtime — the same threads,
//! channels, state machine and validation — while deciding exactly when audio
//! arrives, when the model answers and when playback finishes. Gates replace
//! sleeps: a test opens a gate when it wants a stage to complete, so nothing
//! depends on timing luck.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use rpg_domain::ports::{
    InferenceEngine, InferenceError, InferenceRequest, InferenceResponse, RecognitionError,
    SpeechRecognizer, Transcript,
};
use rpg_speech::dsp::{downmix, Level};

use crate::backends::*;
use crate::protocol::MicInfo;

// ---------------------------------------------------------------------------
// Gate
// ---------------------------------------------------------------------------

/// A latch a backend waits on until the test opens it.
#[derive(Clone, Default)]
pub struct Gate {
    inner: Arc<(Mutex<bool>, Condvar)>,
    waiting: Arc<AtomicUsize>,
}

impl Gate {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn open(&self) {
        let (lock, cv) = &*self.inner;
        *lock.lock().unwrap() = true;
        cv.notify_all();
    }

    pub fn close(&self) {
        *self.inner.0.lock().unwrap() = false;
    }

    /// How many waits have started (a waiter has *reached* the gate).
    pub fn arrivals(&self) -> usize {
        self.waiting.load(Ordering::SeqCst)
    }

    /// Wait until open, `abort` is set, or `timeout` passes. Returns whether
    /// the gate was opened.
    pub fn wait(&self, abort: Option<&AtomicBool>, timeout: Duration) -> bool {
        self.waiting.fetch_add(1, Ordering::SeqCst);
        let (lock, cv) = &*self.inner;
        let deadline = std::time::Instant::now() + timeout;
        let mut open = lock.lock().unwrap();
        while !*open {
            if abort.is_some_and(|a| a.load(Ordering::SeqCst)) {
                return false;
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                return false;
            }
            let step = (deadline - now).min(Duration::from_millis(2));
            open = cv.wait_timeout(open, step).unwrap().0;
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

/// A "real model" whose answers and timing the test controls.
pub struct FakeEngine {
    pub responses: Arc<Mutex<VecDeque<String>>>,
    pub default_response: String,
    pub gate: Option<Gate>,
    /// A genuine wall-clock delay per request (for the slow-model test).
    pub delay: Option<Duration>,
    pub interrupt: Arc<AtomicBool>,
    pub seen: Arc<Mutex<Vec<u64>>>,
}

impl FakeEngine {
    pub fn new() -> Self {
        Self {
            responses: Arc::default(),
            default_response: r#"{"action":"speak","text":"Good day to you."}"#.into(),
            gate: None,
            delay: None,
            interrupt: Arc::default(),
            seen: Arc::default(),
        }
    }

    pub fn with_gate(mut self, gate: Gate) -> Self {
        self.gate = Some(gate);
        self
    }

    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = Some(delay);
        self
    }

    pub fn with_default(mut self, json: &str) -> Self {
        self.default_response = json.into();
        self
    }

    /// Package as a factory, keeping the interrupt flag wired.
    pub fn factory(self) -> EngineFactory {
        Box::new(move || {
            let interrupt = Some(self.interrupt.clone());
            Ok(LoadedEngine {
                engine: Box::new(self),
                interrupt,
            })
        })
    }
}

impl Default for FakeEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceEngine for FakeEngine {
    fn engine_name(&self) -> String {
        "fake-engine".into()
    }

    fn model_id(&self) -> String {
        "fake-model".into()
    }

    fn warm(&mut self) -> Result<(), InferenceError> {
        Ok(())
    }

    fn complete(&mut self, req: &InferenceRequest) -> Result<InferenceResponse, InferenceError> {
        self.seen.lock().unwrap().push(req.id);
        if let Some(gate) = &self.gate {
            if !gate.wait(Some(&self.interrupt), Duration::from_secs(120)) {
                return Err(InferenceError::Cancelled);
            }
        }
        if let Some(d) = self.delay {
            let end = std::time::Instant::now() + d;
            while std::time::Instant::now() < end {
                if self.interrupt.load(Ordering::SeqCst) {
                    return Err(InferenceError::Cancelled);
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        }
        let text = self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| self.default_response.clone());
        Ok(InferenceResponse {
            text,
            completion_tokens: 8,
            ..Default::default()
        })
    }
}

/// A deterministic, non-model engine (the explicit heuristic mode).
pub fn heuristic_engine() -> EngineFactory {
    Box::new(|| {
        Ok(LoadedEngine {
            engine: Box::new(rpg_agents::HeuristicEngine::new()),
            interrupt: None,
        })
    })
}

// ---------------------------------------------------------------------------
// Speech recognition
// ---------------------------------------------------------------------------

/// Records what it was asked to transcribe and answers with a script.
pub struct FakeAsr {
    pub transcripts: Arc<Mutex<VecDeque<String>>>,
    pub default_text: String,
    pub gate: Option<Gate>,
    /// `(samples, sample_rate, rms)` of every request.
    pub received: Arc<Mutex<Vec<(usize, u32, f32)>>>,
    pub fail: Option<String>,
}

impl FakeAsr {
    pub fn new(text: &str) -> Self {
        Self {
            transcripts: Arc::default(),
            default_text: text.into(),
            gate: None,
            received: Arc::default(),
            fail: None,
        }
    }

    pub fn factory(self) -> AsrFactory {
        Box::new(move || Ok(Box::new(self) as Box<dyn SpeechRecognizer>))
    }
}

impl SpeechRecognizer for FakeAsr {
    fn backend_name(&self) -> String {
        "fake-asr".into()
    }

    fn warm(&mut self) -> Result<(), RecognitionError> {
        Ok(())
    }

    fn transcribe(&mut self, pcm: &[f32], rate: u32) -> Result<Transcript, RecognitionError> {
        self.received
            .lock()
            .unwrap()
            .push((pcm.len(), rate, Level::of(pcm).rms));
        if let Some(gate) = &self.gate {
            gate.wait(None, Duration::from_secs(120));
        }
        if let Some(e) = &self.fail {
            return Err(RecognitionError::Backend(e.clone()));
        }
        let text = self
            .transcripts
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| self.default_text.clone());
        Ok(Transcript {
            text,
            confidence: 1.0,
            is_final: true,
            latency_ms: 0.0,
            fallback: false,
        })
    }
}

// ---------------------------------------------------------------------------
// Microphone + VAD
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct FakeMicHandle {
    pub info: Arc<Mutex<MicInfo>>,
    queue: Arc<Mutex<VecDeque<Vec<f32>>>>,
    error: Arc<Mutex<Option<String>>>,
    pub fail_open: Arc<AtomicBool>,
    pub opens: Arc<AtomicUsize>,
    pub opened_with: Arc<Mutex<Vec<Option<String>>>>,
}

impl FakeMicHandle {
    /// Deliver interleaved device frames the way a real device callback
    /// does: downmixed to mono exactly once, at the device rate.
    pub fn push_interleaved(&self, frames: &[f32]) {
        let ch = self.info.lock().unwrap().channels;
        self.queue.lock().unwrap().push_back(downmix(frames, ch));
    }

    /// Deliver `samples` (mono, device rate) in device-sized chunks.
    pub fn push_mono_chunked(&self, samples: &[f32], chunk: usize) {
        let mut q = self.queue.lock().unwrap();
        for c in samples.chunks(chunk.max(1)) {
            q.push_back(c.to_vec());
        }
    }

    pub fn queued_chunks(&self) -> usize {
        self.queue.lock().unwrap().len()
    }

    /// Simulate the device disappearing.
    pub fn fail(&self, message: &str) {
        *self.error.lock().unwrap() = Some(message.into());
    }
}

pub struct FakeMicBackend {
    handle: FakeMicHandle,
    devices: Vec<String>,
}

impl FakeMicBackend {
    pub fn new(sample_rate: u32, channels: u16) -> (Self, FakeMicHandle) {
        let handle = FakeMicHandle {
            info: Arc::new(Mutex::new(MicInfo {
                name: "fake-mic".into(),
                sample_rate,
                channels,
                sample_format: "F32".into(),
            })),
            queue: Arc::default(),
            error: Arc::default(),
            fail_open: Arc::default(),
            opens: Arc::default(),
            opened_with: Arc::default(),
        };
        (
            Self {
                handle: handle.clone(),
                devices: vec!["fake-mic".into(), "fake-mic-2".into()],
            },
            handle,
        )
    }
}

impl MicBackend for FakeMicBackend {
    fn open(&mut self, device: Option<&str>) -> Result<Box<dyn MicSource>, String> {
        self.handle.opens.fetch_add(1, Ordering::SeqCst);
        self.handle
            .opened_with
            .lock()
            .unwrap()
            .push(device.map(str::to_string));
        if self.handle.fail_open.load(Ordering::SeqCst) {
            return Err("fake device refused to open".into());
        }
        if let Some(name) = device {
            self.handle.info.lock().unwrap().name = name.to_string();
        }
        *self.handle.error.lock().unwrap() = None;
        Ok(Box::new(FakeMicSource {
            handle: self.handle.clone(),
        }))
    }

    fn list_devices(&mut self) -> Vec<String> {
        self.devices.clone()
    }
}

struct FakeMicSource {
    handle: FakeMicHandle,
}

impl MicSource for FakeMicSource {
    fn info(&self) -> MicInfo {
        self.handle.info.lock().unwrap().clone()
    }

    fn try_recv(&mut self) -> Option<Vec<f32>> {
        self.handle.queue.lock().unwrap().pop_front()
    }

    fn clear(&mut self) -> usize {
        let mut q = self.handle.queue.lock().unwrap();
        let n = q.iter().map(Vec::len).sum();
        q.clear();
        n
    }

    fn dropped_chunks(&self) -> u64 {
        0
    }

    fn take_error(&mut self) -> Option<String> {
        self.handle.error.lock().unwrap().take()
    }
}

/// A deterministic energy-threshold VAD over 16 kHz mono (20 ms frames).
pub struct EnergyVad {
    pub threshold: f32,
    /// Silent frames that end a segment.
    pub hangover_frames: usize,
    frame: Vec<f32>,
    current: Vec<f32>,
    silent: usize,
    speaking: bool,
    done: Vec<Vec<f32>>,
    pub resets: Arc<AtomicUsize>,
}

const VAD_FRAME: usize = 320;

impl EnergyVad {
    pub fn new() -> Self {
        Self {
            threshold: 0.02,
            hangover_frames: 15,
            frame: Vec::new(),
            current: Vec::new(),
            silent: 0,
            speaking: false,
            done: Vec::new(),
            resets: Arc::default(),
        }
    }

    pub fn factory() -> VadFactory {
        Box::new(|| Ok(Box::new(EnergyVad::new()) as Box<dyn Vad>))
    }

    fn frame_done(&mut self) {
        let frame = std::mem::take(&mut self.frame);
        let loud = Level::of(&frame).rms >= self.threshold;
        if loud {
            self.speaking = true;
            self.silent = 0;
            self.current.extend_from_slice(&frame);
        } else if self.speaking {
            self.current.extend_from_slice(&frame);
            self.silent += 1;
            if self.silent >= self.hangover_frames {
                self.speaking = false;
                self.silent = 0;
                self.done.push(std::mem::take(&mut self.current));
            }
        }
    }
}

impl Default for EnergyVad {
    fn default() -> Self {
        Self::new()
    }
}

impl Vad for EnergyVad {
    fn accept(&mut self, samples: &[f32]) -> Vec<Vec<f32>> {
        for s in samples {
            self.frame.push(*s);
            if self.frame.len() == VAD_FRAME {
                self.frame_done();
            }
        }
        std::mem::take(&mut self.done)
    }

    fn flush(&mut self) -> Vec<Vec<f32>> {
        if !self.current.is_empty() {
            self.done.push(std::mem::take(&mut self.current));
        }
        self.speaking = false;
        std::mem::take(&mut self.done)
    }

    fn speaking(&self) -> bool {
        self.speaking
    }

    fn reset(&mut self) {
        self.frame.clear();
        self.current.clear();
        self.done.clear();
        self.silent = 0;
        self.speaking = false;
        self.resets.fetch_add(1, Ordering::SeqCst);
    }
}

// ---------------------------------------------------------------------------
// Synthesis + playback
// ---------------------------------------------------------------------------

/// Emits `chunks` tone chunks per line, optionally waiting on a gate before
/// each one (to hold the pipeline in "synthesizing").
pub struct FakeSynth {
    pub rate: u32,
    pub chunks: usize,
    pub chunk_samples: usize,
    pub gate: Option<Gate>,
    pub spoken: Arc<Mutex<Vec<String>>>,
    pub fail: Option<String>,
}

impl FakeSynth {
    pub fn new() -> Self {
        Self {
            rate: 24_000,
            chunks: 3,
            chunk_samples: 2_400,
            gate: None,
            spoken: Arc::default(),
            fail: None,
        }
    }
}

impl Default for FakeSynth {
    fn default() -> Self {
        Self::new()
    }
}

impl Synthesizer for FakeSynth {
    fn backend_name(&self) -> String {
        "fake-voice".into()
    }

    fn sample_rate(&self) -> u32 {
        self.rate
    }

    fn voice_for(&self, speaker: &str) -> String {
        format!("voice-{speaker}")
    }

    fn synthesize(
        &mut self,
        text: &str,
        _voice: &str,
        mut sink: ChunkSink,
    ) -> Result<SynthStats, String> {
        if let Some(e) = &self.fail {
            return Err(e.clone());
        }
        self.spoken.lock().unwrap().push(text.to_string());
        let mut stats = SynthStats::default();
        let chunk: Vec<f32> = (0..self.chunk_samples)
            .map(|i| (i as f32 * 0.05).sin() * 0.3)
            .collect();
        for _ in 0..self.chunks {
            if let Some(g) = &self.gate {
                g.wait(None, Duration::from_secs(120));
            }
            stats.samples += chunk.len();
            if !sink(&chunk) {
                stats.stopped = true;
                break;
            }
        }
        Ok(stats)
    }
}

/// Shared state of a [`FakeSink`].
#[derive(Default)]
pub struct FakeSinkState {
    queue: VecDeque<(u64, Vec<f32>)>,
    pub queued: u64,
    pub consumed: u64,
    /// Samples that actually reached the "speaker", per epoch.
    pub heard: Vec<(u64, usize)>,
    pub discarded: u64,
    /// Play everything the moment it is queued.
    pub instant: bool,
}

#[derive(Clone)]
pub struct FakeSinkHandle {
    pub state: Arc<Mutex<FakeSinkState>>,
    epoch: Arc<AtomicU64>,
}

impl FakeSinkHandle {
    /// Play (or discard, if stale) everything queued so far.
    pub fn play_all(&self) {
        let live = self.epoch.load(Ordering::Acquire);
        let mut s = self.state.lock().unwrap();
        while let Some((epoch, buf)) = s.queue.pop_front() {
            s.consumed += buf.len() as u64;
            if epoch < live {
                s.discarded += buf.len() as u64;
            } else {
                s.heard.push((epoch, buf.len()));
            }
        }
    }

    pub fn heard_samples(&self) -> usize {
        self.state.lock().unwrap().heard.iter().map(|h| h.1).sum()
    }

    pub fn pending(&self) -> usize {
        self.state.lock().unwrap().queue.len()
    }
}

pub struct FakeSink {
    rate: u32,
    handle: FakeSinkHandle,
}

impl AudioSink for FakeSink {
    fn device_name(&self) -> String {
        "fake-speaker".into()
    }

    fn sample_rate(&self) -> u32 {
        self.rate
    }

    fn try_submit(&mut self, epoch: u64, samples: Vec<f32>) -> Result<(), Vec<f32>> {
        let instant = {
            let mut s = self.handle.state.lock().unwrap();
            s.queued += samples.len() as u64;
            s.queue.push_back((epoch, samples));
            s.instant
        };
        if instant {
            self.handle.play_all();
        }
        Ok(())
    }

    fn queued(&self) -> u64 {
        self.handle.state.lock().unwrap().queued
    }

    fn consumed(&self) -> u64 {
        self.handle.state.lock().unwrap().consumed
    }

    fn underruns(&self) -> u64 {
        0
    }

    fn set_streaming(&mut self, _streaming: bool) {}

    fn take_error(&mut self) -> Option<String> {
        None
    }
}

/// A TTS factory pairing `synth` with a manually-driven fake speaker.
/// The returned handle becomes usable once the TTS worker has started.
pub fn fake_tts(
    synth: FakeSynth,
    device_rate: u32,
    instant: bool,
) -> (TtsFactory, Arc<Mutex<Option<FakeSinkHandle>>>) {
    let slot: Arc<Mutex<Option<FakeSinkHandle>>> = Arc::default();
    let out = slot.clone();
    let factory: TtsFactory = Box::new(move |epoch: Arc<AtomicU64>| {
        let handle = FakeSinkHandle {
            state: Arc::new(Mutex::new(FakeSinkState {
                instant,
                ..Default::default()
            })),
            epoch,
        };
        *slot.lock().unwrap() = Some(handle.clone());
        Ok((
            Box::new(synth) as Box<dyn Synthesizer>,
            Box::new(FakeSink {
                rate: device_rate,
                handle,
            }) as Box<dyn AudioSink>,
        ))
    });
    (factory, out)
}

/// `secs` of a tone at `rate`, suitable as fake speech.
pub fn tone(rate: u32, hz: f32, secs: f32, amp: f32) -> Vec<f32> {
    let n = (rate as f32 * secs) as usize;
    (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * hz * i as f32 / rate as f32).sin() * amp)
        .collect()
}

/// Interleave mono into `channels` identical channels.
pub fn interleave(mono: &[f32], channels: u16) -> Vec<f32> {
    mono.iter()
        .flat_map(|s| std::iter::repeat_n(*s, channels as usize))
        .collect()
}
