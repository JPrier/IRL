//! Worker threads. Each owns one native backend (created on that thread by
//! its factory), receives requests over an unbounded channel and reports
//! results to the simulation inbox. A worker may block inside its backend —
//! that is what it is for — but nothing ever waits on a worker.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use rpg_domain::ports::{InferenceError, SpeechAudio};
use rpg_speech::dsp::{Level, LinearResampler};

use crate::backends::*;
use crate::protocol::*;
use crate::telemetry::{Telemetry, NONE};

/// Sample rate of everything the VAD and ASR see.
pub const ASR_RATE: u32 = 16_000;

fn spawn_named(name: &str, f: impl FnOnce() + Send + 'static) -> JoinHandle<()> {
    thread::Builder::new()
        .name(name.into())
        .spawn(f)
        .expect("spawn worker thread")
}

// ---------------------------------------------------------------------------
// Microphone + VAD
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct MicConfig {
    /// Initial device; `None` uses the host default.
    pub device: Option<String>,
    /// A recording stops by itself after this long.
    pub max_record: Duration,
    /// Stop automatically once the VAD has heard speech followed by this
    /// much silence (push-to-talk still works; this only ends a turn early).
    pub auto_endpoint: Option<Duration>,
    /// How often the device is polled.
    pub poll: Duration,
    /// How long to wait before reopening a failed device.
    pub retry: Duration,
    /// Write the 16 kHz audio sent to ASR here, for debugging.
    pub dump_utterance: Option<PathBuf>,
}

impl Default for MicConfig {
    fn default() -> Self {
        Self {
            device: None,
            max_record: Duration::from_secs(15),
            auto_endpoint: None,
            poll: Duration::from_millis(5),
            retry: Duration::from_secs(2),
            dump_utterance: None,
        }
    }
}

struct Recording {
    conv: ConvId,
    started: Instant,
    /// 16 kHz mono since the key went down.
    audio: Vec<f32>,
    segments: Vec<Vec<f32>>,
    last_speech: Option<Instant>,
}

struct SelfTest {
    path: String,
    seconds: f32,
    started: Instant,
    samples: Vec<f32>,
    dropped_at_start: u64,
}

struct MicWorker {
    cfg: MicConfig,
    backend: Box<dyn MicBackend>,
    vad: Option<Box<dyn Vad>>,
    vad_error: Option<String>,
    source: Option<Box<dyn MicSource>>,
    device: Option<String>,
    reopen_at: Option<Instant>,
    resampler: LinearResampler,
    scratch: Vec<f32>,
    rec: Option<Recording>,
    self_test: Option<SelfTest>,
    out: Sender<Inbox>,
    tel: Arc<Telemetry>,
    rate_window: (Instant, u64),
}

pub fn spawn_mic(
    cfg: MicConfig,
    backend: Box<dyn MicBackend>,
    vad: VadFactory,
    rx: Receiver<MicRequest>,
    out: Sender<Inbox>,
    tel: Arc<Telemetry>,
) -> JoinHandle<()> {
    spawn_named("irl-mic", move || {
        let (vad, vad_error) = match vad() {
            Ok(v) => (Some(v), None),
            Err(e) => (None, Some(e)),
        };
        let mut w = MicWorker {
            device: cfg.device.clone(),
            cfg,
            backend,
            vad,
            vad_error,
            source: None,
            reopen_at: None,
            resampler: LinearResampler::new(ASR_RATE, ASR_RATE),
            scratch: Vec::new(),
            rec: None,
            self_test: None,
            out,
            tel,
            rate_window: (Instant::now(), 0),
        };
        w.run(rx);
    })
}

impl MicWorker {
    fn send(&self, ev: MicEvent) {
        let _ = self.out.send(Inbox::Mic(ev));
    }

    fn run(&mut self, rx: Receiver<MicRequest>) {
        if let Some(e) = &self.vad_error {
            self.send(MicEvent::DeviceError(format!(
                "voice activity detector: {e}"
            )));
            return;
        }
        let devices = self.backend.list_devices();
        self.send(MicEvent::Devices(devices));
        self.open();
        loop {
            match rx.recv_timeout(self.cfg.poll) {
                Ok(MicRequest::Shutdown) => break,
                Ok(req) => self.request(req),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => break,
            }
            self.pump();
            self.check_device();
            self.check_auto_stop();
            self.update_rate();
        }
        self.tel.mic_listening.store(false, Ordering::Relaxed);
    }

    fn open(&mut self) {
        match self.backend.open(self.device.as_deref()) {
            Ok(src) => {
                let info = src.info();
                self.resampler = LinearResampler::new(info.sample_rate, ASR_RATE);
                self.source = Some(src);
                self.reopen_at = None;
                self.send(MicEvent::DeviceReady(info));
            }
            Err(e) => {
                self.source = None;
                self.reopen_at = Some(Instant::now() + self.cfg.retry);
                self.send(MicEvent::DeviceError(e));
            }
        }
    }

    fn request(&mut self, req: MicRequest) {
        match req {
            MicRequest::Start { conv } => {
                let Some(src) = self.source.as_mut() else {
                    self.send(MicEvent::DeviceError("no microphone is open".into()));
                    return;
                };
                // Everything captured before the key press belongs to no one.
                let discarded = src.clear();
                self.resampler.reset();
                if let Some(vad) = self.vad.as_mut() {
                    vad.reset();
                }
                self.rec = Some(Recording {
                    conv,
                    started: Instant::now(),
                    audio: Vec::with_capacity(ASR_RATE as usize * 4),
                    segments: Vec::new(),
                    last_speech: None,
                });
                self.tel.mic_listening.store(true, Ordering::Relaxed);
                self.tel.recording_ms.store(0, Ordering::Relaxed);
                self.send(MicEvent::Listening { conv, discarded });
            }
            MicRequest::Stop { conv } => {
                if self.rec.as_ref().is_some_and(|r| r.conv == conv) {
                    // Take the audio up to the key release before finishing.
                    self.pump();
                    self.finish_recording();
                }
            }
            MicRequest::Cancel { conv } => {
                if self.rec.as_ref().is_some_and(|r| r.conv == conv) {
                    self.rec = None;
                    self.tel.mic_listening.store(false, Ordering::Relaxed);
                    self.tel.vad_speaking.store(false, Ordering::Relaxed);
                    if let Some(vad) = self.vad.as_mut() {
                        vad.reset();
                    }
                }
            }
            MicRequest::SelfTest { path, seconds } => {
                let dropped = self
                    .source
                    .as_ref()
                    .map(|s| s.dropped_chunks())
                    .unwrap_or(0);
                if let Some(src) = self.source.as_mut() {
                    src.clear();
                }
                self.self_test = Some(SelfTest {
                    path,
                    seconds: seconds.clamp(0.5, 30.0),
                    started: Instant::now(),
                    samples: Vec::new(),
                    dropped_at_start: dropped,
                });
            }
            MicRequest::Select(sel) => {
                let devices = self.backend.list_devices();
                // "Next" is relative to the device actually open, which may be
                // the host default rather than an explicitly chosen one.
                let current = self
                    .source
                    .as_ref()
                    .map(|s| s.info().name)
                    .or_else(|| self.device.clone());
                self.device = match sel {
                    MicSelection::Default => None,
                    MicSelection::Named(n) => Some(n),
                    MicSelection::Next => next_device(&devices, current.as_deref()),
                };
                self.send(MicEvent::Devices(devices));
                self.rec = None;
                self.source = None;
                self.open();
            }
            MicRequest::Shutdown => {}
        }
    }

    /// Drain the device queue. Runs every poll, also while idle, so the level
    /// meter is live and the device queue never fills with stale audio.
    fn pump(&mut self) {
        let Some(src) = self.source.as_mut() else {
            return;
        };
        let mut level = None;
        let mut got = 0u64;
        while let Some(chunk) = src.try_recv() {
            got += chunk.len() as u64;
            level = Some(Level::of(&chunk));
            if let Some(t) = self.self_test.as_mut() {
                t.samples.extend_from_slice(&chunk);
            }
            if let Some(rec) = self.rec.as_mut() {
                self.scratch.clear();
                self.resampler.process_into(&chunk, &mut self.scratch);
                rec.audio.extend_from_slice(&self.scratch);
                if let Some(vad) = self.vad.as_mut() {
                    let segs = vad.accept(&self.scratch);
                    let speaking = vad.speaking();
                    if speaking || !segs.is_empty() {
                        rec.last_speech = Some(Instant::now());
                    }
                    rec.segments.extend(segs);
                    self.tel.vad_speaking.store(speaking, Ordering::Relaxed);
                }
                self.tel.recording_ms.store(
                    rec.audio.len() as u64 * 1000 / ASR_RATE as u64,
                    Ordering::Relaxed,
                );
            }
        }
        self.tel
            .mic_dropped_chunks
            .store(src.dropped_chunks(), Ordering::Relaxed);
        self.tel.mic_samples.fetch_add(got, Ordering::Relaxed);
        if let Some(l) = level {
            // Peak hold with decay, so the meter reads as a level instead of
            // flickering with every 10 ms device buffer.
            let held = (self.tel.mic_peak.get() * 0.85).max(l.peak);
            self.tel.mic_rms.set(l.rms);
            self.tel.mic_peak.set(held);
        }
        self.check_self_test();
    }

    fn check_self_test(&mut self) {
        let done = self
            .self_test
            .as_ref()
            .is_some_and(|t| t.started.elapsed().as_secs_f32() >= t.seconds);
        if !done {
            return;
        }
        let Some(t) = self.self_test.take() else {
            return;
        };
        let result = match self.source.as_ref() {
            None => Err("the microphone closed during the test".to_string()),
            Some(src) => {
                let info = src.info();
                let level = Level::of(&t.samples);
                let audio = SpeechAudio {
                    samples: t.samples,
                    sample_rate: info.sample_rate,
                    synth_ms: 0.0,
                    first_audio_ms: 0.0,
                };
                rpg_speech::write_wav(&t.path, &audio)
                    .map(|()| SelfTestReport {
                        path: t.path.clone(),
                        requested_seconds: t.seconds,
                        samples: audio.samples.len(),
                        wall_seconds: t.started.elapsed().as_secs_f32(),
                        rms: level.rms,
                        peak: level.peak,
                        dropped_chunks: src.dropped_chunks().saturating_sub(t.dropped_at_start),
                        device: info,
                    })
                    .map_err(|e| format!("cannot write {}: {e}", t.path))
            }
        };
        self.send(MicEvent::SelfTest(result));
    }

    fn check_device(&mut self) {
        if let Some(src) = self.source.as_mut() {
            if let Some(e) = src.take_error() {
                self.source = None;
                if let Some(rec) = self.rec.take() {
                    let _ = rec;
                    self.tel.mic_listening.store(false, Ordering::Relaxed);
                }
                self.reopen_at = Some(Instant::now() + self.cfg.retry);
                self.send(MicEvent::DeviceError(e));
            }
        } else if self.reopen_at.is_some_and(|t| Instant::now() >= t) {
            self.open();
        }
    }

    fn check_auto_stop(&mut self) {
        let Some(rec) = self.rec.as_ref() else {
            return;
        };
        let too_long = rec.started.elapsed() >= self.cfg.max_record;
        let endpoint = self.cfg.auto_endpoint.is_some_and(|silence| {
            !rec.segments.is_empty()
                && !self.tel.vad_speaking.load(Ordering::Relaxed)
                && rec.last_speech.is_some_and(|t| t.elapsed() >= silence)
        });
        if too_long || endpoint {
            self.finish_recording();
        }
    }

    fn finish_recording(&mut self) {
        let Some(mut rec) = self.rec.take() else {
            return;
        };
        self.tel.mic_listening.store(false, Ordering::Relaxed);
        self.tel.vad_speaking.store(false, Ordering::Relaxed);
        if let Some(vad) = self.vad.as_mut() {
            rec.segments.extend(vad.flush());
            vad.reset();
        }
        let recorded_seconds = rec.audio.len() as f32 / ASR_RATE as f32;
        if rec.segments.is_empty() {
            let level = Level::of(&rec.audio);
            self.send(MicEvent::NoSpeech {
                conv: rec.conv,
                held_seconds: rec.started.elapsed().as_secs_f32(),
                recorded_seconds,
                rms: level.rms,
                peak: level.peak,
            });
            return;
        }
        // Join the speech segments with a short pause; silence-only stretches
        // are left out so the recognizer does not hallucinate over them.
        let gap = vec![0.0f32; ASR_RATE as usize / 8];
        let mut samples = Vec::new();
        for (i, seg) in rec.segments.iter().enumerate() {
            if i > 0 {
                samples.extend_from_slice(&gap);
            }
            samples.extend_from_slice(seg);
        }
        if let Some(path) = &self.cfg.dump_utterance {
            let _ = rpg_speech::write_wav(
                path,
                &SpeechAudio {
                    samples: samples.clone(),
                    sample_rate: ASR_RATE,
                    synth_ms: 0.0,
                    first_audio_ms: 0.0,
                },
            );
        }
        self.send(MicEvent::Captured {
            conv: rec.conv,
            speech_seconds: samples.len() as f32 / ASR_RATE as f32,
            samples,
            sample_rate: ASR_RATE,
            recorded_seconds,
        });
    }

    fn update_rate(&mut self) {
        let (since, at) = self.rate_window;
        let elapsed = since.elapsed();
        if elapsed >= Duration::from_secs(1) {
            let total = self.tel.mic_samples.load(Ordering::Relaxed);
            self.tel
                .mic_samples_per_second
                .set((total - at) as f32 / elapsed.as_secs_f32());
            self.rate_window = (Instant::now(), total);
        }
    }
}

fn next_device(devices: &[String], current: Option<&str>) -> Option<String> {
    if devices.is_empty() {
        return None;
    }
    let idx = current
        .and_then(|c| devices.iter().position(|d| d == c))
        .map(|i| (i + 1) % devices.len())
        .unwrap_or(0);
    devices.get(idx).cloned()
}

// ---------------------------------------------------------------------------
// Speech recognition
// ---------------------------------------------------------------------------

pub fn spawn_asr(
    factory: AsrFactory,
    rx: Receiver<AsrRequest>,
    out: Sender<Inbox>,
    tel: Arc<Telemetry>,
) -> JoinHandle<()> {
    spawn_named("irl-asr", move || {
        let mut asr = match factory() {
            Ok(a) => a,
            Err(e) => {
                let _ = out.send(Inbox::Asr(AsrEvent::Unavailable(e)));
                return;
            }
        };
        let _ = out.send(Inbox::Asr(AsrEvent::Ready {
            backend: asr.backend_name(),
        }));
        let mut queue: VecDeque<(ConvId, Vec<f32>, u32)> = VecDeque::new();
        loop {
            // Block for work only when there is none queued.
            let first = if queue.is_empty() {
                match rx.recv() {
                    Ok(r) => Some(r),
                    Err(_) => break,
                }
            } else {
                None
            };
            let mut shutdown = false;
            for req in first.into_iter().chain(rx.try_iter()) {
                match req {
                    AsrRequest::Transcribe {
                        conv,
                        samples,
                        sample_rate,
                    } => queue.push_back((conv, samples, sample_rate)),
                    AsrRequest::Cancel { conv } => queue.retain(|(c, ..)| *c != conv),
                    AsrRequest::Shutdown => shutdown = true,
                }
            }
            if shutdown {
                break;
            }
            tel.asr_queue.store(queue.len(), Ordering::Relaxed);
            let Some((conv, samples, rate)) = queue.pop_front() else {
                continue;
            };
            tel.asr_busy.store(true, Ordering::Relaxed);
            let start = Instant::now();
            let ev = match asr.transcribe(&samples, rate) {
                Ok(t) => AsrEvent::Transcript {
                    conv,
                    text: t.text,
                    latency_ms: start.elapsed().as_secs_f32() * 1000.0,
                },
                Err(rpg_domain::ports::RecognitionError::NoSpeech) => AsrEvent::Transcript {
                    conv,
                    text: String::new(),
                    latency_ms: start.elapsed().as_secs_f32() * 1000.0,
                },
                Err(e) => AsrEvent::Failed {
                    conv,
                    error: e.to_string(),
                },
            };
            tel.asr_busy.store(false, Ordering::Relaxed);
            tel.asr_queue.store(queue.len(), Ordering::Relaxed);
            if out.send(Inbox::Asr(ev)).is_err() {
                break;
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

/// Where the inference worker publishes its engine's interrupt flag once the
/// model has loaded. The worker cannot read its channel while the model runs,
/// so the runtime cancels the *running* request by raising this flag directly
/// — a lock-free store that never waits.
pub type InterruptSlot = Arc<OnceLock<Arc<AtomicBool>>>;

pub fn spawn_inference(
    factory: EngineFactory,
    interrupt_slot: InterruptSlot,
    rx: Receiver<InferRequest>,
    out: Sender<Inbox>,
    tel: Arc<Telemetry>,
) -> JoinHandle<()> {
    spawn_named("irl-inference", move || {
        let loaded = match factory() {
            Ok(l) => {
                let _ = out.send(Inbox::Infer(InferEvent::Ready {
                    engine: l.engine.engine_name(),
                    model_id: l.engine.model_id(),
                    is_real_model: l.engine.is_real_model(),
                }));
                Some(l)
            }
            Err(e) => {
                let _ = out.send(Inbox::Infer(InferEvent::LoadFailed(e)));
                None
            }
        };
        let (mut engine, interrupt) = match loaded {
            Some(l) => (Some(l.engine), l.interrupt),
            None => (None, None),
        };
        if let Some(flag) = &interrupt {
            let _ = interrupt_slot.set(flag.clone());
        }
        let mut queue: VecDeque<(InferJob, Instant)> = VecDeque::new();
        loop {
            let first = if queue.is_empty() {
                match rx.recv() {
                    Ok(r) => Some(r),
                    Err(_) => break,
                }
            } else {
                None
            };
            let mut shutdown = false;
            for req in first.into_iter().chain(rx.try_iter()) {
                match req {
                    InferRequest::Decide(job) => queue.push_back((job, Instant::now())),
                    // Cancels for the *running* request are delivered by the
                    // runtime through the interrupt flag; here we drop queued
                    // jobs so a cancelled turn never reaches the model.
                    InferRequest::Cancel { request_id } => {
                        queue.retain(|(j, _)| j.request_id != request_id)
                    }
                    InferRequest::Shutdown => shutdown = true,
                }
            }
            if shutdown {
                break;
            }
            tel.infer_queue.store(queue.len(), Ordering::Relaxed);
            let Some((job, queued_at)) = queue.pop_front() else {
                continue;
            };
            let queue_ms = queued_at.elapsed().as_secs_f32() * 1000.0;
            if let Some(flag) = &interrupt {
                flag.store(false, Ordering::SeqCst);
            }
            tel.active_inference.store(job.request_id, Ordering::SeqCst);
            let start = Instant::now();
            let result = match engine.as_mut() {
                Some(e) => e.complete(&job.request),
                None => Err(InferenceError::NotLoaded("model failed to load".into())),
            };
            tel.active_inference.store(NONE, Ordering::SeqCst);
            if let Some(flag) = &interrupt {
                flag.store(false, Ordering::SeqCst);
            }
            tel.infer_queue.store(queue.len(), Ordering::Relaxed);
            let done = InferDone {
                conv: job.conv,
                request_id: job.request_id,
                world_epoch: job.world_epoch,
                result,
                queue_ms,
                infer_ms: start.elapsed().as_secs_f32() * 1000.0,
            };
            if out.send(Inbox::Infer(InferEvent::Done(done))).is_err() {
                break;
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Speech synthesis + playback
// ---------------------------------------------------------------------------

/// Playback state shared between the TTS worker loop and the synthesis sink
/// closure. Both run on the TTS thread, so `Rc<RefCell<_>>` suffices and the
/// output device never leaves that thread.
struct Playback {
    sink: Box<dyn AudioSink>,
    /// Buffers the device queue could not take yet, retried every poll.
    backlog: VecDeque<(u64, Vec<f32>)>,
    resampler: LinearResampler,
}

impl Playback {
    fn submit(&mut self, epoch: u64, samples: Vec<f32>) {
        self.backlog.push_back((epoch, samples));
        self.flush_backlog();
    }

    fn flush_backlog(&mut self) {
        while let Some((epoch, buf)) = self.backlog.pop_front() {
            if let Err(buf) = self.sink.try_submit(epoch, buf) {
                self.backlog.push_front((epoch, buf));
                break;
            }
        }
    }

    fn backlog_samples(&self) -> u64 {
        self.backlog.iter().map(|(_, b)| b.len() as u64).sum()
    }
}

/// The line currently audible (or about to be).
struct Playing {
    job: SpeakJob,
    /// Done once the device has consumed this many samples.
    end_marker: u64,
    audio_seconds: f32,
    synth_ms: f32,
}

pub fn spawn_tts(
    factory: TtsFactory,
    epoch: Arc<AtomicU64>,
    rx: Receiver<TtsRequest>,
    out: Sender<Inbox>,
    tel: Arc<Telemetry>,
) -> JoinHandle<()> {
    spawn_named("irl-tts", move || {
        let (mut synth, sink) = match factory(epoch.clone()) {
            Ok(pair) => pair,
            Err(e) => {
                let _ = out.send(Inbox::Tts(TtsEvent::Unavailable(e)));
                return;
            }
        };
        let _ = out.send(Inbox::Tts(TtsEvent::Ready {
            backend: synth.backend_name(),
            device: sink.device_name(),
            sample_rate: sink.sample_rate(),
        }));
        let device_rate = sink.sample_rate();
        let voice_rate = synth.sample_rate();
        let pb = Rc::new(RefCell::new(Playback {
            sink,
            backlog: VecDeque::new(),
            resampler: LinearResampler::new(voice_rate, device_rate),
        }));
        let mut jobs: VecDeque<SpeakJob> = VecDeque::new();
        let mut playing: Option<Playing> = None;
        loop {
            match rx.recv_timeout(Duration::from_millis(5)) {
                Ok(TtsRequest::Speak(job)) => jobs.push_back(job),
                Ok(TtsRequest::Shutdown) | Err(RecvTimeoutError::Disconnected) => break,
                Err(RecvTimeoutError::Timeout) => {}
            }
            tel.tts_queue.store(jobs.len(), Ordering::Relaxed);
            let live = epoch.load(Ordering::Acquire);

            // Report on the line in flight.
            {
                let mut p = pb.borrow_mut();
                p.flush_backlog();
                if let Some(e) = p.sink.take_error() {
                    let _ = out.send(Inbox::Tts(TtsEvent::DeviceError(e)));
                }
                tel.playback_underruns
                    .store(p.sink.underruns(), Ordering::Relaxed);
                let pending = p.sink.queued() + p.backlog_samples();
                let left = pending.saturating_sub(p.sink.consumed());
                tel.playback_buffered_ms
                    .set(left as f32 * 1000.0 / device_rate.max(1) as f32);
                tel.playback_active.store(left > 0, Ordering::Relaxed);
                if let Some(cur) = playing.as_ref() {
                    let stale = cur.job.audio_epoch < live;
                    if stale {
                        p.backlog.clear();
                    }
                    if stale || p.sink.consumed() >= cur.end_marker {
                        let cur = playing.take().expect("checked above");
                        let ev = if stale {
                            TtsEvent::Cancelled {
                                conv: cur.job.conv,
                                line: cur.job.line,
                            }
                        } else {
                            TtsEvent::Finished {
                                conv: cur.job.conv,
                                line: cur.job.line,
                                audio_seconds: cur.audio_seconds,
                                synth_ms: cur.synth_ms,
                            }
                        };
                        let _ = out.send(Inbox::Tts(ev));
                    }
                }
            }
            // Lines are spoken one after another: start the next one only
            // when the previous has finished (or was superseded).
            if playing.is_some() {
                continue;
            }
            let Some(job) = jobs.pop_front() else {
                continue;
            };
            tel.tts_queue.store(jobs.len(), Ordering::Relaxed);
            if job.audio_epoch < epoch.load(Ordering::Acquire) {
                let _ = out.send(Inbox::Tts(TtsEvent::Cancelled {
                    conv: job.conv,
                    line: job.line,
                }));
                continue;
            }
            playing = speak(&mut *synth, &pb, &job, &epoch, &out, &tel);
        }
        // Silence anything still queued on the way out.
        epoch.fetch_add(1, Ordering::AcqRel);
        tel.playback_active.store(false, Ordering::Relaxed);
    })
}

/// Synthesize one line, streaming each chunk to the device as it appears.
fn speak(
    synth: &mut dyn Synthesizer,
    pb: &Rc<RefCell<Playback>>,
    job: &SpeakJob,
    epoch: &Arc<AtomicU64>,
    out: &Sender<Inbox>,
    tel: &Arc<Telemetry>,
) -> Option<Playing> {
    let start = Instant::now();
    {
        let mut p = pb.borrow_mut();
        p.resampler.reset();
        p.sink.set_streaming(true);
    }
    tel.tts_synthesizing.store(true, Ordering::Relaxed);
    let sink_pb = pb.clone();
    let sink_epoch = epoch.clone();
    let sink_out = out.clone();
    let (conv, line, my_epoch) = (job.conv, job.line, job.audio_epoch);
    let mut started = false;
    let sink = Box::new(move |chunk: &[f32]| -> bool {
        if sink_epoch.load(Ordering::Acquire) != my_epoch {
            return false;
        }
        let mut p = sink_pb.borrow_mut();
        let mut buf = Vec::new();
        p.resampler.process_into(chunk, &mut buf);
        p.submit(my_epoch, buf);
        if !started {
            started = true;
            let _ = sink_out.send(Inbox::Tts(TtsEvent::Started {
                conv,
                line,
                first_audio_ms: start.elapsed().as_secs_f32() * 1000.0,
            }));
        }
        true
    });
    let voice = synth.voice_for(&job.speaker);
    let result = synth.synthesize(&job.text, &voice, sink);
    tel.tts_synthesizing.store(false, Ordering::Relaxed);
    let mut p = pb.borrow_mut();
    p.sink.set_streaming(false);
    let superseded = epoch.load(Ordering::Acquire) != job.audio_epoch;
    match result {
        _ if superseded => {
            p.backlog.clear();
            let _ = out.send(Inbox::Tts(TtsEvent::Cancelled { conv, line }));
            None
        }
        Ok(stats) => {
            let end_marker = p.sink.queued() + p.backlog_samples();
            Some(Playing {
                job: job.clone(),
                end_marker,
                audio_seconds: stats.samples as f32 / synth.sample_rate().max(1) as f32,
                synth_ms: stats.synth_ms,
            })
        }
        Err(e) => {
            let _ = out.send(Inbox::Tts(TtsEvent::Failed {
                conv,
                line,
                error: e,
            }));
            None
        }
    }
}
