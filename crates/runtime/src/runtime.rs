//! Thread wiring and the fixed-tick simulation loop.
//!
//! ```text
//!   renderer ──Command──►┐                          ┌──► irl-mic        (cpal + VAD)
//!                        │   irl-sim                ├──► irl-asr        (Whisper)
//!   irl-mic  ──event──►  ├─► inbox ─► SimCore ─effects┼──► irl-inference  (llama.cpp)
//!   irl-asr  ──event──►  │   fixed tick schedule    └──► irl-tts        (Kokoro + speaker)
//!   irl-infer ─event──►  │        │
//!   irl-tts  ──event──►  ┘        └──Snapshot──► renderer
//! ```
//!
//! The simulation thread waits only on its own inbox, with a timeout equal to
//! the time left until the next tick. Every send it performs is on an
//! unbounded channel, every telemetry access is an atomic, and the one
//! cross-thread interrupt (cancelling the running model request) is an atomic
//! store. So no microphone, VAD, ASR, model, TTS or playback operation can
//! delay a tick.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, sync_channel, Receiver, RecvTimeoutError, Sender, SyncSender};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use rpg_eval::Session;

use crate::backends::Backends;
use crate::core::{CoreConfig, SimCore};
use crate::protocol::*;
use crate::telemetry::Telemetry;
use crate::workers::{self, InterruptSlot, MicConfig};

/// Snapshots buffered for the renderer before new ones are dropped.
pub const SNAPSHOT_BACKLOG: usize = 64;

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub seed: u64,
    /// Fixed simulation step.
    pub tick_period: Duration,
    /// Minimum spacing of snapshots published between ticks (voice-state
    /// changes are pushed promptly, but never more often than this).
    pub min_publish_interval: Duration,
    pub core: CoreConfig,
    pub mic: MicConfig,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            seed: 2026,
            tick_period: Duration::from_millis(120),
            min_publish_interval: Duration::from_millis(30),
            core: CoreConfig::default(),
            mic: MicConfig::default(),
        }
    }
}

/// The renderer's handle on a running runtime.
pub struct RuntimeHandle {
    inbox: Sender<Inbox>,
    pub snapshots: Receiver<Snapshot>,
    pub telemetry: Arc<Telemetry>,
    /// The playback epoch shared with the output device.
    pub audio_epoch: Arc<AtomicU64>,
    sim: Option<JoinHandle<()>>,
    workers: Vec<(&'static str, JoinHandle<()>)>,
}

/// What happened during shutdown.
#[derive(Debug, Default)]
pub struct ShutdownReport {
    pub joined: Vec<&'static str>,
    /// Threads still busy after the timeout (e.g. a model mid-completion).
    /// They hold no shared state and are detached.
    pub detached: Vec<&'static str>,
}

impl RuntimeHandle {
    /// Send a command. Never blocks.
    pub fn send(&self, cmd: Command) {
        let _ = self.inbox.send(Inbox::Ui(cmd));
    }

    /// A sender for injecting raw inbox messages (tests and tools).
    pub fn inbox(&self) -> Sender<Inbox> {
        self.inbox.clone()
    }

    /// Ask everything to stop and wait up to `timeout` for the threads.
    pub fn shutdown(mut self, timeout: Duration) -> ShutdownReport {
        self.send(Command::Quit);
        let deadline = Instant::now() + timeout;
        let mut report = ShutdownReport::default();
        let mut pending: Vec<(&'static str, JoinHandle<()>)> = Vec::new();
        if let Some(sim) = self.sim.take() {
            pending.push(("irl-sim", sim));
        }
        pending.append(&mut self.workers);
        while !pending.is_empty() {
            let (done, busy): (Vec<_>, Vec<_>) =
                pending.into_iter().partition(|(_, h)| h.is_finished());
            for (name, h) in done {
                let _ = h.join();
                report.joined.push(name);
            }
            pending = busy;
            if pending.is_empty() || Instant::now() >= deadline {
                break;
            }
            thread::sleep(Duration::from_millis(2));
        }
        report.detached = pending.into_iter().map(|(n, _)| n).collect();
        report
    }
}

struct WorkerLinks {
    mic: Option<Sender<MicRequest>>,
    asr: Option<Sender<AsrRequest>>,
    infer: Sender<InferRequest>,
    tts: Option<Sender<TtsRequest>>,
    interrupt: InterruptSlot,
    audio_epoch: Arc<AtomicU64>,
    telemetry: Arc<Telemetry>,
}

impl WorkerLinks {
    /// Hand effects to their workers. A dead or absent worker turns the
    /// request into an immediate failure event, so the conversation can
    /// never wait forever on a component that is gone.
    fn dispatch(&self, effects: Vec<Effect>, inbox: &Sender<Inbox>) {
        for effect in effects {
            match effect {
                Effect::SetAudioEpoch(e) => self.audio_epoch.store(e, Ordering::Release),
                // Native saves go through `Command::Save(path)`; export to a
                // platform store is only requested by the browser client.
                Effect::SaveData { .. } => {}
                Effect::Mic(req) => {
                    let conv = match &req {
                        MicRequest::Start { conv } | MicRequest::Stop { conv } => Some(*conv),
                        _ => None,
                    };
                    let sent = self.mic.as_ref().is_some_and(|tx| tx.send(req).is_ok());
                    if !sent && conv.is_some() {
                        let _ = inbox.send(Inbox::Mic(MicEvent::DeviceError(
                            "microphone worker is not running".into(),
                        )));
                    }
                }
                Effect::Asr(req) => {
                    let conv = match &req {
                        AsrRequest::Transcribe { conv, .. } => Some(*conv),
                        _ => None,
                    };
                    let sent = self.asr.as_ref().is_some_and(|tx| tx.send(req).is_ok());
                    if let (false, Some(conv)) = (sent, conv) {
                        let _ = inbox.send(Inbox::Asr(AsrEvent::Failed {
                            conv,
                            error: "speech recognizer is not running".into(),
                        }));
                    }
                }
                Effect::Infer(req) => {
                    if let InferRequest::Cancel { request_id } = &req {
                        if self.telemetry.active_inference.load(Ordering::SeqCst) == *request_id {
                            if let Some(flag) = self.interrupt.get() {
                                flag.store(true, Ordering::SeqCst);
                            }
                        }
                    }
                    if let Err(e) = self.infer.send(req) {
                        if let InferRequest::Decide(job) = e.0 {
                            let _ = inbox.send(Inbox::Infer(InferEvent::Done(InferDone {
                                conv: job.conv,
                                request_id: job.request_id,
                                world_epoch: job.world_epoch,
                                result: Err(rpg_domain::ports::InferenceError::NotLoaded(
                                    "inference worker is not running".into(),
                                )),
                                queue_ms: 0.0,
                                infer_ms: 0.0,
                            })));
                        }
                    }
                }
                Effect::Tts(req) => {
                    let line = match &req {
                        TtsRequest::Speak(j) => Some((j.conv, j.line)),
                        _ => None,
                    };
                    let sent = self.tts.as_ref().is_some_and(|tx| tx.send(req).is_ok());
                    if let (false, Some((conv, line))) = (sent, line) {
                        let _ = inbox.send(Inbox::Tts(TtsEvent::Failed {
                            conv,
                            line,
                            error: "voice worker is not running".into(),
                        }));
                    }
                }
            }
        }
    }

    fn shutdown(&self) {
        if let Some(tx) = &self.mic {
            let _ = tx.send(MicRequest::Shutdown);
        }
        if let Some(tx) = &self.asr {
            let _ = tx.send(AsrRequest::Shutdown);
        }
        let _ = self.infer.send(InferRequest::Shutdown);
        if let Some(flag) = self.interrupt.get() {
            flag.store(true, Ordering::SeqCst);
        }
        if let Some(tx) = &self.tts {
            let _ = tx.send(TtsRequest::Shutdown);
        }
        self.audio_epoch.fetch_add(1, Ordering::AcqRel);
    }
}

/// Start the simulation and every worker. Returns immediately: the model and
/// speech engines load on their own threads while the world already runs.
pub fn spawn(cfg: RuntimeConfig, session: Session, backends: Backends) -> RuntimeHandle {
    let telemetry = Arc::new(Telemetry::new());
    let audio_epoch = Arc::new(AtomicU64::new(0));
    let (inbox_tx, inbox_rx) = channel::<Inbox>();
    // Bounded: a renderer that stops reading (minimised, stalled) costs at
    // most this many snapshots of memory, and publishing never blocks.
    let (snap_tx, snap_rx) = sync_channel::<Snapshot>(SNAPSHOT_BACKLOG);
    let mut threads = Vec::new();

    let interrupt: InterruptSlot = Arc::new(OnceLock::new());
    let (infer_tx, infer_rx) = channel();
    threads.push((
        "irl-inference",
        workers::spawn_inference(
            backends.engine,
            interrupt.clone(),
            infer_rx,
            inbox_tx.clone(),
            telemetry.clone(),
        ),
    ));

    let mic = backends.mic.map(|(backend, vad)| {
        let (tx, rx) = channel();
        threads.push((
            "irl-mic",
            workers::spawn_mic(
                cfg.mic.clone(),
                backend,
                vad,
                rx,
                inbox_tx.clone(),
                telemetry.clone(),
            ),
        ));
        tx
    });
    let asr = backends.asr.map(|factory| {
        let (tx, rx) = channel();
        threads.push((
            "irl-asr",
            workers::spawn_asr(factory, rx, inbox_tx.clone(), telemetry.clone()),
        ));
        tx
    });
    let tts = backends.tts.map(|factory| {
        let (tx, rx) = channel();
        threads.push((
            "irl-tts",
            workers::spawn_tts(
                factory,
                audio_epoch.clone(),
                rx,
                inbox_tx.clone(),
                telemetry.clone(),
            ),
        ));
        tx
    });

    let mut core = SimCore::new(session, cfg.core.clone(), telemetry.clone());
    core.mark_unavailable(mic.is_none(), asr.is_none(), tts.is_none(), "disabled");
    let links = WorkerLinks {
        mic,
        asr,
        infer: infer_tx,
        tts,
        interrupt,
        audio_epoch: audio_epoch.clone(),
        telemetry: telemetry.clone(),
    };
    let sim_inbox = inbox_tx.clone();
    let sim = thread::Builder::new()
        .name("irl-sim".into())
        .spawn(move || run_sim(cfg, core, links, inbox_rx, sim_inbox, snap_tx))
        .expect("spawn simulation thread");

    RuntimeHandle {
        inbox: inbox_tx,
        snapshots: snap_rx,
        telemetry,
        audio_epoch,
        sim: Some(sim),
        workers: threads,
    }
}

/// Rolling tick-rate measurement.
struct TickMeter {
    window: std::collections::VecDeque<Instant>,
    max_lag: Duration,
}

impl TickMeter {
    fn record(&mut self, at: Instant, lag: Duration, tel: &Telemetry) {
        self.window.push_back(at);
        while self
            .window
            .front()
            .is_some_and(|t| at.duration_since(*t) > Duration::from_secs(2))
        {
            self.window.pop_front();
        }
        if let (Some(first), n) = (self.window.front(), self.window.len()) {
            let span = at.duration_since(*first).as_secs_f32();
            if n > 1 && span > 0.0 {
                tel.ticks_per_second.set((n - 1) as f32 / span);
            }
        }
        self.max_lag = self.max_lag.max(lag);
        tel.tick_delay_ms.set(lag.as_secs_f32() * 1000.0);
        tel.max_tick_lag_ms.set(self.max_lag.as_secs_f32() * 1000.0);
        tel.ticks.fetch_add(1, Ordering::Relaxed);
    }
}

fn run_sim(
    cfg: RuntimeConfig,
    mut core: SimCore,
    links: WorkerLinks,
    inbox: Receiver<Inbox>,
    self_tx: Sender<Inbox>,
    snapshots: SyncSender<Snapshot>,
) {
    let period = cfg.tick_period.max(Duration::from_millis(1));
    let target_tps = 1.0 / period.as_secs_f32();
    let start = Instant::now();
    let mut next_tick = start + period;
    let mut meter = TickMeter {
        window: Default::default(),
        max_lag: Duration::ZERO,
    };
    let publish = |core: &SimCore| {
        let mut snap = core.snapshot();
        snap.diagnostics.target_ticks_per_second = target_tps;
        snap.diagnostics.platform = std::env::consts::OS.to_string();
        if snapshots.try_send(snap).is_err() {
            links
                .telemetry
                .dropped_snapshots
                .fetch_add(1, Ordering::Relaxed);
        }
    };
    publish(&core);
    let mut last_publish = Instant::now();
    let mut dirty = false;

    loop {
        let now = Instant::now();
        if now >= next_tick {
            let lag = now - next_tick;
            core.tick(now - start);
            meter.record(now, lag, &links.telemetry);
            links.dispatch(core.take_effects(), &self_tx);
            next_tick += period;
            // After a long stall (debugger, suspended laptop) resume the
            // schedule instead of replaying a burst of catch-up ticks.
            if now > next_tick + period * 4 {
                let behind = ((now - next_tick).as_secs_f64() / period.as_secs_f64()) as u64;
                links
                    .telemetry
                    .skipped_ticks
                    .fetch_add(behind, Ordering::Relaxed);
                next_tick = now + period;
            }
            publish(&core);
            last_publish = Instant::now();
            dirty = false;
            continue;
        }
        match inbox.recv_timeout(next_tick - now) {
            Ok(msg) => {
                let now = Instant::now();
                core.handle(msg, now - start);
                links.dispatch(core.take_effects(), &self_tx);
                dirty = true;
                if core.quit_requested() {
                    break;
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
        if dirty && last_publish.elapsed() >= cfg.min_publish_interval {
            publish(&core);
            last_publish = Instant::now();
            dirty = false;
        }
    }
    links.shutdown();
    publish(&core);
}
