//! End-to-end tests of the threaded runtime with controllable fake devices
//! and models.
//!
//! Each test holds the pipeline in one voice stage with a gate and then
//! *waits for the simulation tick to advance* while asserting the stage did
//! not change. Waits are condition-based with generous ceilings, never fixed
//! sleeps, so the tests are not timing-sensitive; the measured tick rates are
//! printed (run with `--nocapture`) for the record.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rpg_domain::geom::Vec2;
use rpg_eval::Session;
use rpg_runtime::backends::Backends;
use rpg_runtime::testing::*;
use rpg_runtime::*;

const TICK_MS: u64 = 10;
const LONG: Duration = Duration::from_secs(20);

struct Opts {
    mic_rate: u32,
    mic_channels: u16,
    engine: FakeEngine,
    asr: FakeAsr,
    synth: FakeSynth,
    instant_playback: bool,
    tick_ms: u64,
    core: CoreConfig,
    mic: MicConfig,
    pre_ticks: u32,
    near: &'static str,
    vad: EnergyVad,
}

impl Default for Opts {
    fn default() -> Self {
        Self {
            mic_rate: 48_000,
            mic_channels: 2,
            engine: FakeEngine::new(),
            asr: FakeAsr::new("Hello there, father."),
            synth: FakeSynth::new(),
            instant_playback: true,
            tick_ms: TICK_MS,
            core: CoreConfig::default(),
            mic: MicConfig::default(),
            pre_ticks: 0,
            near: "priest",
            vad: EnergyVad::new(),
        }
    }
}

struct Harness {
    rt: Option<RuntimeHandle>,
    latest: Snapshot,
    mic: FakeMicHandle,
    sink: Arc<Mutex<Option<FakeSinkHandle>>>,
    asr_received: Arc<Mutex<Vec<(usize, u32, f32)>>>,
    engine_seen: Arc<Mutex<Vec<u64>>>,
    spoken: Arc<Mutex<Vec<String>>>,
    vad_resets: Arc<std::sync::atomic::AtomicUsize>,
    tick_ms: u64,
}

impl Harness {
    fn new(o: Opts) -> Self {
        let mut session = Session::new(2026);
        session.ticks(o.pre_ticks);
        session.teleport_player_to(o.near);
        let (mic_backend, mic) = FakeMicBackend::new(o.mic_rate, o.mic_channels);
        let asr_received = o.asr.received.clone();
        let engine_seen = o.engine.seen.clone();
        let spoken = o.synth.spoken.clone();
        let vad_resets = o.vad.resets.clone();
        let vad = o.vad;
        let (tts, sink) = fake_tts(o.synth, 48_000, o.instant_playback);
        let backends = Backends {
            engine: o.engine.factory(),
            mic: Some((
                Box::new(mic_backend),
                Box::new(move || Ok(Box::new(vad) as Box<dyn rpg_runtime::backends::Vad>)),
            )),
            asr: Some(o.asr.factory()),
            tts: Some(tts),
        };
        let cfg = RuntimeConfig {
            tick_period: Duration::from_millis(o.tick_ms),
            core: o.core,
            mic: o.mic,
            ..Default::default()
        };
        let rt = spawn(cfg, session, backends);
        let first = rt.snapshots.recv_timeout(LONG).expect("first snapshot");
        let mut h = Self {
            rt: Some(rt),
            latest: first,
            mic,
            sink,
            asr_received,
            engine_seen,
            spoken,
            vad_resets,
            tick_ms: o.tick_ms,
        };
        h.wait("voice ready", |s| {
            s.voice.mic.is_ready() && s.voice.asr.is_ready() && s.voice.tts.is_ready()
        });
        h.wait("engine ready", |s| s.engine_name == "fake-engine");
        h
    }

    fn rt(&self) -> &RuntimeHandle {
        self.rt.as_ref().unwrap()
    }

    fn send(&self, cmd: Command) {
        self.rt().send(cmd);
    }

    /// Drain snapshots until `pred` holds (or panic after a long ceiling).
    fn wait(&mut self, what: &str, pred: impl Fn(&Snapshot) -> bool) -> Snapshot {
        self.wait_observing(what, pred, |_| {})
    }

    fn wait_observing(
        &mut self,
        what: &str,
        pred: impl Fn(&Snapshot) -> bool,
        mut each: impl FnMut(&Snapshot),
    ) -> Snapshot {
        let deadline = Instant::now() + LONG;
        loop {
            if pred(&self.latest) {
                return self.latest.clone();
            }
            let left = deadline.saturating_duration_since(Instant::now());
            match self.rt().snapshots.recv_timeout(left) {
                Ok(s) => {
                    each(&s);
                    self.latest = s;
                }
                Err(_) => panic!(
                    "timed out waiting for {what}; stage {:?}, tick {}, status '{}', last lines {:?}",
                    self.latest.voice.stage,
                    self.latest.tick,
                    self.latest.status,
                    self.latest.dialogue.iter().rev().take(3).collect::<Vec<_>>()
                ),
            }
        }
    }

    fn stage(&mut self, stage: VoiceStage) -> Snapshot {
        self.wait(&format!("stage {stage:?}"), |s| s.voice.stage == stage)
    }

    /// The core invariant: while the pipeline is held in `stage`, the world
    /// keeps ticking at (close to) its fixed rate.
    fn assert_ticks_while(&mut self, stage: VoiceStage, n: u64) -> TickReport {
        let start = self.stage(stage);
        let t0 = Instant::now();
        let mut left_stage = None;
        let end = self.wait_observing(
            &format!("{n} ticks during {stage:?}"),
            |s| s.tick >= start.tick + n,
            |s| {
                if s.voice.stage != stage && left_stage.is_none() {
                    left_stage = Some(s.voice.stage);
                }
            },
        );
        assert_eq!(left_stage, None, "the pipeline left {stage:?} while held");
        let elapsed = t0.elapsed();
        let ticks = end.tick - start.tick;
        let tps = ticks as f32 / elapsed.as_secs_f32();
        let target = 1000.0 / self.tick_ms as f32;
        let report = TickReport {
            stage,
            ticks,
            elapsed,
            tps,
            target,
            max_lag_ms: end.diagnostics.max_tick_lag_ms,
        };
        eprintln!("{report}");
        assert!(
            tps >= target * 0.5,
            "simulation slowed during {stage:?}: {tps:.1} tps vs target {target:.1}"
        );
        report
    }

    fn sink(&self) -> FakeSinkHandle {
        self.sink.lock().unwrap().clone().expect("tts started")
    }

    /// Hold push-to-talk, feed `mono` (device-rate, mono content) as the
    /// device would deliver it, then release.
    fn say_into_mic(&mut self, mono: &[f32]) {
        self.send(Command::PushToTalk(true));
        self.stage(VoiceStage::Listening);
        self.wait_mic_listening();
        self.feed(mono);
        self.send(Command::PushToTalk(false));
    }

    fn wait_mic_listening(&mut self) {
        let tel = self.rt().telemetry.clone();
        let deadline = Instant::now() + LONG;
        while !tel.mic_listening.load(Ordering::SeqCst) {
            assert!(Instant::now() < deadline, "mic never started listening");
            std::thread::yield_now();
        }
    }

    fn feed(&self, mono: &[f32]) {
        let info = self.mic.info.lock().unwrap().clone();
        // 10 ms device buffers, interleaved like a real stream.
        let frames = (info.sample_rate / 100) as usize;
        for chunk in mono.chunks(frames) {
            self.mic.push_interleaved(&interleave(chunk, info.channels));
        }
    }

    fn speech(&self, secs: f32) -> Vec<f32> {
        let rate = self.mic.info.lock().unwrap().sample_rate;
        let mut s = tone(rate, 220.0, secs, 0.3);
        s.extend(std::iter::repeat_n(0.0, rate as usize / 2));
        s
    }

    fn npc_lines(&self) -> Vec<String> {
        self.latest
            .dialogue
            .iter()
            .filter(|l| l.kind == LineKind::Npc)
            .map(|l| l.text.clone())
            .collect()
    }

    fn shutdown(mut self) -> ShutdownReport {
        self.rt.take().unwrap().shutdown(Duration::from_secs(5))
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        if let Some(rt) = self.rt.take() {
            let _ = rt.shutdown(Duration::from_secs(2));
        }
    }
}

struct TickReport {
    stage: VoiceStage,
    ticks: u64,
    elapsed: Duration,
    tps: f32,
    target: f32,
    max_lag_ms: f32,
}

impl std::fmt::Display for TickReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[tick-measure] {:<13} {:>4} ticks in {:>6.0} ms = {:>6.1} tps (target {:.0}), max lag {:.1} ms",
            format!("{:?}", self.stage),
            self.ticks,
            self.elapsed.as_secs_f32() * 1000.0,
            self.tps,
            self.target,
            self.max_lag_ms
        )
    }
}

fn gated(gate: &Gate) -> Option<Gate> {
    Some(gate.clone())
}

// ---------------------------------------------------------------------------
// The world keeps ticking at every voice stage
// ---------------------------------------------------------------------------

#[test]
fn ticks_continue_while_recording_silence() {
    let mut h = Harness::new(Opts::default());
    h.send(Command::PushToTalk(true));
    h.wait_mic_listening();
    let rate = h.mic.info.lock().unwrap().sample_rate;
    // A live but silent device: more audio than the hold will last.
    let silence = vec![0.0f32; rate as usize * 2];
    h.feed(&silence);
    h.assert_ticks_while(VoiceStage::Listening, 50);
    h.send(Command::PushToTalk(false));
    let s = h.stage(VoiceStage::Idle);
    assert!(s.dialogue.last().unwrap().text.contains("no audio"));
    assert!(h.asr_received.lock().unwrap().is_empty());
}

#[test]
fn ticks_continue_while_recording_speech() {
    let mut h = Harness::new(Opts::default());
    h.send(Command::PushToTalk(true));
    h.wait_mic_listening();
    let speech = h.speech(1.0);
    h.feed(&speech);
    h.assert_ticks_while(VoiceStage::Listening, 50);
    h.send(Command::PushToTalk(false));
    h.stage(VoiceStage::Idle);
    assert_eq!(h.asr_received.lock().unwrap().len(), 1);
    assert_eq!(h.npc_lines(), vec!["Good day to you."]);
}

#[test]
fn ticks_continue_while_asr_is_processing() {
    let gate = Gate::new();
    let mut asr = FakeAsr::new("Hello.");
    asr.gate = gated(&gate);
    let mut h = Harness::new(Opts {
        asr,
        ..Default::default()
    });
    let speech = h.speech(0.6);
    h.say_into_mic(&speech);
    h.assert_ticks_while(VoiceStage::Transcribing, 50);
    gate.open();
    h.stage(VoiceStage::Idle);
    assert_eq!(h.npc_lines().len(), 1);
}

#[test]
fn ticks_and_npc_movement_continue_while_the_model_is_stalled() {
    let gate = Gate::new();
    // NPC schedules start moving around tick 220 of this seed.
    let mut h = Harness::new(Opts {
        engine: FakeEngine::new().with_gate(gate.clone()),
        pre_ticks: 200,
        ..Default::default()
    });
    let before = h.latest.clone();
    h.send(Command::Say("Tell me about the town.".into()));
    h.stage(VoiceStage::Deciding);

    // The player can still walk while the NPC thinks.
    let target = Vec2::new(before.player_pos.x + 2.0, before.player_pos.y);
    h.send(Command::MoveTo(target));
    h.assert_ticks_while(VoiceStage::Deciding, 80);
    let during = h.latest.clone();
    let moved: Vec<&str> = during
        .actors
        .iter()
        .filter(|a| {
            before
                .actor_pos(&a.id)
                .is_some_and(|p| p.distance(a.pos) > 0.01)
        })
        .map(|a| a.id.as_str())
        .collect();
    assert!(
        moved.iter().any(|id| *id != "player"),
        "no NPC moved while the model was stalled"
    );
    assert!(
        during.player_pos.distance(before.player_pos) > 0.5,
        "the player could not move during inference"
    );
    assert_eq!(during.voice.stage, VoiceStage::Deciding);
    assert!(during.diagnostics.active_inference.is_some());

    gate.open();
    h.stage(VoiceStage::Idle);
    assert_eq!(h.npc_lines(), vec!["Good day to you."]);
}

#[test]
fn ticks_continue_while_tts_is_synthesizing() {
    let gate = Gate::new();
    let mut synth = FakeSynth::new();
    synth.gate = gated(&gate);
    let mut h = Harness::new(Opts {
        synth,
        ..Default::default()
    });
    h.send(Command::Say("Hello.".into()));
    h.assert_ticks_while(VoiceStage::Synthesizing, 50);
    gate.open();
    h.stage(VoiceStage::Idle);
}

#[test]
fn ticks_continue_while_audio_is_playing() {
    let mut h = Harness::new(Opts {
        instant_playback: false,
        ..Default::default()
    });
    h.send(Command::Say("Hello.".into()));
    h.assert_ticks_while(VoiceStage::Speaking, 50);
    assert!(h.sink().pending() > 0, "audio should still be queued");
    h.sink().play_all();
    h.stage(VoiceStage::Idle);
    assert!(h.sink().heard_samples() > 0);
}

#[test]
fn cancelling_a_turn_discards_the_late_answer_and_its_audio() {
    let gate = Gate::new();
    let mut h = Harness::new(Opts {
        engine: FakeEngine::new().with_gate(gate.clone()),
        ..Default::default()
    });
    h.send(Command::Say("Hello.".into()));
    h.stage(VoiceStage::Deciding);
    h.send(Command::CancelVoice);
    h.stage(VoiceStage::Idle);
    // The interrupt reached the running request; open the gate anyway to
    // simulate a model that answers regardless.
    gate.open();
    h.assert_ticks_while(VoiceStage::Idle, 30);
    assert!(h.npc_lines().is_empty());
    assert!(h.spoken.lock().unwrap().is_empty());
    assert_eq!(h.sink().heard_samples(), 0);
    // A new turn works normally afterwards.
    h.send(Command::Say("Hello again.".into()));
    h.wait("second reply", |s| {
        s.dialogue.iter().any(|l| l.kind == LineKind::Npc)
    });
}

#[test]
fn model_timeout_falls_back_without_stopping_the_world() {
    let gate = Gate::new();
    let mut h = Harness::new(Opts {
        engine: FakeEngine::new().with_gate(gate.clone()),
        core: CoreConfig {
            decide_timeout: Duration::from_millis(400),
            ..Default::default()
        },
        ..Default::default()
    });
    h.send(Command::Say("Hello.".into()));
    h.assert_ticks_while(VoiceStage::Deciding, 15);
    let s = h.wait("fallback answer", |s| s.fallback_decisions == 1);
    assert_eq!(s.diagnostics.timeouts, 1);
    h.stage(VoiceStage::Idle);
    gate.open();
}

#[test]
fn device_error_is_reported_and_recovered_while_ticking() {
    let mut h = Harness::new(Opts {
        mic: MicConfig {
            retry: Duration::from_millis(100),
            ..Default::default()
        },
        ..Default::default()
    });
    h.send(Command::PushToTalk(true));
    h.wait_mic_listening();
    h.mic.fail("device unplugged");
    let s = h.wait("mic error", |s| !s.voice.mic.is_ready());
    assert_eq!(s.voice.stage, VoiceStage::Idle);
    h.assert_ticks_while(VoiceStage::Idle, 20);
    h.wait("mic recovered", |s| s.voice.mic.is_ready());
    assert!(h.mic.opens.load(Ordering::SeqCst) >= 2);
    // And it records again.
    let speech = h.speech(0.6);
    h.say_into_mic(&speech);
    h.wait("reply after recovery", |s| {
        s.dialogue.iter().any(|l| l.kind == LineKind::Npc)
    });
}

#[test]
fn a_ten_second_inference_does_not_stop_ticks_or_npcs() {
    let mut h = Harness::new(Opts {
        engine: FakeEngine::new().with_delay(Duration::from_secs(10)),
        tick_ms: 50,
        pre_ticks: 200,
        ..Default::default()
    });
    let before = h.latest.clone();
    h.send(Command::Say("Take your time.".into()));
    let start = h.stage(VoiceStage::Deciding);
    let t0 = Instant::now();
    let mut min_tps = f32::INFINITY;
    let end = h.wait_observing(
        "the slow answer",
        |s| s.voice.stage != VoiceStage::Deciding,
        |s| {
            if s.diagnostics.ticks_per_second > 0.0 && t0.elapsed() > Duration::from_secs(3) {
                min_tps = min_tps.min(s.diagnostics.ticks_per_second);
            }
        },
    );
    let elapsed = t0.elapsed();
    let ticks = end.tick - start.tick;
    let expected = elapsed.as_secs_f32() * 20.0;
    eprintln!(
        "[tick-measure] 10s inference: {ticks} ticks in {:.2}s (expected ~{expected:.0}), min rolling tps {min_tps:.1}, max lag {:.1} ms",
        elapsed.as_secs_f32(),
        end.diagnostics.max_tick_lag_ms
    );
    assert!(elapsed >= Duration::from_secs(9), "the model was not slow");
    assert!(
        ticks as f32 >= expected * 0.8,
        "only {ticks} ticks during a {elapsed:?} inference"
    );
    let moved = end
        .actors
        .iter()
        .filter(|a| a.kind == KindView::Npc)
        .filter(|a| {
            before
                .actor_pos(&a.id)
                .is_some_and(|p| p.distance(a.pos) > 0.5)
        })
        .count();
    assert!(moved >= 2, "only {moved} NPCs moved during the inference");
}

// ---------------------------------------------------------------------------
// Microphone recording correctness
// ---------------------------------------------------------------------------

/// The capture path must hand ASR 16 kHz mono of the right duration for
/// mono and stereo devices at common rates. (The original bug downmixed
/// stereo twice, halving the duration and doubling the pitch.)
#[test]
fn mono_and_stereo_devices_deliver_correct_16k_audio() {
    for (rate, channels) in [
        (48_000, 2),
        (48_000, 1),
        (44_100, 2),
        (44_100, 1),
        (16_000, 1),
    ] {
        let mut h = Harness::new(Opts {
            mic_rate: rate,
            mic_channels: channels,
            ..Default::default()
        });
        let speech = h.speech(1.0);
        h.say_into_mic(&speech);
        h.stage(VoiceStage::Idle);
        let got = h.asr_received.lock().unwrap().clone();
        assert_eq!(got.len(), 1, "{rate}/{channels}: {got:?}");
        let (samples, sr, rms) = got[0];
        assert_eq!(sr, 16_000);
        // 1.0 s of speech plus the VAD's 0.3 s hangover.
        let secs = samples as f32 / 16_000.0;
        assert!(
            (secs - 1.3).abs() < 0.05,
            "{rate} Hz x{channels}: ASR got {secs:.3}s of audio"
        );
        assert!(rms > 0.15, "{rate} Hz x{channels}: level lost (rms {rms})");
    }
}

#[test]
fn repeated_recordings_all_work() {
    let mut h = Harness::new(Opts::default());
    for i in 0..5 {
        let speech = h.speech(0.5 + 0.1 * i as f32);
        h.say_into_mic(&speech);
        h.wait(&format!("reply {i}"), |s| {
            s.voice.stage == VoiceStage::Idle
                && s.dialogue
                    .iter()
                    .filter(|l| l.kind == LineKind::Npc)
                    .count()
                    == i + 1
        });
    }
    let got = h.asr_received.lock().unwrap().clone();
    assert_eq!(got.len(), 5);
    for (i, (samples, _, _)) in got.iter().enumerate() {
        let expect = 0.5 + 0.1 * i as f32 + 0.3;
        let secs = *samples as f32 / 16_000.0;
        assert!((secs - expect).abs() < 0.05, "turn {i}: {secs} vs {expect}");
    }
}

#[test]
fn an_empty_microphone_is_diagnosed_not_hung() {
    let mut h = Harness::new(Opts::default());
    h.send(Command::PushToTalk(true));
    h.wait_mic_listening();
    // Hold for a while; the device delivers nothing at all.
    h.assert_ticks_while(VoiceStage::Listening, 40);
    h.send(Command::PushToTalk(false));
    let s = h.stage(VoiceStage::Idle);
    assert!(
        s.dialogue
            .last()
            .unwrap()
            .text
            .contains("capture is failing"),
        "{:?}",
        s.dialogue.last()
    );
}

#[test]
fn a_quick_tap_of_v_is_explained_not_misdiagnosed() {
    let mut h = Harness::new(Opts::default());
    h.send(Command::PushToTalk(true));
    h.send(Command::PushToTalk(false));
    let s = h.wait("tap explained", |s| {
        s.dialogue.iter().any(|l| l.text.contains("too short"))
    });
    assert_eq!(s.voice.stage, VoiceStage::Idle);
}

#[test]
fn a_renderer_that_stops_reading_cannot_stall_or_bloat_the_simulation() {
    let h = Harness::new(Opts::default());
    let tel = h.rt().telemetry.clone();
    let before = tel.ticks.load(Ordering::SeqCst);
    // Deliberately do not read snapshots for a while.
    std::thread::sleep(Duration::from_millis(1500));
    let ticks = tel.ticks.load(Ordering::SeqCst) - before;
    let dropped = tel.dropped_snapshots.load(Ordering::SeqCst);
    eprintln!("[tick-measure] renderer stalled 1.5s: {ticks} ticks, {dropped} snapshots dropped");
    assert!(
        ticks >= 100,
        "only {ticks} ticks while the renderer stalled"
    );
    assert!(dropped > 0, "the snapshot queue should be bounded");
}

#[test]
fn audio_from_before_the_key_press_is_never_recorded() {
    let mut h = Harness::new(Opts::default());
    // Loud audio arrives before the player presses V…
    let stale = h.speech(2.0);
    h.feed(&stale);
    let resets_before = h.vad_resets.load(Ordering::SeqCst);
    // …then a short real utterance.
    let speech = h.speech(0.5);
    h.say_into_mic(&speech);
    h.stage(VoiceStage::Idle);
    let got = h.asr_received.lock().unwrap().clone();
    let secs = got[0].0 as f32 / 16_000.0;
    assert!(
        (secs - 0.8).abs() < 0.05,
        "stale audio leaked into the recording: {secs:.3}s"
    );
    assert!(
        h.vad_resets.load(Ordering::SeqCst) > resets_before,
        "VAD not reset"
    );
}

#[test]
fn rapid_v_presses_never_queue_duplicate_turns() {
    let gate = Gate::new();
    let mut h = Harness::new(Opts {
        engine: FakeEngine::new().with_gate(gate.clone()),
        ..Default::default()
    });
    let speech = h.speech(0.5);
    h.say_into_mic(&speech);
    h.stage(VoiceStage::Deciding);
    for i in 0..20 {
        h.send(Command::PushToTalk(i % 2 == 0));
    }
    h.wait("presses counted", |s| s.diagnostics.refused_ptt >= 10);
    h.assert_ticks_while(VoiceStage::Deciding, 20);
    gate.open();
    h.stage(VoiceStage::Idle);
    assert_eq!(h.engine_seen.lock().unwrap().len(), 1);
    assert_eq!(h.asr_received.lock().unwrap().len(), 1);
    assert_eq!(h.npc_lines().len(), 1);
    // And the next press records normally.
    let speech = h.speech(0.5);
    h.say_into_mic(&speech);
    h.wait("second turn", |s| {
        s.dialogue
            .iter()
            .filter(|l| l.kind == LineKind::Npc)
            .count()
            == 2
    });
}

#[test]
fn mic_self_test_writes_a_playable_wav_without_vad_or_asr() {
    let mut h = Harness::new(Opts {
        mic_rate: 44_100,
        mic_channels: 2,
        ..Default::default()
    });
    let path = std::env::temp_dir().join(format!("irl-selftest-{}.wav", std::process::id()));
    let path_s = path.to_string_lossy().into_owned();
    h.send(Command::MicSelfTest {
        path: path_s.clone(),
        seconds: 0.5,
    });
    // A device delivering audio in real time while the test records.
    let rate = 44_100usize;
    let signal = tone(rate as u32, 220.0, 1.0, 0.3);
    let t0 = Instant::now();
    let mut fed = 0usize;
    while t0.elapsed() < Duration::from_millis(800) {
        let due = ((t0.elapsed().as_secs_f64() * rate as f64) as usize).min(signal.len());
        if due > fed {
            h.feed(&signal[fed..due]);
            fed = due;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let s = h.wait("self-test report", |s| {
        s.dialogue
            .iter()
            .any(|l| l.text.starts_with("mic self-test"))
    });
    let line = s
        .dialogue
        .iter()
        .find(|l| l.text.starts_with("mic self-test"))
        .unwrap();
    assert!(
        line.text.contains("OK: the microphone records audio"),
        "{}",
        line.text
    );
    let wav = rpg_speech::read_wav(&path).expect("playable wav");
    assert_eq!(wav.sample_rate, 44_100);
    assert!(!wav.samples.is_empty());
    assert!(
        h.asr_received.lock().unwrap().is_empty(),
        "self-test used ASR"
    );
    let _ = std::fs::remove_file(path);
}

#[test]
fn selecting_another_microphone_reopens_the_device() {
    let mut h = Harness::new(Opts::default());
    h.send(Command::SelectMic(MicSelection::Next));
    h.wait("reopened", |s| {
        s.voice.mic.is_ready() && s.diagnostics.mic_device.starts_with("fake-mic-2")
    });
    let opened = h.mic.opened_with.lock().unwrap().clone();
    assert_eq!(opened.last().unwrap().as_deref(), Some("fake-mic-2"));
}

// ---------------------------------------------------------------------------
// Save/load, shutdown, sustained load
// ---------------------------------------------------------------------------

#[test]
fn save_and_load_while_the_model_is_thinking() {
    let gate = Gate::new();
    let mut h = Harness::new(Opts {
        engine: FakeEngine::new().with_gate(gate.clone()),
        ..Default::default()
    });
    let dir = std::env::temp_dir().join(format!("irl-rt-save-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("slot.json").to_string_lossy().into_owned();

    h.send(Command::Save(path.clone()));
    let saved = h.wait("saved", |s| {
        s.dialogue.iter().any(|l| l.text.starts_with("saved"))
    });
    h.send(Command::Say("Hello.".into()));
    h.assert_ticks_while(VoiceStage::Deciding, 30);
    h.send(Command::Load(path));
    let loaded = h.wait("loaded", |s| s.world_epoch == 1);
    assert_eq!(loaded.voice.stage, VoiceStage::Idle);
    assert!(loaded.tick < saved.tick + 30 + 5);
    gate.open();
    h.assert_ticks_while(VoiceStage::Idle, 20);
    assert!(
        h.npc_lines().is_empty(),
        "an answer from the old timeline leaked"
    );
    assert!(h.spoken.lock().unwrap().is_empty());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn shutdown_is_clean_while_recording() {
    let mut h = Harness::new(Opts::default());
    h.send(Command::PushToTalk(true));
    h.wait_mic_listening();
    let speech = h.speech(0.3);
    h.feed(&speech);
    let report = h.shutdown();
    assert!(report.detached.is_empty(), "{report:?}");
    assert_eq!(report.joined.len(), 5);
}

#[test]
fn shutdown_is_clean_while_the_model_and_playback_are_busy() {
    let gate = Gate::new();
    let mut h = Harness::new(Opts {
        engine: FakeEngine::new().with_gate(gate.clone()),
        instant_playback: false,
        ..Default::default()
    });
    h.send(Command::Say("Hello.".into()));
    h.stage(VoiceStage::Deciding);
    let report = h.shutdown();
    assert!(report.detached.is_empty(), "{report:?}");
    // Playback variant.
    let mut h = Harness::new(Opts {
        instant_playback: false,
        ..Default::default()
    });
    h.send(Command::Say("Hello.".into()));
    h.stage(VoiceStage::Speaking);
    let report = h.shutdown();
    assert!(report.detached.is_empty(), "{report:?}");
}

#[test]
fn fixed_tick_cadence_holds_under_sustained_voice_load() {
    let mut h = Harness::new(Opts {
        tick_ms: 20,
        ..Default::default()
    });
    let t0 = Instant::now();
    let start_tick = h.latest.tick;
    let mut turns = 0;
    while t0.elapsed() < Duration::from_secs(3) {
        let speech = h.speech(0.3);
        h.say_into_mic(&speech);
        turns += 1;
        h.wait("turn done", |s| {
            s.voice.stage == VoiceStage::Idle
                && s.dialogue
                    .iter()
                    .filter(|l| l.kind == LineKind::Npc)
                    .count()
                    == turns
        });
    }
    let elapsed = t0.elapsed();
    let ticks = h.latest.tick - start_tick;
    let tps = ticks as f32 / elapsed.as_secs_f32();
    eprintln!(
        "[tick-measure] sustained load: {turns} full voice turns, {ticks} ticks in {:.2}s = {tps:.1} tps (target 50), max lag {:.1} ms",
        elapsed.as_secs_f32(),
        h.latest.diagnostics.max_tick_lag_ms
    );
    assert!(turns >= 3);
    assert!(tps >= 40.0, "cadence fell to {tps:.1} tps");
}
