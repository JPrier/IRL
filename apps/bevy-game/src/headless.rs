//! Windowless diagnostics for real hardware.
//!
//! * `--mic-selftest` records raw microphone PCM straight from the device to a
//!   WAV file — no VAD, ASR, model or TTS — and reports level and sample rate,
//!   separating "the microphone records nothing" from "speech is not
//!   detected".
//! * `--voice-check` runs the production runtime (the same threads and
//!   backends as the game) through full voice turns, from a live microphone
//!   or a WAV file standing in for one, and measures the simulation tick rate
//!   during every voice stage.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use rpg_runtime::{
    Command, LineKind, MicInfo, RuntimeConfig, SelfTestReport, Snapshot, VoiceStage,
};
use rpg_speech::audio::{AudioCapture, CaptureConfig};

use crate::voice::{self, VoiceOptions};

/// Record `seconds` of raw audio from the device and write it to `out`.
pub fn mic_selftest(device: Option<String>, seconds: f32, out: PathBuf) -> i32 {
    println!("Input devices:");
    match AudioCapture::list_devices() {
        Ok(list) if !list.is_empty() => {
            for d in list {
                let mark = if d.is_default { " (default)" } else { "" };
                println!("  - {}{mark}", d.name);
            }
        }
        Ok(_) => println!("  (none)"),
        Err(e) => println!("  cannot enumerate: {e}"),
    }
    let capture = match AudioCapture::open(&CaptureConfig {
        device_name: device,
        ..CaptureConfig::default()
    }) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("FAIL: cannot open the microphone: {e}");
            return 2;
        }
    };
    let info = MicInfo {
        name: capture.device_name.clone(),
        sample_rate: capture.native_sample_rate,
        channels: capture.device_channels,
        sample_format: capture.sample_format.clone(),
    };
    println!(
        "Recording {seconds:.1}s from {} — speak now…",
        info.describe()
    );
    let start = Instant::now();
    let mut samples = Vec::new();
    let mut error = None;
    while start.elapsed().as_secs_f32() < seconds {
        while let Some(chunk) = capture.try_recv() {
            samples.extend_from_slice(&chunk);
        }
        if let Some(e) = capture.take_error() {
            error = Some(e);
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let wall = start.elapsed().as_secs_f32();
    let level = rpg_speech::dsp::Level::of(&samples);
    let audio = rpg_domain::ports::SpeechAudio {
        samples,
        sample_rate: info.sample_rate,
        synth_ms: 0.0,
        first_audio_ms: 0.0,
    };
    if let Err(e) = rpg_speech::write_wav(&out, &audio) {
        eprintln!("FAIL: cannot write {}: {e}", out.display());
        return 2;
    }
    let report = SelfTestReport {
        path: out.display().to_string(),
        requested_seconds: seconds,
        samples: audio.samples.len(),
        wall_seconds: wall,
        rms: level.rms,
        peak: level.peak,
        dropped_chunks: capture.dropped_chunks(),
        device: info,
    };
    println!("{}", report.summary());
    if let Some(e) = error {
        println!("stream error during the test: {e}");
    }
    println!(
        "Play {} back to hear exactly what the game hears.",
        report.path
    );
    if report.verdict().starts_with("FAIL") {
        1
    } else {
        0
    }
}

pub struct VoiceCheck {
    pub opts: VoiceOptions,
    pub seed: u64,
    pub near: String,
    pub turns: usize,
    /// Seconds to hold push-to-talk per turn.
    pub hold: f32,
}

#[derive(Default)]
struct StageStats {
    wall: Duration,
    ticks: u64,
    entries: usize,
}

/// Run full voice turns through the production runtime and report tick
/// rates per stage. Returns a process exit code.
pub fn voice_check(check: VoiceCheck) -> i32 {
    let mut session = rpg_eval::Session::new(check.seed);
    session.teleport_player_to(&check.near);
    let cfg = RuntimeConfig {
        seed: check.seed,
        ..RuntimeConfig::default()
    };
    let target = 1.0 / cfg.tick_period.as_secs_f32();
    let rt = rpg_runtime::spawn(cfg, session, voice::backends(&check.opts));
    let mut latest: Option<Snapshot> = None;
    let mut stats: BTreeMap<String, StageStats> = BTreeMap::new();
    let mut last_mark: Option<(VoiceStage, Instant, u64)> = None;
    let mut observe = |s: &Snapshot, stats: &mut BTreeMap<String, StageStats>| {
        let now = Instant::now();
        match last_mark {
            Some((stage, ..)) if stage == s.voice.stage => {}
            Some((stage, t, tick)) => {
                let e = stats.entry(format!("{stage:?}")).or_default();
                e.wall += now - t;
                e.ticks += s.tick.saturating_sub(tick);
                e.entries += 1;
                last_mark = Some((s.voice.stage, now, s.tick));
            }
            None => last_mark = Some((s.voice.stage, now, s.tick)),
        }
    };
    let mut wait = |what: &str,
                    limit: Duration,
                    pred: &dyn Fn(&Snapshot) -> bool,
                    latest: &mut Option<Snapshot>,
                    stats: &mut BTreeMap<String, StageStats>|
     -> bool {
        let deadline = Instant::now() + limit;
        loop {
            if latest.as_ref().is_some_and(pred) {
                return true;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                eprintln!("timed out waiting for {what}");
                return false;
            }
            if let Ok(s) = rt
                .snapshots
                .recv_timeout(left.min(Duration::from_millis(100)))
            {
                observe(&s, stats);
                *latest = Some(s);
            }
        }
    };

    println!("Loading model and speech engines…");
    let loaded = wait(
        "engines",
        Duration::from_secs(300),
        &|s| {
            s.voice.mic.is_ready()
                && s.voice.asr.is_ready()
                && s.voice.tts.is_ready()
                && !s.engine_name.starts_with("loading")
        },
        &mut latest,
        &mut stats,
    );
    if let Some(s) = &latest {
        println!("engine : {} ({})", s.engine_name, s.model_id);
        println!("mic    : {}", s.voice.mic.describe());
        println!("asr    : {}", s.voice.asr.describe());
        println!("tts    : {}", s.voice.tts.describe());
    }
    if !loaded {
        let _ = rt.shutdown(Duration::from_secs(2));
        return 2;
    }
    stats.clear();
    let mut ok = true;
    for turn in 1..=check.turns {
        let lines_before = latest.as_ref().map(|s| s.dialogue.len()).unwrap_or(0);
        println!("\n--- turn {turn}: recording for {:.1}s ---", check.hold);
        rt.send(Command::PushToTalk(true));
        let hold_until = Instant::now() + Duration::from_secs_f32(check.hold);
        wait(
            "recording",
            Duration::from_secs_f32(check.hold + 1.0),
            &|_| Instant::now() >= hold_until,
            &mut latest,
            &mut stats,
        );
        rt.send(Command::PushToTalk(false));
        // Give the pipeline a moment to leave Listening, then wait for Idle.
        wait(
            "processing",
            Duration::from_secs(5),
            &|s| s.voice.stage != VoiceStage::Listening,
            &mut latest,
            &mut stats,
        );
        let finished = wait(
            "turn to finish",
            Duration::from_secs(180),
            &|s| s.voice.stage == VoiceStage::Idle,
            &mut latest,
            &mut stats,
        );
        ok &= finished;
        if let Some(s) = &latest {
            for line in s.dialogue.iter().skip(lines_before) {
                let who = match line.kind {
                    LineKind::Player => "you".to_string(),
                    LineKind::Npc => line.speaker.clone(),
                    LineKind::System => "system".to_string(),
                };
                println!("{who:>10}: {}", line.text);
            }
            let d = &s.diagnostics;
            println!(
                "timing    : asr {}  llm queue {} / infer {}  tts first audio {} / synth {}",
                opt_ms(d.asr_latency_ms),
                opt_ms(d.llm_queue_ms),
                opt_ms(d.llm_infer_ms),
                opt_ms(d.tts_first_audio_ms),
                opt_ms(d.tts_synth_ms)
            );
            println!(
                "provenance: {} model / {} canonical / {} fallback",
                s.model_decisions, s.canonical_decisions, s.fallback_decisions
            );
            ok &= d.last_transcript.as_deref().is_some_and(|t| !t.is_empty());
        }
    }

    println!("\nSimulation ticks per voice stage (target {target:.1} tps):");
    println!("{:<14}{:>10}{:>8}{:>10}", "stage", "wall s", "ticks", "tps");
    for (stage, st) in &stats {
        let secs = st.wall.as_secs_f32();
        let tps = if secs > 0.0 {
            st.ticks as f32 / secs
        } else {
            0.0
        };
        println!("{stage:<14}{secs:>10.2}{:>8}{tps:>10.1}", st.ticks);
        // Very short stages cannot hold a full tick; judge the rest.
        if secs >= 0.5 && tps < target * 0.5 {
            println!("  ^ simulation slowed during {stage}");
            ok = false;
        }
    }
    if let Some(s) = &latest {
        println!(
            "max tick lag {:.1} ms, skipped ticks {}, stale responses {}, timeouts {}",
            s.diagnostics.max_tick_lag_ms,
            s.diagnostics.skipped_ticks,
            s.diagnostics.stale_responses,
            s.diagnostics.timeouts
        );
    }
    let report = rt.shutdown(Duration::from_secs(5));
    println!(
        "shutdown: joined {:?}, detached {:?}",
        report.joined, report.detached
    );
    println!("\nvoice-check: {}", if ok { "PASS" } else { "FAIL" });
    if ok {
        0
    } else {
        1
    }
}

fn opt_ms(v: Option<f32>) -> String {
    v.map(|m| format!("{m:.0} ms"))
        .unwrap_or_else(|| "—".into())
}
