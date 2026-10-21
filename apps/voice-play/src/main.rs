//! # voice-play — the offline voice conversation client for IRL
//!
//! This is Milestone B: the complete, authoritative voice loop.
//!
//! ```text
//!   microphone
//!       │  (cpal capture, device-native rate)
//!       ▼
//!   resample → 16 kHz mono
//!       │
//!       ▼
//!   Silero VAD  ──►  utterance segment
//!       │
//!       ▼
//!   Whisper ASR ──►  transcript
//!       │
//!       ▼
//!   NPC selection (nearest in-range actor, or a pinned target)
//!       │
//!       ▼
//!   rpg_eval::Session::say ──► NpcPolicy ──► local LLM decision
//!       │                                         │
//!       │                                         ▼
//!       │                              rpg_sim validates + commits
//!       ▼
//!   grounded NPC dialogue text (the action the simulation actually accepted)
//!       │
//!       ▼
//!   Kokoro neural TTS ──► resample to device rate ──► speaker
//! ```
//!
//! The language model proposes; Rust validates and commits. Nothing the model
//! says is treated as truth, and the text spoken back is exactly the text of
//! the `Speak` action the authoritative simulation accepted — never a guess.
//!
//! ## Commands
//!
//! * `voice-play`                 — live microphone loop (needs audio devices)
//! * `voice-play once <file.wav>` — headless: transcribe a WAV, run one NPC turn
//! * `voice-play say "<text>"`    — headless: skip ASR, run one NPC turn from text
//! * `voice-play npcs`            — list the named NPCs and where they are
//! * `voice-play help`
//!
//! Headless commands need no audio device and are suitable for CI; `once`/`say`
//! still write a reply WAV so the neural TTS output can be inspected.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use rpg_domain::ids::{ActorId, PLAYER};
use rpg_domain::ports::{
    InferenceEngine, SpeechAudio, SpeechRecognizer, SpeechRequest, SpeechSynthesizer,
};
use rpg_domain::world::ActorKind;
use rpg_eval::Session;
use rpg_inference::{LlamaEngine, ModelProfile};
use rpg_speech::assets::SpeechAssets;
use rpg_speech::{SherpaAsr, SherpaTts};

/// Default neural voice. Any Kokoro English voice id works.
const DEFAULT_VOICE: &str = "af_sarah";
/// Conversation range, matching `SimConfig::talk_radius`.
const TALK_RADIUS: f32 = 3.5;
/// Everything in this app fails with a human-readable string.
type AppResult<T> = Result<T, String>;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("mic");
    let opts = match Opts::parse(&args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("error: {e}\n");
            print_help();
            std::process::exit(2);
        }
    };
    if opts.help {
        print_help();
        return;
    }
    let result = match cmd {
        "mic" | "play" => run_mic(&opts),
        "once" => run_once_wav(&opts),
        "say" => run_say_text(&opts),
        "npcs" => run_list_npcs(&opts),
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        other => {
            eprintln!("unknown command: {other}\n");
            print_help();
            std::process::exit(2);
        }
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

struct Opts {
    seed: u64,
    model: Option<String>,
    profile: Option<String>,
    heuristic: bool,
    voice: String,
    near: String,
    npc: Option<String>,
    cache: PathBuf,
    manifest: PathBuf,
    threads: i32,
    out: Option<String>,
    wav: Option<String>,
    text: Option<String>,
    help: bool,
}

impl Opts {
    fn parse(args: &[String]) -> AppResult<Self> {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut o = Opts {
            seed: 2026,
            model: None,
            profile: None,
            heuristic: false,
            voice: DEFAULT_VOICE.to_string(),
            near: "grocer".to_string(),
            npc: None,
            cache: workspace.join(".cache/speech"),
            manifest: workspace.join("assets/speech/manifest.json"),
            threads: 4,
            out: None,
            wav: None,
            text: None,
            help: false,
        };
        let mut i = 2;
        while i < args.len() {
            let a = args[i].as_str();
            match a {
                "--model" => o.model = Some(next(args, &mut i, a)?),
                "--profile" => o.profile = Some(next(args, &mut i, a)?),
                "--heuristic" => {
                    o.heuristic = true;
                    i += 1;
                }
                "--voice" => o.voice = next(args, &mut i, a)?,
                "--near" => o.near = next(args, &mut i, a)?,
                "--npc" => o.npc = Some(next(args, &mut i, a)?),
                "--cache" => o.cache = PathBuf::from(next(args, &mut i, a)?),
                "--manifest" => o.manifest = PathBuf::from(next(args, &mut i, a)?),
                "--threads" => o.threads = parse_next(args, &mut i, a)?,
                "--seed" => o.seed = parse_next(args, &mut i, a)?,
                "--out" => o.out = Some(next(args, &mut i, a)?),
                "--text" => o.text = Some(next(args, &mut i, a)?),
                "--help" | "-h" => {
                    o.help = true;
                    i += 1;
                }
                other if !other.starts_with('-') && o.wav.is_none() => {
                    o.wav = Some(other.to_string());
                    i += 1;
                }
                other => return Err(format!("unknown option '{other}'")),
            }
        }
        Ok(o)
    }
}

fn next(args: &[String], i: &mut usize, flag: &str) -> AppResult<String> {
    let v = args
        .get(*i + 1)
        .cloned()
        .ok_or_else(|| format!("{flag} needs a value"))?;
    *i += 2;
    Ok(v)
}

fn parse_next<T: std::str::FromStr>(args: &[String], i: &mut usize, flag: &str) -> AppResult<T> {
    next(args, i, flag)?
        .parse()
        .map_err(|_| format!("{flag} has an invalid value"))
}

// ---------------------------------------------------------------------------
// Session construction
// ---------------------------------------------------------------------------

fn build_session(o: &Opts) -> AppResult<Session> {
    if o.heuristic || (o.profile.is_none() && o.model.is_none()) {
        return Ok(Session::new(o.seed));
    }
    let profile = build_profile(o)?;
    let mut engine = LlamaEngine::load(profile).map_err(|e| e.to_string())?;
    engine.warm().map_err(|e| e.to_string())?;
    Ok(Session::with_engine(o.seed, Box::new(engine)))
}

fn build_profile(o: &Opts) -> AppResult<ModelProfile> {
    if let Some(path) = &o.profile {
        return ModelProfile::load(path).map_err(|e| format!("profile {path}: {e}"));
    }
    let model = o
        .model
        .as_ref()
        .ok_or_else(|| "no --profile or --model supplied".to_string())?;
    if !Path::new(model).exists() {
        return Err(format!("model file not found: {model}"));
    }
    let path = model.replace('\\', "/");
    let threads = o.threads.max(1);
    let json = format!(
        r#"{{"id":"voice-play","model_path":"{path}","n_ctx":4096,"n_threads":{threads},"max_tokens":160,"temperature":0.4,"no_think":true,"deadline_ms":60000}}"#
    );
    ModelProfile::from_json(&json).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Speech stack
// ---------------------------------------------------------------------------

struct Speech {
    asr: SherpaAsr,
    tts: SherpaTts,
    assets: SpeechAssets,
}

fn load_speech(o: &Opts) -> AppResult<Speech> {
    let assets = SpeechAssets::load(&o.cache, &o.manifest)
        .map_err(|e| format!("speech assets: {e} (run `voice-demo install` first)"))?;
    let mut asr = SherpaAsr::from_assets(&assets, rpg_speech::DEFAULT_ASR_BUNDLE, o.threads)
        .map_err(|e| e.to_string())?;
    asr.warm().map_err(|e| e.to_string())?;
    let mut tts = SherpaTts::from_assets(&assets, rpg_speech::DEFAULT_TTS_BUNDLE, o.threads)
        .map_err(|e| e.to_string())?;
    tts.warm(&o.voice).map_err(|e| e.to_string())?;
    Ok(Speech { asr, tts, assets })
}

// ---------------------------------------------------------------------------
// NPC selection
// ---------------------------------------------------------------------------

/// Choose who the player is talking to.
///
/// A pinned `--npc` always wins. Otherwise the nearest living, named NPC in the
/// same room and within conversation range is selected — the same rule the
/// simulation uses to accept speech, so the client never addresses someone the
/// validator would reject.
fn select_npc(session: &Session, pinned: Option<&str>) -> Option<ActorId> {
    if let Some(id) = pinned {
        return Some(ActorId::new(id));
    }
    let player = session.player();
    session
        .sim
        .world
        .actors
        .values()
        .filter(|a| a.id.0 != PLAYER)
        .filter(|a| a.kind == ActorKind::Npc)
        .filter(|a| a.pos.distance(player.pos) <= TALK_RADIUS)
        .filter(|a| a.location == player.location)
        .min_by(|a, b| {
            a.pos
                .distance(player.pos)
                .partial_cmp(&b.pos.distance(player.pos))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|a| a.id.clone())
}

// ---------------------------------------------------------------------------
// Live microphone loop
// ---------------------------------------------------------------------------

fn run_mic(o: &Opts) -> AppResult<()> {
    use rpg_speech::audio::{
        AudioCapture, AudioPlayback, CaptureConfig, LinearResampler, PlaybackConfig,
    };
    use rpg_speech::VadSegmenter;

    let mut session = build_session(o)?;
    announce(&session);
    let mut speech = load_speech(o)?;
    let assets = speech.assets.clone();

    // Put the player next to the requested character so the first utterance is
    // already in range. The player can still move with the text client; this is
    // a convenience for a voice-first session.
    session.teleport_player_to(&o.near);
    println!("You are standing near '{}'.", o.near);

    let capture =
        AudioCapture::open(&CaptureConfig::default()).map_err(|e| format!("microphone: {e}"))?;
    let playback = AudioPlayback::open(&PlaybackConfig::new(speech.tts.sample_rate()))
        .map_err(|e| format!("speaker: {e}"))?;
    let mut vad = VadSegmenter::from_assets(&assets, rpg_speech::DEFAULT_VAD_BUNDLE, 16_000)
        .map_err(|e| e.to_string())?;
    let mut to_asr = LinearResampler::new(capture.native_sample_rate, 16_000);

    println!(
        "Listening on the default microphone ({} Hz). Speak to an NPC; Ctrl+C quits.",
        capture.native_sample_rate
    );
    println!("Voice: {}   Engine: {}", o.voice, session.engine_name());
    println!("(Tip: if nobody answers, move next to an NPC in the text client first.)\n");

    loop {
        let Some(chunk) = capture.recv() else {
            println!("microphone closed.");
            break;
        };
        let mono16 = to_asr.process(&chunk);
        for seg in vad.accept(&mono16) {
            handle_utterance(&mut session, &mut speech, &playback, &seg, o);
        }
    }
    Ok(())
}

/// One full turn: ASR → NPC selection → decision → validation → TTS → playback.
fn handle_utterance(
    session: &mut Session,
    speech: &mut Speech,
    playback: &rpg_speech::audio::AudioPlayback,
    seg: &rpg_speech::SpeechSegmentAudio,
    o: &Opts,
) {
    let t0 = Instant::now();
    let transcript = match speech.asr.transcribe(&seg.samples, seg.sample_rate) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[asr] {e}");
            return;
        }
    };
    let asr_ms = t0.elapsed().as_secs_f32() * 1000.0;
    let text = transcript.text.trim().to_string();
    if text.is_empty() {
        return;
    }
    println!(
        "\n[you] {text}   (asr {asr_ms:.0} ms, {:.2}s audio)",
        seg.audio_seconds()
    );
    run_turn(session, speech, Some(playback), &text, o);
}

/// Drive the authoritative simulation for one player utterance and speak the
/// grounded reply.
///
/// `playback` is `Some` only when a real output device is open; headless runs
/// pass `None` so they never touch audio hardware.
fn run_turn(
    session: &mut Session,
    speech: &mut Speech,
    playback: Option<&rpg_speech::audio::AudioPlayback>,
    text: &str,
    o: &Opts,
) {
    let target = match select_npc(session, o.npc.as_deref()) {
        Some(t) => t,
        None => {
            println!("[npc] nobody is close enough to hear you. Move nearer and try again.");
            return;
        }
    };
    let d0 = Instant::now();
    let reply = session.say(target.as_str(), text);
    let decide_ms = d0.elapsed().as_secs_f32() * 1000.0;
    // Advance the world so time, schedules and background actors progress.
    let _ = session.tick();
    match reply {
        Some(reply_text) => {
            println!("[{}] {}", target.as_str(), reply_text);
            println!(
                "      (decision {decide_ms:.0} ms, engine {})",
                session.engine_name()
            );
            if let Some(pb) = playback {
                speak(&mut speech.tts, pb, &reply_text, &o.voice);
            }
            if o.out.is_some() {
                write_reply(o, &mut speech.tts, &reply_text);
            }
        }
        None => {
            // The NPC heard the player but chose a non-verbal action (an offer,
            // a promise, a threat), or the action was rejected. We do not invent
            // dialogue; report exactly what the authoritative simulation did.
            let trace = session.trace.last().cloned().unwrap_or_default();
            println!(
                "[{}] (no spoken reply)   ({decide_ms:.0} ms)",
                target.as_str()
            );
            println!("      {trace}");
        }
    }
}

/// Synthesize and play a reply, resampling to the real device rate.
fn speak(
    tts: &mut SherpaTts,
    playback: &rpg_speech::audio::AudioPlayback,
    text: &str,
    voice: &str,
) {
    let audio = match tts.synthesize(&SpeechRequest {
        text: text.to_string(),
        voice_id: voice.to_string(),
        sample_rate: playback.sample_rate,
    }) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("[tts] {e}");
            return;
        }
    };
    let samples = resample_for_device(&audio, playback.sample_rate);
    println!(
        "[tts] {:.2}s audio in {:.0} ms ({:.2}x real time)",
        audio.audio_seconds(),
        audio.synth_ms,
        audio.real_time_factor()
    );
    if let Err(e) = playback.queue(&samples) {
        eprintln!("[audio] {e}");
        return;
    }
    let seconds = samples.len() as f32 / playback.sample_rate as f32;
    if !playback.wait_until_drained(Duration::from_secs_f32(seconds + 2.0)) {
        eprintln!("[audio] warning: playback did not drain in time");
    }
}

fn resample_for_device(audio: &SpeechAudio, device_rate: u32) -> Vec<f32> {
    if audio.sample_rate == device_rate {
        audio.samples.clone()
    } else {
        rpg_speech::audio::LinearResampler::new(audio.sample_rate, device_rate)
            .process(&audio.samples)
    }
}

// ---------------------------------------------------------------------------
// Headless commands
// ---------------------------------------------------------------------------

fn run_once_wav(o: &Opts) -> AppResult<()> {
    let wav = o.wav.as_ref().ok_or_else(|| {
        "usage: voice-play once <file.wav> [--model ...] [--out reply.wav]".to_string()
    })?;
    let mut session = build_session(o)?;
    announce(&session);
    session.teleport_player_to(&o.near);
    let mut speech = load_speech(o)?;
    let audio = rpg_speech::read_wav(wav).map_err(|e| e.to_string())?;
    let t0 = Instant::now();
    let transcript = speech
        .asr
        .transcribe(&audio.samples, audio.sample_rate)
        .map_err(|e| e.to_string())?;
    println!(
        "[asr] {} ({:.0} ms, {:.2}s audio)",
        transcript.text.trim(),
        t0.elapsed().as_secs_f32() * 1000.0,
        audio.audio_seconds()
    );
    run_turn(&mut session, &mut speech, None, transcript.text.trim(), o);
    print_summary(&session);
    Ok(())
}

fn run_say_text(o: &Opts) -> AppResult<()> {
    // The text may be given positionally (`say "..."`) or with `--text`.
    let text = o.text.as_ref().or(o.wav.as_ref()).ok_or_else(|| {
        "usage: voice-play say \"<text>\" [--model ...] [--out reply.wav]".to_string()
    })?;
    let mut session = build_session(o)?;
    announce(&session);
    session.teleport_player_to(&o.near);
    let mut speech = load_speech(o)?;
    println!("[you] {text}");
    run_turn(&mut session, &mut speech, None, text, o);
    print_summary(&session);
    Ok(())
}

/// Write the synthesized reply to `--out` when the caller does not want live
/// playback. If `--out` is absent this is a no-op so headless runs stay silent.
fn write_reply(o: &Opts, tts: &mut SherpaTts, text: &str) {
    let Some(path) = &o.out else {
        return;
    };
    match tts.synthesize(&SpeechRequest {
        text: text.to_string(),
        voice_id: o.voice.clone(),
        sample_rate: 24_000,
    }) {
        Ok(audio) => match rpg_speech::write_wav(path, &audio) {
            Ok(()) => println!(
                "[tts] wrote {path} ({:.2}s audio in {:.0} ms, {:.2}x real time)",
                audio.audio_seconds(),
                audio.synth_ms,
                audio.real_time_factor()
            ),
            Err(e) => eprintln!("[tts] {e}"),
        },
        Err(e) => eprintln!("[tts] {e}"),
    }
}

fn run_list_npcs(o: &Opts) -> AppResult<()> {
    let session = build_session(o)?;
    println!("Named NPCs in the town:");
    for a in session.sim.world.actors.values() {
        if a.kind == ActorKind::Npc {
            println!(
                "  {:<16} {:<18} at {:<20} ({:.1}, {:.1})",
                a.id.as_str(),
                a.name,
                a.location.as_str(),
                a.pos.x,
                a.pos.y
            );
        }
    }
    Ok(())
}

fn announce(session: &Session) {
    println!("IRL — offline voice client (Milestone B pipeline)");
    if session.is_real_model() {
        println!(
            "Engine: {} (model {}) — real local model",
            session.engine_name(),
            session.model_id()
        );
    } else {
        println!(
            "Engine: {} — DETERMINISTIC BASELINE (pass --model <gguf> for a real model)",
            session.engine_name()
        );
    }
}

fn print_summary(session: &Session) {
    println!(
        "\nSession: {} model decision(s), {} canonical obligation(s), {} deterministic fallback(s).",
        session.model_decisions(),
        session.canonical_decisions(),
        session.fallback_decisions()
    );
}

fn print_help() {
    println!(
        "voice-play — offline voice conversation client for IRL\n\
         \n\
         USAGE:\n\
         \x20 voice-play [mic] [options]           live microphone conversation\n\
         \x20 voice-play once <file.wav> [options]  headless: WAV -> ASR -> one NPC turn\n\
         \x20 voice-play say \"<text>\" [options]     headless: text -> one NPC turn (no ASR)\n\
         \x20 voice-play npcs [options]             list named NPCs and locations\n\
         \n\
         OPTIONS:\n\
         \x20 --model <gguf>       Resident local GGUF model (omit for deterministic baseline)\n\
         \x20 --profile <json>     Model profile file (overrides --model)\n\
         \x20 --heuristic          Force the deterministic no-LLM baseline\n\
         \x20 --voice <id>         Kokoro TTS voice id (default {DEFAULT_VOICE})\n\
         \x20 --near <npc>         Start next to this NPC (default grocer)\n\
         \x20 --npc <id>           Pin the conversation target instead of auto-selecting\n\
         \x20 --cache <dir>        Speech asset cache (default .cache/speech)\n\
         \x20 --manifest <path>    Speech asset manifest\n\
         \x20 --threads <n>        Worker threads for LLM/TTS/ASR (default 4)\n\
         \x20 --seed <n>           World seed (default 2026)\n\
         \x20 --out <path>         Write the reply WAV in headless mode\n\
         \n\
         Prerequisites:\n\
         \x20 voice-demo install                 # download + verify neural speech assets\n\
         \x20 --model .cache/models/<name>.gguf  # a local GGUF for real NPC decisions\n\
         \n\
         The model only proposes; the authoritative Rust simulation validates and\n\
         commits every action. The reply you hear is the accepted Speak action's text."
    );
}
