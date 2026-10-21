//! # voice-demo
//!
//! A small, honest offline voice harness for IRL. It exercises the real neural
//! speech stack end to end:
//!
//! ```text
//!   microphone ──► Silero VAD ──► Whisper ASR ──► (reply text)
//!                                                   │
//!   speaker   ◄── audio device ◄── Kokoro neural TTS ◄┘
//! ```
//!
//! Headless commands (`asr`, `tts`, `roundtrip`, `bench`) need no audio device
//! and are suitable for CI; `mic` needs a real input and output device.
//!
//! Run `cargo run -p voice-demo -- --help` for usage.

use std::path::{Path, PathBuf};
use std::time::Instant;

use rpg_domain::ports::{SpeechError, SpeechRequest, SpeechSynthesizer};
use rpg_domain::ports::SpeechRecognizer;
use rpg_speech::assets::SpeechAssets;
use rpg_speech::{SherpaAsr, SherpaTts};

/// The default voice used when `--voice` is not supplied.
const DEFAULT_VOICE: &str = "af_sarah";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }
    let cmd = args.get(1).map(String::as_str).unwrap_or("help");
    let opts = Opts::parse(&args);
    let result = match cmd {
        "list" => cmd_list(&opts),
        "install" => cmd_install(&opts),
        "asr" => cmd_asr(&opts),
        "tts" => cmd_tts(&opts),
        "roundtrip" => cmd_roundtrip(&opts),
        "bench" => cmd_bench(&opts),
        "mic" => cmd_mic(&opts),
        "help" => {
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

/// Parsed command-line options.
struct Opts {
    cache: PathBuf,
    manifest: PathBuf,
    voice: String,
    threads: i32,
    text: Option<String>,
    wav: Option<String>,
    out: Option<String>,
    play: bool,
}

impl Opts {
    fn parse(args: &[String]) -> Self {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut o = Opts {
            cache: workspace.join(".cache/speech"),
            manifest: workspace.join("assets/speech/manifest.json"),
            voice: DEFAULT_VOICE.to_string(),
            threads: 4,
            text: None,
            wav: None,
            out: None,
            play: false,
        };
        let mut i = 2;
        while i < args.len() {
            match args[i].as_str() {
                "--cache" => {
                    if let Some(v) = args.get(i + 1) {
                        o.cache = PathBuf::from(v);
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                "--manifest" => {
                    if let Some(v) = args.get(i + 1) {
                        o.manifest = PathBuf::from(v);
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                "--voice" => {
                    if let Some(v) = args.get(i + 1) {
                        o.voice = v.clone();
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                "--threads" => {
                    if let Some(v) = args.get(i + 1) {
                        o.threads = v.parse().unwrap_or(4);
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                "--text" => {
                    if let Some(v) = args.get(i + 1) {
                        o.text = Some(v.clone());
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                "--out" => {
                    if let Some(v) = args.get(i + 1) {
                        o.out = Some(v.clone());
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                "--play" => {
                    o.play = true;
                    i += 1;
                }
                other if !other.starts_with("--") && o.wav.is_none() => {
                    o.wav = Some(other.to_string());
                    i += 1;
                }
                _ => i += 1,
            }
        }
        o
    }
}

fn load_assets(o: &Opts) -> Result<SpeechAssets, SpeechError> {
    SpeechAssets::load(&o.cache, &o.manifest)
        .map_err(|e| SpeechError::MissingAsset(format!("{e} (manifest: {})", o.manifest.display())))
}

fn open_tts(o: &Opts, assets: &SpeechAssets) -> Result<SherpaTts, SpeechError> {
    SherpaTts::from_assets(assets, rpg_speech::DEFAULT_TTS_BUNDLE, o.threads)
}

fn open_asr(o: &Opts, assets: &SpeechAssets) -> Result<SherpaAsr, SpeechError> {
    SherpaAsr::from_assets(assets, rpg_speech::DEFAULT_ASR_BUNDLE, o.threads)
        .map_err(|e| SpeechError::Backend(e.to_string()))
}

fn cmd_list(o: &Opts) -> Result<(), SpeechError> {
    let assets = load_assets(o)?;
    println!("manifest : {}", o.manifest.display());
    println!("cache    : {}", o.cache.display());
    println!("models   : {}", assets.models_dir().display());
    println!("\nasset bundles:");
    for b in &assets.manifest().bundles {
        let state = if assets.is_installed(&b.id) {
            "INSTALLED"
        } else {
            "missing"
        };
        println!(
            "  [{state:>9}] {:<22} {:?}  license={}  {} bytes",
            b.id, b.kind, b.license, b.bytes
        );
    }
    println!("\ninput devices:");
    match rpg_speech::audio::AudioCapture::list_devices() {
        Ok(devs) if !devs.is_empty() => {
            for d in devs {
                println!("  {}{}", d.name, if d.is_default { "  (default)" } else { "" });
            }
        }
        Ok(_) => println!("  (none found)"),
        Err(e) => println!("  unavailable: {e}"),
    }
    println!("\noutput devices:");
    match rpg_speech::audio::AudioPlayback::list_devices() {
        Ok(devs) if !devs.is_empty() => {
            for d in devs {
                println!("  {}{}", d.name, if d.is_default { "  (default)" } else { "" });
            }
        }
        Ok(_) => println!("  (none found)"),
        Err(e) => println!("  unavailable: {e}"),
    }
    Ok(())
}

fn cmd_install(o: &Opts) -> Result<(), SpeechError> {
    let assets = load_assets(o)?;
    println!("installing speech assets into {}", o.cache.display());
    let mut log = |line: String| println!("{line}");
    assets
        .install_all(&mut log)
        .map_err(|e| SpeechError::Backend(e.to_string()))?;
    println!("done.");
    Ok(())
}

fn cmd_asr(o: &Opts) -> Result<(), SpeechError> {
    let wav = o.wav.as_ref().ok_or_else(|| {
        SpeechError::Backend("usage: voice-demo asr <file.wav>".into())
    })?;
    let assets = load_assets(o)?;
    let audio = rpg_speech::read_wav(wav)?;
    println!(
        "read {} : {} samples @ {} Hz ({:.2}s)",
        wav,
        audio.samples.len(),
        audio.sample_rate,
        audio.audio_seconds()
    );
    let mut asr = open_asr(o, &assets)?;
    asr.warm().map_err(|e| SpeechError::Backend(e.to_string()))?;
    let t0 = Instant::now();
    let t = asr
        .transcribe(&audio.samples, audio.sample_rate)
        .map_err(|e| SpeechError::Backend(e.to_string()))?;
    println!("backend   : {}", asr.backend_name());
    println!("latency   : {:.1} ms (reported {:.1} ms)", t0.elapsed().as_secs_f32() * 1000.0, t.latency_ms);
    println!("transcript: {}", t.text);
    Ok(())
}

fn cmd_tts(o: &Opts) -> Result<(), SpeechError> {
    let text = o
        .text
        .as_ref()
        .ok_or_else(|| SpeechError::Backend("usage: voice-demo tts --text \"...\" [--out x.wav]".into()))?;
    let assets = load_assets(o)?;
    let mut tts = open_tts(o, &assets)?;
    tts.warm(&o.voice)?;
    let audio = tts.synthesize(&SpeechRequest {
        text: text.clone(),
        voice_id: o.voice.clone(),
        sample_rate: 24_000,
    })?;
    let out = o
        .out
        .clone()
        .unwrap_or_else(|| "voice-demo-tts.wav".to_string());
    rpg_speech::write_wav(&out, &audio).map_err(|e| SpeechError::Backend(e.to_string()))?;
    println!("backend      : {}", tts.backend_name());
    println!("voice        : {}", o.voice);
    println!("audio        : {:.2}s @ {} Hz", audio.audio_seconds(), audio.sample_rate);
    println!("synth        : {:.1} ms", audio.synth_ms);
    println!("real-time x  : {:.2}x", audio.real_time_factor());
    println!("wrote        : {out}");
    if o.play {
        play_audio(&audio)?;
    }
    Ok(())
}

fn cmd_roundtrip(o: &Opts) -> Result<(), SpeechError> {
    let wav = o.wav.as_ref().ok_or_else(|| {
        SpeechError::Backend("usage: voice-demo roundtrip <in.wav> [--out out.wav]".into())
    })?;
    let assets = load_assets(o)?;
    let audio = rpg_speech::read_wav(wav)?;
    let mut asr = open_asr(o, &assets)?;
    let mut tts = open_tts(o, &assets)?;
    asr.warm().map_err(|e| SpeechError::Backend(e.to_string()))?;
    tts.warm(&o.voice)?;

    let a0 = Instant::now();
    let t = asr
        .transcribe(&audio.samples, audio.sample_rate)
        .map_err(|e| SpeechError::Backend(e.to_string()))?;
    let asr_ms = a0.elapsed().as_secs_f32() * 1000.0;

    let s0 = Instant::now();
    let reply = tts.synthesize(&SpeechRequest {
        text: t.text.clone(),
        voice_id: o.voice.clone(),
        sample_rate: 24_000,
    })?;
    let tts_ms = s0.elapsed().as_secs_f32() * 1000.0;

    let out = o
        .out
        .clone()
        .unwrap_or_else(|| "voice-demo-roundtrip.wav".to_string());
    rpg_speech::write_wav(&out, &reply).map_err(|e| SpeechError::Backend(e.to_string()))?;
    println!("ASR  : {} ({:.0} ms)", t.text, asr_ms);
    println!("TTS  : {:.2}s audio in {:.0} ms ({:.2}x real time)", reply.audio_seconds(), tts_ms, reply.real_time_factor());
    println!("wrote: {out}");
    if o.play {
        play_audio(&reply)?;
    }
    Ok(())
}

/// Play a synthesized buffer through the default output device.
///
/// Resamples when the device cannot run at the synthesis rate, so the audio is
/// never played at the wrong speed.
fn play_audio(audio: &rpg_domain::ports::SpeechAudio) -> Result<(), SpeechError> {
    use rpg_speech::audio::{AudioPlayback, LinearResampler, PlaybackConfig};
    let playback = AudioPlayback::open(&PlaybackConfig::new(audio.sample_rate))?;
    let samples = if playback.sample_rate == audio.sample_rate {
        audio.samples.clone()
    } else {
        println!(
            "[audio] device runs at {} Hz; resampling from {} Hz",
            playback.sample_rate, audio.sample_rate
        );
        LinearResampler::new(audio.sample_rate, playback.sample_rate).process(&audio.samples)
    };
    let seconds = samples.len() as f32 / playback.sample_rate as f32;
    playback.queue(&samples)?;
    let timeout = std::time::Duration::from_secs_f32(seconds + 2.0);
    if !playback.wait_until_drained(timeout) {
        eprintln!("[audio] warning: playback did not drain within the timeout");
    }
    Ok(())
}

fn cmd_bench(o: &Opts) -> Result<(), SpeechError> {
    let assets = load_assets(o)?;
    let mut asr = open_asr(o, &assets)?;
    let mut tts = open_tts(o, &assets)?;
    asr.warm().map_err(|e| SpeechError::Backend(e.to_string()))?;
    tts.warm(&o.voice)?;
    println!("# voice-demo bench (real neural TTS + ASR)");
    println!("asr backend: {}", asr.backend_name());
    println!("tts backend: {}", tts.backend_name());
    println!("voice      : {}", o.voice);

    let sentences = [
        "The ledger is behind the flour.",
        "I will give you the medicine if you tell me who signed the book.",
        "Come back after dark and we will talk about the shipment.",
    ];
    println!("\n{:<52} {:>8} {:>8} {:>10} {:>9}", "text", "audio_s", "synth_ms", "first_ms", "rtf");
    let mut total_audio = 0.0f32;
    let mut total_wall = 0.0f32;
    for s in sentences {
        let audio = tts.synthesize(&SpeechRequest {
            text: s.to_string(),
            voice_id: o.voice.clone(),
            sample_rate: 24_000,
        })?;
        total_audio += audio.audio_seconds();
        total_wall += audio.synth_ms / 1000.0;
        println!(
            "{:<52} {:>8.2} {:>8.1} {:>10.1} {:>9.2}",
            truncate(s, 50),
            audio.audio_seconds(),
            audio.synth_ms,
            audio.first_audio_ms,
            audio.real_time_factor()
        );
    }
    println!(
        "\naggregate: {:.2}s audio in {:.2}s wall = {:.2}x real time",
        total_audio,
        total_wall,
        if total_wall > 0.0 { total_audio / total_wall } else { f32::INFINITY }
    );

    // ASR latency on a real fixture if present in the cache.
    let fixture = assets
        .models_dir()
        .join("sherpa-onnx-whisper-tiny.en/test_wavs/0.wav");
    if fixture.is_file() {
        let audio = rpg_speech::read_wav(&fixture)?;
        let t = asr
            .transcribe(&audio.samples, audio.sample_rate)
            .map_err(|e| SpeechError::Backend(e.to_string()))?;
        println!(
            "asr       : {:.1} ms for {:.2}s audio  ({:.2}x real time)",
            t.latency_ms,
            audio.audio_seconds(),
            audio.audio_seconds() / (t.latency_ms / 1000.0)
        );
        println!("asr text  : {}", t.text);
    } else {
        println!("asr       : (no fixture at {})", fixture.display());
    }
    Ok(())
}

fn cmd_mic(o: &Opts) -> Result<(), SpeechError> {
    use rpg_speech::audio::{AudioCapture, AudioPlayback, CaptureConfig, LinearResampler, PlaybackConfig};
    use rpg_speech::VadSegmenter;

    let assets = load_assets(o)?;
    let mut asr = open_asr(o, &assets)?;
    let mut tts = open_tts(o, &assets)?;
    asr.warm().map_err(|e| SpeechError::Backend(e.to_string()))?;
    tts.warm(&o.voice)?;

    let capture = AudioCapture::open(&CaptureConfig::default())?;
    let playback = AudioPlayback::open(&PlaybackConfig::new(tts.sample_rate()))?;
    let mut vad = VadSegmenter::from_assets(&assets, rpg_speech::DEFAULT_VAD_BUNDLE, 16_000)?;

    // Resample captured audio from the device rate to 16 kHz for VAD/ASR.
    let mut resampler = rpg_speech::audio::LinearResampler::new(
        capture.native_sample_rate,
        16_000,
    );

    println!("listening on '{}' @ {} Hz", "default input", capture.native_sample_rate);
    println!("speak; press Ctrl+C to quit. The reply is spoken back with neural TTS.");
    println!("(voice: {})", o.voice);

    let mut listening = true;
    while listening {
        let Some(chunk) = capture.recv() else { break };
        let mono16 = resampler.process(&chunk);
        let segments = vad.accept(&mono16);
        for seg in segments {
            let t0 = Instant::now();
            let transcript = match asr.transcribe(&seg.samples, seg.sample_rate) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("[asr] {e}");
                    continue;
                }
            };
            println!(
                "\n[you] {}  (asr {:.0} ms, {:.2}s audio)",
                transcript.text,
                t0.elapsed().as_secs_f32() * 1000.0,
                seg.audio_seconds()
            );
            // Milestone A: canned reply. LLM grounding arrives in Milestone B.
            let reply = format!("I heard you say: {}", transcript.text);
            let audio = match tts.synthesize(&SpeechRequest {
                text: reply.clone(),
                voice_id: o.voice.clone(),
                sample_rate: playback.sample_rate,
            }) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("[tts] {e}");
                    continue;
                }
            };
            println!(
                "[npc] {}  (tts {:.0} ms, {:.2}x real time)",
                reply,
                audio.synth_ms,
                audio.real_time_factor()
            );
            // The neural model renders at its native rate; resample to whatever
            // the output device actually runs at so playback speed is correct.
            let to_play = if audio.sample_rate == playback.sample_rate {
                audio.samples.clone()
            } else {
                LinearResampler::new(audio.sample_rate, playback.sample_rate)
                    .process(&audio.samples)
            };
            playback.queue(&to_play)?;
        }
        if !vad.speaking() {
            listening = true;
        }
    }
    Ok(())
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(n - 1).collect();
        out.push('…');
        out
    }
}

fn print_help() {
    println!(
        "voice-demo — offline neural speech harness (no cloud services)\n\
         \n\
         USAGE:\n\
         \x20 voice-demo <command> [options]\n\
         \n\
         COMMANDS:\n\
         \x20 list                 List audio devices and installed speech assets\n\
         \x20 install              Download, verify and unpack the pinned assets\n\
         \x20 asr <file.wav>       Transcribe a WAV file (no audio device needed)\n\
         \x20 tts --text \"...\"     Synthesize text to a WAV (no audio device needed)\n\
         \x20 roundtrip <in.wav>   ASR then neural TTS; writes a reply WAV\n\
         \x20 bench                Measure neural TTS real-time factor and ASR latency\n\
         \x20 mic                  Live microphone -> ASR -> neural TTS -> speaker\n\
         \n\
         OPTIONS:\n\
         \x20 --cache <dir>        Speech asset cache (default .cache/speech)\n\
         \x20 --manifest <path>    Asset manifest (default assets/speech/manifest.json)\n\
         \x20 --voice <id>         TTS voice id (default {DEFAULT_VOICE})\n\
         \x20 --threads <n>        Worker threads for TTS/ASR (default 4)\n\
         \x20 --out <path>         Output WAV for tts/roundtrip\n\
         \x20 --play               Play the result through the default speaker"
    );
}
