//! Graphical, voice-native client for IRL.
//!
//! The authoritative simulation runs in [`rpg_runtime`] on its own fixed-tick
//! thread, with the microphone, speech recognizer, language model and voice
//! on dedicated workers. This binary is a *renderer and input device*: it
//! sends [`Command`]s and draws the immutable [`Snapshot`]s that come back.
//! It never mutates world state and never waits on the simulation, and
//! `rpg_sim` never depends on Bevy — the crate boundary is the invariant the
//! design asks for.
//!
//! The presentation itself (world, HUD, dialogue, offers, inventory,
//! diagnostics) is [`irl_client`], shared verbatim with the browser build;
//! this binary adds the native runtime, devices, model loading and CLI.
//!
//! NPC decisions come from the explicitly selected provider: local llama.cpp
//! (default), the deterministic policy, or OpenRouter (`--provider
//! openrouter`, key from `$OPENROUTER_API_KEY`). All of them are subordinate
//! to the same canonical validation and commit path.

mod headless;
mod paths;
mod voice;

use std::path::PathBuf;
use std::time::Duration;

use bevy::prelude::*;
use irl_client::{ClientPlatform, Link};
use rpg_runtime::{Command, MicConfig, RuntimeConfig, RuntimeHandle, Snapshot};

use paths::default_save_path;
use voice::{Provider, VoiceOptions};

/// The native platform link: the threaded `rpg_runtime`, reached through
/// its inbox and bounded snapshot queue.
struct NativeLink {
    runtime: Option<RuntimeHandle>,
}

impl Link for NativeLink {
    fn send(&mut self, cmd: Command) {
        if let Some(rt) = &self.runtime {
            rt.send(cmd);
        }
    }

    /// Drain every snapshot the simulation published and keep the newest.
    fn poll(&mut self) -> Option<Snapshot> {
        let rt = self.runtime.as_ref()?;
        let mut newest = None;
        while let Ok(snap) = rt.snapshots.try_recv() {
            newest = Some(snap);
        }
        newest
    }

    fn quick_save(&mut self) {
        self.send(Command::Save(default_save_path()));
    }

    fn quick_load(&mut self) {
        self.send(Command::Load(default_save_path()));
    }

    fn mic_self_test(&mut self) {
        let path = paths::home_root().join("mic_selftest.wav");
        self.send(Command::MicSelfTest {
            path: path.to_string_lossy().into_owned(),
            seconds: 4.0,
        });
    }

    /// Stop the runtime so the microphone, speaker and model are released
    /// instead of being killed with the process.
    fn shutdown(&mut self) {
        if let Some(rt) = self.runtime.take() {
            let report = rt.shutdown(Duration::from_millis(1500));
            if !report.detached.is_empty() {
                eprintln!("[shutdown] still busy at exit: {:?}", report.detached);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

struct Args {
    seed: u64,
    opts: VoiceOptions,
    auto_endpoint: bool,
    dump_utterance: bool,
}

fn main() {
    let args = parse_args();
    let mut session = rpg_eval::Session::new(args.seed);
    // The session's own engine is unused: decisions run on the runtime's
    // inference worker. A default placement keeps the first turn simple.
    session.teleport_player_to("grocer");
    let cfg = RuntimeConfig {
        seed: args.seed,
        mic: MicConfig {
            device: args.opts.mic_device.clone(),
            auto_endpoint: args.auto_endpoint.then_some(Duration::from_millis(900)),
            dump_utterance: args
                .dump_utterance
                .then(|| paths::home_root().join("last_utterance.wav")),
            ..MicConfig::default()
        },
        ..RuntimeConfig::default()
    };
    let runtime = rpg_runtime::spawn(cfg, session, voice::backends(&args.opts));
    let window = Window {
        title: "IRL — Hidden Mafia".into(),
        ..default()
    };
    irl_client::app(
        Box::new(NativeLink {
            runtime: Some(runtime),
        }),
        ClientPlatform::Native,
        window,
    )
    .run();
}

fn parse_args() -> Args {
    let mut seed = 2026u64;
    let mut opts = VoiceOptions {
        voice: "af_sarah".into(),
        ..VoiceOptions::default()
    };
    let mut auto_endpoint = false;
    let mut dump_utterance = false;
    let mut selftest: Option<f32> = None;
    let mut voice_check = false;
    let mut near = "grocer".to_string();
    let mut turns = 1usize;
    let mut hold: Option<f32> = None;

    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    let next = |i: &mut usize| -> Option<String> {
        *i += 1;
        argv.get(*i).cloned()
    };
    while i < argv.len() {
        match argv[i].as_str() {
            "--seed" => {
                if let Some(v) = next(&mut i).and_then(|s| s.parse().ok()) {
                    seed = v;
                }
            }
            "--model" => opts.model_path = next(&mut i),
            "--provider" => match next(&mut i).as_deref() {
                Some("local") => opts.provider = Provider::Local,
                Some("openrouter") => opts.provider = Provider::OpenRouter,
                Some("heuristic") => opts.provider = Provider::Heuristic,
                other => {
                    eprintln!(
                        "unknown --provider {other:?} (expected local, openrouter or heuristic)"
                    );
                    std::process::exit(2);
                }
            },
            "--openrouter-model" => {
                if let Some(v) = next(&mut i) {
                    opts.openrouter.model = v;
                }
            }
            "--openrouter-key-file" => opts.openrouter_key_file = next(&mut i).map(PathBuf::from),
            "--openrouter-max-tokens" => {
                if let Some(v) = next(&mut i).and_then(|s| s.parse().ok()) {
                    opts.openrouter.max_output_tokens = v;
                }
            }
            "--openrouter-max-requests" => {
                opts.openrouter.budget.max_requests = next(&mut i).and_then(|s| s.parse().ok());
            }
            "--openrouter-max-cost" => {
                opts.openrouter.budget.max_cost_usd = next(&mut i).and_then(|s| s.parse().ok());
            }
            "--openrouter-timeout-ms" => {
                if let Some(v) = next(&mut i).and_then(|s| s.parse().ok()) {
                    opts.openrouter.timeout_ms = v;
                }
            }
            "--openrouter-structured" => {
                match next(&mut i)
                    .as_deref()
                    .and_then(rpg_openrouter::StructuredMode::parse)
                {
                    Some(m) => opts.openrouter.structured = m,
                    None => {
                        eprintln!(
                            "--openrouter-structured expects json_schema, json_object or prompt"
                        );
                        std::process::exit(2);
                    }
                }
            }
            "--openrouter-no-stream" => opts.openrouter.stream = false,
            "--openrouter-key" | "--api-key" => {
                eprintln!(
                    "refusing a key on the command line (shell history and process listings keep it): \
                     set OPENROUTER_API_KEY or use --openrouter-key-file"
                );
                std::process::exit(2);
            }
            "--voice" => {
                if let Some(v) = next(&mut i) {
                    opts.voice = v;
                }
            }
            "--no-voice" => opts.no_voice = true,
            "--mic" => opts.mic_device = next(&mut i),
            "--mic-wav" => opts.mic_wav = next(&mut i).map(PathBuf::from),
            "--auto-endpoint" => auto_endpoint = true,
            "--dump-utterance" => dump_utterance = true,
            "--list-devices" => {
                list_devices();
                std::process::exit(0);
            }
            "--mic-selftest" => {
                selftest = Some(
                    argv.get(i + 1)
                        .and_then(|s| s.parse().ok())
                        .inspect(|_| i += 1)
                        .unwrap_or(5.0),
                );
            }
            "--voice-check" => voice_check = true,
            "--near" => {
                if let Some(v) = next(&mut i) {
                    near = v;
                }
            }
            "--turns" => {
                if let Some(v) = next(&mut i).and_then(|s| s.parse().ok()) {
                    turns = v;
                }
            }
            "--hold" => hold = next(&mut i).and_then(|s| s.parse().ok()),
            "--doctor" | "--paths" => {
                println!("{}", paths::doctor_report());
                std::process::exit(0);
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    // With no explicit `--model`, fall back to a model bundled next to the
    // executable so a shipped build runs with no arguments. The default
    // provider stays local: OpenRouter is used only when explicitly selected.
    opts.model_path = opts.model_path.or_else(paths::bundled_model);

    if let Some(seconds) = selftest {
        let out = paths::home_root().join("mic_selftest.wav");
        std::process::exit(headless::mic_selftest(
            opts.mic_device.clone(),
            seconds,
            out,
        ));
    }
    if voice_check {
        // A WAV "microphone" needs the whole file plus a moment of silence.
        let hold = hold.unwrap_or_else(|| {
            opts.mic_wav
                .as_ref()
                .and_then(|p| rpg_speech::read_wav(p).ok())
                .map(|a| a.audio_seconds() + 1.0)
                .unwrap_or(5.0)
        });
        std::process::exit(headless::voice_check(headless::VoiceCheck {
            opts,
            seed,
            near,
            turns,
            hold,
        }));
    }

    Args {
        seed,
        opts,
        auto_endpoint,
        dump_utterance,
    }
}

fn list_devices() {
    use rpg_speech::audio::{AudioCapture, AudioPlayback};
    for (label, list) in [
        ("Input", AudioCapture::list_devices()),
        ("Output", AudioPlayback::list_devices()),
    ] {
        println!("{label} devices:");
        match list {
            Ok(list) => {
                for d in list {
                    let mark = if d.is_default { " (default)" } else { "" };
                    println!("  - {}{mark}", d.name);
                }
            }
            Err(e) => println!("  cannot enumerate: {e}"),
        }
    }
}

fn print_help() {
    println!(
        "bevy-game — graphical, voice-native IRL client\n\
         \n\
         Usage: bevy-game [--seed N] [--model path.gguf] [--voice ID] [--no-voice]\n\
         \x20                [--mic NAME] [--auto-endpoint] [--dump-utterance]\n\
         \n\
         Inference provider (default: local; no network is ever used unless selected):\n\
           --provider local          llama.cpp with --model (or the bundled .gguf); the default\n\
           --provider heuristic      deterministic policy, no model (tests, baselines)\n\
           --provider openrouter     OpenRouter API; key from $OPENROUTER_API_KEY or\n\
             [--openrouter-key-file F]   the first line of F (never on the command line)\n\
             [--openrouter-model SLUG]   default: {default_model}\n\
             [--openrouter-max-tokens N] [--openrouter-max-requests N] [--openrouter-max-cost USD]\n\
             [--openrouter-timeout-ms MS] [--openrouter-no-stream]\n\
             [--openrouter-structured json_schema|json_object|prompt]\n\
                                       json_schema (default) requires strict structured outputs;\n\
                                       json_object is an explicit fallback with no schema guarantee\n\
         \n\
         Diagnostics (no window):\n\
           --list-devices            list audio input/output devices\n\
           --mic-selftest [SECS]     record raw microphone audio to mic_selftest.wav\n\
           --voice-check             run full voice turns through the real runtime and\n\
                                     report simulation ticks per voice stage\n\
             [--mic-wav FILE]        speak FILE into the pipeline instead of the mic\n\
             [--near NPC] [--turns N] [--hold SECS]\n\
           --doctor                  show where assets, models and saves are found\n\
         \n\
         Controls:\n\
           WASD / arrows   walk (always available, also while an NPC is thinking)\n\
           V (hold)        push-to-talk: speak while held, release to send\n\
           C               cancel the current conversation turn / interrupt speech\n\
           Enter           type a line instead of speaking (Enter sends, Esc cancels)\n\
           Y / N / Tab     accept / reject / cycle offers\n\
           E               investigate the current location\n\
           F5 / F9         quick save / load\n\
           F3              developer diagnostics\n\
           F7              switch microphone\n\
           F8              microphone self-test (writes mic_selftest.wav)\n\
           Esc             quit\n\
         \n\
         Voices (Kokoro): af_alloy af_bella af_nicole af_sarah af_sky am_adam\n\
                          am_michael bf_emma bf_isabella bm_george bm_lewis\n",
        default_model = rpg_openrouter::DEFAULT_MODEL
    );
}
