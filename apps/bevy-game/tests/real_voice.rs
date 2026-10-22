//! Real-model, real-voice end-to-end check of the shipped executable.
//!
//! Runs `bevy-game --voice-check`, which drives the production runtime (the
//! same threads and Silero/Whisper/llama.cpp/Kokoro backends as the game)
//! through full voice turns with a WAV fixture standing in for the
//! microphone, and fails if a turn produces no transcript or the simulation
//! slows during any voice stage. It needs the bundled model, the speech
//! assets and an output device, so it is ignored by default:
//!
//! ```bash
//! cargo test --release -p bevy-game -- --ignored
//! ```
//!
//! It runs the built binary rather than linking the backends into the test
//! harness: on Windows the harness executable would otherwise resolve the
//! system's older `onnxruntime.dll` instead of the one shipped beside the game.

use std::path::Path;
use std::process::Command;

#[test]
#[ignore]
fn real_voice_pipeline_keeps_the_world_ticking() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture = root.join(".cache/speech/fixture-ledger.wav");
    assert!(fixture.is_file(), "missing {}", fixture.display());
    let out = Command::new(env!("CARGO_BIN_EXE_bevy-game"))
        .env("IRL_HOME", &root)
        .args([
            "--voice-check",
            "--near",
            "grocer",
            "--turns",
            "2",
            "--mic-wav",
        ])
        .arg(&fixture)
        .output()
        .expect("run bevy-game");
    let stdout = String::from_utf8_lossy(&out.stdout);
    println!("{stdout}");
    assert!(out.status.success(), "voice-check failed:\n{stdout}");
    assert!(stdout.contains("voice-check: PASS"));
    assert!(
        stdout.to_lowercase().contains("ledger"),
        "fixture not transcribed"
    );
}
