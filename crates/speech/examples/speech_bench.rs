//! Diagnostic, NON-NEURAL speech/buffer benchmark.
//!
//! Identical measurement to `model-bench --speech`, but it lives in `rpg_speech`
//! so the binary links only the speech stack. The combined `model-bench` binary
//! also links `llama.cpp`, and on some Linux runners the static initialisers of
//! llama.cpp + onnxruntime abort with `std::bad_alloc` before `main` runs. CI
//! therefore runs this example, which exercises the same WAV and buffer
//! accounting.
//!
//! This is explicitly non-neural: it proves buffering and throughput, not
//! speech quality. Never cite it as neural-TTS evidence.

use std::time::Instant;

use rpg_domain::ports::{SpeechRequest, SpeechSynthesizer};
use rpg_speech::{measure_streaming, write_wav, DiagnosticToneSynth, MissingNeuralBackend};

fn main() {
    println!("# rpg_speech: integrated speech pipeline (DIAGNOSTIC, NON-NEURAL)");
    println!("# This measures buffering/throughput, not speech quality.\n");

    let mut synth = DiagnosticToneSynth::new();
    synth.warm("diag-low").unwrap();

    let mut chunk_metrics = Vec::new();
    for (label, words) in [("5s", 16usize), ("15s", 48), ("40s", 128)] {
        let text = "the quick brown fox jumps over the lazy dog. ".repeat(words / 9 + 1);
        let t0 = Instant::now();
        let audio = synth
            .synthesize(&SpeechRequest {
                text,
                voice_id: "diag-low".into(),
                sample_rate: 24_000,
            })
            .unwrap();
        let wall = t0.elapsed().as_secs_f64() as f32;
        let path = std::env::temp_dir().join(format!("agentic_rpg_bench_{label}.wav"));
        write_wav(&path, &audio).unwrap();
        println!(
            "{label:>3}: audio={:.2}s wall={:.3}s rtf={:.1}x bytes={} file={}",
            audio.audio_seconds(),
            wall.max(0.0001),
            audio.audio_seconds() / wall.max(0.0001),
            audio.samples.len() * 2,
            path.display()
        );
        chunk_metrics.push((audio.audio_seconds(), wall.max(0.0001)));
    }

    let m = measure_streaming(&chunk_metrics);
    println!(
        "\nstream: chunks={} audio={:.2}s wall={:.3}s speed_ratio={:.2}x underruns={} min_lead={:.2}s",
        m.chunks,
        m.audio_seconds,
        m.wall_seconds,
        m.speed_ratio(),
        m.underruns,
        m.min_lead_seconds
    );
    println!(
        "Note: target is >=1.5x at P05 for a REAL neural backend; the diagnostic \
         device trivially exceeds it and proves nothing about neural TTS."
    );

    let mut missing = MissingNeuralBackend {
        expected_asset: "assets/voices/<licensed-model>.onnx".into(),
    };
    match missing.warm("any") {
        Ok(()) => println!("unexpected: neural backend reported present"),
        Err(e) => println!("neural backend status: {e} (expected until assets are installed)"),
    }
}
