//! reproducible local-model + speech benchmark harness.
//!
//! What this binary *does* verify today:
//!   * deterministic scenario outcomes and wall-clock cost of the full
//!     simulated-player loop under the no-LLM baseline;
//!   * WAV synthesis and streaming/buffer accounting with the diagnostic
//!     (non-neural) device.
//!
//! What it explicitly does **not** claim:
//!   * real neural TTS/ASR quality — no model assets are bundled;
//!   * real LLM latency/quality — no GGUF weights are bundled.
//!
//! `--list-models` prints the candidate matrix with `NOT TESTED` status and the
//! exact commands to measure each one once assets are installed. Never replace
//! this with invented numbers.

use rpg_domain::ports::{SpeechRequest, SpeechSynthesizer};
use rpg_eval::run_all;
use rpg_speech::{measure_streaming, write_wav, DiagnosticToneSynth, MissingNeuralBackend};
use std::io::Write;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        help();
        return;
    }
    if args.iter().any(|a| a == "--list-models") {
        list_models();
        return;
    }
    if args.iter().any(|a| a == "--speech") {
        speech_bench();
        return;
    }

    let seed: u64 = arg_value(&args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2026);
    let out = arg_value(&args, "--out");
    let json = args.iter().any(|a| a == "--json");

    println!("# model-bench: integrated scenario baseline");
    println!("# engine=heuristic (no-LLM latency floor), model=heuristic-v1");
    println!("# status: VERIFIED-DETERMINISTIC (real LLM profiles are NOT TESTED)");
    println!("# seed: {seed}\n");

    let t0 = Instant::now();
    let results = run_all(seed);
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut jsonl = String::new();
    let mut passed = 0;
    for r in &results {
        if r.passed() {
            passed += 1;
        }
        let line = format!(
            "{{\"engine\":\"heuristic\",\"model\":\"heuristic-v1\",\"status\":\"VERIFIED-DETERMINISTIC\",\"scenario\":\"{}\",\"seed\":{},\"passed\":{},\"checks\":{},\"events\":{},\"checksum\":\"{:x}\"}}",
            r.name,
            r.seed,
            r.passed(),
            r.checks.len(),
            r.event_count,
            r.checksum
        );
        jsonl.push_str(&line);
        jsonl.push('\n');
        if !json {
            println!("{}", r.summary());
        }
    }

    if let Some(path) = out {
        let mut f = std::fs::File::create(path).expect("create output");
        f.write_all(jsonl.as_bytes()).expect("write output");
        println!("\nwrote {path}");
    }
    if json {
        print!("{jsonl}");
    }

    println!(
        "\nSummary: {passed}/{} scenarios passed in {wall_ms:.2} ms total.",
        results.len()
    );
    println!("Caveat: this is the deterministic baseline, not a neural model.");
    if passed != results.len() {
        std::process::exit(1);
    }
}

fn arg_value<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

fn help() {
    println!(
        "model-bench [options]\n\
         \x20 --seed N         scenario seed (default 2026)\n\
         \x20 --out PATH       write per-scenario JSONL\n\
         \x20 --json           print JSONL to stdout\n\
         \x20 --speech         run the integrated speech/buffer benchmark\n\
         \x20 --list-models    print the candidate model matrix (status)\n\
         \x20 --help"
    );
}

#[derive(serde::Serialize)]
struct Candidate {
    model: &'static str,
    format: &'static str,
    params: &'static str,
    status: &'static str,
    command: &'static str,
}

fn list_models() {
    let candidates = vec![
        Candidate {
            model: "Qwen3.5-0.8B",
            format: "GGUF 4-bit",
            params: "0.8B",
            status: "NOT TESTED",
            command: "docs/benchmarks/model-matrix.md#qwen35-08b",
        },
        Candidate {
            model: "Qwen3.5-2B",
            format: "GGUF 4-bit",
            params: "2B",
            status: "NOT TESTED",
            command: "docs/benchmarks/model-matrix.md#qwen35-2b",
        },
        Candidate {
            model: "Qwen3.5-4B",
            format: "GGUF 4-bit",
            params: "4B",
            status: "NOT TESTED",
            command: "docs/benchmarks/model-matrix.md#qwen35-4b",
        },
        Candidate {
            model: "Qwen3-1.7B",
            format: "GGUF 4/5/8-bit",
            params: "1.7B",
            status: "NOT TESTED",
            command: "docs/benchmarks/model-matrix.md#qwen3-17b",
        },
        Candidate {
            model: "LFM2.5-1.2B-Instruct",
            format: "GGUF",
            params: "1.2B",
            status: "NOT TESTED",
            command: "docs/benchmarks/model-matrix.md#lfm25",
        },
        Candidate {
            model: "Gemma 3 1B",
            format: "GGUF",
            params: "1B",
            status: "NOT TESTED",
            command: "docs/benchmarks/model-matrix.md#gemma3-1b",
        },
        Candidate {
            model: "heuristic (no-LLM baseline)",
            format: "in-process",
            params: "-",
            status: "VERIFIED-DETERMINISTIC",
            command: "cargo run -p model-bench",
        },
        Candidate {
            model: "scripted (test double)",
            format: "in-process",
            params: "-",
            status: "VERIFIED-DETERMINISTIC",
            command: "cargo test -p rpg_agents",
        },
    ];
    println!("model,format,params,status,evidence");
    for c in &candidates {
        println!(
            "{},{},{},{},{}",
            c.model, c.format, c.params, c.status, c.command
        );
    }
    println!("\nNo real-model measurements are claimed. See docs/benchmarks/README.md.");
}

fn speech_bench() {
    println!("# model-bench: integrated speech pipeline (DIAGNOSTIC, NON-NEURAL)");
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
        m.chunks, m.audio_seconds, m.wall_seconds, m.speed_ratio(), m.underruns, m.min_lead_seconds
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
