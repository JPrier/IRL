//! Reproducible local-model + speech benchmark harness.
//!
//! Two honest modes:
//!   * **baseline** (`--seed`, no profile): deterministic scenarios and the
//!     diagnostic (non-neural) speech device. Clearly labelled as *not* AI.
//!   * **real model** (`--profile <file.json>`): loads the GGUF once through
//!     `rpg_inference`, measures load/warm/prefill/decode, checks the
//!     structured-action rate, and runs the grocer bargain with the real
//!     engine. A missing model asset is reported and never replaced with
//!     heuristic numbers.
//!
//! What is still not claimed: neural TTS/ASR quality (no assets bundled).

use rpg_domain::action::Action;
use rpg_domain::ids::*;
use rpg_domain::ports::{
    InferenceEngine, InferenceRequest, InferenceRole, SpeechRequest, SpeechSynthesizer,
};
use rpg_inference::{backend_info, LlamaEngine, ModelProfile};
use rpg_speech::{measure_streaming, write_wav, DiagnosticToneSynth, MissingNeuralBackend};
use serde::Serialize;
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
    if let Some(profile) = arg_value(&args, "--profile") {
        let code = profile_bench(&args, profile);
        if code != 0 {
            std::process::exit(code);
        }
        return;
    }
    baseline_bench(&args);
}

// ---------------------------------------------------------------------------
// Deterministic baseline
// ---------------------------------------------------------------------------

fn baseline_bench(args: &[String]) {
    let seed: u64 = arg_value(args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2026);
    let out = arg_value(args, "--out");
    let json = args.iter().any(|a| a == "--json");

    println!("# model-bench: integrated scenario baseline");
    println!("# engine=heuristic (no-LLM latency floor), model=heuristic-v1");
    println!("# status: VERIFIED-DETERMINISTIC (real LLM profiles are NOT TESTED)");
    println!("# seed: {seed}\n");

    let t0 = Instant::now();
    let results = rpg_eval::run_all(seed);
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

// ---------------------------------------------------------------------------
// Real model
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct BatteryReport {
    requests: usize,
    valid_json: usize,
    invalid_json: usize,
    valid_rate: f32,
    total_prompt_tokens: u32,
    total_completion_tokens: u32,
    avg_prefill_ms: f32,
    avg_first_token_ms: f32,
    avg_decode_ms: f32,
    tokens_per_sec: f32,
    structured: bool,
}

#[derive(Serialize)]
struct GameplayReport {
    scenario: &'static str,
    seed: u64,
    turns: usize,
    model_decisions: usize,
    canonical_decisions: usize,
    fallbacks: usize,
    ledger_disclosed: bool,
    memory_consistent: bool,
}

#[derive(Serialize)]
struct DialogueReport {
    text: String,
    completion_tokens: u32,
    decode_ms: f32,
    empty: bool,
    contains_reasoning: bool,
}

fn profile_bench(args: &[String], profile_path: &str) -> i32 {
    let seed: u64 = arg_value(args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2026);
    let turns: usize = arg_value(args, "--turns")
        .and_then(|s| s.parse().ok())
        .unwrap_or(5)
        .max(1);
    let out = arg_value(args, "--out");
    let json = args.iter().any(|a| a == "--json");

    let profile = match ModelProfile::load(profile_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("model-bench: cannot load profile {profile_path}: {e}");
            return 2;
        }
    };
    if !std::path::Path::new(&profile.model_path).exists() {
        eprintln!(
            "model-bench: MISSING MODEL ASSET '{}' for profile '{}'",
            profile.model_path, profile.id
        );
        eprintln!("model-bench: refusing to substitute heuristic results for a real profile.");
        return 3;
    }

    println!("# model-bench: REAL local model");
    println!("# profile: {} ({})", profile.id, profile_path);
    println!("# backend: {}", backend_info());
    println!("# status: REAL-MODEL-MEASURED\n");

    let weights_bytes = profile.weight_bytes();
    let load_start = Instant::now();
    let mut engine = match LlamaEngine::load(profile.clone()) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("model-bench: failed to load model: {e}");
            return 4;
        }
    };
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    let warm_start = Instant::now();
    if let Err(e) = engine.warm() {
        eprintln!("model-bench: warm-up failed: {e}");
        return 5;
    }
    let warm_ms = warm_start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "load={load_ms:.0}ms warm={warm_ms:.0}ms weights={} MiB",
        weights_bytes / (1024 * 1024)
    );

    // --- structured tool-decision battery ---------------------------------
    let world = rpg_content::build_town();
    let npc = ActorId::new("grocer");
    let player = ActorId::new(PLAYER);
    let phrasings = [
        "I need to see your ledger.",
        "Will you show me the ledger?",
        "Hand over the ledger.",
        "What do you know about the payments?",
        "The ledger, please.",
    ];
    let mut battery = BatteryReport {
        requests: 0,
        valid_json: 0,
        invalid_json: 0,
        valid_rate: 0.0,
        total_prompt_tokens: 0,
        total_completion_tokens: 0,
        avg_prefill_ms: 0.0,
        avg_first_token_ms: 0.0,
        avg_decode_ms: 0.0,
        tokens_per_sec: 0.0,
        structured: true,
    };
    let mut prefill_sum = 0.0f32;
    let mut first_sum = 0.0f32;
    let mut decode_sum = 0.0f32;
    let mut decode_tokens = 0u32;
    for i in 0..turns {
        let utterance = phrasings[i % phrasings.len()];
        let prompt = rpg_agents::render_npc_context(&world, &npc, &player, utterance);
        let req = InferenceRequest {
            id: i as u64 + 1,
            role: InferenceRole::NpcDecision,
            system: "Reply with one JSON object only.".into(),
            prompt,
            max_tokens: 160,
            temperature: 0.0,
            seed: i as u64,
            response_contract: Some("action-json-v1".into()),
        };
        match engine.complete(&req) {
            Ok(r) => {
                battery.requests += 1;
                battery.total_prompt_tokens += r.prompt_tokens;
                battery.total_completion_tokens += r.completion_tokens;
                prefill_sum += r.prefill_ms;
                first_sum += r.first_token_ms;
                decode_sum += r.decode_ms;
                decode_tokens += r.completion_tokens;
                let valid = serde_json::from_str::<serde_json::Value>(&r.text)
                    .ok()
                    .and_then(|v| v.get("action").map(|_| ()))
                    .is_some();
                if valid {
                    battery.valid_json += 1;
                } else {
                    battery.invalid_json += 1;
                }
                println!(
                    "  turn {i}: valid_json={valid} tokens={} prefill={:.0}ms first={:.0}ms decode={:.0}ms",
                    r.completion_tokens, r.prefill_ms, r.first_token_ms, r.decode_ms
                );
            }
            Err(e) => {
                battery.requests += 1;
                battery.invalid_json += 1;
                println!("  turn {i}: ERROR {e}");
            }
        }
    }
    if battery.requests > 0 {
        let n = battery.requests as f32;
        battery.valid_rate = battery.valid_json as f32 / n;
        battery.avg_prefill_ms = prefill_sum / n;
        battery.avg_first_token_ms = first_sum / n;
        battery.avg_decode_ms = decode_sum / n;
        let secs = decode_sum / 1000.0;
        battery.tokens_per_sec = if secs > 0.0 {
            decode_tokens as f32 / secs
        } else {
            0.0
        };
    }

    // --- free-text dialogue probe -----------------------------------------
    let dialogue_req = InferenceRequest {
        id: 9001,
        role: InferenceRole::Dialogue,
        system: "You are Marta Voss, a guarded grocer whose daughter is ill. Stay in character."
            .into(),
        prompt: "The player asks: 'How is your daughter?' Reply in one short sentence.".into(),
        max_tokens: 64,
        temperature: 0.3,
        seed: 99,
        response_contract: None,
    };
    let dialogue = match engine.complete(&dialogue_req) {
        Ok(r) => DialogueReport {
            text: r.text.clone(),
            completion_tokens: r.completion_tokens,
            decode_ms: r.decode_ms,
            empty: r.text.trim().is_empty(),
            contains_reasoning: r.text.contains("think"),
        },
        Err(e) => DialogueReport {
            text: format!("error: {e}"),
            completion_tokens: 0,
            decode_ms: 0.0,
            empty: true,
            contains_reasoning: false,
        },
    };

    // --- real-engine gameplay loop over several seeds ---------------------
    let mut engine: Box<dyn InferenceEngine> = Box::new(engine);
    let mut gameplays = Vec::new();
    for gseed in [seed, seed.wrapping_add(1), seed.wrapping_add(2)] {
        let (g, returned) = run_gameplay(engine, gseed);
        gameplays.push(g);
        engine = returned;
    }
    let gameplay_pass_rate = if gameplays.is_empty() {
        0.0
    } else {
        gameplays.iter().filter(|g| g.ledger_disclosed).count() as f32 / gameplays.len() as f32
    };

    let peak_rss = peak_rss_bytes();

    let report = serde_json::json!({
        "harness": "model-bench",
        "real_model": true,
        "engine": "llama.cpp",
        "model_id": profile.id,
        "backend": backend_info(),
        "profile": profile_path,
        "host": {
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "available_parallelism": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
            "n_ctx": profile.n_ctx,
            "n_threads": profile.n_threads,
        },
        "weights_bytes": weights_bytes,
        "load_ms": load_ms,
        "warm_ms": warm_ms,
        "peak_rss_bytes": peak_rss,
        "battery": battery,
        "dialogue": dialogue,
        "gameplay": gameplays,
        "gameplay_pass_rate": gameplay_pass_rate,
        "seed": seed,
    });

    print_report(&report, out, json);
    println!(
        "\nStructured validity: {}/{} ({:.0}%), {:.1} tok/s decode",
        battery.valid_json,
        battery.requests,
        battery.valid_rate * 100.0,
        battery.tokens_per_sec
    );
    if gameplay_pass_rate < 1.0 {
        eprintln!("model-bench: FAIL: a real-engine gameplay loop did not disclose the ledger");
        return 1;
    }
    0
}

fn run_gameplay(
    engine: Box<dyn InferenceEngine>,
    seed: u64,
) -> (GameplayReport, Box<dyn InferenceEngine>) {
    let mut s = rpg_eval::Session::with_engine(seed, engine);
    s.teleport_player_to("grocer");
    s.say("grocer", "I need to see your ledger.");

    let offer = s
        .sim
        .world
        .offers
        .values()
        .find(|o| o.from.as_str() == "grocer")
        .map(|o| o.id);
    let mut turns = 1usize;
    if let Some(offer) = offer {
        let _ = s.player_do(Action::AcceptOffer { offer });
    }
    turns += 1;
    s.give("medicine", 1);
    let _ = s.player_do(Action::TransferItem {
        target: ActorId::new("grocer"),
        item: ItemId::new("medicine"),
        qty: 1,
    });
    s.tick();
    turns += 1;
    s.say("grocer", "Now the ledger, as we agreed.");
    turns += 1;

    let player = ActorId::new(PLAYER);
    let ledger = s
        .sim
        .world
        .believes(&player, &FactId::new("f_grocer_ledger"));
    let daughter = s
        .sim
        .world
        .believes(&player, &FactId::new("f_daughter_ill"));
    let report = GameplayReport {
        scenario: "grocer_conditional_bargain_model",
        seed,
        turns,
        model_decisions: s.model_decisions(),
        canonical_decisions: s.canonical_decisions(),
        fallbacks: s.fallback_decisions(),
        ledger_disclosed: ledger,
        memory_consistent: ledger && daughter,
    };
    (report, s.into_engine())
}

fn print_report(report: &serde_json::Value, out: Option<&str>, json: bool) {
    let text = serde_json::to_string_pretty(report).unwrap_or_default();
    if let Some(path) = out {
        if let Err(e) = std::fs::write(path, format!("{text}\n")) {
            eprintln!("model-bench: could not write {path}: {e}");
        } else {
            println!("\nwrote {path}");
        }
    }
    if json {
        println!("{text}");
    }
}

/// Peak resident set size when the platform exposes it (Linux `/proc`).
fn peak_rss_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
                return Some(kb * 1024);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn arg_value<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

fn help() {
    println!(
        "model-bench [options]\n\
         \x20 --profile PATH    run a real ModelProfile JSON (REAL model)\n\
         \x20 --seed N          seed (default 2026)\n\
         \x20 --turns N         structured-decision battery size (default 5)\n\
         \x20 --out PATH        write the JSON report\n\
         \x20 --json            print the JSON report to stdout\n\
         \x20 --speech          run the integrated speech/buffer benchmark\n\
         \x20 --list-models     print the candidate model matrix (status)\n\
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
            status: "MEASURED (local probe + model-bench --profile)",
            command: "cargo run -p model-bench -- --profile .cache/models/qwen35-0.8b.json",
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
            command: "cargo test -p rpg_eval",
        },
    ];
    println!("model,format,params,status,evidence");
    for c in &candidates {
        println!(
            "{},{},{},{},{}",
            c.model, c.format, c.params, c.status, c.command
        );
    }
    println!("\nA real-model row is only MEASURED once `model-bench --profile` has run.");
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
