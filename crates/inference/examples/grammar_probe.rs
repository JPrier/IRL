//! Isolated reproducer for the grammar-sampler native abort.
//!
//! Run with a real GGUF:
//!   cargo run -p rpg_inference --example grammar_probe -- <model.gguf>
//!
//! It runs the *same* prompt twice: once unconstrained and once constrained by
//! the `action-json-v1` JSON-schema grammar. The constrained run is the one
//! that historically hit `GGML_ASSERT(!stacks.empty())` inside llama.cpp once
//! the grammar reached its terminal state and the loop kept sampling.

use rpg_domain::ports::{InferenceEngine, InferenceRequest, InferenceRole};
use rpg_inference::{LlamaEngine, ModelProfile};

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| ".cache/models/Qwen3.5-0.8B-Q4_K_M.gguf".into());
    let normalized = path.replace(std::path::MAIN_SEPARATOR, "/");
    let profile = ModelProfile::from_json(&format!(
        r#"{{"id":"grammar-probe","model_path":"{}","n_ctx":2048,"n_threads":8,"max_tokens":256,"no_think":true}}"#,
        normalized
    ))
    .unwrap();
    let mut engine = LlamaEngine::load(profile).unwrap();
    engine.warm().unwrap();

    let make = |contract: Option<&str>| InferenceRequest {
        id: 1,
        role: InferenceRole::NpcDecision,
        system: "You are Marta Voss, a guarded grocer. Reply with ONE JSON object only.".into(),
        prompt: "The player says: \"Show me your ledger.\" Decide what to do.".into(),
        max_tokens: 256,
        temperature: 0.2,
        seed: 7,
        response_contract: contract.map(str::to_string),
    };

    println!("== unconstrained ==");
    match engine.complete(&make(None)) {
        Ok(r) => println!(
            "text={:?}\ntokens={} structured={} prefill={:.0}ms decode={:.0}ms",
            r.text, r.completion_tokens, r.structured, r.prefill_ms, r.decode_ms
        ),
        Err(e) => println!("error: {e}"),
    }

    println!("\n== grammar-constrained (action-json-v1) ==");
    match engine.complete(&make(Some("action-json-v1"))) {
        Ok(r) => println!(
            "text={:?}\ntokens={} structured={} prefill={:.0}ms decode={:.0}ms",
            r.text, r.completion_tokens, r.structured, r.prefill_ms, r.decode_ms
        ),
        Err(e) => println!("error: {e}"),
    }
    println!("\nSURVIVED");
}
