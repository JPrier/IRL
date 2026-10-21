use rpg_domain::ports::{InferenceEngine, InferenceRequest, InferenceRole};
use rpg_inference::{LlamaEngine, ModelProfile};

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| ".cache/models/Qwen3.5-0.8B-Q4_K_M.gguf".into());
    let normalized = path.replace(std::path::MAIN_SEPARATOR, "/");
    let profile = ModelProfile::from_json(&format!(
        r#"{{"id":"qwen35-0.8b-q4","model_path":"{}","n_ctx":2048,"n_threads":8,"max_tokens":96}}"#,
        normalized
    ))
    .unwrap();
    let mut engine = LlamaEngine::load(profile).unwrap();
    engine.warm().unwrap();
    let req = InferenceRequest {
        id: 1,
        role: InferenceRole::NpcDecision,
        system: "You are Marta Voss, a guarded grocer whose daughter is ill. Reply with ONE JSON object only.".into(),
        prompt: "The player says: \"Show me your ledger.\" Respond with {\"action\":\"speak\",\"text\":\"...\"} or {\"action\":\"offer\",\"want\":[\"medicine\"],\"note\":\"...\"}.".into(),
        max_tokens: 96,
        temperature: 0.4,
        seed: 7,
        response_contract: None,
    };
    let r = engine.complete(&req).unwrap();
    println!("TEXT: {}", r.text);
    println!(
        "tokens={} prefill={:.0}ms decode={:.0}ms first={:.0}ms",
        r.completion_tokens, r.prefill_ms, r.decode_ms, r.first_token_ms
    );
}
