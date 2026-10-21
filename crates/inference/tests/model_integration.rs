//! Real-GGUF regression tests.
//!
//! These require weights, so they are `#[ignore]`d and must be run explicitly:
//!
//! ```text
//! RPG_TEST_MODEL=.cache/models/Qwen3.5-0.8B-Q4_K_M.gguf \
//!   cargo test -p rpg_inference -- --ignored --nocapture
//! ```
//!
//! They cover the issues fixed on this branch: repeated schema-valid JSON tool
//! decisions, malformed grammars failing loudly, cancellation, and timeouts.
//! Ordinary `cargo test` does not need any weights.

use rpg_domain::ports::{InferenceEngine, InferenceError, InferenceRequest, InferenceRole};
use rpg_inference::{LlamaEngine, ModelProfile};
use std::sync::atomic::Ordering;
use std::time::Duration;

fn workspace_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

fn model_path() -> String {
    let raw = std::env::var("RPG_TEST_MODEL")
        .unwrap_or_else(|_| ".cache/models/Qwen3.5-0.8B-Q4_K_M.gguf".into());
    let path = std::path::PathBuf::from(&raw);
    if path.is_absolute() {
        raw
    } else {
        workspace_root().join(path).to_string_lossy().into_owned()
    }
}

fn profile(extra: &str) -> ModelProfile {
    let path = model_path().replace(char::from(92), "/");
    ModelProfile::from_json(&format!(
        r#"{{"id":"integration","model_path":"{path}","n_ctx":2048,"n_threads":8,"max_tokens":192,"no_think":true{extra}}}"#
    ))
    .unwrap()
}

fn engine(extra: &str) -> LlamaEngine {
    let p = profile(extra);
    assert!(
        std::path::Path::new(&p.model_path).exists(),
        "set RPG_TEST_MODEL to a GGUF file; {} is missing",
        p.model_path
    );
    let mut e = LlamaEngine::load(p).expect("load model");
    e.warm().expect("warm model");
    e
}

fn decision_request(id: u64, prompt: &str, contract: Option<&str>) -> InferenceRequest {
    InferenceRequest {
        id,
        role: InferenceRole::NpcDecision,
        system: "Reply with one JSON object only.".into(),
        prompt: prompt.into(),
        max_tokens: 256,
        temperature: 0.0,
        seed: id,
        response_contract: contract.map(str::to_string),
    }
}

#[test]
#[ignore = "requires RPG_TEST_MODEL (real GGUF weights)"]
fn repeated_decisions_are_schema_valid_json() {
    let mut e = engine("");
    for i in 0..5u64 {
        let req = decision_request(
            i + 1,
            &format!(
                "You are Marta Voss, a grocer. The player says: \"Show me the ledger.\" (turn {}). \
                 Reply with exactly one JSON object and no prose, for example \
                 {{\"action\":\"speak\",\"text\":\"short reply\"}} or \
                 {{\"action\":\"offer\",\"condition\":\"has_item:medicine\",\"note\":\"short note\"}}. \
                 Keep every string under 20 words.",
                i + 1
            ),
            Some("action-json-v1"),
        );
        let r = e.complete(&req).expect("completion");
        assert!(r.structured, "grammar must be applied");
        let v: serde_json::Value =
            serde_json::from_str(&r.text).expect("output must be valid JSON");
        assert!(v.get("action").is_some(), "missing action: {}", r.text);
        assert!(r.completion_tokens > 0);
        assert!(r.prompt_tokens > 0);
    }
}

#[test]
#[ignore = "requires RPG_TEST_MODEL (real GGUF weights)"]
fn malformed_grammar_fails_loudly() {
    let mut e = engine("");
    let req = decision_request(1, "Say something.", Some("root ::= ["));
    let err = e.complete(&req).expect_err("malformed grammar must fail");
    assert!(
        matches!(err, InferenceError::Backend(_)),
        "expected Backend error, got {err:?}"
    );
}

#[test]
#[ignore = "requires RPG_TEST_MODEL (real GGUF weights)"]
fn cancellation_stops_generation() {
    let mut e = engine("");
    let handle = e.cancel_handle();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(1500));
        handle.store(true, Ordering::SeqCst);
    });
    let mut req = decision_request(1, "Tell me a long story about the docks.", None);
    req.max_tokens = 1024;
    let err = e
        .complete(&req)
        .expect_err("cancellation must stop generation");
    assert!(
        matches!(err, InferenceError::Timeout(_)),
        "expected cancellation/timeout, got {err:?}"
    );
}

#[test]
#[ignore = "requires RPG_TEST_MODEL (real GGUF weights)"]
fn deadline_is_enforced() {
    // Do not warm: warm-up would itself hit the 1 ms deadline.
    let p = profile(r#","deadline_ms":1"#);
    let mut e = LlamaEngine::load(p).expect("load model");
    let req = decision_request(1, "Say something long.", None);
    let err = e.complete(&req).expect_err("deadline must expire");
    assert!(
        matches!(err, InferenceError::Timeout(1)),
        "expected Timeout(1), got {err:?}"
    );
}
