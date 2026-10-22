//! Opt-in, billable integration test against the real OpenRouter API.
//!
//! Never runs by default (`#[ignore]`), never in ordinary CI and never on
//! untrusted pull requests. Run it explicitly with a disposable,
//! spending-limited key:
//!
//! ```bash
//! OPENROUTER_API_KEY=sk-or-... \
//! IRL_OPENROUTER_MODEL=openai/gpt-6-luna \
//!   cargo test -p rpg_openrouter --test real_openrouter -- --ignored --nocapture
//! ```
//!
//! It uses a strict budget (at most 4 requests, 300 output tokens each),
//! drives a real gameplay turn through the shared canonical path, and prints
//! evidence (latency, tokens, reported vs estimated cost, provenance) with no
//! secret in it. The evidence is also written to
//! `target/openrouter-evidence.json`.

use rpg_domain::ports::InferenceEngine;
use rpg_openrouter::native::OpenRouterEngine;
use rpg_openrouter::{Budget, OpenRouterConfig, API_KEY_ENV};

#[test]
#[ignore = "billable: needs OPENROUTER_API_KEY and network"]
fn real_openrouter_turn_goes_through_canonical_validation() {
    let Ok(raw) = std::env::var(API_KEY_ENV) else {
        panic!("set {API_KEY_ENV} to run this opt-in test");
    };
    let model = std::env::var("IRL_OPENROUTER_MODEL")
        .unwrap_or_else(|_| rpg_openrouter::DEFAULT_MODEL.to_string());
    let cfg = OpenRouterConfig {
        model: model.clone(),
        max_output_tokens: 300,
        budget: Budget {
            max_requests: Some(4),
            max_cost_usd: Some(0.05),
            ..Budget::default()
        },
        ..OpenRouterConfig::default()
    };
    let key = rpg_openrouter::ApiKey::new(raw).expect("key");
    let mut engine = OpenRouterEngine::new(cfg, key);
    let t = std::time::Instant::now();
    engine.warm().expect("key + model verification");
    let verify_ms = t.elapsed().as_millis();
    let model_info = engine.model_info().cloned().expect("model info");
    assert!(
        model_info.supports_structured(),
        "{model} lacks structured outputs"
    );

    // 1. One direct request with the exact NPC input contract, to measure
    //    latency/tokens/cost and check the output against the shared schema.
    let probe = rpg_eval::Session::new(2026);
    let mut world = probe.sim.world.clone();
    let player = rpg_domain::ids::ActorId::new(rpg_domain::ids::PLAYER);
    let grocer = rpg_domain::ids::ActorId::new("grocer");
    if let (Some(pos), Some(loc)) = (
        world.actor(&grocer).map(|a| a.pos),
        world.actor(&grocer).map(|a| a.location.clone()),
    ) {
        let p = world.actor_mut(&player).unwrap();
        p.pos = pos;
        p.location = loc;
    }
    let req = rpg_agents::NpcPolicy.request(&world, &grocer, &player, "Good evening.", 1);
    let resp = engine.complete(&req).expect("a real completion");
    let contract = rpg_domain::contract::validate_action_json(&resp.text);
    assert!(contract.is_ok(), "contract: {contract:?} for {}", resp.text);
    let remote = resp.remote.clone().expect("provenance");
    let meter = engine.meter().clone();

    // 2. A real gameplay turn through the shared canonical path.
    let mut s = rpg_eval::Session::with_engine(2026, Box::new(engine));
    s.teleport_player_to("grocer");
    let before = s.sim.events.len();
    let t = std::time::Instant::now();
    let spoken = s.say(
        "grocer",
        "Evening, Marta. How has business been at the shop?",
    );
    let turn_ms = t.elapsed().as_millis();
    let (_, source) = s.decisions.last().cloned().expect("a decision");
    let after = s.sim.events.len();
    assert!(after > before, "the turn committed events");
    assert!(
        !matches!(source, rpg_agents::DecisionSource::HeuristicFallback { .. }),
        "the real model's answer should pass the contract: {}",
        source.label()
    );

    let evidence = serde_json::json!({
        "model_requested": model,
        "model_name": model_info.name,
        "structured_outputs": model_info.supports_structured(),
        "verify_ms": verify_ms,
        "probe": {
            "structured": resp.structured,
            "latency_ms": remote.latency_ms,
            "first_token_ms": resp.first_token_ms,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "reported_cost_usd": remote.reported_cost_usd,
            "estimated_cost_usd": remote.estimated_cost_usd,
            "served_model": remote.model,
            "provider": remote.provider,
            "generation_id": remote.generation_id,
            "streamed": remote.streamed,
            "raw_action": resp.text,
        },
        "session_requests_before_turn": meter.requests,
        "turn_ms": turn_ms,
        "decision_source": source.label(),
        "npc_line": spoken,
        "events_committed": after - before,
        "trace_tail": s.trace.iter().rev().take(3).collect::<Vec<_>>(),
    });
    let text = serde_json::to_string_pretty(&evidence).unwrap();
    assert!(!text.contains("sk-or"), "evidence must not contain the key");
    println!("{text}");
    let _ = std::fs::create_dir_all("../../target");
    let _ = std::fs::write("../../target/openrouter-evidence.json", &text);
}
