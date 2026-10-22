//! The platform-neutral parts of `SimCore` that the browser client relies
//! on: save export/import through the platform (no filesystem), platform
//! notices, streamed-progress reports, and session inference accounting
//! with remote provenance. Deterministic: no threads, no clocks.

use std::sync::Arc;
use std::time::Duration;

use rpg_domain::ports::{InferenceError, InferenceResponse, RemoteProvenance};
use rpg_eval::Session;
use rpg_runtime::*;

fn ms(v: u64) -> Duration {
    Duration::from_millis(v)
}

fn core() -> SimCore {
    let mut c = SimCore::new(
        Session::new(2026),
        CoreConfig::default(),
        Arc::new(Telemetry::new()),
    );
    c.mark_unavailable(true, true, true, "browser test");
    c.handle(
        Inbox::Infer(InferEvent::Ready {
            engine: "openrouter".into(),
            model_id: "test/structured".into(),
            is_real_model: true,
        }),
        ms(0),
    );
    c.take_effects();
    c
}

fn remote(text: &str, cost: f64) -> Result<InferenceResponse, InferenceError> {
    Ok(InferenceResponse {
        text: text.into(),
        prompt_tokens: 100,
        completion_tokens: 10,
        structured: true,
        remote: Some(RemoteProvenance {
            gateway: "openrouter".into(),
            requested_model: "test/structured".into(),
            model: Some("test/structured".into()),
            provider: Some("MockAI".into()),
            reported_cost_usd: Some(cost),
            estimated_cost_usd: Some(cost / 2.0),
            ..RemoteProvenance::default()
        }),
        ..Default::default()
    })
}

fn turn(c: &mut SimCore, npc: &str, text: &str, at: u64) -> InferJob {
    c.session.teleport_player_to(npc);
    c.handle(Inbox::Ui(Command::Say(text.into())), ms(at));
    assert_eq!(c.stage(), VoiceStage::Deciding);
    c.take_effects()
        .into_iter()
        .find_map(|e| match e {
            Effect::Infer(InferRequest::Decide(j)) => Some(j),
            _ => None,
        })
        .expect("inference request")
}

fn done(job: &InferJob, result: Result<InferenceResponse, InferenceError>) -> Inbox {
    Inbox::Infer(InferEvent::Done(InferDone {
        conv: job.conv,
        request_id: job.request_id,
        world_epoch: job.world_epoch,
        result,
        queue_ms: 0.0,
        infer_ms: 1234.0,
    }))
}

fn exported(c: &mut SimCore) -> String {
    c.handle(
        Inbox::Ui(Command::ExportSave {
            slot: "quick".into(),
        }),
        ms(5),
    );
    c.take_effects()
        .into_iter()
        .find_map(|e| match e {
            Effect::SaveData { slot, json } if slot == "quick" => Some(json),
            _ => None,
        })
        .expect("SaveData effect")
}

#[test]
fn export_produces_the_canonical_versioned_save_and_import_restores_it() {
    let mut c = core();
    for i in 1..=20 {
        c.tick(ms(i * 120));
    }
    let tick = c.session.sim.world.tick;
    let checksum = c.session.checksum();
    let json = exported(&mut c);
    let parsed = rpg_persistence::from_json_str(&json).expect("valid save");
    assert_eq!(parsed.version, rpg_persistence::SAVE_VERSION);
    assert_eq!(parsed.events.checksum(), checksum);

    for i in 21..=40 {
        c.tick(ms(i * 120));
    }
    assert!(c.session.sim.world.tick > tick);
    let epoch = c.world_epoch();
    c.handle(
        Inbox::Ui(Command::ImportSave {
            label: "browser storage".into(),
            json: json.clone(),
        }),
        ms(6000),
    );
    assert_eq!(c.session.sim.world.tick, tick);
    assert_eq!(c.session.checksum(), checksum);
    assert_eq!(c.world_epoch(), epoch + 1);
    assert!(c
        .dialogue()
        .iter()
        .any(|l| l.text.contains("loaded from browser storage")));

    // Deterministic restoration: both timelines continue identically.
    let mut a = core();
    a.handle(
        Inbox::Ui(Command::ImportSave {
            label: "a".into(),
            json: json.clone(),
        }),
        ms(0),
    );
    let mut b = core();
    b.handle(
        Inbox::Ui(Command::ImportSave {
            label: "b".into(),
            json,
        }),
        ms(0),
    );
    for i in 1..=50 {
        a.tick(ms(i * 120));
        b.tick(ms(i * 120));
    }
    assert_eq!(a.session.checksum(), b.session.checksum());
}

#[test]
fn import_rejects_bad_saves_without_touching_the_world() {
    let mut c = core();
    let before = c.session.checksum();
    let epoch = c.world_epoch();
    for (label, json) in [
        ("garbage", "not json at all".to_string()),
        ("future", r#"{"version":999}"#.to_string()),
        ("unversioned", r#"{"seed":1}"#.to_string()),
    ] {
        c.handle(
            Inbox::Ui(Command::ImportSave {
                label: label.into(),
                json,
            }),
            ms(0),
        );
        assert!(
            c.dialogue()
                .iter()
                .any(|l| l.text.contains(&format!("load failed ({label})"))),
            "{label}"
        );
    }
    assert_eq!(c.session.checksum(), before);
    assert_eq!(c.world_epoch(), epoch);
}

#[test]
fn importing_during_inference_discards_the_late_answer() {
    let mut c = core();
    let json = exported(&mut c);
    let job = turn(&mut c, "priest", "Hello.", 10);
    c.handle(
        Inbox::Ui(Command::ImportSave {
            label: "slot".into(),
            json,
        }),
        ms(20),
    );
    assert_eq!(c.stage(), VoiceStage::Idle);
    c.handle(
        done(&job, remote(r#"{"action":"speak","text":"Old."}"#, 0.01)),
        ms(30),
    );
    assert_eq!(c.counters().stale_responses, 1);
    assert!(!c.dialogue().iter().any(|l| l.text == "Old."));
    // The answer still cost something and is accounted for.
    assert_eq!(c.inference_stats().completed, 1);
}

#[test]
fn exporting_during_inference_keeps_the_turn_alive() {
    let mut c = core();
    let job = turn(&mut c, "priest", "Hello.", 0);
    let json = exported(&mut c);
    assert!(rpg_persistence::from_json_str(&json).is_ok());
    assert_eq!(c.stage(), VoiceStage::Deciding);
    c.handle(
        done(
            &job,
            remote(r#"{"action":"speak","text":"Still here."}"#, 0.0),
        ),
        ms(50),
    );
    assert!(c.dialogue().iter().any(|l| l.text == "Still here."));
}

#[test]
fn notices_show_in_the_dialogue_log() {
    let mut c = core();
    c.handle(
        Inbox::Ui(Command::Notice("saved to browser storage".into())),
        ms(0),
    );
    assert_eq!(
        c.dialogue().last().map(|l| (l.kind, l.text.as_str())),
        Some((LineKind::System, "saved to browser storage"))
    );
}

#[test]
fn accounting_keeps_reported_and_estimated_cost_apart() {
    let mut c = core();
    let job = turn(&mut c, "priest", "Hello.", 0);
    c.handle(
        done(&job, remote(r#"{"action":"speak","text":"Peace."}"#, 0.002)),
        ms(100),
    );
    let job = turn(&mut c, "priest", "Again.", 200);
    c.handle(
        done(
            &job,
            Err(InferenceError::Backend("OpenRouter rate limited".into())),
        ),
        ms(300),
    );
    let s = c.inference_stats().clone();
    assert_eq!((s.requests, s.completed, s.errors), (2, 1, 1));
    assert_eq!((s.prompt_tokens, s.completion_tokens), (100, 10));
    assert_eq!(s.reported_cost_usd, Some(0.002));
    assert_eq!(s.estimated_cost_usd, Some(0.001));
    assert_eq!(s.last_latency_ms, Some(1234.0));
    assert!(s.last_remote.as_deref().unwrap().contains("MockAI"));
    assert!(s.last_error.as_deref().unwrap().contains("rate limited"));
    let snap = c.snapshot();
    assert_eq!(snap.diagnostics.inference, s);
    assert!(snap
        .diagnostics
        .last_decision
        .as_deref()
        .unwrap()
        .contains("heuristic-fallback"));
    assert_eq!(snap.model_decisions, 1);
    assert_eq!(snap.fallback_decisions, 1);
}

#[test]
fn progress_is_a_count_only_and_ignored_when_stale() {
    let mut c = core();
    let job = turn(&mut c, "priest", "Hello.", 0);
    c.handle(
        Inbox::Infer(InferEvent::Progress {
            conv: job.conv,
            request_id: job.request_id,
            chunks: 3,
        }),
        ms(10),
    );
    assert!(c.snapshot().status.contains("answer arriving"));
    assert_eq!(c.snapshot().diagnostics.llm_stream_chunks, Some(3));
    // Nothing was committed or shown from the partial answer.
    assert_eq!(c.stage(), VoiceStage::Deciding);
    assert!(!c.dialogue().iter().any(|l| l.kind == LineKind::Npc));

    c.handle(Inbox::Ui(Command::CancelVoice), ms(20));
    c.handle(
        Inbox::Infer(InferEvent::Progress {
            conv: job.conv,
            request_id: job.request_id,
            chunks: 9,
        }),
        ms(30),
    );
    assert!(!c.snapshot().status.contains("answer arriving"));
}

#[test]
fn delayed_duplicate_and_cancelled_answers_commit_at_most_once() {
    let mut c = core();
    let job = turn(&mut c, "priest", "Hello.", 0);
    // The world keeps ticking for 30 simulated seconds of remote latency.
    for i in 1..=250 {
        c.tick(ms(i * 120));
    }
    assert_eq!(c.stage(), VoiceStage::Deciding);
    let answer = || remote(r#"{"action":"speak","text":"Finally."}"#, 0.0);
    c.handle(done(&job, answer()), ms(30_100));
    c.handle(done(&job, answer()), ms(30_200));
    let lines = c.dialogue().iter().filter(|l| l.text == "Finally.").count();
    assert_eq!(lines, 1);
    assert_eq!(c.counters().stale_responses, 1);

    // Cancel, then the answer arrives: nothing is committed.
    let job = turn(&mut c, "priest", "One more thing.", 31_000);
    c.handle(Inbox::Ui(Command::CancelVoice), ms(31_010));
    let events = c.session.sim.events.len();
    c.handle(done(&job, answer()), ms(31_500));
    assert_eq!(c.session.sim.events.len(), events);
    assert_eq!(c.counters().stale_responses, 2);
}
