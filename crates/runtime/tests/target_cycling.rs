//! Regression tests for `Command::CycleTarget`.
//!
//! The original implementation used an infinite iterator
//! (`candidates.iter().cycle().skip_while(...).nth(1)`) that never terminated
//! when the player had no active target or when the current target was no
//! longer in the candidate list. These tests drive the real `SimCore::handle`
//! path and assert that every command returns and the simulation keeps
//! ticking afterwards.

use std::sync::Arc;
use std::time::Duration;

use rpg_domain::geom::Vec2;
use rpg_domain::ids::{ActorId, PLAYER};
use rpg_domain::ports::{InferenceError, InferenceResponse};
use rpg_domain::world::ActorStatus;
use rpg_eval::Session;
use rpg_runtime::*;

fn ms(v: u64) -> Duration {
    Duration::from_millis(v)
}

/// A core with every component reported ready and a "real" model.
fn ready_core() -> SimCore {
    let mut c = SimCore::new(
        Session::new(2026),
        CoreConfig::default(),
        Arc::new(Telemetry::new()),
    );
    c.handle(
        Inbox::Infer(InferEvent::Ready {
            engine: "fake".into(),
            model_id: "fake-model".into(),
            is_real_model: true,
        }),
        ms(0),
    );
    c.handle(
        Inbox::Mic(MicEvent::DeviceReady(MicInfo {
            name: "mic".into(),
            sample_rate: 48_000,
            channels: 2,
            sample_format: "F32".into(),
        })),
        ms(0),
    );
    c.handle(
        Inbox::Asr(AsrEvent::Ready {
            backend: "asr".into(),
        }),
        ms(0),
    );
    c.handle(
        Inbox::Tts(TtsEvent::Ready {
            backend: "tts".into(),
            device: "spk".into(),
            sample_rate: 48_000,
        }),
        ms(0),
    );
    c.take_effects();
    c
}

fn talking_to(c: &SimCore) -> Option<String> {
    c.snapshot().talking_to
}

/// Put the player in `anchor`'s location and arrange `others` within the talk
/// radius so `nearby_all` returns a controlled candidate set.
fn stage(c: &mut SimCore, anchor: &str, others: &[&str]) {
    c.session.teleport_player_to(anchor);
    let player_id = ActorId::new(PLAYER);
    let (pos, loc) = {
        let p = c.session.sim.world.actor(&player_id).unwrap();
        (p.pos, p.location.clone())
    };
    for (i, npc) in others.iter().enumerate() {
        if let Some(a) = c.session.sim.world.actor_mut(&ActorId::new(*npc)) {
            a.pos = Vec2::new(pos.x + (i as f32 + 1.0) * 0.5, pos.y);
            a.location = loc.clone();
            a.status = ActorStatus::Alive;
        }
    }
}

/// Move an actor far away so it no longer qualifies as a nearby candidate.
fn banish(c: &mut SimCore, npc: &str) {
    if let Some(a) = c.session.sim.world.actor_mut(&ActorId::new(npc)) {
        a.location = rpg_domain::ids::LocationId::new("nowhere");
        a.pos = Vec2::new(10_000.0, 10_000.0);
    }
}

fn kill(c: &mut SimCore, npc: &str) {
    if let Some(a) = c.session.sim.world.actor_mut(&ActorId::new(npc)) {
        a.status = ActorStatus::Dead;
    }
}

fn cycle(c: &mut SimCore, at: u64) {
    c.handle(Inbox::Ui(Command::CycleTarget), ms(at));
}

fn select(c: &mut SimCore, npc: &str, at: u64) {
    c.handle(Inbox::Ui(Command::SelectTarget(ActorId::new(npc))), ms(at));
}

fn assert_ticks(c: &mut SimCore, start: u64, now: u64) {
    let before = c.snapshot().tick;
    c.tick(ms(now));
    assert_eq!(
        c.snapshot().tick,
        before + 1,
        "simulation did not advance a tick after the command"
    );
    assert!(c.snapshot().tick > start);
}

#[test]
fn cycle_target_with_no_candidates_is_safe_and_keeps_ticking() {
    let mut c = ready_core();
    // Move every NPC out of the player's reach.
    for npc in [
        "grocer",
        "priest",
        "guard_captain",
        "guard_rookie",
        "merchant_head",
        "spy",
        "tavern_keeper",
        "dockworker",
        "courier",
        "mafia_lieutenant",
        "accountant",
        "neighbor",
        "child",
    ] {
        banish(&mut c, npc);
    }
    let start = c.snapshot().tick;
    cycle(&mut c, 10);
    assert_eq!(talking_to(&c), None);
    // The user is told why, not left hanging.
    assert!(c
        .dialogue()
        .iter()
        .any(|l| l.text.contains("nobody is close enough")));
    assert_ticks(&mut c, start, 20);
}

#[test]
fn cycle_target_with_one_candidate_stays_on_it() {
    let mut c = ready_core();
    stage(&mut c, "priest", &["priest"]);
    let start = c.snapshot().tick;
    cycle(&mut c, 10);
    assert_eq!(talking_to(&c).as_deref(), Some("priest"));
    // Cycling again with a single candidate wraps to itself and returns.
    cycle(&mut c, 20);
    assert_eq!(talking_to(&c).as_deref(), Some("priest"));
    assert_ticks(&mut c, start, 30);
}

#[test]
fn cycle_target_cycles_multiple_and_wraps() {
    let mut c = ready_core();
    // Place three NPCs in the same room, all in range.
    stage(&mut c, "priest", &["priest", "spy", "tavern_keeper"]);
    let order = c.nearby_all();
    assert!(
        order.len() >= 3,
        "expected at least three candidates, got {order:?}"
    );
    let expected: Vec<String> = order.iter().map(|a| a.as_str().to_string()).collect();

    // First cycle: no current target -> first candidate.
    cycle(&mut c, 10);
    assert_eq!(talking_to(&c).as_deref(), Some(expected[0].as_str()));

    // Each subsequent cycle advances one position and wraps at the end.
    for i in 1..=expected.len() {
        cycle(&mut c, 20 + i as u64);
        let want = expected[i % expected.len()].as_str();
        assert_eq!(
            talking_to(&c).as_deref(),
            Some(want),
            "cycle step {i} selected the wrong target"
        );
    }
}

#[test]
fn cycle_target_falls_back_when_current_absent() {
    let mut c = ready_core();
    stage(&mut c, "priest", &["priest", "spy", "tavern_keeper"]);
    // Select the spy explicitly, then make it vanish from the candidate set.
    select(&mut c, "spy", 10);
    assert_eq!(talking_to(&c).as_deref(), Some("spy"));
    banish(&mut c, "spy");
    let start = c.snapshot().tick;
    cycle(&mut c, 20);
    let chosen = talking_to(&c).expect("a fallback target should be chosen");
    assert_ne!(chosen, "spy", "the absent target must not be selected");
    assert!(c.nearby_all().iter().any(|a| a.as_str() == chosen));
    assert_ticks(&mut c, start, 30);
}

#[test]
fn cycle_target_skips_dead_npc() {
    let mut c = ready_core();
    stage(&mut c, "priest", &["priest", "spy"]);
    select(&mut c, "spy", 10);
    kill(&mut c, "spy");
    let start = c.snapshot().tick;
    cycle(&mut c, 20);
    let chosen = talking_to(&c).expect("a living target should be chosen");
    assert_ne!(chosen, "spy", "a dead NPC must not be selected");
    assert_ticks(&mut c, start, 30);
}

#[test]
fn cycle_target_wraps_last_to_first() {
    let mut c = ready_core();
    stage(&mut c, "priest", &["priest", "spy", "tavern_keeper"]);
    let order: Vec<String> = c
        .nearby_all()
        .iter()
        .map(|a| a.as_str().to_string())
        .collect();
    let last = order.last().unwrap().clone();
    select(&mut c, &last, 10);
    assert_eq!(talking_to(&c).as_deref(), Some(last.as_str()));
    cycle(&mut c, 20);
    assert_eq!(
        talking_to(&c).as_deref(),
        Some(order[0].as_str()),
        "cycling from the last candidate must wrap to the first"
    );
}

#[test]
fn explicit_select_overrides_and_cycle_continues_from_there() {
    let mut c = ready_core();
    stage(&mut c, "priest", &["priest", "spy", "tavern_keeper"]);
    let order: Vec<String> = c
        .nearby_all()
        .iter()
        .map(|a| a.as_str().to_string())
        .collect();
    // Select the second candidate explicitly.
    let second = order.get(1).expect("need at least two candidates").clone();
    select(&mut c, &second, 10);
    assert_eq!(talking_to(&c).as_deref(), Some(second.as_str()));
    // A cycle continues from the explicitly selected position.
    cycle(&mut c, 20);
    let want = order.get(2).unwrap_or(&order[0]).clone();
    assert_eq!(talking_to(&c).as_deref(), Some(want.as_str()));
}

#[test]
fn end_conversation_then_cycle_restarts_at_first() {
    let mut c = ready_core();
    stage(&mut c, "priest", &["priest", "spy", "tavern_keeper"]);
    let order: Vec<String> = c
        .nearby_all()
        .iter()
        .map(|a| a.as_str().to_string())
        .collect();
    select(&mut c, order.last().unwrap(), 10);
    c.handle(Inbox::Ui(Command::EndConversation), ms(20));
    assert_eq!(talking_to(&c), None);
    let start = c.snapshot().tick;
    cycle(&mut c, 30);
    assert_eq!(talking_to(&c).as_deref(), Some(order[0].as_str()));
    assert_ticks(&mut c, start, 40);
}

#[test]
fn repeated_cycling_never_hangs_and_always_returns() {
    let mut c = ready_core();
    stage(&mut c, "priest", &["priest", "spy", "tavern_keeper"]);
    let start = c.snapshot().tick;
    // Hammer the command; each call must return promptly (the test finishing
    // is the proof it does not spin forever).
    for i in 0..50 {
        cycle(&mut c, 10 + i);
    }
    assert_ticks(&mut c, start, 100);
}

// ---------------------------------------------------------------------------
// Lifecycle hardening: target changes during an in-flight inference
// ---------------------------------------------------------------------------

fn speak(text: &str) -> Result<InferenceResponse, InferenceError> {
    Ok(InferenceResponse {
        text: format!(r#"{{"action":"speak","text":"{text}"}}"#),
        completion_tokens: 5,
        ..Default::default()
    })
}

fn infer_job(effects: &[Effect]) -> InferJob {
    effects
        .iter()
        .find_map(|e| match e {
            Effect::Infer(InferRequest::Decide(j)) => Some(j.clone()),
            _ => None,
        })
        .expect("expected an inference request")
}

fn done(job: &InferJob, result: Result<InferenceResponse, InferenceError>) -> Inbox {
    Inbox::Infer(InferEvent::Done(InferDone {
        conv: job.conv,
        request_id: job.request_id,
        world_epoch: job.world_epoch,
        result,
        queue_ms: 1.0,
        infer_ms: 2.0,
    }))
}

fn typed_turn(c: &mut SimCore, npc: &str, text: &str, at: u64) -> InferJob {
    c.session.teleport_player_to(npc);
    c.handle(Inbox::Ui(Command::Say(text.into())), ms(at));
    infer_job(&c.take_effects())
}

fn npc_lines(c: &SimCore) -> Vec<String> {
    c.dialogue()
        .iter()
        .filter(|l| l.kind == LineKind::Npc)
        .map(|l| l.text.clone())
        .collect()
}

#[test]
fn switching_target_cancels_pending_inference() {
    let mut c = ready_core();
    stage(&mut c, "priest", &["priest", "spy"]);
    select(&mut c, "priest", 10);
    let _job = typed_turn(&mut c, "priest", "Hello.", 20);
    let cancelled_before = c.counters().cancelled;
    // Switch to the spy while the priest's request is still pending.
    select(&mut c, "spy", 30);
    assert!(
        c.counters().cancelled > cancelled_before,
        "switching targets must cancel the pending request"
    );
    assert_eq!(talking_to(&c).as_deref(), Some("spy"));
    let fx = c.take_effects();
    assert!(
        fx.iter()
            .any(|e| matches!(e, Effect::Infer(InferRequest::Cancel { .. }))),
        "a cancel effect must be sent to the inference worker"
    );
}

#[test]
fn ending_conversation_cancels_pending_inference() {
    let mut c = ready_core();
    stage(&mut c, "priest", &["priest"]);
    select(&mut c, "priest", 10);
    let _job = typed_turn(&mut c, "priest", "Hello.", 20);
    let cancelled_before = c.counters().cancelled;
    c.handle(Inbox::Ui(Command::EndConversation), ms(30));
    assert!(
        c.counters().cancelled > cancelled_before,
        "ending the conversation must cancel the pending request"
    );
    assert_eq!(talking_to(&c), None);
    let fx = c.take_effects();
    assert!(
        fx.iter()
            .any(|e| matches!(e, Effect::Infer(InferRequest::Cancel { .. }))),
        "a cancel effect must be sent"
    );
}

#[test]
fn cycling_target_cancels_pending_inference() {
    let mut c = ready_core();
    stage(&mut c, "priest", &["priest", "spy", "tavern_keeper"]);
    select(&mut c, "priest", 10);
    let _job = typed_turn(&mut c, "priest", "Hello.", 20);
    let cancelled_before = c.counters().cancelled;
    cycle(&mut c, 30);
    let new_target = talking_to(&c);
    assert!(new_target.is_some());
    assert_ne!(
        new_target.as_deref(),
        Some("priest"),
        "cycling must move away from the current target"
    );
    assert!(
        c.counters().cancelled > cancelled_before,
        "cycling to a different target must cancel the pending request"
    );
}

#[test]
fn late_response_after_switch_is_stale_and_not_voiced() {
    let mut c = ready_core();
    stage(&mut c, "priest", &["priest", "spy"]);
    select(&mut c, "priest", 10);
    let job = typed_turn(&mut c, "priest", "Hello.", 20);
    // Switch to the spy, cancelling the priest's request.
    select(&mut c, "spy", 30);
    let _ = c.take_effects();
    let stale_before = c.counters().stale_responses;
    let lines_before = npc_lines(&c).len();
    // The priest's response arrives late.
    c.handle(done(&job, speak("I'm still here.")), ms(40));
    // The response must not be voiced.
    assert_eq!(
        npc_lines(&c).len(),
        lines_before,
        "a cancelled request's late answer must not be voiced"
    );
    assert!(
        c.counters().stale_responses > stale_before,
        "the late answer must be counted as stale"
    );
}
