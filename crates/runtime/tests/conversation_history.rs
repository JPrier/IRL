//! Regression tests for conversation-history propagation and boundaries.
//!
//! Review finding #7: the decision contexts (`NpcPolicy.resolve` and the
//! heuristic fallback) were built with `history: &[]`, so a canonical bargain
//! follow-up was resolved without the conversation that produced it. The
//! history now comes from the active conversation and obeys its boundaries:
//!
//! * a follow-up turn carries the prior exchange (multi-turn bargains work);
//! * the current utterance is never duplicated into the history;
//! * switching/restarting a conversation starts empty (no inheritance);
//! * loading a save ends the conversation (no cross-timeline leak).

use std::sync::Arc;
use std::time::Duration;

use rpg_domain::geom::Vec2;
use rpg_domain::ids::{ActorId, FactId, ItemId, OfferId, PLAYER};
use rpg_domain::ports::{InferenceError, InferenceResponse};
use rpg_domain::world::ActorStatus;
use rpg_eval::Session;
use rpg_runtime::*;

fn ms(v: u64) -> Duration {
    Duration::from_millis(v)
}

/// A core with every component reported ready. `real` selects a "real" model
/// (async inference jobs) or the deterministic policy (synchronous fallback
/// path in `request_decision`).
fn ready_core(real: bool) -> SimCore {
    let mut c = SimCore::new(
        Session::new(2026),
        CoreConfig::default(),
        Arc::new(Telemetry::new()),
    );
    c.handle(
        Inbox::Infer(InferEvent::Ready {
            engine: "fake".into(),
            model_id: "fake-model".into(),
            is_real_model: real,
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

fn near(c: &mut SimCore, npc: &str) {
    c.session.teleport_player_to(npc);
}

/// Put the player in `anchor`'s location with `others` ranged around it.
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

fn speak(text: &str) -> Result<InferenceResponse, InferenceError> {
    Ok(InferenceResponse {
        text: format!(r#"{{"action":"speak","text":"{text}"}}"#),
        completion_tokens: 5,
        ..Default::default()
    })
}

fn infer_job(effects: &[Effect]) -> Option<InferJob> {
    effects.iter().find_map(|e| match e {
        Effect::Infer(InferRequest::Decide(j)) => Some(j.clone()),
        _ => None,
    })
}

fn expect_infer_job(effects: &[Effect]) -> InferJob {
    infer_job(effects).expect("expected an inference request")
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

/// Complete the spoken line `line` of `conv` so the turn returns to idle and
/// the next typed turn is accepted.
fn finish_line(c: &mut SimCore, conv: u64, line: u64, at: u64) {
    c.handle(
        Inbox::Tts(TtsEvent::Finished {
            conv,
            line,
            audio_seconds: 0.5,
            synth_ms: 1.0,
        }),
        ms(at),
    );
    let _ = c.take_effects();
}

fn grocer_offer(c: &SimCore) -> OfferId {
    c.session
        .sim
        .world
        .offers
        .values()
        .find(|o| o.from.as_str() == "grocer")
        .map(|o| o.id)
        .expect("the grocer should have made a conditional offer")
}

fn player_heard_ledger(c: &SimCore) -> bool {
    c.session
        .sim
        .world
        .has_heard(&ActorId::new(PLAYER), &FactId::new("f_grocer_ledger"))
}

fn npc_lines(c: &SimCore) -> Vec<String> {
    c.dialogue()
        .iter()
        .filter(|l| l.kind == LineKind::Npc)
        .map(|l| l.text.clone())
        .collect()
}

// ---------------------------------------------------------------------------
// Multi-turn canonical bargain through the realtime runtime
// ---------------------------------------------------------------------------

/// The full bargain loop across two turns: the canonical agenda turns a
/// stalling model into a conditional offer, the player accepts and delivers,
/// and the follow-up ("what about it?") is honoured through the canonical
/// obligation at *resolution* time — which used to run with `history: &[]`.
#[test]
fn multi_turn_bargain_honours_follow_up_after_delivery() {
    let mut c = ready_core(true);
    near(&mut c, "grocer");

    // Turn 1: ask for the ledger. A first contact has no history.
    c.handle(
        Inbox::Ui(Command::Say(
            "I need to see your ledger. The one with the dock payments.".into(),
        )),
        ms(10),
    );
    let job1 = expect_infer_job(&c.take_effects());
    assert!(
        !job1.request.prompt.contains("Recent conversation"),
        "a first contact must start with an empty history"
    );
    // The model stalls ("speak"); the canonical agenda must replace it.
    c.handle(done(&job1, speak("Hmm. Maybe.")), ms(20));
    let offer_id = grocer_offer(&c);
    {
        let offer = &c.session.sim.world.offers[&offer_id];
        assert!(
            offer.disclose.as_ref().map(|f| f.as_str()) == Some("f_grocer_ledger"),
            "the offer must promise the ledger fact"
        );
    }
    finish_line(&mut c, job1.conv, 1, 30);

    // Accept the deal and deliver the medicine.
    c.handle(Inbox::Ui(Command::AcceptOffer(offer_id)), ms(40));
    c.session.give("medicine", 1);
    let gave = c
        .session
        .player_do(rpg_domain::action::Action::TransferItem {
            target: ActorId::new("grocer"),
            item: ItemId::new("medicine"),
            qty: 1,
        });
    assert!(
        matches!(gave, rpg_domain::action::ValidationOutcome::Accepted),
        "medicine delivery must be accepted: {gave:?}"
    );
    c.tick(ms(50));
    assert!(
        c.session
            .sim
            .world
            .commitments
            .values()
            .any(|cm| cm.state == rpg_domain::social::CommitmentState::Satisfied),
        "delivery must satisfy the deal"
    );

    // Turn 2: a vague follow-up. The prompt must carry the prior exchange.
    near(&mut c, "grocer");
    c.handle(
        Inbox::Ui(Command::Say("Now the ledger, as we agreed.".into())),
        ms(60),
    );
    let job2 = expect_infer_job(&c.take_effects());
    let prompt = &job2.request.prompt;
    assert!(
        prompt.contains("Recent conversation"),
        "the follow-up prompt must include the conversation history"
    );
    assert!(
        prompt.contains("I need to see your ledger"),
        "the prior player turn must be in the history"
    );
    assert!(
        prompt.contains("Bring me medicine"),
        "the NPC's committed offer line must be in the history"
    );
    assert_eq!(
        prompt.matches("Now the ledger, as we agreed.").count(),
        1,
        "the current utterance must appear exactly once (not duplicated in history)"
    );

    // The model stalls again; the canonical obligation honours the deal.
    c.handle(done(&job2, speak("It's a deal then.")), ms(70));
    assert!(
        player_heard_ledger(&c),
        "the satisfied deal must be honoured on the follow-up turn"
    );
    let ledger_line = npc_lines(&c).iter().any(|l| l.contains("ledger"));
    assert!(ledger_line, "the NPC must have spoken the ledger statement");
}

/// The deterministic fallback path (no real model) also runs the full bargain
/// loop; its decision context carries the same conversation history.
#[test]
fn fallback_policy_completes_the_multi_turn_bargain() {
    let mut c = ready_core(false);
    near(&mut c, "grocer");

    // Turn 1: the heuristic offers the designed conditional bargain.
    c.handle(
        Inbox::Ui(Command::Say("I need to see your ledger.".into())),
        ms(10),
    );
    let fx = c.take_effects();
    assert!(
        infer_job(&fx).is_none(),
        "the fallback path must decide synchronously, not via inference"
    );
    let offer_id = grocer_offer(&c);
    finish_line(&mut c, 1, 1, 20);

    // Accept and deliver.
    c.handle(Inbox::Ui(Command::AcceptOffer(offer_id)), ms(30));
    c.session.give("medicine", 1);
    let gave = c
        .session
        .player_do(rpg_domain::action::Action::TransferItem {
            target: ActorId::new("grocer"),
            item: ItemId::new("medicine"),
            qty: 1,
        });
    assert!(matches!(
        gave,
        rpg_domain::action::ValidationOutcome::Accepted
    ));
    c.tick(ms(40));

    // Turn 2: the honoured bargain discloses the ledger.
    near(&mut c, "grocer");
    c.handle(Inbox::Ui(Command::Say("Now the ledger?".into())), ms(50));
    assert!(player_heard_ledger(&c));
}

// ---------------------------------------------------------------------------
// Boundaries: unrelated conversations must not inherit history
// ---------------------------------------------------------------------------

/// Selecting a different NPC starts a fresh conversation: the new addressee's
/// prompt carries no trace of the previous NPC's exchange.
#[test]
fn switching_to_another_npc_does_not_inherit_history() {
    let mut c = ready_core(true);
    stage(&mut c, "priest", &["priest", "spy"]);

    c.handle(
        Inbox::Ui(Command::SelectTarget(ActorId::new("priest"))),
        ms(10),
    );
    c.handle(
        Inbox::Ui(Command::Say("purple penguins parade".into())),
        ms(20),
    );
    let job1 = expect_infer_job(&c.take_effects());
    c.handle(done(&job1, speak("The crypt is quiet tonight.")), ms(30));
    finish_line(&mut c, job1.conv, 1, 40);

    // Switch to the spy and ask a new question.
    c.handle(
        Inbox::Ui(Command::SelectTarget(ActorId::new("spy"))),
        ms(50),
    );
    assert_eq!(c.snapshot().talking_to.as_deref(), Some("spy"));
    c.handle(
        Inbox::Ui(Command::Say("Do you watch the docks?".into())),
        ms(60),
    );
    let job2 = expect_infer_job(&c.take_effects());
    let prompt = &job2.request.prompt;
    assert!(
        !prompt.contains("purple penguins parade"),
        "the spy's prompt must not inherit the priest's conversation"
    );
    assert!(
        !prompt.contains("The crypt is quiet tonight."),
        "the spy's prompt must not contain another NPC's reply"
    );
    assert!(
        !prompt.contains("Recent conversation"),
        "a new conversation starts with an empty history"
    );
}

/// Ending a conversation (X / EndConversation) resets its memory: talking to
/// the same NPC again starts empty rather than continuing the old exchange.
#[test]
fn ended_conversation_does_not_inherit_history() {
    let mut c = ready_core(true);
    stage(&mut c, "priest", &["priest", "spy"]);

    c.handle(
        Inbox::Ui(Command::Say("purple penguins parade".into())),
        ms(10),
    );
    let job1 = expect_infer_job(&c.take_effects());
    c.handle(done(&job1, speak("The crypt is quiet tonight.")), ms(20));
    finish_line(&mut c, job1.conv, 1, 30);

    c.handle(Inbox::Ui(Command::EndConversation), ms(40));
    assert_eq!(c.snapshot().talking_to, None);

    c.handle(
        Inbox::Ui(Command::Say("Do you watch the docks?".into())),
        ms(50),
    );
    let job2 = expect_infer_job(&c.take_effects());
    let prompt = &job2.request.prompt;
    assert!(
        !prompt.contains("purple penguins parade"),
        "a restarted conversation must not inherit its predecessor's turns"
    );
    assert!(
        !prompt.contains("The crypt is quiet tonight."),
        "a restarted conversation must not inherit prior replies"
    );
}

/// Loading a save ends the conversation: its turns belong to the old timeline
/// and must never leak into the restored world's prompts (an imported save may
/// even come from a different session entirely).
#[test]
fn loading_a_save_does_not_inherit_conversation_history() {
    let mut c = ready_core(true);
    near(&mut c, "priest");
    c.handle(
        Inbox::Ui(Command::Say("purple penguins parade".into())),
        ms(10),
    );
    let job1 = expect_infer_job(&c.take_effects());
    c.handle(done(&job1, speak("The crypt is quiet tonight.")), ms(20));
    finish_line(&mut c, job1.conv, 1, 30);
    assert_eq!(c.snapshot().talking_to.as_deref(), Some("priest"));

    let path = std::env::temp_dir().join("irl_runtime_conversation_history_test.json");
    let _ = std::fs::remove_file(&path);
    c.handle(
        Inbox::Ui(Command::Save(path.to_string_lossy().into_owned())),
        ms(40),
    );
    c.handle(
        Inbox::Ui(Command::Load(path.to_string_lossy().into_owned())),
        ms(50),
    );
    assert!(
        c.dialogue().iter().any(|l| l.text.contains("loaded from")),
        "the load must be reported"
    );
    // The conversation ended with the old timeline.
    assert_eq!(
        c.snapshot().talking_to,
        None,
        "loading a save must end the active conversation"
    );

    // A fresh turn next to the same NPC starts with an empty history.
    c.handle(
        Inbox::Ui(Command::Say("Do you watch the docks?".into())),
        ms(60),
    );
    let job2 = expect_infer_job(&c.take_effects());
    let prompt = &job2.request.prompt;
    assert!(
        !prompt.contains("purple penguins parade"),
        "the restored world must not inherit pre-save conversation history"
    );
    assert!(
        !prompt.contains("The crypt is quiet tonight."),
        "the restored world must not inherit pre-save replies"
    );
    assert_eq!(c.snapshot().talking_to.as_deref(), Some("priest"));
    let _ = std::fs::remove_file(&path);
}
