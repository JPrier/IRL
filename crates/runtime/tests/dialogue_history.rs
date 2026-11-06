//! Regression test for the dialogue-history duplication bug.
//!
//! The player's latest utterance must be passed to the model prompt as the
//! *current* line and must not also appear as a past turn in the "Recent
//! conversation" section. The fix in `request_decision` excludes the last
//! turn (`ac.turns.len() - 1`) from the history it forwards.

use std::sync::Arc;
use std::time::Duration;

use rpg_domain::ports::{InferenceError, InferenceResponse};
use rpg_eval::Session;
use rpg_runtime::*;

fn ms(v: u64) -> Duration {
    Duration::from_millis(v)
}

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

fn near(c: &mut SimCore, npc: &str) {
    c.session.teleport_player_to(npc);
}

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

fn typed_turn(c: &mut SimCore, npc: &str, text: &str, now: u64) -> InferJob {
    near(c, npc);
    c.handle(Inbox::Ui(Command::Say(text.into())), ms(now));
    infer_job(&c.take_effects())
}

#[test]
fn current_utterance_appears_exactly_once_in_prompt() {
    let mut c = ready_core();

    // Turn 1: the player greets the priest.
    let job1 = typed_turn(&mut c, "priest", "Hello there!", 10);
    c.handle(done(&job1, speak("Good day to you.")), ms(20));
    // Let the turn complete (voice the line and return to idle).
    let _ = c.take_effects();
    // Simulate TTS completion so the NPC's turn is recorded in history.
    // (If TTS does not finish, the next typed turn is refused; we drain by
    // driving the turn to Idle via a fresh typed turn only after it settles.)
    // The core records the NPC's turn when the decision is committed; the
    // active conversation holds both turns by the time turn 2 is requested.
    for t in 21..40 {
        c.handle(
            Inbox::Tts(TtsEvent::Finished {
                conv: job1.conv,
                line: 1,
                audio_seconds: 0.5,
                synth_ms: 1.0,
            }),
            ms(t),
        );
    }
    let _ = c.take_effects();

    // Turn 2: the player asks a follow-up. This is where duplication used to
    // happen: the follow-up was pushed into `ac.turns` and then the whole
    // `ac.turns` was rendered into the history.
    let job2 = typed_turn(&mut c, "priest", "How are you?", 100);
    let prompt = &job2.request.prompt;

    assert!(
        prompt.contains("How are you?"),
        "the current utterance must be present in the prompt"
    );
    assert_eq!(
        prompt.matches("How are you?").count(),
        1,
        "the current utterance must appear exactly once (not duplicated in history)"
    );

    // The prior turn is still available as short-term memory.
    assert!(
        prompt.contains("Hello there!"),
        "the previous player turn should remain in the recent conversation"
    );
    assert_eq!(
        prompt.matches("Hello there!").count(),
        1,
        "the previous turn must appear exactly once"
    );
}

#[test]
fn history_excludes_only_the_current_turn() {
    let mut c = ready_core();
    // Turn 1.
    let job1 = typed_turn(&mut c, "priest", "First line.", 10);
    c.handle(done(&job1, speak("First reply.")), ms(20));
    for t in 21..40 {
        c.handle(
            Inbox::Tts(TtsEvent::Finished {
                conv: job1.conv,
                line: 1,
                audio_seconds: 0.5,
                synth_ms: 1.0,
            }),
            ms(t),
        );
    }
    let _ = c.take_effects();
    // Turn 2.
    let job2 = typed_turn(&mut c, "priest", "Second line.", 100);
    c.handle(done(&job2, speak("Second reply.")), ms(110));
    for t in 111..130 {
        c.handle(
            Inbox::Tts(TtsEvent::Finished {
                conv: job2.conv,
                line: 2,
                audio_seconds: 0.5,
                synth_ms: 1.0,
            }),
            ms(t),
        );
    }
    let _ = c.take_effects();
    // Turn 3: verify the prompt contains the two prior turns but not the
    // current one twice.
    let job3 = typed_turn(&mut c, "priest", "Third line.", 200);
    let prompt = &job3.request.prompt;
    assert_eq!(
        prompt.matches("Third line.").count(),
        1,
        "the current (third) line must appear exactly once"
    );
    // The two completed turns are in the history.
    assert!(prompt.contains("First line."));
    assert!(prompt.contains("First reply."));
    assert!(prompt.contains("Second line."));
    assert!(prompt.contains("Second reply."));
}
