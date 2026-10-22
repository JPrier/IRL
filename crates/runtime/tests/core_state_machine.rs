//! Deterministic tests of the authoritative core's conversation state
//! machine. No threads, no clocks: every input is injected with an explicit
//! timestamp, so stale, duplicate, out-of-order, cancelled and timed-out
//! responses are reproduced exactly.

use std::sync::Arc;
use std::time::Duration;

use rpg_domain::ids::{ActorId, ItemId, PLAYER};
use rpg_domain::ports::{InferenceError, InferenceResponse};
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

fn speak_job(effects: &[Effect]) -> Option<SpeakJob> {
    effects.iter().find_map(|e| match e {
        Effect::Tts(TtsRequest::Speak(j)) => Some(j.clone()),
        _ => None,
    })
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

fn npc_lines(c: &SimCore) -> Vec<String> {
    c.dialogue()
        .iter()
        .filter(|l| l.kind == LineKind::Npc)
        .map(|l| l.text.clone())
        .collect()
}

/// Start a typed turn to `npc` and return its inference job.
fn typed_turn(c: &mut SimCore, npc: &str, text: &str, at: u64) -> InferJob {
    near(c, npc);
    c.handle(Inbox::Ui(Command::Say(text.into())), ms(at));
    assert_eq!(c.stage(), VoiceStage::Deciding);
    infer_job(&c.take_effects())
}

#[test]
fn full_voice_turn_walks_every_stage_while_the_world_ticks() {
    let mut c = ready_core();
    near(&mut c, "priest");
    let mut now = 0;
    let mut tick = |c: &mut SimCore, n: u32| {
        let before = c.session.sim.world.tick;
        for _ in 0..n {
            now += 120;
            c.tick(ms(now));
        }
        assert_eq!(c.session.sim.world.tick, before + n as u64);
    };

    c.handle(Inbox::Ui(Command::PushToTalk(true)), ms(0));
    let fx = c.take_effects();
    let conv = match fx.as_slice() {
        [Effect::Mic(MicRequest::Start { conv })] => *conv,
        other => panic!("unexpected effects {other:?}"),
    };
    assert_eq!(c.stage(), VoiceStage::Listening);
    tick(&mut c, 5);

    c.handle(Inbox::Ui(Command::PushToTalk(false)), ms(700));
    assert!(matches!(
        c.take_effects().as_slice(),
        [Effect::Mic(MicRequest::Stop { .. })]
    ));
    tick(&mut c, 2);
    c.handle(
        Inbox::Mic(MicEvent::Captured {
            conv,
            samples: vec![0.1; 16_000],
            sample_rate: 16_000,
            recorded_seconds: 1.0,
            speech_seconds: 1.0,
        }),
        ms(1000),
    );
    assert_eq!(c.stage(), VoiceStage::Transcribing);
    assert!(matches!(
        c.take_effects().as_slice(),
        [Effect::Asr(AsrRequest::Transcribe {
            sample_rate: 16_000,
            ..
        })]
    ));
    tick(&mut c, 5);

    c.handle(
        Inbox::Asr(AsrEvent::Transcript {
            conv,
            text: "Good evening, father.".into(),
            latency_ms: 300.0,
        }),
        ms(1600),
    );
    assert_eq!(c.stage(), VoiceStage::Deciding);
    let job = infer_job(&c.take_effects());
    assert_eq!(job.conv, conv);
    tick(&mut c, 20);

    c.handle(done(&job, speak("Peace be with you.")), ms(4000));
    let speak = speak_job(&c.take_effects()).expect("a committed line is voiced");
    assert_eq!(speak.text, "Peace be with you.");
    assert_eq!(speak.audio_epoch, c.audio_epoch());
    assert_eq!(c.stage(), VoiceStage::Synthesizing);
    tick(&mut c, 3);

    c.handle(
        Inbox::Tts(TtsEvent::Started {
            conv,
            line: speak.line,
            first_audio_ms: 90.0,
        }),
        ms(4500),
    );
    assert_eq!(c.stage(), VoiceStage::Speaking);
    tick(&mut c, 10);
    c.handle(
        Inbox::Tts(TtsEvent::Finished {
            conv,
            line: speak.line,
            audio_seconds: 1.5,
            synth_ms: 200.0,
        }),
        ms(6000),
    );
    assert_eq!(c.stage(), VoiceStage::Idle);

    let stages: Vec<VoiceStage> = c.transitions().into_iter().map(|(_, s)| s).collect();
    assert_eq!(
        stages,
        vec![
            VoiceStage::Listening,
            VoiceStage::Transcribing,
            VoiceStage::Deciding,
            VoiceStage::Validating,
            VoiceStage::Synthesizing,
            VoiceStage::Speaking,
            VoiceStage::Idle,
        ]
    );
    assert_eq!(npc_lines(&c), vec!["Peace be with you."]);
    assert_eq!(c.session.model_decisions(), 1);
    assert_eq!(c.counters().stale_responses, 0);
}

#[test]
fn typed_and_spoken_lines_take_the_same_gameplay_path() {
    let mut typed = ready_core();
    let job_typed = typed_turn(&mut typed, "priest", "Who runs the docks?", 0);

    let mut spoken = ready_core();
    near(&mut spoken, "priest");
    spoken.handle(Inbox::Ui(Command::PushToTalk(true)), ms(0));
    spoken.handle(Inbox::Ui(Command::PushToTalk(false)), ms(10));
    let conv = spoken.conversation_id().unwrap();
    spoken.handle(
        Inbox::Mic(MicEvent::Captured {
            conv,
            samples: vec![0.1; 8000],
            sample_rate: 16_000,
            recorded_seconds: 0.5,
            speech_seconds: 0.5,
        }),
        ms(20),
    );
    spoken.handle(
        Inbox::Asr(AsrEvent::Transcript {
            conv,
            text: "Who runs the docks?".into(),
            latency_ms: 1.0,
        }),
        ms(30),
    );
    let job_spoken = infer_job(&spoken.take_effects());
    assert_eq!(job_typed.request.prompt, job_spoken.request.prompt);
    assert_eq!(job_typed.request.system, job_spoken.request.system);
}

#[test]
fn cancel_then_late_model_answer_commits_nothing_and_says_nothing() {
    let mut c = ready_core();
    let job = typed_turn(&mut c, "priest", "Hello.", 0);
    let decisions_before = c.session.decisions.len();
    let events_before = c.session.sim.events.len();

    c.handle(Inbox::Ui(Command::CancelVoice), ms(100));
    let fx = c.take_effects();
    assert!(fx.iter().any(|e| matches!(
        e,
        Effect::Infer(InferRequest::Cancel { request_id }) if *request_id == job.request_id
    )));
    assert_eq!(c.stage(), VoiceStage::Idle);

    c.handle(done(&job, speak("Too late.")), ms(5000));
    assert!(speak_job(&c.take_effects()).is_none(), "stale line voiced");
    assert!(npc_lines(&c).is_empty());
    assert_eq!(c.session.decisions.len(), decisions_before);
    assert_eq!(c.session.sim.events.len(), events_before);
    assert_eq!(c.counters().stale_responses, 1);
    assert_eq!(c.counters().cancelled, 1);
}

#[test]
fn duplicate_responses_commit_once() {
    let mut c = ready_core();
    let job = typed_turn(&mut c, "priest", "Hello.", 0);
    c.handle(done(&job, speak("Once.")), ms(100));
    c.handle(done(&job, speak("Twice.")), ms(110));
    assert_eq!(npc_lines(&c), vec!["Once."]);
    assert_eq!(c.counters().stale_responses, 1);
    assert_eq!(c.session.model_decisions(), 1);
}

#[test]
fn out_of_order_responses_only_commit_the_current_conversation() {
    let mut c = ready_core();
    let first = typed_turn(&mut c, "priest", "First question.", 0);
    c.handle(Inbox::Ui(Command::CancelVoice), ms(50));
    let second = typed_turn(&mut c, "priest", "Second question.", 100);
    assert_ne!(first.request_id, second.request_id);

    // The newer answer arrives first, the abandoned one after it.
    c.handle(done(&second, speak("Answer two.")), ms(200));
    c.handle(done(&first, speak("Answer one.")), ms(210));
    assert_eq!(npc_lines(&c), vec!["Answer two."]);
    assert_eq!(c.counters().stale_responses, 1);
}

#[test]
fn npc_who_walks_away_during_inference_does_not_answer() {
    let mut c = ready_core();
    let job = typed_turn(&mut c, "grocer", "Is the shop open?", 0);
    // While the model thinks, the grocer's schedule takes them elsewhere.
    let grocer = c
        .session
        .sim
        .world
        .actor_mut(&ActorId::new("grocer"))
        .unwrap();
    grocer.location = rpg_domain::ids::LocationId::new("plaza");
    grocer.pos = rpg_domain::geom::Vec2::new(20.0, 12.0);
    let decisions_before = c.session.decisions.len();

    c.handle(done(&job, speak("Yes, come in.")), ms(3000));
    assert!(speak_job(&c.take_effects()).is_none());
    assert!(npc_lines(&c).is_empty());
    assert_eq!(c.session.decisions.len(), decisions_before);
    assert_eq!(c.counters().invalidated, 1);
    assert_eq!(c.stage(), VoiceStage::Idle);
}

#[test]
fn npc_moving_with_its_real_schedule_mid_inference_is_handled() {
    let mut c = ready_core();
    // The grocer's schedule moves them at tick 240.
    c.session.ticks(235);
    let job = typed_turn(&mut c, "grocer", "Do you have a minute?", 0);
    let start = c.session.player().pos;
    let mut now = 0;
    for _ in 0..40 {
        now += 120;
        c.tick(ms(now));
    }
    let grocer = c.session.sim.world.actor(&ActorId::new("grocer")).unwrap();
    let away = grocer.pos.distance(start) > TALK_RADIUS + 1.5
        || grocer.location != c.session.player().location;
    assert!(away, "the schedule should have walked the grocer away");
    c.handle(done(&job, speak("Sure.")), ms(now));
    assert!(npc_lines(&c).is_empty());
    assert_eq!(c.counters().invalidated, 1);
}

#[test]
fn rejected_proposal_is_regenerated_from_the_current_world() {
    let mut c = ready_core();
    let priest = ActorId::new("priest");
    c.session
        .sim
        .world
        .actor_mut(&priest)
        .unwrap()
        .inventory
        .add(ItemId::new("medicine"), 1);
    let job = typed_turn(&mut c, "priest", "Can you help my sister?", 0);
    // The item changes hands while the model is still deciding to give it.
    c.session
        .sim
        .world
        .actor_mut(&priest)
        .unwrap()
        .inventory
        .remove(&ItemId::new("medicine"), 1);
    let offer = Ok(InferenceResponse {
        text: r#"{"action":"offer","target":"player","give":["medicine"],"note":"Take this medicine."}"#
            .into(),
        ..Default::default()
    });
    c.handle(done(&job, offer), ms(1000));
    let fx = c.take_effects();
    assert!(
        speak_job(&fx).is_none(),
        "an impossible offer must not be voiced"
    );
    assert_eq!(c.counters().regenerations, 1);
    assert_eq!(c.stage(), VoiceStage::Deciding);
    let regen = infer_job(&fx);
    assert_ne!(regen.request_id, job.request_id);

    // The original request answering again is stale.
    c.handle(done(&job, speak("Stale.")), ms(1100));
    assert_eq!(c.counters().stale_responses, 1);

    c.handle(
        done(&regen, speak("I have nothing to give, child.")),
        ms(2000),
    );
    assert_eq!(npc_lines(&c), vec!["I have nothing to give, child."]);
    assert!(c
        .session
        .sim
        .world
        .offers
        .values()
        .all(|o| o.from != priest));
}

#[test]
fn second_rejection_ends_the_turn_silently() {
    let mut c = ready_core();
    let job = typed_turn(&mut c, "priest", "Give me something.", 0);
    let bad = || {
        Ok(InferenceResponse {
            text: r#"{"action":"offer","target":"player","give":["medicine"]}"#.into(),
            ..Default::default()
        })
    };
    c.handle(done(&job, bad()), ms(100));
    let regen = infer_job(&c.take_effects());
    c.handle(done(&regen, bad()), ms(200));
    assert!(speak_job(&c.take_effects()).is_none());
    assert!(npc_lines(&c).is_empty());
    assert_eq!(c.stage(), VoiceStage::Idle);
}

#[test]
fn model_timeout_cancels_and_falls_back_but_never_hangs() {
    let mut c = ready_core();
    let job = typed_turn(&mut c, "priest", "Hello?", 0);
    let mut now = 0;
    while now < 44_000 {
        now += 120;
        c.tick(ms(now));
    }
    assert_eq!(c.stage(), VoiceStage::Deciding);
    c.tick(ms(45_100));
    let fx = c.take_effects();
    assert!(fx.iter().any(|e| matches!(
        e,
        Effect::Infer(InferRequest::Cancel { request_id }) if *request_id == job.request_id
    )));
    assert_eq!(c.counters().timeouts, 1);
    assert_eq!(c.session.fallback_decisions(), 1);
    assert!(matches!(
        c.session.decisions.last().unwrap().1,
        rpg_agents::DecisionSource::HeuristicFallback { .. }
    ));
    assert_ne!(c.stage(), VoiceStage::Deciding);

    // The model finally answers: too late, ignored.
    c.handle(done(&job, speak("Sorry, I was thinking.")), ms(50_000));
    assert_eq!(c.counters().stale_responses, 1);
    assert!(!npc_lines(&c).contains(&"Sorry, I was thinking.".to_string()));
}

#[test]
fn every_stage_has_a_deadline_so_the_turn_is_never_stuck() {
    for stage in [
        VoiceStage::Listening,
        VoiceStage::Transcribing,
        VoiceStage::Synthesizing,
        VoiceStage::Speaking,
    ] {
        let mut c = ready_core();
        near(&mut c, "priest");
        c.handle(Inbox::Ui(Command::PushToTalk(true)), ms(0));
        let conv = c.conversation_id().unwrap();
        if stage != VoiceStage::Listening {
            c.handle(
                Inbox::Mic(MicEvent::Captured {
                    conv,
                    samples: vec![0.1; 1600],
                    sample_rate: 16_000,
                    recorded_seconds: 0.1,
                    speech_seconds: 0.1,
                }),
                ms(0),
            );
        }
        if matches!(stage, VoiceStage::Synthesizing | VoiceStage::Speaking) {
            c.handle(
                Inbox::Asr(AsrEvent::Transcript {
                    conv,
                    text: "Hi".into(),
                    latency_ms: 1.0,
                }),
                ms(0),
            );
            let job = infer_job(&c.take_effects());
            c.handle(done(&job, speak("Hello.")), ms(0));
            if stage == VoiceStage::Speaking {
                let line = speak_job(&c.take_effects()).unwrap().line;
                c.handle(
                    Inbox::Tts(TtsEvent::Started {
                        conv,
                        line,
                        first_audio_ms: 1.0,
                    }),
                    ms(0),
                );
            }
        }
        assert_eq!(c.stage(), stage);
        c.tick(ms(200_000));
        assert_eq!(c.stage(), VoiceStage::Idle, "{stage:?} never timed out");
        assert_eq!(c.counters().timeouts, 1);
    }
}

#[test]
fn load_during_inference_discards_the_old_timeline() {
    let dir = std::env::temp_dir().join(format!("irl-load-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("save.json").to_string_lossy().into_owned();

    let mut c = ready_core();
    c.handle(Inbox::Ui(Command::Save(path.clone())), ms(0));
    let saved_tick = c.session.sim.world.tick;
    let job = typed_turn(&mut c, "priest", "Hello.", 0);
    for i in 1..=10 {
        c.tick(ms(i * 120));
    }
    let epoch = c.world_epoch();
    let audio = c.audio_epoch();
    c.handle(Inbox::Ui(Command::Load(path)), ms(1500));
    let fx = c.take_effects();
    assert_eq!(c.world_epoch(), epoch + 1);
    assert!(c.audio_epoch() > audio);
    assert!(fx
        .iter()
        .any(|e| matches!(e, Effect::SetAudioEpoch(n) if *n == c.audio_epoch())));
    assert_eq!(c.session.sim.world.tick, saved_tick);
    assert_eq!(c.stage(), VoiceStage::Idle);

    c.handle(done(&job, speak("From the old timeline.")), ms(2000));
    assert!(speak_job(&c.take_effects()).is_none());
    assert!(npc_lines(&c).is_empty());
    assert_eq!(c.counters().stale_responses, 1);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn saving_during_inference_keeps_the_turn_alive() {
    let dir = std::env::temp_dir().join(format!("irl-save-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("save.json").to_string_lossy().into_owned();
    let mut c = ready_core();
    let job = typed_turn(&mut c, "priest", "Hello.", 0);
    c.handle(Inbox::Ui(Command::Save(path.clone())), ms(10));
    assert!(std::path::Path::new(&path).is_file());
    assert_eq!(c.stage(), VoiceStage::Deciding);
    c.handle(done(&job, speak("Still here.")), ms(20));
    assert_eq!(npc_lines(&c), vec!["Still here."]);
    // The save holds canonical state only; it reloads cleanly.
    assert!(rpg_persistence::load_json(&path).is_ok());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn push_to_talk_during_a_turn_is_refused_not_queued() {
    let mut c = ready_core();
    let _job = typed_turn(&mut c, "priest", "Hello.", 0);
    for i in 0..10 {
        c.handle(Inbox::Ui(Command::PushToTalk(i % 2 == 0)), ms(10 + i));
    }
    let fx = c.take_effects();
    assert!(
        !fx.iter()
            .any(|e| matches!(e, Effect::Mic(MicRequest::Start { .. }))),
        "a second recording must not start behind the first"
    );
    assert_eq!(c.counters().refused_ptt, 5);
    assert_eq!(c.counters().conversations, 1);
    assert_eq!(c.stage(), VoiceStage::Deciding);
}

#[test]
fn talking_over_an_npc_interrupts_and_silences_it() {
    let mut c = ready_core();
    let job = typed_turn(&mut c, "priest", "Tell me a story.", 0);
    c.handle(done(&job, speak("Once upon a time…")), ms(100));
    let line = speak_job(&c.take_effects()).unwrap();
    c.handle(
        Inbox::Tts(TtsEvent::Started {
            conv: line.conv,
            line: line.line,
            first_audio_ms: 50.0,
        }),
        ms(200),
    );
    assert_eq!(c.stage(), VoiceStage::Speaking);

    c.handle(Inbox::Ui(Command::PushToTalk(true)), ms(300));
    let fx = c.take_effects();
    assert!(fx
        .iter()
        .any(|e| matches!(e, Effect::SetAudioEpoch(n) if *n > line.audio_epoch)));
    assert!(fx
        .iter()
        .any(|e| matches!(e, Effect::Mic(MicRequest::Start { .. }))));
    assert_eq!(c.stage(), VoiceStage::Listening);
    assert_eq!(c.counters().interrupted, 1);

    // The interrupted line's completion event is recognised as stale.
    c.handle(
        Inbox::Tts(TtsEvent::Finished {
            conv: line.conv,
            line: line.line,
            audio_seconds: 1.0,
            synth_ms: 1.0,
        }),
        ms(400),
    );
    assert_eq!(c.stage(), VoiceStage::Listening);
    assert_eq!(c.counters().stale_responses, 1);
}

#[test]
fn failures_at_every_stage_return_to_idle() {
    // Microphone fails while recording.
    let mut c = ready_core();
    near(&mut c, "priest");
    c.handle(Inbox::Ui(Command::PushToTalk(true)), ms(0));
    c.handle(Inbox::Mic(MicEvent::DeviceError("unplugged".into())), ms(5));
    assert_eq!(c.stage(), VoiceStage::Idle);
    // …and a new press is refused with a clear message, not a hang.
    c.handle(Inbox::Ui(Command::PushToTalk(true)), ms(10));
    assert_eq!(c.stage(), VoiceStage::Idle);
    assert!(c
        .dialogue()
        .last()
        .unwrap()
        .text
        .contains("microphone UNAVAILABLE"));
    // Recovery.
    c.handle(
        Inbox::Mic(MicEvent::DeviceReady(MicInfo {
            name: "mic".into(),
            sample_rate: 44_100,
            channels: 1,
            sample_format: "I16".into(),
        })),
        ms(20),
    );
    c.handle(Inbox::Ui(Command::PushToTalk(true)), ms(30));
    assert_eq!(c.stage(), VoiceStage::Listening);

    // No speech in the recording.
    let conv = c.conversation_id().unwrap();
    c.handle(
        Inbox::Mic(MicEvent::NoSpeech {
            conv,
            held_seconds: 2.0,
            recorded_seconds: 2.0,
            rms: 0.0,
            peak: 0.0,
        }),
        ms(40),
    );
    assert_eq!(c.stage(), VoiceStage::Idle);
    assert!(c.dialogue().last().unwrap().text.contains("no audio"));

    // ASR fails.
    c.handle(Inbox::Ui(Command::PushToTalk(true)), ms(50));
    let conv = c.conversation_id().unwrap();
    c.handle(
        Inbox::Mic(MicEvent::Captured {
            conv,
            samples: vec![0.2; 1600],
            sample_rate: 16_000,
            recorded_seconds: 0.1,
            speech_seconds: 0.1,
        }),
        ms(60),
    );
    c.handle(
        Inbox::Asr(AsrEvent::Failed {
            conv,
            error: "decoder error".into(),
        }),
        ms(70),
    );
    assert_eq!(c.stage(), VoiceStage::Idle);

    // Empty transcript.
    c.handle(Inbox::Ui(Command::PushToTalk(true)), ms(80));
    let conv = c.conversation_id().unwrap();
    c.handle(
        Inbox::Mic(MicEvent::Captured {
            conv,
            samples: vec![0.2; 1600],
            sample_rate: 16_000,
            recorded_seconds: 0.1,
            speech_seconds: 0.1,
        }),
        ms(90),
    );
    c.handle(
        Inbox::Asr(AsrEvent::Transcript {
            conv,
            text: "  ".into(),
            latency_ms: 1.0,
        }),
        ms(95),
    );
    assert_eq!(c.stage(), VoiceStage::Idle);

    // TTS fails.
    let job = typed_turn(&mut c, "priest", "Hello.", 100);
    c.handle(done(&job, speak("Hello.")), ms(110));
    let line = speak_job(&c.take_effects()).unwrap();
    c.handle(
        Inbox::Tts(TtsEvent::Failed {
            conv: line.conv,
            line: line.line,
            error: "no device".into(),
        }),
        ms(120),
    );
    assert_eq!(c.stage(), VoiceStage::Idle);

    // Inference error falls back to the policy (and says so).
    let job = typed_turn(&mut c, "priest", "Hello again.", 200);
    c.handle(
        done(&job, Err(InferenceError::Backend("boom".into()))),
        ms(210),
    );
    assert!(matches!(
        c.session.decisions.last().unwrap().1,
        rpg_agents::DecisionSource::HeuristicFallback { .. }
    ));
}

#[test]
fn no_speech_messages_separate_tap_capture_failure_and_silence() {
    let cases = [
        (0.1, 0.1, 0.0, "too short"),
        (2.0, 0.2, 0.0, "capture is failing"),
        (2.0, 2.0, 0.0, "digital silence"),
        (2.0, 2.0, 0.01, "very quiet"),
        (2.0, 2.0, 0.4, "no speech"),
    ];
    for (held, recorded, peak, expect) in cases {
        let mut c = ready_core();
        c.handle(Inbox::Ui(Command::PushToTalk(true)), ms(0));
        let conv = c.conversation_id().unwrap();
        c.handle(
            Inbox::Mic(MicEvent::NoSpeech {
                conv,
                held_seconds: held,
                recorded_seconds: recorded,
                rms: peak / 3.0,
                peak,
            }),
            ms(10),
        );
        let last = &c.dialogue().last().unwrap().text;
        assert!(last.contains(expect), "{held}/{recorded}/{peak}: {last}");
        assert_eq!(c.stage(), VoiceStage::Idle);
    }
}

#[test]
fn nobody_in_earshot_ends_the_turn_without_inference() {
    let mut c = ready_core();
    c.session.player_mut().location = rpg_domain::ids::LocationId::new("empty_lot");
    c.handle(Inbox::Ui(Command::Say("Anyone?".into())), ms(0));
    let fx = c.take_effects();
    assert!(!fx.iter().any(|e| matches!(e, Effect::Infer(_))));
    assert_eq!(c.stage(), VoiceStage::Idle);
}

#[test]
fn without_a_real_model_the_policy_answers_immediately() {
    let mut c = ready_core();
    c.handle(
        Inbox::Infer(InferEvent::Ready {
            engine: "heuristic".into(),
            model_id: "none".into(),
            is_real_model: false,
        }),
        ms(0),
    );
    near(&mut c, "grocer");
    c.handle(
        Inbox::Ui(Command::Say("I need to see your ledger.".into())),
        ms(0),
    );
    let fx = c.take_effects();
    assert!(!fx.iter().any(|e| matches!(e, Effect::Infer(_))));
    assert_eq!(c.session.decisions.len(), 1);
    assert!(matches!(
        c.session.decisions[0].1,
        rpg_agents::DecisionSource::HeuristicEngine
    ));
}

#[test]
fn a_failed_model_load_is_reported_as_fallback_provenance() {
    let mut c = ready_core();
    c.handle(
        Inbox::Infer(InferEvent::LoadFailed("weights missing".into())),
        ms(0),
    );
    let _ = typed_turn_no_infer(&mut c, "priest", "Hello.");
    assert!(matches!(
        c.session.decisions[0].1,
        rpg_agents::DecisionSource::HeuristicFallback { .. }
    ));
}

fn typed_turn_no_infer(c: &mut SimCore, npc: &str, text: &str) -> Vec<Effect> {
    near(c, npc);
    c.handle(Inbox::Ui(Command::Say(text.into())), ms(0));
    c.take_effects()
}

#[test]
fn player_can_move_and_trade_while_an_npc_thinks() {
    let mut c = ready_core();
    let _job = typed_turn(&mut c, "priest", "Hello.", 0);
    let start = c.session.player().pos;
    c.handle(
        Inbox::Ui(Command::MoveTo(rpg_domain::geom::Vec2::new(
            start.x + 2.0,
            start.y,
        ))),
        ms(10),
    );
    for i in 1..=20 {
        c.tick(ms(10 + i * 120));
    }
    let moved = c.session.player().pos.distance(start);
    assert!(
        moved > 0.5,
        "player did not move during inference ({moved})"
    );
    assert_eq!(c.stage(), VoiceStage::Deciding);
    assert_eq!(
        c.session.player().id,
        ActorId::new(PLAYER),
        "sanity: player is the mover"
    );
}
