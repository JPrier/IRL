//! # rpg_web_runtime
//!
//! The browser runtime: the same authoritative [`SimCore`] the native client
//! runs on its `irl-sim` thread, driven instead from the browser event loop.
//!
//! The native runtime blocks on an inbox with a timeout and runs workers on
//! threads. A standard `wasm32-unknown-unknown` page has one thread, so
//! here:
//!
//! * [`WebRuntime::pump`] is called once per rendered frame. It drains the
//!   inbox (results posted by finished futures and browser callbacks), runs
//!   every fixed tick that is due, and dispatches the effects the core asked
//!   for. It never waits for anything.
//! * Slow work — OpenRouter requests, speech, storage — runs as futures or
//!   browser callbacks started by a [`Dispatcher`]; each posts its result to
//!   the inbox with the correlation ids it was given, and `SimCore` rejects
//!   anything stale exactly as on native.
//! * The page also pumps from a timer, so the world keeps ticking when frames
//!   are slow or the tab is in the background (where browsers throttle
//!   animation frames). After a longer stall at most [`MAX_BURST`] catch-up
//!   ticks run and the rest are counted as skipped (as the native runtime
//!   does after a stall).
//!
//! Nothing in this module is browser-specific, so the scheduling, dispatch
//! and staleness behaviour is tested natively with a fake dispatcher; the
//! browser dispatcher lives in [`browser`] (wasm only).

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use rpg_eval::Session;
use rpg_runtime::{Command, CoreConfig, Effect, Inbox, SimCore, Snapshot, Telemetry};

#[cfg(target_arch = "wasm32")]
pub mod browser;

/// Messages for the simulation, posted by futures and callbacks.
pub type InboxQueue = Rc<RefCell<VecDeque<Inbox>>>;

/// Default fixed step, matching the native runtime.
pub const TICK_PERIOD_MS: f64 = 120.0;

/// Performs effects for the browser runtime. Implementations must return
/// immediately; anything slow is started here and reports back through the
/// inbox.
pub trait Dispatcher {
    fn dispatch(&mut self, effect: Effect, inbox: &InboxQueue);
}

/// Catch-up ticks allowed in one pump. Generous, so slow frames (software
/// rendering, a busy main thread) delay ticks rather than drop them; only a
/// real stall of several seconds (a throttled background tab) skips ticks.
pub const MAX_BURST: u32 = 25;

/// Fixed-timestep scheduling against a monotonic millisecond clock.
#[derive(Clone, Debug)]
pub struct Scheduler {
    period_ms: f64,
    next_ms: f64,
    /// Most ticks run in one pump; beyond this the schedule resyncs.
    pub max_burst: u32,
    skipped: u64,
}

impl Scheduler {
    pub fn new(period_ms: f64, start_ms: f64) -> Self {
        let period_ms = period_ms.max(1.0);
        Self {
            period_ms,
            next_ms: start_ms + period_ms,
            max_burst: MAX_BURST,
            skipped: 0,
        }
    }

    pub fn period_ms(&self) -> f64 {
        self.period_ms
    }

    /// Ticks skipped after stalls (backgrounded tab, debugger).
    pub fn skipped(&self) -> u64 {
        self.skipped
    }

    /// How many ticks are due at `now_ms`, and the lag of the first one.
    pub fn due(&mut self, now_ms: f64) -> (u32, f64) {
        if now_ms < self.next_ms {
            return (0, 0.0);
        }
        let lag = now_ms - self.next_ms;
        let behind = (lag / self.period_ms).floor() as u64 + 1;
        if behind > self.max_burst as u64 {
            // Resume the schedule instead of replaying a burst.
            self.skipped += behind - self.max_burst as u64;
            self.next_ms = now_ms + self.period_ms;
            return (self.max_burst, lag);
        }
        self.next_ms += self.period_ms * behind as f64;
        (behind as u32, lag)
    }
}

/// Rolling tick-rate measurement (the native runtime's meter, on ms).
#[derive(Debug, Default)]
struct TickMeter {
    window: VecDeque<f64>,
    max_lag_ms: f64,
}

impl TickMeter {
    fn record(&mut self, at_ms: f64, lag_ms: f64, tel: &Telemetry) {
        self.window.push_back(at_ms);
        while self.window.front().is_some_and(|t| at_ms - *t > 2000.0) {
            self.window.pop_front();
        }
        if let (Some(first), n) = (self.window.front(), self.window.len()) {
            let span = (at_ms - first) / 1000.0;
            if n > 1 && span > 0.0 {
                tel.ticks_per_second.set(((n - 1) as f64 / span) as f32);
            }
        }
        self.max_lag_ms = self.max_lag_ms.max(lag_ms);
        tel.tick_delay_ms.set(lag_ms as f32);
        tel.max_tick_lag_ms.set(self.max_lag_ms as f32);
        tel.ticks.fetch_add(1, Ordering::Relaxed);
    }
}

/// The browser's simulation driver.
pub struct WebRuntime<D: Dispatcher> {
    core: SimCore,
    sched: Scheduler,
    meter: TickMeter,
    inbox: InboxQueue,
    dispatcher: D,
    telemetry: Arc<Telemetry>,
    start_ms: f64,
    platform: String,
    /// Upper bound on inbox messages handled per pump.
    max_messages: usize,
}

impl<D: Dispatcher> WebRuntime<D> {
    pub fn new(session: Session, cfg: CoreConfig, dispatcher: D, now_ms: f64) -> Self {
        Self::with_period(session, cfg, dispatcher, now_ms, TICK_PERIOD_MS)
    }

    pub fn with_period(
        session: Session,
        cfg: CoreConfig,
        dispatcher: D,
        now_ms: f64,
        period_ms: f64,
    ) -> Self {
        let telemetry = Arc::new(Telemetry::new());
        Self {
            core: SimCore::new(session, cfg, telemetry.clone()),
            sched: Scheduler::new(period_ms, now_ms),
            meter: TickMeter::default(),
            inbox: Rc::new(RefCell::new(VecDeque::new())),
            dispatcher,
            telemetry,
            start_ms: now_ms,
            platform: "browser".into(),
            max_messages: 512,
        }
    }

    /// A handle futures and callbacks use to post results.
    pub fn inbox(&self) -> InboxQueue {
        self.inbox.clone()
    }

    pub fn send(&self, cmd: Command) {
        self.inbox.borrow_mut().push_back(Inbox::Ui(cmd));
    }

    pub fn core(&self) -> &SimCore {
        &self.core
    }

    pub fn core_mut(&mut self) -> &mut SimCore {
        &mut self.core
    }

    pub fn dispatcher(&self) -> &D {
        &self.dispatcher
    }

    pub fn dispatcher_mut(&mut self) -> &mut D {
        &mut self.dispatcher
    }

    pub fn telemetry(&self) -> &Arc<Telemetry> {
        &self.telemetry
    }

    pub fn set_platform(&mut self, platform: impl Into<String>) {
        self.platform = platform.into();
    }

    fn since_start(&self, now_ms: f64) -> Duration {
        Duration::from_secs_f64(((now_ms - self.start_ms) / 1000.0).max(0.0))
    }

    fn flush_effects(&mut self) {
        for effect in self.core.take_effects() {
            self.dispatcher.dispatch(effect, &self.inbox);
        }
    }

    /// Advance the browser runtime to `now_ms`. Returns a snapshot when
    /// anything changed. Never blocks.
    pub fn pump(&mut self, now_ms: f64) -> Option<Snapshot> {
        let mut changed = false;
        // Messages first, so a result that arrived before a tick is judged
        // against the world it raced with.
        for _ in 0..self.max_messages {
            let Some(msg) = self.inbox.borrow_mut().pop_front() else {
                break;
            };
            self.core.handle(msg, self.since_start(now_ms));
            self.flush_effects();
            changed = true;
        }
        let (ticks, lag) = self.sched.due(now_ms);
        for i in 0..ticks {
            let at = self.since_start(now_ms);
            self.core.tick(at);
            self.meter
                .record(now_ms, if i == 0 { lag } else { 0.0 }, &self.telemetry);
            self.flush_effects();
            changed = true;
        }
        self.telemetry
            .skipped_ticks
            .store(self.sched.skipped(), Ordering::Relaxed);
        changed.then(|| self.snapshot())
    }

    /// The current snapshot, with the driver's own diagnostics filled in.
    pub fn snapshot(&self) -> Snapshot {
        let mut snap = self.core.snapshot();
        snap.diagnostics.platform = self.platform.clone();
        snap.diagnostics.target_ticks_per_second = (1000.0 / self.sched.period_ms()) as f32;
        snap
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rpg_domain::ports::{InferenceError, InferenceResponse};
    use rpg_runtime::{InferDone, InferEvent, InferJob, InferRequest, LineKind, VoiceStage};

    /// Records effects; inference answers are released by the test.
    #[derive(Default)]
    struct FakeDispatcher {
        jobs: Vec<InferJob>,
        cancels: Vec<u64>,
        saves: Vec<(String, String)>,
    }

    impl Dispatcher for FakeDispatcher {
        fn dispatch(&mut self, effect: Effect, inbox: &InboxQueue) {
            match effect {
                Effect::Infer(InferRequest::Decide(job)) => self.jobs.push(job),
                Effect::Infer(InferRequest::Cancel { request_id }) => self.cancels.push(request_id),
                Effect::SaveData { slot, json } => {
                    self.saves.push((slot.clone(), json));
                    inbox
                        .borrow_mut()
                        .push_back(Inbox::Ui(Command::Notice(format!("saved {slot}"))));
                }
                _ => {}
            }
        }
    }

    fn runtime() -> WebRuntime<FakeDispatcher> {
        let mut session = Session::new(2026);
        session.teleport_player_to("priest");
        let mut rt = WebRuntime::new(
            session,
            CoreConfig::default(),
            FakeDispatcher::default(),
            0.0,
        );
        rt.core_mut().mark_unavailable(true, true, true, "test");
        rt.inbox()
            .borrow_mut()
            .push_back(Inbox::Infer(InferEvent::Ready {
                engine: "openrouter".into(),
                model_id: "test/model".into(),
                is_real_model: true,
            }));
        rt.pump(0.0);
        rt
    }

    fn answer(rt: &WebRuntime<FakeDispatcher>, job: &InferJob, text: &str) {
        rt.inbox()
            .borrow_mut()
            .push_back(Inbox::Infer(InferEvent::Done(InferDone {
                conv: job.conv,
                request_id: job.request_id,
                world_epoch: job.world_epoch,
                result: Ok(InferenceResponse {
                    text: text.into(),
                    ..Default::default()
                }),
                queue_ms: 0.0,
                infer_ms: 1.0,
            })));
    }

    /// Frames at 60 fps from `from` to `to` ms; returns the last snapshot.
    fn frames(rt: &mut WebRuntime<FakeDispatcher>, from: f64, to: f64) -> Option<Snapshot> {
        let mut t = from;
        let mut last = None;
        while t <= to {
            if let Some(s) = rt.pump(t) {
                last = Some(s);
            }
            t += 1000.0 / 60.0;
        }
        last
    }

    #[test]
    fn fixed_ticks_follow_the_clock_not_the_frame_rate() {
        let mut s = Scheduler::new(120.0, 0.0);
        assert_eq!(s.due(100.0).0, 0);
        assert_eq!(s.due(120.0).0, 1);
        assert_eq!(s.due(130.0).0, 0);
        assert_eq!(s.due(480.0).0, 3);
        // A slow 900 ms frame catches up instead of dropping ticks.
        assert_eq!(s.due(1_380.0).0, 7);
        assert_eq!(s.skipped(), 0);
        // A 10 s stall (background tab) runs a bounded burst and resyncs.
        let (n, lag) = s.due(10_480.0);
        assert_eq!(n, s.max_burst);
        assert!(lag > 9_000.0);
        assert!(s.skipped() > 50);
        assert_eq!(s.due(10_500.0).0, 0);
        assert_eq!(s.due(10_600.0).0, 1);
    }

    #[test]
    fn the_world_keeps_ticking_through_a_slow_remote_answer() {
        let mut rt = runtime();
        let tick0 = rt.core().session.sim.world.tick;
        rt.send(Command::Say("Hello, father.".into()));
        rt.pump(10.0);
        assert_eq!(rt.core().stage(), VoiceStage::Deciding);
        let job = rt.dispatcher().jobs[0].clone();

        // Ten seconds of frames while the request is outstanding.
        let snap = frames(&mut rt, 10.0, 10_010.0).unwrap();
        let ticked = rt.core().session.sim.world.tick - tick0;
        assert!(
            (80..=85).contains(&ticked),
            "ticks during the wait: {ticked}"
        );
        assert_eq!(snap.voice.stage, VoiceStage::Deciding);
        assert!(
            snap.diagnostics.ticks_per_second > 7.5,
            "{}",
            snap.diagnostics.ticks_per_second
        );
        assert_eq!(snap.diagnostics.platform, "browser");
        assert_eq!(snap.diagnostics.pending_inference, 1);

        answer(&rt, &job, r#"{"action":"speak","text":"Peace to you."}"#);
        let snap = rt.pump(10_020.0).unwrap();
        assert!(snap
            .dialogue
            .iter()
            .any(|l| l.kind == LineKind::Npc && l.text == "Peace to you."));
        assert_eq!(snap.voice.stage, VoiceStage::Idle);
        assert_eq!(snap.model_decisions, 1);
    }

    #[test]
    fn cancelled_requests_are_aborted_and_their_late_answers_ignored() {
        let mut rt = runtime();
        rt.send(Command::Say("Hello.".into()));
        rt.pump(10.0);
        let job = rt.dispatcher().jobs[0].clone();
        rt.send(Command::CancelVoice);
        rt.pump(20.0);
        assert_eq!(rt.dispatcher().cancels, vec![job.request_id]);
        let events = rt.core().session.sim.events.len();
        answer(&rt, &job, r#"{"action":"speak","text":"Too late."}"#);
        rt.pump(30.0);
        assert_eq!(rt.core().counters().stale_responses, 1);
        assert_eq!(rt.core().session.sim.events.len(), events);
        assert!(!rt.core().dialogue().iter().any(|l| l.text == "Too late."));
    }

    #[test]
    fn errors_return_the_game_to_a_usable_state() {
        let mut rt = runtime();
        rt.send(Command::Say("Hello.".into()));
        rt.pump(10.0);
        let job = rt.dispatcher().jobs[0].clone();
        rt.inbox()
            .borrow_mut()
            .push_back(Inbox::Infer(InferEvent::Done(InferDone {
                conv: job.conv,
                request_id: job.request_id,
                world_epoch: job.world_epoch,
                result: Err(InferenceError::Backend("OpenRouter network error".into())),
                queue_ms: 0.0,
                infer_ms: 1.0,
            })));
        let snap = rt.pump(20.0).unwrap();
        assert_eq!(snap.voice.stage, VoiceStage::Idle);
        assert_eq!(snap.fallback_decisions, 1);
        assert_eq!(snap.diagnostics.inference.errors, 1);
        // And the next turn works.
        rt.send(Command::Say("Again.".into()));
        rt.pump(30.0);
        assert_eq!(rt.core().stage(), VoiceStage::Deciding);
    }

    #[test]
    fn saves_go_to_the_platform_and_back() {
        let mut rt = runtime();
        frames(&mut rt, 0.0, 2_000.0);
        let tick = rt.core().session.sim.world.tick;
        rt.send(Command::ExportSave {
            slot: "quick".into(),
        });
        // The platform's completion notice is handled in the same pump.
        let snap = rt.pump(2_010.0).unwrap();
        let (slot, json) = rt.dispatcher().saves[0].clone();
        assert_eq!(slot, "quick");
        assert!(!json.contains("sk-or"), "saves carry no credentials");
        assert!(snap.dialogue.iter().any(|l| l.text == "saved quick"));
        frames(&mut rt, 2_030.0, 4_000.0);
        rt.send(Command::ImportSave {
            label: "browser storage".into(),
            json,
        });
        rt.pump(4_010.0);
        assert_eq!(rt.core().session.sim.world.tick, tick);
    }

    #[test]
    fn player_moves_while_a_request_is_outstanding() {
        let mut rt = runtime();
        rt.send(Command::Say("Hello.".into()));
        rt.pump(10.0);
        let before = rt.core().snapshot().player_pos;
        rt.send(Command::MoveTo(rpg_domain::geom::Vec2::new(
            before.x + 3.0,
            before.y,
        )));
        frames(&mut rt, 20.0, 3_000.0);
        let after = rt.core().snapshot().player_pos;
        assert!(after.x > before.x, "{before:?} -> {after:?}");
        assert_eq!(rt.core().stage(), VoiceStage::Deciding);
    }
}
