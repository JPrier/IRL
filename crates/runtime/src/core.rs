//! The authoritative simulation core.
//!
//! [`SimCore`] owns the canonical [`Session`] and the conversation state
//! machine. It is deliberately single-threaded and *pure with respect to
//! time and I/O*: every input is a message plus the current monotonic time,
//! and every side effect it wants (start the microphone, run a model request,
//! voice a line, silence stale audio) is returned as an [`Effect`] for the
//! runtime to dispatch. Nothing in here can block, so the fixed tick schedule
//! never waits on audio, recognition, inference or playback — and the same
//! code is driven step by step by the deterministic tests.
//!
//! ## The world keeps moving while an NPC thinks
//!
//! A decision request is rendered from the world at time *t₀* and answered
//! at some later *t₁*. Between the two the NPC may walk away, an offer may
//! expire, an item may change hands, the player may cancel or load a save.
//! So when a response arrives the core:
//!
//! 1. checks its correlation ids (conversation, request, world epoch) and the
//!    conversation stage — anything else is counted as stale and dropped;
//! 2. re-checks the *situation* (the NPC is still alive, present and in
//!    earshot) against the current world;
//! 3. resolves the model output against the current world (canonical agenda
//!    and obligations included), then validates and commits it through the
//!    same [`Session::commit_npc_decision`] path every other client uses;
//! 4. only a committed action with words is ever sent to TTS, tagged with
//!    the current audio epoch, so a cancelled or superseded line can never be
//!    heard announcing something that did not happen.
//!
//! A proposal rejected by validation is regenerated once from the fresh
//! world when it came from the model; otherwise the NPC says nothing.

use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use rpg_agents::{NpcPolicy, NpcRequestContext};
use rpg_domain::action::{Action, Loudness, ValidationOutcome};
use rpg_domain::geom::Vec2;
use rpg_domain::ids::{ActorId, OfferId, PLAYER};
use rpg_domain::ports::{InferenceError, InferenceRequest};
use rpg_domain::world::{ActorKind, ActorStatus, World};
use rpg_eval::Session;

use crate::protocol::*;
use crate::telemetry::{Telemetry, NONE};

/// Interaction radius, in world units. Normal speech carries ~6 units, but the
/// simulation also applies occlusion and location factors, so this stays
/// conservative: the UI never offers a target the validator would reject.
pub const TALK_RADIUS: f32 = 3.5;

/// How long an idle conversation remains open before it is considered over.
pub const CONVERSATION_TIMEOUT: Duration = Duration::from_secs(300);

/// Maximum number of recent exchanges kept for short-term memory.
pub const MAX_CONVERSATION_TURNS: usize = 12;

/// Timeouts and limits for the conversation state machine.
#[derive(Clone, Debug)]
pub struct CoreConfig {
    pub talk_radius: f32,
    /// How far an NPC may drift (beyond `talk_radius`) during inference and
    /// still answer. Further than this, the answer is discarded.
    pub answer_slack: f32,
    /// Hard ceiling on a recording, after which the core gives up on the mic.
    pub listen_timeout: Duration,
    pub transcribe_timeout: Duration,
    /// After this the model is cancelled and the deterministic policy answers.
    pub decide_timeout: Duration,
    pub synth_timeout: Duration,
    pub speak_timeout: Duration,
    /// How many times a rejected model proposal may be regenerated.
    pub max_regenerations: u8,
    /// Dialogue lines kept in memory.
    pub dialogue_limit: usize,
}

impl Default for CoreConfig {
    fn default() -> Self {
        Self {
            talk_radius: TALK_RADIUS,
            answer_slack: 1.5,
            listen_timeout: Duration::from_secs(20),
            transcribe_timeout: Duration::from_secs(30),
            decide_timeout: Duration::from_secs(45),
            synth_timeout: Duration::from_secs(30),
            speak_timeout: Duration::from_secs(90),
            max_regenerations: 1,
            dialogue_limit: 200,
        }
    }
}

#[derive(Clone, Debug)]
enum EngineState {
    Loading,
    Ready {
        name: String,
        model_id: String,
        real: bool,
    },
    Failed(String),
}

#[derive(Clone, Debug)]
struct PendingDecision {
    request_id: u64,
    world_epoch: u64,
    request: InferenceRequest,
}

#[derive(Clone, Debug)]
struct Conversation {
    id: ConvId,
    stage: VoiceStage,
    stage_since: Duration,
    npc: Option<ActorId>,
    utterance: Option<String>,
    pending: Option<PendingDecision>,
    /// Line currently being voiced.
    line: Option<u64>,
    regenerations: u8,
}

/// A persistent social interaction with one NPC. Unlike [`Conversation`],
/// which tracks one voice/inference turn, this survives across turns until
/// the player switches targets, ends it, or the NPC leaves.
#[derive(Clone, Debug)]
struct ActiveConversation {
    npc: ActorId,
    last_active: Duration,
    turns: VecDeque<ConversationTurn>,
}

/// One exchange in an active conversation, kept for short-term memory.
#[derive(Clone, Debug)]
struct ConversationTurn {
    speaker: ActorId,
    text: String,
}

/// Counters surfaced in the diagnostics overlay and asserted by tests.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    pub conversations: u64,
    pub completed: u64,
    pub cancelled: u64,
    /// Responses (any worker) that arrived for a conversation, request or
    /// epoch that is no longer current.
    pub stale_responses: u64,
    pub timeouts: u64,
    /// Push-to-talk presses refused because a turn was mid-flight.
    pub refused_ptt: u64,
    pub regenerations: u64,
    /// Answers discarded because the situation changed during inference.
    pub invalidated: u64,
    pub interrupted: u64,
}

/// Session-wide accounting of every decision request answered by an
/// inference backend, stale or not (a late answer still cost tokens).
///
/// Costs keep the provider's *reported* figure apart from the local
/// *estimate*; they are never summed together.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InferenceStats {
    /// Requests handed to the inference backend.
    pub requests: u64,
    /// Answers that arrived (successful completions).
    pub completed: u64,
    /// Requests that ended in an error (network, provider, timeout, cancel).
    pub errors: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// Sum of provider-reported costs, when any were reported.
    pub reported_cost_usd: Option<f64>,
    /// Sum of locally estimated costs, when any were estimated.
    pub estimated_cost_usd: Option<f64>,
    /// End-to-end latency of the most recent request.
    pub last_latency_ms: Option<f32>,
    /// Gateway/provider/model of the most recent remote answer.
    pub last_remote: Option<String>,
    /// The most recent inference error, for the diagnostics overlay.
    pub last_error: Option<String>,
}

impl InferenceStats {
    fn record(&mut self, done: &InferDone) {
        self.last_latency_ms = Some(done.infer_ms);
        match &done.result {
            Ok(resp) => {
                self.completed += 1;
                self.prompt_tokens += resp.prompt_tokens as u64;
                self.completion_tokens += resp.completion_tokens as u64;
                if let Some(r) = &resp.remote {
                    if let Some(c) = r.reported_cost_usd {
                        *self.reported_cost_usd.get_or_insert(0.0) += c;
                    }
                    if let Some(c) = r.estimated_cost_usd {
                        *self.estimated_cost_usd.get_or_insert(0.0) += c;
                    }
                    self.last_remote = Some(format!(
                        "{} / {}{}",
                        r.gateway,
                        r.model.as_deref().unwrap_or(&r.requested_model),
                        r.provider
                            .as_deref()
                            .map(|p| format!(" via {p}"))
                            .unwrap_or_default()
                    ));
                }
            }
            Err(e) => {
                self.errors += 1;
                self.last_error = Some(e.to_string());
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
struct LastTurn {
    transcript: Option<String>,
    asr_ms: Option<f32>,
    llm_queue_ms: Option<f32>,
    llm_infer_ms: Option<f32>,
    tts_first_ms: Option<f32>,
    tts_synth_ms: Option<f32>,
    recorded_seconds: f32,
    stream_chunks: Option<u32>,
    llm_first_ms: Option<f32>,
}

pub struct SimCore {
    pub session: Session,
    cfg: CoreConfig,
    telemetry: Arc<Telemetry>,
    engine: EngineState,
    mic: Readiness,
    mic_info: Option<MicInfo>,
    mic_devices: Vec<String>,
    asr: Readiness,
    tts: Readiness,
    conv: Option<Conversation>,
    active_conversation: Option<ActiveConversation>,
    next_conv: ConvId,
    next_line: u64,
    world_epoch: u64,
    audio_epoch: u64,
    dialogue: Vec<DialogueLine>,
    status: String,
    counters: Counters,
    inference: InferenceStats,
    rejected_actions: u64,
    last_decision: Option<String>,
    last: LastTurn,
    transitions: VecDeque<(ConvId, VoiceStage)>,
    effects: Vec<Effect>,
    now: Duration,
    quit: bool,
}

impl SimCore {
    pub fn new(session: Session, cfg: CoreConfig, telemetry: Arc<Telemetry>) -> Self {
        Self {
            session,
            cfg,
            telemetry,
            engine: EngineState::Loading,
            mic: Readiness::Loading,
            mic_info: None,
            mic_devices: Vec::new(),
            asr: Readiness::Loading,
            tts: Readiness::Loading,
            conv: None,
            active_conversation: None,
            next_conv: 1,
            next_line: 1,
            world_epoch: 0,
            audio_epoch: 0,
            dialogue: Vec::new(),
            status: "ready — hold V to talk, Enter to type, WASD to walk".into(),
            counters: Counters::default(),
            inference: InferenceStats::default(),
            rejected_actions: 0,
            last_decision: None,
            last: LastTurn::default(),
            transitions: VecDeque::new(),
            effects: Vec::new(),
            now: Duration::ZERO,
            quit: false,
        }
    }

    // -----------------------------------------------------------------
    // Inputs
    // -----------------------------------------------------------------

    /// Mark voice components that will never start (e.g. `--no-voice`).
    pub fn mark_unavailable(&mut self, mic: bool, asr: bool, tts: bool, why: &str) {
        let r = Readiness::Unavailable(why.to_string());
        if mic {
            self.mic = r.clone();
        }
        if asr {
            self.asr = r.clone();
        }
        if tts {
            self.tts = r;
        }
    }

    /// Handle one inbox message at monotonic time `now`.
    pub fn handle(&mut self, msg: Inbox, now: Duration) {
        self.now = now;
        match msg {
            Inbox::Ui(cmd) => self.on_command(cmd),
            Inbox::Mic(ev) => self.on_mic(ev),
            Inbox::Asr(ev) => self.on_asr(ev),
            Inbox::Infer(ev) => self.on_infer(ev),
            Inbox::Tts(ev) => self.on_tts(ev),
        }
    }

    /// Advance the world one fixed step, then enforce stage deadlines.
    pub fn tick(&mut self, now: Duration) {
        self.now = now;
        let _ = self.session.tick();
        self.check_deadlines();
        self.check_active_conversation();
    }

    /// End the active conversation if its NPC can no longer participate, or
    /// the interaction has been idle for too long.
    fn check_active_conversation(&mut self) {
        let Some(ac) = &self.active_conversation else {
            return;
        };
        let npc = ac.npc.clone();
        if !self.npc_can_participate(&npc) {
            let name = actor_name(&self.session.sim.world, &npc);
            self.active_conversation = None;
            self.system(&format!("{name} is no longer close enough to talk"));
            return;
        }
        if self.now.saturating_sub(ac.last_active) > CONVERSATION_TIMEOUT {
            self.active_conversation = None;
            self.system("the conversation has gone quiet");
        }
    }

    /// Whether an NPC is alive, present, and within talking distance of the
    /// player. Used to keep a conversation open across turns.
    fn npc_can_participate(&self, npc: &ActorId) -> bool {
        let world = &self.session.sim.world;
        let Some(player) = world.actor(&ActorId::new(PLAYER)) else {
            return false;
        };
        let Some(a) = world.actor(npc) else {
            return false;
        };
        a.kind == ActorKind::Npc
            && a.status == ActorStatus::Alive
            && a.location == player.location
            && a.pos.distance(player.pos) <= self.cfg.talk_radius
    }

    /// Side effects produced since the last call, in order.
    pub fn take_effects(&mut self) -> Vec<Effect> {
        std::mem::take(&mut self.effects)
    }

    pub fn quit_requested(&self) -> bool {
        self.quit
    }

    pub fn counters(&self) -> &Counters {
        &self.counters
    }

    pub fn inference_stats(&self) -> &InferenceStats {
        &self.inference
    }

    pub fn stage(&self) -> VoiceStage {
        self.conv
            .as_ref()
            .map(|c| c.stage)
            .unwrap_or(VoiceStage::Idle)
    }

    pub fn conversation_id(&self) -> Option<ConvId> {
        self.conv.as_ref().map(|c| c.id)
    }

    pub fn pending_request_id(&self) -> Option<u64> {
        self.conv
            .as_ref()
            .and_then(|c| c.pending.as_ref())
            .map(|p| p.request_id)
    }

    pub fn world_epoch(&self) -> u64 {
        self.world_epoch
    }

    pub fn audio_epoch(&self) -> u64 {
        self.audio_epoch
    }

    pub fn dialogue(&self) -> &[DialogueLine] {
        &self.dialogue
    }

    /// Every stage each conversation entered, oldest first (bounded).
    pub fn transitions(&self) -> Vec<(ConvId, VoiceStage)> {
        self.transitions.iter().cloned().collect()
    }

    // -----------------------------------------------------------------
    // Renderer commands
    // -----------------------------------------------------------------

    fn on_command(&mut self, cmd: Command) {
        match cmd {
            Command::Quit => {
                self.quit = true;
                self.cancel_active("shutting down");
            }
            Command::MoveTo(pos) => {
                let _ = self.session.player_do(Action::MoveToPos { pos });
            }
            Command::AcceptOffer(id) => self.offer(id, true),
            Command::RejectOffer(id) => self.offer(id, false),
            Command::Investigate(target) => {
                let _ = self.session.player_do(Action::Investigate { target });
            }
            Command::Save(path) => self.save(&path),
            Command::Load(path) => self.load(&path),
            Command::ExportSave { slot } => self.export_save(slot),
            Command::ImportSave { label, json } => match rpg_persistence::from_json_str(&json) {
                Ok(save) => self.restore_save(save, &label),
                Err(e) => self.system(&format!("load failed ({label}): {e}")),
            },
            Command::Notice(text) => self.system(&text),
            Command::VoiceUnavailable { mic, asr, tts, why } => {
                if mic && self.stage() == VoiceStage::Listening {
                    self.cancel_active("voice input turned off");
                }
                self.mark_unavailable(mic, asr, tts, &why);
            }
            Command::PushToTalk(true) => self.ptt_down(),
            Command::PushToTalk(false) => self.ptt_up(),
            Command::Say(text) => self.typed(text),
            Command::SelectTarget(id) => {
                if self.npc_can_participate(&id) {
                    // Switching to a different NPC while a turn is in flight
                    // would otherwise let the old NPC's late answer be voiced
                    // after the player has moved on. Cancel it first so pending
                    // work is handled consistently.
                    if self
                        .conv
                        .as_ref()
                        .is_some_and(|c| c.npc.as_ref() != Some(&id))
                    {
                        self.cancel_active("switched target");
                    }
                    self.begin_active_conversation(id.clone());
                    let name = actor_name(&self.session.sim.world, &id);
                    self.status = format!("talking to {name}");
                } else {
                    self.system("that person is not close enough to talk to");
                }
            }
            Command::CycleTarget => {
                let candidates = self.nearby_all();
                if candidates.is_empty() {
                    self.system("nobody is close enough to talk to");
                } else {
                    let current = self.active_conversation.as_ref().map(|ac| ac.npc.clone());
                    // Finite, deterministic selection. `candidates` is already
                    // ordered (nearest first, ties broken by id), so:
                    // - no current target  -> first candidate
                    // - current present    -> next candidate, wrapping to first
                    // - current absent     -> first candidate (it left/died)
                    let next = match current {
                        Some(current_npc) => candidates
                            .iter()
                            .position(|c| *c == current_npc)
                            .map(|idx| candidates[(idx + 1) % candidates.len()].clone())
                            .unwrap_or_else(|| candidates[0].clone()),
                        None => candidates[0].clone(),
                    };
                    // A target switch invalidates an in-flight turn for a
                    // different NPC; cancel it so its late answer cannot be
                    // voiced after the player has moved on.
                    if self
                        .conv
                        .as_ref()
                        .is_some_and(|c| c.npc.as_ref() != Some(&next))
                    {
                        self.cancel_active("switched target");
                    }
                    self.begin_active_conversation(next.clone());
                    let name = actor_name(&self.session.sim.world, &next);
                    self.status = format!("talking to {name}");
                }
            }
            Command::EndConversation => {
                // Cancel any in-flight turn first: ending the conversation must
                // not leave a model request that later voices a ghost line.
                if self.conv.is_some() {
                    self.cancel_active("conversation ended");
                }
                if self.active_conversation.is_some() {
                    self.active_conversation = None;
                    self.status = "conversation ended".into();
                }
            }
            Command::CancelVoice => {
                if self.conv.is_some() {
                    self.cancel_active("cancelled");
                    self.system("conversation cancelled");
                }
            }
            Command::MicSelfTest { path, seconds } => {
                if self.mic.is_ready() {
                    self.status = format!("mic self-test: recording {seconds:.0}s of raw audio…");
                    self.effects
                        .push(Effect::Mic(MicRequest::SelfTest { path, seconds }));
                } else {
                    self.system(&format!(
                        "mic self-test unavailable: microphone {}",
                        self.mic.describe()
                    ));
                }
            }
            Command::SelectMic(sel) => {
                if matches!(self.stage(), VoiceStage::Listening) {
                    self.cancel_active("switching microphone");
                }
                self.mic = Readiness::Loading;
                self.effects.push(Effect::Mic(MicRequest::Select(sel)));
            }
        }
    }

    fn offer(&mut self, id: OfferId, accept: bool) {
        let action = if accept {
            Action::AcceptOffer { offer: id }
        } else {
            Action::RejectOffer { offer: id }
        };
        if let ValidationOutcome::Rejected(reason) = self.session.player_do(action) {
            self.system(&format!("offer #{}: {reason:?}", id.raw()));
        }
    }

    fn ptt_down(&mut self) {
        match self.stage() {
            VoiceStage::Idle => self.start_listening(),
            // Key repeat, or a second press while already recording.
            VoiceStage::Listening => {}
            // Barge-in: talking over an NPC interrupts them.
            VoiceStage::Synthesizing | VoiceStage::Speaking => {
                self.counters.interrupted += 1;
                self.cancel_active("interrupted");
                self.start_listening();
            }
            VoiceStage::Transcribing | VoiceStage::Deciding | VoiceStage::Validating => {
                // Never queue a second turn silently behind the first.
                self.counters.refused_ptt += 1;
                self.status = "still answering your last line — press C to cancel it".into();
            }
        }
    }

    fn ptt_up(&mut self) {
        if let Some(c) = &self.conv {
            if c.stage == VoiceStage::Listening {
                let conv = c.id;
                self.status = "processing recording…".into();
                self.effects.push(Effect::Mic(MicRequest::Stop { conv }));
            }
        }
    }

    fn start_listening(&mut self) {
        if !self.mic.is_ready() {
            self.system(&format!(
                "microphone {} — F7 switches device, Enter types instead",
                self.mic.describe()
            ));
            return;
        }
        let id = self.begin_conversation();
        self.set_stage(VoiceStage::Listening);
        self.status = "listening — release V when done (C cancels)".into();
        self.effects
            .push(Effect::Mic(MicRequest::Start { conv: id }));
    }

    fn typed(&mut self, text: String) {
        let text = text.trim().to_string();
        if text.is_empty() {
            return;
        }
        match self.stage() {
            VoiceStage::Idle => {}
            VoiceStage::Listening => self.cancel_active("replaced by typed line"),
            VoiceStage::Synthesizing | VoiceStage::Speaking => {
                self.counters.interrupted += 1;
                self.cancel_active("interrupted");
            }
            VoiceStage::Transcribing | VoiceStage::Deciding | VoiceStage::Validating => {
                self.counters.refused_ptt += 1;
                self.status = "still answering your last line — press C to cancel it".into();
                return;
            }
        }
        self.begin_conversation();
        self.player_turn(text);
    }

    // -----------------------------------------------------------------
    // The shared gameplay path for typed and spoken dialogue
    // -----------------------------------------------------------------

    fn player_turn(&mut self, text: String) {
        self.dialogue_push(DialogueLine {
            speaker: "you".into(),
            text: text.clone(),
            kind: LineKind::Player,
        });

        // An active conversation is authoritative; proximity only chooses the
        // first addressee. This is what stops the nearest NPC from stealing a
        // continuing conversation.
        let npc = match self.resolve_addressee() {
            Some(n) => n,
            None => {
                self.system("nobody is close enough to hear you");
                self.finish(false);
                return;
            }
        };

        self.begin_active_conversation(npc.clone());

        // Record the player's turn for short-term memory.
        if let Some(ac) = self.active_conversation.as_mut() {
            ac.turns.push_back(ConversationTurn {
                speaker: ActorId::new(PLAYER),
                text: text.clone(),
            });
            ac.last_active = self.now;
            while ac.turns.len() > MAX_CONVERSATION_TURNS {
                ac.turns.pop_front();
            }
        }

        let out = self.session.player_speak(npc.as_str(), &text);
        if let ValidationOutcome::Rejected(reason) = out {
            let name = actor_name(&self.session.sim.world, &npc);
            self.system(&format!("{name} did not hear you ({reason:?})"));
            self.finish(false);
            return;
        }
        if let Some(c) = self.conv.as_mut() {
            c.npc = Some(npc);
            c.utterance = Some(text);
        }
        self.request_decision();
    }

    /// Resolve who the player is addressing. An active conversation is
    /// authoritative until the NPC leaves or the player explicitly switches.
    fn resolve_addressee(&self) -> Option<ActorId> {
        if let Some(ac) = &self.active_conversation {
            if self.npc_can_participate(&ac.npc) {
                return Some(ac.npc.clone());
            }
        }
        self.nearby()
    }

    /// Ask for the active conversation's NPC decision: synchronously from the
    /// deterministic policy when no real model is in use, otherwise as an
    /// asynchronous job for the inference worker.
    fn request_decision(&mut self) {
        let Some(c) = self.conv.as_ref() else { return };
        let conv = c.id;
        let (Some(npc), Some(text)) = (c.npc.clone(), c.utterance.clone()) else {
            return;
        };
        let player = ActorId::new(PLAYER);
        let fallback_reason = match &self.engine {
            EngineState::Ready { real: true, .. } | EngineState::Loading => None,
            EngineState::Ready { real: false, .. } => Some(None),
            EngineState::Failed(e) => Some(Some(format!("model unavailable: {e}"))),
        };
        if let Some(reason) = fallback_reason {
            self.set_stage(VoiceStage::Validating);
            let ctx = NpcRequestContext {
                world: &self.session.sim.world,
                npc: &npc,
                speaker: &player,
                utterance: &text,
                history: &[],
            };
            let decision = NpcPolicy.heuristic(&ctx, reason);
            self.commit(npc, decision, false);
            return;
        }
        let request_id = self.session.next_request_id();
        // Build prior-turn history, excluding the player's current utterance
        // (recorded in `turns` by `player_turn`) because that utterance is
        // passed separately to the model prompt. Including it here would
        // render the same line twice.
        let history: Vec<(ActorId, String)> = self
            .active_conversation
            .as_ref()
            .filter(|ac| ac.npc == npc)
            .map(|ac| {
                let prior = ac.turns.len().saturating_sub(1);
                ac.turns
                    .iter()
                    .take(prior)
                    .map(|t| (t.speaker.clone(), t.text.clone()))
                    .collect()
            })
            .unwrap_or_default();
        let request = NpcPolicy.request(
            &NpcRequestContext {
                world: &self.session.sim.world,
                npc: &npc,
                speaker: &player,
                utterance: &text,
                history: &history,
            },
            request_id,
        );
        if let Some(c) = self.conv.as_mut() {
            c.pending = Some(PendingDecision {
                request_id,
                world_epoch: self.world_epoch,
                request: request.clone(),
            });
        }
        self.set_stage(VoiceStage::Deciding);
        self.inference.requests += 1;
        self.last.stream_chunks = None;
        self.status = format!("{} is thinking…", actor_name(&self.session.sim.world, &npc));
        self.effects
            .push(Effect::Infer(InferRequest::Decide(InferJob {
                conv,
                request_id,
                world_epoch: self.world_epoch,
                request,
            })));
    }

    /// Validate and commit a decision; voice it if it carries words.
    fn commit(&mut self, npc: ActorId, decision: rpg_agents::NpcDecision, from_model: bool) {
        let name = actor_name(&self.session.sim.world, &npc);
        let commit = self.session.commit_npc_decision(npc.as_str(), decision);
        self.last_decision = Some(commit.source.label());
        if let ValidationOutcome::Rejected(reason) = &commit.outcome {
            self.rejected_actions += 1;
            let can_retry = from_model
                && self
                    .conv
                    .as_ref()
                    .is_some_and(|c| c.regenerations < self.cfg.max_regenerations);
            if can_retry {
                // The world moved under the proposal: ask again from the
                // current state rather than voicing something impossible.
                self.counters.regenerations += 1;
                if let Some(c) = self.conv.as_mut() {
                    c.regenerations += 1;
                }
                self.request_decision();
                return;
            }
            self.system(&format!("{name} does not answer ({reason:?})"));
            self.finish(true);
            return;
        }
        match commit.spoken {
            Some(line) => {
                // Record the NPC's reply for short-term memory.
                if let Some(ac) = self.active_conversation.as_mut() {
                    if ac.npc == npc {
                        ac.turns.push_back(ConversationTurn {
                            speaker: npc.clone(),
                            text: line.clone(),
                        });
                        ac.last_active = self.now;
                        while ac.turns.len() > MAX_CONVERSATION_TURNS {
                            ac.turns.pop_front();
                        }
                    }
                }
                self.dialogue_push(DialogueLine {
                    speaker: npc.as_str().to_string(),
                    text: line.clone(),
                    kind: LineKind::Npc,
                });
                if self.tts.is_ready() {
                    let id = self.next_line;
                    self.next_line += 1;
                    let conv = self.conv.as_ref().map(|c| c.id).unwrap_or_default();
                    if let Some(c) = self.conv.as_mut() {
                        c.line = Some(id);
                    }
                    self.set_stage(VoiceStage::Synthesizing);
                    self.status = format!("{name} is speaking");
                    self.effects.push(Effect::Tts(TtsRequest::Speak(SpeakJob {
                        conv,
                        line: id,
                        speaker: npc.as_str().to_string(),
                        text: line,
                        audio_epoch: self.audio_epoch,
                    })));
                } else {
                    self.finish(true);
                }
            }
            None => {
                // The NPC acted without words. Show what the simulation did,
                // so silence is never mistaken for broken audio.
                let trace = self.session.trace.last().cloned().unwrap_or_default();
                self.dialogue_push(DialogueLine {
                    speaker: npc.as_str().to_string(),
                    text: format!("(no spoken reply) {trace}"),
                    kind: LineKind::System,
                });
                self.finish(true);
            }
        }
    }

    // -----------------------------------------------------------------
    // Worker events
    // -----------------------------------------------------------------

    fn on_mic(&mut self, ev: MicEvent) {
        match ev {
            MicEvent::DeviceReady(info) => {
                self.mic = Readiness::Ready(info.describe());
                self.mic_info = Some(info);
            }
            MicEvent::DeviceError(e) => {
                self.mic = Readiness::Unavailable(e.clone());
                if self.stage() == VoiceStage::Listening {
                    self.cancel_active("microphone failed");
                }
                self.system(&format!("microphone error: {e} (retrying)"));
            }
            MicEvent::Devices(list) => self.mic_devices = list,
            MicEvent::Listening { conv, .. } => {
                if !self.is_current(conv, VoiceStage::Listening) {
                    self.counters.stale_responses += 1;
                }
            }
            MicEvent::Captured {
                conv,
                samples,
                sample_rate,
                recorded_seconds,
                ..
            } => {
                if !self.is_current(conv, VoiceStage::Listening) {
                    self.counters.stale_responses += 1;
                    return;
                }
                self.last.recorded_seconds = recorded_seconds;
                if !self.asr.is_ready() {
                    self.system(&format!("speech recognizer {}", self.asr.describe()));
                    self.finish(false);
                    return;
                }
                self.set_stage(VoiceStage::Transcribing);
                self.status = "transcribing…".into();
                self.effects.push(Effect::Asr(AsrRequest::Transcribe {
                    conv,
                    samples,
                    sample_rate,
                }));
            }
            MicEvent::NoSpeech {
                conv,
                held_seconds,
                recorded_seconds,
                rms,
                peak,
            } => {
                if !self.is_current(conv, VoiceStage::Listening) {
                    self.counters.stale_responses += 1;
                    return;
                }
                self.last.recorded_seconds = recorded_seconds;
                let msg = if held_seconds < 0.25 {
                    "that was too short — hold V while you speak, release when done".to_string()
                } else if recorded_seconds < held_seconds * 0.5 {
                    format!(
                        "the microphone delivered only {recorded_seconds:.1}s of audio in {held_seconds:.1}s — capture is failing; try F7 or the self-test (F8)"
                    )
                } else if peak < 1e-4 {
                    format!(
                        "no audio from the microphone ({recorded_seconds:.1}s of digital silence) — check the device (F7) or run the self-test (F8)"
                    )
                } else if peak < 0.02 {
                    format!(
                        "microphone is very quiet (peak {peak:.3}, rms {rms:.4}) — speak closer or raise the input gain"
                    )
                } else {
                    format!("heard sound but no speech (peak {peak:.2}) — hold V while you speak")
                };
                self.system(&msg);
                self.finish(false);
            }
            MicEvent::SelfTest(result) => {
                let text = match result {
                    Ok(report) => report.summary(),
                    Err(e) => format!("mic self-test failed: {e}"),
                };
                self.status = "ready".into();
                self.system(&text);
            }
        }
    }

    fn on_asr(&mut self, ev: AsrEvent) {
        match ev {
            AsrEvent::Ready { backend } => self.asr = Readiness::Ready(backend),
            AsrEvent::Unavailable(e) => self.asr = Readiness::Unavailable(e),
            AsrEvent::Transcript {
                conv,
                text,
                latency_ms,
            } => {
                if !self.is_current(conv, VoiceStage::Transcribing) {
                    self.counters.stale_responses += 1;
                    return;
                }
                self.last.asr_ms = Some(latency_ms);
                let text = text.trim().to_string();
                self.last.transcript = Some(text.clone());
                if text.is_empty() {
                    self.system("did not catch that — hold V and speak clearly");
                    self.finish(false);
                    return;
                }
                self.player_turn(text);
            }
            AsrEvent::Failed { conv, error } => {
                if !self.is_current(conv, VoiceStage::Transcribing) {
                    self.counters.stale_responses += 1;
                    return;
                }
                self.system(&format!("did not catch that ({error})"));
                self.finish(false);
            }
        }
    }

    fn on_infer(&mut self, ev: InferEvent) {
        match ev {
            InferEvent::Ready {
                engine,
                model_id,
                is_real_model,
            } => {
                self.engine = EngineState::Ready {
                    name: engine,
                    model_id,
                    real: is_real_model,
                };
            }
            InferEvent::LoadFailed(e) => {
                self.system(&format!(
                    "model unavailable: {e} — using the deterministic policy"
                ));
                self.engine = EngineState::Failed(e);
            }
            InferEvent::Progress {
                conv,
                request_id,
                chunks,
            } => {
                let current = self.conv.as_ref().is_some_and(|c| {
                    c.id == conv
                        && c.stage == VoiceStage::Deciding
                        && c.pending
                            .as_ref()
                            .is_some_and(|p| p.request_id == request_id)
                });
                if current {
                    self.last.stream_chunks = Some(chunks);
                    if let Some(npc) = self.conv.as_ref().and_then(|c| c.npc.clone()) {
                        let name = actor_name(&self.session.sim.world, &npc);
                        self.status = format!("{name} is thinking… (answer arriving)");
                    }
                }
            }
            InferEvent::Done(done) => self.on_decision(done),
        }
    }

    fn on_decision(&mut self, done: InferDone) {
        // Account for every answer, including stale ones: a late answer was
        // still generated (and, for a remote provider, billed).
        self.inference.record(&done);
        let current = self.conv.as_ref().is_some_and(|c| {
            c.id == done.conv
                && c.stage == VoiceStage::Deciding
                && c.pending.as_ref().is_some_and(|p| {
                    p.request_id == done.request_id && p.world_epoch == done.world_epoch
                })
        });
        if !current || done.world_epoch != self.world_epoch {
            self.counters.stale_responses += 1;
            return;
        }
        self.last.llm_queue_ms = Some(done.queue_ms);
        self.last.llm_infer_ms = Some(done.infer_ms);
        self.last.llm_first_ms = done.result.as_ref().ok().map(|r| r.first_token_ms);
        self.resolve_pending(done.result);
    }

    /// Validate a decision result (or a timeout) against the current world.
    fn resolve_pending(
        &mut self,
        result: Result<rpg_domain::ports::InferenceResponse, InferenceError>,
    ) {
        let Some(c) = self.conv.as_mut() else { return };
        let Some(pending) = c.pending.take() else {
            return;
        };
        let (Some(npc), Some(text)) = (c.npc.clone(), c.utterance.clone()) else {
            return;
        };
        self.set_stage(VoiceStage::Validating);
        if let Err(why) = self.can_still_answer(&npc) {
            self.counters.invalidated += 1;
            let name = actor_name(&self.session.sim.world, &npc);
            self.system(&format!("{name} {why} before answering"));
            self.finish(true);
            return;
        }
        let from_model = result.is_ok();
        let model_id = match &self.engine {
            EngineState::Ready { model_id, .. } => model_id.clone(),
            _ => "unknown".into(),
        };
        let decision = NpcPolicy.resolve(
            &NpcRequestContext {
                world: &self.session.sim.world,
                npc: &npc,
                speaker: &ActorId::new(PLAYER),
                utterance: &text,
                history: &[],
            },
            pending.request,
            result,
            &model_id,
        );
        self.commit(npc, decision, from_model);
    }

    fn on_tts(&mut self, ev: TtsEvent) {
        match ev {
            TtsEvent::Ready {
                backend,
                device,
                sample_rate,
            } => {
                self.tts = Readiness::Ready(format!("{backend} → {device} @ {sample_rate} Hz"));
            }
            TtsEvent::Unavailable(e) => self.tts = Readiness::Unavailable(e),
            TtsEvent::DeviceError(e) => self.system(&format!("speaker error: {e}")),
            TtsEvent::Started {
                conv,
                line,
                first_audio_ms,
            } => {
                if !self.is_current_line(conv, line, VoiceStage::Synthesizing) {
                    self.counters.stale_responses += 1;
                    return;
                }
                self.last.tts_first_ms = Some(first_audio_ms);
                self.set_stage(VoiceStage::Speaking);
            }
            TtsEvent::Finished {
                conv,
                line,
                synth_ms,
                ..
            } => {
                let current = self.is_current_line(conv, line, VoiceStage::Speaking)
                    || self.is_current_line(conv, line, VoiceStage::Synthesizing);
                if !current {
                    self.counters.stale_responses += 1;
                    return;
                }
                self.last.tts_synth_ms = Some(synth_ms);
                self.finish(true);
            }
            TtsEvent::Cancelled { .. } => {}
            TtsEvent::Failed { conv, line, error } => {
                let current = self.is_current_line(conv, line, VoiceStage::Speaking)
                    || self.is_current_line(conv, line, VoiceStage::Synthesizing);
                if !current {
                    self.counters.stale_responses += 1;
                    return;
                }
                self.system(&format!("voice failed: {error}"));
                self.finish(true);
            }
        }
    }

    // -----------------------------------------------------------------
    // Lifecycle helpers
    // -----------------------------------------------------------------

    fn begin_conversation(&mut self) -> ConvId {
        let id = self.next_conv;
        self.next_conv += 1;
        self.counters.conversations += 1;
        self.conv = Some(Conversation {
            id,
            stage: VoiceStage::Idle,
            stage_since: self.now,
            npc: None,
            utterance: None,
            pending: None,
            line: None,
            regenerations: 0,
        });
        id
    }

    /// Start or refresh a persistent conversation with an NPC.
    fn begin_active_conversation(&mut self, npc: ActorId) {
        let already = self
            .active_conversation
            .as_ref()
            .map(|ac| ac.npc == npc)
            .unwrap_or(false);
        if already {
            if let Some(ac) = self.active_conversation.as_mut() {
                ac.last_active = self.now;
            }
            return;
        }
        self.active_conversation = Some(ActiveConversation {
            npc,
            last_active: self.now,
            turns: VecDeque::new(),
        });
    }

    fn set_stage(&mut self, stage: VoiceStage) {
        if let Some(c) = self.conv.as_mut() {
            c.stage = stage;
            c.stage_since = self.now;
            self.transitions.push_back((c.id, stage));
            while self.transitions.len() > 64 {
                self.transitions.pop_front();
            }
        }
    }

    /// End the active conversation normally (or after a handled failure).
    fn finish(&mut self, reached_npc: bool) {
        if let Some(c) = self.conv.take() {
            self.transitions.push_back((c.id, VoiceStage::Idle));
            while self.transitions.len() > 64 {
                self.transitions.pop_front();
            }
        }
        if reached_npc {
            self.counters.completed += 1;
        }
        self.status = "ready".into();
    }

    /// Cancel the active conversation (player request, barge-in, quit).
    fn cancel_active(&mut self, why: &str) {
        if self.abort_active(why) {
            self.counters.cancelled += 1;
        }
    }

    /// Abort the active conversation at whatever stage it is in, telling the
    /// worker that owns that stage to stop, and silence any of its audio.
    /// Returns whether there was a conversation to abort.
    fn abort_active(&mut self, why: &str) -> bool {
        let Some(c) = self.conv.as_ref() else {
            return false;
        };
        let conv = c.id;
        match c.stage {
            VoiceStage::Listening => self.effects.push(Effect::Mic(MicRequest::Cancel { conv })),
            VoiceStage::Transcribing => self.effects.push(Effect::Asr(AsrRequest::Cancel { conv })),
            VoiceStage::Deciding => {
                if let Some(p) = &c.pending {
                    self.effects.push(Effect::Infer(InferRequest::Cancel {
                        request_id: p.request_id,
                    }));
                }
            }
            VoiceStage::Synthesizing | VoiceStage::Speaking => self.bump_audio_epoch(),
            VoiceStage::Validating | VoiceStage::Idle => {}
        }
        self.finish(false);
        self.status = why.to_string();
        true
    }

    fn bump_audio_epoch(&mut self) {
        self.audio_epoch += 1;
        self.effects.push(Effect::SetAudioEpoch(self.audio_epoch));
    }

    fn is_current(&self, conv: ConvId, stage: VoiceStage) -> bool {
        self.conv
            .as_ref()
            .is_some_and(|c| c.id == conv && c.stage == stage)
    }

    fn is_current_line(&self, conv: ConvId, line: u64, stage: VoiceStage) -> bool {
        self.conv
            .as_ref()
            .is_some_and(|c| c.id == conv && c.stage == stage && c.line == Some(line))
    }

    fn check_deadlines(&mut self) {
        let Some(c) = self.conv.as_ref() else { return };
        let elapsed = self.now.saturating_sub(c.stage_since);
        let limit = match c.stage {
            VoiceStage::Listening => self.cfg.listen_timeout,
            VoiceStage::Transcribing => self.cfg.transcribe_timeout,
            VoiceStage::Deciding => self.cfg.decide_timeout,
            VoiceStage::Synthesizing => self.cfg.synth_timeout,
            VoiceStage::Speaking => self.cfg.speak_timeout,
            VoiceStage::Validating | VoiceStage::Idle => return,
        };
        if elapsed < limit {
            return;
        }
        self.counters.timeouts += 1;
        match c.stage {
            VoiceStage::Deciding => {
                // Stop the model and let the deterministic policy answer, so
                // a stalled model never leaves the NPC (or the UI) stuck.
                if let Some(p) = &c.pending {
                    self.effects.push(Effect::Infer(InferRequest::Cancel {
                        request_id: p.request_id,
                    }));
                }
                let ms = limit.as_millis() as u64;
                self.resolve_pending(Err(InferenceError::Timeout(ms)));
            }
            stage => {
                let what = stage.label();
                self.abort_active(&format!("{what} timed out"));
                self.system(&format!(
                    "{what} timed out after {:.0}s",
                    limit.as_secs_f32()
                ));
            }
        }
    }

    /// Whether the NPC can still deliver an answer in the *current* world.
    fn can_still_answer(&self, npc: &ActorId) -> Result<(), String> {
        let world = &self.session.sim.world;
        let player = world
            .actor(&ActorId::new(PLAYER))
            .ok_or("lost track of you")?;
        let Some(a) = world.actor(npc) else {
            return Err("is gone".into());
        };
        match a.status {
            ActorStatus::Alive => {}
            ActorStatus::Dead => return Err("died".into()),
            ActorStatus::Departed => return Err("left town".into()),
        }
        if a.location != player.location {
            return Err("walked away".into());
        }
        if a.pos.distance(player.pos) > self.cfg.talk_radius + self.cfg.answer_slack {
            return Err("moved out of earshot".into());
        }
        Ok(())
    }

    // -----------------------------------------------------------------
    // Save / load
    // -----------------------------------------------------------------

    /// Persist canonical state. The save file is versioned and contains the
    /// simulation seed, world snapshot and event checksum; loading it must
    /// reproduce the same canonical outcomes. In-flight conversations are
    /// transient and deliberately not saved.
    fn save(&mut self, path: &str) {
        let save = rpg_persistence::SaveFile::from_sim(&self.session.sim);
        let message = match rpg_persistence::save_json(&save, path) {
            Ok(()) => format!("saved to {path}"),
            Err(e) => format!("save failed: {e}"),
        };
        self.system(&message);
    }

    /// Restore canonical state. Anything in flight belongs to the old
    /// timeline: it is cancelled, the world epoch moves on (so late model
    /// answers are recognised as stale) and queued audio is silenced.
    fn load(&mut self, path: &str) {
        match rpg_persistence::load_json(path) {
            Ok(save) => self.restore_save(save, path),
            Err(e) => self.system(&format!("load failed: {e}")),
        }
    }

    /// The platform-neutral half of [`SimCore::save`]: produce the canonical
    /// save JSON and hand it to the platform to store.
    fn export_save(&mut self, slot: String) {
        let save = rpg_persistence::SaveFile::from_sim(&self.session.sim);
        match rpg_persistence::to_json_string(&save) {
            Ok(json) => self.effects.push(Effect::SaveData { slot, json }),
            Err(e) => self.system(&format!("save failed: {e}")),
        }
    }

    /// Replace the canonical state with a validated save, on any platform.
    fn restore_save(&mut self, save: rpg_persistence::SaveFile, label: &str) {
        self.abort_active("loaded a save");
        self.bump_audio_epoch();
        self.world_epoch += 1;
        let config = rpg_sim::SimConfig {
            enable_background: true,
            enable_director: true,
            ..rpg_sim::SimConfig::default()
        };
        self.session.sim = rpg_persistence::restore(save, config);
        self.dialogue.clear();
        self.system(&format!("loaded from {label}"));
    }

    // -----------------------------------------------------------------
    // Output
    // -----------------------------------------------------------------

    fn system(&mut self, text: &str) {
        self.dialogue_push(DialogueLine {
            speaker: "system".into(),
            text: text.to_string(),
            kind: LineKind::System,
        });
    }

    fn dialogue_push(&mut self, line: DialogueLine) {
        self.dialogue.push(line);
        let excess = self.dialogue.len().saturating_sub(self.cfg.dialogue_limit);
        if excess > 0 {
            self.dialogue.drain(..excess);
        }
    }

    /// All living named NPCs in the same room within the talk radius, sorted
    /// by distance to the player.
    pub fn nearby_all(&self) -> Vec<ActorId> {
        let world = &self.session.sim.world;
        let Some(player) = world.actor(&ActorId::new(PLAYER)) else {
            return Vec::new();
        };
        let mut candidates: Vec<_> = world
            .actors
            .values()
            .filter(|a| a.id.0 != PLAYER)
            .filter(|a| a.kind == ActorKind::Npc)
            .filter(|a| a.status == ActorStatus::Alive)
            .filter(|a| a.location == player.location)
            .filter(|a| a.pos.distance(player.pos) <= self.cfg.talk_radius)
            .map(|a| (a.pos.distance(player.pos), a.id.clone()))
            .collect();
        candidates.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        candidates.into_iter().map(|(_, id)| id).collect()
    }

    /// Nearest living named NPC in the same room, within the talk radius.
    pub fn nearby(&self) -> Option<ActorId> {
        let world = &self.session.sim.world;
        let player = world.actor(&ActorId::new(PLAYER))?;
        world
            .actors
            .values()
            .filter(|a| a.id.0 != PLAYER)
            .filter(|a| a.kind == ActorKind::Npc)
            .filter(|a| a.status == ActorStatus::Alive)
            .filter(|a| a.location == player.location)
            .filter(|a| a.pos.distance(player.pos) <= self.cfg.talk_radius)
            .min_by(|a, b| {
                a.pos
                    .distance(player.pos)
                    .partial_cmp(&b.pos.distance(player.pos))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|a| a.id.clone())
    }

    /// Build the immutable view the renderer draws.
    pub fn snapshot(&self) -> Snapshot {
        let world = &self.session.sim.world;
        let player = world.actor(&ActorId::new(PLAYER));
        let player_pos = player.map(|a| a.pos).unwrap_or(Vec2::ZERO);
        let player_location = player
            .map(|a| a.location.as_str().to_string())
            .unwrap_or_default();
        let nearby = self.nearby().map(|a| a.as_str().to_string());

        let locations = world
            .locations
            .values()
            .map(|l| LocationView {
                id: l.id.as_str().to_string(),
                name: l.name.clone(),
                x: l.bounds.x,
                y: l.bounds.y,
                w: l.bounds.w,
                h: l.bounds.h,
                private: l.private,
            })
            .collect();

        let actors = world
            .actors
            .values()
            .map(|a| ActorView {
                id: a.id.as_str().to_string(),
                name: a.name.clone(),
                pos: a.pos,
                kind: match a.kind {
                    ActorKind::Player => KindView::Player,
                    ActorKind::Npc => KindView::Npc,
                    ActorKind::Background => KindView::Background,
                },
                status: match a.status {
                    ActorStatus::Alive => StatusView::Alive,
                    ActorStatus::Dead => StatusView::Dead,
                    ActorStatus::Departed => StatusView::Departed,
                },
                location: a.location.as_str().to_string(),
                nearby: nearby.as_deref() == Some(a.id.as_str()),
            })
            .collect();

        let offers = world
            .offers
            .values()
            .map(|o| OfferView {
                id: o.id.raw(),
                from: actor_name(world, &o.from),
                to: actor_name(world, &o.to),
                to_player: o.to.0 == PLAYER,
                give: format_stacks(world, &o.give),
                want: format_stacks(world, &o.want),
                price: o.price,
                note: o.note.clone(),
                state: format!("{:?}", o.state).to_ascii_lowercase(),
            })
            .collect();

        let commitments = world
            .commitments
            .values()
            .map(|c| CommitmentView {
                id: c.id.raw(),
                kind: format!("{:?}", c.kind),
                debtor: actor_name(world, &c.debtor),
                creditor: actor_name(world, &c.creditor),
                terms: c.terms.clone(),
                state: format!("{:?}", c.state),
            })
            .collect();

        let player_inv = player.map(|a| &a.inventory);
        let inventory = player_inv
            .map(|inv| {
                inv.items
                    .iter()
                    .map(|s| ItemView {
                        name: world
                            .items
                            .get(&s.item)
                            .map(|i| i.name.clone())
                            .unwrap_or_else(|| s.item.as_str().to_string()),
                        qty: s.qty,
                    })
                    .collect()
            })
            .unwrap_or_default();
        let money = player_inv.map(|inv| inv.money).unwrap_or(0);

        // --- Developer diagnostics (F3 overlay only) ---------------------
        let nearby_beliefs = nearby
            .as_deref()
            .and_then(|id| world.actor(&ActorId::new(id)))
            .map(|a| {
                a.beliefs
                    .keys()
                    .filter_map(|f| world.facts.get(f))
                    .map(|f| format!("[{:?}] {}", f.sensitivity, f.statement).to_ascii_lowercase())
                    .collect()
            })
            .unwrap_or_default();
        let nearby_relationship = nearby
            .as_deref()
            .and_then(|id| world.actor(&ActorId::new(id)))
            .map(|a| {
                let r = a.relationship(&ActorId::new(PLAYER));
                format!(
                    "trust {:.2}  affection {:.2}  respect {:.2}  fear {:.2}  anger {:.2}  suspicion {:.2}  familiar {:.2}",
                    r.trust, r.affection, r.respect, r.fear, r.anger, r.suspicion, r.familiarity
                )
            });
        let perception = vec![
            format!("conversation radius {}", self.cfg.talk_radius),
            format!(
                "hearing: whisper {} / normal {} / loud {} / shout {}",
                Loudness::Whisper.radius(),
                Loudness::Normal.radius(),
                Loudness::Loud.radius(),
                Loudness::Shout.radius()
            ),
            "line of sight is wall-occluded (rpg_perception)".to_string(),
        ];
        let recent_events = self
            .session
            .sim
            .events
            .recent(8)
            .iter()
            .map(|e| format!("t{}  {}", e.tick, e.describe()))
            .collect();

        let (engine_name, model_id) = match &self.engine {
            EngineState::Loading => ("loading model…".to_string(), String::new()),
            EngineState::Ready { name, model_id, .. } => (name.clone(), model_id.clone()),
            EngineState::Failed(e) => (format!("deterministic (model failed: {e})"), String::new()),
        };

        let t = &self.telemetry;
        let stage = self.stage();
        let voice = VoiceView {
            stage,
            conversation: self.conversation_id(),
            npc: self
                .conv
                .as_ref()
                .and_then(|c| c.npc.as_ref())
                .or_else(|| self.active_conversation.as_ref().map(|ac| &ac.npc))
                .map(|n| n.as_str().to_string()),
            stage_seconds: self
                .conv
                .as_ref()
                .map(|c| self.now.saturating_sub(c.stage_since).as_secs_f32())
                .unwrap_or(0.0),
            mic: self.mic.clone(),
            asr: self.asr.clone(),
            tts: self.tts.clone(),
            mic_rms: t.mic_rms.get(),
            mic_peak: t.mic_peak.get(),
            speech_detected: t.vad_speaking.load(Ordering::Relaxed),
            last_transcript: self.last.transcript.clone(),
        };
        let active = t.active_inference.load(Ordering::Relaxed);
        let diagnostics = Diagnostics {
            platform: String::new(),
            ticks_per_second: t.ticks_per_second.get(),
            target_ticks_per_second: 0.0,
            tick_delay_ms: t.tick_delay_ms.get(),
            max_tick_lag_ms: t.max_tick_lag_ms.get(),
            skipped_ticks: t.skipped_ticks.load(Ordering::Relaxed),
            mic_device: self
                .mic_info
                .as_ref()
                .map(|m| m.describe())
                .unwrap_or_else(|| self.mic.describe()),
            mic_samples_per_second: t.mic_samples_per_second.get(),
            mic_rms: t.mic_rms.get(),
            mic_peak: t.mic_peak.get(),
            mic_dropped_chunks: t.mic_dropped_chunks.load(Ordering::Relaxed),
            vad_speaking: t.vad_speaking.load(Ordering::Relaxed),
            recording_seconds: if stage == VoiceStage::Listening {
                t.recording_ms.load(Ordering::Relaxed) as f32 / 1000.0
            } else {
                self.last.recorded_seconds
            },
            last_transcript: self.last.transcript.clone(),
            asr_latency_ms: self.last.asr_ms,
            conversation: self.conversation_id(),
            request_id: self.pending_request_id(),
            active_inference: (active != NONE).then_some(active),
            llm_queue_ms: self.last.llm_queue_ms,
            llm_infer_ms: self.last.llm_infer_ms,
            llm_first_token_ms: self.last.llm_first_ms,
            llm_stream_chunks: self.last.stream_chunks,
            pending_inference: usize::from(self.pending_request_id().is_some()),
            inference: self.inference.clone(),
            last_decision: self.last_decision.clone(),
            rejected_actions: self.rejected_actions,
            tts_first_audio_ms: self.last.tts_first_ms,
            tts_synth_ms: self.last.tts_synth_ms,
            playback: t.playback_status(),
            playback_underruns: t.playback_underruns.load(Ordering::Relaxed),
            asr_queue: t.asr_queue.load(Ordering::Relaxed),
            infer_queue: t.infer_queue.load(Ordering::Relaxed),
            tts_queue: t.tts_queue.load(Ordering::Relaxed),
            cancelled: self.counters.cancelled,
            stale_responses: self.counters.stale_responses,
            timeouts: self.counters.timeouts,
            refused_ptt: self.counters.refused_ptt,
            regenerations: self.counters.regenerations,
            invalidated: self.counters.invalidated,
            transitions: self
                .transitions
                .iter()
                .rev()
                .take(8)
                .map(|(c, s)| format!("#{c} {}", s.label()))
                .collect(),
        };

        Snapshot {
            tick: world.tick,
            day: world.day,
            minute_of_day: world.minute_of_day,
            world_epoch: self.world_epoch,
            locations,
            actors,
            player_pos,
            player_location,
            nearby,
            talking_to: self
                .active_conversation
                .as_ref()
                .map(|ac| ac.npc.as_str().to_string()),
            dialogue: self.dialogue.clone(),
            engine_name,
            model_id,
            model_decisions: self.session.model_decisions(),
            canonical_decisions: self.session.canonical_decisions(),
            fallback_decisions: self.session.fallback_decisions(),
            status: self.status.clone(),
            voice,
            diagnostics,
            recent_events,
            nearby_beliefs,
            nearby_relationship,
            perception,
            offers,
            commitments,
            inventory,
            money,
        }
    }
}

/// Human-readable name for an actor id (falls back to the raw id).
fn actor_name(world: &World, id: &ActorId) -> String {
    world
        .actor(id)
        .map(|a| a.name.clone())
        .unwrap_or_else(|| id.as_str().to_string())
}

/// "medicine x2, ledger" for display in offers.
fn format_stacks(world: &World, stacks: &[rpg_domain::world::ItemStack]) -> String {
    if stacks.is_empty() {
        return "—".into();
    }
    stacks
        .iter()
        .map(|s| {
            let name = world
                .items
                .get(&s.item)
                .map(|i| i.name.clone())
                .unwrap_or_else(|| s.item.as_str().to_string());
            if s.qty > 1 {
                format!("{name} x{}", s.qty)
            } else {
                name
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}
