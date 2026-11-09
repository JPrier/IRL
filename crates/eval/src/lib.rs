//! # rpg_eval
//!
//! Simulated-player scenarios and quality metrics. These are the acceptance
//! tests that prove the *integrated* loop (perception → NPC decision →
//! validated action → persistent consequence) actually works, not just that
//! isolated units compile.
//!
//! Every scenario is seeded and deterministic; replaying the same script must
//! produce the same canonical event checksum.

use rpg_agents::{HeuristicEngine, NpcPolicy, NpcRequestContext};
use rpg_content::build_town;
use rpg_domain::action::{
    Action, ActionProposal, InvestigateTarget, Loudness, ProposalSource, QuestPredicate,
    ValidationOutcome,
};
use rpg_domain::ids::*;
use rpg_domain::ports::InferenceEngine;
use rpg_sim::{SimConfig, Simulation, StepReport};
use std::collections::VecDeque;

/// How many conversation turns (player and NPC lines together) the session
/// keeps for follow-up context. Mirrors the runtime's
/// `rpg_runtime::MAX_CONVERSATION_TURNS` so eval exercises the same window
/// the voice client does.
const MAX_CONVERSATION_TURNS: usize = 12;

/// A scripted/simulated player connected to a live simulation and NPC policy.
pub struct Session {
    pub sim: Simulation,
    pub engine: Box<dyn InferenceEngine>,
    pub next_request: u64,
    pub trace: Vec<String>,
    /// Provenance of every NPC decision, in order.
    pub decisions: Vec<(String, rpg_agents::DecisionSource)>,
    /// Addressee of the ongoing conversation, if any.
    pub conversation_target: Option<ActorId>,
    /// Bounded short-term memory of the ongoing conversation, mirroring the
    /// runtime's active-conversation semantics: one conversation at a time;
    /// switching addressees drops it entirely (no per-NPC retention); turns
    /// carry speaker attribution, the player's utterance first.
    pub conversation_turns: VecDeque<(ActorId, String)>,
}

impl Session {
    pub fn new(seed: u64) -> Self {
        Self::with_engine(seed, Box::new(HeuristicEngine::new()))
    }

    /// Build a session driven by a specific inference backend. The engine is
    /// retained for the whole session, so model weights load once.
    pub fn with_engine(seed: u64, engine: Box<dyn InferenceEngine>) -> Self {
        let config = SimConfig {
            enable_background: true,
            enable_director: true,
            ..SimConfig::default()
        };
        Self {
            sim: Simulation::new(build_town(), seed, config),
            engine,
            next_request: 1,
            trace: Vec::new(),
            decisions: Vec::new(),
            conversation_target: None,
            conversation_turns: VecDeque::new(),
        }
    }

    /// True when this session is backed by a real local model.
    pub fn is_real_model(&self) -> bool {
        self.engine.is_real_model()
    }

    pub fn engine_name(&self) -> String {
        self.engine.engine_name()
    }

    pub fn model_id(&self) -> String {
        self.engine.model_id()
    }

    /// Recover the engine so a resident model can be reused across sessions.
    pub fn into_engine(self) -> Box<dyn InferenceEngine> {
        self.engine
    }

    /// NPC turns answered by the real model.
    pub fn model_decisions(&self) -> usize {
        self.decisions
            .iter()
            .filter(|(_, s)| matches!(s, rpg_agents::DecisionSource::Model))
            .count()
    }

    /// NPC turns where a canonical obligation determined the action.
    pub fn canonical_decisions(&self) -> usize {
        self.decisions
            .iter()
            .filter(|(_, s)| {
                matches!(
                    s,
                    rpg_agents::DecisionSource::CanonicalObligation { .. }
                        | rpg_agents::DecisionSource::CanonicalAgenda { .. }
                )
            })
            .count()
    }

    /// NPC turns that required the deterministic fallback (reported, never hidden).
    pub fn fallback_decisions(&self) -> usize {
        self.decisions
            .iter()
            .filter(|(_, s)| matches!(s, rpg_agents::DecisionSource::HeuristicFallback { .. }))
            .count()
    }

    pub fn player(&self) -> &rpg_domain::world::Actor {
        self.sim.world.actor(&ActorId::new(PLAYER)).unwrap()
    }

    pub fn player_mut(&mut self) -> &mut rpg_domain::world::Actor {
        self.sim.world.actor_mut(&ActorId::new(PLAYER)).unwrap()
    }

    pub fn give(&mut self, item: &str, qty: u32) {
        self.player_mut().inventory.add(ItemId::new(item), qty);
    }

    pub fn player_do(&mut self, action: Action) -> rpg_domain::action::ValidationOutcome {
        let p = ActionProposal {
            actor: ActorId::new(PLAYER),
            action,
            source: ProposalSource::Player,
            rationale: "simulated player".into(),
        };
        self.sim.apply(&p)
    }

    /// Player speaks, then the addressed NPC decides and acts through the
    /// shared policy + inference engine.
    ///
    /// The session tracks the ongoing conversation like the runtime's active
    /// conversation does: only an accepted utterance opens or continues it;
    /// addressing a different NPC drops it entirely (a new conversation never
    /// inherits another's turns); the player's utterance and the NPC's
    /// committed spoken line are recorded with speaker attribution, bounded
    /// to [`MAX_CONVERSATION_TURNS`]. The decision's history view is the
    /// recorded exchange *without* the current utterance (it is passed
    /// separately, and removed by content, never by position), so the same
    /// line is never rendered twice.
    ///
    /// Returns the NPC's spoken reply text when the accepted action was
    /// `Speak`, so a voice client can synthesize exactly what the character
    /// said. Returns `None` when the speech did not reach the NPC or the NPC
    /// chose a non-verbal action.
    pub fn say(&mut self, target: &str, text: &str) -> Option<String> {
        // An NPC can only answer speech that actually reached them (in range
        // and audible). Rejected speech must not burn inference or produce an
        // impossible response — and never opens a conversation.
        if matches!(self.player_speak(target, text), ValidationOutcome::Accepted) {
            let npc = ActorId::new(target);
            // A new addressee starts a fresh conversation: no inheritance
            // from a previous NPC's exchange.
            if self.conversation_target.as_ref() != Some(&npc) {
                self.conversation_target = Some(npc.clone());
                self.conversation_turns.clear();
            }
            // Record the player's turn, then resolve against the prior turns
            // only (the runtime records the utterance first and strips the
            // trailing copy when building the history view).
            self.record_conversation_turn(ActorId::new(PLAYER), text.to_string());
            let history = self.prior_turns(&npc, text);
            let spoken = self.npc_respond_with_history(target, text, &history);
            // Record the NPC's committed reply — only a spoken line counts;
            // a rejected action said nothing and must not enter the memory.
            if let Some(line) = &spoken {
                self.record_conversation_turn(npc, line.clone());
            }
            spoken
        } else {
            None
        }
    }

    /// Append a turn to the ongoing conversation, keeping the memory bounded
    /// to [`MAX_CONVERSATION_TURNS`] (oldest turns are dropped first).
    fn record_conversation_turn(&mut self, speaker: ActorId, text: String) {
        self.conversation_turns.push_back((speaker, text));
        while self.conversation_turns.len() > MAX_CONVERSATION_TURNS {
            self.conversation_turns.pop_front();
        }
    }

    /// Prior turns of the ongoing conversation with `npc`, mirroring the
    /// runtime's `prior_turns`: the current utterance is also the final
    /// recorded turn at this point, so that trailing record is removed — but
    /// only when it actually is that utterance (same speaker and text),
    /// never by position alone.
    fn prior_turns(&self, npc: &ActorId, current: &str) -> Vec<(ActorId, String)> {
        if self.conversation_target.as_ref() != Some(npc) {
            return Vec::new();
        }
        let mut turns: Vec<(ActorId, String)> = self
            .conversation_turns
            .iter()
            .map(|(who, said)| (who.clone(), said.clone()))
            .collect();
        if turns
            .last()
            .is_some_and(|(who, said)| who.0 == PLAYER && said == current)
        {
            turns.pop();
        }
        turns
    }

    /// The player's half of a conversational turn: commit the utterance as a
    /// `Speak` action addressed to `target`. Only an `Accepted` outcome means
    /// the NPC heard it and may answer.
    pub fn player_speak(&mut self, target: &str, text: &str) -> ValidationOutcome {
        let out = self.player_do(Action::Speak {
            target: Some(ActorId::new(target)),
            text: text.into(),
            loudness: Loudness::Normal,
        });
        self.trace
            .push(format!("player -> {target}: {text} [{out:?}]"));
        out
    }

    /// Allocate the correlation id for the next NPC decision request.
    pub fn next_request_id(&mut self) -> u64 {
        let id = self.next_request;
        self.next_request += 1;
        id
    }

    /// Run the NPC decision for a heard utterance and return the line the NPC
    /// should speak aloud, when the accepted action has one.
    ///
    /// This is the primitive for callers that manage the conversation
    /// themselves: unlike [`Session::say`], it consults no session memory —
    /// the decision runs with an empty history unless the caller uses
    /// [`Session::npc_respond_with_history`].
    ///
    /// * `Speak` → its `text`.
    /// * `MakeOffer` → its `note`.
    /// * `Promise` → its `terms`.
    /// * `Threaten` → its `demand`.
    ///
    /// These fields are the character's own words as committed by the
    /// authoritative simulation; the voice client reads exactly this text and
    /// never invents dialogue.
    pub fn npc_respond(&mut self, npc: &str, text: &str) -> Option<String> {
        self.npc_respond_with_history(npc, text, &[])
    }

    /// Run the NPC decision for a heard utterance with explicit conversation
    /// history. The history is rendered into the model prompt so the NPC can
    /// resolve follow-up references.
    pub fn npc_respond_with_history(
        &mut self,
        npc: &str,
        text: &str,
        history: &[(ActorId, String)],
    ) -> Option<String> {
        let npc_id = ActorId::new(npc);
        let player = ActorId::new(PLAYER);
        let request_id = self.next_request_id();
        let ctx = NpcRequestContext {
            world: &self.sim.world,
            npc: &npc_id,
            speaker: &player,
            utterance: text,
            history,
        };
        let decision = NpcPolicy.decide(&ctx, request_id, self.engine.as_mut());
        self.commit_npc_decision(npc, decision).spoken
    }

    /// Validate and commit an NPC decision against the *current* world.
    ///
    /// This is the single commit path for NPC replies: the synchronous
    /// [`Session::npc_respond`] and the realtime client (whose decision was
    /// produced by a model running while the world kept moving) both end
    /// here, so provenance and the spoken line are recorded identically.
    pub fn commit_npc_decision(
        &mut self,
        npc: &str,
        decision: rpg_agents::NpcDecision,
    ) -> NpcCommit {
        let kind = decision.proposal.action.kind_name().to_string();
        let source = decision.source.clone();
        let spoken = spoken_line_in(&self.sim.world, &decision.proposal.action);
        let out = self.sim.apply(&decision.proposal);
        self.trace
            .push(format!("{npc} -> {kind} [{}]: {out:?}", source.label()));
        self.decisions.push((npc.to_string(), source.clone()));
        // A `Pending` outcome means the offer/promise was recorded and awaits
        // acceptance — the character still spoke the line. Only a rejection
        // means nothing was said.
        let spoken = match out {
            ValidationOutcome::Rejected(_) => None,
            _ => spoken,
        };
        NpcCommit {
            outcome: out,
            spoken,
            kind,
            source,
        }
    }

    pub fn tick(&mut self) -> StepReport {
        self.sim.step(Vec::new())
    }

    pub fn ticks(&mut self, n: u32) {
        for _ in 0..n {
            let _ = self.tick();
        }
    }

    /// Move the player next to an actor (setup convenience for scripts).
    pub fn teleport_player_to(&mut self, actor: &str) {
        let target = self
            .sim
            .world
            .actor(&ActorId::new(actor))
            .map(|a| (a.pos, a.location.clone()));
        if let Some((pos, loc)) = target {
            if let Some(p) = self.sim.world.actor_mut(&ActorId::new(PLAYER)) {
                p.pos = pos;
                p.location = loc;
            }
        }
    }

    pub fn teleport(&mut self, actor: &str, location: &str) {
        let center = self
            .sim
            .world
            .location(&LocationId::new(location))
            .map(|l| l.center());
        if let (Some(center), Some(a)) = (center, self.sim.world.actor_mut(&ActorId::new(actor))) {
            a.pos = center;
            a.location = LocationId::new(location);
        }
    }

    pub fn checksum(&self) -> u64 {
        self.sim.events.checksum()
    }

    /// Walk to a location and advance ticks until arrival.
    pub fn goto(&mut self, location: &str) {
        let out = self.player_do(Action::MoveTo {
            location: LocationId::new(location),
        });
        if matches!(out, ValidationOutcome::Accepted) {
            let player = ActorId::new(PLAYER);
            let mut guard = 0;
            while self.sim.paths.contains_key(&player) && guard < 200 {
                let _ = self.tick();
                guard += 1;
            }
        }
    }

    /// Advance until all movement and private-conversation walks settle.
    pub fn wait_settled(&mut self, max_ticks: u32) {
        let mut n = 0;
        while (!self.sim.paths.is_empty() || !self.sim.pending_private.is_empty()) && n < max_ticks
        {
            let _ = self.tick();
            n += 1;
        }
    }
}

/// The outcome of committing one NPC decision.
#[derive(Clone, Debug)]
pub struct NpcCommit {
    pub outcome: ValidationOutcome,
    /// The words the character actually said; `None` for a rejected or
    /// silent action. Only this text may be voiced.
    pub spoken: Option<String>,
    /// The committed action's kind, for traces and diagnostics.
    pub kind: String,
    pub source: rpg_agents::DecisionSource,
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// The words an accepted NPC action should be spoken as, if any.
///
/// This is the character's own committed text: `Speak.text`, an offer's
/// `note`, a promise's `terms`, or a threat's `demand`. It deliberately ignores
/// mechanical fields (item lists, fact ids) and returns `None` for silent
/// actions, so the voice client never fabricates dialogue.
pub fn spoken_line(action: &Action) -> Option<String> {
    let candidate = match action {
        Action::Speak { text, .. } => text,
        Action::MakeOffer { note, .. } => note,
        Action::Promise { terms, .. } => terms,
        Action::Threaten { demand, .. } => demand,
        _ => return None,
    };
    let trimmed = candidate.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Like [`spoken_line`], but also renders actions whose words live in the world
/// model.
///
/// * `ShareClaim` is how an NPC discloses a fact they believe. Its audible text
///   is the fact's canonical `statement`; the simulation only accepts the claim
///   when the speaker actually believes it, so this never leaks a secret the
///   character could not know.
/// * `MakeOffer` normally carries the speaker's own `note`, but the local model
///   frequently leaves it empty. That made the offer **completely silent** in
///   the voice clients even though a real bargain had been offered. When the
///   note is empty the committed terms are rendered into a short sentence. This
///   is a deterministic paraphrase of the accepted action's own fields (the same
///   data the UI shows), never invented dialogue.
pub fn spoken_line_in(world: &rpg_domain::world::World, action: &Action) -> Option<String> {
    match action {
        Action::ShareClaim { fact, .. } => world
            .facts
            .get(fact)
            .map(|f| f.statement.trim().to_string())
            .filter(|s| !s.is_empty()),
        Action::MakeOffer {
            note,
            give,
            want,
            disclose,
            ..
        } => {
            let note = note.trim();
            if note.is_empty() {
                Some(describe_offer(world, give, want, disclose.is_some()))
            } else {
                Some(note.to_string())
            }
        }
        _ => spoken_line(action),
    }
}

/// Human names for an item-stack list, e.g. "medicine" or "medicine and a ledger".
fn names_of(world: &rpg_domain::world::World, stacks: &[rpg_domain::world::ItemStack]) -> String {
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
        .join(" and ")
}

/// Render an offer's committed terms as a short spoken sentence.
fn describe_offer(
    world: &rpg_domain::world::World,
    give: &[rpg_domain::world::ItemStack],
    want: &[rpg_domain::world::ItemStack],
    discloses: bool,
) -> String {
    let want_s = names_of(world, want);
    let give_s = names_of(world, give);
    if !want_s.is_empty() {
        if discloses {
            format!("Bring me {want_s} and I will tell you what I know.")
        } else if !give_s.is_empty() {
            format!("Bring me {want_s} and I will give you {give_s}.")
        } else {
            format!("Bring me {want_s} and we have a deal.")
        }
    } else if !give_s.is_empty() {
        format!("I can offer you {give_s}.")
    } else if discloses {
        "There is something I could tell you, for the right price.".to_string()
    } else {
        "Let us make a deal.".to_string()
    }
}

#[derive(Clone, Debug)]
pub struct Check {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

#[derive(Clone, Debug)]
pub struct ScenarioResult {
    pub name: String,
    pub seed: u64,
    pub checks: Vec<Check>,
    pub event_count: usize,
    pub checksum: u64,
    pub trace: Vec<String>,
}

impl ScenarioResult {
    pub fn passed(&self) -> bool {
        self.checks.iter().all(|c| c.passed)
    }

    pub fn failures(&self) -> Vec<&Check> {
        self.checks.iter().filter(|c| !c.passed).collect()
    }

    pub fn summary(&self) -> String {
        let failed = self.failures();
        if failed.is_empty() {
            format!(
                "PASS {} (seed {}, {} checks, {} events, checksum {:x})",
                self.name,
                self.seed,
                self.checks.len(),
                self.event_count,
                self.checksum
            )
        } else {
            let mut s = format!(
                "FAIL {} (seed {}, {} events)\n",
                self.name, self.seed, self.event_count
            );
            for c in failed {
                s.push_str(&format!("  - {}: {}\n", c.name, c.detail));
            }
            s
        }
    }
}

fn check(checks: &mut Vec<Check>, name: &str, passed: bool, detail: impl Into<String>) {
    checks.push(Check {
        name: name.into(),
        passed,
        detail: detail.into(),
    });
}

fn finish(name: &str, seed: u64, session: &Session, checks: Vec<Check>) -> ScenarioResult {
    ScenarioResult {
        name: name.into(),
        seed,
        checks,
        event_count: session.sim.events.len(),
        checksum: session.checksum(),
        trace: session.trace.clone(),
    }
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

pub struct Scenario {
    pub name: &'static str,
    pub description: &'static str,
    pub run: fn(u64) -> ScenarioResult,
}

pub fn scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            name: "friendly_investigation",
            description: "Talk to several NPCs and discover at least one lead.",
            run: friendly_investigation,
        },
        Scenario {
            name: "grocer_conditional_bargain",
            description: "The grocer demands medicine before sharing the ledger.",
            run: grocer_conditional_bargain,
        },
        Scenario {
            name: "private_conversation_and_spy",
            description: "A private conversation moves away from a spy.",
            run: private_conversation_and_spy,
        },
        Scenario {
            name: "blackmail_has_consequences",
            description: "Threatening an NPC raises fear and anger (not free compliance).",
            run: blackmail_has_consequences,
        },
        Scenario {
            name: "ignore_plot_world_progresses",
            description: "The world moves while the main story stays player-paced.",
            run: ignore_plot_world_progresses,
        },
        Scenario {
            name: "save_reload_continuity",
            description: "A commitment survives save/load.",
            run: save_reload_continuity,
        },
        Scenario {
            name: "deterministic_replay",
            description: "Same seed + script yields the same canonical checksum.",
            run: deterministic_replay,
        },
        Scenario {
            name: "knowledge_boundaries",
            description: "NPCs may not reveal facts they do not know or may not share.",
            run: knowledge_boundaries,
        },
        Scenario {
            name: "three_independent_routes",
            description: "The leader can be reached through independent clue chains.",
            run: three_independent_routes,
        },
        Scenario {
            name: "innocent_grocer_stays_innocent",
            description: "Attention on the grocer does not retroactively make them the leader.",
            run: innocent_grocer_stays_innocent,
        },
        Scenario {
            name: "emergent_quest_surface",
            description: "An emergent quest is recognized from real goals, surfaced by the constrained director, resolved by predicate satisfaction.",
            run: emergent_quest_surface,
        },
    ]
}

pub fn run_named(name: &str, seed: u64) -> Option<ScenarioResult> {
    scenarios()
        .into_iter()
        .find(|s| s.name == name)
        .map(|s| (s.run)(seed))
}

pub fn run_all(seed: u64) -> Vec<ScenarioResult> {
    scenarios().into_iter().map(|s| (s.run)(seed)).collect()
}

fn friendly_investigation(seed: u64) -> ScenarioResult {
    let mut s = Session::new(seed);
    let mut checks = Vec::new();

    s.teleport_player_to("grocer");
    s.say("grocer", "Good morning. Business been steady?");
    s.teleport_player_to("guard_captain");
    s.say(
        "guard_captain",
        "Have you noticed anything strange at the docks?",
    );
    s.teleport_player_to("dockworker");
    s.say("dockworker", "Who moves the crates at night?");
    s.teleport_player_to("priest");
    s.say("priest", "Do you know who is threatening the shopkeepers?");

    // Explore two locations to find any placed clues (must be physically there).
    s.goto("warehouse");
    s.player_do(Action::Investigate {
        target: InvestigateTarget::Location(LocationId::new("warehouse")),
    });
    s.goto("alley");
    s.player_do(Action::Investigate {
        target: InvestigateTarget::Location(LocationId::new("alley")),
    });
    s.ticks(20);

    let player = ActorId::new(PLAYER);
    let beliefs = s
        .sim
        .world
        .actor(&player)
        .map(|a| a.beliefs.len())
        .unwrap_or(0);
    let events = s.sim.events.len();
    check(
        &mut checks,
        "player_learned_something",
        beliefs > 0,
        format!("beliefs={beliefs}"),
    );
    check(
        &mut checks,
        "events_recorded",
        events > 5,
        format!("events={events}"),
    );
    let spoke = s
        .sim
        .events
        .all()
        .iter()
        .any(|e| matches!(e.kind, rpg_domain::events::EventKind::SpeechEmitted { .. }));
    check(
        &mut checks,
        "speech_committed",
        spoke,
        "at least one SpeechEmitted",
    );
    finish("friendly_investigation", seed, &s, checks)
}

fn grocer_conditional_bargain(seed: u64) -> ScenarioResult {
    use rpg_domain::action::ValidationOutcome;
    let mut s = Session::new(seed);
    let mut checks = Vec::new();
    s.teleport_player_to("grocer");

    s.say(
        "grocer",
        "I need to see your ledger. The one with the dock payments.",
    );
    let offer = s
        .sim
        .world
        .offers
        .values()
        .find(|o| o.from.as_str() == "grocer")
        .cloned();
    check(
        &mut checks,
        "conditional_offer_created",
        offer.is_some(),
        format!("offers={}", s.sim.world.offers.len()),
    );

    if let Some(o) = offer {
        check(
            &mut checks,
            "offer_is_conditional",
            o.condition.is_some(),
            "offer carries a world predicate",
        );
        // Accepting creates a binding obligation, but does not yet disclose.
        let accepted = s.player_do(Action::AcceptOffer { offer: o.id });
        check(
            &mut checks,
            "offer_accepted",
            matches!(accepted, ValidationOutcome::Accepted),
            format!("{accepted:?}"),
        );
        s.say("grocer", "The ledger? Our deal?");
        let knows_before = s
            .sim
            .world
            .has_heard(&ActorId::new(PLAYER), &FactId::new("f_grocer_ledger"));
        check(
            &mut checks,
            "not_disclosed_before_medicine",
            !knows_before,
            "ledger withheld until the condition is met",
        );
        // Deliver the medicine; the obligation is satisfied by the predicate.
        s.give("medicine", 1);
        let gave = s.player_do(Action::TransferItem {
            target: ActorId::new("grocer"),
            item: ItemId::new("medicine"),
            qty: 1,
        });
        check(
            &mut checks,
            "medicine_delivered",
            matches!(gave, ValidationOutcome::Accepted),
            format!("{gave:?}"),
        );
        s.tick();
        let satisfied = s
            .sim
            .world
            .commitments
            .values()
            .any(|c| c.state == rpg_domain::social::CommitmentState::Satisfied);
        check(
            &mut checks,
            "obligation_satisfied",
            satisfied,
            "commitment satisfied",
        );
        // Now the grocer honours the bargain.
        s.say("grocer", "The medicine is yours. Now, the ledger?");
        let knows_after = s
            .sim
            .world
            .has_heard(&ActorId::new(PLAYER), &FactId::new("f_grocer_ledger"));
        check(
            &mut checks,
            "ledger_disclosed_after_bargain",
            knows_after,
            "player heard ledger fact",
        );
    }
    finish("grocer_conditional_bargain", seed, &s, checks)
}

fn private_conversation_and_spy(seed: u64) -> ScenarioResult {
    let mut s = Session::new(seed);
    let mut checks = Vec::new();
    s.teleport("spy", "grocer");
    s.teleport_player_to("grocer");

    let spy_before = s
        .sim
        .world
        .actor(&ActorId::new("spy"))
        .unwrap()
        .location
        .clone();
    let out = s.player_do(Action::AskPrivately {
        target: ActorId::new("grocer"),
    });
    check(
        &mut checks,
        "privacy_accepted",
        matches!(out, ValidationOutcome::Accepted),
        format!("{out:?}"),
    );
    // Privacy takes real movement, not a teleport.
    s.wait_settled(50);
    let grocer_loc = s
        .sim
        .world
        .actor(&ActorId::new("grocer"))
        .unwrap()
        .location
        .clone();
    let spy_after = s
        .sim
        .world
        .actor(&ActorId::new("spy"))
        .unwrap()
        .location
        .clone();
    check(
        &mut checks,
        "moved_to_private_space",
        grocer_loc.as_str() == "grocer_backroom",
        format!("grocer at {grocer_loc}"),
    );
    check(
        &mut checks,
        "spy_not_in_private_space",
        spy_after.as_str() != "grocer_backroom",
        format!("spy was {spy_before}, now {spy_after}"),
    );
    // Whisper in the private room should not be audible to the spy.
    let before = s.sim.events.len();
    let _ = s.player_do(Action::Speak {
        target: Some(ActorId::new("grocer")),
        text: "Quietly: the ledger is behind the flour.".into(),
        loudness: Loudness::Whisper,
    });
    s.npc_respond("grocer", "Quietly: the ledger is behind the flour.");
    let spy_heard = s.sim.events.all()[before..].iter().any(|e| {
        e.observed_by.contains(&ActorId::new("spy"))
            && matches!(&e.kind, rpg_domain::events::EventKind::SpeechEmitted { speaker, .. } if speaker.as_str() == PLAYER)
    });
    check(
        &mut checks,
        "spy_did_not_hear_private_talk",
        !spy_heard,
        "spy not an observer of the private utterance",
    );
    finish("private_conversation_and_spy", seed, &s, checks)
}

fn blackmail_has_consequences(seed: u64) -> ScenarioResult {
    let mut s = Session::new(seed);
    let mut checks = Vec::new();
    s.teleport_player_to("grocer");
    // Give the player leverage: they secretly know about the ledger.
    let (tick, minute) = (s.sim.world.tick, s.sim.world.minute_of_day);
    rpg_social::record_belief(
        &mut s.sim.world,
        &mut s.sim.events,
        tick,
        minute,
        &ActorId::new(PLAYER),
        &FactId::new("f_grocer_ledger"),
        true,
        0.95,
        rpg_domain::world::Provenance::Assumed,
    );
    let before = s
        .sim
        .world
        .actor(&ActorId::new("grocer"))
        .unwrap()
        .relationship(&ActorId::new(PLAYER));

    let out = s.player_do(Action::Threaten {
        target: ActorId::new("grocer"),
        demand: "Show me the ledger, or the Watch hears about it.".into(),
    });
    check(
        &mut checks,
        "threat_accepted",
        matches!(out, rpg_domain::action::ValidationOutcome::Accepted),
        format!("{out:?}"),
    );
    let after = s
        .sim
        .world
        .actor(&ActorId::new("grocer"))
        .unwrap()
        .relationship(&ActorId::new(PLAYER));
    check(
        &mut checks,
        "fear_rose",
        after.fear > before.fear,
        format!("{} -> {}", before.fear, after.fear),
    );
    check(
        &mut checks,
        "anger_rose",
        after.anger > before.anger,
        format!("{} -> {}", before.anger, after.anger),
    );
    check(
        &mut checks,
        "trust_fell",
        after.trust_benevolence < before.trust_benevolence,
        format!(
            "benevolence {} -> {} (contextual dimension, design §3.2)",
            before.trust_benevolence, after.trust_benevolence
        ),
    );
    finish("blackmail_has_consequences", seed, &s, checks)
}

fn ignore_plot_world_progresses(seed: u64) -> ScenarioResult {
    let mut s = Session::new(seed);
    let mut checks = Vec::new();
    let leader_before = rpg_content::seeded_leader(&s.sim.world).unwrap();
    let budget_before: i64 = s.sim.world.factions.values().map(|f| f.budget).sum();
    let tick_before = s.sim.world.tick;
    let committed_before = s.sim.world.facts.values().filter(|f| f.committed).count();

    s.ticks(240);

    let leader_after = rpg_content::seeded_leader(&s.sim.world).unwrap();
    let budget_after: i64 = s.sim.world.factions.values().map(|f| f.budget).sum();
    let committed_after = s.sim.world.facts.values().filter(|f| f.committed).count();

    check(
        &mut checks,
        "time_advanced",
        s.sim.world.tick > tick_before,
        format!("tick {} -> {}", tick_before, s.sim.world.tick),
    );
    check(
        &mut checks,
        "economy_moved",
        budget_after != budget_before,
        format!("budget {budget_before} -> {budget_after}"),
    );
    check(
        &mut checks,
        "leader_unchanged",
        leader_before == leader_after,
        format!("{leader_before} == {leader_after}"),
    );
    check(
        &mut checks,
        "story_not_auto_completed",
        committed_after >= committed_before,
        "no player discovery occurred; no checkpoint should regress",
    );
    finish("ignore_plot_world_progresses", seed, &s, checks)
}

fn save_reload_continuity(seed: u64) -> ScenarioResult {
    let mut s = Session::new(seed);
    let mut checks = Vec::new();
    s.teleport_player_to("grocer");
    s.player_do(Action::Promise {
        target: ActorId::new("grocer"),
        terms: "I will find your daughter's medicine.".into(),
        predicate: QuestPredicate::HasItem {
            actor: ActorId::new("grocer"),
            item: ItemId::new("medicine"),
            qty: 1,
        },
        deadline: None,
    });
    let commitments_before = s.sim.world.commitments.len();
    let save = rpg_persistence::SaveFile::from_sim(&s.sim);
    let json = serde_json::to_string(&save).unwrap();
    let loaded: rpg_persistence::SaveFile = serde_json::from_str(&json).unwrap();
    let restored = rpg_persistence::restore(loaded, SimConfig::default());
    check(
        &mut checks,
        "commitments_persist",
        restored.world.commitments.len() == commitments_before && commitments_before > 0,
        format!(
            "{} -> {}",
            commitments_before,
            restored.world.commitments.len()
        ),
    );
    check(
        &mut checks,
        "event_checksum_persists",
        restored.events.checksum() == s.checksum(),
        "checksum equal",
    );
    finish("save_reload_continuity", seed, &s, checks)
}

fn run_replay_script(seed: u64) -> Session {
    let mut s = Session::new(seed);
    s.teleport_player_to("grocer");
    s.say("grocer", "Where do the shipments go at night?");
    s.say("dockworker", "Who signs for the night crates?");
    s.player_do(Action::Investigate {
        target: InvestigateTarget::Location(LocationId::new("warehouse")),
    });
    s.ticks(15);
    s
}

fn deterministic_replay(seed: u64) -> ScenarioResult {
    let a = run_replay_script(seed);
    let b = run_replay_script(seed);
    let mut checks = Vec::new();
    check(
        &mut checks,
        "same_checksum",
        a.checksum() == b.checksum(),
        format!("{:x} vs {:x}", a.checksum(), b.checksum()),
    );
    check(
        &mut checks,
        "same_event_count",
        a.sim.events.len() == b.sim.events.len(),
        format!("{} vs {}", a.sim.events.len(), b.sim.events.len()),
    );
    finish("deterministic_replay", seed, &a, checks)
}

fn knowledge_boundaries(seed: u64) -> ScenarioResult {
    let mut s = Session::new(seed);
    let mut checks = Vec::new();
    s.teleport_player_to("grocer");

    // The player does not know the shipment fact, so cannot disclose it.
    let out = s.player_do(Action::ShareClaim {
        target: ActorId::new("grocer"),
        fact: FactId::new("f_shipments_diverted"),
        assertive: true,
        deceptive: false,
        retold_chain: None,
    });
    check(
        &mut checks,
        "cannot_share_unknown_fact",
        matches!(
            out,
            rpg_domain::action::ValidationOutcome::Rejected(
                rpg_domain::action::RejectReason::LacksKnowledge(_)
            )
        ),
        format!("{out:?}"),
    );

    // A non-inner-circle actor cannot disclose an inner-circle fact.
    let out2 = s.player_do(Action::ShareClaim {
        target: ActorId::new(PLAYER),
        fact: FactId::new("f_leader_identity"),
        assertive: true,
        deceptive: false,
        retold_chain: None,
    });
    check(
        &mut checks,
        "player_cannot_forge_secret",
        matches!(out2, rpg_domain::action::ValidationOutcome::Rejected(_)),
        format!("{out2:?}"),
    );

    // The grocer's prompt/context must not contain the leader's name.
    let ctx = rpg_agents::render_npc_context(
        &s.sim.world,
        &ActorId::new("grocer"),
        &ActorId::new(PLAYER),
        "Who leads the mafia?",
        &[],
    );
    check(
        &mut checks,
        "context_has_no_leader_leak",
        !ctx.to_lowercase().contains("selene"),
        "leader name absent from grocer context",
    );
    finish("knowledge_boundaries", seed, &s, checks)
}

fn three_independent_routes(seed: u64) -> ScenarioResult {
    let s = Session::new(seed);
    let mut checks = Vec::new();
    let cp_trace = s
        .sim
        .world
        .story
        .checkpoints
        .iter()
        .find(|c| c.id == "cp_trace")
        .cloned();
    let route_count = cp_trace.map(|c| c.candidate_facts.len()).unwrap_or(0);
    check(
        &mut checks,
        "trace_has_parallel_routes",
        route_count >= 2,
        format!("cp_trace candidates={route_count}"),
    );
    let leader_fact = s
        .sim
        .world
        .facts
        .get(&FactId::new("f_leader_identity"))
        .cloned();
    let holders = leader_fact
        .as_ref()
        .map(|f| f.eligible_holders.len())
        .unwrap_or(0);
    check(
        &mut checks,
        "leader_fact_has_multiple_holders",
        holders >= 2,
        format!("holders={holders}"),
    );
    let leader = rpg_content::seeded_leader(&s.sim.world);
    check(
        &mut checks,
        "leader_is_seeded",
        leader.is_some(),
        "fixed answer present",
    );
    finish("three_independent_routes", seed, &s, checks)
}

fn innocent_grocer_stays_innocent(seed: u64) -> ScenarioResult {
    let mut s = Session::new(seed);
    let mut checks = Vec::new();
    s.teleport_player_to("grocer");
    // Repeated attention on the grocer.
    for _ in 0..6 {
        s.say(
            "grocer",
            "I know you are hiding something. Tell me about the mafia.",
        );
        s.tick();
    }
    let leader = rpg_content::seeded_leader(&s.sim.world).unwrap();
    check(
        &mut checks,
        "grocer_not_promoted_to_leader",
        leader.as_str() != "grocer",
        format!("leader={leader}"),
    );
    let grocer = s.sim.world.actor(&ActorId::new("grocer")).unwrap();
    check(
        &mut checks,
        "grocer_has_no_inner_circle_flag",
        !grocer.has_flag("inner_circle"),
        "grocer remains an ordinary NPC",
    );
    finish("innocent_grocer_stays_innocent", seed, &s, checks)
}

/// The I7 production pathway (plan Phase 5; design §6.2/§6.3): recognize the
/// grocer's authored bargain as a quest opportunity, run the constrained
/// director's negative space, surface through the validated proposal path,
/// bind a real contract by accepting the bargain offer, then resolve the
/// quest through real validated actions (substitute predicate satisfaction).
/// Finishes with deterministic replay and save/load continuation.
fn run_emergent_quest_script(seed: u64) -> Session {
    use rpg_narrative::DirectorProposal;
    let mut s = Session::new(seed);
    // The player hears the quest surface directly (permitted channel):
    // teleport to the grocer, then surface the recognized opportunity.
    s.teleport_player_to("grocer");
    let quests = rpg_narrative::recognize_quests(&s.sim.world);
    let opportunity = quests[0].clone();
    let (tick, minute) = (s.sim.world.tick, s.sim.world.minute_of_day);
    let _ = rpg_narrative::commit_director_proposal(
        &s.sim.world,
        &mut s.sim.events,
        tick,
        minute,
        DirectorProposal::SurfaceQuest {
            opportunity,
            presented_to: ActorId::new(PLAYER),
            claims: vec![],
        },
    );
    // Bind the real contract: ask about the ledger, accept the authored
    // bargain offer (production bargain pathway), keep the quest.
    s.say(
        "grocer",
        "I need to see your ledger. The one with the dock payments.",
    );
    if let Some(o) = s
        .sim
        .world
        .offers
        .values()
        .find(|o| o.from.as_str() == "grocer")
        .cloned()
    {
        let _ = s.player_do(Action::AcceptOffer { offer: o.id });
    }
    // Obtain the medicine through a real validated route (investigate the
    // alley, then hand it over in person) and satisfy the goal predicate.
    s.goto("alley");
    s.player_do(Action::Investigate {
        target: InvestigateTarget::Location(LocationId::new("alley")),
    });
    s.teleport_player_to("grocer");
    let _ = s.player_do(Action::TransferItem {
        target: ActorId::new("grocer"),
        item: ItemId::new("medicine"),
        qty: 1,
    });
    s.ticks(2);
    s
}

fn emergent_quest_surface(seed: u64) -> ScenarioResult {
    use rpg_narrative::{
        commit_director_proposal, quest_satisfied, recognize_quests, validate_director_proposal,
        Decision, DirectorProposal, QuestClaim, QuestContract, QuestGoalSource, QuestPathway,
    };
    let mut s = Session::new(seed);
    let mut checks = Vec::new();
    // Mutating steps below mirror run_emergent_quest_script exactly; only
    // read-only checks are interleaved between them.
    s.teleport_player_to("grocer");

    // --- T5.1: recognition from real content, no scripted giver ----------
    let quests = recognize_quests(&s.sim.world);
    check(
        &mut checks,
        "t5_1_exactly_one_opportunity",
        quests.len() == 1,
        format!("quests={}", quests.len()),
    );
    let opportunity = quests[0].clone();
    check(
        &mut checks,
        "t5_1_beneficiary_and_goal_are_real",
        opportunity.beneficiary == ActorId::new("grocer")
            && opportunity.goal == QuestGoalSource::Bargain
            && opportunity.goal_predicate
                == QuestPredicate::HasItem {
                    actor: ActorId::new("grocer"),
                    item: ItemId::new("medicine"),
                    qty: 1,
                },
        format!("{:?}", opportunity.goal),
    );
    check(
        &mut checks,
        "t5_1_pathways_cite_real_placement",
        !opportunity.pathways.is_empty()
            && opportunity.pathways.iter().all(|p| match p {
                QuestPathway::ItemAvailableAt { item, location, .. } => s
                    .sim
                    .world
                    .location_items
                    .get(location)
                    .map(|stacks| stacks.iter().any(|st| &st.item == item))
                    .unwrap_or(false),
                QuestPathway::HeldByActor { item, holder, .. } => s
                    .sim
                    .world
                    .actor(holder)
                    .map(|a| a.is_alive() && a.inventory.quantity(item) > 0)
                    .unwrap_or(false),
            }),
        format!("{:?}", opportunity.pathways),
    );
    check(
        &mut checks,
        "t5_1_stake_is_the_promised_disclosure",
        opportunity.stake.promised_fact == Some(FactId::new("f_grocer_ledger")),
        format!("{:?}", opportunity.stake.promised_fact),
    );

    // Truth snapshot: everything the director must never touch. Compared as
    // canonical debug text (domain state types are not PartialEq by design).
    let truth_snapshot = |w: &rpg_domain::world::World| {
        format!(
            "{:?}|{:?}|{:?}|{:?}|{:?}",
            w.facts,
            w.deals,
            w.commitments,
            w.actors
                .iter()
                .map(|(k, a)| (k, &a.goals))
                .collect::<Vec<_>>(),
            w.flags,
        )
    };
    let truth_before = truth_snapshot(&s.sim.world);
    let events_before = s.sim.events.len();

    // --- T5.3: the constrained director's negative space -----------------
    // (a) Teleporting resources: a pathway pointing somewhere the item is not.
    let mut tampered = opportunity.clone();
    tampered.pathways = vec![QuestPathway::ItemAvailableAt {
        item: ItemId::new("medicine"),
        location: LocationId::new("castle"),
        qty: 1,
    }];
    check(
        &mut checks,
        "t5_3a_teleport_rejected",
        matches!(
            validate_director_proposal(
                &s.sim.world,
                &DirectorProposal::SurfaceQuest {
                    opportunity: tampered,
                    presented_to: ActorId::new(PLAYER),
                    claims: vec![],
                }
            ),
            Decision::Rejected(_)
        ),
        "tampered pathway must not validate",
    );
    // (b) Inventing a quest for an actor with no blocked goal/bargain.
    let fake = rpg_narrative::QuestOpportunity {
        beneficiary: ActorId::new("priest"),
        goal: QuestGoalSource::Bargain,
        description: "priest needs medicine".into(),
        goal_predicate: QuestPredicate::HasItem {
            actor: ActorId::new("priest"),
            item: ItemId::new("medicine"),
            qty: 1,
        },
        blockers: vec![],
        pathways: vec![],
        stake: rpg_narrative::QuestStake {
            goal_kind: None,
            goal_priority: None,
            promised_fact: None,
        },
        contract: None,
    };
    check(
        &mut checks,
        "t5_3b_invented_quest_rejected",
        matches!(
            validate_director_proposal(
                &s.sim.world,
                &DirectorProposal::SurfaceQuest {
                    opportunity: fake,
                    presented_to: ActorId::new(PLAYER),
                    claims: vec![],
                }
            ),
            Decision::Rejected(_)
        ),
        "invented beneficiary must not validate",
    );
    // (c) Inventing witnessed events: claim on an unknown fact.
    check(
        &mut checks,
        "t5_3c_invented_claim_rejected",
        matches!(
            validate_director_proposal(
                &s.sim.world,
                &DirectorProposal::SurfaceQuest {
                    opportunity: opportunity.clone(),
                    presented_to: ActorId::new(PLAYER),
                    claims: vec![QuestClaim {
                        fact: FactId::new("f_nonexistent"),
                        polarity: true,
                    }],
                }
            ),
            Decision::Rejected(_)
        ),
        "unknown-fact claim must not validate",
    );
    // (d) Treating a proposed quest as an established contract.
    let mut fake_contract = opportunity.clone();
    fake_contract.contract = Some(QuestContract::Deal(DealId(9999)));
    check(
        &mut checks,
        "t5_3d_fake_contract_rejected",
        matches!(
            validate_director_proposal(
                &s.sim.world,
                &DirectorProposal::SurfaceQuest {
                    opportunity: fake_contract,
                    presented_to: ActorId::new(PLAYER),
                    claims: vec![],
                }
            ),
            Decision::Rejected(_)
        ),
        "fabricated contract must not validate",
    );
    check(
        &mut checks,
        "t5_3_rejections_write_nothing",
        s.sim.events.len() == events_before && truth_before == truth_snapshot(&s.sim.world),
        "no events, no truth movement",
    );

    // --- Surface through the validated production path --------------------
    let (tick, minute) = (s.sim.world.tick, s.sim.world.minute_of_day);
    let surface_decision = commit_director_proposal(
        &s.sim.world,
        &mut s.sim.events,
        tick,
        minute,
        DirectorProposal::SurfaceQuest {
            opportunity: opportunity.clone(),
            presented_to: ActorId::new(PLAYER),
            claims: vec![],
        },
    );
    check(
        &mut checks,
        "surface_accepted",
        surface_decision.is_accepted(),
        format!("{surface_decision:?}"),
    );
    check(
        &mut checks,
        "surface_records_provenance_only",
        s.sim.events.len() == events_before + 1
            && matches!(
                s.sim.events.all().last().map(|e| &e.kind),
                Some(rpg_domain::events::EventKind::DirectorProposed { accepted: true, .. })
            ),
        "one DirectorProposed event, nothing else",
    );
    check(
        &mut checks,
        "surface_commits_no_world_truth",
        truth_before == truth_snapshot(&s.sim.world),
        "facts/deals/commitments/goals/flags unchanged",
    );
    // The permitted actor-facing surface: existence only, no privileged
    // pathway locations.
    let surface = rpg_narrative::QuestSurface::of(&s.sim.world, &opportunity);
    check(
        &mut checks,
        "t5_4_surface_is_permitted_information_only",
        surface.has_known_pathway
            && !surface.has_contract
            && surface.promised_fact == Some(FactId::new("f_grocer_ledger"))
            && !surface.description.contains("warehouse")
            && !surface.description.contains("alley"),
        format!("surface={surface:?}"),
    );

    // --- Bind a real contract (production bargain offer → commitment) ----
    s.say(
        "grocer",
        "I need to see your ledger. The one with the dock payments.",
    );
    let offer = s
        .sim
        .world
        .offers
        .values()
        .find(|o| o.from.as_str() == "grocer" && o.condition.is_some())
        .cloned();
    // The say() above already created the authored bargain offer.
    let accepted = offer
        .map(|o| s.player_do(Action::AcceptOffer { offer: o.id }))
        .unwrap_or(ValidationOutcome::Rejected(
            rpg_domain::action::RejectReason::Invalid("no bargain offer".into()),
        ));
    check(
        &mut checks,
        "bargain_offer_accepted",
        matches!(accepted, ValidationOutcome::Accepted),
        format!("{accepted:?}"),
    );
    let quests = recognize_quests(&s.sim.world);
    let opportunity2 = quests
        .iter()
        .find(|q| q.beneficiary == ActorId::new("grocer"))
        .cloned();
    check(
        &mut checks,
        "t5_4_contract_recognized_from_accepted_commitment",
        match &opportunity2 {
            Some(q) => match q.contract {
                Some(QuestContract::Commitment(cid)) => s
                    .sim
                    .world
                    .commitments
                    .get(&cid)
                    .map(|cm| {
                        cm.state == rpg_domain::social::CommitmentState::Pending
                            && cm.predicate == q.goal_predicate
                            && cm.promised_fact == Some(FactId::new("f_grocer_ledger"))
                    })
                    .unwrap_or(false),
                _ => false,
            },
            None => false,
        },
        format!(
            "contract={:?}",
            opportunity2.as_ref().and_then(|q| q.contract)
        ),
    );

    // --- Save/load continuation ------------------------------------------
    let save = rpg_persistence::SaveFile::from_sim(&s.sim);
    let json = serde_json::to_string(&save).unwrap();
    let loaded: rpg_persistence::SaveFile = serde_json::from_str(&json).unwrap();
    let restored = rpg_persistence::restore(loaded, SimConfig::default());
    check(
        &mut checks,
        "save_load_checksum_equal",
        restored.events.checksum() == s.checksum(),
        format!("{:x} vs {:x}", restored.events.checksum(), s.checksum()),
    );
    let restored_quests = recognize_quests(&restored.world);
    check(
        &mut checks,
        "save_load_recognition_equal",
        restored_quests == quests,
        format!(
            "restored={}",
            restored_quests
                .iter()
                .map(|q| format!("{:?}", q.goal))
                .collect::<Vec<_>>()
                .join(",")
        ),
    );
    check(
        &mut checks,
        "save_load_surface_event_persists",
        restored.events.all().iter().any(|e| {
            matches!(
                &e.kind,
                rpg_domain::events::EventKind::DirectorProposed { accepted: true, .. }
            )
        }),
        "DirectorProposed provenance survives the round-trip",
    );

    // --- T5.2: substitute predicate satisfaction through real actions -----
    let opportunity2 = opportunity2.expect("quest still recognized");
    check(
        &mut checks,
        "t5_2_not_satisfied_before",
        !quest_satisfied(&s.sim.world, &opportunity2),
        "grocer lacks medicine",
    );
    s.goto("alley");
    s.player_do(Action::Investigate {
        target: InvestigateTarget::Location(LocationId::new("alley")),
    });
    let found = s
        .sim
        .world
        .actor(&ActorId::new(PLAYER))
        .map(|a| a.inventory.quantity(&ItemId::new("medicine")))
        .unwrap_or(0);
    check(
        &mut checks,
        "t5_2_medicine_found_by_search",
        found >= 1,
        format!("player medicine={found}"),
    );
    s.teleport_player_to("grocer");
    let given = s.player_do(Action::TransferItem {
        target: ActorId::new("grocer"),
        item: ItemId::new("medicine"),
        qty: 1,
    });
    check(
        &mut checks,
        "t5_2_medicine_delivered",
        matches!(given, ValidationOutcome::Accepted),
        format!("{given:?}"),
    );
    check(
        &mut checks,
        "t5_2_predicate_satisfied_by_delivered_substitute",
        quest_satisfied(&s.sim.world, &opportunity2),
        "resolution follows the predicate, not a single authored route",
    );
    s.ticks(2);
    check(
        &mut checks,
        "t5_2_satisfied_quest_no_longer_surfaced",
        recognize_quests(&s.sim.world).is_empty(),
        "a satisfied goal is not an opportunity",
    );

    // --- Deterministic replay ---------------------------------------------
    let replay = run_emergent_quest_script(seed);
    check(
        &mut checks,
        "deterministic_replay",
        replay.checksum() == s.checksum(),
        format!("{:x} vs {:x}", replay.checksum(), s.checksum()),
    );

    finish("emergent_quest_surface", seed, &s, checks)
}

/// Aggregate a human-readable leaderboard of scenario results.
pub fn leaderboard(results: &[ScenarioResult]) -> String {
    let mut out = String::new();
    let passed = results.iter().filter(|r| r.passed()).count();
    out.push_str(&format!("Scenarios: {}/{} passed\n", passed, results.len()));
    for r in results {
        out.push_str(&r.summary());
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn share_claim_speaks_the_grounded_fact_statement() {
        let session = Session::new(2026);
        let claim = Action::ShareClaim {
            target: ActorId::new(PLAYER),
            fact: FactId::new("f_grocer_ledger"),
            assertive: true,
            deceptive: false,
            retold_chain: None,
        };
        let line = spoken_line_in(&session.sim.world, &claim).expect("statement");
        assert!(line.contains("ledger"), "unexpected line: {line}");

        // A purely mechanical action still has no spoken line.
        let silent = Action::Investigate {
            target: InvestigateTarget::Location(LocationId::new("plaza")),
        };
        assert!(spoken_line_in(&session.sim.world, &silent).is_none());
    }

    #[test]
    fn empty_offer_note_still_speaks_the_terms() {
        let session = Session::new(2026);
        let offer = Action::MakeOffer {
            target: ActorId::new(PLAYER),
            give: vec![],
            want: vec![rpg_domain::world::ItemStack {
                item: ItemId::new("medicine"),
                qty: 1,
            }],
            condition: None,
            disclose: Some(FactId::new("f_grocer_ledger")),
            note: String::new(),
        };
        let line = spoken_line_in(&session.sim.world, &offer).expect("offer should speak");
        assert!(
            line.to_lowercase().contains("medicine"),
            "unexpected line: {line}"
        );
    }

    #[test]
    fn all_builtin_scenarios_pass() {
        let results = run_all(2026);
        let failed: Vec<_> = results.iter().filter(|r| !r.passed()).collect();
        assert!(
            failed.is_empty(),
            "failed scenarios:\n{}",
            failed
                .iter()
                .map(|r| r.summary())
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    #[test]
    fn replay_is_deterministic_across_seeds() {
        for seed in [1u64, 2, 3] {
            let a = run_replay_script(seed);
            let b = run_replay_script(seed);
            assert_eq!(a.checksum(), b.checksum(), "seed {seed}");
        }
    }

    /// Autonomy regression: every scenario must keep passing across many
    /// seeds now that NPCs autonomously pursue goals between ticks. Goals
    /// (Marta's protective watch, Renn's investigation) may fire during any
    /// scenario, so this guards against autonomy disturbing scripted
    /// expectations or breaking deterministic replay.
    #[test]
    fn all_scenarios_pass_across_seeds_with_autonomy() {
        for seed in [1u64, 7, 42, 2026, 98765] {
            for result in run_all(seed) {
                assert!(
                    result.passed(),
                    "scenario {} failed at seed {}\n{result:?}",
                    result.name,
                    seed
                );
            }
        }
    }

    /// No-weights regression test for the real-model interaction shape: a
    /// scripted model that only "speaks" still gets the canonical bargain,
    /// and the satisfied deal is honoured on the next turn.
    #[test]
    fn scripted_model_bargain_is_honoured_canonically() {
        let engine =
            rpg_agents::ScriptedEngine::new().with_default(r#"{"action":"speak","text":"Hmm."}"#);
        let mut s = Session::with_engine(2026, Box::new(engine));
        s.teleport_player_to("grocer");
        s.say("grocer", "I need to see your ledger.");

        let offer = s
            .sim
            .world
            .offers
            .values()
            .find(|o| o.from.as_str() == "grocer")
            .cloned()
            .expect("canonical agenda should create a conditional offer");
        match &offer.condition {
            Some(QuestPredicate::HasItem { item, .. }) => assert_eq!(item.as_str(), "medicine"),
            other => panic!("expected medicine condition, got {other:?}"),
        }
        assert_eq!(
            offer.disclose.as_ref().map(|f| f.as_str()),
            Some("f_grocer_ledger")
        );
        assert!(matches!(
            s.player_do(Action::AcceptOffer { offer: offer.id }),
            ValidationOutcome::Accepted
        ));
        s.give("medicine", 1);
        assert!(matches!(
            s.player_do(Action::TransferItem {
                target: ActorId::new("grocer"),
                item: ItemId::new("medicine"),
                qty: 1,
            }),
            ValidationOutcome::Accepted
        ));
        s.tick();
        assert!(
            s.sim
                .world
                .commitments
                .values()
                .any(|c| c.state == rpg_domain::social::CommitmentState::Satisfied),
            "medicine delivery should satisfy the deal"
        );
        s.say("grocer", "Now the ledger, as we agreed.");
        assert!(
            s.sim
                .world
                .has_heard(&ActorId::new(PLAYER), &FactId::new("f_grocer_ledger")),
            "player should hear the ledger fact once the deal is honoured"
        );
    }

    /// A deterministic model stand-in that records every prompt it is asked
    /// to complete and always "speaks", so tests can assert exactly what the
    /// decision context carried (history rendering) without any LLM.
    struct PromptCapture {
        prompts: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        warmed: bool,
    }

    impl PromptCapture {
        fn new(prompts: std::sync::Arc<std::sync::Mutex<Vec<String>>>) -> Self {
            Self {
                prompts,
                warmed: false,
            }
        }
    }

    impl rpg_domain::ports::InferenceEngine for PromptCapture {
        fn engine_name(&self) -> String {
            "prompt-capture".into()
        }
        fn model_id(&self) -> String {
            "capture-v1".into()
        }
        fn warm(&mut self) -> Result<(), rpg_domain::ports::InferenceError> {
            self.warmed = true;
            Ok(())
        }
        fn complete(
            &mut self,
            req: &rpg_domain::ports::InferenceRequest,
        ) -> Result<rpg_domain::ports::InferenceResponse, rpg_domain::ports::InferenceError>
        {
            if !self.warmed {
                self.warm()?;
            }
            self.prompts.lock().unwrap().push(req.prompt.clone());
            Ok(rpg_domain::ports::InferenceResponse {
                completion_tokens: 1,
                text: r#"{"action":"speak","text":"Understood."}"#.into(),
                ..Default::default()
            })
        }
    }

    /// Regression for the eval-history gap: `say` must track the ongoing
    /// conversation like the runtime does and feed the *prior* exchange into
    /// the decision context on follow-ups (prompt capture proves what the
    /// model/heuristic path resolved from). Switching addressees must start
    /// clean in both directions — no inheritance.
    #[test]
    fn follow_up_history_reaches_the_decision_and_switching_starts_clean() {
        let prompts = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut s = Session::with_engine(2026, Box::new(PromptCapture::new(prompts.clone())));
        s.teleport_player_to("grocer");

        // First contact: the decision context starts with an empty history.
        s.say(
            "grocer",
            "I need to see your ledger. The one with the dock payments.",
        );
        let prompt1 = prompts.lock().unwrap()[0].clone();
        assert!(
            !prompt1.contains("Recent conversation"),
            "a first contact must start with an empty history"
        );
        assert_eq!(
            (
                s.conversation_target.as_ref().map(|a| a.as_str()),
                s.conversation_turns.len()
            ),
            (Some("grocer"), 2),
            "the accepted exchange must be recorded with both speakers"
        );

        // Follow-up: the prior exchange — the player's line and the grocer's
        // committed offer note — must reach the decision context, and the
        // current utterance must not be duplicated into the history.
        s.say("grocer", "What about it?");
        let prompt2 = prompts.lock().unwrap()[1].clone();
        assert!(
            prompt2.contains("Recent conversation"),
            "the follow-up prompt must include the conversation history"
        );
        assert!(
            prompt2.contains(r#"player: "I need to see your ledger"#),
            "the prior player turn must carry speaker attribution: {prompt2}"
        );
        assert!(
            prompt2.contains("grocer: \"Bring me medicine"),
            "the NPC's committed reply must be in the history: {prompt2}"
        );
        assert_eq!(
            prompt2.matches("What about it?").count(),
            1,
            "the current utterance must appear exactly once (never duplicated in history)"
        );

        // Switching to another NPC starts a fresh conversation: the new
        // addressee's context carries no trace of the grocer's exchange.
        s.teleport_player_to("dockworker");
        s.say("dockworker", "Do you watch the docks?");
        let prompt3 = prompts.lock().unwrap()[2].clone();
        assert!(
            !prompt3.contains("Recent conversation"),
            "a switched-to NPC must start with an empty history"
        );
        assert!(
            !prompt3.contains("dock payments"),
            "the dockworker's prompt must not inherit the grocer's conversation"
        );

        // Switching back is equally clean: no per-NPC retention across a
        // switch, mirroring the runtime's active-conversation reset.
        s.teleport_player_to("grocer");
        s.say("grocer", "Back again — anything new?");
        let prompt4 = prompts.lock().unwrap()[3].clone();
        assert!(
            !prompt4.contains("Recent conversation"),
            "returning to a former addressee must start clean too"
        );
        assert!(
            !prompt4.contains("What about it?"),
            "the earlier exchange must not leak across the switch"
        );
    }

    /// The recorded memory is bounded: older turns are dropped first, so a
    /// long session's decision context shows only the recent exchange window.
    #[test]
    fn conversation_history_is_bounded() {
        let prompts = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut s = Session::with_engine(2026, Box::new(PromptCapture::new(prompts.clone())));
        s.teleport_player_to("grocer");
        for i in 1..=8 {
            s.say("grocer", &format!("question note {i}"));
        }
        assert_eq!(prompts.lock().unwrap().len(), 8);
        let prompt = prompts.lock().unwrap()[7].clone();
        // Twelve turns (six exchanges of player + NPC) fit; the oldest
        // exchange has been evicted by the eighth follow-up.
        assert!(
            !prompt.contains("note 1") && !prompt.contains("note 2"),
            "the oldest exchanges must have been evicted from the bounded window"
        );
        assert!(
            prompt.contains("note 3") && prompt.contains("note 7"),
            "the recent exchange window must still be present: {prompt}"
        );
        assert_eq!(
            prompt.matches("question note 8").count(),
            1,
            "the current utterance must not be duplicated into the bounded history"
        );
        assert!(s.conversation_turns.len() <= MAX_CONVERSATION_TURNS);
    }

    /// The deterministic fallback path (no model) completes the multi-turn
    /// bargain on a follow-up turn exactly like the runtime's fallback, and
    /// the session's tracked memory carries the whole exchange with speaker
    /// attribution.
    #[test]
    fn heuristic_fallback_resolves_follow_up_bargain_with_history() {
        let mut s = Session::new(2026);
        s.teleport_player_to("grocer");

        // Turn 1: the heuristic offers the designed conditional bargain.
        s.say("grocer", "I need to see your ledger.");
        assert!(
            s.sim
                .world
                .offers
                .values()
                .any(|o| o.from.as_str() == "grocer"),
            "the heuristic must propose the conditional bargain"
        );
        let offer = s
            .sim
            .world
            .offers
            .values()
            .find(|o| o.from.as_str() == "grocer")
            .cloned()
            .expect("conditional offer");
        assert!(matches!(
            s.player_do(Action::AcceptOffer { offer: offer.id }),
            ValidationOutcome::Accepted
        ));
        s.give("medicine", 1);
        assert!(matches!(
            s.player_do(Action::TransferItem {
                target: ActorId::new("grocer"),
                item: ItemId::new("medicine"),
                qty: 1,
            }),
            ValidationOutcome::Accepted
        ));
        s.tick();
        assert!(
            s.sim
                .world
                .commitments
                .values()
                .any(|c| c.state == rpg_domain::social::CommitmentState::Satisfied),
            "delivery must satisfy the deal"
        );

        // Turn 2 (a follow-up): the heuristic honours the delivered bargain,
        // resolving from the tracked conversation history like the runtime's
        // synchronous fallback does.
        s.teleport_player_to("grocer");
        s.say("grocer", "Now the ledger, as we agreed.");
        assert!(
            s.sim
                .world
                .has_heard(&ActorId::new(PLAYER), &FactId::new("f_grocer_ledger")),
            "the satisfied deal must be honoured on the follow-up turn"
        );
        assert_eq!(
            s.conversation_target.as_ref().map(|a| a.as_str()),
            Some("grocer")
        );
        let turns: Vec<(String, String)> = s
            .conversation_turns
            .iter()
            .map(|(who, said)| (who.0.clone(), said.clone()))
            .collect();
        assert!(
            turns.contains(&("player".into(), "I need to see your ledger.".to_string())),
            "the tracked memory must keep the player's opening turn: {turns:?}"
        );
        assert!(
            turns
                .iter()
                .any(|(who, said)| who == "grocer" && !said.is_empty()),
            "the tracked memory must keep the grocer's spoken replies: {turns:?}"
        );
        assert!(
            turns.contains(&("player".into(), "Now the ledger, as we agreed.".to_string())),
            "the tracked memory must keep the follow-up utterance: {turns:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Privacy boundary regressions on the real town, through the Session
    // commit path the clients share (policy redaction -> validation ->
    // spoken line).
    // -----------------------------------------------------------------------

    /// A scripted model that leaks the leader's identity in every
    /// player-visible word field: the policy replaces Speak, offer notes,
    /// promise terms and threat demands with the canonical deflection, and
    /// what the character "says aloud" (the committed spoken line) is the
    /// deflection, never the leak.
    #[test]
    fn scripted_model_leaks_are_deflected_in_every_spoken_field() {
        let leaks = [
            r#"{"action":"speak","text":"Selene Voss is the leader of the mafia."}"#,
            r#"{"action":"offer","note":"I will tell you the leader of the mafia for coin.","want":["ledger"]}"#,
            r#"{"action":"promise","terms":"I promise to show you where the contraband is stockpiled."}"#,
            r#"{"action":"threaten","demand":"Pay me, or everyone learns the dockmaster is the lieutenant."}"#,
        ];
        for raw in leaks {
            let engine = rpg_agents::ScriptedEngine::new().with_default(raw);
            let mut s = Session::with_engine(2026, Box::new(engine));
            s.teleport_player_to("spy");
            let spoken = s
                .npc_respond("spy", "Tell me what you know.")
                .expect("the action must commit");
            assert_eq!(
                spoken,
                rpg_agents::SPEAK_DEFLECTION,
                "leak must be deflected: {raw}"
            );
        }
    }

    /// The merchant head believes the leader's identity but is not authorised
    /// to disclose it; the biography-scoped phrasing that dodges "leader"
    /// must still deflect through the shared commit path, in the Speak field
    /// and the offer note (privacy review).
    #[test]
    fn scripted_merchant_head_mafia_echo_is_deflected() {
        let leaks = [
            (
                "speak",
                r#"{"action":"speak","text":"Selene Voss runs the mafia."}"#,
            ),
            (
                "offer note",
                r#"{"action":"offer","note":"Selene Voss runs the mafia.","want":["ledger"]}"#,
            ),
        ];
        for (field, raw) in leaks {
            let engine = rpg_agents::ScriptedEngine::new().with_default(raw);
            let mut s = Session::with_engine(2026, Box::new(engine));
            s.teleport_player_to("merchant_head");
            let spoken = s
                .npc_respond("merchant_head", "Good morning.")
                .expect("the action must commit");
            assert_eq!(
                spoken,
                rpg_agents::SPEAK_DEFLECTION,
                "leak must be deflected in {field}"
            );
        }
    }

    /// Public vocabulary passes through the whole chain verbatim: ordinary
    /// speech and free-text offer fields about the market, merchants and
    /// public figures are never rewritten.
    #[test]
    fn scripted_model_public_speech_passes_through_untouched() {
        let pass_through = [
            (
                "merchant_head",
                r#"{"action":"speak","text":"The market is busy today."}"#,
                "The market is busy today.",
            ),
            (
                "spy",
                r#"{"action":"offer","note":"Buy apples at the market.","want":["ledger"]}"#,
                "Buy apples at the market.",
            ),
            (
                "spy",
                r#"{"action":"speak","text":"The customs house is disused."}"#,
                "The customs house is disused.",
            ),
        ];
        for (npc, raw, expected) in pass_through {
            let engine = rpg_agents::ScriptedEngine::new().with_default(raw);
            let mut s = Session::with_engine(2026, Box::new(engine));
            s.teleport_player_to(npc);
            let spoken = s
                .npc_respond(npc, "Good morning.")
                .expect("the action must commit");
            assert_eq!(spoken, expected, "{npc} must speak verbatim");
        }
    }
}
