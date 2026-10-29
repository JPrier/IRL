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

/// A scripted/simulated player connected to a live simulation and NPC policy.
pub struct Session {
    pub sim: Simulation,
    pub engine: Box<dyn InferenceEngine>,
    pub next_request: u64,
    pub trace: Vec<String>,
    /// Provenance of every NPC decision, in order.
    pub decisions: Vec<(String, rpg_agents::DecisionSource)>,
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
    /// Returns the NPC's spoken reply text when the accepted action was
    /// `Speak`, so a voice client can synthesize exactly what the character
    /// said. Returns `None` when the speech did not reach the NPC or the NPC
    /// chose a non-verbal action.
    pub fn say(&mut self, target: &str, text: &str) -> Option<String> {
        // An NPC can only answer speech that actually reached them (in range
        // and audible). Rejected speech must not burn inference or produce an
        // impossible response.
        if matches!(self.player_speak(target, text), ValidationOutcome::Accepted) {
            self.npc_respond(target, text)
        } else {
            None
        }
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
    /// * `Speak` → its `text`.
    /// * `MakeOffer` → its `note` ("what you promise in return").
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
        after.trust < before.trust,
        format!("{} -> {}", before.trust, after.trust),
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
}
