//! # rpg_sim
//!
//! The authoritative simulation. It owns the canonical [`World`], the
//! append-only [`EventLog`], the deterministic RNG, and the perception engine.
//!
//! **Nothing else may mutate canonical state.** Every change flows through
//! [`Simulation::apply`], which validates a typed [`ActionProposal`] and emits
//! typed events. Model output is always an untrusted proposal.

pub mod autonomy;
pub mod pathfind;

use rpg_domain::action::{
    Action, ActionProposal, InvestigateTarget, Loudness, ProposalSource, QuestPredicate,
    RejectReason, SanctionKind, ValidationOutcome,
};
use rpg_domain::behavior_types::{ActionOutcome, CausalTrace, Effect, GoalChange};
use rpg_domain::deal::{self as deal_fsm, Deal, DealEvent, Party, ViolationReason};
use rpg_domain::events::{EventKind, EventLog};
use rpg_domain::geom::Vec2;
use rpg_domain::ids::*;
use rpg_domain::rng::DeterministicRng;
use rpg_domain::social::{CommitmentKind, Offer, OfferState};
use rpg_domain::world::{
    evaluate_predicate, Actor, ActorKind, ContainerKind, ItemStack as Stack, LocationKind,
    Sensitivity, World,
};
use rpg_narrative::{self as narrative, Decision, NarrativeChange};
use rpg_perception::PerceptionEngine;
use rpg_social as social;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::autonomy::GoalAutonomy;

use rpg_behavior::{AppraisedEvent, DecayRates};
use rpg_domain::emotions::{EmotionResponse, EmotionsConfig};
use rpg_domain::needs::{NeedCrossing, NeedKind, NeedsConfig, ThresholdDirection};

/// The need a goal kind maps onto for urgency (design §3 `N` → goal
/// urgency; T3.8). Goal kinds without a needs analog map to nothing.
fn goal_kind_need(kind: rpg_domain::world::GoalKind) -> Option<NeedKind> {
    use rpg_domain::needs::NeedKind as N;
    use rpg_domain::world::GoalKind as G;
    match kind {
        G::Survival => Some(N::Health),
        G::Wealth => Some(N::Wealth),
        G::Safety | G::Protect => Some(N::Safety),
        G::Affection => Some(N::Belonging),
        G::Status => Some(N::Status),
        G::Duty | G::Ideology | G::Ambition | G::Revenge | G::Investigate | G::Conceal => None,
    }
}

/// Tunable, scenario-fixed simulation parameters.
#[derive(Clone, Debug)]
pub struct SimConfig {
    /// Simulated minutes advanced per `step`.
    pub minutes_per_tick: u32,
    /// Maximum conversational distance for targeted actions.
    pub talk_radius: f32,
    /// Base movement distance per tick (scaled by actor speed).
    pub move_units_per_tick: f32,
    /// Run faction budgets and ambient incidents.
    pub enable_background: bool,
    /// Run the deterministic director (checkpoints + clue placement).
    pub enable_director: bool,
    /// Enable deterministic goal-driven autonomy (pursuits) for NPCs.
    pub enable_goal_autonomy: bool,
    /// Pacing margin (in ticks) added on top of the estimated travel time for
    /// an autonomous pursuit; the deadline scales with the planned path.
    pub goal_pursuit_margin_ticks: u32,
    /// Ticks to pause goal autonomy after a completed pursuit.
    pub goal_cooldown_ticks: u32,
    /// Minimum minutes between ambient incident rolls.
    pub incident_interval_minutes: u32,
    /// Grace window (simulated minutes) after a crossed deal deadline before
    /// the deal is violated (plan §4.5 renegotiation window).
    pub deal_grace_minutes: u64,
    /// Needs state tuning (design §3 `N`, plan I-4a): per-need decay,
    /// satiation thresholds, urgency thresholds with hysteresis.
    pub needs: NeedsConfig,
    /// Emotions state tuning (design §3 `E`, plan I-4b): per-emotion decay,
    /// base sensitivities, response thresholds.
    pub emotions: EmotionsConfig,
    /// Directed contextual relationship decay rates + appraisal-memory
    /// salience decay (design §3.2, plan I-4).
    pub relationship_decay: DecayRates,
    /// Salience lost per tick by an appraisal memory (0 keeps memories).
    pub rel_memory_salience_decay: f32,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            minutes_per_tick: 1,
            talk_radius: 3.5,
            move_units_per_tick: 2.0,
            enable_background: true,
            enable_director: true,
            enable_goal_autonomy: true,
            goal_pursuit_margin_ticks: 10,
            goal_cooldown_ticks: 60,
            incident_interval_minutes: 60,
            deal_grace_minutes: 30,
            needs: NeedsConfig::default(),
            emotions: EmotionsConfig::default(),
            relationship_decay: DecayRates::default(),
            rel_memory_salience_decay: 0.002,
        }
    }
}

/// Summary of one `step`, used by tests, the scenario runner and the UI.
#[derive(Clone, Debug, Default)]
pub struct StepReport {
    pub accepted: usize,
    pub pending: usize,
    pub rejected: usize,
    pub satisfied_commitments: Vec<CommitmentId>,
    pub activated_checkpoints: Vec<String>,
    pub placed_clues: Vec<FactId>,
    pub background_incidents: Vec<String>,
    pub director_notes: Vec<String>,
    /// Emotion response classes currently gated by threshold crossings
    /// (design §11 Phase 3: flee/retaliate/comfort/reconcile are *gated*,
    /// not hardcoded; the I-6 planner consumes these).
    pub gated_responses: Vec<(ActorId, EmotionResponse)>,
}

/// A pending private conversation: both actors are walking to `location`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PendingPrivate {
    pub a: ActorId,
    pub b: ActorId,
    pub location: LocationId,
}

/// State that is not canonical world truth but must survive save/load so that
/// subsequent simulation is bit-identical.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SimTransient {
    pub paths: BTreeMap<ActorId, Vec<Vec2>>,
    pub pending_private: Vec<PendingPrivate>,
    pub last_incident_minute: u64,
    /// Transient autonomy state, persisted for exact continuation.
    #[serde(default)]
    pub autonomy: GoalAutonomy,
    /// Adopted deal-response intentions (design §3 `J`): mind-local,
    /// persisted for exact continuation (an adopted intention survives
    /// save/load like the rest of the step pipeline state).
    #[serde(default)]
    pub deal_intentions: BTreeMap<ActorId, rpg_behavior::Intention>,
}

/// A recorded public entry point, used for input/proposal replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SimOp {
    /// A direct `apply` call. Boxed to keep the enum small: `ActionProposal`
    /// is much larger than a `Vec` of proposals and this is a hot replay type.
    Apply(Box<ActionProposal>),
    /// A full `step` with its external proposals.
    Step(Vec<ActionProposal>),
}

#[derive(Clone, Debug)]
pub struct Simulation {
    /// The original seed, distinct from the live RNG state.
    pub seed: u64,
    pub world: World,
    pub events: EventLog,
    pub rng: DeterministicRng,
    pub perception: PerceptionEngine,
    pub config: SimConfig,
    /// Transient movement paths (persisted so continuation is exact).
    pub paths: BTreeMap<ActorId, Vec<Vec2>>,
    /// Human-readable trace used by debug overlays and failure diagnostics.
    pub trace: Vec<String>,
    last_incident_minute: u64,
    /// World state at the start of the simulation, kept for exact replay.
    pub initial_world: World,
    /// Recorded public entry points (proposal replay).
    pub ops: Vec<SimOp>,
    /// Pending private-conversation walks.
    pub pending_private: Vec<PendingPrivate>,
    /// Transient goal-autonomy state.
    pub autonomy: GoalAutonomy,
    /// Adopted deal-response intentions (design §3 `J`).
    pub deal_intentions: BTreeMap<ActorId, rpg_behavior::Intention>,
    /// When false, `apply`/`step` do not append to `ops` (used during replay).
    pub recording: bool,
    /// Typed `ActionOutcome` records for every validated proposal
    /// (instrumentation; rebuilt identically by replay, not persisted).
    pub outcomes: Vec<ActionOutcome>,
    /// Inspectable causal traces ("did X because of goal Y, belief Z,
    /// observed event E, cost C") — one per validated proposal.
    pub causal_traces: Vec<CausalTrace>,
    /// Scratch buffer for the primitive effects produced by the dispatch
    /// handler currently running. Drained into the `ActionOutcome`.
    effects_buf: Vec<Effect>,
}

impl Simulation {
    pub fn new(world: World, seed: u64, config: SimConfig) -> Self {
        Self {
            seed,
            initial_world: world.clone(),
            world,
            events: EventLog::new(),
            rng: DeterministicRng::new(seed),
            perception: PerceptionEngine::default(),
            config,
            paths: BTreeMap::new(),
            trace: Vec::new(),
            last_incident_minute: 0,
            ops: Vec::new(),
            pending_private: Vec::new(),
            autonomy: GoalAutonomy::default(),
            deal_intentions: BTreeMap::new(),
            recording: true,
            outcomes: Vec::new(),
            causal_traces: Vec::new(),
            effects_buf: Vec::new(),
        }
    }

    /// Rebuild a simulation from a persisted world + event log.
    pub fn restore(world: World, events: EventLog, seed: u64, config: SimConfig) -> Self {
        let last_incident_minute = world.day as u64 * 1440 + world.minute_of_day as u64;
        let mut sim = Self::new(world, seed, config);
        sim.events = events;
        sim.last_incident_minute = last_incident_minute;
        sim
    }

    /// Restore the full state needed for bit-identical continuation.
    #[allow(clippy::too_many_arguments)]
    pub fn restore_full(
        seed: u64,
        world: World,
        events: EventLog,
        rng: DeterministicRng,
        transient: SimTransient,
        ops: Vec<SimOp>,
        initial_world: World,
        config: SimConfig,
    ) -> Self {
        Self {
            seed,
            world,
            events,
            rng,
            perception: PerceptionEngine::default(),
            config,
            paths: transient.paths,
            trace: Vec::new(),
            last_incident_minute: transient.last_incident_minute,
            initial_world,
            ops,
            pending_private: transient.pending_private,
            autonomy: transient.autonomy,
            deal_intentions: transient.deal_intentions,
            recording: true,
            outcomes: Vec::new(),
            causal_traces: Vec::new(),
            effects_buf: Vec::new(),
        }
    }

    /// Transient state that must be persisted for exact continuation.
    pub fn transient(&self) -> SimTransient {
        SimTransient {
            paths: self.paths.clone(),
            pending_private: self.pending_private.clone(),
            last_incident_minute: self.last_incident_minute,
            autonomy: self.autonomy.clone(),
            deal_intentions: self.deal_intentions.clone(),
        }
    }

    /// Re-run a recorded entry-point log from the initial world. This is an
    /// input/proposal replay: it reconstructs canonical state exactly without
    /// re-running stochastic inference.
    pub fn replay_ops(initial_world: World, seed: u64, config: SimConfig, ops: &[SimOp]) -> Self {
        let mut sim = Self::new(initial_world, seed, config);
        sim.recording = false;
        for op in ops {
            match op {
                SimOp::Apply(p) => {
                    let _ = sim.apply_impl(p);
                }
                SimOp::Step(props) => {
                    let _ = sim.step_impl(props.clone());
                }
            }
        }
        sim
    }

    pub fn tick(&self) -> u64 {
        self.world.tick
    }

    pub fn minute(&self) -> u32 {
        self.world.minute_of_day
    }

    // -----------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------

    pub fn validate(&self, proposal: &ActionProposal) -> Result<(), RejectReason> {
        let Some(actor) = self.world.actor(&proposal.actor) else {
            return Err(RejectReason::UnknownActor(proposal.actor.clone()));
        };
        if !actor.is_alive() {
            return Err(RejectReason::ActorNotAlive(proposal.actor.clone()));
        }
        let in_range = |t: &ActorId| -> Result<(), RejectReason> {
            let Some(target) = self.world.actor(t) else {
                return Err(RejectReason::UnknownActor(t.clone()));
            };
            if !target.is_alive() {
                return Err(RejectReason::ActorNotAlive(t.clone()));
            }
            let same_room = actor.location == target.location;
            let near = actor.pos.distance(target.pos) <= self.config.talk_radius;
            if same_room && near {
                Ok(())
            } else {
                Err(RejectReason::OutOfRange)
            }
        };

        match &proposal.action {
            Action::Speak { target, text, .. } => {
                if text.trim().is_empty() {
                    return Err(RejectReason::Invalid("empty speech".into()));
                }
                if let Some(t) = target {
                    in_range(t)?;
                }
            }
            Action::AskPrivately { target } => {
                in_range(target)?;
                if self.find_private_location_near(actor).is_none() {
                    return Err(RejectReason::Invalid(
                        "no private location available".into(),
                    ));
                }
            }
            Action::MoveTo { location } => {
                let Some(loc) = self.world.location(location) else {
                    return Err(RejectReason::UnknownLocation(location.clone()));
                };
                if pathfind::plan_world_path(&self.world.nav, actor.pos, loc.center()).is_none() {
                    return Err(RejectReason::Unreachable);
                }
            }
            Action::MoveToPos { pos } => {
                if pathfind::plan_world_path(&self.world.nav, actor.pos, *pos).is_none() {
                    return Err(RejectReason::Unreachable);
                }
            }
            Action::MakeOffer { target, give, .. } => {
                in_range(target)?;
                for stack in give {
                    if !self.world.items.contains_key(&stack.item) {
                        return Err(RejectReason::UnknownItem(stack.item.clone()));
                    }
                    if actor.inventory.quantity(&stack.item) < stack.qty {
                        return Err(RejectReason::InsufficientItems(stack.item.clone()));
                    }
                }
            }
            Action::AcceptOffer { offer } | Action::RejectOffer { offer } => {
                let Some(o) = self.world.offers.get(offer) else {
                    return Err(RejectReason::UnknownOffer(*offer));
                };
                if o.state != OfferState::Pending {
                    return Err(RejectReason::Invalid("offer is not pending".into()));
                }
                if o.to != proposal.actor {
                    return Err(RejectReason::Invalid(
                        "only the recipient may respond".into(),
                    ));
                }
            }
            Action::Promise { target, .. } => {
                in_range(target)?;
            }
            Action::BreakPromise { commitment } => {
                let Some(c) = self.world.commitments.get(commitment) else {
                    return Err(RejectReason::UnknownCommitment(*commitment));
                };
                if c.debtor != proposal.actor {
                    return Err(RejectReason::Invalid("not your commitment".into()));
                }
            }
            Action::ProposeDeal {
                target,
                condition,
                performance,
                give,
                money,
                ..
            } => {
                in_range(target)?;
                // A contract is an agreement between two distinct parties
                // (design §5.2: proposal, acceptance, consideration between
                // debtor and creditor). A self-deal has no counterparty and
                // no real exchange — reject it outright.
                if *target == proposal.actor {
                    return Err(RejectReason::Invalid(
                        "a deal requires two distinct parties".into(),
                    ));
                }
                self.validate_predicate_refs(condition)?;
                self.validate_predicate_refs(performance)?;
                for stack in give {
                    if !self.world.items.contains_key(&stack.item) {
                        return Err(RejectReason::UnknownItem(stack.item.clone()));
                    }
                }
                // Consideration custody is validated at acceptance/settlement
                // time (plan §4.2: the debtor commits only resources it
                // controls; there is no v1 lock), but negative money would
                // *mint* value, so it is structurally rejected now.
                if *money < 0 {
                    return Err(RejectReason::Invalid(
                        "deal consideration cannot be negative".into(),
                    ));
                }
            }
            Action::AcceptDeal { deal } => {
                let Some(d) = self.world.deals.get(deal) else {
                    return Err(RejectReason::UnknownDeal(*deal));
                };
                if d.state != rpg_domain::deal::DealState::Proposed {
                    return Err(RejectReason::Invalid("deal is not proposed".into()));
                }
                if d.creditor != proposal.actor {
                    return Err(RejectReason::Invalid(
                        "only the creditor may accept a deal".into(),
                    ));
                }
                // Escrow (§4.2): validate the *whole* consideration bundle
                // before anything mutates — every component or nothing.
                // Quantities aggregate per item across duplicate stacks with
                // checked arithmetic, so a bundle like [{X,10},{X,10}] held
                // against 10 is rejected here instead of silently
                // under-delivering at dispatch.
                let Some(debtor) = self.world.actor(&d.debtor) else {
                    return Err(RejectReason::UnknownActor(d.debtor.clone()));
                };
                let mut need: BTreeMap<&ItemId, u32> = BTreeMap::new();
                for stack in &d.give {
                    if !self.world.items.contains_key(&stack.item) {
                        return Err(RejectReason::UnknownItem(stack.item.clone()));
                    }
                    let total = need.entry(&stack.item).or_insert(0);
                    *total = total.checked_add(stack.qty).ok_or_else(|| {
                        RejectReason::Invalid(format!(
                            "consideration bundle overflows for item {}",
                            stack.item.as_str()
                        ))
                    })?;
                }
                for (item, qty) in &need {
                    if debtor.inventory.quantity(item) < *qty {
                        return Err(RejectReason::InsufficientItems((*item).clone()));
                    }
                }
                if d.money > 0 && debtor.inventory.money < d.money {
                    return Err(RejectReason::InsufficientFunds);
                }
            }
            Action::RejectDeal { deal } => {
                let Some(d) = self.world.deals.get(deal) else {
                    return Err(RejectReason::UnknownDeal(*deal));
                };
                if d.state != rpg_domain::deal::DealState::Proposed {
                    return Err(RejectReason::Invalid("deal is not proposed".into()));
                }
                if d.creditor != proposal.actor {
                    return Err(RejectReason::Invalid(
                        "only the creditor may reject a deal".into(),
                    ));
                }
            }
            Action::CancelDeal { deal } => {
                let Some(d) = self.world.deals.get(deal) else {
                    return Err(RejectReason::UnknownDeal(*deal));
                };
                if proposal.actor != d.debtor && proposal.actor != d.creditor {
                    return Err(RejectReason::Invalid(
                        "only a deal party may cancel it".into(),
                    ));
                }
            }
            Action::SettleDeal {
                deal,
                deliver,
                money,
            } => {
                let Some(d) = self.world.deals.get(deal) else {
                    return Err(RejectReason::UnknownDeal(*deal));
                };
                if d.state != rpg_domain::deal::DealState::PerformanceDue {
                    return Err(RejectReason::Invalid(
                        "deal is not due for performance".into(),
                    ));
                }
                if d.debtor != proposal.actor {
                    return Err(RejectReason::Invalid("only the debtor may settle".into()));
                }
                if *money < 0 {
                    return Err(RejectReason::Invalid(
                        "settlement cannot be negative".into(),
                    ));
                }
                // Escrow (§4.2): the settlement bundle is one validated
                // proposal. Custody for every component is checked here —
                // aggregated per item across duplicate stacks with checked
                // arithmetic — so dispatch either performs every transfer or
                // — on any validation failure — mutates nothing.
                let mut need: BTreeMap<&ItemId, u32> = BTreeMap::new();
                for stack in deliver {
                    if !self.world.items.contains_key(&stack.item) {
                        return Err(RejectReason::UnknownItem(stack.item.clone()));
                    }
                    let total = need.entry(&stack.item).or_insert(0);
                    *total = total.checked_add(stack.qty).ok_or_else(|| {
                        RejectReason::Invalid(format!(
                            "settlement bundle overflows for item {}",
                            stack.item.as_str()
                        ))
                    })?;
                }
                for (item, qty) in &need {
                    if actor.inventory.quantity(item) < *qty {
                        return Err(RejectReason::InsufficientItems((*item).clone()));
                    }
                }
                if actor.inventory.money < *money {
                    return Err(RejectReason::InsufficientFunds);
                }
            }
            Action::RenegotiateDeal { deal, .. } => {
                let Some(d) = self.world.deals.get(deal) else {
                    return Err(RejectReason::UnknownDeal(*deal));
                };
                if d.state != rpg_domain::deal::DealState::PerformanceDue {
                    return Err(RejectReason::Invalid(
                        "only a due deal can be renegotiated".into(),
                    ));
                }
                if d.creditor != proposal.actor {
                    return Err(RejectReason::Invalid(
                        "renegotiation requires creditor consent".into(),
                    ));
                }
                if !d.can_renegotiate() {
                    return Err(RejectReason::Invalid(
                        "the single renegotiation cycle is spent".into(),
                    ));
                }
            }
            Action::TransferItem { target, item, qty } => {
                in_range(target)?;
                if actor.inventory.quantity(item) < *qty {
                    return Err(RejectReason::InsufficientItems(item.clone()));
                }
            }
            Action::Pay { target, amount } => {
                in_range(target)?;
                // Payment must be a real transfer: zero or negative amounts
                // would move nothing (or mint money) — reject before any
                // mutation.
                if *amount <= 0 {
                    return Err(RejectReason::Invalid("pay amount must be positive".into()));
                }
                if actor.inventory.money < *amount {
                    return Err(RejectReason::InsufficientFunds);
                }
            }
            Action::ShareClaim {
                target,
                fact,
                deceptive,
                ..
            } => {
                in_range(target)?;
                self.validate_disclosure_knowledge(actor, fact, *deceptive)?;
                self.validate_disclosure_permission(actor, fact)?;
            }
            Action::WarnContact { target, fact } => {
                in_range(target)?;
                self.validate_disclosure_knowledge(actor, fact, false)?;
                self.validate_disclosure_permission(actor, fact)?;
            }
            Action::Query { target, about } => {
                in_range(target)?;
                if !self.world.facts.contains_key(about) {
                    return Err(RejectReason::UnknownFact(about.clone()));
                }
            }
            Action::Sanction {
                target,
                kind,
                evidence,
            } => {
                in_range(target)?;
                // AUTHORITY: a sanction is an institutional act; the proposer
                // must actually hold institutional authority (capability or
                // the canonical authority flag). Without it the act is
                // *not permitted* — distinct from not possible.
                if !actor.capabilities.can_arrest && !actor.has_flag("authority") {
                    return Err(RejectReason::NoCapability("impose sanctions".into()));
                }
                // EVIDENCE: the authority must itself hold (know of) every
                // cited fact — provenance-bound grounds, not bare assertion.
                if evidence.is_empty() {
                    return Err(RejectReason::Invalid(
                        "a sanction requires cited evidence".into(),
                    ));
                }
                for fact in evidence {
                    if !self.world.facts.contains_key(fact) {
                        return Err(RejectReason::UnknownFact(fact.clone()));
                    }
                    if !rpg_behavior::knows_of(&self.world, &actor.id, fact) {
                        return Err(RejectReason::LacksKnowledge(fact.clone()));
                    }
                }
                // COST: a fine is a real transfer; it must be payable (a
                // fine the subject cannot pay would mint or clamp money).
                if let SanctionKind::Fine { amount } = kind {
                    if *amount <= 0 {
                        return Err(RejectReason::Invalid("fine amount must be positive".into()));
                    }
                    let target_actor = self.world.actor(target).expect("checked in_range");
                    if target_actor.inventory.money < *amount {
                        return Err(RejectReason::InsufficientFunds);
                    }
                }
            }
            Action::ExcuseDeal { deal, evidence } => {
                let Some(d) = self.world.deals.get(deal) else {
                    return Err(RejectReason::UnknownDeal(*deal));
                };
                // Authority/institution evidence excuses a due deal (plan
                // §4.4); every other state is an illegal FSM arrow.
                if d.state != rpg_domain::deal::DealState::PerformanceDue {
                    return Err(RejectReason::Invalid(format!(
                        "only a due deal can be excused (state: {})",
                        d.state.name()
                    )));
                }
                if !actor.capabilities.can_arrest && !actor.has_flag("authority") {
                    return Err(RejectReason::NoCapability("excuse contracts".into()));
                }
                if evidence.is_empty() {
                    return Err(RejectReason::Invalid(
                        "an excuse requires cited authority evidence".into(),
                    ));
                }
                for fact in evidence {
                    if !self.world.facts.contains_key(fact) {
                        return Err(RejectReason::UnknownFact(fact.clone()));
                    }
                    if !rpg_behavior::knows_of(&self.world, &actor.id, fact) {
                        return Err(RejectReason::LacksKnowledge(fact.clone()));
                    }
                }
            }
            Action::SpreadRumor { .. } => {}
            Action::Investigate { target } => match target {
                InvestigateTarget::Location(l) => {
                    if self.world.location(l).is_none() {
                        return Err(RejectReason::UnknownLocation(l.clone()));
                    }
                    // You must actually be there; no remote investigation.
                    if actor.location != *l {
                        return Err(RejectReason::OutOfRange);
                    }
                }
                InvestigateTarget::Actor(a) => {
                    in_range(a)?;
                }
                InvestigateTarget::Item(i) => {
                    if !self.world.items.contains_key(i) {
                        return Err(RejectReason::UnknownItem(i.clone()));
                    }
                }
            },
            Action::ReportCrime { authority, .. } => {
                let Some(auth) = self.world.actor(authority) else {
                    return Err(RejectReason::UnknownActor(authority.clone()));
                };
                if !auth.capabilities.can_arrest && !auth.has_flag("authority") {
                    return Err(RejectReason::NoCapability("report authority".into()));
                }
            }
            Action::HideEvidence { item, location } => {
                if actor.inventory.quantity(item) == 0 {
                    return Err(RejectReason::InsufficientItems(item.clone()));
                }
                if self.world.location(location).is_none() {
                    return Err(RejectReason::UnknownLocation(location.clone()));
                }
            }
            Action::AllocateBudget {
                faction, amount, ..
            } => {
                let Some(f) = self.world.faction(faction) else {
                    return Err(RejectReason::UnknownFaction(faction.clone()));
                };
                if !actor.capabilities.can_allocate_budget && !f.members.contains(&proposal.actor) {
                    return Err(RejectReason::NoCapability("allocate budget".into()));
                }
                if *amount < 0 || f.budget < *amount {
                    return Err(RejectReason::InsufficientFunds);
                }
            }
            Action::Repair { location } => {
                if self.world.location(location).is_none() {
                    return Err(RejectReason::UnknownLocation(location.clone()));
                }
                if self.world.damage_at(location) <= 0.0 {
                    return Err(RejectReason::Invalid("nothing to repair".into()));
                }
                if !actor.capabilities.can_repair && actor.affiliations.is_empty() {
                    return Err(RejectReason::NoCapability("repair".into()));
                }
            }
            Action::Recruit { target } => {
                in_range(target)?;
            }
            Action::ScheduleMeeting {
                target, location, ..
            } => {
                in_range(target)?;
                if self.world.location(location).is_none() {
                    return Err(RejectReason::UnknownLocation(location.clone()));
                }
            }
            Action::Threaten { target, .. } => {
                in_range(target)?;
            }
            Action::Comfort { target } => {
                in_range(target)?;
            }
            Action::PlaceClue { fact, .. } => {
                if let Some(f) = self.world.fact(fact) {
                    if f.committed {
                        return Err(RejectReason::Invalid("fact already committed".into()));
                    }
                } else {
                    return Err(RejectReason::UnknownFact(fact.clone()));
                }
            }
            Action::Idle => {}
        }
        Ok(())
    }

    /// The knowledge half of disclosure validation (plan I-5, G1): the
    /// truthful path requires the speaker to *believe* the fact; a
    /// `deceptive` (intentional lie) proposal instead requires the speaker to
    /// *know of* the proposition — authored belief of either polarity or
    /// ledger evidence — because a lie is exactly an assertion contrary to
    /// the speaker's own belief. World truth is never consulted here.
    fn validate_disclosure_knowledge(
        &self,
        actor: &Actor,
        fact: &FactId,
        deceptive: bool,
    ) -> Result<(), RejectReason> {
        if self.world.fact(fact).is_none() {
            return Err(RejectReason::UnknownFact(fact.clone()));
        }
        let knows = if deceptive {
            rpg_behavior::knows_of(&self.world, &actor.id, fact)
        } else {
            self.world.believes(&actor.id, fact)
        };
        if !knows {
            return Err(RejectReason::LacksKnowledge(fact.clone()));
        }
        Ok(())
    }

    /// The permission half of disclosure validation: knowledge is not
    /// permission (design §9.1.2), and a deceptive assertion does not relax
    /// it — lying about a secret is still a disclosure of the secret.
    fn validate_disclosure_permission(
        &self,
        actor: &Actor,
        fact: &FactId,
    ) -> Result<(), RejectReason> {
        let Some(f) = self.world.fact(fact) else {
            return Err(RejectReason::UnknownFact(fact.clone()));
        };
        let holder = f.eligible_holders.is_empty() || f.eligible_holders.contains(&actor.id);
        let authorized = match f.sensitivity {
            Sensitivity::Public | Sensitivity::Restricted => true,
            Sensitivity::Private | Sensitivity::Secret => holder,
            Sensitivity::InnerCircle => actor.has_flag("inner_circle") && holder,
        };
        if !authorized {
            return Err(RejectReason::NotPermittedToDisclose(fact.clone()));
        }
        Ok(())
    }

    // -----------------------------------------------------------------
    // Application
    // -----------------------------------------------------------------

    pub fn apply(&mut self, proposal: &ActionProposal) -> ValidationOutcome {
        if self.recording {
            self.ops.push(SimOp::Apply(Box::new(proposal.clone())));
        }
        self.apply_impl(proposal)
    }

    fn apply_impl(&mut self, proposal: &ActionProposal) -> ValidationOutcome {
        self.effects_buf.clear();
        let action_kind = proposal.action.kind_name().to_string();
        if let Err(reason) = self.validate(proposal) {
            // Rejection mutates nothing: no effects, no new world state.
            let trace = self.build_trace(proposal, &[], Some(&reason), self.events.next_id().raw());
            let outcome = rpg_domain::behavior_types::rejected_outcome(
                proposal.actor.clone(),
                &action_kind,
                reason.clone(),
                trace.clone(),
            );
            self.outcomes.push(outcome);
            self.causal_traces.push(trace);
            self.reject(proposal, &reason);
            return ValidationOutcome::Rejected(reason);
        }
        let first_event = self.events.next_id().raw();
        let outcome = self.dispatch(proposal);
        if let ValidationOutcome::Rejected(reason) = &outcome {
            self.reject(proposal, reason);
        }
        // Event-sourced mutation record: every event the handler emitted,
        // every primitive effect it produced, one causal trace. A rejection
        // mutates nothing, so its outcome carries no effects by construction.
        let last_event = self.events.next_id().raw();
        let events: Vec<rpg_domain::ids::EventId> = (first_event..last_event)
            .map(rpg_domain::ids::EventId)
            .collect();
        let effects = if matches!(outcome, ValidationOutcome::Rejected(_)) {
            self.effects_buf.clear();
            Vec::new()
        } else {
            std::mem::take(&mut self.effects_buf)
        };
        let trace = self.build_trace(proposal, &effects, None, first_event);
        self.outcomes.push(ActionOutcome {
            actor: proposal.actor.clone(),
            action_kind: action_kind.clone(),
            validation: outcome.clone(),
            effects: effects.clone(),
            events: events.clone(),
            trace: trace.clone(),
        });
        self.causal_traces.push(trace);
        outcome
    }

    /// Build an inspectable causal trace for one validated proposal:
    /// *NPC did X because of goal Y, belief Z, observed event E, and cost C.*
    ///
    /// The trace is instrumentation, so it may only cite real causes:
    /// `up_to` bounds the event log to what existed *before* this proposal
    /// was dispatched, and observations must have been directly perceived by
    /// the actor. When there is no genuine cause to cite for a field, the
    /// field stays empty — empty is preferable to false causality.
    fn build_trace(
        &self,
        proposal: &ActionProposal,
        effects: &[Effect],
        rejected: Option<&RejectReason>,
        up_to: u64,
    ) -> CausalTrace {
        let action_kind = proposal.action.kind_name().to_string();
        let actor = self.world.actor(&proposal.actor);

        // Goals: active goals of the actor, highest priority first.
        let mut because_goals: Vec<String> = actor
            .map(|a| {
                let mut goals: Vec<&rpg_domain::world::Goal> = a
                    .goals
                    .iter()
                    .filter(|g| g.status == rpg_domain::world::GoalStatus::Active)
                    .collect();
                goals.sort_by(|x, y| {
                    y.priority
                        .partial_cmp(&x.priority)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                goals
                    .iter()
                    .take(2)
                    .map(|g| g.description.clone())
                    .collect()
            })
            .unwrap_or_default();

        // Beliefs: only the facts this action is actually about, as held by
        // the actor. An unrelated most-recent belief is never substituted.
        let mut because_beliefs: Vec<FactId> = action_facts(&proposal.action)
            .into_iter()
            .filter(|f| actor.map(|a| a.beliefs.contains_key(f)).unwrap_or(false))
            .collect();
        because_beliefs.sort();
        because_beliefs.dedup();

        // Observations: recent events the actor *directly perceived* before
        // this proposal was dispatched. Self-caused-but-unobserved events and
        // the proposal's own events are not causes of the decision.
        let because_observations: Vec<EventId> = self
            .events
            .all()
            .iter()
            .rev()
            .filter(|e| e.id.raw() < up_to && e.observed_by.iter().any(|a| a == &proposal.actor))
            .take(3)
            .map(|e| e.id)
            .collect();

        // Costs: resource costs actually carried by the effects. Nothing is
        // invented — when an action carried no resource cost, the list stays
        // empty.
        let mut costs: Vec<String> = Vec::new();
        for e in effects {
            match e {
                Effect::Transfer {
                    item: Some(item),
                    qty,
                    ..
                } => costs.push(format!("{qty}x{}", item.as_str())),
                Effect::Transfer {
                    item: None, qty, ..
                } => costs.push(format!("{qty} coins")),
                Effect::Transform { consumed, .. } if *consumed > 0 => {
                    costs.push(format!("{consumed} consumed"))
                }
                _ => {}
            }
        }
        if let Some(reason) = rejected {
            costs.clear();
            costs.push(format!("rejected: {}", reason.message()));
            because_goals.clear();
        }

        CausalTrace {
            actor: proposal.actor.clone(),
            action_kind,
            because_goals,
            because_beliefs,
            because_observations,
            costs,
            effects: effects.to_vec(),
        }
    }

    fn reject(&mut self, proposal: &ActionProposal, reason: &RejectReason) {
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        self.events.push(
            tick,
            minute,
            Some(proposal.actor.clone()),
            EventKind::ActionRejected {
                actor: proposal.actor.clone(),
                action: proposal.action.kind_name().to_string(),
                reason: reason.clone(),
            },
            Vec::new(),
            Vec::new(),
        );
        self.trace.push(format!(
            "reject {}: {}",
            proposal.action.kind_name(),
            reason.message()
        ));
        // T2.1 abandonment accounting (design §9.2.1): a *rejected attempt*
        // of an adopted deal intention feeds its failure count exactly once
        // per attempt. Proposal generation is not an attempt — only
        // dispatches that actually reach this rejection hook count, and only
        // when the rejected action is the adopted option's own action (so
        // unrelated rejections never accumulate).
        if let Some(intent) = self.deal_intentions.get(&proposal.actor).cloned() {
            let attempted = rpg_agents::deal_response_options(
                &self.world,
                &proposal.actor,
                None,
                None,
                &self.config.needs,
            )
            .and_then(|opts| rpg_agents::deal_response_action(&self.world, &opts, &intent.option))
            .is_some_and(|action| action == proposal.action);
            if attempted {
                if let Some(intent) = self.deal_intentions.get_mut(&proposal.actor) {
                    intent.record_failure();
                }
            }
        }
    }

    fn dispatch(&mut self, proposal: &ActionProposal) -> ValidationOutcome {
        match proposal.action.clone() {
            Action::Speak {
                target,
                text,
                loudness,
            } => self.do_speak(proposal, target, text, loudness),
            Action::AskPrivately { target } => self.do_ask_privately(proposal, target),
            Action::MoveTo { location } => self.do_move_to_location(proposal, location),
            Action::MoveToPos { pos } => self.do_move_to_pos(proposal, pos),
            Action::MakeOffer {
                target,
                give,
                want,
                condition,
                disclose,
                note,
            } => self.do_make_offer(proposal, target, give, want, condition, disclose, note),
            Action::AcceptOffer { offer } => self.do_accept_offer(proposal, offer),
            Action::RejectOffer { offer } => self.do_reject_offer(proposal, offer),
            Action::Promise {
                target,
                terms,
                predicate,
                deadline,
            } => self.do_promise(proposal, target, terms, predicate, deadline),
            Action::BreakPromise { commitment } => self.do_break_promise(proposal, commitment),
            Action::ProposeDeal {
                target,
                terms,
                condition,
                performance,
                condition_deadline_minute,
                performance_deadline_minute,
                give,
                money,
                note,
            } => self.do_propose_deal(
                proposal,
                target,
                terms,
                condition,
                performance,
                condition_deadline_minute,
                performance_deadline_minute,
                give,
                money,
                note,
            ),
            Action::AcceptDeal { deal } => self.do_accept_deal(proposal, deal),
            Action::RejectDeal { deal } => self.do_reject_deal(proposal, deal),
            Action::CancelDeal { deal } => self.do_cancel_deal(proposal, deal),
            Action::SettleDeal {
                deal,
                deliver,
                money,
            } => self.do_settle_deal(proposal, deal, deliver, money),
            Action::RenegotiateDeal {
                deal,
                new_condition,
                new_performance,
                new_condition_deadline_minute,
                new_performance_deadline_minute,
            } => self.do_renegotiate_deal(
                proposal,
                deal,
                new_condition,
                new_performance,
                new_condition_deadline_minute,
                new_performance_deadline_minute,
            ),
            Action::ExcuseDeal { deal, evidence } => self.do_excuse_deal(proposal, deal, evidence),
            Action::TransferItem { target, item, qty } => {
                self.do_transfer(proposal, target, item, qty)
            }
            Action::Pay { target, amount } => self.do_pay(proposal, target, amount),
            Action::ShareClaim {
                target,
                fact,
                assertive,
                deceptive,
            } => self.do_share_claim(proposal, target, fact, assertive, deceptive),
            Action::Query { target, about } => self.do_query(proposal, target, about),
            Action::Sanction {
                target,
                kind,
                evidence,
            } => self.do_sanction(proposal, target, kind, evidence),
            Action::SpreadRumor { text, about } => self.do_spread_rumor(proposal, text, about),
            Action::Investigate { target } => self.do_investigate(proposal, target),
            Action::ReportCrime {
                authority,
                evidence,
            } => self.do_report_crime(proposal, authority, evidence),
            Action::HideEvidence { item, location } => {
                self.do_hide_evidence(proposal, item, location)
            }
            Action::WarnContact { target, fact } => {
                self.do_share_claim(proposal, target, fact, true, false)
            }
            Action::AllocateBudget {
                faction,
                amount,
                purpose,
            } => self.do_allocate_budget(proposal, faction, amount, purpose),
            Action::Repair { location } => self.do_repair(proposal, location),
            Action::Recruit { target } => self.do_recruit(proposal, target),
            Action::ScheduleMeeting {
                target,
                location,
                at_minute,
            } => self.do_schedule_meeting(proposal, target, location, at_minute),
            Action::Threaten { target, demand } => self.do_threaten(proposal, target, demand),
            Action::Comfort { target } => self.do_comfort(proposal, target),
            Action::PlaceClue {
                fact,
                kind,
                location,
                holder,
            } => self.do_place_clue(proposal, fact, kind, location, holder),
            Action::Idle => ValidationOutcome::Accepted,
        }
    }

    // --- individual action handlers -----------------------------------

    fn do_speak(
        &mut self,
        proposal: &ActionProposal,
        target: Option<ActorId>,
        text: String,
        loudness: Loudness,
    ) -> ValidationOutcome {
        let actor = self.world.actor(&proposal.actor).cloned().unwrap();
        let location = self
            .world
            .location_at(actor.pos)
            .cloned()
            .unwrap_or_else(|| actor.location.clone());
        let routing = self.perception.route_speech(
            &self.world,
            &actor.id,
            target.as_ref(),
            loudness,
            actor.pos,
            actor.facing,
            &location,
        );
        let private = actor.in_private_conversation_with.is_some();
        let mut observed: Vec<ActorId> = routing.overhearers.clone();
        if let Some(a) = &routing.addressee {
            observed.push(a.clone());
        }
        // Speaking builds familiarity with everyone who heard it.
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        self.events.push(
            tick,
            minute,
            Some(actor.id.clone()),
            EventKind::SpeechEmitted {
                speaker: actor.id.clone(),
                text,
                addressee: routing.addressee.clone(),
                loudness,
                location,
                private,
            },
            Vec::new(),
            observed.clone(),
        );
        for listener in &observed {
            social::adjust_relationship(
                &mut self.world,
                listener,
                &actor.id,
                &social::RelationshipDelta::new().familiarity(0.05),
            );
            // The listener's tie toward the speaker moved — the social
            // mutation is typed. (Routing never includes the speaker, so
            // every entry here is a real relationship mutation.)
            self.effects_buf.push(Effect::ChangeTie {
                from: listener.clone(),
                to: actor.id.clone(),
                summary: "heard the utterance (familiarity)".into(),
            });
            // OBSERVE: the utterance was delivered to this listener.
            self.effects_buf.push(Effect::Observe {
                observer: listener.clone(),
                observed: actor.id.as_str().to_string(),
                channel: rpg_domain::behavior_types::AcquisitionChannel::Overheard,
                origin_event: None,
            });
        }
        self.effects_buf.push(Effect::CreateEvent {
            summary: format!("speech by {}", actor.id),
        });
        ValidationOutcome::Accepted
    }

    /// Nearest private location reachable from `actor`.
    fn find_private_location_near(&self, actor: &Actor) -> Option<LocationId> {
        let mut candidates: Vec<(f32, LocationId)> = self
            .world
            .locations
            .values()
            .filter(|l| l.private || l.kind == LocationKind::Backroom)
            .filter(|l| pathfind::plan_world_path(&self.world.nav, actor.pos, l.center()).is_some())
            .map(|l| (actor.pos.distance(l.center()), l.id.clone()))
            .collect();
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        candidates.into_iter().map(|(_, id)| id).next()
    }

    /// Nearest private location reachable by both actors.
    fn find_private_location_for(&self, a: &Actor, b: &Actor) -> Option<LocationId> {
        let mut candidates: Vec<(f32, LocationId)> = self
            .world
            .locations
            .values()
            .filter(|l| l.private || l.kind == LocationKind::Backroom)
            .filter(|l| {
                pathfind::plan_world_path(&self.world.nav, a.pos, l.center()).is_some()
                    && pathfind::plan_world_path(&self.world.nav, b.pos, l.center()).is_some()
            })
            .map(|l| {
                (
                    a.pos.distance(l.center()) + b.pos.distance(l.center()),
                    l.id.clone(),
                )
            })
            .collect();
        candidates.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
        candidates.into_iter().map(|(_, id)| id).next()
    }

    /// Consent is a real social check; actors walk to privacy, they are not
    /// teleported.
    fn do_ask_privately(
        &mut self,
        proposal: &ActionProposal,
        target: ActorId,
    ) -> ValidationOutcome {
        let Some(actor) = self.world.actor(&proposal.actor).cloned() else {
            return ValidationOutcome::Rejected(RejectReason::UnknownActor(proposal.actor.clone()));
        };
        let Some(target_actor) = self.world.actor(&target).cloned() else {
            return ValidationOutcome::Rejected(RejectReason::UnknownActor(target.clone()));
        };
        let rel = target_actor.relationship(&proposal.actor);
        let willingness = rel.trust * 0.5 + rel.affection * 0.3 + rel.familiarity * 0.3
            - rel.fear * 0.3
            - rel.anger * 0.3
            - rel.suspicion * 0.2;
        let willing = willingness > -0.1
            || target_actor.has_flag("friendly")
            || target_actor.has_flag("shopkeeper");
        if !willing {
            return ValidationOutcome::Rejected(RejectReason::TargetUnwilling);
        }
        let Some(loc) = self.find_private_location_for(&actor, &target_actor) else {
            return ValidationOutcome::Rejected(RejectReason::Invalid(
                "no reachable private location".into(),
            ));
        };
        let center = self
            .world
            .location(&loc)
            .map(|l| l.center())
            .unwrap_or(actor.pos);
        let (Some(path_a), Some(path_b)) = (
            pathfind::plan_world_path(&self.world.nav, actor.pos, center),
            pathfind::plan_world_path(&self.world.nav, target_actor.pos, center),
        ) else {
            return ValidationOutcome::Rejected(RejectReason::Unreachable);
        };
        self.paths.insert(proposal.actor.clone(), path_a);
        self.paths.insert(target.clone(), path_b);
        self.pending_private.push(PendingPrivate {
            a: proposal.actor.clone(),
            b: target.clone(),
            location: loc.clone(),
        });
        // SCHEDULE the private meeting; the ask itself is a REQUEST.
        self.effects_buf.push(Effect::Request {
            from: proposal.actor.clone(),
            to: target.clone(),
            demand: "move somewhere private".into(),
        });
        self.effects_buf.push(Effect::Schedule {
            by: proposal.actor.clone(),
            with: target.clone(),
            at_minute: self.world.minute_of_day,
        });
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        self.events.push(
            tick,
            minute,
            Some(proposal.actor.clone()),
            EventKind::MeetingScheduled {
                with: proposal.actor.clone(),
                target,
                location: loc,
                at_minute: minute,
            },
            Vec::new(),
            Vec::new(),
        );
        ValidationOutcome::Accepted
    }

    /// Called after movement: promote arrived pairs to a private conversation.
    fn finalize_private(&mut self) {
        let pending = std::mem::take(&mut self.pending_private);
        let mut keep = Vec::new();
        for p in pending {
            let a_arrived = !self.paths.contains_key(&p.a)
                && self
                    .world
                    .actor(&p.a)
                    .map(|x| x.location == p.location)
                    .unwrap_or(false);
            let b_arrived = !self.paths.contains_key(&p.b)
                && self
                    .world
                    .actor(&p.b)
                    .map(|x| x.location == p.location)
                    .unwrap_or(false);
            if a_arrived && b_arrived {
                if let Some(a) = self.world.actor_mut(&p.a) {
                    a.in_private_conversation_with = Some(p.b.clone());
                }
                if let Some(b) = self.world.actor_mut(&p.b) {
                    b.in_private_conversation_with = Some(p.a.clone());
                }
                let (tick, minute) = (self.world.tick, self.world.minute_of_day);
                self.events.push(
                    tick,
                    minute,
                    Some(p.a.clone()),
                    EventKind::MeetingScheduled {
                        with: p.a.clone(),
                        target: p.b.clone(),
                        location: p.location.clone(),
                        at_minute: minute,
                    },
                    Vec::new(),
                    vec![p.b.clone()],
                );
            } else {
                keep.push(p);
            }
        }
        self.pending_private = keep;
    }

    fn do_move_to_location(
        &mut self,
        proposal: &ActionProposal,
        location: LocationId,
    ) -> ValidationOutcome {
        let pos = self
            .world
            .location(&location)
            .map(|l| l.center())
            .unwrap_or(Vec2::ZERO);
        self.set_path(&proposal.actor, pos)
    }

    fn do_move_to_pos(&mut self, proposal: &ActionProposal, pos: Vec2) -> ValidationOutcome {
        self.set_path(&proposal.actor, pos)
    }

    fn set_path(&mut self, actor: &ActorId, dest: Vec2) -> ValidationOutcome {
        // Moving away ends any private conversation.
        let partner = self
            .world
            .actor(actor)
            .and_then(|a| a.in_private_conversation_with.clone());
        if let Some(partner) = partner {
            if let Some(a) = self.world.actor_mut(actor) {
                a.in_private_conversation_with = None;
            }
            if let Some(p) = self.world.actor_mut(&partner) {
                p.in_private_conversation_with = None;
            }
        }
        let from = self.world.actor(actor).map(|a| a.pos).unwrap_or(Vec2::ZERO);
        match pathfind::plan_world_path(&self.world.nav, from, dest) {
            Some(path) => {
                self.paths.insert(actor.clone(), path);
                // MOVE: the committed path (execution produces ActorMoved
                // events over the following ticks).
                self.effects_buf.push(Effect::Move {
                    actor: actor.clone(),
                    from,
                    to: dest,
                });
                ValidationOutcome::Accepted
            }
            None => ValidationOutcome::Rejected(RejectReason::Unreachable),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn do_make_offer(
        &mut self,
        proposal: &ActionProposal,
        target: ActorId,
        give: Vec<rpg_domain::world::ItemStack>,
        want: Vec<rpg_domain::world::ItemStack>,
        condition: Option<QuestPredicate>,
        disclose: Option<FactId>,
        note: String,
    ) -> ValidationOutcome {
        let id = OfferId(self.world.next_offer_id);
        self.world.next_offer_id += 1;
        self.world.offers.insert(
            id,
            Offer {
                id,
                from: proposal.actor.clone(),
                to: target.clone(),
                give: give.clone(),
                want: want.clone(),
                price: 0,
                condition: condition.clone(),
                disclose: disclose.clone(),
                note: note.clone(),
                state: OfferState::Pending,
                created_tick: self.world.tick,
            },
        );
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        self.effects_buf.push(Effect::Offer {
            from: proposal.actor.clone(),
            to: target.clone(),
            offer: id,
        });
        self.events.push(
            tick,
            minute,
            Some(proposal.actor.clone()),
            EventKind::OfferCreated {
                offer: id,
                from: proposal.actor.clone(),
                to: target,
                give,
                want,
                note,
            },
            Vec::new(),
            Vec::new(),
        );
        ValidationOutcome::Pending {
            offer: Some(id),
            commitment: None,
        }
    }

    fn do_accept_offer(&mut self, proposal: &ActionProposal, offer: OfferId) -> ValidationOutcome {
        let Some(o) = self.world.offers.get(&offer).cloned() else {
            return ValidationOutcome::Rejected(RejectReason::UnknownOffer(offer));
        };
        // A trade can only complete if both sides actually hold their side.
        for stack in &o.give {
            let held = self
                .world
                .actor(&o.from)
                .map(|a| a.inventory.quantity(&stack.item))
                .unwrap_or(0);
            if held < stack.qty {
                return ValidationOutcome::Rejected(RejectReason::InsufficientItems(
                    stack.item.clone(),
                ));
            }
        }
        for stack in &o.want {
            let held = self
                .world
                .actor(&o.to)
                .map(|a| a.inventory.quantity(&stack.item))
                .unwrap_or(0);
            if held < stack.qty {
                return ValidationOutcome::Rejected(RejectReason::InsufficientItems(
                    stack.item.clone(),
                ));
            }
        }
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        // Transfer what `from` gives to the acceptor.
        for stack in &o.give {
            self.move_item(&o.from, &o.to, &stack.item, stack.qty);
        }
        for stack in &o.want {
            self.move_item(&o.to, &o.from, &stack.item, stack.qty);
        }
        if let Some(c) = &o.condition {
            let cid = social::create_commitment(
                &mut self.world,
                &mut self.events,
                tick,
                minute,
                CommitmentKind::Contract,
                &o.from,
                &o.to,
                o.note.clone(),
                c.clone(),
                None,
            );
            // A conditional deal discloses only once its predicate is met; the
            // promised fact is stored so the NPC can honour it later.
            self.effects_buf.push(Effect::Commit {
                commitment: cid,
                debtor: o.from.clone(),
                creditor: o.to.clone(),
            });
            if let Some(fact) = &o.disclose {
                if let Some(cm) = self.world.commitments.get_mut(&cid) {
                    cm.promised_fact = Some(fact.clone());
                }
            }
        } else if let Some(fact) = &o.disclose {
            // Unconditional offers disclose immediately on acceptance.
            let _ = social::share_claim(
                &mut self.world,
                &mut self.events,
                &mut self.rng,
                tick,
                minute,
                &o.from,
                &o.to,
                fact,
                true,
            );
            self.effects_buf.push(Effect::Assert {
                speaker: o.from.clone(),
                proposition: fact.clone(),
                asserted_polarity: true,
                believed: None,
            });
        }
        if let Some(of) = self.world.offers.get_mut(&offer) {
            of.state = OfferState::Accepted;
        }
        self.effects_buf.push(Effect::CreateEvent {
            summary: format!("offer #{} accepted", offer.raw()),
        });
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        self.events.push(
            tick,
            minute,
            Some(proposal.actor.clone()),
            EventKind::OfferAccepted {
                offer,
                by: proposal.actor.clone(),
            },
            Vec::new(),
            Vec::new(),
        );
        ValidationOutcome::Accepted
    }

    fn do_reject_offer(&mut self, proposal: &ActionProposal, offer: OfferId) -> ValidationOutcome {
        if let Some(of) = self.world.offers.get_mut(&offer) {
            of.state = OfferState::Rejected;
        }
        self.effects_buf.push(Effect::CreateEvent {
            summary: format!("offer #{} rejected", offer.raw()),
        });
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        self.events.push(
            tick,
            minute,
            Some(proposal.actor.clone()),
            EventKind::OfferRejected {
                offer,
                by: proposal.actor.clone(),
            },
            Vec::new(),
            Vec::new(),
        );
        ValidationOutcome::Accepted
    }

    fn do_promise(
        &mut self,
        proposal: &ActionProposal,
        target: ActorId,
        terms: String,
        predicate: QuestPredicate,
        deadline: Option<u64>,
    ) -> ValidationOutcome {
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        let id = social::create_commitment(
            &mut self.world,
            &mut self.events,
            tick,
            minute,
            CommitmentKind::Promise,
            &proposal.actor,
            &target,
            terms,
            predicate,
            deadline.map(|d| d as u32),
        );
        self.effects_buf.push(Effect::Commit {
            commitment: id,
            debtor: proposal.actor.clone(),
            creditor: target,
        });
        ValidationOutcome::Pending {
            offer: None,
            commitment: Some(id),
        }
    }

    fn do_break_promise(
        &mut self,
        proposal: &ActionProposal,
        commitment: CommitmentId,
    ) -> ValidationOutcome {
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        social::break_commitment(
            &mut self.world,
            &mut self.events,
            tick,
            minute,
            commitment,
            &proposal.actor,
        );
        self.effects_buf.push(Effect::ChangeTie {
            from: proposal.actor.clone(),
            to: proposal.actor.clone(),
            summary: format!("broke commitment #{}", commitment.raw()),
        });
        self.effects_buf.push(Effect::CreateEvent {
            summary: format!("commitment #{} broken", commitment.raw()),
        });
        ValidationOutcome::Accepted
    }

    // --- deal FSM handlers (behavior-engine Phase 2, plan §4) ----------

    /// Validate that a predicate only references actors/items/facts/
    /// locations/factions that exist. Grounding, not possibility: a
    /// well-typed predicate about a dead actor still validates here (the
    /// world predicate itself evaluates false).
    fn validate_predicate_refs(&self, pred: &QuestPredicate) -> Result<(), RejectReason> {
        use QuestPredicate as P;
        match pred {
            P::HasItem { actor, item, .. } => {
                if !self.world.actors.contains_key(actor) {
                    return Err(RejectReason::UnknownActor(actor.clone()));
                }
                if !self.world.items.contains_key(item) {
                    return Err(RejectReason::UnknownItem(item.clone()));
                }
                Ok(())
            }
            P::HasMoney { actor, .. } => {
                if !self.world.actors.contains_key(actor) {
                    return Err(RejectReason::UnknownActor(actor.clone()));
                }
                Ok(())
            }
            P::ActorAlive { actor }
            | P::ActorDead { actor }
            | P::ActorAt { actor, .. }
            | P::BelievesFact { actor, .. }
            | P::HeardFact { actor, .. } => {
                if !self.world.actors.contains_key(actor) {
                    return Err(RejectReason::UnknownActor(actor.clone()));
                }
                match pred {
                    QuestPredicate::ActorAt { location, .. } => {
                        if !self.world.locations.contains_key(location) {
                            return Err(RejectReason::UnknownLocation(location.clone()));
                        }
                    }
                    QuestPredicate::BelievesFact { fact, .. }
                    | QuestPredicate::HeardFact { fact, .. }
                        if !self.world.facts.contains_key(fact) =>
                    {
                        return Err(RejectReason::UnknownFact(fact.clone()));
                    }
                    _ => {}
                }
                Ok(())
            }
            P::FactCommitted { fact } => {
                if !self.world.facts.contains_key(fact) {
                    return Err(RejectReason::UnknownFact(fact.clone()));
                }
                Ok(())
            }
            P::LocationRepaired { location } => {
                if !self.world.locations.contains_key(location) {
                    return Err(RejectReason::UnknownLocation(location.clone()));
                }
                Ok(())
            }
            P::FactionPaid { faction, .. } => {
                if !self.world.factions.contains_key(faction) {
                    return Err(RejectReason::UnknownFaction(faction.clone()));
                }
                Ok(())
            }
            P::All(parts) | P::Any(parts) => {
                for p in parts {
                    self.validate_predicate_refs(p)?;
                }
                Ok(())
            }
            P::Not(inner) => self.validate_predicate_refs(inner),
            P::Flag(_) => Ok(()),
        }
    }

    /// Propose a typed conditional contract: the deal enters `Proposed`.
    /// Nothing moves yet; acceptance ≠ fulfillment.
    #[allow(clippy::too_many_arguments)]
    fn do_propose_deal(
        &mut self,
        proposal: &ActionProposal,
        target: ActorId,
        terms: String,
        condition: QuestPredicate,
        performance: QuestPredicate,
        condition_deadline_minute: Option<u64>,
        performance_deadline_minute: Option<u64>,
        give: Vec<Stack>,
        money: i64,
        note: String,
    ) -> ValidationOutcome {
        let id = DealId(self.world.next_deal_id);
        self.world.next_deal_id += 1;
        let deal = Deal::proposed(
            id,
            proposal.actor.clone(),
            target.clone(),
            note,
            condition,
            performance,
            condition_deadline_minute,
            performance_deadline_minute,
            self.config.deal_grace_minutes,
            give,
            money,
            Vec::new(),
            self.world.tick,
        );
        self.world.deals.insert(id, deal);
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        self.effects_buf.push(Effect::CreateEvent {
            summary: format!("deal #{} proposed", id.raw()),
        });
        self.events.push(
            tick,
            minute,
            Some(proposal.actor.clone()),
            EventKind::DealProposed {
                deal: id,
                debtor: proposal.actor.clone(),
                creditor: target,
                terms,
            },
            Vec::new(),
            Vec::new(),
        );
        ValidationOutcome::Pending {
            offer: None,
            commitment: None,
        }
    }

    /// Creditor accepts: give-on-acceptance consideration moves atomically
    /// at `Accepted`, before the FSM enters `Active` (§4.2). Validation
    /// already proved the debtor holds the whole bundle, so this handler
    /// cannot fail midway — the proposal is one atomic validate_then_apply.
    fn do_accept_deal(&mut self, proposal: &ActionProposal, deal_id: DealId) -> ValidationOutcome {
        let Some(deal) = self.world.deals.get(&deal_id).cloned() else {
            return ValidationOutcome::Rejected(RejectReason::UnknownDeal(deal_id));
        };
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        // Escrow: consideration moves atomically at Accepted (§4.2).
        for stack in &deal.give {
            self.move_item(&deal.debtor, &deal.creditor, &stack.item, stack.qty);
        }
        if deal.money > 0 {
            self.transfer_money(&deal.debtor, &deal.creditor, deal.money);
        }
        if let Some(d) = self.world.deals.get_mut(&deal_id) {
            let _ = d.advance(DealEvent::Accept, minute as u64);
            let _ = d.advance(DealEvent::Activate, minute as u64);
        }
        self.effects_buf.push(Effect::CreateEvent {
            summary: format!("deal #{} accepted and active", deal_id.raw()),
        });
        self.events.push(
            tick,
            minute,
            Some(proposal.actor.clone()),
            EventKind::DealAccepted {
                deal: deal_id,
                by: proposal.actor.clone(),
            },
            Vec::new(),
            Vec::new(),
        );
        self.events.push(
            tick,
            minute,
            Some(proposal.actor.clone()),
            EventKind::DealActivated { deal: deal_id },
            Vec::new(),
            Vec::new(),
        );
        ValidationOutcome::Accepted
    }

    /// Creditor rejects the proposal (terminal, exactly-once).
    fn do_reject_deal(&mut self, proposal: &ActionProposal, deal_id: DealId) -> ValidationOutcome {
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        if let Some(d) = self.world.deals.get_mut(&deal_id) {
            let _ = d.advance(DealEvent::Reject, minute as u64);
        }
        self.effects_buf.push(Effect::CreateEvent {
            summary: format!("deal #{} rejected", deal_id.raw()),
        });
        self.events.push(
            tick,
            minute,
            Some(proposal.actor.clone()),
            EventKind::DealRejected {
                deal: deal_id,
                by: proposal.actor.clone(),
            },
            Vec::new(),
            Vec::new(),
        );
        ValidationOutcome::Accepted
    }

    /// Cancel per the §4.4 authorization matrix. The requesting party is the
    /// caller; a unilateral debtor past ConditionMet is mapped by the FSM
    /// itself onto the Violated path.
    fn do_cancel_deal(&mut self, proposal: &ActionProposal, deal_id: DealId) -> ValidationOutcome {
        let Some(deal) = self.world.deals.get(&deal_id).cloned() else {
            return ValidationOutcome::Rejected(RejectReason::UnknownDeal(deal_id));
        };
        let by = if proposal.actor == deal.debtor {
            Party::Debtor
        } else if proposal.actor == deal.creditor {
            Party::Creditor
        } else {
            return ValidationOutcome::Rejected(RejectReason::Invalid(
                "only a deal party may cancel it".into(),
            ));
        };
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        let _ = tick;
        // §4.4: the debtor may cancel as `Canceled` only before ConditionMet
        // with no consideration exchanged (§4.2: it moved atomically at
        // `Accepted`); once exchanged, the kernel has no such arrow and the
        // attempt is rejected — neutral, nothing mutates, and the creditor's
        // option set (consent, authority, deadline lapse) is unchanged.
        let outcome = deal_fsm::transition(
            deal.state,
            DealEvent::Cancel { by },
            deal.consideration_exchanged(),
        );
        match outcome {
            Ok(rpg_domain::deal::DealState::Violated) => {
                let _ = self.apply_deal_event(
                    deal_id,
                    DealEvent::Violate {
                        reason: ViolationReason::DebtorReneged,
                    },
                    minute as u64,
                    Vec::new(),
                );
            }
            Ok(_) => {
                self.apply_deal_event(deal_id, DealEvent::Cancel { by }, minute as u64, Vec::new());
            }
            Err(_) => {
                // The FSM rejected the cancellation (e.g. terminal or the
                // transient ConditionMet state): rejection mutates nothing.
                return ValidationOutcome::Rejected(RejectReason::Invalid(format!(
                    "cancellation not permitted in state {}",
                    deal.state.name()
                )));
            }
        }
        ValidationOutcome::Accepted
    }

    /// Deliver a settlement bundle (escrow §4.2). All custody was validated
    /// before dispatch, so the transfers below either all happen (they were
    /// already validated) or the proposal was rejected outright. A short
    /// delivery records `DealPartialDelivery` and mutates nothing contractual
    /// (§4.3); a satisfied performance performs the deal exactly-once (§4.6).
    fn do_settle_deal(
        &mut self,
        proposal: &ActionProposal,
        deal_id: DealId,
        deliver: Vec<Stack>,
        money: i64,
    ) -> ValidationOutcome {
        let Some(deal) = self.world.deals.get(&deal_id).cloned() else {
            return ValidationOutcome::Rejected(RejectReason::UnknownDeal(deal_id));
        };
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        // The bundle: every component transfers now. Delivery evidence
        // (§4.3) records the *actual* transfers, not the requested
        // quantities: `move_item` reports what really moved, so a
        // shortfall can never be over-reported here.
        let mut got_by_item: BTreeMap<ItemId, u32> = BTreeMap::new();
        for stack in &deliver {
            let moved = self.move_item(&deal.debtor, &deal.creditor, &stack.item, stack.qty);
            let got = got_by_item.entry(stack.item.clone()).or_insert(0);
            *got = got.saturating_add(moved);
        }
        if money > 0 {
            self.transfer_money(&deal.debtor, &deal.creditor, money);
        }

        let performance = if deal.installments.is_empty() {
            deal.performance.clone()
        } else {
            // Ordered installments: the first unsettled one is next (§4.3).
            deal.installments
                .iter()
                .find(|i| !i.settled)
                .map(|i| i.performance.clone())
                .unwrap_or_else(|| deal.performance.clone())
        };
        let expected_qty = if deal.installments.is_empty() {
            deal_fsm::predicate_delivery(&performance).map(|(_, _, q)| q)
        } else {
            deal.installments
                .iter()
                .find(|i| !i.settled)
                .and_then(|i| (i.expected_qty > 0).then_some(i.expected_qty))
        };

        if evaluate_predicate(&self.world, &performance) {
            let settled_all = if deal.installments.is_empty() {
                true
            } else {
                // Settle the first unsettled installment exactly-once.
                let idx = deal
                    .installments
                    .iter()
                    .position(|i| !i.settled)
                    .expect("unsettled installment exists");
                let all = self
                    .world
                    .deals
                    .get_mut(&deal_id)
                    .and_then(|d| d.settle_installment(idx, tick).ok())
                    .unwrap_or(false);
                let total = deal.total_installments();
                let performed = self
                    .world
                    .deals
                    .get(&deal_id)
                    .map(|d| d.performed_installments())
                    .unwrap_or(0);
                self.effects_buf.push(Effect::CreateEvent {
                    summary: format!("deal #{} installment {} settled", deal_id.raw(), idx),
                });
                self.events.push(
                    tick,
                    minute,
                    Some(proposal.actor.clone()),
                    EventKind::DealInstallmentSettled {
                        deal: deal_id,
                        index: idx,
                        performed,
                        total,
                    },
                    Vec::new(),
                    Vec::new(),
                );
                all
            };
            if settled_all {
                self.perform_deal(&deal_id, &proposal.actor, tick, minute);
            }
            return ValidationOutcome::Accepted;
        }

        // Short delivery: the deal stays PerformanceDue, a typed event
        // records the shortfall as creditor evidence, nothing contractual
        // mutates (§4.3). The delivered goods are real transfers; the
        // creditor decides (renegotiate / excuse / let the deadline lapse).
        if let Some(expected) = expected_qty {
            let got = deal_fsm::predicate_item(&performance)
                .and_then(|item| got_by_item.get(&item).copied())
                .unwrap_or(0);
            self.effects_buf.push(Effect::CreateEvent {
                summary: format!(
                    "deal #{} partial delivery {}/{}",
                    deal_id.raw(),
                    got,
                    expected
                ),
            });
            self.events.push(
                tick,
                minute,
                Some(proposal.actor.clone()),
                EventKind::DealPartialDelivery {
                    deal: deal_id,
                    got,
                    expected,
                },
                Vec::new(),
                Vec::new(),
            );
        }
        ValidationOutcome::Accepted
    }

    /// Renegotiate a due deal: creditor consent (validated), single cycle,
    /// then the deal re-enters `Active` with the amended terms (§4.1).
    #[allow(clippy::too_many_arguments)]
    fn do_renegotiate_deal(
        &mut self,
        proposal: &ActionProposal,
        deal_id: DealId,
        new_condition: Option<QuestPredicate>,
        new_performance: Option<QuestPredicate>,
        new_condition_deadline_minute: Option<u64>,
        new_performance_deadline_minute: Option<u64>,
    ) -> ValidationOutcome {
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        if let Some(c) = &new_condition {
            if let Err(reason) = self.validate_predicate_refs(c) {
                return ValidationOutcome::Rejected(reason);
            }
        }
        if let Some(p) = &new_performance {
            if let Err(reason) = self.validate_predicate_refs(p) {
                return ValidationOutcome::Rejected(reason);
            }
        }
        if let Some(d) = self.world.deals.get_mut(&deal_id) {
            if let Some(c) = new_condition {
                d.condition = c;
            }
            if let Some(p) = new_performance {
                d.performance = p;
            }
            if new_condition_deadline_minute.is_some() {
                d.condition_deadline_minute = new_condition_deadline_minute;
            }
            if new_performance_deadline_minute.is_some() {
                d.performance_deadline_minute = new_performance_deadline_minute;
            }
            d.condition_overdue_emitted = false;
            d.performance_overdue_emitted = false;
            d.grace_until_minute = None;
            let _ = d.advance(DealEvent::Renegotiate, minute as u64);
            let _ = d.advance(DealEvent::Reactivate, minute as u64);
        }
        self.effects_buf.push(Effect::CreateEvent {
            summary: format!("deal #{} renegotiated", deal_id.raw()),
        });
        self.events.push(
            tick,
            minute,
            Some(proposal.actor.clone()),
            EventKind::DealRenegotiated {
                deal: deal_id,
                by: proposal.actor.clone(),
            },
            Vec::new(),
            Vec::new(),
        );
        self.events.push(
            tick,
            minute,
            Some(proposal.actor.clone()),
            EventKind::DealActivated { deal: deal_id },
            Vec::new(),
            Vec::new(),
        );
        ValidationOutcome::Accepted
    }

    /// Apply one deal lifecycle event through the pinned transition kernel
    /// and emit its typed event. `evidence` carries the authority's cited
    /// facts for an `Excuse` release (ignored by every other event). Returns
    /// the post-transition state, or `None` when the kernel rejected the
    /// event (exactly-once guard).
    fn apply_deal_event(
        &mut self,
        deal_id: DealId,
        event: DealEvent,
        minute: u64,
        evidence: Vec<FactId>,
    ) -> Option<rpg_domain::deal::DealState> {
        let (tick, m) = (self.world.tick, self.world.minute_of_day);
        let next = self
            .world
            .deals
            .get_mut(&deal_id)?
            .advance(event, minute)
            .ok()?;
        match event {
            DealEvent::Accept => {
                let by = self.world.deals.get(&deal_id)?.creditor.clone();
                self.effects_buf.push(Effect::CreateEvent {
                    summary: format!("deal #{} accepted", deal_id.raw()),
                });
                self.events.push(
                    tick,
                    m,
                    None,
                    EventKind::DealAccepted { deal: deal_id, by },
                    Vec::new(),
                    Vec::new(),
                );
            }
            DealEvent::Activate => {
                self.effects_buf.push(Effect::CreateEvent {
                    summary: format!("deal #{} active", deal_id.raw()),
                });
                self.events.push(
                    tick,
                    m,
                    None,
                    EventKind::DealActivated { deal: deal_id },
                    Vec::new(),
                    Vec::new(),
                );
            }
            DealEvent::ConditionMet => {
                self.effects_buf.push(Effect::CreateEvent {
                    summary: format!("deal #{} condition met", deal_id.raw()),
                });
                self.events.push(
                    tick,
                    m,
                    None,
                    EventKind::DealConditionMet { deal: deal_id },
                    Vec::new(),
                    Vec::new(),
                );
            }
            DealEvent::ObligationDue => {
                self.effects_buf.push(Effect::CreateEvent {
                    summary: format!("deal #{} performance due", deal_id.raw()),
                });
                self.events.push(
                    tick,
                    m,
                    None,
                    EventKind::DealPerformanceDue { deal: deal_id },
                    Vec::new(),
                    Vec::new(),
                );
            }
            DealEvent::Cancel { by } => {
                let who = match by {
                    Party::Debtor => self.world.deals.get(&deal_id)?.debtor.clone(),
                    Party::Creditor | Party::Mutual => {
                        self.world.deals.get(&deal_id)?.creditor.clone()
                    }
                };
                self.effects_buf.push(Effect::CreateEvent {
                    summary: format!("deal #{} canceled", deal_id.raw()),
                });
                self.events.push(
                    tick,
                    m,
                    None,
                    EventKind::DealCanceled {
                        deal: deal_id,
                        by: who,
                    },
                    Vec::new(),
                    Vec::new(),
                );
            }
            DealEvent::Violate { reason } => {
                // A breach is appraised by the creditor (mind-local; the tie
                // update is the social layer's directed edge write).
                let (debtor, creditor) = {
                    let d = self.world.deals.get(&deal_id)?;
                    (d.debtor.clone(), d.creditor.clone())
                };
                social::adjust_relationship(
                    &mut self.world,
                    &creditor,
                    &debtor,
                    &social::RelationshipDelta::new()
                        .trust(-0.3)
                        .anger(0.25)
                        .respect(-0.2),
                );
                self.effects_buf.push(Effect::ChangeTie {
                    from: creditor,
                    to: debtor,
                    summary: format!("deal #{} violated ({})", deal_id.raw(), reason.name()),
                });
                self.effects_buf.push(Effect::CreateEvent {
                    summary: format!("deal #{} violated", deal_id.raw()),
                });
                self.events.push(
                    tick,
                    m,
                    None,
                    EventKind::DealViolated {
                        deal: deal_id,
                        reason,
                    },
                    Vec::new(),
                    Vec::new(),
                );
            }
            DealEvent::Excuse => {
                // Authority/institution release (plan §4.4): the typed
                // terminal event records the cited evidence; the deal's
                // history stays intact for audit.
                self.effects_buf.push(Effect::CreateEvent {
                    summary: format!("deal #{} excused", deal_id.raw()),
                });
                self.events.push(
                    tick,
                    m,
                    None,
                    EventKind::DealExcused {
                        deal: deal_id,
                        evidence,
                    },
                    Vec::new(),
                    Vec::new(),
                );
            }
            _ => {}
        }
        Some(next)
    }

    /// Perform a due deal: the contracted benefit was delivered and
    /// validated. Exactly-once via the transition kernel's state guard.
    fn perform_deal(&mut self, deal_id: &DealId, by: &ActorId, tick: u64, minute: u32) {
        if let Some(d) = self.world.deals.get_mut(deal_id) {
            let _ = d.advance(DealEvent::Perform, minute as u64);
        }
        self.effects_buf.push(Effect::CreateEvent {
            summary: format!("deal #{} performed", deal_id.raw()),
        });
        self.events.push(
            tick,
            minute,
            Some(by.clone()),
            EventKind::DealPerformed {
                deal: *deal_id,
                by: by.clone(),
            },
            Vec::new(),
            Vec::new(),
        );
    }

    /// Move money between actors (contractual component of deals). The
    /// caller validates funds first, so this conserves the total exactly.
    fn transfer_money(&mut self, from: &ActorId, to: &ActorId, amount: i64) {
        if let Some(a) = self.world.actor_mut(from) {
            a.inventory.money -= amount;
        }
        if let Some(a) = self.world.actor_mut(to) {
            a.inventory.money += amount;
        }
        self.effects_buf.push(Effect::Transfer {
            from: from.clone(),
            to: to.clone(),
            item: None,
            qty: amount.max(0) as u32,
        });
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        self.events.push(
            tick,
            minute,
            Some(from.clone()),
            EventKind::MoneyPaid {
                from: from.clone(),
                to: to.clone(),
                amount,
            },
            Vec::new(),
            Vec::new(),
        );
    }

    /// Move up to `qty` of `item`; returns how many actually moved (the
    /// caller validates custody first, so this is `qty` in validated flows).
    /// Settlement evidence must record this *actual* count, not the
    /// requested one (§4.3: `DealPartialDelivery` is evidence of what was
    /// really delivered).
    fn move_item(&mut self, from: &ActorId, to: &ActorId, item: &ItemId, qty: u32) -> u32 {
        let removed = self
            .world
            .actor_mut(from)
            .map(|a| a.inventory.remove(item, qty))
            .unwrap_or(0);
        if removed > 0 {
            if let Some(a) = self.world.actor_mut(to) {
                a.inventory.add(item.clone(), removed);
            }
        }
        let value = self
            .world
            .items
            .get(item)
            .map(|i| i.base_value)
            .unwrap_or(1);
        let value_norm = (value as f32 / 100.0).clamp(0.05, 1.0);
        // TRANSFER with ledger integrity (the move above) plus the social
        // meaning of receiving a gift (CHANGE_TIE, mind-local appraisal).
        self.effects_buf.push(Effect::Transfer {
            from: from.clone(),
            to: to.clone(),
            item: Some(item.clone()),
            qty: removed,
        });
        self.effects_buf.push(Effect::ChangeTie {
            from: to.clone(),
            to: from.clone(),
            summary: format!("received {}x{}", removed, item.as_str()),
        });
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        let transfer_event = self.events.push(
            tick,
            minute,
            Some(from.clone()),
            EventKind::ItemTransferred {
                from: from.clone(),
                to: to.clone(),
                item: item.clone(),
                qty: removed,
            },
            Vec::new(),
            vec![to.clone()],
        );
        // The recipient's APPRAISE of the gift: gratitude or bribery, per
        // its own values (design §3.2 separation; plan I-4).
        if removed > 0 {
            self.effects_buf.push(Effect::Appraise {
                subject: to.clone(),
                regarding: from.clone(),
                summary: "received a gift".into(),
            });
            self.record_appraisal(
                to,
                from,
                AppraisedEvent::GiftReceived { value: value_norm },
                transfer_event,
            );
        }
        removed
    }

    fn do_transfer(
        &mut self,
        _proposal: &ActionProposal,
        target: ActorId,
        item: ItemId,
        qty: u32,
    ) -> ValidationOutcome {
        let from = _proposal.actor.clone();
        self.move_item(&from, &target, &item, qty);
        ValidationOutcome::Accepted
    }

    fn do_pay(
        &mut self,
        proposal: &ActionProposal,
        target: ActorId,
        amount: i64,
    ) -> ValidationOutcome {
        let from = proposal.actor.clone();
        if let Some(a) = self.world.actor_mut(&from) {
            a.inventory.money -= amount;
        }
        if let Some(a) = self.world.actor_mut(&target) {
            a.inventory.money += amount;
        }
        let norm = (amount as f32 / 100.0).clamp(0.05, 1.0);
        self.effects_buf.push(Effect::Transfer {
            from: from.clone(),
            to: target.clone(),
            item: None,
            qty: amount.max(0) as u32,
        });
        self.effects_buf.push(Effect::ChangeTie {
            from: target.clone(),
            to: from.clone(),
            summary: format!("received {amount} coins"),
        });
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        let payment_event = self.events.push(
            tick,
            minute,
            Some(from.clone()),
            EventKind::MoneyPaid {
                from: from.clone(),
                to: target.clone(),
                amount,
            },
            Vec::new(),
            Vec::new(),
        );
        // The recipient's APPRAISE of the payment (design §3.2; plan I-4): a
        // directed, contextual GiftReceived appraisal exactly once — no
        // universal aggregate deltas.
        self.effects_buf.push(Effect::Appraise {
            subject: target.clone(),
            regarding: from.clone(),
            summary: "received a payment".into(),
        });
        self.record_appraisal(
            &target,
            &from,
            AppraisedEvent::GiftReceived { value: norm },
            payment_event,
        );
        ValidationOutcome::Accepted
    }

    fn do_share_claim(
        &mut self,
        proposal: &ActionProposal,
        target: ActorId,
        fact: FactId,
        assertive: bool,
        deceptive: bool,
    ) -> ValidationOutcome {
        // The asserted polarity (design §3.1; plan I-5 G1): the truthful
        // path asserts the fact as stated; a deceptive assertion asserts the
        // polarity the speaker believes false (falling back to a plain
        // affirmation for a speaker who only *knows of* the proposition).
        // The speaker's belief is never consulted for truth — and world
        // truth is never touched.
        let asserted_polarity = if deceptive {
            let speaker_belief = self
                .world
                .actor(&proposal.actor)
                .and_then(|a| a.beliefs.get(&fact))
                .map(|b| b.believes_true);
            !speaker_belief.unwrap_or(true)
        } else {
            true
        };
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        let outcome = social::assert_claim(
            &mut self.world,
            &mut self.events,
            &mut self.rng,
            tick,
            minute,
            &proposal.actor,
            &target,
            &fact,
            assertive,
            asserted_polarity,
            deceptive,
        );
        // ASSERT: a claim record with provenance, never automatic belief.
        // The asserted polarity is what the recipient was told — not what
        // the speaker believes, and not the world's truth.
        self.effects_buf.push(Effect::Assert {
            speaker: proposal.actor.clone(),
            proposition: fact.clone(),
            asserted_polarity,
            believed: Some(outcome.believed),
        });
        self.effects_buf.push(Effect::ChangeTie {
            from: target.clone(),
            to: proposal.actor.clone(),
            summary: "was told a claim (familiarity)".into(),
        });
        ValidationOutcome::Accepted
    }

    /// Authoritative QUERY lowering (design §1.1 "no omniscient answer";
    /// plan I-5 G2): the question is dispatched as `Effect::Query`, the
    /// answer is derived from the target's own evidence ledger via
    /// `answer_query` — never from world truth — and both the question and
    /// the answer are event-sourced. When the target holds evidence, the
    /// answer is *delivered* as a truthful assertion of what they hold (the
    /// asker gains provenance-bound evidence via the shared acceptance
    /// mechanic); when they hold nothing, the typed answer is Unknown.
    fn do_query(
        &mut self,
        proposal: &ActionProposal,
        target: ActorId,
        about: FactId,
    ) -> ValidationOutcome {
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        self.effects_buf.push(Effect::Query {
            asker: proposal.actor.clone(),
            target: target.clone(),
            about: about.clone(),
        });
        // CREATE_EVENT first: the question is the canonical origin the
        // answer cites as provenance.
        let question_event = self.events.push(
            tick,
            minute,
            Some(proposal.actor.clone()),
            EventKind::QuestionAsked {
                from: proposal.actor.clone(),
                to: target.clone(),
                about: about.clone(),
            },
            Vec::new(),
            vec![target.clone()],
        );
        // The answer is the target's own epistemic state, never world truth.
        let answer = rpg_behavior::answer_query(&self.world, &target, &about);
        match &answer {
            rpg_behavior::QueryAnswer::Unknown => {
                // Knowing nothing is a real answer: the target cannot be
                // made to disclose or invent the world's truth.
                self.effects_buf.push(Effect::CreateEvent {
                    summary: format!("query about {} answered unknown", about.as_str()),
                });
            }
            rpg_behavior::QueryAnswer::Heard { polarities, .. } => {
                // The answerer asserts what they hold: the majority polarity
                // of their evidence (contradictions stay visible in the
                // typed answer event; the asker's own appraisal of it is
                // theirs to make).
                let trues = polarities.iter().filter(|p| **p).count();
                let asserted_polarity = trues * 2 >= polarities.len();
                let _ = rpg_behavior::assert_proposition(
                    &mut self.world,
                    &mut self.events,
                    &mut self.rng,
                    tick,
                    minute,
                    &target,
                    &proposal.actor,
                    &about,
                    asserted_polarity,
                    false,
                );
                // Being asked and answered builds familiarity, as any
                // delivered claim does.
                social::adjust_relationship(
                    &mut self.world,
                    &target,
                    &proposal.actor,
                    &social::RelationshipDelta::new().familiarity(0.05),
                );
                self.effects_buf.push(Effect::ChangeTie {
                    from: target.clone(),
                    to: proposal.actor.clone(),
                    summary: "answered a question (familiarity)".into(),
                });
            }
        }
        let (polarities, confidence, contradicted, unknown) = match answer {
            rpg_behavior::QueryAnswer::Unknown => (Vec::new(), 0.0, false, true),
            rpg_behavior::QueryAnswer::Heard {
                polarities,
                confidence,
                contradicted,
            } => (polarities, confidence, contradicted, false),
        };
        self.events.push(
            tick,
            minute,
            Some(target.clone()),
            EventKind::QueryAnswered {
                answerer: target.clone(),
                asker: proposal.actor.clone(),
                about: about.clone(),
                polarities,
                confidence,
                contradicted,
                unknown,
                origin: question_event,
            },
            vec![question_event],
            vec![proposal.actor.clone()],
        );
        ValidationOutcome::Accepted
    }

    /// Authoritative SANCTION lowering (design §1.1 "authority, process, cost
    /// and evidence"; plan I-5 G3): the institutional act is dispatched as
    /// `Effect::Sanction`, the process is the typed `SanctionIssued` event,
    /// the cost is the fine's real transfer, and the evidence prerequisites
    /// were validated (the authority must hold every cited fact). The
    /// subject's APPRAISE of being sanctioned routes through the directed
    /// contextual pathway (design §3.2; plan I-4) — no universal deltas.
    fn do_sanction(
        &mut self,
        proposal: &ActionProposal,
        target: ActorId,
        kind: SanctionKind,
        evidence: Vec<FactId>,
    ) -> ValidationOutcome {
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        self.effects_buf.push(Effect::Sanction {
            authority: proposal.actor.clone(),
            subject: target.clone(),
            action: kind.name().to_string(),
        });
        // PROCESS: the typed institutional record (the subject observes its
        // own sanction).
        let sanction_event = self.events.push(
            tick,
            minute,
            Some(proposal.actor.clone()),
            EventKind::SanctionIssued {
                authority: proposal.actor.clone(),
                subject: target.clone(),
                kind: kind.name().to_string(),
                evidence: evidence.clone(),
            },
            Vec::new(),
            vec![target.clone()],
        );
        // COST: a fine is a real transfer of the stated amount.
        if let SanctionKind::Fine { amount } = kind {
            self.transfer_money(&target, &proposal.actor, amount);
        }
        // CUSTODY: an arrest takes the subject out of play (world state,
        // recorded by the sanction event above).
        if matches!(kind, SanctionKind::Arrest) {
            if let Some(a) = self.world.actor_mut(&target) {
                a.status = rpg_domain::world::ActorStatus::Departed;
            }
        }
        // The subject's APPRAISE of the sanction (mind-local, contextual):
        // an institutional commendation reads as praise; a penalty as an
        // institutional affront, severity-scaled by what was taken.
        let severity = match kind {
            SanctionKind::Fine { amount } => (amount as f32 / 100.0).clamp(0.3, 1.0),
            SanctionKind::Arrest => 1.0,
            SanctionKind::Boycott => 0.7,
            SanctionKind::Praise => 0.0,
        };
        let appraisal = if matches!(kind, SanctionKind::Praise) {
            AppraisedEvent::PraiseReceived
        } else {
            AppraisedEvent::Sanctioned { severity }
        };
        self.effects_buf.push(Effect::Appraise {
            subject: target.clone(),
            regarding: proposal.actor.clone(),
            summary: format!("was {} by the authority", kind.name()),
        });
        self.record_appraisal(&target, &proposal.actor, appraisal, sanction_event);
        ValidationOutcome::Accepted
    }

    /// Authority-backed contract release (plan §4.4): from `PerformanceDue`,
    /// authority/institution evidence excuses the deal. The state check and
    /// the exactly-once guard live in the transition kernel; the authority
    /// and evidence prerequisites were validated.
    fn do_excuse_deal(
        &mut self,
        proposal: &ActionProposal,
        deal_id: DealId,
        evidence: Vec<FactId>,
    ) -> ValidationOutcome {
        let minute = self.world.minute_of_day as u64;
        let _ = proposal;
        match self.apply_deal_event(deal_id, DealEvent::Excuse, minute, evidence) {
            Some(_) => ValidationOutcome::Accepted,
            None => ValidationOutcome::Rejected(RejectReason::Invalid(format!(
                "deal #{} cannot be excused from its current state (exactly-once)",
                deal_id.raw()
            ))),
        }
    }

    fn do_spread_rumor(
        &mut self,
        proposal: &ActionProposal,
        text: String,
        about: Option<FactId>,
    ) -> ValidationOutcome {
        let truth = about
            .as_ref()
            .and_then(|f| self.world.fact(f))
            .map(|f| f.truth)
            .unwrap_or(false);
        let speaker_belief = about.as_ref().and_then(|f| {
            self.world
                .actor(&proposal.actor)
                .and_then(|a| a.beliefs.get(f))
        });
        let conf = speaker_belief.map(|b| b.confidence).unwrap_or(0.5);
        // The asserted polarity is the *speaker's own belief* — what they are
        // actually claiming — never the world's canonical truth. A speaker
        // with no belief at all passes the claim on as heard (polarity true).
        let asserted_polarity = speaker_belief.map(|b| b.believes_true).unwrap_or(true);
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        let rumor_proposition = about.clone().unwrap_or_else(|| FactId::new("<rumor>"));
        social::add_rumor(
            &mut self.world,
            &mut self.events,
            tick,
            minute,
            &proposal.actor,
            text,
            about,
            truth,
            conf,
        );
        // ASSERT (public) + the rumor record itself.
        self.effects_buf.push(Effect::Assert {
            speaker: proposal.actor.clone(),
            proposition: rumor_proposition,
            asserted_polarity,
            believed: None,
        });
        self.effects_buf.push(Effect::CreateEvent {
            summary: "rumor spread publicly".into(),
        });
        ValidationOutcome::Accepted
    }

    /// Deliberate investigation semantics (review finding: an autonomous NPC
    /// search must never silently make quest items unavailable).
    ///
    /// * A *player* search of a location is a recovery: physical items left
    ///   at the site are taken into the inventory (observable `ItemFound`).
    /// * An *NPC* search is a look-around, not a seizure: the site is marked
    ///   inspected and committed clue containers surface as beliefs (each an
    ///   observable `FactBelieved` event, the player-visible signal of the
    ///   search), but `location_items` stay where they are. Confiscating
    ///   key ingredients is an authored obstacle ("the apothecary may refuse
    ///   because the guard confiscated key ingredients",
    ///   `docs/design/emergent_quest_driven_rpg_design.md`), delivered
    ///   through authored changes — never an emergent side effect of
    ///   autonomous walking.
    fn do_investigate(
        &mut self,
        proposal: &ActionProposal,
        target: InvestigateTarget,
    ) -> ValidationOutcome {
        let tick = self.world.tick;
        let minute = self.world.minute_of_day;
        match target {
            InvestigateTarget::Location(loc) => {
                if let Some(a) = self.world.actor_mut(&proposal.actor) {
                    a.inspected.insert(loc.as_str().to_string());
                }
                // Recover any physical items left here — the player's channel
                // only; see the deliberate-semantics doc above.
                let is_player = self
                    .world
                    .actor(&proposal.actor)
                    .is_some_and(|a| a.kind == ActorKind::Player);
                let found = if is_player {
                    self.world.location_items.remove(&loc).unwrap_or_default()
                } else {
                    Vec::new()
                };
                for stack in found {
                    if let Some(a) = self.world.actor_mut(&proposal.actor) {
                        a.inventory.add(stack.item.clone(), stack.qty);
                    }
                    self.effects_buf.push(Effect::Transfer {
                        from: proposal.actor.clone(),
                        to: proposal.actor.clone(),
                        item: Some(stack.item.clone()),
                        qty: stack.qty,
                    });
                    self.events.push(
                        tick,
                        minute,
                        Some(proposal.actor.clone()),
                        EventKind::ItemFound {
                            actor: proposal.actor.clone(),
                            item: stack.item,
                            qty: stack.qty,
                            location: loc.clone(),
                        },
                        Vec::new(),
                        vec![proposal.actor.clone()],
                    );
                }
                // Reveal any clue containers physically present here.
                let facts_here: Vec<FactId> = self
                    .world
                    .story
                    .committed_containers
                    .values()
                    .filter(|c| c.location.as_ref() == Some(&loc))
                    .map(|c| c.fact.clone())
                    .collect();
                for fact in facts_here {
                    social::record_belief(
                        &mut self.world,
                        &mut self.events,
                        tick,
                        minute,
                        &proposal.actor,
                        &fact,
                        true,
                        0.95,
                        rpg_domain::world::Provenance::Read {
                            item: ItemId::new("investigated"),
                        },
                    );
                    // OBSERVE: the investigator perceived the clue container.
                    self.effects_buf.push(Effect::Observe {
                        observer: proposal.actor.clone(),
                        observed: format!("clue for {}", fact.as_str()),
                        channel: rpg_domain::behavior_types::AcquisitionChannel::Read,
                        origin_event: None,
                    });
                }
                self.effects_buf.push(Effect::CreateEvent {
                    summary: format!("investigated {}", loc.as_str()),
                });
            }
            InvestigateTarget::Actor(other) => {
                if let Some(a) = self.world.actor_mut(&proposal.actor) {
                    a.inspected.insert(other.as_str().to_string());
                }
                // Investigating a person surfaces facts they are an eligible
                // holder of *only if* they already believe them.
                let held: Vec<FactId> = self
                    .world
                    .actor(&other)
                    .map(|o| {
                        o.beliefs
                            .iter()
                            .filter(|(_, b)| b.believes_true)
                            .map(|(f, _)| f.clone())
                            .collect()
                    })
                    .unwrap_or_default();
                for fact in held {
                    let is_clue = self
                        .world
                        .story
                        .committed_containers
                        .values()
                        .any(|c| c.holder.as_ref() == Some(&other) && c.fact == fact);
                    if is_clue {
                        social::record_belief(
                            &mut self.world,
                            &mut self.events,
                            tick,
                            minute,
                            &proposal.actor,
                            &fact,
                            true,
                            0.9,
                            rpg_domain::world::Provenance::Told {
                                by: other.clone(),
                                event: None,
                            },
                        );
                        self.effects_buf.push(Effect::Observe {
                            observer: proposal.actor.clone(),
                            observed: format!("clue held by {}", other.as_str()),
                            channel: rpg_domain::behavior_types::AcquisitionChannel::Told,
                            origin_event: None,
                        });
                    }
                }
                self.effects_buf.push(Effect::CreateEvent {
                    summary: format!("investigated {}", other.as_str()),
                });
            }
            InvestigateTarget::Item(item) => {
                if let Some(a) = self.world.actor_mut(&proposal.actor) {
                    a.inspected.insert(item.as_str().to_string());
                }
                self.effects_buf.push(Effect::CreateEvent {
                    summary: format!("investigated item {}", item.as_str()),
                });
            }
        }
        ValidationOutcome::Accepted
    }

    fn do_report_crime(
        &mut self,
        proposal: &ActionProposal,
        authority: ActorId,
        evidence: Vec<FactId>,
    ) -> ValidationOutcome {
        // Guards gain respect for credible reports and suspicion of the accused
        // is *not* automatic; only the report itself is recorded.
        social::adjust_relationship(
            &mut self.world,
            &authority,
            &proposal.actor,
            &social::RelationshipDelta::new().trust(0.1).respect(0.05),
        );
        // The authority's directed tie toward the reporter moved — the social
        // mutation is typed (a self-report mutates nothing, so no effect).
        if authority != proposal.actor {
            self.effects_buf.push(Effect::ChangeTie {
                from: authority.clone(),
                to: proposal.actor.clone(),
                summary: "received a credible report (trust, respect)".into(),
            });
        }
        // Shared report mechanic (behavior-engine Phase 1): the authority
        // *receives* each reported fact as provenance-bound evidence — claim
        // separation means a report does not silently become belief.
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        for fact in &evidence {
            rpg_behavior::record_provenance_evidence(
                &mut self.world,
                &mut self.events,
                tick,
                minute,
                &authority,
                fact,
                true,
                &rpg_domain::world::Provenance::Told {
                    by: proposal.actor.clone(),
                    event: None,
                },
            );
        }
        // ASSERT (claim to the institution) + REQUEST (for action).
        self.effects_buf.push(Effect::Assert {
            speaker: proposal.actor.clone(),
            proposition: evidence
                .first()
                .cloned()
                .unwrap_or_else(|| FactId::new("<crime>")),
            asserted_polarity: true,
            believed: None,
        });
        self.effects_buf.push(Effect::Request {
            from: proposal.actor.clone(),
            to: authority.clone(),
            demand: "investigate the reported crime".into(),
        });
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        self.events.push(
            tick,
            minute,
            Some(proposal.actor.clone()),
            EventKind::CrimeReported {
                by: proposal.actor.clone(),
                authority,
                evidence,
            },
            Vec::new(),
            Vec::new(),
        );
        ValidationOutcome::Accepted
    }

    fn do_hide_evidence(
        &mut self,
        proposal: &ActionProposal,
        item: ItemId,
        location: LocationId,
    ) -> ValidationOutcome {
        if let Some(a) = self.world.actor_mut(&proposal.actor) {
            a.inventory.remove(&item, u32::MAX);
        }
        self.world
            .set_flag(format!("hidden:{}@{}", item.as_str(), location.as_str()));
        // TRANSFORM: the evidence left the actor's custody (consumed into
        // concealment); the concealment is recorded — and event-sourced.
        self.effects_buf.push(Effect::Transform {
            actor: proposal.actor.clone(),
            subject: item.clone(),
            consumed: 1,
            produced: 0,
            note: format!("hidden at {}", location.as_str()),
        });
        self.effects_buf.push(Effect::CreateEvent {
            summary: format!("evidence {} hidden at {}", item.as_str(), location.as_str()),
        });
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        self.events.push(
            tick,
            minute,
            Some(proposal.actor.clone()),
            EventKind::EvidenceHidden {
                actor: proposal.actor.clone(),
                item,
                location,
            },
            Vec::new(),
            Vec::new(),
        );
        ValidationOutcome::Accepted
    }

    fn do_allocate_budget(
        &mut self,
        proposal: &ActionProposal,
        faction: FactionId,
        amount: i64,
        purpose: String,
    ) -> ValidationOutcome {
        if let Some(f) = self.world.faction_mut(&faction) {
            f.budget -= amount;
        }
        *self
            .world
            .ledger
            .entry(format!("{faction}:{purpose}"))
            .or_insert(0) += amount;
        // Resolve any priority whose predicate is now satisfied.
        let resolved: Vec<usize> = self
            .world
            .faction(&faction)
            .map(|f| {
                f.priorities
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| {
                        !p.resolved
                            && p.predicate
                                .as_ref()
                                .map(|pred| evaluate_predicate(&self.world, pred))
                                .unwrap_or(false)
                    })
                    .map(|(i, _)| i)
                    .collect()
            })
            .unwrap_or_default();
        let resolved_count = resolved.len();
        if let Some(f) = self.world.faction_mut(&faction) {
            for i in resolved {
                if let Some(p) = f.priorities.get_mut(i) {
                    p.resolved = true;
                }
            }
        }
        // The faction budget is an institutional pool, not an actor's
        // custody: the honest lowering is TRANSFORM (consumption of the
        // faction's budget object), never a fabricated self-to-self
        // TRANSFER. The typed `BudgetAllocated` event carries the
        // institutional truth (faction, amount, purpose).
        self.effects_buf.push(Effect::Transform {
            actor: proposal.actor.clone(),
            subject: ItemId::new(faction.as_str()),
            consumed: amount.max(0) as u32,
            produced: 0,
            note: format!("{faction} budget allocated to {purpose}"),
        });
        if resolved_count > 0 {
            self.effects_buf.push(Effect::ChangeGoal {
                actor: proposal.actor.clone(),
                goal: format!("{faction}:{purpose}"),
                change: GoalChange::Completed,
            });
        }
        self.effects_buf.push(Effect::CreateEvent {
            summary: format!("{faction} allocated {amount} to {purpose}"),
        });
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        self.events.push(
            tick,
            minute,
            Some(proposal.actor.clone()),
            EventKind::BudgetAllocated {
                faction,
                amount,
                purpose: purpose.clone(),
                priority: purpose,
            },
            Vec::new(),
            Vec::new(),
        );
        ValidationOutcome::Accepted
    }

    fn do_repair(&mut self, proposal: &ActionProposal, location: LocationId) -> ValidationOutcome {
        let damage = self.world.damage_at(&location);
        let cost = (damage.clamp(0.0, 1.0) * 200.0) as i64;
        // Find a faction responsible for this location that can afford it.
        let mut payer: Option<FactionId> = None;
        for f in self.world.factions.values() {
            if f.responsible_for.contains(&location) && f.budget >= cost && f.staff > 0 {
                payer = Some(f.id.clone());
                break;
            }
        }
        let Some(fid) = payer else {
            return ValidationOutcome::Rejected(RejectReason::Invalid(
                "no responsible faction can afford the repair".into(),
            ));
        };
        if let Some(f) = self.world.faction_mut(&fid) {
            f.budget -= cost;
        }
        self.world.damage.insert(location.clone(), 0.0);
        // TRANSFORM (the location is restored) plus the honest lowering of
        // the institutional budget consumption: TRANSFORM of the paying
        // faction's budget object — never a fabricated self-to-self TRANSFER.
        // The typed `RepairCompleted` event carries the real payer and cost.
        self.effects_buf.push(Effect::Transform {
            actor: proposal.actor.clone(),
            subject: ItemId::new(location.as_str()),
            consumed: 0,
            produced: 1,
            note: "location repaired".into(),
        });
        self.effects_buf.push(Effect::Transform {
            actor: proposal.actor.clone(),
            subject: ItemId::new(fid.as_str()),
            consumed: cost.max(0) as u32,
            produced: 0,
            note: format!("{fid} budget spent on the repair"),
        });
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        self.events.push(
            tick,
            minute,
            Some(proposal.actor.clone()),
            EventKind::RepairCompleted {
                location,
                by: fid,
                cost,
            },
            Vec::new(),
            Vec::new(),
        );
        ValidationOutcome::Accepted
    }

    fn do_recruit(&mut self, proposal: &ActionProposal, target: ActorId) -> ValidationOutcome {
        let rel = self
            .world
            .actor(&target)
            .map(|t| t.relationship(&proposal.actor))
            .unwrap_or_default();
        if rel.warmth() < 0.35 {
            return ValidationOutcome::Rejected(RejectReason::TargetUnwilling);
        }
        if let Some(t) = self.world.actor_mut(&target) {
            t.flags.insert(format!("crew:{}", proposal.actor.as_str()));
        }
        if let Some(a) = self.world.actor_mut(&proposal.actor) {
            a.flags.insert(format!("crew_of:{}", target.as_str()));
        }
        // AUTHORIZE: a bounded role grant (crew membership), plus the social
        // meaning of joining (CHANGE_TIE).
        self.effects_buf.push(Effect::Authorize {
            grantor: proposal.actor.clone(),
            grantee: target.clone(),
            scope: format!("crew of {}", proposal.actor.as_str()),
        });
        self.effects_buf.push(Effect::ChangeTie {
            from: target.clone(),
            to: proposal.actor.clone(),
            summary: "joined the crew".into(),
        });
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        self.events.push(
            tick,
            minute,
            Some(proposal.actor.clone()),
            EventKind::Recruited {
                recruiter: proposal.actor.clone(),
                recruit: target,
            },
            Vec::new(),
            Vec::new(),
        );
        ValidationOutcome::Accepted
    }

    fn do_schedule_meeting(
        &mut self,
        proposal: &ActionProposal,
        target: ActorId,
        location: LocationId,
        at_minute: u32,
    ) -> ValidationOutcome {
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        self.effects_buf.push(Effect::Schedule {
            by: proposal.actor.clone(),
            with: target.clone(),
            at_minute,
        });
        self.events.push(
            tick,
            minute,
            Some(proposal.actor.clone()),
            EventKind::MeetingScheduled {
                with: proposal.actor.clone(),
                target,
                location,
                at_minute,
            },
            Vec::new(),
            Vec::new(),
        );
        ValidationOutcome::Accepted
    }

    fn do_threaten(
        &mut self,
        proposal: &ActionProposal,
        target: ActorId,
        demand: String,
    ) -> ValidationOutcome {
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        // CREATE_EVENT first: the threat is the canonical origin the
        // appraisal cites as provenance.
        let threat_event = self.events.push(
            tick,
            minute,
            Some(proposal.actor.clone()),
            EventKind::ThreatMade {
                from: proposal.actor.clone(),
                to: target.clone(),
                demand: demand.clone(),
            },
            Vec::new(),
            Vec::new(),
        );
        // REQUEST (demanded act) + the target's mind-local APPRAISE (fear and
        // anger, computed from *its* values/needs/current edge — no
        // universal delta) + the directed relationship update.
        self.effects_buf.push(Effect::Request {
            from: proposal.actor.clone(),
            to: target.clone(),
            demand: demand.clone(),
        });
        self.effects_buf.push(Effect::Appraise {
            subject: target.clone(),
            regarding: proposal.actor.clone(),
            summary: "was threatened".into(),
        });
        self.record_appraisal(
            &target,
            &proposal.actor,
            AppraisedEvent::ThreatReceived { severity: 1.0 },
            threat_event,
        );
        self.effects_buf.push(Effect::ChangeTie {
            from: target.clone(),
            to: proposal.actor.clone(),
            summary: "fear and anger rose".into(),
        });
        ValidationOutcome::Accepted
    }

    fn do_comfort(&mut self, proposal: &ActionProposal, target: ActorId) -> ValidationOutcome {
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        let comfort_event = self.events.push(
            tick,
            minute,
            Some(proposal.actor.clone()),
            EventKind::Comforted {
                from: proposal.actor.clone(),
                to: target.clone(),
            },
            Vec::new(),
            Vec::new(),
        );
        // APPRAISE (mind-local: the target's grief relief / warmth, computed
        // from its own profile) + CHANGE_TIE.
        self.effects_buf.push(Effect::Appraise {
            subject: target.clone(),
            regarding: proposal.actor.clone(),
            summary: "was comforted".into(),
        });
        self.record_appraisal(
            &target,
            &proposal.actor,
            AppraisedEvent::ComfortReceived { sincerity: 0.8 },
            comfort_event,
        );
        self.effects_buf.push(Effect::ChangeTie {
            from: target.clone(),
            to: proposal.actor.clone(),
            summary: "felt comforted".into(),
        });
        ValidationOutcome::Accepted
    }

    /// Apply the mind-local `APPRAISE` of one event through the directed
    /// contextual pathway (design §3.2; plan I-4/I-4b): the observer's value
    /// profile, needs-driven goal urgency and current edge condition the
    /// deltas (no universal deltas); the same call writes the observer's
    /// emotion state (separate mind-local state, T3.12) and records the
    /// appraisal provenance. Event-sourced as typed `AppraisalRecorded`.
    fn record_appraisal(
        &mut self,
        subject: &ActorId,
        regarding: &ActorId,
        event: AppraisedEvent,
        origin: EventId,
    ) {
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        if let Some(outcome) = social::apply_appraised_event(
            &mut self.world,
            subject,
            regarding,
            event,
            Some(origin),
            &self.config.emotions,
            &self.config.needs,
        ) {
            self.events.push(
                tick,
                minute,
                Some(subject.clone()),
                EventKind::AppraisalRecorded {
                    subject: subject.clone(),
                    regarding: regarding.clone(),
                    event: Some(origin),
                    interpretation: outcome.interpretation.clone(),
                },
                vec![origin],
                Vec::new(),
            );
            self.trace.push(format!(
                "appraise: {} {} {}",
                subject, outcome.interpretation, regarding
            ));
        }
    }

    fn do_place_clue(
        &mut self,
        _proposal: &ActionProposal,
        fact: FactId,
        kind: ContainerKind,
        location: Option<LocationId>,
        holder: Option<ActorId>,
    ) -> ValidationOutcome {
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        let clue_summary = format!("clue for {} placed as {:?}", fact.as_str(), kind);
        let decision = narrative::commit_change(
            &mut self.world,
            &mut self.events,
            tick,
            minute,
            NarrativeChange::PlaceClue {
                fact,
                kind,
                location,
                holder,
            },
        );
        match decision {
            Decision::Accepted => {
                self.effects_buf.push(Effect::CreateEvent {
                    summary: clue_summary,
                });
                ValidationOutcome::Accepted
            }
            Decision::Rejected(r) => ValidationOutcome::Rejected(RejectReason::Invalid(r)),
        }
    }

    // -----------------------------------------------------------------
    // Stepping, time, routines, background simulation
    // -----------------------------------------------------------------

    /// Run one full simulation step: apply proposals, run routines, move,
    /// then run background economy/ambient systems and story maintenance.
    pub fn step(&mut self, proposals: Vec<ActionProposal>) -> StepReport {
        if self.recording {
            self.ops.push(SimOp::Step(proposals.clone()));
        }
        self.step_impl(proposals)
    }

    fn step_impl(&mut self, proposals: Vec<ActionProposal>) -> StepReport {
        self.advance_clock();
        let mut report = StepReport::default();
        let mut acted: BTreeSet<ActorId> = BTreeSet::new();

        // Deterministic ordering regardless of caller order.
        let mut props = proposals;
        props.sort_by(|a, b| {
            a.actor
                .cmp(&b.actor)
                .then_with(|| a.action.kind_name().cmp(b.action.kind_name()))
        });

        for p in &props {
            match self.apply_impl(p) {
                ValidationOutcome::Accepted => report.accepted += 1,
                ValidationOutcome::Pending { .. } => report.pending += 1,
                ValidationOutcome::Rejected(_) => report.rejected += 1,
            }
            acted.insert(p.actor.clone());
        }

        if self.config.enable_director {
            self.director_tick(&mut report);
        }

        let mut routine = self.routine_proposals(&acted);
        // At-post gate: autonomy may replace a routine proposal only when the
        // actor is at its scheduled post (routine idles) or has no schedule
        // entry; it never hijacks schedule-driven moves. Pursuing actors are
        // absent from `routine` (the goal layer owns their turns), so their
        // re-proposed walks and arrival actions always apply. Fresh pursuits
        // are recorded only *after* the gate accepts them — a rejected
        // pursuit must never suppress the schedule it lost to.
        for p in self.autonomy_proposals(&acted) {
            let at_post = match routine.iter().find(|r| r.actor == p.proposal.actor) {
                Some(r) => matches!(r.action, Action::Idle),
                None => true,
            };
            if !at_post {
                continue;
            }
            if let Some(pursuit) = p.pursuit {
                self.commit_pursuit(p.proposal.actor.clone(), pursuit);
            }
            if let Some(pos) = routine.iter().position(|r| r.actor == p.proposal.actor) {
                routine[pos] = p.proposal;
            } else {
                routine.push(p.proposal);
            }
        }
        // Deal responses (behavior-engine Phase 2): an actor with a due
        // obligation answers it through the ranked options. The same at-post
        // gate applies: a due deal never hijacks a schedule move.
        for p in self.deal_response_proposals(&acted) {
            let at_post = match routine.iter().find(|r| r.actor == p.actor) {
                Some(r) => matches!(r.action, Action::Idle),
                None => true,
            };
            if !at_post {
                continue;
            }
            if let Some(pos) = routine.iter().position(|r| r.actor == p.actor) {
                routine[pos] = p;
            } else {
                routine.push(p);
            }
        }
        for p in &routine {
            let _ = self.apply_impl(p);
        }

        self.move_actors();
        self.finalize_private();
        if self.config.enable_background {
            self.faction_tick();
            self.ambient_incidents(&mut report);
        }
        self.maintain(&mut report);
        self.mind_and_social_maintenance(&mut report);
        report
    }

    /// Mind and social maintenance (behavior-engine Phase 3; plan I-4a,
    /// I-4b and G4): the deterministic end-of-tick pass over needs, emotions,
    /// relationship and appraisal-memory maintenance.
    ///
    /// **Background mutation classification (plan G4, metric M1):** regular
    /// needs/emotions/relationship decay and memory-salience decay are
    /// *declared non-evented background mutations* — they mutate canonical
    /// actor state in this pipeline without a per-tick event; this function
    /// is the single declared site for that classification. Threshold
    /// crossings (need urgency latch) and the mapped goal-urgency raise they
    /// trigger ARE event-sourced as typed `NeedThresholdCrossed`,
    /// `GoalUrgencyRaised` / `GoalUrgencyReleased` events. Everything here is
    /// a pure function of canonical state in deterministic actor-id order,
    /// so replay reproduces it exactly.
    fn mind_and_social_maintenance(&mut self, report: &mut StepReport) {
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        // Pass 1 — decay (classified non-evented) + crossing detection, in
        // deterministic actor-id order.
        let mut crossings: Vec<(ActorId, Vec<NeedCrossing>)> = Vec::new();
        for (id, actor) in self.world.actors.iter_mut() {
            if !actor.is_alive() {
                continue;
            }
            let crossed = actor.needs.decay(&self.config.needs);
            if !crossed.is_empty() {
                crossings.push((id.clone(), crossed));
            }
            actor.emotions.decay(&self.config.emotions);
            // Threshold-gated response classes (design §11 Phase 3): the
            // gated classes are reported, not hardcoded reactions — the
            // I-6 planner (Phase 4) consumes them.
            for r in actor.emotions.responses(&self.config.emotions) {
                report.gated_responses.push((id.clone(), r));
            }
        }
        social::decay_relationships(
            &mut self.world,
            &self.config.relationship_decay,
            self.config.rel_memory_salience_decay,
        );
        // Pass 2 — event-sourced threshold crossings and the bounded,
        // hysteresis-guarded goal-urgency raise they map onto.
        for (actor_id, crossed) in crossings {
            for c in crossed {
                self.events.push(
                    tick,
                    minute,
                    Some(actor_id.clone()),
                    EventKind::NeedThresholdCrossed {
                        actor: actor_id.clone(),
                        need: c.kind,
                        direction: c.direction,
                        value: c.value,
                    },
                    Vec::new(),
                    Vec::new(),
                );
                match c.direction {
                    ThresholdDirection::FellBelow => {
                        // Raise the mapped active goals' urgency by the
                        // bounded mapped amount (T3.8); recorded so the
                        // release can subtract exactly what was added.
                        let urgency = self
                            .world
                            .actor(&actor_id)
                            .map(|a| a.needs.urgency(c.kind, &self.config.needs))
                            .unwrap_or(0.0);
                        if urgency <= 0.0 {
                            continue;
                        }
                        let need = c.kind;
                        let Some(actor) = self.world.actor_mut(&actor_id) else {
                            continue;
                        };
                        for goal in actor.goals.iter_mut() {
                            if goal.status != rpg_domain::world::GoalStatus::Active
                                || goal_kind_need(goal.kind) != Some(need)
                            {
                                continue;
                            }
                            goal.need_boost = urgency;
                            goal.priority += urgency;
                            self.events.push(
                                tick,
                                minute,
                                Some(actor_id.clone()),
                                EventKind::GoalUrgencyRaised {
                                    actor: actor_id.clone(),
                                    goal: goal.id,
                                    need,
                                    urgency,
                                },
                                Vec::new(),
                                Vec::new(),
                            );
                        }
                    }
                    ThresholdDirection::Recovered => {
                        // Release: remove exactly the boost applied at the
                        // crossing (hysteresis guard, T3.8).
                        let need = c.kind;
                        let Some(actor) = self.world.actor_mut(&actor_id) else {
                            continue;
                        };
                        for goal in actor.goals.iter_mut() {
                            if goal_kind_need(goal.kind) != Some(need) || goal.need_boost <= 0.0 {
                                continue;
                            }
                            goal.priority -= goal.need_boost;
                            goal.need_boost = 0.0;
                            self.events.push(
                                tick,
                                minute,
                                Some(actor_id.clone()),
                                EventKind::GoalUrgencyReleased {
                                    actor: actor_id.clone(),
                                    goal: goal.id,
                                    need,
                                },
                                Vec::new(),
                                Vec::new(),
                            );
                        }
                    }
                }
            }
        }
    }

    /// Production deal responses (behavior-engine Phase 2): an actor with a
    /// due obligation answers it through the ranked options.
    ///
    /// * **Fast path** (T2.1): an adopted intention continues without
    ///   re-deliberation while it stays valid (its satisfaction predicate —
    ///   the deal's performance predicate — has not begun to hold and its
    ///   abandonment criteria are unmet) *and* is still the ranked best.
    /// * **Re-plan**: otherwise the options are re-ranked with the switch
    ///   penalty against the standing intention (η_i·switch_i deters
    ///   oscillation) and the best acceptable option is adopted. Rejected
    ///   attempts count toward the abandonment criteria; an intention whose
    ///   configured failure threshold is reached is abandoned and is not
    ///   re-adopted.
    ///
    /// Deterministic: the ranking is a pure function of world state; no
    /// sampling in the pipeline (the seeded softmax sampler remains available
    /// to callers that track an RNG stream).
    pub fn deal_response_proposals(
        &mut self,
        already_acted: &BTreeSet<ActorId>,
    ) -> Vec<ActionProposal> {
        let mut out = Vec::new();
        if self.world.deals.is_empty() {
            return out;
        }
        // Retire intentions that are satisfied, even when no deal is due
        // anymore (an intention is not an obligation). *Abandoned* intentions
        // are deliberately kept: the configured failure threshold has
        // declared the goal given up, and keeping the abandoned intention in
        // place is what stops the pipeline from re-adopting the same
        // abandoned plan on a later tick (T2.1).
        let stale: Vec<ActorId> = self
            .deal_intentions
            .iter()
            .filter(|(id, intent)| {
                matches!(
                    rpg_behavior::consider_replan(intent, &self.world),
                    rpg_behavior::Replan::Replan(rpg_behavior::ReplanReason::GoalSatisfied)
                ) || !self
                    .world
                    .actor(id)
                    .is_some_and(|a| a.is_alive() && a.kind != ActorKind::Player)
            })
            .map(|(id, _)| id.clone())
            .collect();
        for id in stale {
            self.deal_intentions.remove(&id);
        }
        let due_debtors: Vec<ActorId> = {
            let mut ids: BTreeSet<ActorId> = self
                .world
                .deals
                .values()
                .filter(|d| d.state == rpg_domain::deal::DealState::PerformanceDue)
                .map(|d| d.debtor.clone())
                .collect();
            ids.retain(|id| {
                self.world.actor(id).is_some_and(|a| {
                    a.is_alive()
                        && a.kind != ActorKind::Player
                        && !already_acted.contains(&a.id)
                        && !self.paths.contains_key(&a.id)
                        && a.in_private_conversation_with.is_none()
                        && !self
                            .pending_private
                            .iter()
                            .any(|p| p.a == a.id || p.b == a.id)
                        && !self.autonomy.is_pursuing(&a.id)
                })
            });
            ids.into_iter().collect()
        };
        for npc in due_debtors {
            let tick = self.world.tick;
            // Fast path: the adopted intention continues.
            if let Some(intent) = self.deal_intentions.get(&npc).cloned() {
                match rpg_behavior::consider_replan(&intent, &self.world) {
                    rpg_behavior::Replan::Continue => {
                        let options = rpg_agents::deal_response_options(
                            &self.world,
                            &npc,
                            Some(&intent),
                            Some(self.config.talk_radius),
                            &self.config.needs,
                        );
                        if let Some(options) = options {
                            if options
                                .ranking
                                .ranked
                                .first()
                                .is_some_and(|b| b.id == intent.option)
                            {
                                if let Some(action) = rpg_agents::deal_response_action(
                                    &self.world,
                                    &options,
                                    &intent.option,
                                ) {
                                    out.push(ActionProposal {
                                        actor: npc,
                                        action,
                                        source: ProposalSource::NpcPolicy,
                                        rationale: format!(
                                            "deal intention: {} (fast path)",
                                            intent.option
                                        ),
                                    });
                                    continue;
                                }
                            }
                        }
                    }
                    rpg_behavior::Replan::Replan(rpg_behavior::ReplanReason::GoalSatisfied) => {
                        // Invalidated (satisfied): clear it and re-deliberate
                        // below.
                        self.deal_intentions.remove(&npc);
                    }
                    rpg_behavior::Replan::Replan(rpg_behavior::ReplanReason::Abandoned) => {
                        // The configured failure threshold declared the goal
                        // abandoned: the actor gives it up. No proposal is
                        // generated and the abandoned intention is *not*
                        // re-adopted — re-deliberating would deterministically
                        // re-choose the same option and loop the failure
                        // counter forever. The deadline path (§4.5) still
                        // owns the contract's breach resolution.
                        continue;
                    }
                }
            }
            // Re-plan: rank and adopt the best acceptable option.
            let previous = self.deal_intentions.get(&npc).cloned();
            let Some(options) = rpg_agents::deal_response_options(
                &self.world,
                &npc,
                previous.as_ref(),
                Some(self.config.talk_radius),
                &self.config.needs,
            ) else {
                continue;
            };
            let Some(best) = options.ranking.ranked.first() else {
                continue;
            };
            let Some(action) = rpg_agents::deal_response_action(&self.world, &options, &best.id)
            else {
                continue;
            };
            let satisfaction = self
                .world
                .deals
                .get(&options.deal)
                .map(|d| d.performance.clone());
            let intention = rpg_behavior::Intention::adopt(
                format!("respond to deal #{}", options.deal.raw()),
                best.id.clone(),
                satisfaction,
                rpg_domain::behavior_types::AbandonmentCriteria {
                    after_failed_attempts: Some(3),
                    predicate: None,
                },
                tick,
                previous.as_ref(),
            );
            self.deal_intentions.insert(npc.clone(), intention);
            out.push(ActionProposal {
                actor: npc,
                action,
                source: ProposalSource::NpcPolicy,
                rationale: format!("deal #{}: {}", options.deal.raw(), best.id),
            });
        }
        out
    }

    /// Record a failed attempt on the actor's adopted deal intention (a
    /// rejection of its proposed action), feeding the abandonment criteria.
    pub fn record_deal_intention_failure(&mut self, actor: &ActorId) {
        if let Some(intent) = self.deal_intentions.get_mut(actor) {
            intent.record_failure();
        }
    }

    fn advance_clock(&mut self) {
        self.world.tick += 1;
        self.world.minute_of_day += self.config.minutes_per_tick;
        while self.world.minute_of_day >= 1440 {
            self.world.minute_of_day -= 1440;
            self.world.day += 1;
        }
    }

    /// Deterministic routine proposals for actors that did not act this step.
    pub fn routine_proposals(&self, already_acted: &BTreeSet<ActorId>) -> Vec<ActionProposal> {
        let mut out = Vec::new();
        let minute = self.world.minute_of_day;
        for actor in self.world.actors.values() {
            if !actor.is_alive()
                || actor.kind == ActorKind::Player
                || already_acted.contains(&actor.id)
                || self.paths.contains_key(&actor.id)
                || actor.in_private_conversation_with.is_some()
                || self
                    .pending_private
                    .iter()
                    .any(|p| p.a == actor.id || p.b == actor.id)
            {
                continue;
            }
            // Pursuing actors: the goal layer owns their movement while a
            // pursuit is active; do not fight it with schedule moves.
            if self.autonomy.is_pursuing(&actor.id) {
                continue;
            }
            let scheduled = actor
                .schedule
                .iter()
                .find(|e| minute >= e.start_min && minute < e.end_min);
            if let Some(entry) = scheduled {
                if actor.location != entry.location {
                    out.push(ActionProposal {
                        actor: actor.id.clone(),
                        action: Action::MoveTo {
                            location: entry.location.clone(),
                        },
                        source: ProposalSource::NpcPolicy,
                        rationale: format!("schedule: {}", entry.activity),
                    });
                    continue;
                }
            }
            out.push(ActionProposal {
                actor: actor.id.clone(),
                action: Action::Idle,
                source: ProposalSource::NpcPolicy,
                rationale: "routine".into(),
            });
        }
        out
    }

    pub fn move_actors(&mut self) {
        let dt = self.config.move_units_per_tick;
        let ids: Vec<ActorId> = self.paths.keys().cloned().collect();
        for id in ids {
            let Some(mut path) = self.paths.remove(&id) else {
                continue;
            };
            let (speed, mut pos, mut facing, fallback_loc) = match self.world.actor(&id) {
                Some(a) => (
                    a.capabilities.speed.max(0.1),
                    a.pos,
                    a.facing,
                    a.location.clone(),
                ),
                None => continue,
            };
            let mut budget = dt * speed;
            let old_pos = pos;
            while budget > 0.0 {
                let Some(next) = path.first().copied() else {
                    break;
                };
                let d = pos.distance(next);
                if d <= budget {
                    facing = pos.direction_to(next).y.atan2(pos.direction_to(next).x);
                    pos = next;
                    path.remove(0);
                    budget -= d;
                } else {
                    let dir = pos.direction_to(next);
                    facing = dir.y.atan2(dir.x);
                    pos = Vec2::new(pos.x + dir.x * budget, pos.y + dir.y * budget);
                    budget = 0.0;
                }
            }
            let new_location = self.world.location_at(pos).cloned().unwrap_or(fallback_loc);
            let moved = old_pos.distance(pos) > 0.001;
            if let Some(a) = self.world.actor_mut(&id) {
                a.pos = pos;
                a.facing = facing;
                a.location = new_location.clone();
            }
            if !path.is_empty() {
                self.paths.insert(id.clone(), path);
            }
            if moved {
                let (tick, minute) = (self.world.tick, self.world.minute_of_day);
                self.events.push(
                    tick,
                    minute,
                    Some(id.clone()),
                    EventKind::ActorMoved {
                        actor: id,
                        from: old_pos,
                        to: pos,
                        location: new_location,
                    },
                    Vec::new(),
                    Vec::new(),
                );
            }
        }
    }

    /// Faction budgets: pay for the highest-value affordable priority. This is
    /// where "cost of inaction" shows up — unresolved damage accumulates and
    /// priorities compete for a finite budget.
    pub fn faction_tick(&mut self) {
        let faction_ids: Vec<FactionId> = self.world.factions.keys().cloned().collect();
        for fid in faction_ids {
            // Unresolved damage to responsible locations accumulates.
            let damaged: Vec<LocationId> = self
                .world
                .faction(&fid)
                .map(|f| {
                    f.responsible_for
                        .iter()
                        .filter(|l| self.world.damage_at(l) > 0.0)
                        .cloned()
                        .collect()
                })
                .unwrap_or_default();
            for loc in &damaged {
                // Damage slowly worsens while ignored.
                let cur = self.world.damage_at(loc);
                self.world.damage.insert(loc.clone(), (cur + 0.01).min(1.0));
            }
            let Some(idx) = self
                .world
                .faction(&fid)
                .and_then(|f| f.best_affordable_priority())
            else {
                continue;
            };
            let (cost, desc) = {
                let f = self.world.faction(&fid).unwrap();
                let p = &f.priorities[idx];
                (p.cost, p.description.clone())
            };
            if let Some(f) = self.world.faction_mut(&fid) {
                f.budget -= cost;
                if let Some(p) = f.priorities.get_mut(idx) {
                    p.resolved = true;
                }
            }
            *self
                .world
                .ledger
                .entry(format!("{fid}:{desc}"))
                .or_insert(0) += cost;
            self.trace.push(format!("{fid} funded '{desc}' for {cost}"));
        }
    }

    fn ambient_incidents(&mut self, report: &mut StepReport) {
        let now = self.world.day as u64 * 1440 + self.world.minute_of_day as u64;
        if now.saturating_sub(self.last_incident_minute)
            < self.config.incident_interval_minutes as u64
        {
            return;
        }
        self.last_incident_minute = now;
        // A small deterministic chance of a localized incident.
        if !self.rng.chance(0.4) {
            return;
        }
        let locs: Vec<LocationId> = self.world.locations.keys().cloned().collect();
        if locs.is_empty() {
            return;
        }
        let loc = locs[self.rng.pick_index(locs.len())].clone();
        let severity = 0.05 + 0.1 * self.rng.next_f32();
        let cur = self.world.damage_at(&loc);
        self.world
            .damage
            .insert(loc.clone(), (cur + severity).min(1.0));
        let desc = "an unattended accident caused damage".to_string();
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        self.events.push(
            tick,
            minute,
            None,
            EventKind::BackgroundIncident {
                location: loc.clone(),
                description: desc.clone(),
                severity,
            },
            Vec::new(),
            Vec::new(),
        );
        report.background_incidents.push(desc);
    }

    /// Story maintenance: satisfy commitments, activate checkpoints, place
    /// clues, and advance deals. Runs after every step when the director is
    /// enabled.
    fn maintain(&mut self, report: &mut StepReport) {
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        report
            .satisfied_commitments
            .extend(social::satisfy_commitments(
                &mut self.world,
                &mut self.events,
                tick,
                minute,
            ));
        self.advance_deals();
    }

    /// Deal pipeline (plan §4.5: deadline processing is part of the
    /// deterministic step pipeline, never an out-of-band mutation).
    ///
    /// Per tick, per deal, in id order:
    /// * `Active`: the condition predicate holding moves the deal
    ///   ConditionMet and — in the same deterministic pass — makes the
    ///   obligation due (design §7 step 5: separate events, separate state).
    /// * `PerformanceDue`: the performance predicate holding performs the
    ///   deal exactly-once (validated world predicate or settled transfer
    ///   evidence; a fulfillment *claim* alone never moves the FSM).
    /// * Deadlines: crossing emits exactly one typed `ObligationDue` event
    ///   (idempotent by state check) and starts the grace policy; an elapsed
    ///   grace window violates the deal with a typed `DealViolated` event.
    fn advance_deals(&mut self) {
        let now = self.world.day as u64 * 1440 + self.world.minute_of_day as u64;
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        let ids: Vec<DealId> = self.world.deals.keys().copied().collect();
        for id in ids {
            // At most one full Active → ConditionMet → PerformanceDue →
            // Performed chain per deal per tick.
            for _ in 0..4 {
                let Some(d) = self.world.deals.get(&id).cloned() else {
                    break;
                };
                if d.is_terminal() {
                    break;
                }
                match d.state {
                    rpg_domain::deal::DealState::Active => {
                        if evaluate_predicate(&self.world, &d.condition) {
                            if self
                                .apply_deal_event(id, DealEvent::ConditionMet, now, Vec::new())
                                .is_none()
                            {
                                break;
                            }
                            continue;
                        }
                        if d.condition_deadline_minute
                            .is_some_and(|deadline| now >= deadline)
                        {
                            let crossed = self
                                .world
                                .deals
                                .get_mut(&id)
                                .map(|x| x.cross_deadline(false, now))
                                .unwrap_or(false);
                            if crossed {
                                self.events.push(
                                    tick,
                                    minute,
                                    None,
                                    EventKind::ObligationDue { deal: id },
                                    Vec::new(),
                                    Vec::new(),
                                );
                            }
                        }
                        if d.condition_overdue_emitted && d.grace_elapsed(now) {
                            self.apply_deal_event(
                                id,
                                DealEvent::Violate {
                                    reason: ViolationReason::DeadlineLapsed,
                                },
                                now,
                                Vec::new(),
                            );
                        }
                        break;
                    }
                    rpg_domain::deal::DealState::ConditionMet => {
                        // The obligation becomes due: the trigger predicate
                        // held, the benefit is still pending.
                        self.apply_deal_event(id, DealEvent::ObligationDue, now, Vec::new());
                    }
                    rpg_domain::deal::DealState::PerformanceDue => {
                        if evaluate_predicate(&self.world, &d.performance) {
                            self.perform_deal(&id, &d.debtor, tick, minute);
                            break;
                        }
                        if d.performance_deadline_minute
                            .is_some_and(|deadline| now >= deadline)
                        {
                            let crossed = self
                                .world
                                .deals
                                .get_mut(&id)
                                .map(|x| x.cross_deadline(true, now))
                                .unwrap_or(false);
                            if crossed {
                                self.events.push(
                                    tick,
                                    minute,
                                    None,
                                    EventKind::ObligationDue { deal: id },
                                    Vec::new(),
                                    Vec::new(),
                                );
                            }
                        }
                        if d.performance_overdue_emitted && d.grace_elapsed(now) {
                            self.apply_deal_event(
                                id,
                                DealEvent::Violate {
                                    reason: ViolationReason::DeadlineLapsed,
                                },
                                now,
                                Vec::new(),
                            );
                        }
                        break;
                    }
                    _ => break,
                }
            }
        }
    }

    fn director_tick(&mut self, report: &mut StepReport) {
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        let activated =
            narrative::activate_ready_checkpoints(&mut self.world, &mut self.events, tick, minute);
        report.activated_checkpoints.extend(activated);
        if let Some(fact) = narrative::place_next_clue(
            &mut self.world,
            &mut self.events,
            &mut self.rng,
            tick,
            minute,
        ) {
            report.placed_clues.push(fact);
        }
    }

    /// Advance many ticks with no external proposals.
    pub fn run_ticks(&mut self, n: u32) -> StepReport {
        let mut total = StepReport::default();
        for _ in 0..n {
            let r = self.step(Vec::new());
            total.accepted += r.accepted;
            total.pending += r.pending;
            total.rejected += r.rejected;
            total.satisfied_commitments.extend(r.satisfied_commitments);
            total.activated_checkpoints.extend(r.activated_checkpoints);
            total.placed_clues.extend(r.placed_clues);
            total.background_incidents.extend(r.background_incidents);
            total.director_notes.extend(r.director_notes);
        }
        total
    }

    /// Convenience: the player actor id.
    pub fn player_id() -> ActorId {
        ActorId::new(PLAYER)
    }
}

/// Facts an action is epistemically about (used by causal traces).
fn action_facts(action: &Action) -> Vec<FactId> {
    match action {
        Action::ShareClaim { fact, .. } | Action::WarnContact { fact, .. } => vec![fact.clone()],
        Action::Query { about, .. } => vec![about.clone()],
        Action::Sanction { evidence, .. } | Action::ExcuseDeal { evidence, .. } => evidence.clone(),
        Action::SpreadRumor { about, .. } => about.clone().into_iter().collect(),
        Action::ReportCrime { evidence, .. } => evidence.clone(),
        Action::MakeOffer { disclose, .. } => disclose.clone().into_iter().collect(),
        Action::PlaceClue { fact, .. } => vec![fact.clone()],
        _ => Vec::new(),
    }
}
