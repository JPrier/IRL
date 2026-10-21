//! # rpg_sim
//!
//! The authoritative simulation. It owns the canonical [`World`], the
//! append-only [`EventLog`], the deterministic RNG, and the perception engine.
//!
//! **Nothing else may mutate canonical state.** Every change flows through
//! [`Simulation::apply`], which validates a typed [`ActionProposal`] and emits
//! typed events. Model output is always an untrusted proposal.

pub mod pathfind;

use rpg_domain::action::{
    Action, ActionProposal, InvestigateTarget, Loudness, ProposalSource, QuestPredicate,
    RejectReason, ValidationOutcome,
};
use rpg_domain::events::{EventKind, EventLog};
use rpg_domain::geom::Vec2;
use rpg_domain::ids::*;
use rpg_domain::rng::DeterministicRng;
use rpg_domain::social::{CommitmentKind, Offer, OfferState};
use rpg_domain::world::{
    evaluate_predicate, Actor, ActorKind, ContainerKind, LocationKind, Sensitivity, World,
};
use rpg_narrative::{self as narrative, Decision, NarrativeChange};
use rpg_perception::PerceptionEngine;
use rpg_social as social;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

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
    /// Minimum minutes between ambient incident rolls.
    pub incident_interval_minutes: u32,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            minutes_per_tick: 1,
            talk_radius: 3.5,
            move_units_per_tick: 2.0,
            enable_background: true,
            enable_director: true,
            incident_interval_minutes: 60,
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
}

/// A recorded public entry point, used for input/proposal replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SimOp {
    /// A direct `apply` call.
    Apply(ActionProposal),
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
    /// When false, `apply`/`step` do not append to `ops` (used during replay).
    pub recording: bool,
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
            recording: true,
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
            recording: true,
        }
    }

    /// Transient state that must be persisted for exact continuation.
    pub fn transient(&self) -> SimTransient {
        SimTransient {
            paths: self.paths.clone(),
            pending_private: self.pending_private.clone(),
            last_incident_minute: self.last_incident_minute,
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
            Action::TransferItem { target, item, qty } => {
                in_range(target)?;
                if actor.inventory.quantity(item) < *qty {
                    return Err(RejectReason::InsufficientItems(item.clone()));
                }
            }
            Action::Pay { target, amount } => {
                in_range(target)?;
                if actor.inventory.money < *amount {
                    return Err(RejectReason::InsufficientFunds);
                }
            }
            Action::ShareClaim { target, fact, .. } => {
                in_range(target)?;
                self.validate_disclosure(actor, fact)?;
            }
            Action::WarnContact { target, fact } => {
                in_range(target)?;
                self.validate_disclosure(actor, fact)?;
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

    /// A fact may only be disclosed if the actor believes it and is permitted
    /// by its sensitivity (inner-circle secrets require inner-circle status).
    fn validate_disclosure(&self, actor: &Actor, fact: &FactId) -> Result<(), RejectReason> {
        let Some(f) = self.world.fact(fact) else {
            return Err(RejectReason::UnknownFact(fact.clone()));
        };
        if !self.world.believes(&actor.id, fact) {
            return Err(RejectReason::LacksKnowledge(fact.clone()));
        }
        if f.sensitivity == Sensitivity::InnerCircle && !actor.has_flag("inner_circle") {
            return Err(RejectReason::NotPermittedToDisclose(fact.clone()));
        }
        Ok(())
    }

    // -----------------------------------------------------------------
    // Application
    // -----------------------------------------------------------------

    pub fn apply(&mut self, proposal: &ActionProposal) -> ValidationOutcome {
        if self.recording {
            self.ops.push(SimOp::Apply(proposal.clone()));
        }
        self.apply_impl(proposal)
    }

    fn apply_impl(&mut self, proposal: &ActionProposal) -> ValidationOutcome {
        if let Err(reason) = self.validate(proposal) {
            self.reject(proposal, &reason);
            return ValidationOutcome::Rejected(reason);
        }
        let outcome = self.dispatch(proposal);
        if let ValidationOutcome::Rejected(reason) = &outcome {
            self.reject(proposal, reason);
        }
        outcome
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
                note,
            } => self.do_make_offer(proposal, target, give, want, condition, note),
            Action::AcceptOffer { offer } => self.do_accept_offer(proposal, offer),
            Action::RejectOffer { offer } => self.do_reject_offer(proposal, offer),
            Action::Promise {
                target,
                terms,
                predicate,
                deadline,
            } => self.do_promise(proposal, target, terms, predicate, deadline),
            Action::BreakPromise { commitment } => self.do_break_promise(proposal, commitment),
            Action::TransferItem { target, item, qty } => {
                self.do_transfer(proposal, target, item, qty)
            }
            Action::Pay { target, amount } => self.do_pay(proposal, target, amount),
            Action::ShareClaim {
                target,
                fact,
                assertive,
            } => self.do_share_claim(proposal, target, fact, assertive),
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
                self.do_share_claim(proposal, target, fact, true)
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
        }
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
                ValidationOutcome::Accepted
            }
            None => ValidationOutcome::Rejected(RejectReason::Unreachable),
        }
    }

    fn do_make_offer(
        &mut self,
        proposal: &ActionProposal,
        target: ActorId,
        give: Vec<rpg_domain::world::ItemStack>,
        want: Vec<rpg_domain::world::ItemStack>,
        condition: Option<QuestPredicate>,
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
                disclose: None,
                note: note.clone(),
                state: OfferState::Pending,
                created_tick: self.world.tick,
            },
        );
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
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
            social::create_commitment(
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
        }
        if let Some(fact) = &o.disclose {
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
        }
        if let Some(of) = self.world.offers.get_mut(&offer) {
            of.state = OfferState::Accepted;
        }
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
        ValidationOutcome::Accepted
    }

    fn move_item(&mut self, from: &ActorId, to: &ActorId, item: &ItemId, qty: u32) {
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
        social::apply_gift(&mut self.world, from, to, value_norm);
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        self.events.push(
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
        social::apply_gift(&mut self.world, &from, &target, norm);
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        self.events.push(
            tick,
            minute,
            Some(from.clone()),
            EventKind::MoneyPaid {
                from,
                to: target,
                amount,
            },
            Vec::new(),
            Vec::new(),
        );
        ValidationOutcome::Accepted
    }

    fn do_share_claim(
        &mut self,
        proposal: &ActionProposal,
        target: ActorId,
        fact: FactId,
        assertive: bool,
    ) -> ValidationOutcome {
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        social::share_claim(
            &mut self.world,
            &mut self.events,
            &mut self.rng,
            tick,
            minute,
            &proposal.actor,
            &target,
            &fact,
            assertive,
        );
        ValidationOutcome::Accepted
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
        let conf = about
            .as_ref()
            .and_then(|f| {
                self.world
                    .actor(&proposal.actor)
                    .and_then(|a| a.beliefs.get(f))
            })
            .map(|b| b.confidence)
            .unwrap_or(0.5);
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
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
        ValidationOutcome::Accepted
    }

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
                // Recover any physical items left here.
                let found = self.world.location_items.remove(&loc).unwrap_or_default();
                for stack in found {
                    if let Some(a) = self.world.actor_mut(&proposal.actor) {
                        a.inventory.add(stack.item.clone(), stack.qty);
                    }
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
                }
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
                    }
                }
            }
            InvestigateTarget::Item(item) => {
                if let Some(a) = self.world.actor_mut(&proposal.actor) {
                    a.inspected.insert(item.as_str().to_string());
                }
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
        if let Some(f) = self.world.faction_mut(&faction) {
            for i in resolved {
                if let Some(p) = f.priorities.get_mut(i) {
                    p.resolved = true;
                }
            }
        }
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
        social::apply_threat(&mut self.world, &proposal.actor, &target);
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        self.events.push(
            tick,
            minute,
            Some(proposal.actor.clone()),
            EventKind::ThreatMade {
                from: proposal.actor.clone(),
                to: target,
                demand,
            },
            Vec::new(),
            Vec::new(),
        );
        ValidationOutcome::Accepted
    }

    fn do_comfort(&mut self, proposal: &ActionProposal, target: ActorId) -> ValidationOutcome {
        social::apply_comfort(&mut self.world, &proposal.actor, &target);
        let (tick, minute) = (self.world.tick, self.world.minute_of_day);
        self.events.push(
            tick,
            minute,
            Some(proposal.actor.clone()),
            EventKind::Comforted {
                from: proposal.actor.clone(),
                to: target,
            },
            Vec::new(),
            Vec::new(),
        );
        ValidationOutcome::Accepted
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
            Decision::Accepted => ValidationOutcome::Accepted,
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

        let routine = self.routine_proposals(&acted);
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
        report
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
    /// clues. Runs after every step when the director is enabled.
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
