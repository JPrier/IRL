//! Shared schema for the compositional behavior engine (design §1, §3, §11
//! Phase 0). These are the only cross-crate vocabulary types for intents,
//! speech acts, evidence, effects, outcomes, goal predicates and causal
//! traces.
//!
//! Everything here is plain serializable data with no behaviour logic: the
//! cognition core lives in `rpg_behavior`, the authoritative transitions in
//! `rpg_sim`. All new state is `serde(default)`, so older saves migrate
//! forward (see `rpg_persistence::SAVE_VERSION`).

use crate::action::{QuestPredicate, RejectReason, ValidationOutcome};
use crate::geom::Vec2;
use crate::ids::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// The 16-primitive effect algebra (design §1.1)
// ---------------------------------------------------------------------------

/// The closed vocabulary of typed interaction effects. Every authoritative
/// world mutation lowers into zero or more of these; a high-level action is a
/// *composition* of them. World-authoritative effects (Transfer, Transform)
/// and mind-local effects (Appraise, ChangeGoal) are deliberately distinct:
/// observing an objective effect is not guaranteed, and appraisal is not
/// world truth.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Effect {
    /// Change position or containment.
    Move {
        actor: ActorId,
        from: Vec2,
        to: Vec2,
    },
    /// Change custody or ownership of resources (items or money).
    Transfer {
        from: ActorId,
        to: ActorId,
        item: Option<ItemId>,
        qty: u32,
    },
    /// Produce/consume/alter an object.
    Transform {
        actor: ActorId,
        subject: ItemId,
        consumed: u32,
        produced: u32,
        note: String,
    },
    /// Deliver a sensory observation to a perceiver. `observed` describes
    /// what was perceived (an actor, a clue container, an event) because
    /// observation targets are not limited to actors.
    Observe {
        observer: ActorId,
        observed: String,
        channel: AcquisitionChannel,
        origin_event: Option<EventId>,
    },
    /// Transmit a proposition as speech or writing; creates a claim record,
    /// never automatic belief.
    Assert {
        speaker: ActorId,
        proposition: FactId,
        asserted_polarity: bool,
        believed: Option<bool>,
    },
    /// Request information (the answer is whatever the target's own evidence
    /// supports — never an omniscient reply).
    Query {
        asker: ActorId,
        target: ActorId,
        about: FactId,
    },
    /// Ask for behaviour (request ≠ compliance).
    Request {
        from: ActorId,
        to: ActorId,
        demand: String,
    },
    /// Propose a conditional exchange (acceptance ≠ fulfillment).
    Offer {
        from: ActorId,
        to: ActorId,
        offer: OfferId,
    },
    /// Adopt an obligation/plan (identities, conditions, lifecycle).
    Commit {
        commitment: CommitmentId,
        debtor: ActorId,
        creditor: ActorId,
    },
    /// Grant or revoke permission or role (bounded capability and scope).
    Authorize {
        grantor: ActorId,
        grantee: ActorId,
        scope: String,
    },
    /// Subjective evaluation of an experience (mind-local, actor-specific).
    Appraise {
        subject: ActorId,
        regarding: ActorId,
        summary: String,
    },
    /// Impose institutional/social consequences.
    Sanction {
        authority: ActorId,
        subject: ActorId,
        action: String,
    },
    /// Reserve future time/actions (clock and conflict handling).
    Schedule {
        by: ActorId,
        with: ActorId,
        at_minute: u32,
    },
    /// Create, reprioritize, complete or abandon a goal (mind-local; never
    /// mutates world truth).
    ChangeGoal {
        actor: ActorId,
        goal: String,
        change: GoalChange,
    },
    /// Update a directed social relation (never an unmediated global score).
    ChangeTie {
        from: ActorId,
        to: ActorId,
        summary: String,
    },
    /// Record an occurrence, stimulus or environmental incident under an
    /// immutable event identifier.
    CreateEvent { summary: String },
}

impl Effect {
    /// Short family name, one of the 16 primitives (design §1.1).
    pub fn family(&self) -> &'static str {
        match self {
            Effect::Move { .. } => "move",
            Effect::Transfer { .. } => "transfer",
            Effect::Transform { .. } => "transform",
            Effect::Observe { .. } => "observe",
            Effect::Assert { .. } => "assert",
            Effect::Query { .. } => "query",
            Effect::Request { .. } => "request",
            Effect::Offer { .. } => "offer",
            Effect::Commit { .. } => "commit",
            Effect::Authorize { .. } => "authorize",
            Effect::Appraise { .. } => "appraise",
            Effect::Sanction { .. } => "sanction",
            Effect::Schedule { .. } => "schedule",
            Effect::ChangeGoal { .. } => "change_goal",
            Effect::ChangeTie { .. } => "change_tie",
            Effect::CreateEvent { .. } => "create_event",
        }
    }

    /// The actor whose agency produced the effect (where applicable).
    pub fn actor(&self) -> Option<&ActorId> {
        match self {
            Effect::Move { actor, .. }
            | Effect::Transform { actor, .. }
            | Effect::ChangeGoal { actor, .. }
            | Effect::Schedule { by: actor, .. } => Some(actor),
            Effect::Transfer { from, .. }
            | Effect::Assert { speaker: from, .. }
            | Effect::Query { asker: from, .. }
            | Effect::Request { from, .. }
            | Effect::Offer { from, .. }
            | Effect::Authorize { grantor: from, .. }
            | Effect::Sanction {
                authority: from, ..
            } => Some(from),
            Effect::Observe { observer, .. } => Some(observer),
            Effect::Appraise { subject, .. } => Some(subject),
            Effect::Commit { debtor, .. } => Some(debtor),
            Effect::ChangeTie { from, .. } => Some(from),
            Effect::CreateEvent { .. } => None,
        }
    }
}

/// Direction of a [`Effect::ChangeGoal`] (design: create, reprioritize,
/// complete or abandon).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoalChange {
    Created,
    Reprioritized,
    Completed,
    Abandoned,
}

// ---------------------------------------------------------------------------
// Acquisition channels and provenance chains (design §3.1)
// ---------------------------------------------------------------------------

/// How an observer acquired an evidence item. The channel constrains what can
/// be recorded: no observation without a defined route.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcquisitionChannel {
    /// Directly perceived by the observer.
    Witnessed,
    /// Delivered by another actor as testimony.
    Told,
    /// Picked up incidentally (in range but not addressed).
    Overheard,
    /// Read from a document or object.
    Read,
    /// Derived by the observer's own reasoning.
    Inferred,
    /// Authored/assumed without an in-world acquisition (kept distinct on
    /// purpose so authored knowledge stays auditable).
    Assumed,
}

impl AcquisitionChannel {
    /// Default perceived reliability for the channel. Calibration values are
    /// design choices (plan §9: explicitly approximate, acceptance is
    /// behavioural).
    pub fn default_reliability(self) -> f32 {
        match self {
            AcquisitionChannel::Witnessed => 0.95,
            AcquisitionChannel::Read => 0.9,
            AcquisitionChannel::Assumed => 0.8,
            AcquisitionChannel::Told => 0.6,
            AcquisitionChannel::Overheard => 0.55,
            AcquisitionChannel::Inferred => 0.5,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            AcquisitionChannel::Witnessed => "witnessed",
            AcquisitionChannel::Told => "told",
            AcquisitionChannel::Overheard => "overheard",
            AcquisitionChannel::Read => "read",
            AcquisitionChannel::Inferred => "inferred",
            AcquisitionChannel::Assumed => "assumed",
        }
    }
}

/// Stable identifier of a provenance chain: one origin plus the ordered
/// transmission path. All retellings of one claim carry the same chain id, so
/// five repeated reports from one fabricated source never count as five
/// independent confirmations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SourceChainId(pub u64);

impl SourceChainId {
    pub fn raw(self) -> u64 {
        self.0
    }
}

/// The immutable identity of a chain's origin: what makes two evidence items
/// correlated (same originator or same origin event) versus independent.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceChain {
    pub id: SourceChainId,
    /// The canonical event the chain originates from, if any.
    pub origin_event: Option<EventId>,
    /// The ultimate source actor of the claim, if any.
    pub originator: Option<ActorId>,
    /// Ordered, de-duplicated transmission path (originator first).
    pub path: Vec<ActorId>,
}

/// One evidence item in an actor's per-proposition ledger (design §3.1).
/// Supports positive or negative polarity; an actor may hold contradictory
/// reports.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    /// The proposition the evidence is about.
    pub proposition_id: FactId,
    /// Who holds this evidence item.
    pub observer_id: ActorId,
    /// The polarity that was asserted (what the observer was told / saw).
    pub asserted_polarity: bool,
    /// The event the claim originates from, when known.
    pub origin_event_id: Option<EventId>,
    /// The direct source of this report, when another actor.
    pub source_actor_id: Option<ActorId>,
    pub acquisition_channel: AcquisitionChannel,
    /// Subjective assessment of the source's accuracy, 0..1.
    pub perceived_reliability: f32,
    pub received_tick: u64,
    /// Provenance chain: retellings of one origin share this id.
    pub source_chain_id: SourceChainId,
    /// Correlation group: evidence with the same group id shares an origin
    /// (same originator or same origin event) and is *not* independent.
    pub independent_corrob_group: Option<u64>,
}

/// Canonical epistemic state: per-actor, per-proposition evidence ledgers and
/// the chain registry. This is world-canonical data (serialized with the
/// world); the write paths live in `rpg_behavior` and are invoked from the
/// authoritative sim/social transitions.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Epistemics {
    pub evidence: BTreeMap<ActorId, BTreeMap<FactId, Vec<Evidence>>>,
    pub chains: BTreeMap<u64, SourceChain>,
    pub next_chain_id: u64,
}

impl Epistemics {
    /// All evidence items `observer` holds about `fact`, in acquisition order.
    pub fn evidence_about(&self, observer: &ActorId, fact: &FactId) -> &[Evidence] {
        self.evidence
            .get(observer)
            .and_then(|by_fact| by_fact.get(fact))
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub fn chain(&self, id: SourceChainId) -> Option<&SourceChain> {
        self.chains.get(&id.raw())
    }
}

// ---------------------------------------------------------------------------
// Intents and speech acts (design §1, §1.2)
// ---------------------------------------------------------------------------

/// What the actor *wanted* to happen. Deliberately distinct from what the
/// authoritative sim actually did with the proposal.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Intent {
    pub kind: IntentKind,
    pub targets: Vec<ActorId>,
    /// Proposition the intent is about, when epistemic.
    pub about: Option<FactId>,
    /// Free-form description for traces; never authoritative.
    pub description: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntentKind {
    Inform,
    Deceive,
    Inquire,
    RequestAction,
    Trade,
    Promise,
    Threaten,
    Comfort,
    Investigate,
    Conceal,
    Relocate,
    Refuse,
    Authorize,
    Sanction,
    Idle,
    Unknown,
}

/// A typed candidate speech act. A natural sentence may contain several; the
/// parser/policy proposes candidates and the sim validates before anything is
/// committed. Never lets raw text set a belief, resource or world fact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SpeechAct {
    /// Transmit a proposition with a polarity; `deceptive` marks an
    /// intentional assertion contrary to the speaker's own belief (a lie is a
    /// proposition choice, never a truth mutation).
    Assert {
        proposition: FactId,
        polarity: bool,
        deceptive: bool,
    },
    Query {
        about: FactId,
    },
    Request {
        demand: String,
    },
    Offer {
        note: String,
    },
    Threat {
        condition: String,
        demand: String,
    },
    Commit {
        terms: String,
    },
    Refuse {
        what: String,
    },
    Acknowledge,
    Emote {
        expression: String,
    },
    /// Noncanonical or unparseable wording; requires clarification or a
    /// harmless fallback.
    Unknown {
        raw: String,
    },
}

impl SpeechAct {
    /// Short machine name for logs and traces.
    pub fn kind_name(&self) -> &'static str {
        match self {
            SpeechAct::Assert { .. } => "assert",
            SpeechAct::Query { .. } => "query",
            SpeechAct::Request { .. } => "request",
            SpeechAct::Offer { .. } => "offer",
            SpeechAct::Threat { .. } => "threat",
            SpeechAct::Commit { .. } => "commit",
            SpeechAct::Refuse { .. } => "refuse",
            SpeechAct::Acknowledge => "acknowledge",
            SpeechAct::Emote { .. } => "emote",
            SpeechAct::Unknown { .. } => "unknown",
        }
    }
}

// ---------------------------------------------------------------------------
// Goal predicates (design §3: predicate, urgency, beneficiary, abandonment)
// ---------------------------------------------------------------------------

/// A typed goal condition: the world predicate that satisfies the goal, who
/// benefits, how urgent it is, and when the actor may give it up.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GoalPredicate {
    pub predicate: QuestPredicate,
    pub beneficiary: ActorId,
    /// Relative urgency 0..1; higher preempts lower-priority goals.
    pub urgency: f32,
    /// Criteria under which the goal may be abandoned.
    pub abandonment: AbandonmentCriteria,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AbandonmentCriteria {
    /// Give up after this many consecutive failed attempts, if set.
    pub after_failed_attempts: Option<u32>,
    /// Give up once this predicate holds (e.g. the obstacle is moot).
    pub predicate: Option<QuestPredicate>,
}

// ---------------------------------------------------------------------------
// Outcomes and causal traces (design §11 Phase 0 deliverable)
// ---------------------------------------------------------------------------

/// The full typed record of one validated proposal: what was decided, which
/// primitive effects were produced, which events were emitted, and the causal
/// trace explaining why. Rejected proposals carry empty effects and no event
/// ids: rejection mutates nothing.
#[derive(Clone, Debug, PartialEq)]
pub struct ActionOutcome {
    pub actor: ActorId,
    pub action_kind: String,
    pub validation: ValidationOutcome,
    pub effects: Vec<Effect>,
    pub events: Vec<EventId>,
    pub trace: CausalTrace,
}

/// An inspectable explanation: *NPC did X because of goal Y, belief Z,
/// observed event E, and cost C*. Traces are instrumentation (not canonical
/// world state); the sim rebuilds them identically during replay.
#[derive(Clone, Debug, PartialEq)]
pub struct CausalTrace {
    pub actor: ActorId,
    pub action_kind: String,
    /// Active goals that motivated the action (highest priority first).
    pub because_goals: Vec<String>,
    /// Propositions the actor held that bear on the action.
    pub because_beliefs: Vec<FactId>,
    /// Recent events the actor perceived that bear on the action.
    pub because_observations: Vec<EventId>,
    /// What the action cost (items, money, risk, attention…).
    pub costs: Vec<String>,
    /// The primitive effects the action actually produced.
    pub effects: Vec<Effect>,
    /// The recorded production decision (design §4.2: "include the RNG seed
    /// and sample in the replay record"): which option the seeded utility
    /// selection chose, from which scores, with which recorded sample. This
    /// is how sampling is wired into the *recorded* decision stream rather
    /// than remaining a kernel-only demo. `None` for non-sampled paths.
    #[allow(dead_code)]
    pub decision: Option<DecisionRecord>,
}

/// The recorded seeded-utility selection behind one decision (design §4.2):
/// the candidate scores, the softmax temperature, and the actual sample
/// drawn. Recorded per sampled decision; replay reproduces it from the same
/// RNG stream.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionRecord {
    /// The chosen option id.
    pub option: String,
    /// The ranking the sample walked (option id → utility), best first.
    pub utilities: Vec<(String, f32)>,
    /// The uniform sample drawn (`None` = unsampled argmax).
    pub sample: Option<f32>,
    /// Softmax temperature τ_i used.
    pub tau: f32,
}

impl CausalTrace {
    /// Human-readable one-line rendering of the trace.
    pub fn render(&self) -> String {
        let mut line = format!("{} did {}", self.actor, self.action_kind);
        if !self.because_goals.is_empty() {
            line.push_str(&format!(
                " because of goal [{}]",
                self.because_goals.join("; ")
            ));
        }
        if !self.because_beliefs.is_empty() {
            line.push_str(&format!(
                ", belief [{}]",
                beliefs_str(&self.because_beliefs)
            ));
        }
        if !self.because_observations.is_empty() {
            let obs: Vec<String> = self
                .because_observations
                .iter()
                .map(|e| format!("event #{}", e.raw()))
                .collect();
            line.push_str(&format!(", observed [{}]", obs.join(", ")));
        }
        if !self.costs.is_empty() {
            line.push_str(&format!(", cost [{}]", self.costs.join("; ")));
        }
        line.push_str(&format!(
            " -> effects [{}]",
            self.effects
                .iter()
                .map(|e| e.family())
                .collect::<Vec<_>>()
                .join(", ")
        ));
        line
    }
}

fn beliefs_str(beliefs: &[FactId]) -> String {
    beliefs
        .iter()
        .map(|b| b.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// The typed action outcome of a rejected proposal carries no effects; kept
/// as a named helper so callers cannot accidentally attach effects to a
/// rejection.
pub fn rejected_outcome(
    actor: ActorId,
    action_kind: &str,
    reject: RejectReason,
    trace: CausalTrace,
) -> ActionOutcome {
    ActionOutcome {
        actor,
        action_kind: action_kind.to_string(),
        validation: ValidationOutcome::Rejected(reject),
        effects: Vec::new(),
        events: Vec::new(),
        trace,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::RejectReason;

    fn actor(id: &str) -> ActorId {
        ActorId::new(id)
    }

    /// The 16 primitives are exactly the design §1.1 vocabulary.
    #[test]
    fn effect_families_are_the_sixteen_primitives() {
        let effects = vec![
            Effect::Move {
                actor: actor("a"),
                from: Vec2::ZERO,
                to: Vec2::ZERO,
            },
            Effect::Transfer {
                from: actor("a"),
                to: actor("b"),
                item: None,
                qty: 1,
            },
            Effect::Transform {
                actor: actor("a"),
                subject: ItemId::new("x"),
                consumed: 1,
                produced: 1,
                note: String::new(),
            },
            Effect::Observe {
                observer: actor("a"),
                observed: actor("b").to_string(),
                channel: AcquisitionChannel::Witnessed,
                origin_event: None,
            },
            Effect::Assert {
                speaker: actor("a"),
                proposition: FactId::new("f"),
                asserted_polarity: true,
                believed: None,
            },
            Effect::Query {
                asker: actor("a"),
                target: actor("b"),
                about: FactId::new("f"),
            },
            Effect::Request {
                from: actor("a"),
                to: actor("b"),
                demand: String::new(),
            },
            Effect::Offer {
                from: actor("a"),
                to: actor("b"),
                offer: OfferId(1),
            },
            Effect::Commit {
                commitment: CommitmentId(1),
                debtor: actor("a"),
                creditor: actor("b"),
            },
            Effect::Authorize {
                grantor: actor("a"),
                grantee: actor("b"),
                scope: String::new(),
            },
            Effect::Appraise {
                subject: actor("a"),
                regarding: actor("b"),
                summary: String::new(),
            },
            Effect::Sanction {
                authority: actor("a"),
                subject: actor("b"),
                action: String::new(),
            },
            Effect::Schedule {
                by: actor("a"),
                with: actor("b"),
                at_minute: 0,
            },
            Effect::ChangeGoal {
                actor: actor("a"),
                goal: String::new(),
                change: GoalChange::Created,
            },
            Effect::ChangeTie {
                from: actor("a"),
                to: actor("b"),
                summary: String::new(),
            },
            Effect::CreateEvent {
                summary: String::new(),
            },
        ];
        let families: Vec<&str> = effects.iter().map(|e| e.family()).collect();
        assert_eq!(families.len(), 16);
        let expected = [
            "move",
            "transfer",
            "transform",
            "observe",
            "assert",
            "query",
            "request",
            "offer",
            "commit",
            "authorize",
            "appraise",
            "sanction",
            "schedule",
            "change_goal",
            "change_tie",
            "create_event",
        ];
        for (got, want) in families.iter().zip(expected.iter()) {
            assert_eq!(*got, *want);
        }
    }

    #[test]
    fn causal_trace_renders_goal_belief_event_cost() {
        let trace = CausalTrace {
            actor: actor("grocer"),
            action_kind: "share_claim".into(),
            because_goals: vec!["protect family".into()],
            because_beliefs: vec![FactId::new("f_grocer_ledger")],
            because_observations: vec![EventId(7)],
            costs: vec!["1x medicine".into()],
            effects: vec![Effect::Assert {
                speaker: actor("grocer"),
                proposition: FactId::new("f_grocer_ledger"),
                asserted_polarity: true,
                believed: Some(true),
            }],
            decision: None,
        };
        let rendered = trace.render();
        assert!(rendered.contains("grocer did share_claim"));
        assert!(rendered.contains("goal [protect family]"));
        assert!(rendered.contains("belief [f_grocer_ledger]"));
        assert!(rendered.contains("event #7"));
        assert!(rendered.contains("cost [1x medicine]"));
        assert!(rendered.contains("effects [assert]"));
    }

    #[test]
    fn rejected_outcome_has_no_effects() {
        let out = rejected_outcome(
            actor("a"),
            "share_claim",
            RejectReason::LacksKnowledge(FactId::new("f")),
            CausalTrace {
                actor: actor("a"),
                action_kind: "share_claim".into(),
                because_goals: vec![],
                because_beliefs: vec![],
                because_observations: vec![],
                costs: vec![],
                effects: vec![],
                decision: None,
            },
        );
        assert!(matches!(out.validation, ValidationOutcome::Rejected(_)));
        assert!(out.effects.is_empty());
        assert!(out.events.is_empty());
    }

    #[test]
    fn evidence_serializes_roundtrip() {
        let e = Evidence {
            proposition_id: FactId::new("f_x"),
            observer_id: actor("b"),
            asserted_polarity: true,
            origin_event_id: Some(EventId(3)),
            source_actor_id: Some(actor("a")),
            acquisition_channel: AcquisitionChannel::Told,
            perceived_reliability: 0.6,
            received_tick: 12,
            source_chain_id: SourceChainId(1),
            independent_corrob_group: Some(99),
        };
        let json = serde_json::to_string(&e).unwrap();
        let back: Evidence = serde_json::from_str(&json).unwrap();
        assert_eq!(back, e);
        // Design §3.1 field names.
        for key in [
            "proposition_id",
            "observer_id",
            "asserted_polarity",
            "origin_event_id",
            "source_actor_id",
            "acquisition_channel",
            "perceived_reliability",
            "received_tick",
            "source_chain_id",
            "independent_corrob_group",
        ] {
            assert!(json.contains(key), "missing {key} in {json}");
        }
    }

    #[test]
    fn epistemics_defaults_empty() {
        let e = Epistemics::default();
        assert!(e.evidence_about(&actor("a"), &FactId::new("f")).is_empty());
        assert!(e.chains.is_empty());
        assert_eq!(e.next_chain_id, 0);
    }
}
