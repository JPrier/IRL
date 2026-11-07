//! Append-only canonical event log.
//!
//! Every authoritative state change is represented by one or more `Event`s.
//! Replaying the log (optionally re-using recorded model proposals) must
//! reproduce the exact same canonical state without re-running inference.

use crate::action::{Loudness, RejectReason};
use crate::behavior_types::{AcquisitionChannel, SourceChainId};
use crate::geom::Vec2;
use crate::ids::*;
use crate::world::{Belief, ContainerKind, ItemStack, Provenance};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Event {
    pub id: EventId,
    pub tick: u64,
    /// Simulated minute-of-day when the event occurred.
    pub sim_minute: u32,
    pub actor: Option<ActorId>,
    pub kind: EventKind,
    /// Events that causally precede this one.
    #[serde(default)]
    pub caused_by: Vec<EventId>,
    /// Actors who directly perceived this event.
    #[serde(default)]
    pub observed_by: Vec<ActorId>,
}

impl Event {
    pub fn describe(&self) -> String {
        self.kind.describe()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EventKind {
    /// The simulation clock advanced one step.
    Tick,
    ActorMoved {
        actor: ActorId,
        from: Vec2,
        to: Vec2,
        location: LocationId,
    },
    /// Speech actually emitted into the world (post-validation).
    SpeechEmitted {
        speaker: ActorId,
        text: String,
        addressee: Option<ActorId>,
        loudness: Loudness,
        location: LocationId,
        /// True if this was inside a private conversation.
        private: bool,
    },
    ItemTransferred {
        from: ActorId,
        to: ActorId,
        item: ItemId,
        qty: u32,
    },
    /// Evidence left an actor's custody into concealment (behavior-engine
    /// Phase 0: the previously event-less hide path is now typed and logged).
    EvidenceHidden {
        actor: ActorId,
        item: ItemId,
        location: LocationId,
    },
    /// An actor recovered an item from a searched location.
    ItemFound {
        actor: ActorId,
        item: ItemId,
        qty: u32,
        location: LocationId,
    },
    MoneyPaid {
        from: ActorId,
        to: ActorId,
        amount: i64,
    },
    OfferCreated {
        offer: OfferId,
        from: ActorId,
        to: ActorId,
        give: Vec<ItemStack>,
        want: Vec<ItemStack>,
        note: String,
    },
    OfferAccepted {
        offer: OfferId,
        by: ActorId,
    },
    OfferRejected {
        offer: OfferId,
        by: ActorId,
    },
    CommitmentCreated {
        commitment: CommitmentId,
        parties: Vec<ActorId>,
        terms: String,
    },
    CommitmentSatisfied {
        commitment: CommitmentId,
    },
    CommitmentBroken {
        commitment: CommitmentId,
        by: ActorId,
    },
    // --- Deal FSM lifecycle (behavior-engine Phase 2, plan §4) -----------
    /// A typed conditional contract was proposed (DealState::Proposed).
    DealProposed {
        deal: DealId,
        debtor: ActorId,
        creditor: ActorId,
        terms: String,
    },
    /// The creditor accepted; give-on-acceptance consideration moved
    /// atomically and the deal is awaiting activation.
    DealAccepted {
        deal: DealId,
        by: ActorId,
    },
    /// The creditor declined the proposal (terminal).
    DealRejected {
        deal: DealId,
        by: ActorId,
    },
    /// The accepted deal entered Active (the obligation is now binding).
    DealActivated {
        deal: DealId,
    },
    /// The trigger predicate holds. *Not* the benefit: ConditionMet ≠
    /// benefit delivered (plan §4.1).
    DealConditionMet {
        deal: DealId,
    },
    /// The promised benefit is due (settlement happens from here).
    DealPerformanceDue {
        deal: DealId,
    },
    /// The contracted benefit was delivered and validated (terminal).
    DealPerformed {
        deal: DealId,
        by: ActorId,
    },
    /// The deal failed: deadline lapsed or debtor reneged (terminal).
    DealViolated {
        deal: DealId,
        reason: crate::deal::ViolationReason,
    },
    /// Released by authority/institution evidence (terminal).
    DealExcused {
        deal: DealId,
        evidence: Vec<FactId>,
    },
    /// Cancelled per the §4.4 authorization matrix (terminal).
    DealCanceled {
        deal: DealId,
        by: ActorId,
    },
    /// The creditor consented to the single renegotiation cycle.
    DealRenegotiated {
        deal: DealId,
        by: ActorId,
    },
    /// A deadline was crossed: exactly one idempotent event, then the grace
    /// policy runs (plan §4.5).
    ObligationDue {
        deal: DealId,
    },
    /// An un-split obligation was delivered short: the deal stays
    /// `PerformanceDue`, nothing contractual mutates (plan §4.3).
    DealPartialDelivery {
        deal: DealId,
        got: u32,
        expected: u32,
    },
    /// One installment of a split obligation settled exactly-once.
    DealInstallmentSettled {
        deal: DealId,
        index: usize,
        performed: usize,
        total: usize,
    },
    /// A fact was told to an actor (may or may not be believed).
    /// `deceptive` marks an intentional lie (design §3.1: the asserted
    /// polarity is the speaker's choice and never a truth mutation); the M2
    /// provenance audit reads it. `#[serde(default)]` keeps older saves
    /// loading.
    ClaimShared {
        from: ActorId,
        to: ActorId,
        fact: FactId,
        assertive: bool,
        believed: bool,
        #[serde(default)]
        deceptive: bool,
    },
    /// An actor came to believe a fact they did not before.
    FactBelieved {
        actor: ActorId,
        fact: FactId,
        source: Provenance,
    },
    /// A provenance-bound evidence item was recorded in an actor's ledger
    /// (behavior-engine Phase 1: every epistemic write is event-sourced).
    EvidenceRecorded {
        observer: ActorId,
        fact: FactId,
        asserted_polarity: bool,
        channel: AcquisitionChannel,
        source_actor: Option<ActorId>,
        chain: SourceChainId,
        origin_event: Option<EventId>,
    },
    /// A physical clue container was placed for a fact.
    CluePlaced {
        fact: FactId,
        kind: ContainerKind,
        location: Option<LocationId>,
        holder: Option<ActorId>,
    },
    CrimeReported {
        by: ActorId,
        authority: ActorId,
        evidence: Vec<FactId>,
    },
    BudgetAllocated {
        faction: FactionId,
        amount: i64,
        purpose: String,
        priority: String,
    },
    RepairCompleted {
        location: LocationId,
        by: FactionId,
        cost: i64,
    },
    ThreatMade {
        from: ActorId,
        to: ActorId,
        demand: String,
    },
    /// A request for behaviour was made (request ≠ compliance).
    RequestMade {
        from: ActorId,
        to: ActorId,
        demand: String,
    },
    Comforted {
        from: ActorId,
        to: ActorId,
    },
    Recruited {
        recruiter: ActorId,
        recruit: ActorId,
    },
    MeetingScheduled {
        with: ActorId,
        target: ActorId,
        location: LocationId,
        at_minute: u32,
    },
    RumorSpread {
        rumor: RumorId,
        by: ActorId,
        text: String,
    },
    ActorDied {
        actor: ActorId,
        cause: String,
    },
    LocationDamaged {
        location: LocationId,
        amount: f32,
        cause: String,
    },
    /// A background/ambient incident occurred during simulation.
    BackgroundIncident {
        location: LocationId,
        description: String,
        severity: f32,
    },
    /// The director proposed a narrative change (not yet committed).
    DirectorProposed {
        proposal: String,
        accepted: bool,
        reason: String,
    },
    /// A checkpoint became active or committed.
    CheckpointChanged {
        checkpoint: String,
        state: String,
    },
    /// A proposal was rejected by the authoritative controller.
    ActionRejected {
        actor: ActorId,
        action: String,
        reason: RejectReason,
    },
    /// A model's proposal text, recorded so replay can avoid re-inference.
    ModelProposal {
        actor: ActorId,
        role: String,
        raw: String,
    },
    // --- Needs / emotions maintenance (behavior-engine Phase 3, plan
    // I-4a/I-4b) ---------------------------------------------------------
    /// A need crossed an urgency threshold (event-sourced; the regular
    /// decay itself is a classified non-evented background mutation).
    NeedThresholdCrossed {
        actor: ActorId,
        need: crate::needs::NeedKind,
        direction: crate::needs::ThresholdDirection,
        value: f32,
    },
    /// The crossed need raised a mapped goal's urgency (bounded by the
    /// tuning's urgency cap; cleared again by [`EventKind::GoalUrgencyReleased`]).
    GoalUrgencyRaised {
        actor: ActorId,
        goal: u64,
        need: crate::needs::NeedKind,
        urgency: f32,
    },
    /// The need recovered above the release threshold (hysteresis); the
    /// previously applied goal urgency boost is removed.
    GoalUrgencyReleased {
        actor: ActorId,
        goal: u64,
        need: crate::needs::NeedKind,
    },
    /// An authoritative question was asked (QUERY lowering, design §1.1:
    /// no omniscient answer). The typed answer follows as
    /// [`EventKind::QueryAnswered`] plus, when the answerer has evidence,
    /// the delivery (`ClaimShared`/`EvidenceRecorded`) it causes.
    QuestionAsked {
        from: ActorId,
        to: ActorId,
        about: FactId,
    },
    /// The answer an actor gave to an authoritative question: exactly what
    /// their own evidence ledger supports (polarities, confidence, whether
    /// their reports conflict), or `unknown` — never the world's truth.
    QueryAnswered {
        answerer: ActorId,
        asker: ActorId,
        about: FactId,
        polarities: Vec<bool>,
        confidence: f32,
        contradicted: bool,
        unknown: bool,
        origin: EventId,
    },
    /// An institutional/social consequence was imposed (SANCTION lowering,
    /// design §1.1: authority, process, cost and evidence). The typed record
    /// of the process step; the consequences (payment, custody, appraisal)
    /// are the sanction's own effects.
    SanctionIssued {
        authority: ActorId,
        subject: ActorId,
        kind: String,
        evidence: Vec<FactId>,
    },
    /// A mind-local appraisal was recorded (design §3.2: ΔR_ij depends on
    /// the observer's values, beliefs, goals, prior relation and context —
    /// never a universal delta). Provenance for relationship deltas (T3.6)
    /// and for emotion writes (T3.12).
    AppraisalRecorded {
        subject: ActorId,
        regarding: ActorId,
        event: Option<EventId>,
        interpretation: String,
    },
}

impl EventKind {
    pub fn describe(&self) -> String {
        match self {
            EventKind::Tick => "tick".into(),
            EventKind::ActorMoved {
                actor, location, ..
            } => {
                format!("{actor} moved within {location}")
            }
            EventKind::SpeechEmitted {
                speaker,
                addressee,
                text,
                ..
            } => match addressee {
                Some(t) => format!("{speaker} -> {t}: \"{text}\""),
                None => format!("{speaker} said: \"{text}\""),
            },
            EventKind::ItemTransferred {
                from,
                to,
                item,
                qty,
            } => format!("{from} gave {qty}x{item} to {to}"),
            EventKind::EvidenceHidden {
                actor,
                item,
                location,
            } => format!("{actor} hid {item} at {location}"),
            EventKind::ItemFound {
                actor,
                item,
                qty,
                location,
            } => format!("{actor} found {qty}x{item} in {location}"),
            EventKind::MoneyPaid { from, to, amount } => {
                format!("{from} paid {amount} to {to}")
            }
            EventKind::OfferCreated {
                offer, from, to, ..
            } => {
                format!("offer #{offer} from {from} to {to}")
            }
            EventKind::OfferAccepted { offer, by } => format!("offer #{offer} accepted by {by}"),
            EventKind::OfferRejected { offer, by } => format!("offer #{offer} rejected by {by}"),
            EventKind::CommitmentCreated {
                commitment,
                parties,
                terms,
            } => format!("commitment #{commitment} ({terms}) among {parties:?}"),
            EventKind::CommitmentSatisfied { commitment } => {
                format!("commitment #{commitment} satisfied")
            }
            EventKind::CommitmentBroken { commitment, by } => {
                format!("commitment #{commitment} broken by {by}")
            }
            EventKind::DealProposed {
                deal,
                debtor,
                creditor,
                terms,
            } => format!("deal #{deal} proposed: {debtor} -> {creditor} ({terms})"),
            EventKind::DealAccepted { deal, by } => format!("deal #{deal} accepted by {by}"),
            EventKind::DealRejected { deal, by } => format!("deal #{deal} rejected by {by}"),
            EventKind::DealActivated { deal } => format!("deal #{deal} active"),
            EventKind::DealConditionMet { deal } => {
                format!("deal #{deal} condition met (benefit not yet delivered)")
            }
            EventKind::DealPerformanceDue { deal } => format!("deal #{deal} performance due"),
            EventKind::DealPerformed { deal, by } => {
                format!("deal #{deal} performed (settled by {by})")
            }
            EventKind::DealViolated { deal, reason } => {
                format!("deal #{deal} violated ({})", reason.name())
            }
            EventKind::DealExcused { deal, .. } => format!("deal #{deal} excused"),
            EventKind::DealCanceled { deal, by } => format!("deal #{deal} canceled by {by}"),
            EventKind::DealRenegotiated { deal, by } => {
                format!("deal #{deal} renegotiated by {by}")
            }
            EventKind::ObligationDue { deal } => format!("deal #{deal} obligation due"),
            EventKind::DealPartialDelivery {
                deal,
                got,
                expected,
            } => format!("deal #{deal} partial delivery: {got}/{expected}"),
            EventKind::DealInstallmentSettled {
                deal,
                index,
                performed,
                total,
            } => format!("deal #{deal} installment {index} settled ({performed}/{total})"),
            EventKind::ClaimShared {
                from,
                to,
                fact,
                believed,
                deceptive,
                ..
            } => {
                if *deceptive {
                    format!("{from} told {to} about {fact} (believed={believed}, INTENTIONAL LIE)")
                } else {
                    format!("{from} told {to} about {fact} (believed={believed})")
                }
            }
            EventKind::FactBelieved { actor, fact, .. } => format!("{actor} believes {fact}"),
            EventKind::EvidenceRecorded {
                observer,
                fact,
                channel,
                chain,
                ..
            } => {
                format!(
                    "{observer} recorded {channel:?} evidence for {fact} (chain #{})",
                    chain.raw()
                )
            }
            EventKind::CluePlaced {
                fact,
                kind,
                location,
                holder,
            } => format!("clue for {fact} placed as {kind:?} at {location:?}/{holder:?}"),
            EventKind::CrimeReported { by, authority, .. } => {
                format!("{by} reported a crime to {authority}")
            }
            EventKind::BudgetAllocated {
                faction,
                amount,
                purpose,
                ..
            } => format!("{faction} allocated {amount} to {purpose}"),
            EventKind::RepairCompleted { location, by, cost } => {
                format!("{by} repaired {location} for {cost}")
            }
            EventKind::ThreatMade { from, to, demand } => {
                format!("{from} threatened {to}: {demand}")
            }
            EventKind::RequestMade { from, to, demand } => {
                format!("{from} asked {to}: {demand}")
            }
            EventKind::Comforted { from, to } => format!("{from} comforted {to}"),
            EventKind::Recruited { recruiter, recruit } => {
                format!("{recruiter} recruited {recruit}")
            }
            EventKind::MeetingScheduled { with, target, .. } => {
                format!("{with} scheduled a meeting with {target}")
            }
            EventKind::RumorSpread { by, text, .. } => format!("{by} spread rumor: \"{text}\""),
            EventKind::ActorDied { actor, cause } => format!("{actor} died ({cause})"),
            EventKind::LocationDamaged {
                location,
                amount,
                cause,
            } => format!("{location} damaged by {amount} ({cause})"),
            EventKind::BackgroundIncident {
                location,
                description,
                ..
            } => format!("background incident at {location}: {description}"),
            EventKind::DirectorProposed {
                proposal, accepted, ..
            } => format!("director proposed {proposal} (accepted={accepted})"),
            EventKind::CheckpointChanged { checkpoint, state } => {
                format!("checkpoint {checkpoint} -> {state}")
            }
            EventKind::ActionRejected {
                actor,
                action,
                reason,
            } => format!("{actor} action {action} rejected: {}", reason.message()),
            EventKind::ModelProposal { actor, role, .. } => {
                format!("{actor} model proposal ({role})")
            }
            EventKind::NeedThresholdCrossed {
                actor,
                need,
                direction,
                value,
            } => match direction {
                crate::needs::ThresholdDirection::FellBelow => {
                    format!(
                        "{actor}'s {} fell below the urgency threshold ({value:.2})",
                        need.name()
                    )
                }
                crate::needs::ThresholdDirection::Recovered => {
                    format!(
                        "{actor}'s {} recovered above the release threshold ({value:.2})",
                        need.name()
                    )
                }
            },
            EventKind::GoalUrgencyRaised {
                actor,
                goal,
                need,
                urgency,
            } => format!(
                "{actor}'s goal #{goal} urgency raised to {urgency:.2} ({} need)",
                need.name()
            ),
            EventKind::GoalUrgencyReleased { actor, goal, need } => {
                format!(
                    "{actor}'s goal #{goal} urgency released ({} need recovered)",
                    need.name()
                )
            }
            EventKind::AppraisalRecorded {
                subject,
                regarding,
                event,
                interpretation,
            } => match event {
                Some(id) => {
                    format!("{subject} appraised {regarding} (event #{id}): {interpretation}")
                }
                None => format!("{subject} appraised {regarding}: {interpretation}"),
            },
            EventKind::QuestionAsked { from, to, about } => {
                format!("{from} asked {to} about {about}")
            }
            EventKind::QueryAnswered {
                answerer,
                asker,
                about,
                polarities,
                confidence,
                contradicted,
                unknown,
                ..
            } => {
                if *unknown {
                    format!("{answerer} could not answer {asker}'s question about {about}")
                } else {
                    format!(
                        "{answerer} answered {asker} about {about}: polarities {polarities:?} \
                         (confidence {confidence:.2}, contradicted={contradicted})"
                    )
                }
            }
            EventKind::SanctionIssued {
                authority,
                subject,
                kind,
                ..
            } => format!("{authority} sanctioned {subject} ({kind})"),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EventLog {
    events: Vec<Event>,
    next_id: u64,
}

impl EventLog {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            next_id: 1,
        }
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn all(&self) -> &[Event] {
        &self.events
    }

    pub fn next_id(&self) -> EventId {
        EventId(self.next_id)
    }

    /// Append a new event, assigning the next id. Returns the assigned id.
    pub fn push(
        &mut self,
        tick: u64,
        sim_minute: u32,
        actor: Option<ActorId>,
        kind: EventKind,
        caused_by: Vec<EventId>,
        observed_by: Vec<ActorId>,
    ) -> EventId {
        let id = EventId(self.next_id);
        self.next_id += 1;
        self.events.push(Event {
            id,
            tick,
            sim_minute,
            actor,
            kind,
            caused_by,
            observed_by,
        });
        id
    }

    /// Most recent `n` events.
    pub fn recent(&self, n: usize) -> &[Event] {
        let start = self.events.len().saturating_sub(n);
        &self.events[start..]
    }

    pub fn get(&self, id: EventId) -> Option<&Event> {
        self.events.iter().find(|e| e.id == id)
    }

    /// Stable checksum over the canonical log, used by replay tests.
    pub fn checksum(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for e in &self.events {
            let bytes = format!(
                "{}|{}|{}|{}|{}",
                e.id.raw(),
                e.tick,
                e.sim_minute,
                e.actor.as_ref().map(|a| a.as_str()).unwrap_or(""),
                e.describe()
            );
            for b in bytes.bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01B3);
            }
        }
        h
    }

    /// Record a belief change for replay/observability.
    #[allow(clippy::too_many_arguments)]
    pub fn push_belief(
        &mut self,
        tick: u64,
        sim_minute: u32,
        actor: ActorId,
        belief: &Belief,
    ) -> EventId {
        self.push(
            tick,
            sim_minute,
            Some(actor.clone()),
            EventKind::FactBelieved {
                actor,
                fact: belief.fact.clone(),
                source: belief.source.clone(),
            },
            Vec::new(),
            Vec::new(),
        )
    }
}
