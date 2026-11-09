//! The typed capability ("tool") API.
//!
//! Models may only *propose* values of [`Action`]. The simulation validates
//! every proposal and is the sole authority allowed to mutate world state.

use crate::geom::Vec2;
use crate::ids::*;
use crate::world::{ContainerKind, ItemStack};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Loudness {
    Whisper,
    Normal,
    Loud,
    Shout,
}

impl Loudness {
    /// Base audible radius in world units before occlusion/location factors.
    pub fn radius(self) -> f32 {
        match self {
            Loudness::Whisper => 1.5,
            Loudness::Normal => 6.0,
            Loudness::Loud => 12.0,
            Loudness::Shout => 22.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InvestigateTarget {
    Location(LocationId),
    Actor(ActorId),
    Item(ItemId),
}

/// A state-changing request. Note that these are *proposals*; each maps to a
/// vetted controller path in `sim`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Action {
    /// Speak aloud. `target` is the intended addressee, if any.
    Speak {
        target: Option<ActorId>,
        text: String,
        loudness: Loudness,
    },
    /// Ask the target to relocate somewhere more private.
    AskPrivately {
        target: ActorId,
    },
    /// Walk to a named location (path planned by sim).
    MoveTo {
        location: LocationId,
    },
    /// Walk to a precise point (used by movement/leash logic).
    MoveToPos {
        pos: Vec2,
    },
    /// Offer an exchange: give `give` in return for `want`.
    MakeOffer {
        target: ActorId,
        give: Vec<ItemStack>,
        want: Vec<ItemStack>,
        /// Optional world condition the offer is contingent on.
        condition: Option<QuestPredicate>,
        /// Fact the proposer promises to disclose once the deal is honoured.
        disclose: Option<FactId>,
        note: String,
    },
    AcceptOffer {
        offer: OfferId,
    },
    RejectOffer {
        offer: OfferId,
    },
    /// Promise to do (or not do) something; creates a commitment.
    Promise {
        target: ActorId,
        terms: String,
        predicate: QuestPredicate,
        deadline: Option<u64>,
    },
    /// Break an existing commitment.
    BreakPromise {
        commitment: CommitmentId,
    },
    /// Propose a typed conditional contract (the deal FSM). The deal enters
    /// `Proposed` and only the named creditor may accept, reject or
    /// renegotiate; acceptance ≠ fulfillment (design §5.2).
    ProposeDeal {
        target: ActorId,
        terms: String,
        /// Trigger predicate: when it holds, the obligation becomes due.
        condition: QuestPredicate,
        /// Promised-benefit predicate: settlement validates against it.
        performance: QuestPredicate,
        #[serde(default)]
        condition_deadline_minute: Option<u64>,
        #[serde(default)]
        performance_deadline_minute: Option<u64>,
        /// Give-on-acceptance consideration (moves atomically at `Accepted`).
        #[serde(default)]
        give: Vec<ItemStack>,
        /// Money component of the give-on-acceptance consideration.
        #[serde(default)]
        money: i64,
        note: String,
    },
    /// Accept a proposed deal as its creditor. Consideration moves
    /// atomically at `Accepted`, before the FSM enters `Active`.
    AcceptDeal {
        deal: DealId,
    },
    /// Reject a proposed deal as its creditor (terminal).
    RejectDeal {
        deal: DealId,
    },
    /// Cancel a deal (authorization matrix: party inferred from the caller;
    /// a unilateral debtor past ConditionMet takes the Violated path).
    CancelDeal {
        deal: DealId,
    },
    /// Deliver a settlement bundle for a due obligation. Validated as one
    /// atomic bundle (escrow: everything moves or nothing does); a short
    /// delivery records `DealPartialDelivery` and mutates nothing contractual.
    SettleDeal {
        deal: DealId,
        deliver: Vec<ItemStack>,
        money: i64,
    },
    /// Renegotiate a due deal (single cycle, creditor consent required).
    RenegotiateDeal {
        deal: DealId,
        #[serde(default)]
        new_condition: Option<QuestPredicate>,
        #[serde(default)]
        new_performance: Option<QuestPredicate>,
        #[serde(default)]
        new_condition_deadline_minute: Option<u64>,
        #[serde(default)]
        new_performance_deadline_minute: Option<u64>,
    },
    /// Release a due deal through authority/institution evidence (plan §4.4:
    /// from `PerformanceDue`, authority/institution evidence ⇒ `Excused`).
    /// The sim validates the authority and that it holds the cited evidence;
    /// only a due deal can be excused (the FSM rejects every other state).
    ExcuseDeal {
        deal: DealId,
        evidence: Vec<FactId>,
    },
    TransferItem {
        target: ActorId,
        item: ItemId,
        qty: u32,
    },
    /// Pay money to another actor.
    Pay {
        target: ActorId,
        amount: i64,
    },
    /// Tell another actor a fact. The default (`deceptive: false`) asserts
    /// what the speaker believes true; `deceptive: true` is an intentional
    /// lie — the speaker asserts the polarity they believe false (a lie is a
    /// proposition choice, never a truth mutation; design §3.1). Both lower
    /// into the same ASSERT machinery and require the speaker to at least
    /// *know of* the proposition; knowledge is not permission.
    ShareClaim {
        target: ActorId,
        fact: FactId,
        /// Whether the speaker asserts it as certain.
        assertive: bool,
        /// Intentional deception: assert the polarity the speaker believes
        /// false. Never mutates world truth.
        #[serde(default)]
        deceptive: bool,
        /// The provenance chain this retelling explicitly continues (plan
        /// I-6 I6b: autonomous retellings select *which* chain they pass on,
        /// which need not be the speaker's latest). The epistemic layer
        /// validates the selection — the chain must be registered and owned
        /// by the speaker for this proposition — and an invalid selection
        /// falls back to the legacy inference (a bogus pointer never
        /// corrupts the chain registry). `None` keeps the legacy behaviour.
        #[serde(default)]
        retold_chain: Option<crate::behavior_types::SourceChainId>,
    },
    /// Ask another actor about a proposition (the QUERY primitive, design
    /// §1.1: no omniscient answer). The authoritative lowering derives the
    /// answer from the target's own evidence ledger — a target with no
    /// evidence answers Unknown even when the world's truth says otherwise.
    Query {
        target: ActorId,
        about: FactId,
    },
    /// Impose an institutional/social consequence (the SANCTION primitive,
    /// design §1.1: authority, process, cost and evidence). The sim validates
    /// that the proposer actually holds the authority and the cited evidence;
    /// an unauthorized sanction is rejected (not-permitted, not applied).
    Sanction {
        target: ActorId,
        kind: SanctionKind,
        /// Facts the authority cites as grounds; each must be in the
        /// authority's own knowledge.
        evidence: Vec<FactId>,
    },
    /// Spread a (possibly false) claim to the public.
    SpreadRumor {
        text: String,
        about: Option<FactId>,
    },
    Investigate {
        target: InvestigateTarget,
    },
    ReportCrime {
        authority: ActorId,
        evidence: Vec<FactId>,
    },
    HideEvidence {
        item: ItemId,
        location: LocationId,
    },
    WarnContact {
        target: ActorId,
        fact: FactId,
    },
    AllocateBudget {
        faction: FactionId,
        amount: i64,
        purpose: String,
    },
    Repair {
        location: LocationId,
    },
    Recruit {
        target: ActorId,
    },
    ScheduleMeeting {
        target: ActorId,
        location: LocationId,
        at_minute: u32,
    },
    Threaten {
        target: ActorId,
        demand: String,
    },
    Comfort {
        target: ActorId,
    },
    /// Place a physical clue container for a fact (world/director only).
    PlaceClue {
        fact: FactId,
        kind: ContainerKind,
        location: Option<LocationId>,
        holder: Option<ActorId>,
    },
    /// Do nothing this tick; NPCs use this to idle at their routine.
    Idle,
}

impl Action {
    /// Short machine name used in logs and prompts.
    pub fn kind_name(&self) -> &'static str {
        match self {
            Action::Speak { .. } => "speak",
            Action::AskPrivately { .. } => "ask_privately",
            Action::MoveTo { .. } => "move_to",
            Action::MoveToPos { .. } => "move_to_pos",
            Action::MakeOffer { .. } => "make_offer",
            Action::AcceptOffer { .. } => "accept_offer",
            Action::RejectOffer { .. } => "reject_offer",
            Action::Promise { .. } => "promise",
            Action::BreakPromise { .. } => "break_promise",
            Action::ProposeDeal { .. } => "propose_deal",
            Action::AcceptDeal { .. } => "accept_deal",
            Action::RejectDeal { .. } => "reject_deal",
            Action::CancelDeal { .. } => "cancel_deal",
            Action::SettleDeal { .. } => "settle_deal",
            Action::RenegotiateDeal { .. } => "renegotiate_deal",
            Action::ExcuseDeal { .. } => "excuse_deal",
            Action::Query { .. } => "query",
            Action::Sanction { .. } => "sanction",
            Action::TransferItem { .. } => "transfer_item",
            Action::Pay { .. } => "pay",
            Action::ShareClaim { .. } => "share_claim",
            Action::SpreadRumor { .. } => "spread_rumor",
            Action::Investigate { .. } => "investigate",
            Action::ReportCrime { .. } => "report_crime",
            Action::HideEvidence { .. } => "hide_evidence",
            Action::WarnContact { .. } => "warn_contact",
            Action::AllocateBudget { .. } => "allocate_budget",
            Action::Repair { .. } => "repair",
            Action::Recruit { .. } => "recruit",
            Action::ScheduleMeeting { .. } => "schedule_meeting",
            Action::Threaten { .. } => "threaten",
            Action::Comfort { .. } => "comfort",
            Action::PlaceClue { .. } => "place_clue",
            Action::Idle => "idle",
        }
    }

    /// True if the action is a physical activity that occupies the actor and
    /// should suppress a second concurrent move this tick.
    pub fn is_movement(&self) -> bool {
        matches!(self, Action::MoveTo { .. } | Action::MoveToPos { .. })
    }
}

/// A world condition that can be evaluated deterministically. Used by quests,
/// commitments, faction priorities and checkpoint completion.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum QuestPredicate {
    /// Target actor possesses at least `qty` of `item`.
    HasItem {
        actor: ActorId,
        item: ItemId,
        qty: u32,
    },
    /// Actor is alive.
    ActorAlive { actor: ActorId },
    /// Actor is dead.
    ActorDead { actor: ActorId },
    /// The last-known location of `actor` is `location`.
    ActorAt {
        actor: ActorId,
        location: LocationId,
    },
    /// Actor believes `fact` to be true.
    BelievesFact { actor: ActorId, fact: FactId },
    /// Actor has *heard* about `fact` (delivered evidence), whether or not
    /// they came to believe it. Settlement of a disclosure obligation uses
    /// this: the promised disclosure is performed when it was actually
    /// delivered, not when the recipient chose to believe it.
    HeardFact { actor: ActorId, fact: FactId },
    /// The fact has been committed/placed in the world.
    FactCommitted { fact: FactId },
    /// Location's damage has been repaired (damage == 0).
    LocationRepaired { location: LocationId },
    /// A faction has paid at least `amount` toward resolving `priority`.
    FactionPaid {
        faction: FactionId,
        priority: String,
        amount: i64,
    },
    /// Actor has at least `amount` money.
    HasMoney { actor: ActorId, amount: i64 },
    /// All sub-predicates hold.
    All(Vec<QuestPredicate>),
    /// At least one sub-predicate holds.
    Any(Vec<QuestPredicate>),
    /// Negation.
    Not(Box<QuestPredicate>),
    /// Opaque, explicit, scenario-defined predicate resolved by the sim.
    Flag(String),
}

/// The institutional consequence a [`Action::Sanction`] imposes (design
/// §1.1: "fine, arrest, boycott, praise").
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum SanctionKind {
    /// Extract a monetary penalty from the subject (the sanction's cost).
    Fine { amount: i64 },
    /// Take the subject into custody.
    Arrest,
    /// The institution refuses trade/cooperation with the subject.
    Boycott,
    /// Institutional commendation.
    Praise,
}

impl SanctionKind {
    pub fn name(self) -> &'static str {
        match self {
            SanctionKind::Fine { .. } => "fine",
            SanctionKind::Arrest => "arrest",
            SanctionKind::Boycott => "boycott",
            SanctionKind::Praise => "praise",
        }
    }
}

/// Who is asking for the action to happen.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalSource {
    Player,
    NpcPolicy,
    NpcModel,
    Director,
    Scripted,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionProposal {
    pub actor: ActorId,
    pub action: Action,
    pub source: ProposalSource,
    /// Free-form rationale recorded for observability; never authoritative.
    #[serde(default)]
    pub rationale: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ValidationOutcome {
    /// Applied and committed as one or more events.
    Accepted,
    /// Recorded as a pending offer/commitment that requires acceptance.
    Pending {
        offer: Option<OfferId>,
        commitment: Option<CommitmentId>,
    },
    /// Rejected; no world state changed.
    Rejected(RejectReason),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum RejectReason {
    UnknownActor(ActorId),
    ActorNotAlive(ActorId),
    UnknownLocation(LocationId),
    UnknownItem(ItemId),
    UnknownFact(FactId),
    UnknownFaction(FactionId),
    UnknownOffer(OfferId),
    UnknownCommitment(CommitmentId),
    /// Target is too far away / not in the same room.
    OutOfRange,
    /// Path to the destination is blocked.
    Unreachable,
    /// Speaker does not believe the fact, or lacks provenance.
    LacksKnowledge(FactId),
    /// Speaker is not authorised to disclose (e.g. inner-circle secret).
    NotPermittedToDisclose(FactId),
    /// Insufficient inventory.
    InsufficientItems(ItemId),
    InsufficientFunds,
    /// Target is not willing (validated by social/decision layer).
    TargetUnwilling,
    /// Capability not held by actor.
    NoCapability(String),
    /// A prior commitment already exists.
    AlreadyCommitted,
    /// Unknown deal (typed conditional contract).
    UnknownDeal(DealId),
    /// Generic invariant violation with explanation.
    Invalid(String),
}

impl RejectReason {
    pub fn message(&self) -> String {
        match self {
            RejectReason::UnknownActor(a) => format!("unknown actor {a}"),
            RejectReason::ActorNotAlive(a) => format!("{a} is not alive"),
            RejectReason::UnknownLocation(l) => format!("unknown location {l}"),
            RejectReason::UnknownItem(i) => format!("unknown item {i}"),
            RejectReason::UnknownFact(f) => format!("unknown fact {f}"),
            RejectReason::UnknownFaction(f) => format!("unknown faction {f}"),
            RejectReason::UnknownOffer(o) => format!("unknown offer {o}"),
            RejectReason::UnknownCommitment(c) => format!("unknown commitment {c}"),
            RejectReason::OutOfRange => "target out of range".into(),
            RejectReason::Unreachable => "destination unreachable".into(),
            RejectReason::LacksKnowledge(f) => format!("actor lacks knowledge of {f}"),
            RejectReason::NotPermittedToDisclose(f) => {
                format!("actor is not permitted to disclose {f}")
            }
            RejectReason::InsufficientItems(i) => format!("insufficient {i}"),
            RejectReason::InsufficientFunds => "insufficient funds".into(),
            RejectReason::TargetUnwilling => "target is unwilling".into(),
            RejectReason::NoCapability(c) => format!("missing capability: {c}"),
            RejectReason::AlreadyCommitted => "already committed".into(),
            RejectReason::UnknownDeal(d) => format!("unknown deal #{d}"),
            RejectReason::Invalid(s) => s.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loudness_radii_are_ordered() {
        assert!(Loudness::Whisper.radius() < Loudness::Normal.radius());
        assert!(Loudness::Normal.radius() < Loudness::Loud.radius());
        assert!(Loudness::Loud.radius() < Loudness::Shout.radius());
    }
}
