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
    /// Tell another actor a fact the speaker believes.
    ShareClaim {
        target: ActorId,
        fact: FactId,
        /// Whether the speaker asserts it as certain.
        assertive: bool,
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
