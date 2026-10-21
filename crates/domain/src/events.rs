//! Append-only canonical event log.
//!
//! Every authoritative state change is represented by one or more `Event`s.
//! Replaying the log (optionally re-using recorded model proposals) must
//! reproduce the exact same canonical state without re-running inference.

use crate::action::{Loudness, RejectReason};
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
    /// A fact was told to an actor (may or may not be believed).
    ClaimShared {
        from: ActorId,
        to: ActorId,
        fact: FactId,
        assertive: bool,
        believed: bool,
    },
    /// An actor came to believe a fact they did not before.
    FactBelieved {
        actor: ActorId,
        fact: FactId,
        source: Provenance,
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
            EventKind::ClaimShared {
                from,
                to,
                fact,
                believed,
                ..
            } => format!("{from} told {to} about {fact} (believed={believed})"),
            EventKind::FactBelieved { actor, fact, .. } => format!("{actor} believes {fact}"),
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
