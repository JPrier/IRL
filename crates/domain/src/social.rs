//! Persistent social records: commitments, exchanges and rumors.
//!
//! These are canonical state. Dialogue can *propose* them, but only the
//! simulation commits them, and they survive save/reload.

use crate::action::QuestPredicate;
use crate::ids::*;
use crate::world::ItemStack;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommitmentKind {
    /// A spoken promise (conditional or not).
    Promise,
    /// A favour one party owes another.
    Favour,
    /// An obligation created by receiving help or a gift.
    Debt,
    /// Coerced agreement (threat/blackmail); carries resentment.
    Coerced,
    /// A standing business contract.
    Contract,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommitmentState {
    Pending,
    Satisfied,
    Broken,
    Expired,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Commitment {
    pub id: CommitmentId,
    pub kind: CommitmentKind,
    /// The actor who owes the obligation.
    pub debtor: ActorId,
    /// The actor owed / expecting the outcome.
    pub creditor: ActorId,
    pub terms: String,
    /// World condition that satisfies the commitment.
    pub predicate: QuestPredicate,
    pub state: CommitmentState,
    pub created_tick: u64,
    #[serde(default)]
    pub deadline_minute: Option<u32>,
    /// How much the debtor resents this (0..1). Rises for coerced deals.
    #[serde(default)]
    pub resentment: f32,
    /// Fact the debtor promised to disclose once the predicate is satisfied.
    #[serde(default)]
    pub promised_fact: Option<FactId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OfferState {
    Pending,
    Accepted,
    Rejected,
    Expired,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Offer {
    pub id: OfferId,
    pub from: ActorId,
    pub to: ActorId,
    /// What `from` will hand over on acceptance.
    #[serde(default)]
    pub give: Vec<ItemStack>,
    /// What `from` wants in return.
    #[serde(default)]
    pub want: Vec<ItemStack>,
    /// Money component (positive = from pays to).
    #[serde(default)]
    pub price: i64,
    /// Optional conditional commitment created on acceptance (e.g. "protect
    /// family before I hand over the ledger").
    #[serde(default)]
    pub condition: Option<QuestPredicate>,
    /// A fact disclosed when the offer completes.
    #[serde(default)]
    pub disclose: Option<FactId>,
    pub note: String,
    pub state: OfferState,
    pub created_tick: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rumor {
    pub id: RumorId,
    pub text: String,
    /// Fact the rumor is about, if any.
    #[serde(default)]
    pub about: Option<FactId>,
    /// Whether the underlying claim is actually true in world truth.
    pub truth: bool,
    /// Who started it.
    pub source: ActorId,
    /// Ordered transmission chain (source first).
    pub chain: Vec<ActorId>,
    /// Average credibility 0..1 among current holders.
    pub confidence: f32,
    pub created_tick: u64,
    /// How many hops the rumor has travelled.
    pub hops: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commitment_defaults_are_pending() {
        let c = Commitment {
            id: CommitmentId(1),
            kind: CommitmentKind::Promise,
            debtor: ActorId::new("a"),
            creditor: ActorId::new("b"),
            terms: "protect family".into(),
            predicate: QuestPredicate::Flag("family_safe".into()),
            state: CommitmentState::Pending,
            created_tick: 0,
            deadline_minute: None,
            resentment: 0.0,
            promised_fact: None,
        };
        assert_eq!(c.state, CommitmentState::Pending);
    }
}
