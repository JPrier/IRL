//! # rpg_domain
//!
//! The stable, serializable heart of the Agentic RPG: identifiers, geometry,
//! world schemas, the typed action ("tool") API, and the abstract ports for
//! replaceable inference/speech backends.
//!
//! This crate has **no** dependency on Bevy, SQLite, or any model runtime, so
//! it can be reused by the headless test runner, the model bench, and the
//! desktop client alike.

pub mod action;
pub mod behavior_types;
pub mod contract;
pub mod deal;
pub mod emotions;
pub mod events;
pub mod geom;
pub mod ids;
pub mod needs;
pub mod ports;
pub mod rng;
pub mod social;
pub mod world;

pub use action::{
    Action, ActionProposal, Loudness, ProposalSource, QuestPredicate, RejectReason,
    ValidationOutcome,
};
pub use behavior_types::{
    AbandonmentCriteria, AcquisitionChannel, ActionOutcome, CausalTrace, Effect, Epistemics,
    Evidence, GoalChange, GoalPredicate, Intent, IntentKind, SourceChain, SourceChainId, SpeechAct,
};
pub use deal::{
    transition, Deal, DealEvent, DealState, DealTransitionError, Installment, InstallmentError,
    Party, ViolationReason, MAX_RENEGOTIATIONS,
};
pub use emotions::{
    EmotionKind, EmotionResponse, EmotionThresholds, Emotions, EmotionsConfig, EMOTION_KINDS,
};
pub use events::{Event, EventKind, EventLog};
pub use geom::{from_grid, to_grid, GridPos, Rect, Vec2};
pub use ids::{
    ActorId, CommitmentId, DealId, EventId, FactId, FactionId, ItemId, LocationId, OfferId,
    RumorId, PLAYER,
};
pub use needs::{
    NeedCrossing, NeedKind, NeedTuning, Needs, NeedsConfig, ThresholdDirection, NEED_KINDS,
};
pub use rng::DeterministicRng;
pub use social::{Commitment, CommitmentKind, CommitmentState, Offer, OfferState, Rumor};
pub use world::*;
