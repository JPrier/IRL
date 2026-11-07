//! # rpg_behavior
//!
//! The pure cognition core of the compositional behavior engine (design §3,
//! §11 Phase 1): the per-actor, per-proposition **evidence ledger**, provenance
//! **source chains** with correlation protection, the epistemic **acceptance**
//! mechanic, and the shared mechanics behind ask / assert / report / lie /
//! question / refuse.
//!
//! Layering rules:
//!
//! * This crate depends only on `rpg_domain` (types), never on `sim` or
//!   `social`, so it stays a leaf every higher layer can call into.
//! * `World::epistemics` is canonical state: the functions here that write it
//!   (`record_evidence`, `accept_assertion`) are part of authoritative
//!   transitions and are only invoked from `rpg_sim` / `rpg_social` (or tests
//!   constructing states directly). Nothing here mutates any other canonical
//!   field.
//! * Acceptance is explicitly approximate (design §3.1): the aggregate is a
//!   calibratable heuristic, not a validated Bayesian posterior. Acceptance is
//!   behavioural — a liar can fool a trusting NPC while a skeptic asks for
//!   more evidence — and the `assertive` flag never forces belief.

pub mod adapters;
pub mod appraisal;
pub mod epistemics;
pub mod intentions;
pub mod mechanics;
pub mod utility;

pub use adapters::{draft_from_provenance, record_provenance_evidence};
pub use appraisal::{
    aftermath, aftermath_anger_decay, apply_delta, appraise, decay_edge, emotion_deltas,
    saturating_add, Aftermath, AftermathThresholds, AppraisalProfile, AppraisedEvent,
    ContextualEdge, DecayRates, DirectedDelta,
};
pub use epistemics::{
    accept_assertion, aggregate_about, knows_of, prior_strength, record_evidence, Acceptance,
    Aggregate, AssertionDraft, RecipientAppraisal,
};
pub use intentions::{consider_replan, Intention, Replan, ReplanReason};
pub use mechanics::{
    answer_query, assert_proposition, inspect, make_request, record_refusal,
    skeptic_seeks_evidence, QueryAnswer,
};
pub use utility::{
    needs_term, rank, rank_with_needs, sample, Feasibility, Params, Ranking, Sampled, UtilityOption,
};
