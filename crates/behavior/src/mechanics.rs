//! Shared epistemic interaction mechanics (design §2, §11 Phase 1): asking,
//! asserting (including intentional lies), reporting, refusing. One set of
//! rules for every speech-act family — no actor-id special cases, no
//! omniscient answers, request ≠ compliance, lying is a proposition choice
//! and never a truth mutation.
//!
//! These functions are the shared mechanics the authoritative sim transitions
//! (and Phase 2 speech-act lowering) call into. They only write epistemic
//! ledger state and append typed events; they never mutate world truth,
//! resources, or relationships.

use crate::epistemics::{
    accept_assertion, aggregate_about, knows_of, Acceptance, AssertionDraft as Draft,
    RecipientAppraisal,
};
use rpg_domain::behavior_types::AcquisitionChannel;
use rpg_domain::events::{EventKind, EventLog};
use rpg_domain::ids::*;
use rpg_domain::rng::DeterministicRng;
use rpg_domain::world::World;

/// The answer an actor can give when asked about a proposition: exactly what
/// their own evidence supports — never the global truth.
#[derive(Clone, Debug, PartialEq)]
pub enum QueryAnswer {
    /// The actor has no evidence and no authored knowledge: they cannot
    /// answer.
    Unknown,
    /// The actor has something to report: the polarities of their evidence
    /// (possibly contradictory), their approximate confidence, and whether
    /// the reports conflict.
    Heard {
        polarities: Vec<bool>,
        confidence: f32,
        contradicted: bool,
    },
}

/// Ask `target` about a proposition. The answer is derived from the target's
/// evidence ledger, falling back to an authored belief of either polarity —
/// never from the world's canonical fact truth. (QUERY: no omniscient
/// answer.)
pub fn answer_query(world: &World, target: &ActorId, fact: &FactId) -> QueryAnswer {
    let entries = world.epistemics.evidence_about(target, fact);
    if !entries.is_empty() {
        let aggregate = aggregate_about(&world.epistemics, target, fact);
        return QueryAnswer::Heard {
            polarities: entries.iter().map(|e| e.asserted_polarity).collect(),
            confidence: aggregate.confidence,
            contradicted: aggregate.contradicted,
        };
    }
    // Authored knowledge (no ledger entry yet) is still the actor's own
    // knowledge — reportable at its authored confidence.
    if let Some(belief) = world.actor(target).and_then(|a| a.beliefs.get(fact)) {
        return QueryAnswer::Heard {
            polarities: vec![belief.believes_true],
            confidence: belief.confidence,
            contradicted: false,
        };
    }
    QueryAnswer::Unknown
}

/// One actor asserts a proposition (any polarity) to another. This is the
/// shared assertion mechanic behind `share_claim` and — for `deceptive`
/// assertions — behind lying: the speaker *knows of* the proposition but
/// their own belief may contradict the asserted polarity. Lying never
/// mutates truth; the recipient's belief is still their own appraisal.
///
/// Returns the acceptance outcome (the evidence item is always recorded;
/// belief is rolled, never forced).
#[allow(clippy::too_many_arguments)]
pub fn assert_proposition(
    world: &mut World,
    events: &mut EventLog,
    rng: &mut DeterministicRng,
    tick: u64,
    minute: u32,
    speaker: &ActorId,
    target: &ActorId,
    fact: &FactId,
    asserted_polarity: bool,
    deceptive: bool,
) -> Option<Acceptance> {
    // A speaker must know of the proposition (authored belief of either
    // polarity, or ledger evidence). Knowledge, not belief-in-truth: this is
    // what makes a *lie* expressible without bypassing knowledge boundaries.
    if !knows_of(world, speaker, fact) {
        return None;
    }
    let rel = world
        .actor(target)
        .map(|a| a.relationship(speaker))
        .unwrap_or_default();
    let speaker_confidence = world
        .actor(speaker)
        .and_then(|a| a.beliefs.get(fact))
        .map(|b| b.confidence)
        .unwrap_or(0.3);
    let appraisal = RecipientAppraisal {
        trust: rel.trust,
        suspicion: rel.suspicion,
        fear: rel.fear,
        assertiveness: 0.8,
    };
    let draft = Draft {
        speaker: Some(speaker.clone()),
        proposition: fact.clone(),
        asserted_polarity,
        channel: AcquisitionChannel::Told,
        origin_event: None,
        reliability: 0.6,
        retold_chain: None,
        deceptive,
    };
    // The typed `ClaimShared` delivery event (and the causal link from the
    // evidence write to it) is emitted by the shared acceptance mechanic.
    let acceptance = accept_assertion(
        world,
        events,
        tick,
        minute,
        target,
        draft,
        appraisal,
        speaker_confidence,
        true,
        rng,
    );
    Some(acceptance)
}

/// Record that a request was made (REQUEST: request ≠ compliance — nothing
/// about the target's behaviour changes here; the response is a separate
/// decision).
pub fn make_request(
    events: &mut EventLog,
    tick: u64,
    minute: u32,
    from: &ActorId,
    to: &ActorId,
    demand: impl Into<String>,
) {
    events.push(
        tick,
        minute,
        Some(from.clone()),
        EventKind::RequestMade {
            from: from.clone(),
            to: to.clone(),
            demand: demand.into(),
        },
        Vec::new(),
        vec![to.clone()],
    );
}

/// Record an explicit refusal. Refusal is a state change only in the
/// observer's ledger of the interaction: nothing is forced on the refusing
/// actor.
pub fn record_refusal(
    world: &mut World,
    events: &mut EventLog,
    tick: u64,
    minute: u32,
    actor: &ActorId,
    target: &ActorId,
    what: impl Into<String>,
) {
    let what = what.into();
    // A refusal is epistemically visible to the target: the target now has
    // evidence *that the actor refused*.
    let draft = Draft {
        speaker: Some(actor.clone()),
        proposition: FactId::new(format!("refusal:{}:{}", actor.as_str(), what)),
        asserted_polarity: true,
        channel: AcquisitionChannel::Told,
        origin_event: None,
        reliability: 0.9,
        retold_chain: None,
        deceptive: false,
    };
    let (evidence, chain) =
        crate::epistemics::record_evidence(&mut world.epistemics, target, &draft, tick);
    events.push(
        tick,
        minute,
        Some(actor.clone()),
        EventKind::EvidenceRecorded {
            observer: target.clone(),
            fact: evidence.proposition_id.clone(),
            asserted_polarity: true,
            channel: evidence.acquisition_channel,
            source_actor: evidence.source_actor_id.clone(),
            chain,
            origin_event: None,
        },
        Vec::new(),
        vec![target.clone()],
    );
}

/// A skeptic's response policy (design §11 Phase 1 deliverable): after an
/// unbelieved assertion with weak or contradictory evidence, the actor asks
/// for more evidence instead of simply disbelieving. Pure function of the
/// aggregate and the acceptance roll — never of actor identity.
pub fn skeptic_seeks_evidence(aggregate: &crate::epistemics::Aggregate, believed: bool) -> bool {
    !believed && (aggregate.contradicted || aggregate.independence_groups < 2)
}

/// Convenience read for tests/inspection: the actor's aggregate about a
/// proposition.
pub fn inspect(world: &World, actor: &ActorId, fact: &FactId) -> crate::epistemics::Aggregate {
    aggregate_about(&world.epistemics, actor, fact)
}
