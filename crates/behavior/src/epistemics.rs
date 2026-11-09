//! The evidence ledger and epistemic acceptance mechanic (design §3.1).
//!
//! * `record_evidence` inserts a provenance-bound evidence item, inheriting
//!   (or creating) the claim's **source chain** so retellings of one origin
//!   share a chain id.
//! * `aggregate_about` folds the ledger into an [`Aggregate`] with
//!   **correlation protection**: evidence items whose chains share an origin
//!   (same origin event, or same ultimate originator) are grouped and
//!   down-weighted instead of counted as independent confirmations.
//! * `accept_assertion` computes the recipient's acceptance probability from
//!   their relationship to the speaker plus the aggregated prior, inserts the
//!   evidence item, emits the typed `EvidenceRecorded` event, and rolls the
//!   belief — never deterministically.
//!
//! Acceptance is explicitly approximate (design §3.1): the aggregate is a
//! calibratable heuristic, not a validated Bayesian posterior. Acceptance is
//! behavioural — a liar can fool a trusting NPC while a skeptic asks for more
//! evidence — and the `assertive` flag weights the odds but never forces
//! belief.

use rpg_domain::behavior_types::{
    AcquisitionChannel, Epistemics, Evidence, SourceChain, SourceChainId,
};
use rpg_domain::events::{EventKind, EventLog};
use rpg_domain::ids::*;
use rpg_domain::rng::DeterministicRng;
use rpg_domain::world::World;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Deterministic FNV-1a 64-bit hash over a byte string. Used only to derive
/// stable correlation-group ids from chain identities.
fn fnv64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// The correlation-group id of a chain: derived from the chain's immutable
/// origin identity, so it is stable for the life of the chain. Chains with
/// the same origin event, or the same ultimate originator, are correlated;
/// chains with neither (self-discovery, authored knowledge) are independent.
fn group_identity(epistemics: &Epistemics, chain_id: SourceChainId) -> u64 {
    match epistemics.chain(chain_id) {
        Some(chain) => match (chain.origin_event, chain.originator.as_ref()) {
            (Some(event), _) => fnv64(format!("event:{}", event.raw()).as_bytes()),
            (None, Some(originator)) => {
                fnv64(format!("originator:{}", originator.as_str()).as_bytes())
            }
            (None, None) => fnv64(format!("chain:{}", chain_id.raw()).as_bytes()),
        },
        None => fnv64(format!("chain:{}", chain_id.raw()).as_bytes()),
    }
}

/// Does the actor know *of* the proposition at all (authored belief of either
/// polarity, or any ledger evidence)? Knowing of a claim is distinct from
/// believing it — and distinct from being permitted to disclose it.
pub fn knows_of(world: &World, actor: &ActorId, fact: &FactId) -> bool {
    if let Some(a) = world.actor(actor) {
        if a.beliefs.contains_key(fact) {
            return true;
        }
    }
    !world.epistemics.evidence_about(actor, fact).is_empty()
}

/// Fold the observer's ledger for one proposition into an [`Aggregate`].
///
/// Correlation protection: items are bucketed by their chain's origin
/// identity; each bucket contributes `max_reliability + 0.08 per extra
/// distinct chain, capped at +0.24` — so five retellings from one origin do
/// not raise certainty like five independent sources. The neutral mass of
/// `0.5` in the denominator keeps single weak reports from becoming
/// certainty.
pub fn aggregate_about(epistemics: &Epistemics, observer: &ActorId, fact: &FactId) -> Aggregate {
    let entries = epistemics.evidence_about(observer, fact);
    if entries.is_empty() {
        return Aggregate {
            confidence: 0.0,
            independence_groups: 0,
            contradicted: false,
            true_groups: 0,
            false_groups: 0,
        };
    }

    // group id → (max reliability, distinct chains in group, polarity votes).
    let mut groups: BTreeMap<u64, (f32, BTreeSet<SourceChainId>, u32)> = BTreeMap::new();
    for e in entries {
        let gid = group_identity(epistemics, e.source_chain_id);
        let entry = groups.entry(gid).or_insert((0.0, BTreeSet::new(), 0));
        entry.0 = entry.0.max(e.perceived_reliability.clamp(0.0, 1.0));
        entry.1.insert(e.source_chain_id);
        if e.asserted_polarity {
            entry.2 += 1;
        }
    }

    // The fold itself is the extracted pure core (the same symbol the T6.8
    // proof exercises); `into_values` yields groups in ascending group-id
    // order and the Vec preserves it, so the accumulation order - and with
    // it the f32 result - is identical to the previous inline fold.
    let group_data: Vec<GroupData> = groups
        .into_values()
        .map(|(max_rel, chains, true_votes)| GroupData {
            max_reliability: max_rel,
            distinct_chains: chains.len() as u32,
            true_votes,
        })
        .collect();
    aggregate_from_groups(&group_data)
}

/// The per-group corroboration weight (I-8/T6.8 proof target): the
/// strongest item in the group, plus a small capped bonus for extra
/// *distinct chains* inside the same origin group - partial corroboration
/// without full independence. Pure, total and side-effect free;
/// [`aggregate_about`] calls this for every correlation group, and the
/// T6.8 Kani harness (`crates/proof-kernels`) proves the cap
/// (`weight <= max_reliability + 0.24` for any chain count) directly
/// against this symbol.
pub fn group_weight(max_reliability: f32, distinct_chains: usize) -> f32 {
    (max_reliability + 0.08 * (distinct_chains.saturating_sub(1) as f32))
        .min(max_reliability + 0.24)
}

/// One pre-grouped correlation group: the strongest perceived reliability
/// among the group's items, how many distinct provenance chains it
/// contains, and how many items asserted the true polarity. The numeric
/// fold input for [`aggregate_from_groups`] (I-8/T6.8 proof target).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroupData {
    pub max_reliability: f32,
    pub distinct_chains: u32,
    pub true_votes: u32,
}

/// The numeric fold of pre-grouped correlation groups into an
/// [`Aggregate`]: per-group capped weight ([`group_weight`]), polarity
/// split, and the neutral-mass confidence. Pure, total and side-effect
/// free; [`aggregate_about`] buckets the ledger by origin identity (the
/// map side of the fold) and calls this with the group data, so the
/// production aggregation and the T6.8 proof exercise the same symbols.
pub fn aggregate_from_groups(groups: &[GroupData]) -> Aggregate {
    let mut score_true = 0.0f32;
    let mut score_false = 0.0f32;
    let mut true_groups = 0u32;
    let mut false_groups = 0u32;
    for group in groups {
        let weight = group_weight(group.max_reliability, group.distinct_chains as usize);
        if group.true_votes > 0 {
            score_true += weight;
            true_groups += 1;
        } else {
            score_false += weight;
            false_groups += 1;
        }
    }

    let confidence = if score_true + score_false <= f32::EPSILON {
        0.0
    } else {
        (score_true / (score_true + score_false + 0.5)).clamp(0.0, 1.0)
    };
    Aggregate {
        confidence,
        independence_groups: true_groups + false_groups,
        contradicted: score_true >= 0.25 && score_false >= 0.25,
        true_groups,
        false_groups,
    }
}

/// Map an aggregate to a 0..1 prior strength for the acceptance formula.
/// Zero with no evidence; capped growth for corroborated evidence, so
/// repeated retellings of one origin never approach certainty on their own.
pub fn prior_strength(aggregate: &Aggregate) -> f32 {
    (((aggregate.confidence - 0.5) * 2.5).clamp(0.0, 1.0))
        * (1.0 - 0.5 * aggregate.contradicted as u8 as f32)
}

/// Record one evidence item in the observer's ledger.
///
/// Chain inheritance: when the item comes from a source actor, the observer's
/// entry joins the source's *latest* chain for the same proposition — claims
/// preserve their origin chain across retellings. Otherwise a fresh chain is
/// created from the draft's origin event (or originator).
///
/// A draft may carry an explicitly selected retold chain
/// ([`AssertionDraft::retold_chain`]): when the caller knows which provenance
/// chain this delivery continues (e.g. the reteller heard the claim on one of
/// several chains they hold), that selection wins over the latest-chain
/// fallback. The selection is validated: the chain must be registered, and the
/// speaker must own it for this proposition (they are the chain's originator,
/// or they hold evidence about the proposition on that chain). An invalid
/// selection is ignored and the legacy inference applies — a bogus pointer
/// must never corrupt the chain registry.
///
/// Emits nothing: the caller emits the typed `EvidenceRecorded` event so the
/// epistemic write is event-sourced; `accept_assertion` does this for the
/// acceptance path.
pub fn record_evidence(
    epistemics: &mut Epistemics,
    observer: &ActorId,
    draft: &AssertionDraft,
    tick: u64,
) -> (Evidence, SourceChainId) {
    // Inherit a chain for this delivery, if any:
    // 1. An explicitly selected retold chain, when it is valid (registered,
    //    and owned by the speaker for this proposition).
    // 2. A repeated delivery of the same claim from the same speaker to the
    //    same observer reuses that pair's chain.
    // 3. Otherwise a retelling from a source who already holds evidence for
    //    the proposition joins the source's latest chain — claims preserve
    //    their origin chain.
    let explicit = draft.retold_chain.filter(|id| {
        let Some(chain) = epistemics.chain(*id) else {
            return false;
        };
        let Some(speaker) = draft.speaker.as_ref() else {
            return false;
        };
        chain.originator.as_ref() == Some(speaker)
            || epistemics
                .evidence_about(speaker, &draft.proposition)
                .iter()
                .any(|e| e.source_chain_id == *id)
    });
    let inherited = explicit.or_else(|| {
        draft.speaker.as_ref().and_then(|speaker| {
            epistemics
                .evidence_about(observer, &draft.proposition)
                .iter()
                .rev()
                .find(|e| e.source_actor_id.as_ref() == Some(speaker))
                .map(|e| e.source_chain_id)
                .or_else(|| {
                    epistemics
                        .evidence_about(speaker, &draft.proposition)
                        .last()
                        .map(|e| e.source_chain_id)
                })
        })
    });

    let chain_id = match inherited {
        Some(id) => id,
        None => {
            let id = SourceChainId(epistemics.next_chain_id);
            epistemics.next_chain_id += 1;
            let mut path = Vec::new();
            if let Some(originator) = &draft.speaker {
                path.push(originator.clone());
            }
            path.push(observer.clone());
            epistemics.chains.insert(
                id.raw(),
                SourceChain {
                    id,
                    origin_event: draft.origin_event,
                    originator: draft.speaker.clone(),
                    path,
                },
            );
            id
        }
    };

    // Extend the transmission path (bookkeeping only; never changes the
    // chain's origin identity or correlation group).
    if let Some(chain) = epistemics.chains.get_mut(&chain_id.raw()) {
        if !chain.path.contains(observer) {
            chain.path.push(observer.clone());
        }
    }

    let evidence = Evidence {
        proposition_id: draft.proposition.clone(),
        observer_id: observer.clone(),
        asserted_polarity: draft.asserted_polarity,
        origin_event_id: draft.origin_event,
        source_actor_id: draft.speaker.clone(),
        acquisition_channel: draft.channel,
        perceived_reliability: draft.reliability.clamp(0.0, 1.0),
        received_tick: tick,
        source_chain_id: chain_id,
        independent_corrob_group: Some(group_identity(epistemics, chain_id)),
    };
    epistemics
        .evidence
        .entry(observer.clone())
        .or_default()
        .entry(draft.proposition.clone())
        .or_default()
        .push(evidence.clone());
    (evidence, chain_id)
}

/// The recipient's relationship-derived appraisal of a speaker, passed in by
/// the social layer so this crate stays relationship-agnostic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecipientAppraisal {
    /// Trust of the observer toward the source, −1..1.
    pub trust: f32,
    /// Suspicion toward the source, −1..1.
    pub suspicion: f32,
    /// Fear of the source, −1..1.
    pub fear: f32,
    /// How forcefully the claim was asserted, 0..1 (never forces belief).
    pub assertiveness: f32,
}

/// A proposition being delivered to an observer, ready to be recorded.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AssertionDraft {
    /// The direct source actor, if the channel is interpersonal.
    pub speaker: Option<ActorId>,
    pub proposition: FactId,
    /// The polarity that was asserted (what the observer was told / saw).
    pub asserted_polarity: bool,
    pub channel: AcquisitionChannel,
    /// The event the claim originates from, when known.
    pub origin_event: Option<EventId>,
    /// The recipient's perceived reliability of this item, 0..1.
    pub reliability: f32,
    /// Explicit retold-chain selection: the provenance chain this delivery
    /// continues, when the caller knows which one the speaker is retelling.
    ///
    /// A speaker may hold evidence on several chains for one proposition
    /// (they heard it from two independent origins); the default inference
    /// would join the *latest* chain, which can merge unrelated origins. An
    /// explicit selection wins; it is validated on use (registered chain,
    /// owned by the speaker for this proposition) and an invalid selection
    /// falls back to the default inference. `#[serde(default)]` keeps older
    /// serialized drafts deserializable.
    #[serde(default)]
    pub retold_chain: Option<SourceChainId>,
    /// Intentional deception marker (design §3.1, plan I-5): this delivery
    /// is an assertion the speaker knows (or believes) to be false. Pure
    /// provenance: it rides on the `ClaimShared` delivery event so the lie is
    /// auditable (M2), and never mutates world truth or forces belief.
    #[serde(default)]
    pub deceptive: bool,
}

/// The folded state of an actor's evidence about one proposition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aggregate {
    /// Approximate posterior-ish strength that the proposition is true,
    /// 0..1 (0 = no evidence; contradicted inputs converge toward 0.5).
    pub confidence: f32,
    /// Number of independence groups contributing evidence.
    pub independence_groups: u32,
    /// True when both polarities are supported above threshold.
    pub contradicted: bool,
    pub true_groups: u32,
    pub false_groups: u32,
}

/// Result of one delivered assertion through [`accept_assertion`].
#[derive(Clone, Debug, PartialEq)]
pub struct Acceptance {
    /// Did the recipient come to believe the asserted polarity?
    pub believed: bool,
    /// The acceptance probability that was rolled against.
    pub probability: f32,
    /// Prior strength from the pre-existing ledger (0 when empty).
    pub prior_strength: f32,
    /// The evidence item that was recorded (always — delivery is guaranteed
    /// even when belief fails).
    pub evidence: Evidence,
    pub chain: SourceChainId,
    /// The aggregate *after* this item was recorded.
    pub aggregate: Aggregate,
    /// The `ClaimShared` delivery event this acceptance emitted, when the
    /// channel is interpersonal (`Told`). Told-path evidence is causally
    /// linked to it (`EvidenceRecorded.caused_by`).
    pub claim_event: Option<EventId>,
}

/// # The acceptance-probability core
///
/// The acceptance-probability core (I-8/T6.8 P1/P2/P5 proof target): the
/// production relationship-sensitive formula plus the bounded prior term,
/// and the perceived-reliability derivation, in one pure, total function.
///
/// Returns `(probability, perceived_reliability)`:
///
/// * `probability` is clamped to [0.05, 0.97] (P1) - the prior term is
///   `0.2 * strength`, exactly `0` for an empty ledger, so the empty-ledger
///   probability reproduces the legacy relationship-only formula exactly
///   (P2);
/// * `perceived_reliability` is the item reliability the acceptance writes
///   into the ledger, clamped to [0.05, 0.98] (P5).
///
/// [`accept_assertion`] computes the pre-ledger strength with
/// [`aggregate_about`]/[`prior_strength`] and calls this - the numeric
/// arbitration and the production acceptance path are the same symbols.
#[allow(clippy::too_many_arguments)]
pub fn acceptance_probability(
    appraisal: &RecipientAppraisal,
    speaker_confidence: f32,
    strength: f32,
) -> (f32, f32) {
    let trust_01 = (1.0 + appraisal.trust) * 0.5;
    let suspicion_01 = (1.0 + appraisal.suspicion) * 0.5;
    // Perceived reliability of this item: the source's own confidence,
    // modulated by the recipient's assessment of the source.
    let reliability =
        (0.5 * speaker_confidence.clamp(0.0, 1.0) + 0.3 * trust_01 + 0.2 * (1.0 - suspicion_01))
            .clamp(0.05, 0.98);

    // Legacy-compatible acceptance probability (the prior term is 0 for an
    // empty ledger, so existing behaviour is unchanged there).
    let mut p = 0.25
        + 0.45 * trust_01
        + 0.25 * appraisal.assertiveness.clamp(0.0, 1.0)
        + 0.2 * speaker_confidence.clamp(0.0, 1.0)
        + 0.2 * strength
        - 0.35 * suspicion_01
        - 0.15 * appraisal.fear;
    p = p.clamp(0.05, 0.97);
    (p, reliability)
}

/// Shared acceptance mechanic for one delivered assertion (design §3.1):
///
/// 1. Aggregate the observer's *pre-existing* ledger (correlation-protected).
/// 2. Compute the acceptance probability: the legacy relationship-sensitive
///    formula plus a bounded prior term from genuinely independent evidence.
///    An empty ledger reproduces the legacy probability exactly.
/// 3. Roll belief against the probability.
/// 4. On the told path, emit the typed `ClaimShared` delivery event carrying
///    the rolled outcome, insert the evidence item (delivery is guaranteed),
///    and emit the typed `EvidenceRecorded` event causally linked to the
///    delivery (`caused_by`) so every told-path evidence write is traceable
///    to the claim that caused it.
///
#[allow(clippy::too_many_arguments)]
pub fn accept_assertion(
    world: &mut World,
    events: &mut EventLog,
    tick: u64,
    minute: u32,
    observer: &ActorId,
    draft: AssertionDraft,
    appraisal: RecipientAppraisal,
    speaker_confidence: f32,
    assertive: bool,
    rng: &mut DeterministicRng,
) -> Acceptance {
    let prior = aggregate_about(&world.epistemics, observer, &draft.proposition);
    let strength = prior_strength(&prior);

    // The numeric core (probability + perceived reliability) is the
    // extracted pure production function the T6.8 harness proves directly
    // (P1/P2/P5) - same symbol the production acceptance path calls.
    let (p, reliability) = acceptance_probability(&appraisal, speaker_confidence, strength);

    let record_draft = AssertionDraft {
        reliability,
        ..draft
    };
    let believed = rng.chance(p);
    // The told path is a real transmission: the `ClaimShared` delivery event
    // is emitted *before* the evidence it causes is recorded, so the evidence
    // write can link back to the claim (`caused_by`) — never the reverse.
    let claim_event = match (record_draft.channel, record_draft.speaker.as_ref()) {
        (AcquisitionChannel::Told, Some(speaker)) if speaker != observer => Some(events.push(
            tick,
            minute,
            Some(speaker.clone()),
            EventKind::ClaimShared {
                from: speaker.clone(),
                to: observer.clone(),
                fact: record_draft.proposition.clone(),
                assertive,
                believed,
                deceptive: record_draft.deceptive,
            },
            Vec::new(),
            vec![observer.clone()],
        )),
        _ => None,
    };
    let (evidence, chain) = record_evidence(&mut world.epistemics, observer, &record_draft, tick);
    events.push(
        tick,
        minute,
        Some(observer.clone()),
        EventKind::EvidenceRecorded {
            observer: observer.clone(),
            fact: evidence.proposition_id.clone(),
            asserted_polarity: evidence.asserted_polarity,
            channel: evidence.acquisition_channel,
            source_actor: evidence.source_actor_id.clone(),
            chain,
            origin_event: evidence.origin_event_id,
        },
        claim_event.into_iter().collect(),
        Vec::new(),
    );

    let aggregate = aggregate_about(&world.epistemics, observer, &evidence.proposition_id);
    Acceptance {
        believed,
        probability: p,
        prior_strength: strength,
        evidence,
        chain,
        aggregate,
        claim_event,
    }
}
