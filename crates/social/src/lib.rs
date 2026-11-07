//! # rpg_social
//!
//! Social causality: multi-dimensional relationships, provenance-aware beliefs,
//! commitments/obligations, and rumor propagation.
//!
//! All functions are deterministic given the same [`DeterministicRng`] state
//! and mutate only canonical [`World`] state (plus the append-only event log).

use rpg_behavior::{
    AftermathThresholds, AppraisalProfile, AppraisedEvent, ContextualEdge, DecayRates,
};
use rpg_domain::action::QuestPredicate;
use rpg_domain::emotions::{EmotionKind, EmotionsConfig};
use rpg_domain::events::{EventKind, EventLog};
use rpg_domain::ids::*;
use rpg_domain::needs::NeedsConfig;
use rpg_domain::rng::DeterministicRng;
use rpg_domain::social::{Commitment, CommitmentKind, CommitmentState, Rumor};
use rpg_domain::world::{evaluate_predicate, Actor, Belief, Provenance, RelMemory, World};

/// A bundle of relationship deltas. Missing fields are zero.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RelationshipDelta {
    pub trust: f32,
    pub affection: f32,
    pub respect: f32,
    pub fear: f32,
    pub anger: f32,
    pub sadness: f32,
    pub temptation: f32,
    pub suspicion: f32,
    pub obligation: f32,
    pub familiarity: f32,
}

impl RelationshipDelta {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn trust(mut self, v: f32) -> Self {
        self.trust = v;
        self
    }
    pub fn affection(mut self, v: f32) -> Self {
        self.affection = v;
        self
    }
    pub fn respect(mut self, v: f32) -> Self {
        self.respect = v;
        self
    }
    pub fn fear(mut self, v: f32) -> Self {
        self.fear = v;
        self
    }
    pub fn anger(mut self, v: f32) -> Self {
        self.anger = v;
        self
    }
    pub fn sadness(mut self, v: f32) -> Self {
        self.sadness = v;
        self
    }
    pub fn temptation(mut self, v: f32) -> Self {
        self.temptation = v;
        self
    }
    pub fn suspicion(mut self, v: f32) -> Self {
        self.suspicion = v;
        self
    }
    pub fn obligation(mut self, v: f32) -> Self {
        self.obligation = v;
        self
    }
    pub fn familiarity(mut self, v: f32) -> Self {
        self.familiarity = v;
        self
    }
}

/// Apply a relationship delta from `from`'s perspective toward `to`.
pub fn adjust_relationship(
    world: &mut World,
    from: &ActorId,
    to: &ActorId,
    delta: &RelationshipDelta,
) {
    if from == to {
        return;
    }
    let Some(actor) = world.actor_mut(from) else {
        return;
    };
    let rel = actor.relationship_mut(to);
    rel.trust += delta.trust;
    rel.affection += delta.affection;
    rel.respect += delta.respect;
    rel.fear += delta.fear;
    rel.anger += delta.anger;
    rel.sadness += delta.sadness;
    rel.temptation += delta.temptation;
    rel.suspicion += delta.suspicion;
    rel.obligation += delta.obligation;
    rel.familiarity += delta.familiarity;
    rel.clamp_all();
}

// ---------------------------------------------------------------------------
// Directed contextual relationship pathway (behavior-engine Phase 3; design
// §3.2, plan I-4). The *appraised* update: ΔR_ij = f(e, V_i, B_i, goals_i,
// R_ij, context) — never a universal delta.
// ---------------------------------------------------------------------------

/// What one appraised event did to the observer's mind (result record; the
/// caller event-sources it). The edge deltas, the emotion writes and the
/// interpretation are all recorded so causal traces and tests can audit the
/// provenance (T3.6, T3.12).
#[derive(Clone, Debug, PartialEq)]
pub struct AppraisalOutcome {
    pub subject: ActorId,
    pub regarding: ActorId,
    /// The observer's interpretation (design §3.2: the same event may be
    /// read as affection, bribery, humiliation, coercion…).
    pub interpretation: String,
    /// Raw directed deltas computed by the appraisal kernel.
    pub delta: rpg_behavior::DirectedDelta,
    /// The saturating-updated edge (already applied to canonical state).
    pub edge_after: rpg_behavior::ContextualEdge,
    /// Emotion deltas actually applied (sensitivity-scaled, pre-clamp).
    pub emotions: Vec<(rpg_domain::emotions::EmotionKind, f32)>,
}

/// Extract the observer's value profile `V_i` from authored personality,
/// value tags and the current needs state (the `goals_i` input of design
/// §3.2: a deprived safety need amplifies fear appraisals). Pure function of
/// canonical state; no actor-id special cases.
pub fn appraisal_profile(actor: &Actor, needs_config: &NeedsConfig) -> AppraisalProfile {
    let p = &actor.personality;
    let tag = |t: &str| actor.values.iter().any(|v| v == t) || p.has_tag(t);
    let mut greed: f32 = 0.5;
    if p.has_tag("greedy") {
        greed += 0.3;
    }
    if tag("wealth") {
        greed += 0.15;
    }
    if tag("generosity") {
        greed -= 0.2;
    }
    let mut pride: f32 = 0.5;
    if p.has_tag("proud") {
        pride += 0.3;
    }
    if tag("self-reliance") || tag("dignity") {
        pride += 0.15;
    }
    if tag("humility") {
        pride -= 0.2;
    }
    AppraisalProfile {
        greed: greed.clamp(0.0, 1.0),
        pride: pride.clamp(0.0, 1.0),
        bravery: (0.5 * p.risk_tolerance + 0.5 * (1.0 - p.self_preservation)).clamp(0.0, 1.0),
        reciprocity: p.reciprocity.clamp(0.0, 1.0),
        forgiveness: (0.5 + 0.4 * p.agreeableness).clamp(0.0, 1.0),
        safety_urgency: actor
            .needs
            .urgency(rpg_domain::needs::NeedKind::Safety, needs_config),
    }
}

/// Per-actor emotion sensitivities (T3.11): personality scales how strongly
/// an appraised event moves each emotion, which shifts where the response
/// thresholds effectively cross. Pure function of canonical state.
pub fn emotion_sensitivity(actor: &Actor, config: &EmotionsConfig) -> [f32; 6] {
    let p = &actor.personality;
    let mut s = config.sensitivity;
    s[EmotionKind::Fear.index()] *= 1.0 + 0.5 * (1.0 - p.risk_tolerance) + 0.25 * p.neuroticism;
    s[EmotionKind::Anger.index()] *= 1.0 + 0.4 * p.reciprocity;
    s[EmotionKind::Joy.index()] *= 1.0 + 0.3 * p.extraversion;
    s[EmotionKind::Grief.index()] *= 1.0 + 0.25 * p.neuroticism;
    s[EmotionKind::Pride.index()] *= 1.0 + 0.3 * (1.0 - p.agreeableness);
    s[EmotionKind::Shame.index()] *= 1.0 + 0.3 * (1.0 - p.risk_tolerance);
    s
}

/// The observer's interpretation of an event (design §3.2: "the recipient
/// may infer affection, bribery, humiliation, coercion, or a hidden
/// motive"). Rule-based over the profile; recorded as appraisal provenance
/// (T3.6).
fn interpret(event: &AppraisedEvent, profile: &AppraisalProfile) -> String {
    match event {
        AppraisedEvent::GiftReceived { .. } => {
            if profile.pride >= 0.7 && profile.greed < 0.6 {
                "read the gift as a bribe"
            } else if profile.greed >= 0.7 {
                "was delighted by the gift"
            } else {
                "felt grateful for the gift"
            }
        }
        AppraisedEvent::AccurateClaim => "found the claim accurate",
        AppraisedEvent::DeceptionExposed => "discovered a deception",
        AppraisedEvent::ThreatReceived { .. } => "felt threatened",
        AppraisedEvent::ComfortReceived { .. } => "felt comforted",
        AppraisedEvent::HarmInflicted { .. } => "was harmed",
        AppraisedEvent::PromiseKept => "saw the promise kept",
        AppraisedEvent::PromiseBroken => "saw the promise broken",
        AppraisedEvent::InsultReceived { .. } => "felt insulted",
        AppraisedEvent::PraiseReceived => "felt appreciated",
        AppraisedEvent::Sanctioned { .. } => "was sanctioned",
    }
    .to_string()
}

/// Apply one appraised event to the observer's mind (design §3.2; plan I-4 +
/// I-4b): compute the directed contextual deltas from the observer's value
/// profile and current edge (no universal deltas), apply them with
/// saturation, write the event's emotions (mind-local, sensitivity-scaled),
/// and record the appraisal memory with event + interpretation provenance
/// (T3.6). Returns `None` for a self-appraisal or an unknown actor.
pub fn apply_appraised_event(
    world: &mut World,
    subject: &ActorId,
    regarding: &ActorId,
    event: AppraisedEvent,
    origin_event: Option<EventId>,
    emotions_config: &EmotionsConfig,
    needs_config: &NeedsConfig,
) -> Option<AppraisalOutcome> {
    if subject == regarding || world.actor(subject).is_none() || world.actor(regarding).is_none() {
        return None;
    }
    let (profile, sensitivities, edge_before) = {
        let actor = world.actor(subject).unwrap();
        (
            appraisal_profile(actor, needs_config),
            emotion_sensitivity(actor, emotions_config),
            ContextualEdge::from_relationship(&actor.relationship(regarding)),
        )
    };
    let delta = rpg_behavior::appraise(&event, &profile, &edge_before);
    let edge_after = rpg_behavior::apply_delta(&edge_before, &delta);
    let interpretation = interpret(&event, &profile);
    let mut emotions_written = Vec::new();
    let now_tick = world.tick;
    {
        let actor = world.actor_mut(subject)?;
        let rel = actor.relationship_mut(regarding);
        rel.trust_accuracy = edge_after.trust_accuracy;
        rel.trust_benevolence = edge_after.trust_benevolence;
        rel.affection = edge_after.affection;
        rel.respect = edge_after.respect;
        rel.fear = edge_after.fear;
        rel.anger = edge_after.anger;
        rel.obligation = edge_after.obligation;
        rel.familiarity = edge_after.familiarity;
        rel.perceived_power = edge_after.perceived_power;
        rel.clamp_all();
        for (kind, d) in rpg_behavior::emotion_deltas(&event, &profile) {
            let scaled = d * sensitivities[kind.index()];
            if scaled == 0.0 {
                continue;
            }
            actor.emotions.adjust(kind, scaled, 1.0);
            emotions_written.push((kind, scaled));
        }
        actor.push_rel_memory(
            regarding.clone(),
            RelMemory {
                event: origin_event,
                interpretation: interpretation.clone(),
                salience: 1.0,
                tick: now_tick,
            },
        );
    }
    Some(AppraisalOutcome {
        subject: subject.clone(),
        regarding: regarding.clone(),
        interpretation,
        delta,
        edge_after,
        emotions: emotions_written,
    })
}

/// Maintenance pass (called from the sim's deterministic step pipeline):
/// per-dimension relationship decay toward 0, appraisal-memory salience
/// decay, and the forgiveness/retaliation aftermath policy applied to anger
/// decay (T3.5: an actor whose state reaches the forgiveness branch cools
/// faster). Classified background mutation (plan G4): regular decay is
/// non-evented.
pub fn decay_relationships(world: &mut World, rates: &DecayRates, salience_decay: f32) {
    for actor in world.actors.values_mut() {
        // The actor's real dispositions drive the aftermath policy and the
        // disposition-scaled forgiveness decay (T3.5).
        let profile = AppraisalProfile {
            reciprocity: actor.personality.reciprocity.clamp(0.0, 1.0),
            forgiveness: (0.5 + 0.4 * actor.personality.agreeableness).clamp(0.0, 1.0),
            ..AppraisalProfile::default()
        };
        for rel in actor.relationships.values_mut() {
            let mut edge = ContextualEdge::from_relationship(rel);
            let outcome = rpg_behavior::aftermath(&edge, &profile, &AftermathThresholds::default());
            let mut rates = *rates;
            rates.anger = rpg_behavior::aftermath_anger_decay(rates.anger, outcome, &profile);
            rpg_behavior::decay_edge(&mut edge, &rates);
            rel.trust_accuracy = edge.trust_accuracy;
            rel.trust_benevolence = edge.trust_benevolence;
            rel.affection = edge.affection;
            rel.respect = edge.respect;
            rel.fear = edge.fear;
            rel.anger = edge.anger;
            rel.obligation = edge.obligation;
            rel.familiarity = edge.familiarity;
            rel.perceived_power = edge.perceived_power;
        }
        for entries in actor.rel_memory.values_mut() {
            for m in entries.iter_mut() {
                m.salience = (m.salience - salience_decay).max(0.0);
            }
            entries.retain(|m| m.salience > 0.0);
        }
    }
}

/// Outcome of one actor telling another a fact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShareOutcome {
    /// Did the recipient come to believe the claim?
    pub believed: bool,
    /// Was this new information to the recipient?
    pub newly_learned: bool,
}

/// Write the legacy single-slot belief snapshot plus its `FactBelieved`
/// event. This is the compatibility surface every downstream reader (prompt
/// rendering, predicates, investigation) sees; the authoritative per-
/// proposition evidence lives in `World::epistemics` (behavior-engine Phase
/// 1). Shared by [`record_belief`] and [`share_claim`].
#[allow(clippy::too_many_arguments)]
fn write_belief_snapshot(
    world: &mut World,
    events: &mut EventLog,
    tick: u64,
    minute: u32,
    actor: &ActorId,
    fact: &FactId,
    believes_true: bool,
    confidence: f32,
    source: Provenance,
) -> bool {
    let Some(a) = world.actor_mut(actor) else {
        return false;
    };
    let previous = a.beliefs.get(fact).map(|b| b.believes_true);
    let newly = previous != Some(believes_true) || previous.is_none();
    a.beliefs.insert(
        fact.clone(),
        Belief {
            fact: fact.clone(),
            believes_true,
            confidence: confidence.clamp(0.0, 1.0),
            source,
            acquired_tick: tick,
        },
    );
    if newly && believes_true {
        events.push(
            tick,
            minute,
            Some(actor.clone()),
            EventKind::FactBelieved {
                actor: actor.clone(),
                fact: fact.clone(),
                source: Provenance::Assumed,
            },
            Vec::new(),
            Vec::new(),
        );
    }
    newly
}

/// Record a belief on an actor. Returns true if the belief is new or changed
/// from disbelief to belief.
///
/// Compatibility shim (behavior-engine Phase 1): the legacy snapshot is
/// written exactly as before, and the same write is *also* recorded in the
/// actor's evidence ledger with full provenance (channel, source chain,
/// reliability), so every canonical belief write is event-sourced.
#[allow(clippy::too_many_arguments)]
pub fn record_belief(
    world: &mut World,
    events: &mut EventLog,
    tick: u64,
    minute: u32,
    actor: &ActorId,
    fact: &FactId,
    believes_true: bool,
    confidence: f32,
    source: Provenance,
) -> bool {
    let newly = write_belief_snapshot(
        world,
        events,
        tick,
        minute,
        actor,
        fact,
        believes_true,
        confidence,
        source.clone(),
    );
    rpg_behavior::record_provenance_evidence(
        world,
        events,
        tick,
        minute,
        actor,
        fact,
        believes_true,
        &source,
    );
    newly
}

/// `from` tells `to` about `fact`. The recipient's trust, the speaker's
/// assertiveness and existing suspicion determine whether it is believed.
///
/// The speaker must already believe the fact (or be lying deliberately); the
/// caller is responsible for validating that. Returns what happened.
///
/// Behavior-engine Phase 1: delivery is now event-sourced — the recipient
/// always records a provenance-bound evidence item (`EvidenceRecorded` event)
/// whose source chain preserves the claim's origin, and acceptance is rolled
/// from the relationship appraisal plus the correlation-protected prior. The
/// legacy probability formula is unchanged for an empty ledger, and the
/// `assertive` flag still weights the odds without ever forcing belief.
#[allow(clippy::too_many_arguments)]
pub fn share_claim(
    world: &mut World,
    events: &mut EventLog,
    rng: &mut DeterministicRng,
    tick: u64,
    minute: u32,
    from: &ActorId,
    to: &ActorId,
    fact: &FactId,
    assertive: bool,
) -> ShareOutcome {
    assert_claim(
        world, events, rng, tick, minute, from, to, fact, assertive, true, false,
    )
}

/// The general ASSERT delivery (plan I-5): the same mechanism as
/// [`share_claim`] but with the asserted polarity and the intentional-
/// deception marker chosen by the caller. The authoritative sim lowering
/// routes both truthful and deceptive claims through this single production
/// pathway: the asserted polarity (never the speaker's belief) is what gets
/// recorded in the evidence ledger and the `ClaimShared` event, `deceptive`
/// marks the lie for the M2 provenance audit, and world truth is untouched.
#[allow(clippy::too_many_arguments)]
pub fn assert_claim(
    world: &mut World,
    events: &mut EventLog,
    rng: &mut DeterministicRng,
    tick: u64,
    minute: u32,
    from: &ActorId,
    to: &ActorId,
    fact: &FactId,
    assertive: bool,
    asserted_polarity: bool,
    deceptive: bool,
) -> ShareOutcome {
    let speaker_confidence = world
        .actor(from)
        .and_then(|a| a.beliefs.get(fact))
        .map(|b| b.confidence)
        .unwrap_or(0.3);

    let rel = world
        .actor(to)
        .map(|a| a.relationship(from))
        .unwrap_or_default();

    // A confident trusted friend is believed; a distrusted one is doubted.
    // Assertiveness influences the odds but never guarantees belief: even a
    // bargain-backed disclosure must still be judged by the recipient's trust
    // and suspicion. The recipient always *hears* the claim (an evidence item
    // is recorded either way); only conviction is rolled.
    let appraisal = rpg_behavior::RecipientAppraisal {
        trust: rel.trust,
        suspicion: rel.suspicion,
        fear: rel.fear,
        assertiveness: if assertive { 1.0 } else { 0.6 },
    };
    let draft = rpg_behavior::AssertionDraft {
        speaker: Some(from.clone()),
        proposition: fact.clone(),
        asserted_polarity,
        channel: rpg_domain::behavior_types::AcquisitionChannel::Told,
        origin_event: None,
        reliability: 0.6,
        retold_chain: None,
        deceptive,
    };
    let acceptance = rpg_behavior::accept_assertion(
        world,
        events,
        tick,
        minute,
        to,
        draft,
        appraisal,
        speaker_confidence,
        assertive,
        rng,
    );
    let believed = acceptance.believed;
    let p = acceptance.probability;
    // Even when not fully believed, the recipient records an uncertain belief
    // with provenance so the rumor can still propagate as a claim. (The
    // evidence item was already inserted by `accept_assertion`; this writes
    // the legacy snapshot only. The provenance links the delivered claim —
    // the `ClaimShared` event the acceptance path emitted.)
    let newly = write_belief_snapshot(
        world,
        events,
        tick,
        minute,
        to,
        fact,
        believed,
        if believed { p } else { (1.0 - p).max(0.1) },
        Provenance::Told {
            by: from.clone(),
            event: acceptance.claim_event,
        },
    );

    // Being told something repeatedly builds familiarity.
    adjust_relationship(world, to, from, &RelationshipDelta::new().familiarity(0.1));

    ShareOutcome {
        believed,
        newly_learned: newly && believed,
    }
}

/// Create a commitment between two actors. Returns its id.
#[allow(clippy::too_many_arguments)]
pub fn create_commitment(
    world: &mut World,
    events: &mut EventLog,
    tick: u64,
    minute: u32,
    kind: CommitmentKind,
    debtor: &ActorId,
    creditor: &ActorId,
    terms: impl Into<String>,
    predicate: QuestPredicate,
    deadline_minute: Option<u32>,
) -> CommitmentId {
    let id = CommitmentId(world.next_commitment_id);
    world.next_commitment_id += 1;
    let terms = terms.into();
    world.commitments.insert(
        id,
        Commitment {
            id,
            kind,
            debtor: debtor.clone(),
            creditor: creditor.clone(),
            terms: terms.clone(),
            predicate,
            state: CommitmentState::Pending,
            created_tick: tick,
            deadline_minute,
            resentment: if kind == CommitmentKind::Coerced {
                0.4
            } else {
                0.0
            },
            promised_fact: None,
        },
    );
    events.push(
        tick,
        minute,
        Some(creditor.clone()),
        EventKind::CommitmentCreated {
            commitment: id,
            parties: vec![debtor.clone(), creditor.clone()],
            terms,
        },
        Vec::new(),
        Vec::new(),
    );
    id
}

/// Evaluate all pending commitments; mark satisfied ones and log them.
/// Returns the ids that became satisfied this call.
pub fn satisfy_commitments(
    world: &mut World,
    events: &mut EventLog,
    tick: u64,
    minute: u32,
) -> Vec<CommitmentId> {
    let mut satisfied = Vec::new();
    let ids: Vec<CommitmentId> = world.commitments.keys().copied().collect();
    for id in ids {
        let done = {
            let Some(c) = world.commitments.get(&id) else {
                continue;
            };
            c.state == CommitmentState::Pending && evaluate_predicate(world, &c.predicate)
        };
        if done {
            if let Some(c) = world.commitments.get_mut(&id) {
                c.state = CommitmentState::Satisfied;
            }
            events.push(
                tick,
                minute,
                None,
                EventKind::CommitmentSatisfied { commitment: id },
                Vec::new(),
                Vec::new(),
            );
            satisfied.push(id);
        }
    }
    satisfied
}

/// Break a pending commitment. Raises the creditor's anger and lowers trust.
pub fn break_commitment(
    world: &mut World,
    events: &mut EventLog,
    tick: u64,
    minute: u32,
    id: CommitmentId,
    by: &ActorId,
) {
    let Some(c) = world.commitments.get_mut(&id) else {
        return;
    };
    if c.state != CommitmentState::Pending {
        return;
    }
    c.state = CommitmentState::Broken;
    let creditor = c.creditor.clone();
    let debtor = c.debtor.clone();
    adjust_relationship(
        world,
        &creditor,
        &debtor,
        &RelationshipDelta::new()
            .trust(-0.4)
            .anger(0.35)
            .respect(-0.2),
    );
    events.push(
        tick,
        minute,
        Some(by.clone()),
        EventKind::CommitmentBroken {
            commitment: id,
            by: by.clone(),
        },
        Vec::new(),
        Vec::new(),
    );
}

/// Register a new rumor. Returns its id.
#[allow(clippy::too_many_arguments)]
pub fn add_rumor(
    world: &mut World,
    events: &mut EventLog,
    tick: u64,
    minute: u32,
    source: &ActorId,
    text: impl Into<String>,
    about: Option<FactId>,
    truth: bool,
    confidence: f32,
) -> RumorId {
    let id = RumorId(world.next_rumor_id);
    world.next_rumor_id += 1;
    let text = text.into();
    world.rumors.push(Rumor {
        id,
        text: text.clone(),
        about,
        truth,
        source: source.clone(),
        chain: vec![source.clone()],
        confidence: confidence.clamp(0.0, 1.0),
        created_tick: tick,
        hops: 0,
    });
    events.push(
        tick,
        minute,
        Some(source.clone()),
        EventKind::RumorSpread {
            rumor: id,
            by: source.clone(),
            text,
        },
        Vec::new(),
        Vec::new(),
    );
    id
}

/// Propagate an existing rumor to another actor, potentially distorting the
/// confidence. Returns true if the rumor reached `to`.
///
/// The delivery passes through the shared acceptance mechanic with the
/// *actual reteller* as the direct source: the reteller passes on their own
/// belief polarity — never the rumor record's world truth — and the
/// recipient's belief is rolled (never forced), with a provenance-bound
/// evidence item on their ledger.
pub fn propagate_rumor(
    world: &mut World,
    events: &mut EventLog,
    rng: &mut DeterministicRng,
    tick: u64,
    minute: u32,
    id: RumorId,
    to: &ActorId,
) -> bool {
    let Some(r) = world.rumors.iter_mut().find(|r| r.id == id) else {
        return false;
    };
    if r.chain.contains(to) {
        return false;
    }
    let decay = 0.85 + 0.1 * rng.next_f32();
    r.confidence = (r.confidence * decay).clamp(0.05, 1.0);
    r.hops += 1;
    r.chain.push(to.clone());
    let (text, about, confidence) = (r.text.clone(), r.about.clone(), r.confidence);
    let speaker = r.chain[r.chain.len() - 2].clone();
    if let Some(fact) = &about {
        // The reteller's own conviction is what gets retold. When they hold
        // no belief at all they pass the claim on as heard (polarity true) at
        // the rumor's carried confidence.
        let (retold_polarity, reteller_confidence) = world
            .actor(&speaker)
            .and_then(|a| a.beliefs.get(fact))
            .map(|b| (b.believes_true, b.confidence))
            .unwrap_or((true, confidence));
        let rel = world
            .actor(to)
            .map(|a| a.relationship(&speaker))
            .unwrap_or_default();
        let appraisal = rpg_behavior::RecipientAppraisal {
            trust: rel.trust,
            suspicion: rel.suspicion,
            fear: rel.fear,
            assertiveness: 0.6,
        };
        let draft = rpg_behavior::AssertionDraft {
            speaker: Some(speaker.clone()),
            proposition: fact.clone(),
            asserted_polarity: retold_polarity,
            channel: rpg_domain::behavior_types::AcquisitionChannel::Told,
            origin_event: None,
            reliability: 0.6,
            retold_chain: None,
            deceptive: false,
        };
        let acceptance = rpg_behavior::accept_assertion(
            world,
            events,
            tick,
            minute,
            to,
            draft,
            appraisal,
            reteller_confidence,
            false,
            rng,
        );
        // Legacy snapshot so the rumor can keep propagating as a claim; the
        // polarity is the rolled acceptance outcome, never world truth.
        write_belief_snapshot(
            world,
            events,
            tick,
            minute,
            to,
            fact,
            acceptance.believed,
            if acceptance.believed {
                acceptance.probability
            } else {
                (1.0 - acceptance.probability).max(0.1)
            },
            Provenance::Told {
                by: speaker,
                event: acceptance.claim_event,
            },
        );
    }
    events.push(
        tick,
        minute,
        Some(to.clone()),
        EventKind::RumorSpread {
            rumor: id,
            by: to.clone(),
            text,
        },
        Vec::new(),
        vec![to.clone()],
    );
    true
}

// ---------------------------------------------------------------------------
// Emotional reactions used by action execution
// ---------------------------------------------------------------------------

/// Apply the social consequences of a threat.
pub fn apply_threat(world: &mut World, from: &ActorId, to: &ActorId) {
    let (risk, nerve) = world
        .actor(to)
        .map(|a| (a.personality.risk_tolerance, a.personality.neuroticism))
        .unwrap_or((0.5, 0.5));
    adjust_relationship(
        world,
        to,
        from,
        &RelationshipDelta::new()
            .fear(0.25 + 0.3 * (1.0 - risk) + 0.2 * nerve)
            .anger(0.2 + 0.2 * risk)
            .trust(-0.25)
            .respect(-0.1)
            .suspicion(0.2),
    );
}

/// Apply the social consequences of comforting someone.
pub fn apply_comfort(world: &mut World, from: &ActorId, to: &ActorId) {
    let (agree, recip) = world
        .actor(to)
        .map(|a| (a.personality.agreeableness, a.personality.reciprocity))
        .unwrap_or((0.5, 0.5));
    adjust_relationship(
        world,
        to,
        from,
        &RelationshipDelta::new()
            .sadness(-0.3)
            .affection(0.15 + 0.15 * agree)
            .trust(0.1 * recip)
            .obligation(0.1),
    );
}

/// Apply the social consequences of receiving a gift/payment.
pub fn apply_gift(world: &mut World, from: &ActorId, to: &ActorId, value: f32) {
    let greed = world
        .actor(to)
        .map(|a| {
            if a.personality.has_tag("greedy") {
                0.3
            } else {
                0.0
            }
        })
        .unwrap_or(0.0);
    let v = value.clamp(0.0, 1.0);
    adjust_relationship(
        world,
        to,
        from,
        &RelationshipDelta::new()
            .affection(0.2 * v + greed)
            .obligation(0.3 * v)
            .temptation(0.2 * v)
            .trust(0.05 * v),
    );
}

/// Apply the social consequences of being harmed by someone (known culprit).
pub fn apply_harm(world: &mut World, victim: &ActorId, culprit: &ActorId, severity: f32) {
    let s = severity.clamp(0.0, 1.0);
    adjust_relationship(
        world,
        victim,
        culprit,
        &RelationshipDelta::new()
            .anger(0.5 * s)
            .fear(0.3 * s)
            .trust(-0.6 * s)
            .affection(-0.4 * s)
            .respect(-0.2 * s)
            .suspicion(0.3 * s),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use rpg_domain::world::{
        Actor, ActorKind, ActorStatus, Capabilities, NavGrid, Personality, StoryState, WorldLore,
    };
    use std::collections::{BTreeMap, BTreeSet};

    fn world_with(actors: Vec<Actor>) -> World {
        let mut actors_map = BTreeMap::new();
        for a in actors {
            actors_map.insert(a.id.clone(), a);
        }
        World {
            tick: 0,
            day: 0,
            minute_of_day: 480,
            locations: BTreeMap::new(),
            actors: actors_map,
            items: BTreeMap::new(),
            factions: BTreeMap::new(),
            facts: BTreeMap::new(),
            nav: NavGrid::new(8, 8),
            story: StoryState {
                mode: rpg_domain::world::MysteryMode::FixedAnswer,
                anchors: Vec::new(),
                checkpoints: Vec::new(),
                fixed_answer: BTreeMap::new(),
                committed_containers: BTreeMap::new(),
            },
            commitments: BTreeMap::new(),
            offers: BTreeMap::new(),
            rumors: Vec::new(),
            damage: BTreeMap::new(),
            location_items: BTreeMap::new(),
            flags: BTreeSet::new(),
            ledger: BTreeMap::new(),
            next_commitment_id: 1,
            next_offer_id: 1,
            deals: BTreeMap::new(),
            next_deal_id: 1,
            next_rumor_id: 1,
            next_goal_id: 1,
            epistemics: Default::default(),
            lore: WorldLore::default(),
        }
    }

    fn actor(id: &str) -> Actor {
        Actor {
            id: ActorId::new(id),
            name: id.to_string(),
            kind: ActorKind::Npc,
            status: ActorStatus::Alive,
            location: LocationId::new("x"),
            pos: rpg_domain::geom::Vec2::ZERO,
            facing: 0.0,
            personality: Personality::default(),
            values: Vec::new(),
            biography: String::new(),
            private_biography: String::new(),
            affiliations: Vec::new(),
            capabilities: Capabilities::default(),
            inventory: Default::default(),
            goals: Vec::new(),
            schedule: Vec::new(),
            relationships: BTreeMap::new(),
            beliefs: BTreeMap::new(),
            inspected: BTreeSet::new(),
            flags: BTreeSet::new(),
            secret_allegiance: None,
            in_private_conversation_with: None,
            bargain: None,
            needs: Default::default(),
            emotions: Default::default(),
            rel_memory: BTreeMap::new(),
        }
    }

    #[test]
    fn relationship_deltas_clamp() {
        let mut w = world_with(vec![actor("a"), actor("b")]);
        for _ in 0..10 {
            adjust_relationship(
                &mut w,
                &ActorId::new("a"),
                &ActorId::new("b"),
                &RelationshipDelta::new().trust(1.0).anger(-1.0),
            );
        }
        let r = w
            .actor(&ActorId::new("a"))
            .unwrap()
            .relationship(&ActorId::new("b"));
        assert_eq!(r.trust, 1.0);
        assert_eq!(r.anger, -1.0);
    }

    #[test]
    fn threat_raises_fear_and_anger() {
        let mut w = world_with(vec![actor("a"), actor("b")]);
        apply_threat(&mut w, &ActorId::new("a"), &ActorId::new("b"));
        let r = w
            .actor(&ActorId::new("b"))
            .unwrap()
            .relationship(&ActorId::new("a"));
        assert!(r.fear > 0.0);
        assert!(r.anger > 0.0);
        assert!(r.trust < 0.0);
    }

    #[test]
    fn commitment_satisfied_by_predicate() {
        let mut w = world_with(vec![actor("a"), actor("b")]);
        let mut events = EventLog::new();
        let id = create_commitment(
            &mut w,
            &mut events,
            1,
            480,
            CommitmentKind::Promise,
            &ActorId::new("a"),
            &ActorId::new("b"),
            "raise the flag",
            QuestPredicate::Flag("done".into()),
            None,
        );
        assert!(satisfy_commitments(&mut w, &mut events, 2, 481).is_empty());
        w.set_flag("done");
        let done = satisfy_commitments(&mut w, &mut events, 3, 482);
        assert_eq!(done, vec![id]);
        assert_eq!(w.commitments[&id].state, CommitmentState::Satisfied);
    }

    #[test]
    fn rumor_propagation_is_tracked() {
        let mut w = world_with(vec![actor("a"), actor("b"), actor("c")]);
        let mut events = EventLog::new();
        let mut rng = DeterministicRng::new(1);
        let id = add_rumor(
            &mut w,
            &mut events,
            1,
            480,
            &ActorId::new("a"),
            "the grocer is a smuggler",
            None,
            false,
            0.9,
        );
        assert!(propagate_rumor(
            &mut w,
            &mut events,
            &mut rng,
            2,
            481,
            id,
            &ActorId::new("b")
        ));
        assert!(propagate_rumor(
            &mut w,
            &mut events,
            &mut rng,
            3,
            482,
            id,
            &ActorId::new("c")
        ));
        let r = w.rumors.iter().find(|r| r.id == id).unwrap();
        assert_eq!(r.hops, 2);
        assert_eq!(r.chain.len(), 3);
        // No duplicate delivery.
        assert!(!propagate_rumor(
            &mut w,
            &mut events,
            &mut rng,
            4,
            483,
            id,
            &ActorId::new("b")
        ));
    }

    /// Regression: ordinary (and even assertive) claims must be judged by the
    /// relationship-sensitive roll — the speaker's `assertive` flag must never
    /// guarantee belief. Delivery (hearing) is still always guaranteed.
    #[test]
    fn share_claim_ordinary_claims_are_probabilistic_even_when_assertive() {
        let seeds = 32;
        let mut believed_count = 0;
        for seed in 0..seeds {
            let mut w = world_with(vec![actor("a"), actor("b")]);
            let mut events = EventLog::new();
            let mut rng = DeterministicRng::new(seed);
            let out = share_claim(
                &mut w,
                &mut events,
                &mut rng,
                1,
                480,
                &ActorId::new("a"),
                &ActorId::new("b"),
                &FactId::new("f_x"),
                true,
            );
            if out.believed {
                believed_count += 1;
            }
            // Delivery is guaranteed regardless of the belief roll: the
            // recipient always records that it heard the claim.
            assert!(
                w.has_heard(&ActorId::new("b"), &FactId::new("f_x")),
                "the recipient must always hear the claim (seed {seed})"
            );
        }
        assert!(
            believed_count < seeds,
            "an assertive claim with a neutral relationship must not be believed deterministically"
        );
    }

    /// Regression: a distrusted, suspected speaker must be doubted — trust and
    /// suspicion must actually lower the belief probability.
    #[test]
    fn share_claim_distrusted_speakers_are_doubted() {
        let seeds = 32;
        let mut believed_count = 0;
        for seed in 0..seeds {
            let mut w = world_with(vec![actor("a"), actor("b")]);
            // The recipient `b` deeply distrusts and suspects the speaker `a`.
            adjust_relationship(
                &mut w,
                &ActorId::new("b"),
                &ActorId::new("a"),
                &RelationshipDelta::new().trust(-1.0).suspicion(1.0),
            );
            let mut events = EventLog::new();
            let mut rng = DeterministicRng::new(seed);
            let out = share_claim(
                &mut w,
                &mut events,
                &mut rng,
                1,
                480,
                &ActorId::new("a"),
                &ActorId::new("b"),
                &FactId::new("f_x"),
                true,
            );
            if out.believed {
                believed_count += 1;
            }
        }
        assert!(
            believed_count * 2 < seeds,
            "a distrusted speaker should be believed in well under half of the seeds, got {believed_count}/{seeds}"
        );
    }

    /// Regression: a fulfilled bargain must guarantee that the promised
    /// information is *delivered* (heard), not that the recipient believes it.
    /// Even when the belief roll fails, the belief record must exist.
    #[test]
    fn share_claim_delivers_even_when_not_believed() {
        let mut found_disbelief = false;
        for seed in 0..32 {
            let mut w = world_with(vec![actor("a"), actor("b")]);
            adjust_relationship(
                &mut w,
                &ActorId::new("b"),
                &ActorId::new("a"),
                &RelationshipDelta::new().trust(-1.0).suspicion(1.0),
            );
            let mut events = EventLog::new();
            let mut rng = DeterministicRng::new(seed);
            let out = share_claim(
                &mut w,
                &mut events,
                &mut rng,
                1,
                480,
                &ActorId::new("a"),
                &ActorId::new("b"),
                &FactId::new("f_x"),
                true,
            );
            if !out.believed {
                found_disbelief = true;
                assert!(
                    w.has_heard(&ActorId::new("b"), &FactId::new("f_x")),
                    "delivery must be guaranteed even when belief fails (seed {seed})"
                );
            }
        }
        assert!(
            found_disbelief,
            "expected at least one seed where a distrusted speaker is not believed"
        );
    }

    /// Regression (F8): rumor propagation passes through the shared
    /// acceptance mechanic with the *actual reteller* — the reteller's own
    /// belief polarity is what the recipient hears and records, never the
    /// rumor record's world truth.
    #[test]
    fn f8_propagate_rumor_retells_the_retellers_belief_not_world_truth() {
        let mut w = world_with(vec![actor("a"), actor("m"), actor("c")]);
        let fact = FactId::new("f_rumor");
        // The mid-chain reteller `m` heard the rumor and DISBELIEVES it;
        // the rumor record's world truth is TRUE.
        w.actor_mut(&ActorId::new("m")).unwrap().beliefs.insert(
            fact.clone(),
            rpg_domain::world::Belief {
                fact: fact.clone(),
                believes_true: false,
                confidence: 0.8,
                source: Provenance::Told {
                    by: ActorId::new("a"),
                    event: None,
                },
                acquired_tick: 0,
            },
        );
        // Seed the rumor record mid-transmission: originated by `a`, now
        // held by `m` (so `m` is the actual reteller of the next hop).
        let id = RumorId(w.next_rumor_id);
        w.next_rumor_id += 1;
        w.rumors.push(Rumor {
            id,
            text: "the grocer is a smuggler".into(),
            about: Some(fact.clone()),
            truth: true,
            source: ActorId::new("a"),
            chain: vec![ActorId::new("a"), ActorId::new("m")],
            confidence: 0.7,
            created_tick: 0,
            hops: 1,
        });

        let mut events = EventLog::new();
        let mut rng = DeterministicRng::new(5);
        assert!(propagate_rumor(
            &mut w,
            &mut events,
            &mut rng,
            1,
            480,
            id,
            &ActorId::new("c")
        ));

        // The recipient records provenance-bound told evidence whose asserted
        // polarity is the RETELLER's disbelief — never the world truth.
        let entries = w.epistemics.evidence_about(&ActorId::new("c"), &fact);
        assert_eq!(entries.len(), 1);
        assert!(
            !entries[0].asserted_polarity,
            "the recipient heard m's disbelief, not the world truth"
        );
        assert_eq!(
            entries[0].source_actor_id.as_ref().map(|a| a.as_str()),
            Some("m"),
            "the actual reteller is the evidence's source"
        );
        assert_eq!(
            entries[0].acquisition_channel,
            rpg_domain::behavior_types::AcquisitionChannel::Told
        );
        // The write is event-sourced: one EvidenceRecorded (causally linked
        // to the delivery) and one ClaimShared from the actual reteller.
        let claim = events
            .all()
            .iter()
            .find(|e| {
                matches!(
                    &e.kind,
                    EventKind::ClaimShared { from, to, .. }
                        if from.as_str() == "m" && to.as_str() == "c"
                )
            })
            .expect("the rumor delivery emits a ClaimShared event");
        let recorded = events
            .all()
            .iter()
            .find(|e| {
                matches!(
                    &e.kind,
                    EventKind::EvidenceRecorded { observer, fact: f, .. }
                        if observer.as_str() == "c" && f == &fact
                )
            })
            .expect("the rumor delivery records evidence");
        assert!(recorded.caused_by.contains(&claim.id));
        // The legacy snapshot follows the rolled acceptance, not world truth.
        let snapshot_believed = w
            .actor(&ActorId::new("c"))
            .unwrap()
            .beliefs
            .get(&fact)
            .unwrap()
            .believes_true;
        match &claim.kind {
            EventKind::ClaimShared { believed, .. } => {
                assert_eq!(
                    snapshot_believed, *believed,
                    "the snapshot polarity is the acceptance roll, never world truth"
                );
            }
            _ => unreachable!(),
        }
        // The reteller's own recorded disbelief survives (no world-truth
        // write ever touched their polarity).
        assert!(
            !w.actor(&ActorId::new("m"))
                .unwrap()
                .beliefs
                .get(&fact)
                .unwrap()
                .believes_true
        );
    }
}
