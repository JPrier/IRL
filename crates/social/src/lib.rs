//! # rpg_social
//!
//! Social causality: multi-dimensional relationships, provenance-aware beliefs,
//! commitments/obligations, and rumor propagation.
//!
//! All functions are deterministic given the same [`DeterministicRng`] state
//! and mutate only canonical [`World`] state (plus the append-only event log).

use rpg_domain::action::QuestPredicate;
use rpg_domain::events::{EventKind, EventLog};
use rpg_domain::ids::*;
use rpg_domain::rng::DeterministicRng;
use rpg_domain::social::{Commitment, CommitmentKind, CommitmentState, Rumor};
use rpg_domain::world::{evaluate_predicate, Belief, Provenance, World};

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

/// Outcome of one actor telling another a fact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShareOutcome {
    /// Did the recipient come to believe the claim?
    pub believed: bool,
    /// Was this new information to the recipient?
    pub newly_learned: bool,
}

/// Record a belief on an actor. Returns true if the belief is new or changed
/// from disbelief to belief.
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

/// `from` tells `to` about `fact`. The recipient's trust, the speaker's
/// assertiveness and existing suspicion determine whether it is believed.
///
/// The speaker must already believe the fact (or be lying deliberately); the
/// caller is responsible for validating that. Returns what happened.
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
    let speaker_confidence = world
        .actor(from)
        .and_then(|a| a.beliefs.get(fact))
        .map(|b| b.confidence)
        .unwrap_or(0.3);

    let rel = world
        .actor(to)
        .map(|a| a.relationship(from))
        .unwrap_or_default();
    let trust = (1.0 + rel.trust) * 0.5; // 0..1
    let suspicion = (1.0 + rel.suspicion) * 0.5;
    let assertiveness = if assertive { 1.0 } else { 0.6 };

    // A confident trusted friend is believed; a distrusted one is doubted.
    let mut p = 0.25 + 0.45 * trust + 0.25 * assertiveness + 0.2 * speaker_confidence
        - 0.35 * suspicion
        - 0.15 * rel.fear;
    p = p.clamp(0.05, 0.97);
    let believed = rng.chance(p);
    // Even when not fully believed, the recipient records an uncertain belief
    // with provenance so the rumor can still propagate as a claim.
    let newly = record_belief(
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
            event: None,
        },
    );

    events.push(
        tick,
        minute,
        Some(from.clone()),
        EventKind::ClaimShared {
            from: from.clone(),
            to: to.clone(),
            fact: fact.clone(),
            assertive,
            believed,
        },
        Vec::new(),
        vec![to.clone()],
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
    let (text, about, truth, confidence) = (r.text.clone(), r.about.clone(), r.truth, r.confidence);
    let speaker = r.chain[r.chain.len() - 2].clone();
    if let Some(fact) = &about {
        record_belief(
            world,
            events,
            tick,
            minute,
            to,
            fact,
            truth,
            confidence,
            Provenance::Told {
                by: speaker,
                event: None,
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
        Actor, ActorKind, ActorStatus, Capabilities, NavGrid, Personality, StoryState,
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
            next_rumor_id: 1,
            next_goal_id: 1,
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
}
