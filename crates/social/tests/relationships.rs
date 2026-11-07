//! Phase 3 directed contextual relationship acceptance tests (test-spec
//! T3.1–T3.6; plan I-4) through the production pathway the sim dispatches:
//! `apply_appraised_event` (directed, contextual, saturating, provenance-
//! carrying) and the maintenance decay pass.

use rpg_behavior::AppraisedEvent;
use rpg_domain::emotions::{EmotionKind, EmotionsConfig};
use rpg_domain::ids::*;
use rpg_domain::needs::NeedsConfig;
use rpg_domain::world::{
    Actor, ActorKind, ActorStatus, Capabilities, Personality, Relationship, World,
};
use rpg_social::{apply_appraised_event, decay_relationships};
use std::collections::{BTreeMap, BTreeSet};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn bare_actor(id: &str) -> Actor {
    Actor {
        id: ActorId::new(id),
        name: id.to_string(),
        kind: ActorKind::Npc,
        status: ActorStatus::Alive,
        location: LocationId::new("room"),
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

fn world_with(actors: &[&str]) -> World {
    let mut map = BTreeMap::new();
    for id in actors {
        let a = bare_actor(id);
        map.insert(a.id.clone(), a);
    }
    World {
        tick: 10,
        day: 0,
        minute_of_day: 480,
        locations: BTreeMap::new(),
        actors: map,
        items: BTreeMap::new(),
        factions: BTreeMap::new(),
        facts: BTreeMap::new(),
        nav: rpg_domain::world::NavGrid::new(8, 8),
        story: rpg_domain::world::StoryState {
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
        lore: rpg_domain::world::WorldLore::default(),
    }
}

fn appraised(
    world: &mut World,
    subject: &str,
    regarding: &str,
    event: AppraisedEvent,
    origin: Option<EventId>,
) -> rpg_social::AppraisalOutcome {
    apply_appraised_event(
        world,
        &ActorId::new(subject),
        &ActorId::new(regarding),
        event,
        origin,
        &EmotionsConfig::default(),
        &NeedsConfig::default(),
    )
    .expect("both actors exist")
}

// ---------------------------------------------------------------------------
// T3.1 — directed edges: Rel(i,j) ≠ Rel(j,i) after a one-sided event
// ---------------------------------------------------------------------------

#[test]
fn t3_1_one_sided_event_moves_only_the_observer_s_edge() {
    let mut w = world_with(&["i", "j"]);
    let before_j_of_i = w
        .actor(&ActorId::new("j"))
        .unwrap()
        .relationship(&ActorId::new("i"));
    appraised(
        &mut w,
        "i",
        "j",
        AppraisedEvent::GiftReceived { value: 0.5 },
        None,
    );
    let i_of_j = w
        .actor(&ActorId::new("i"))
        .unwrap()
        .relationship(&ActorId::new("j"));
    let j_of_i = w
        .actor(&ActorId::new("j"))
        .unwrap()
        .relationship(&ActorId::new("i"));
    // The observer's edge toward the other changed…
    assert!(
        i_of_j.affection > 0.0 && i_of_j.obligation > 0.0,
        "Rel(i,j) moved after i received a gift from j"
    );
    // …and the reverse edge is untouched (directed, not a global score).
    assert_eq!(
        j_of_i, before_j_of_i,
        "Rel(j,i) must not move: relationships are directed edges"
    );
    assert!(i_of_j != j_of_i);
}

// ---------------------------------------------------------------------------
// T3.2 — contextual trust: trusting a mechanic ≠ trusting with a secret
// (dimension separation)
// ---------------------------------------------------------------------------

#[test]
fn t3_2_accuracy_events_never_move_benevolence_and_vice_versa() {
    let mut w = world_with(&["i", "j"]);
    // A verified-accurate claim: trust-in-accuracy rises…
    appraised(&mut w, "i", "j", AppraisedEvent::AccurateClaim, None);
    let rel = w
        .actor(&ActorId::new("i"))
        .unwrap()
        .relationship(&ActorId::new("j"));
    assert!(rel.trust_accuracy > 0.0, "accuracy trust rose");
    assert_eq!(
        rel.trust_benevolence, 0.0,
        "trusting someone's accuracy does NOT imply trusting them with a secret"
    );

    // …and a kept promise (benevolence event) moves benevolence, not a
    // blanket accuracy credit.
    let mut w2 = world_with(&["i", "j"]);
    appraised(&mut w2, "i", "j", AppraisedEvent::PromiseKept, None);
    let rel2 = w2
        .actor(&ActorId::new("i"))
        .unwrap()
        .relationship(&ActorId::new("j"));
    assert!(rel2.trust_benevolence > rel2.trust_accuracy);
}

// ---------------------------------------------------------------------------
// T3.3 — no universal deltas: the same event yields different ΔR for
// different value profiles
// ---------------------------------------------------------------------------

#[test]
fn t3_3_same_event_yields_different_deltas_for_different_values() {
    // A greedy actor and a proud-AND-greedy actor (greed held constant —
    // only the pride axis differs) receive the SAME gift from the same
    // donor.
    let mut w = world_with(&["i", "j"]);
    {
        let a = w.actor_mut(&ActorId::new("i")).unwrap();
        a.personality.tags.push("greedy".to_string());
    }
    let out_greedy = appraised(
        &mut w,
        "i",
        "j",
        AppraisedEvent::GiftReceived { value: 0.8 },
        None,
    );

    let mut w2 = world_with(&["i", "j"]);
    {
        let a = w2.actor_mut(&ActorId::new("i")).unwrap();
        a.personality.tags.push("proud".to_string());
    }
    let out_proud = appraised(
        &mut w2,
        "i",
        "j",
        AppraisedEvent::GiftReceived { value: 0.8 },
        None,
    );

    // The proud actor reads the same gift as (partial) humiliation/bribery:
    // strictly less affection, strictly more negative respect.
    assert!(
        out_greedy.delta.affection > out_proud.delta.affection,
        "same event, different Δaffection: greedy {} vs proud {}",
        out_greedy.delta.affection,
        out_proud.delta.affection
    );
    assert!(
        out_greedy.delta.respect > out_proud.delta.respect,
        "the proud actor's respect falls relative to the greedy one: {} vs {}",
        out_greedy.delta.respect,
        out_proud.delta.respect
    );
    // And the interpretations differ — the appraised *meaning* is personal.
    assert_ne!(out_greedy.interpretation, out_proud.interpretation);
}

// ---------------------------------------------------------------------------
// T3.4 — saturation + decay
// ---------------------------------------------------------------------------

#[test]
fn t3_4_repeated_identical_events_asymptote() {
    let mut w = world_with(&["i", "j"]);
    let mut last = 0.0;
    for _ in 0..40 {
        let out = appraised(&mut w, "i", "j", AppraisedEvent::PromiseKept, None);
        let rel = w
            .actor(&ActorId::new("i"))
            .unwrap()
            .relationship(&ActorId::new("j"));
        assert!(rel.trust_benevolence >= last - 1e-6);
        last = rel.trust_benevolence;
        assert!(
            rel.trust_benevolence <= 1.0,
            "never past the bound ({} after {} events)",
            rel.trust_benevolence,
            40
        );
        let _ = out;
    }
    // Asymptote: the last few steps add almost nothing.
    let mut w2 = world_with(&["i", "j"]);
    for _ in 0..60 {
        let _ = appraised(&mut w2, "i", "j", AppraisedEvent::PromiseKept, None);
    }
    let mut w3 = world_with(&["i", "j"]);
    for _ in 0..120 {
        let _ = appraised(&mut w3, "i", "j", AppraisedEvent::PromiseKept, None);
    }
    let a = w2
        .actor(&ActorId::new("i"))
        .unwrap()
        .relationship(&ActorId::new("j"))
        .trust_benevolence;
    let b = w3
        .actor(&ActorId::new("i"))
        .unwrap()
        .relationship(&ActorId::new("j"))
        .trust_benevolence;
    assert!(
        b - a < 0.05,
        "doubling the identical events adds almost nothing: {a} vs {b}"
    );
}

#[test]
fn t3_4b_unused_dimensions_decay_at_configured_rates() {
    let mut w = world_with(&["i", "j"]);
    appraised(
        &mut w,
        "i",
        "j",
        AppraisedEvent::ThreatReceived { severity: 1.0 },
        None,
    );
    let after = w
        .actor(&ActorId::new("i"))
        .unwrap()
        .relationship(&ActorId::new("j"));
    assert!(after.fear > 0.0 && after.anger > 0.0);
    // Maintenance: feelings cool; accuracy/benevolence facts barely move;
    // familiarity and perceived power stay put (rate 0 by configuration).
    let mut decayed_after_ticks = after.clone();
    for _ in 0..50 {
        decay_relationships(&mut w, &rpg_behavior::DecayRates::default(), 0.0);
        decayed_after_ticks = w
            .actor(&ActorId::new("i"))
            .unwrap()
            .relationship(&ActorId::new("j"));
    }
    assert!(
        decayed_after_ticks.anger < after.anger && decayed_after_ticks.fear < after.fear,
        "unused dimensions decay at their configured rates"
    );
    assert_eq!(decayed_after_ticks.familiarity, after.familiarity);
    assert_eq!(decayed_after_ticks.perceived_power, after.perceived_power);
    assert!((decayed_after_ticks.trust_accuracy - after.trust_accuracy).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// T3.5 — forgiveness/retaliation: both paths reachable depending on state
// ---------------------------------------------------------------------------

#[test]
fn t3_5_forgiveness_and_retaliation_both_reachable_by_state() {
    // Retaliation: high anger + a reciprocating profile.
    let mut vengeful = world_with(&["i", "j"]);
    {
        let a = vengeful.actor_mut(&ActorId::new("i")).unwrap();
        a.personality.reciprocity = 0.9;
        a.personality.agreeableness = 0.2;
    }
    let _ = appraised(
        &mut vengeful,
        "i",
        "j",
        AppraisedEvent::HarmInflicted { severity: 1.0 },
        None,
    );
    let edge = rpg_behavior::ContextualEdge::from_relationship(
        &vengeful
            .actor(&ActorId::new("i"))
            .unwrap()
            .relationship(&ActorId::new("j")),
    );
    let profile = rpg_social::appraisal_profile(
        vengeful.actor(&ActorId::new("i")).unwrap(),
        &NeedsConfig::default(),
    );
    assert_eq!(
        rpg_behavior::aftermath(
            &edge,
            &profile,
            &rpg_behavior::AftermathThresholds::default()
        ),
        rpg_behavior::Aftermath::Retaliate,
        "the retaliation path is reachable from a vengeful state"
    );

    // Forgiveness: anger cooled, the tie was warm (a history of praise).
    let mut warm = world_with(&["i", "j"]);
    {
        let a = warm.actor_mut(&ActorId::new("i")).unwrap();
        a.personality.reciprocity = 0.2;
        a.personality.agreeableness = 0.9;
    }
    for _ in 0..3 {
        let _ = appraised(&mut warm, "i", "j", AppraisedEvent::PraiseReceived, None);
    }
    let _ = appraised(
        &mut warm,
        "i",
        "j",
        AppraisedEvent::InsultReceived { severity: 0.2 },
        None,
    );
    let edge = rpg_behavior::ContextualEdge::from_relationship(
        &warm
            .actor(&ActorId::new("i"))
            .unwrap()
            .relationship(&ActorId::new("j")),
    );
    let profile = rpg_social::appraisal_profile(
        warm.actor(&ActorId::new("i")).unwrap(),
        &NeedsConfig::default(),
    );
    let outcome = rpg_behavior::aftermath(
        &edge,
        &profile,
        &rpg_behavior::AftermathThresholds::default(),
    );
    assert_eq!(
        outcome,
        rpg_behavior::Aftermath::Forgive,
        "the forgiveness path is reachable from a warm, low-anger state"
    );
    // And the forgiveness path is *effective* in maintenance: anger cools
    // faster under it.
    let before_anger = edge.anger;
    let mut rates = rpg_behavior::DecayRates::default();
    rates.anger = rpg_behavior::aftermath_anger_decay(rates.anger, outcome, &profile);
    assert!(
        rates.anger > rpg_behavior::DecayRates::default().anger,
        "forgiveness accelerates the anger decay"
    );
    let _ = before_anger;
}

// ---------------------------------------------------------------------------
// T3.6 — appraisal provenance: each delta references the event id + the
// interpretation
// ---------------------------------------------------------------------------

#[test]
fn t3_6_appraisal_provenance_records_event_and_interpretation() {
    let mut w = world_with(&["i", "j"]);
    let out = appraised(
        &mut w,
        "i",
        "j",
        AppraisedEvent::GiftReceived { value: 0.5 },
        Some(EventId(7)),
    );
    assert!(!out.delta.is_zero(), "the event moved the edge");
    assert!(!out.interpretation.is_empty());
    let memory = w
        .actor(&ActorId::new("i"))
        .unwrap()
        .rel_memory
        .get(&ActorId::new("j"))
        .and_then(|v| v.last())
        .expect("an appraisal memory was recorded");
    assert_eq!(
        memory.event,
        Some(EventId(7)),
        "the delta references its event id"
    );
    assert_eq!(memory.interpretation, out.interpretation);
    assert!(memory.salience > 0.0);
    // A second appraisal appends a second provenance entry (a history, not
    // an overwrite).
    let _ = appraised(
        &mut w,
        "i",
        "j",
        AppraisedEvent::DeceptionExposed,
        Some(EventId(8)),
    );
    let entries = w
        .actor(&ActorId::new("i"))
        .unwrap()
        .rel_memory
        .get(&ActorId::new("j"))
        .unwrap();
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[1].event, Some(EventId(8)));
    // Memory salience decays in maintenance and is pruned when spent.
    for _ in 0..600 {
        decay_relationships(&mut w, &rpg_behavior::DecayRates::default(), 0.01);
    }
    let entries = w
        .actor(&ActorId::new("i"))
        .unwrap()
        .rel_memory
        .get(&ActorId::new("j"))
        .cloned()
        .unwrap_or_default();
    assert!(
        entries.iter().all(|m| m.salience > 0.0),
        "spent memories are pruned, not zeroed"
    );
    assert!(
        entries.is_empty() || entries.len() < 2,
        "low-salience memory decays away"
    );
}

// ---------------------------------------------------------------------------
// T3.12 (production side) — emotions ≠ relationships through the pathway
// ---------------------------------------------------------------------------

#[test]
fn t3_12_appraisal_writes_emotions_and_edge_as_separate_states() {
    let mut w = world_with(&["i", "j"]);
    let _ = appraised(
        &mut w,
        "i",
        "j",
        AppraisedEvent::ThreatReceived { severity: 1.0 },
        None,
    );
    let actor = w.actor(&ActorId::new("i")).unwrap();
    let rel = actor.relationship(&ActorId::new("j"));
    assert!(rel.fear > 0.0, "the edge's fear dimension moved");
    assert!(
        actor.emotions.value(EmotionKind::Fear) > 0.0,
        "the emotion state moved"
    );
    // Both states carry their own values; neither implies the other.
    let emotions_only = actor.emotions.value(EmotionKind::Fear);
    assert!(emotions_only > 0.0 && rel.fear > 0.0);
    // The reverse edge is untouched by the one-sided appraisal.
    let reverse = w
        .actor(&ActorId::new("j"))
        .unwrap()
        .relationship(&ActorId::new("i"));
    assert_eq!(reverse, Relationship::default());
}
