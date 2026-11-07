//! Behavior-engine Phase 3 production-integration tests (test-spec T3.7–
//! T3.12; plan I-4/I-4a/I-4b/I-3b): needs decay event-sourced in the real
//! step pipeline, the needs-differential ranking through the production
//! deal-response pathway, emotions written by `APPRAISE` in the authoritative
//! dispatch, sensitivity-altered threshold gating in maintenance, and
//! truthful appraisal provenance.

use rpg_domain::action::{Action, ActionProposal, ProposalSource, QuestPredicate};
use rpg_domain::deal::{Deal, DealEvent};
use rpg_domain::emotions::{EmotionKind, EmotionResponse};
use rpg_domain::events::EventKind;
use rpg_domain::geom::{Rect, Vec2};
use rpg_domain::ids::*;
use rpg_domain::needs::{NeedKind, NeedTuning, NeedsConfig, ThresholdDirection};
use rpg_domain::world::{
    Actor, ActorKind, ActorStatus, Capabilities, Fact, FactCategory, Goal, GoalKind, GoalStatus,
    Location, LocationKind, NavGrid, Personality, Sensitivity, StoryState, World,
};
use rpg_sim::{SimConfig, Simulation};
use std::collections::{BTreeMap, BTreeSet};

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

fn npc(id: &str, x: f32, risk: f32, neuroticism: f32, self_preservation: f32) -> Actor {
    let mut a = Actor {
        id: ActorId::new(id),
        name: id.to_string(),
        kind: ActorKind::Npc,
        status: ActorStatus::Alive,
        location: LocationId::new("plaza"),
        pos: Vec2::new(x, 1.0),
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
    };
    a.personality.risk_tolerance = risk;
    a.personality.neuroticism = neuroticism;
    a.personality.self_preservation = self_preservation;
    a
}

fn fixture(needs_decay: f32, emotions_decay: f32) -> (Simulation, SimConfig) {
    let nav = NavGrid::new(12, 12);
    let mut locations = BTreeMap::new();
    locations.insert(
        LocationId::new("plaza"),
        Location {
            id: LocationId::new("plaza"),
            name: "Plaza".into(),
            kind: LocationKind::Plaza,
            bounds: Rect::new(0.0, 0.0, 12.0, 12.0),
            owner: None,
            sound_transmission: 1.0,
            private: false,
            tags: vec![],
        },
    );
    let mut facts = BTreeMap::new();
    facts.insert(
        FactId::new("f_ledger"),
        Fact {
            id: FactId::new("f_ledger"),
            statement: "the ledger was altered".into(),
            truth: true,
            sensitivity: Sensitivity::Restricted,
            category: FactCategory::Observation,
            prerequisites: vec![],
            eligible_holders: vec![],
            eligible_containers: vec![],
            committed: false,
            placed_at: None,
        },
    );
    let mut actors = BTreeMap::new();
    actors.insert(ActorId::new(PLAYER), {
        let mut p = npc("player", 1.0, 0.5, 0.5, 0.5);
        p.kind = ActorKind::Player;
        p.id = ActorId::new(PLAYER);
        p
    });
    // Sensitive: risk-averse, neurotic, self-preserving.
    actors.insert(
        ActorId::new("sensitive"),
        npc("sensitive", 2.0, 0.0, 1.0, 1.0),
    );
    // Insensitive: fearless, stable, unconcerned with safety.
    actors.insert(ActorId::new("stoic"), npc("stoic", 3.0, 1.0, 0.0, 0.0));
    let mut config = SimConfig {
        enable_background: false,
        enable_director: false,
        enable_goal_autonomy: false,
        ..SimConfig::default()
    };
    let mut tuning = [NeedTuning::default(); 5];
    for t in tuning.iter_mut() {
        t.decay_per_tick = needs_decay;
    }
    config.needs = NeedsConfig { tuning };
    let mut decay = [0.01; 6];
    decay[EmotionKind::Fear.index()] = emotions_decay;
    config.emotions.decay_per_tick = decay;
    let world = World {
        tick: 0,
        day: 0,
        minute_of_day: 480,
        locations,
        actors,
        items: BTreeMap::new(),
        factions: BTreeMap::new(),
        facts,
        nav,
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
        lore: Default::default(),
    };
    let sim = Simulation::new(world, 2026, config.clone());
    (sim, config)
}

fn proposal(actor: &str, action: Action) -> ActionProposal {
    ActionProposal {
        actor: ActorId::new(actor),
        action,
        source: ProposalSource::Player,
        rationale: String::new(),
    }
}

fn threat_events(sim: &Simulation) -> Vec<&rpg_domain::events::Event> {
    sim.events
        .all()
        .iter()
        .filter(|e| matches!(e.kind, EventKind::ThreatMade { .. }))
        .collect()
}

// ---------------------------------------------------------------------------
// T3.7/T3.8 — needs decay in the real pipeline: typed threshold crossings,
// bounded goal-urgency raise with release
// ---------------------------------------------------------------------------

#[test]
fn t3_7_needs_decay_crossings_are_event_sourced_and_raise_goal_urgency() {
    let (mut sim, _config) = fixture(0.1, 0.01);
    // A wealth goal the need maps onto.
    sim.world
        .actor_mut(&ActorId::new("sensitive"))
        .unwrap()
        .goals
        .push(Goal {
            id: 1,
            description: "earn coin".into(),
            kind: GoalKind::Wealth,
            priority: 1.0,
            need_boost: 0.0,
            status: GoalStatus::Active,
            predicate: None,
            target_location: None,
            target_actor: None,
        });
    sim.world
        .actor_mut(&ActorId::new("sensitive"))
        .unwrap()
        .needs
        .set(NeedKind::Wealth, 0.4);

    let base_priority = sim.world.actor(&ActorId::new("sensitive")).unwrap().goals[0].priority;
    let _ = sim.step(vec![]);

    // The crossing was event-sourced…
    let crossed = sim.events.all().iter().any(|e| {
        matches!(&e.kind, EventKind::NeedThresholdCrossed { actor, need, direction, .. }
            if actor.as_str() == "sensitive" && *need == NeedKind::Wealth
                && *direction == ThresholdDirection::FellBelow)
    });
    assert!(crossed, "a typed NeedThresholdCrossed event was recorded");

    // …and the mapped goal urgency was raised, bounded and recorded.
    let actor = sim.world.actor(&ActorId::new("sensitive")).unwrap();
    assert!(
        actor.goals[0].priority > base_priority,
        "urgency raised the priority"
    );
    assert!(actor.goals[0].need_boost > 0.0);
    let raised = sim.events.all().iter().any(|e| {
        matches!(&e.kind, EventKind::GoalUrgencyRaised { actor, need, .. }
            if actor.as_str() == "sensitive" && *need == NeedKind::Wealth)
    });
    assert!(raised);
    let boost = actor.goals[0].need_boost;
    assert!(
        boost <= sim_world_needs_cap(),
        "urgency is bounded by the tuning cap"
    );

    // Latched: further decay fires no more *wealth* crossings (hysteresis in
    // the pipeline — no escalation while the need stays low).
    let events_before = sim.events.len();
    for _ in 0..5 {
        let _ = sim.step(vec![]);
    }
    let extra: Vec<_> = sim.events.all()[events_before..]
        .iter()
        .filter(|e| matches!(&e.kind, EventKind::NeedThresholdCrossed { actor, need, .. } if actor.as_str() == "sensitive" && *need == NeedKind::Wealth))
        .collect();
    assert!(extra.is_empty(), "no threshold ping-pong while latched");

    // Recovery releases exactly what was raised.
    sim.world
        .actor_mut(&ActorId::new("sensitive"))
        .unwrap()
        .needs
        .set(NeedKind::Wealth, 0.7);
    let _ = sim.step(vec![]);
    let actor = sim.world.actor(&ActorId::new("sensitive")).unwrap();
    assert!(
        (actor.goals[0].priority - base_priority).abs() < 1e-5,
        "boost removed"
    );
    assert_eq!(actor.goals[0].need_boost, 0.0);
    let released = sim.events.all().iter().any(|e| {
        matches!(&e.kind, EventKind::GoalUrgencyReleased { actor, need, .. }
            if actor.as_str() == "sensitive" && *need == NeedKind::Wealth)
    });
    assert!(released);
}

fn sim_world_needs_cap() -> f32 {
    NeedTuning::default().urgency_cap
}

/// The replay of a run that included needs maintenance reproduces the full
/// world and event log exactly (the classified background mutation must be
/// replay-stable, plan G4).
#[test]
fn t3_7_needs_maintenance_is_replay_stable() {
    let (mut sim, config) = fixture(0.05, 0.05);
    for _ in 0..40 {
        let _ = sim.step(vec![]);
    }
    let save = rpg_persistence::SaveFile::from_sim(&sim);
    let replayed = rpg_persistence::replay(&save, config);
    assert_eq!(
        serde_json::to_string(&sim.world).unwrap(),
        serde_json::to_string(&replayed.world).unwrap(),
        "needs/emotions/relationship maintenance must replay exactly"
    );
    assert_eq!(sim.events.checksum(), replayed.events.checksum());
}

// ---------------------------------------------------------------------------
// T3.9 — the production deal-response pathway ranks by the needs differential
// ---------------------------------------------------------------------------

#[test]
fn t3_9_production_deal_ranking_differs_by_need_profile() {
    let (mut sim, config) = fixture(0.0, 0.01);
    // Two actors hold the same evidence; each owes the player a due
    // disclosure. One is status-starved, one status-satiated.
    for id in ["sensitive", "stoic"] {
        let (tick, minute) = (sim.world.tick, sim.world.minute_of_day);
        rpg_social::record_belief(
            &mut sim.world,
            &mut sim.events,
            tick,
            minute,
            &ActorId::new(id),
            &FactId::new("f_ledger"),
            true,
            0.9,
            rpg_domain::world::Provenance::Assumed,
        );
        let debtor = ActorId::new(id);
        let did = DealId(sim.world.next_deal_id);
        sim.world.next_deal_id += 1;
        sim.world.deals.insert(
            did,
            Deal::proposed(
                did,
                debtor,
                ActorId::new(PLAYER),
                "bargain",
                QuestPredicate::Flag("unused".into()),
                QuestPredicate::HeardFact {
                    actor: ActorId::new(PLAYER),
                    fact: FactId::new("f_ledger"),
                },
                None,
                None,
                60,
                Vec::new(),
                0,
                Vec::new(),
                0,
            ),
        );
        let d = sim.world.deals.get_mut(&did).unwrap();
        let _ = d.advance(DealEvent::Accept, 0);
        let _ = d.advance(DealEvent::Activate, 0);
        let _ = d.advance(DealEvent::ConditionMet, 0);
        let _ = d.advance(DealEvent::ObligationDue, 0);
    }
    {
        let a = sim.world.actor_mut(&ActorId::new("sensitive")).unwrap();
        a.needs.set(NeedKind::Status, 0.05);
    }
    {
        let a = sim.world.actor_mut(&ActorId::new("stoic")).unwrap();
        a.needs.set(NeedKind::Status, 0.95);
    }
    let starved = rpg_agents::deal_response_options(
        &sim.world,
        &ActorId::new("sensitive"),
        None,
        None,
        &config.needs,
    )
    .unwrap();
    let satiated = rpg_agents::deal_response_options(
        &sim.world,
        &ActorId::new("stoic"),
        None,
        None,
        &config.needs,
    )
    .unwrap();
    let starved_honour = starved
        .ranking
        .ranked
        .iter()
        .find(|s| s.id == "honour_disclosure")
        .expect("honouring is feasible for the knower")
        .utility;
    let satiated_honour = satiated
        .ranking
        .ranked
        .iter()
        .find(|s| s.id == "honour_disclosure")
        .unwrap()
        .utility;
    assert!(
        starved_honour > satiated_honour,
        "the same option set ranks differently for different need profiles: \
         {starved_honour} (status-starved) vs {satiated_honour} (status-satiated)"
    );
    // The lexicographic walk-away is never ranked, in either profile.
    for r in [&starved.ranking, &satiated.ranking] {
        assert!(!r.ranked.iter().any(|s| s.id == "renege"));
        assert_eq!(r.unacceptable, vec!["renege".to_string()]);
    }
    // Both still honour first (the goal term dominates at these levels).
    assert_eq!(starved.ranking.ranked[0].id, "honour_disclosure");
    assert_eq!(satiated.ranking.ranked[0].id, "honour_disclosure");
}

// ---------------------------------------------------------------------------
// T3.10/T3.11/T3.12 — the authoritative threat path: APPRAISE writes
// emotions, gating responds to sensitivity, provenance is truthful
// ---------------------------------------------------------------------------

#[test]
fn t3_10_threat_appraise_writes_emotions_and_contextual_edge() {
    let (mut sim, _config) = fixture(0.0, 0.01);
    let out = sim.apply(&proposal(
        PLAYER,
        Action::Threaten {
            target: ActorId::new("sensitive"),
            demand: "the ledger".into(),
        },
    ));
    assert!(matches!(
        out,
        rpg_domain::action::ValidationOutcome::Accepted
    ));
    // The APPRAISE effect is in the typed outcome…
    let outcome = sim.outcomes.last().unwrap();
    assert!(outcome.effects.iter().any(|e| e.family() == "appraise"));
    // …and its application wrote the target's emotion state (mind-local).
    let actor = sim.world.actor(&ActorId::new("sensitive")).unwrap();
    assert!(actor.emotions.value(EmotionKind::Fear) > 0.0);
    assert!(actor.emotions.value(EmotionKind::Anger) > 0.0);
    // The contextual edge moved on its own dimensions; the legacy aggregate
    // trust is untouched by the new pathway.
    let rel = actor.relationship(&ActorId::new(PLAYER));
    assert!(rel.fear > 0.0 && rel.anger > 0.0);
    assert!(rel.trust_benevolence < 0.0);
    // The reverse edge stayed untouched (directed).
    let reverse = sim
        .world
        .actor(&ActorId::new(PLAYER))
        .unwrap()
        .relationship(&ActorId::new("sensitive"));
    assert_eq!(reverse.fear, 0.0);
    assert_eq!(reverse.anger, 0.0);
}

#[test]
fn t3_11_sensitivity_gates_flee_only_for_the_sensitive_actor() {
    let (mut sim, _config) = fixture(0.0, 0.01);
    let _ = sim.apply(&proposal(
        PLAYER,
        Action::Threaten {
            target: ActorId::new("sensitive"),
            demand: "the ledger".into(),
        },
    ));
    let _ = sim.apply(&proposal(
        PLAYER,
        Action::Threaten {
            target: ActorId::new("stoic"),
            demand: "the ledger".into(),
        },
    ));
    let report = sim.step(vec![]);
    let mut fled: Vec<&str> = Vec::new();
    for (actor, response) in &report.gated_responses {
        if *response == EmotionResponse::Flee {
            fled.push(actor.as_str());
        }
    }
    assert!(
        fled.contains(&"sensitive"),
        "the sensitive actor crosses the flee threshold"
    );
    assert!(
        !fled.contains(&"stoic"),
        "the stoic actor does not cross from the same event (sensitivity alters the crossing point)"
    );
}

#[test]
fn t3_11b_emotions_decay_back_below_threshold_in_the_pipeline() {
    let (mut sim, _config) = fixture(0.0, 0.3);
    let _ = sim.apply(&proposal(
        PLAYER,
        Action::Threaten {
            target: ActorId::new("sensitive"),
            demand: "the ledger".into(),
        },
    ));
    let _ = sim.step(vec![]);
    assert!(
        sim.world
            .actor(&ActorId::new("sensitive"))
            .unwrap()
            .emotions
            .value(EmotionKind::Fear)
            > 0.0
    );
    for _ in 0..5 {
        let _ = sim.step(vec![]);
    }
    let fear = sim
        .world
        .actor(&ActorId::new("sensitive"))
        .unwrap()
        .emotions
        .value(EmotionKind::Fear);
    assert!(
        fear < SimConfig::default().emotions.thresholds.flee_fear,
        "decayed back below the flee threshold in the pipeline: {fear}"
    );
}

#[test]
fn t3_12_appraisal_provenance_is_event_sourced_and_truthful() {
    let (mut sim, _config) = fixture(0.0, 0.01);
    let _ = sim.apply(&proposal(
        PLAYER,
        Action::Threaten {
            target: ActorId::new("sensitive"),
            demand: "the ledger".into(),
        },
    ));
    let threats = threat_events(&sim);
    assert_eq!(threats.len(), 1);
    let threat_id = threats[0].id;
    // The appraisal event cites the threat as its cause (truthful trace)…
    let appraisal = sim
        .events
        .all()
        .iter()
        .find(|e| matches!(e.kind, EventKind::AppraisalRecorded { .. }))
        .expect("the mind-local appraisal was event-sourced");
    assert_eq!(appraisal.caused_by, vec![threat_id]);
    let EventKind::AppraisalRecorded {
        subject,
        regarding,
        interpretation,
        ..
    } = &appraisal.kind
    else {
        unreachable!()
    };
    assert_eq!(subject.as_str(), "sensitive");
    assert_eq!(regarding.as_str(), PLAYER);
    assert!(!interpretation.is_empty(), "the interpretation is recorded");
    // …and the actor's memory carries the same provenance (T3.6 in the
    // production path).
    let memory = sim
        .world
        .actor(subject)
        .unwrap()
        .rel_memory
        .get(regarding)
        .and_then(|v| v.last())
        .unwrap();
    assert_eq!(memory.event, Some(threat_id));
    assert_eq!(memory.interpretation, *interpretation);
}
