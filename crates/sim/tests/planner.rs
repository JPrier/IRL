//! Behavior-engine Phase 4a (plan I-6, I6a slice) acceptance tests, exercised
//! through the *real* `Simulation::step` pipeline — never through helpers
//! alone:
//!
//! * bounded event-driven planning/activation (fair, capped, persistent
//!   plans; relevant-event replanning only — the adversarial unrelated-event
//!   regression included);
//! * seeded utility selection wired into the recorded production decision
//!   stream (replay + save/load continuity);
//! * real needs/emotion-gate consumption in production action selection, with
//!   in-world need satisfaction through real validated actions and truthful
//!   typed `NeedChanged` events;
//! * agents-side QUERY / SANCTION / EXCUSE_DEAL candidates grounded in
//!   actor-local knowledge, authorization and real contract state.

use rpg_behavior::{record_provenance_evidence, UtilityOption};
use rpg_domain::action::{Action, ActionProposal, ProposalSource, QuestPredicate, SanctionKind};
use rpg_domain::deal::{DealState, Party};
use rpg_domain::emotions::EmotionKind;
use rpg_domain::events::EventKind;
use rpg_domain::geom::{Rect, Vec2};
use rpg_domain::ids::*;
use rpg_domain::needs::NeedKind;
use rpg_domain::world::{
    Actor, ActorKind, ActorStatus, Belief, Capabilities, Fact, FactCategory, Item, Location,
    LocationKind, NavGrid, Personality, Provenance, RelMemory, Sensitivity, StoryState, World,
};
use rpg_sim::{SimConfig, Simulation};
use std::collections::{BTreeMap, BTreeSet};

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

const HALL: &str = "hall";
const MARKET: &str = "market";
const DOCKS: &str = "docks";

fn actor(id: &str, room: &str, x: f32) -> Actor {
    Actor {
        id: ActorId::new(id),
        name: id.to_string(),
        kind: ActorKind::Npc,
        status: ActorStatus::Alive,
        location: LocationId::new(room),
        pos: Vec2::new(x, 5.0),
        facing: 0.0,
        personality: Personality::default(),
        values: Vec::new(),
        biography: String::new(),
        private_biography: String::new(),
        affiliations: vec![],
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

fn a(id: &str) -> ActorId {
    ActorId::new(id)
}

/// Three rooms on one walkable map; NPCs placed at each room's center.
fn world_fixture() -> World {
    let nav = NavGrid::new(40, 12);
    let mut locations = BTreeMap::new();
    for (id, x) in [(HALL, 0.0), (MARKET, 14.0), (DOCKS, 28.0)] {
        locations.insert(
            LocationId::new(id),
            Location {
                id: LocationId::new(id),
                name: id.to_string(),
                kind: LocationKind::Plaza,
                bounds: Rect::new(x, 0.0, 10.0, 10.0),
                owner: None,
                sound_transmission: 1.0,
                private: false,
                tags: vec![],
            },
        );
    }
    let mut actors = BTreeMap::new();
    actors.insert(a("n1").clone(), actor("n1", HALL, 5.0));
    actors.insert(a("n2").clone(), actor("n2", HALL, 6.5));
    actors.insert(a("n3").clone(), actor("n3", HALL, 8.0));
    World {
        tick: 0,
        day: 0,
        minute_of_day: 480,
        locations,
        actors,
        items: BTreeMap::new(),
        factions: BTreeMap::new(),
        facts: BTreeMap::new(),
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
        deals: BTreeMap::new(),
        rumors: Vec::new(),
        damage: BTreeMap::new(),
        damage_facts: BTreeMap::new(),
        location_items: BTreeMap::new(),
        flags: BTreeSet::new(),
        ledger: BTreeMap::new(),
        next_commitment_id: 1,
        next_offer_id: 1,
        next_deal_id: 1,
        next_rumor_id: 1,
        next_goal_id: 1,
        epistemics: Default::default(),
        lore: Default::default(),
    }
}

/// The planner pass enabled, everything else quiet; budget 1 activation per
/// step by default.
fn config() -> SimConfig {
    SimConfig {
        enable_background: false,
        enable_director: false,
        enable_goal_autonomy: false,
        ..SimConfig::default()
    }
}

fn sim(world: World) -> Simulation {
    Simulation::new(world, 2026, config())
}

fn apply(sim: &mut Simulation, actor: &str, action: Action) {
    let outcome = sim.apply(&ActionProposal {
        actor: a(actor),
        action,
        source: ProposalSource::Player,
        rationale: String::new(),
    });
    assert!(
        !matches!(outcome, rpg_domain::action::ValidationOutcome::Rejected(_)),
        "fixture action must apply"
    );
}

fn fear_of(sim: &mut Simulation, id: &str, amount: f32) {
    sim.world
        .actor_mut(&a(id))
        .unwrap()
        .emotions
        .adjust(EmotionKind::Fear, amount, 1.0);
}

fn add_fact(w: &mut World, id: &str, category: FactCategory, sensitivity: Sensitivity) {
    w.facts.insert(
        FactId::new(id),
        Fact {
            id: FactId::new(id),
            statement: format!("fact {id}"),
            truth: true,
            sensitivity,
            category,
            prerequisites: vec![],
            eligible_holders: vec![],
            eligible_containers: vec![],
            committed: false,
            placed_at: None,
        },
    );
}

fn last_event(
    sim: &Simulation,
    pred: impl Fn(&EventKind) -> bool,
) -> Option<&rpg_domain::events::Event> {
    sim.events.all().iter().rev().find(|e| pred(&e.kind))
}

/// Fixture knowledge: the actor holds (believes) the fact, so it may
/// lawfully disclose it (a claim needs knowledge AND permission).
fn seed_belief(w: &mut World, holder: &str, fact: &str) {
    w.actor_mut(&a(holder)).unwrap().beliefs.insert(
        FactId::new(fact),
        Belief {
            fact: FactId::new(fact),
            believes_true: true,
            confidence: 0.9,
            source: Provenance::Assumed,
            acquired_tick: 0,
        },
    );
}

// ---------------------------------------------------------------------------
// T4.1a (adversarial): unrelated events neither replan the planner's actors
// nor exceed the activation budget
// ---------------------------------------------------------------------------

#[test]
fn t4_1a_unrelated_events_do_not_replan_the_plan_holder() {
    let mut s = sim(world_fixture());
    // n1 is frightened into a flee plan; n2/n3 are the unrelated pair.
    fear_of(&mut s, "n1", 0.8);
    s.world
        .actor_mut(&a("n1"))
        .unwrap()
        .relationship_mut(&a("bully"))
        .fear = 0.9;
    s.world
        .actors
        .insert(a("bully").clone(), actor("bully", HALL, 4.0));
    add_fact(
        &mut s.world,
        "f_other",
        FactCategory::Observation,
        Sensitivity::Public,
    );
    seed_belief(&mut s.world, "n2", "f_other");

    // Step 1: the fear gate adopts the flee plan.
    s.step(Vec::new());
    let plan = s
        .planner
        .plans
        .get(&a("n1"))
        .expect("flee plan adopted")
        .clone();
    assert!(plan.intention.option.starts_with("flee:"));
    // The step's proposal was the plan's own action.
    assert!(matches!(plan.action, Action::MoveTo { .. }));

    // Let the fear gate fall back below its threshold: from here on, n1's
    // only trigger is its own persistent plan, so any replanning could only
    // have come from the unrelated events under test.
    let fear = rpg_domain::emotions::EmotionKind::Fear;
    let current = s.world.actor(&a("n1")).unwrap().emotions.value(fear);
    s.world
        .actor_mut(&a("n1"))
        .unwrap()
        .emotions
        .adjust(fear, -(current + 0.1), 1.0);
    assert!(
        s.world
            .actor(&a("n1"))
            .unwrap()
            .emotions
            .responses(&s.config.emotions)
            .is_empty(),
        "fixture: the gate must be closed for the adversarial window"
    );
    // Drain the pre-window gate signal exactly as the next natural
    // activation would (the gate closed above).
    s.planner.take_signals(&a("n1"));

    // Unrelated events: n2 tells n3 a claim (a signal for n3 only), twice.
    for _ in 0..2 {
        apply(
            &mut s,
            "n2",
            Action::ShareClaim {
                target: a("n3"),
                fact: FactId::new("f_other"),
                assertive: true,
                deceptive: false,
                retold_chain: None,
            },
        );
    }
    for _ in 0..5 {
        s.step(Vec::new());
        // The plan holder never receives a signal from the unrelated pair.
        assert!(
            s.planner.signals.get(&a("n1")).is_none_or(|q| q.is_empty()),
            "unrelated events must not signal the plan holder"
        );
        // The plan is unchanged: the fast path continues it, never re-plans.
        let now = s.planner.plans.get(&a("n1")).expect("plan persists");
        assert_eq!(now.adopted_tick, plan.adopted_tick);
        assert_eq!(now.intention.option, plan.intention.option);
        assert_eq!(now.action, plan.action);
        // The budget is a hard cap no matter how many events fired.
        assert!(
            s.last_planner_pass().activations.len() <= 1,
            "activation budget exceeded"
        );
        // Every n1 activation was cheap fast-path work: no re-deliberation.
        for act in &s.last_planner_pass().activations {
            if act.actor == a("n1") {
                assert!(
                    !act.replanned,
                    "n1 must not re-deliberate on unrelated events"
                );
                assert_eq!(act.work, 0, "fast path must not generate candidates");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// T4.1b: bounded fair activation makes progress across multiple NPCs
// ---------------------------------------------------------------------------

#[test]
fn t4_1b_fair_activation_makes_progress_across_multiple_npcs() {
    let mut s = sim(world_fixture());
    s.world
        .actors
        .insert(a("bully").clone(), actor("bully", HALL, 4.0));
    for n in ["n1", "n2", "n3"] {
        fear_of(&mut s, n, 0.8);
        s.world
            .actor_mut(&a(n))
            .unwrap()
            .relationship_mut(&a("bully"))
            .fear = 0.9;
    }
    let start: BTreeMap<String, String> = s
        .world
        .actors
        .iter()
        .map(|(id, actor)| (id.0.clone(), actor.location.0.clone()))
        .collect();

    // Budget 1: over a window of steps every frightened NPC is activated and
    // flees the bully's room — bounded work per step, progress for all.
    let mut seen_flee_plans: BTreeSet<String> = BTreeSet::new();
    for step in 0..24 {
        s.step(Vec::new());
        assert!(
            s.last_planner_pass().activations.len() <= 1,
            "budget exceeded at step {step}"
        );
        for act in &s.last_planner_pass().activations {
            if act.replanned {
                seen_flee_plans.insert(act.actor.0.clone());
            }
        }
    }
    assert_eq!(
        seen_flee_plans.len(),
        3,
        "each of the three NPCs must have re-deliberated at least once (fair window)"
    );
    // Progress: every frightened NPC ended up away from the bully's room.
    for n in ["n1", "n2", "n3"] {
        let actor = s.world.actor(&a(n)).unwrap();
        assert_ne!(
            actor.location.0, HALL,
            "{n} must have fled the bully's room (started in {:?})",
            start[n]
        );
    }
}

// ---------------------------------------------------------------------------
// T4.4a: the per-activation work cap and the activation budget are enforced
// ---------------------------------------------------------------------------

#[test]
fn t4_4a_activation_budget_and_work_caps_are_enforced_under_load() {
    let mut s = sim(world_fixture());
    s.world
        .actors
        .insert(a("bully").clone(), actor("bully", HALL, 4.0));
    let mut cfg = config();
    cfg.activation.max_candidates_per_activation = 1;
    let mut s = Simulation::new(s.world, 2026, cfg);
    // Adversarial load: every NPC is frightened *and* holds contradictory
    // evidence (two candidate families each) — the caps must hold.
    add_fact(
        &mut s.world,
        "f_conflict",
        FactCategory::Observation,
        Sensitivity::Public,
    );
    for n in ["n1", "n2", "n3"] {
        fear_of(&mut s, n, 0.8);
        s.world
            .actor_mut(&a(n))
            .unwrap()
            .relationship_mut(&a("bully"))
            .fear = 0.9;
        // Two conflicting sources (independent chains): n3 says true, n1 says
        // false — each NPC ends up with a contradicted ledger for f_conflict…
        // only for the NPC who is *not* a source; n1 and n3 are sources.
    }
    // n2 (not a source) gets one true item and one false item: contradicted.
    record_provenance_evidence(
        &mut s.world,
        &mut s.events,
        0,
        480,
        &a("n2"),
        &FactId::new("f_conflict"),
        true,
        &Provenance::Told {
            by: a("n1"),
            event: None,
        },
    );
    record_provenance_evidence(
        &mut s.world,
        &mut s.events,
        0,
        480,
        &a("n2"),
        &FactId::new("f_conflict"),
        false,
        &Provenance::Told {
            by: a("n3"),
            event: None,
        },
    );

    for step in 0..12 {
        s.step(Vec::new());
        assert!(
            s.last_planner_pass().activations.len() <= 1,
            "activation budget exceeded at step {step}"
        );
        for act in &s.last_planner_pass().activations {
            assert!(
                act.work <= 1,
                "per-activation work cap exceeded: {} candidates considered",
                act.work
            );
        }
    }
}

// ---------------------------------------------------------------------------
// T4.9a: QUERY candidates are grounded in the actor's own conflicted ledger
// and preserve the epistemic boundaries on both the authorized and the
// unauthorized path
// ---------------------------------------------------------------------------

#[test]
fn t4_9a_query_candidate_is_actor_local_and_boundary_preserving() {
    let mut s = sim(world_fixture());
    add_fact(
        &mut s.world,
        "f_secret",
        FactCategory::Crime,
        Sensitivity::Secret,
    );
    s.world
        .facts
        .get_mut(&FactId::new("f_secret"))
        .unwrap()
        .eligible_holders = vec![a("n1")];
    add_fact(
        &mut s.world,
        "f_noise",
        FactCategory::Observation,
        Sensitivity::Public,
    );
    seed_belief(&mut s.world, "n1", "f_noise");
    seed_belief(&mut s.world, "n1", "f_secret");
    // n3 also *knows of* f_secret (holds its own contrary judgment — an
    // independent origin) but is not an eligible holder — knowing is not
    // permission (design §9.1.2).
    let (t0, m0) = (0u64, 480u32);
    record_provenance_evidence(
        &mut s.world,
        &mut s.events,
        t0,
        m0,
        &a("n3"),
        &FactId::new("f_secret"),
        false,
        &Provenance::Assumed,
    );
    // n2 holds contradictory evidence about f_secret: n1's chain says true,
    // n3's independent chain says false. The *latest* source is n3.
    record_provenance_evidence(
        &mut s.world,
        &mut s.events,
        0,
        480,
        &a("n2"),
        &FactId::new("f_secret"),
        true,
        &Provenance::Told {
            by: a("n1"),
            event: None,
        },
    );
    record_provenance_evidence(
        &mut s.world,
        &mut s.events,
        0,
        480,
        &a("n2"),
        &FactId::new("f_secret"),
        false,
        &Provenance::Told {
            by: a("n3"),
            event: None,
        },
    );
    let agg =
        rpg_behavior::aggregate_about(&s.world.epistemics, &a("n2"), &FactId::new("f_secret"));
    assert!(agg.contradicted, "fixture: the ledger must be conflicted");

    // A signal wakes n2 up: an unrelated claim about f_noise is delivered.
    apply(
        &mut s,
        "n1",
        Action::ShareClaim {
            target: a("n2"),
            fact: FactId::new("f_noise"),
            assertive: true,
            deceptive: false,
            retold_chain: None,
        },
    );

    // The activation queries the actor's *own latest source* (n3) — actor-
    // local provenance, not "who really knows".
    let mut asked = false;
    for _ in 0..4 {
        s.step(Vec::new());
        if let Some(e) = last_event(
            &s,
            |k| matches!(k, EventKind::QueryAnswered { asker, .. } if asker == &a("n2")),
        ) {
            asked = true;
            match &e.kind {
                EventKind::QueryAnswered {
                    answerer,
                    refused,
                    unknown,
                    polarities,
                    ..
                } => {
                    assert_eq!(answerer, &a("n3"));
                    // n3 is not an eligible holder of the Secret: the answer
                    // is a refusal carrying no content (knowledge is not
                    // permission; design §9.1.2).
                    assert!(*refused, "an unauthorized answerer must refuse");
                    assert!(!*unknown, "a refusal is never conflated with ignorance");
                    assert!(polarities.is_empty(), "a refusal transmits no polarity");
                }
                _ => unreachable!(),
            }
            break;
        }
    }
    assert!(
        asked,
        "the conflicted actor must have queried its own source"
    );
    // The refusal delivered nothing: n2's ledger about f_secret is unchanged.
    let delivered = s
        .world
        .epistemics
        .evidence_about(&a("n2"), &FactId::new("f_secret"))
        .iter()
        .filter(|e| e.source_actor_id == Some(a("n3")) && e.received_tick > 0)
        .count();
    assert_eq!(delivered, 0, "a refusal must not deliver evidence");

    // The authorized path: n1 (an eligible holder) is asked next time.
    // Directly drive the authorized check through the production pipeline:
    // n2 queries n1 via an explicit proposal (the candidate generation above
    // chose n3; the sim still validates whatever is proposed).
    apply(
        &mut s,
        "n2",
        Action::Query {
            target: a("n1"),
            about: FactId::new("f_secret"),
        },
    );
    let answered = last_event(&s, |k| {
        matches!(k, EventKind::QueryAnswered { asker, answerer, .. } if asker == &a("n2") && answerer == &a("n1"))
    })
    .expect("authorized answer");
    match &answered.kind {
        EventKind::QueryAnswered {
            refused,
            polarities,
            unknown,
            ..
        } => {
            assert!(!refused && !unknown, "the holder may answer");
            assert!(
                !polarities.is_empty(),
                "the holder answers from its ledger or authored belief"
            );
        }
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// T4.10a: SANCTION candidates are gated on authority and real evidence
// ---------------------------------------------------------------------------

#[test]
fn t4_10a_sanction_candidate_requires_authority_and_evidence() {
    let mut s = sim(world_fixture());
    add_fact(
        &mut s.world,
        "f_assault",
        FactCategory::Crime,
        Sensitivity::Public,
    );
    s.world
        .actors
        .insert(a("vandal").clone(), actor("vandal", HALL, 5.0));
    s.world
        .actors
        .insert(a("witness").clone(), actor("witness", HALL, 6.0));
    s.world
        .actors
        .insert(a("captain").clone(), actor("captain", HALL, 6.0));
    s.world
        .actor_mut(&a("captain"))
        .unwrap()
        .capabilities
        .can_arrest = true;
    // The captain trusts its long-time witness: the report lands credible.
    s.world
        .actor_mut(&a("captain"))
        .unwrap()
        .relationship_mut(&a("witness"))
        .trust = 1.0;
    // Everyone has money so a fine can actually be paid.
    for id in ["vandal", "captain"] {
        s.world.actor_mut(&a(id)).unwrap().inventory.money = 100;
    }

    // The crime: vandal threatens witness (the ThreatMade event's actor is
    // the culprit — the chain origin the captain later grounds on).
    apply(
        &mut s,
        "vandal",
        Action::Threaten {
            target: a("witness"),
            demand: "leave the docks".into(),
        },
    );
    let threat_id = last_event(&s, |k| matches!(k, EventKind::ThreatMade { .. }))
        .map(|e| e.id)
        .expect("threat event");
    // Two witnesses *saw* it: provenance-bound evidence whose origin event
    // is the threat (two chains, one origin — independent-ish corroboration
    // with correlation protection).
    let (tick_now, minute_now) = (s.world.tick, s.world.minute_of_day);
    s.world
        .actors
        .insert(a("witness2").clone(), actor("witness2", HALL, 7.0));
    for w in ["witness", "witness2"] {
        record_provenance_evidence(
            &mut s.world,
            &mut s.events,
            tick_now,
            minute_now,
            &a(w),
            &FactId::new("f_assault"),
            true,
            &Provenance::Witnessed { event: threat_id },
        );
        // The witness reports to the captain: the report carries the chain
        // (claims preserve their origin chain), so the captain can ground
        // the subject without ever inventing one.
        apply(
            &mut s,
            w,
            Action::ReportCrime {
                authority: a("captain"),
                evidence: vec![FactId::new("f_assault")],
            },
        );
    }

    // The captain is activated (the report is a relevant event) and, being
    // the authority, proposes the grounded sanction.
    let mut sanctioned = false;
    for _ in 0..4 {
        s.step(Vec::new());
        if let Some(e) = last_event(&s, |k| matches!(k, EventKind::SanctionIssued { .. })) {
            match &e.kind {
                EventKind::SanctionIssued {
                    authority,
                    subject,
                    kind,
                    evidence,
                } => {
                    assert_eq!(authority, &a("captain"));
                    assert_eq!(
                        subject,
                        &a("vandal"),
                        "the subject is the chain-grounded culprit"
                    );
                    assert_eq!(kind, "fine");
                    assert_eq!(evidence, &vec![FactId::new("f_assault")]);
                }
                _ => unreachable!(),
            }
            sanctioned = true;
            break;
        }
    }
    assert!(
        sanctioned,
        "the authority must sanction on its real evidence"
    );

    // The fine was a real transfer (conservation) and real need movement.
    assert_eq!(s.world.actor(&a("vandal")).unwrap().inventory.money, 90);
    assert_eq!(s.world.actor(&a("captain")).unwrap().inventory.money, 110);
    let need_events = s
        .events
        .all()
        .iter()
        .filter(|e| {
            matches!(
                &e.kind,
                EventKind::NeedChanged {
                    need: NeedKind::Wealth,
                    ..
                }
            )
        })
        .count();
    assert!(
        need_events >= 2,
        "the fine's wealth deltas are event-sourced"
    );

    // The *non-authority* never generates a sanction candidate, and an
    // unauthorized proposal is rejected as not-permitted (distinct from
    // not-possible).
    let witness_candidates = rpg_agents::planner_candidates::planner_candidates(
        &s.world,
        &a("witness"),
        &s.events,
        &s.config.needs,
        &s.config.emotions,
        s.config.talk_radius,
        8,
        &BTreeSet::new(),
    );
    assert!(
        !witness_candidates
            .iter()
            .any(|c| c.option.id.starts_with("sanction:")),
        "a non-authority must not generate sanction candidates"
    );
    let rejected = s.apply(&ActionProposal {
        actor: a("witness"),
        action: Action::Sanction {
            target: a("vandal"),
            kind: SanctionKind::Fine { amount: 10 },
            evidence: vec![FactId::new("f_assault")],
        },
        source: ProposalSource::NpcPolicy,
        rationale: String::new(),
    });
    assert!(matches!(
        rejected,
        rpg_domain::action::ValidationOutcome::Rejected(_)
    ));
}

// ---------------------------------------------------------------------------
// T4.10b: EXCUSE_DEAL candidates are gated on real contract state + authority
// ---------------------------------------------------------------------------

#[test]
fn t4_10b_excuse_deal_candidate_is_grounding_gated() {
    let mut s = sim(world_fixture());
    add_fact(
        &mut s.world,
        "f_law",
        FactCategory::Organization,
        Sensitivity::Public,
    );
    for id in ["debtor", "creditor", "sheriff"] {
        s.world
            .actors
            .insert(a(id).clone(), actor(id, HALL, 5.0 + id.len() as f32));
    }
    s.world
        .actor_mut(&a("sheriff"))
        .unwrap()
        .capabilities
        .can_arrest = true;
    seed_belief(&mut s.world, "creditor", "f_law");
    // The sheriff holds citable institutional knowledge.
    s.world.actor_mut(&a("sheriff")).unwrap().beliefs.insert(
        FactId::new("f_law"),
        Belief {
            fact: FactId::new("f_law"),
            believes_true: true,
            confidence: 0.9,
            source: Provenance::Assumed,
            acquired_tick: 0,
        },
    );
    // The creditor holds the medicine the deal's condition needs.
    s.world
        .actor_mut(&a("creditor"))
        .unwrap()
        .inventory
        .add(ItemId::new("medicine"), 1);
    s.world.items.insert(
        ItemId::new("medicine"),
        Item {
            id: ItemId::new("medicine"),
            name: "medicine".into(),
            description: String::new(),
            base_value: 80,
            tags: vec![],
            stackable: true,
        },
    );

    // A disclosure deal whose condition already holds and whose performance
    // deadline is already past: due and overdue — real contract state. The
    // performance fact is one nobody holds, so the deal cannot auto-perform
    // (only the authority's excuse resolves it).
    add_fact(
        &mut s.world,
        "f_promise",
        FactCategory::Ledger,
        Sensitivity::Public,
    );
    let deal_outcome = s.apply(&ActionProposal {
        actor: a("debtor"),
        action: Action::ProposeDeal {
            target: a("creditor"),
            terms: "I will tell you who altered the ledger.".into(),
            condition: QuestPredicate::HasItem {
                actor: a("creditor"),
                item: ItemId::new("medicine"),
                qty: 1,
            },
            performance: QuestPredicate::HeardFact {
                actor: a("creditor"),
                fact: FactId::new("f_promise"),
            },
            condition_deadline_minute: None,
            performance_deadline_minute: Some(480),
            give: vec![],
            money: 0,
            note: String::new(),
        },
        source: ProposalSource::Player,
        rationale: String::new(),
    });
    assert!(matches!(
        deal_outcome,
        rpg_domain::action::ValidationOutcome::Pending { .. }
    ));
    let deal_id = *s.world.deals.keys().next().expect("the proposed deal");
    apply(&mut s, "creditor", Action::AcceptDeal { deal: deal_id });
    // The condition holds: the pipeline makes the obligation due.
    s.step(Vec::new());
    assert_eq!(
        s.world.deals.get(&deal_id).unwrap().state,
        DealState::PerformanceDue
    );

    // Someone wakes the sheriff up (a claim is delivered): the activation
    // sees the overdue due contract and excuses it with its own evidence.
    apply(
        &mut s,
        "creditor",
        Action::ShareClaim {
            target: a("sheriff"),
            fact: FactId::new("f_law"),
            assertive: true,
            deceptive: false,
            retold_chain: None,
        },
    );
    let mut excused = false;
    for _ in 0..4 {
        s.step(Vec::new());
        if s.world.deals.get(&deal_id).unwrap().state == DealState::Excused {
            excused = true;
            break;
        }
    }
    assert!(excused, "the authority must excuse the overdue due deal");

    // Exactly-once: a second excuse from the same state is an illegal FSM
    // arrow and is rejected, mutating nothing.
    let rejected = s.apply(&ActionProposal {
        actor: a("sheriff"),
        action: Action::ExcuseDeal {
            deal: deal_id,
            evidence: vec![FactId::new("f_law")],
        },
        source: ProposalSource::NpcPolicy,
        rationale: String::new(),
    });
    assert!(matches!(
        rejected,
        rpg_domain::action::ValidationOutcome::Rejected(_)
    ));
    assert_eq!(
        s.world.deals.get(&deal_id).unwrap().state,
        DealState::Excused
    );

    // The grounded generation is authority-gated: the debtor (no authority,
    // though it *is* a party) generates no excuse candidate.
    let debtor_candidates = rpg_agents::planner_candidates::planner_candidates(
        &s.world,
        &a("debtor"),
        &s.events,
        &s.config.needs,
        &s.config.emotions,
        s.config.talk_radius,
        8,
        &BTreeSet::new(),
    );
    assert!(
        !debtor_candidates
            .iter()
            .any(|c| c.option.id.starts_with("excuse_deal:")),
        "a non-authority must not generate excuse candidates"
    );
    let _ = Party::Debtor;
}

// ---------------------------------------------------------------------------
// T3.11c: gated emotion responses become real validated actions in the
// production pipeline
// ---------------------------------------------------------------------------

#[test]
fn t3_11c_emotion_gates_produce_validated_response_actions() {
    let mut s = sim(world_fixture());
    // Fear: n1 fears bully (in the hall) and flees.
    s.world
        .actors
        .insert(a("bully").clone(), actor("bully", HALL, 4.0));
    fear_of(&mut s, "n1", 0.8);
    s.world
        .actor_mut(&a("n1"))
        .unwrap()
        .relationship_mut(&a("bully"))
        .fear = 0.9;
    // Shame: n2 was caught deceiving n1 and must make amends.
    s.world
        .actor_mut(&a("n2"))
        .unwrap()
        .emotions
        .adjust(EmotionKind::Shame, 0.8, 1.0);
    s.world.actor_mut(&a("n2")).unwrap().push_rel_memory(
        a("n1"),
        RelMemory {
            event: None,
            interpretation: "discovered a deception".into(),
            salience: 1.0,
            tick: 0,
        },
    );

    let mut fled = false;
    let mut reconciled = false;
    for _ in 0..20 {
        s.step(Vec::new());
        if !fled
            && last_event(
                &s,
                |k| matches!(k, EventKind::ActorMoved { actor, .. } if actor == &a("n1")),
            )
            .is_some()
        {
            // The flee applied as a real move: n1 left the hall for the far
            // dock, away from the feared actor.
            fled = s.world.actor(&a("n1")).unwrap().location.0 == DOCKS;
        }
        if !reconciled
            && last_event(&s, |k| {
                matches!(k, EventKind::Comforted { from, to } if from == &a("n2") && to == &a("n1"))
            })
            .is_some()
        {
            reconciled = true;
        }
        if fled && reconciled {
            break;
        }
    }
    assert!(
        fled,
        "the fear gate must produce a validated flee (real move)"
    );
    assert!(
        reconciled,
        "the shame gate must produce a validated reconciliation (real comfort)"
    );

    // The comfort really satisfied the comforted party's belonging need —
    // through the validated action, with the truthful typed event.
    let changed = s
        .events
        .all()
        .iter()
        .any(|e| matches!(&e.kind, EventKind::NeedChanged { actor, need: NeedKind::Belonging, delta, .. } if actor == &a("n1") && *delta > 0.0));
    assert!(
        changed,
        "receiving comfort satisfies belonging (NeedChanged)"
    );
}

// ---------------------------------------------------------------------------
// T3.9b: real need fulfillment changes later utility
// ---------------------------------------------------------------------------

#[test]
fn t3_9b_real_need_fulfillment_changes_later_utility() {
    let mut s = sim(world_fixture());
    s.world
        .actors
        .insert(a("payer").clone(), actor("payer", HALL, 5.5));
    s.world.actor_mut(&a("payer")).unwrap().inventory.money = 200;
    s.world
        .actor_mut(&a("n1"))
        .unwrap()
        .needs
        .set(NeedKind::Wealth, 0.1);

    let params = rpg_behavior::Params::default();
    let needs_config = s.config.needs;
    let options = |gains: Vec<(NeedKind, f32)>| -> Vec<UtilityOption> {
        vec![UtilityOption {
            id: "wealth_gain".into(),
            goal_delta: 1.0,
            cost: 0.0,
            feasibility: rpg_behavior::Feasibility::Possible,
            needs_gain: gains,
        }]
    };
    let needs_before = s.world.actor(&a("n1")).unwrap().needs.clone();
    let before = rpg_behavior::rank_with_needs(
        &options(vec![(NeedKind::Wealth, 0.5)]),
        None,
        params,
        &needs_before,
        &needs_config,
    )
    .ranked[0]
        .utility;

    // The real validated action: the payer pays n1 100 coins.
    apply(
        &mut s,
        "payer",
        Action::Pay {
            target: a("n1"),
            amount: 100,
        },
    );

    // The need really moved — through the validated action, event-sourced.
    let moved = s
        .events
        .all()
        .iter()
        .any(|e| matches!(&e.kind, EventKind::NeedChanged { actor, need: NeedKind::Wealth, delta, .. } if actor == &a("n1") && *delta > 0.5));
    assert!(moved, "receiving payment satisfies wealth (NeedChanged)");
    let needs_after = s.world.actor(&a("n1")).unwrap().needs.clone();
    assert!(
        needs_after.value(NeedKind::Wealth) > needs_before.value(NeedKind::Wealth),
        "the validated payment must move the need"
    );

    // Later utility: the same wealth-gain option is worth less now that the
    // need is (nearly) satisfied — the same kernel the production pipeline
    // calls ranks differently after real fulfillment.
    let after = rpg_behavior::rank_with_needs(
        &options(vec![(NeedKind::Wealth, 0.5)]),
        None,
        params,
        &needs_after,
        &needs_config,
    )
    .ranked[0]
        .utility;
    assert!(
        after < before,
        "real need fulfillment must lower the later utility of the same gain \
         (before {before}, after {after})"
    );

    // The payer's wealth really dropped too (truthful depletion).
    let payer_lost = s
        .events
        .all()
        .iter()
        .any(|e| matches!(&e.kind, EventKind::NeedChanged { actor, need: NeedKind::Wealth, delta, .. } if actor == &a("payer") && *delta < 0.0));
    assert!(payer_lost, "paying money depletes the payer's wealth need");
}

// ---------------------------------------------------------------------------
// T2.9b + T0.5b: seeded sampling in the recorded production decision stream —
// same seed reproduces; save/load preserves sampling and intentions; replay
// reproduces everything
// ---------------------------------------------------------------------------

fn frightened_fixture() -> World {
    let mut w = world_fixture();
    w.actors
        .insert(a("bully").clone(), actor("bully", HALL, 4.0));
    for n in ["n1", "n2", "n3"] {
        w.actors
            .get_mut(&a(n))
            .unwrap()
            .emotions
            .adjust(EmotionKind::Fear, 0.8, 1.0);
        w.actors
            .get_mut(&a(n))
            .unwrap()
            .relationship_mut(&a("bully"))
            .fear = 0.9;
    }
    w
}

#[test]
fn t2_9b_seeded_sampling_reproduces_and_is_recorded() {
    let seed = 777;
    let mut first = Simulation::new(frightened_fixture(), seed, config());
    let mut second = Simulation::new(frightened_fixture(), seed, config());
    for _ in 0..10 {
        first.step(Vec::new());
        second.step(Vec::new());
    }
    assert_eq!(
        first.events.checksum(),
        second.events.checksum(),
        "same seed must reproduce the same decisions and outcomes"
    );
    assert_eq!(
        serde_json::to_string(&first.world).unwrap(),
        serde_json::to_string(&second.world).unwrap()
    );

    // The sampled decisions are *recorded*: every planner adoption's causal
    // trace carries the decision record (option, scores, sample, tau).
    let sampled: Vec<_> = first
        .causal_traces
        .iter()
        .filter_map(|t| t.decision.as_ref())
        .collect();
    assert!(
        !sampled.is_empty(),
        "planner adoptions must record their sampled decisions"
    );
    for d in &sampled {
        assert!(d.sample.unwrap_or(-1.0) >= 0.0 && d.sample.unwrap_or(2.0) <= 1.0);
        assert!(!d.utilities.is_empty());
        assert!(d.utilities.iter().any(|(id, _)| *id == d.option));
    }

    // Replay over the recorded ops reproduces the full stream.
    let replay = Simulation::replay_ops(
        first.initial_world.clone(),
        seed,
        first.config.clone(),
        &first.ops,
    );
    assert_eq!(replay.events.checksum(), first.events.checksum());
    assert_eq!(
        serde_json::to_string(&replay.world).unwrap(),
        serde_json::to_string(&first.world).unwrap()
    );
}

#[test]
fn t0_5b_save_load_preserves_plans_sampling_and_continues_identically() {
    let seed = 999;
    let mut s = Simulation::new(frightened_fixture(), seed, config());
    for _ in 0..6 {
        s.step(Vec::new());
    }
    assert!(
        !s.planner.plans.is_empty(),
        "fixture: plans must exist before the save"
    );

    // Round-trip the full transient state (plans, signals, rotation) through
    // JSON — exactly what persistence does — and continue both simulations.
    let json = serde_json::to_string(&s.transient()).unwrap();
    let transient: rpg_sim::SimTransient = serde_json::from_str(&json).unwrap();
    assert!(
        !transient.planner.plans.is_empty(),
        "plans survive the save"
    );
    let mut restored = Simulation::restore_full(
        s.seed,
        s.world.clone(),
        s.events.clone(),
        s.rng.clone(),
        transient,
        s.ops.clone(),
        s.initial_world.clone(),
        s.config.clone(),
    );
    for _ in 0..12 {
        s.step(Vec::new());
        restored.step(Vec::new());
    }
    assert_eq!(
        serde_json::to_string(&s.world).unwrap(),
        serde_json::to_string(&restored.world).unwrap(),
        "world diverged after restoring mid-planning (sampling/intentions lost)"
    );
    assert_eq!(s.events.checksum(), restored.events.checksum());
    assert_eq!(
        serde_json::to_string(&s.planner).unwrap(),
        serde_json::to_string(&restored.planner).unwrap(),
        "planner state (plans/signals/rotation) must round-trip identically"
    );

    // A pre-planner save (no planner field) still loads (serde default).
    let legacy = serde_json::json!({
        "paths": {},
        "pending_private": [],
        "last_incident_minute": 0,
        "autonomy": { "pursuing": {}, "cooldowns": {}, "tended": {} },
        "deal_intentions": {}
    });
    let old: rpg_sim::SimTransient = serde_json::from_value(legacy).unwrap();
    assert!(old.planner.plans.is_empty());
}

// ---------------------------------------------------------------------------
// T4.1c (adversarial, review F1): sustained signal load cannot starve the
// rotation. One actor is re-signaled every tick while two others are
// eligible through persistent plans; budget 1 must still rotate through all
// three, and the fairness state must survive save/load exactly.
// ---------------------------------------------------------------------------

#[test]
fn t4_1c_sustained_signal_load_cannot_starve_the_rotation() {
    let mut s = sim(world_fixture());
    // n1 ("X") is perpetually gated *at* its flee destination: the bully
    // stays in the hall, n1 already fled to the docks, and the fear gate
    // keeps firing (re-signaling n1) every single tick.
    s.world
        .actors
        .insert(a("bully").clone(), actor("bully", HALL, 4.0));
    {
        let n1 = s.world.actor_mut(&a("n1")).unwrap();
        n1.location = LocationId::new(DOCKS);
        n1.pos = Vec2::new(33.0, 5.0);
        n1.relationship_mut(&a("bully")).fear = 0.9;
    }
    // n2/n3 ("Y"/"Z") are eligible through persistent, open-ended plans.
    let criteria = rpg_domain::behavior_types::AbandonmentCriteria {
        after_failed_attempts: None,
        predicate: None,
    };
    for (id, other) in [("n2", "n1"), ("n3", "n2")] {
        let plan = rpg_behavior::planner::adopt_plan(
            "stay close to a friend",
            format!("seek_comfort_move:{other}"),
            Action::MoveTo {
                location: LocationId::new(MARKET),
            },
            None,
            criteria.clone(),
            0,
            None,
        );
        s.planner.plans.insert(a(id), plan);
    }

    let mut activated: BTreeMap<String, usize> = BTreeMap::new();
    for step in 0..24 {
        // Sustained load: the fear gate is topped up every tick, so n1 is
        // re-signaled and priority-claimed every single tick.
        fear_of(&mut s, "n1", 0.8);
        s.step(Vec::new());
        assert!(
            s.last_planner_pass().activations.len() <= 1,
            "activation budget exceeded at step {step}"
        );
        for act in &s.last_planner_pass().activations {
            *activated.entry(act.actor.0.clone()).or_default() += 1;
        }
    }
    for id in ["n1", "n2", "n3"] {
        let count = activated.get(id).copied().unwrap_or(0);
        assert!(
            count >= 1,
            "{id} starved under sustained signal load (activations: {activated:?})"
        );
    }

    // Replay stability: the rotation state round-trips through the same
    // persistence path and both runs continue identically.
    let json = serde_json::to_string(&s.transient()).unwrap();
    let transient: rpg_sim::SimTransient = serde_json::from_str(&json).unwrap();
    let mut restored = Simulation::restore_full(
        s.seed,
        s.world.clone(),
        s.events.clone(),
        s.rng.clone(),
        transient,
        s.ops.clone(),
        s.initial_world.clone(),
        s.config.clone(),
    );
    for _ in 0..12 {
        s.step(Vec::new());
        restored.step(Vec::new());
    }
    assert_eq!(s.events.checksum(), restored.events.checksum());
    assert_eq!(
        serde_json::to_string(&s.planner).unwrap(),
        serde_json::to_string(&restored.planner).unwrap(),
        "planner state (incl. rotation history) must round-trip identically"
    );
}

// ---------------------------------------------------------------------------
// T4.10c (adversarial, review F2): sanction identity — one incident
// (evidence fact × subject) is punishable exactly once, across candidate
// generation AND authoritative dispatch, surviving save/load; a genuinely
// distinct incident remains actionable; money stays conserved.
// ---------------------------------------------------------------------------

#[test]
fn t4_10c_sanction_identity_is_enforced_once_across_generation_and_dispatch() {
    let mut s = sim(world_fixture());
    add_fact(
        &mut s.world,
        "f_assault",
        FactCategory::Crime,
        Sensitivity::Public,
    );
    add_fact(
        &mut s.world,
        "f_assault_2",
        FactCategory::Crime,
        Sensitivity::Public,
    );
    s.world
        .actors
        .insert(a("vandal").clone(), actor("vandal", HALL, 5.0));
    s.world
        .actors
        .insert(a("witness").clone(), actor("witness", HALL, 6.0));
    s.world
        .actors
        .insert(a("witness2").clone(), actor("witness2", HALL, 7.0));
    s.world
        .actors
        .insert(a("captain").clone(), actor("captain", HALL, 6.0));
    s.world
        .actor_mut(&a("captain"))
        .unwrap()
        .capabilities
        .can_arrest = true;
    s.world
        .actor_mut(&a("captain"))
        .unwrap()
        .relationship_mut(&a("witness"))
        .trust = 1.0;
    for id in ["vandal", "captain"] {
        s.world.actor_mut(&a(id)).unwrap().inventory.money = 100;
    }
    let money_before: i64 = s.world.actors.values().map(|a| a.inventory.money).sum();

    // Incident 1: vandal threatens the witness; two witnesses saw it and
    // both report (corroboration grounds the subject with real confidence).
    apply(
        &mut s,
        "vandal",
        Action::Threaten {
            target: a("witness"),
            demand: "leave the docks".into(),
        },
    );
    let threat_1 = last_event(&s, |k| matches!(k, EventKind::ThreatMade { .. }))
        .map(|e| e.id)
        .expect("threat event");
    let (t, m) = (s.world.tick, s.world.minute_of_day);
    for w in ["witness", "witness2"] {
        record_provenance_evidence(
            &mut s.world,
            &mut s.events,
            t,
            m,
            &a(w),
            &FactId::new("f_assault"),
            true,
            &Provenance::Witnessed { event: threat_1 },
        );
        apply(
            &mut s,
            w,
            Action::ReportCrime {
                authority: a("captain"),
                evidence: vec![FactId::new("f_assault")],
            },
        );
    }
    let mut sanctioned = false;
    for _ in 0..4 {
        s.step(Vec::new());
        if last_event(&s, |k| matches!(k, EventKind::SanctionIssued { .. })).is_some() {
            sanctioned = true;
            break;
        }
    }
    assert!(sanctioned, "the authority must sanction a real incident");
    assert_eq!(s.world.actor(&a("vandal")).unwrap().inventory.money, 90);
    assert_eq!(s.world.actor(&a("captain")).unwrap().inventory.money, 110);

    // The identity ledger is persisted (transient save state).
    let json = serde_json::to_string(&s.transient()).unwrap();
    let transient: rpg_sim::SimTransient = serde_json::from_str(&json).unwrap();
    assert!(
        transient
            .sanctions
            .contains(&(FactId::new("f_assault"), a("vandal"))),
        "the punished incident identity must persist through save/load"
    );
    let mut restored = Simulation::restore_full(
        s.seed,
        s.world.clone(),
        s.events.clone(),
        s.rng.clone(),
        transient,
        s.ops.clone(),
        s.initial_world.clone(),
        s.config.clone(),
    );

    // Sustained report/retelling load — 12 more steps of the SAME evidence,
    // in the original run and in the restored one: no repeat punishment.
    for target in [&mut s, &mut restored] {
        for _ in 0..12 {
            apply(
                target,
                "witness",
                Action::ReportCrime {
                    authority: a("captain"),
                    evidence: vec![FactId::new("f_assault")],
                },
            );
            target.step(Vec::new());
        }
        let fines = target
            .events
            .all()
            .iter()
            .filter(|e| {
                matches!(
                    &e.kind,
                    EventKind::SanctionIssued { subject, evidence, .. }
                        if subject == &a("vandal")
                            && evidence.contains(&FactId::new("f_assault"))
                )
            })
            .count();
        assert_eq!(
            fines, 1,
            "the same incident/evidence must not permit repeat punishment"
        );
        assert_eq!(
            target.world.actor(&a("vandal")).unwrap().inventory.money,
            90,
            "no repeated fine"
        );
        // Autonomous generation respects the identity: the authority
        // generates no sanction candidate over adjudicated evidence.
        let captain_candidates = rpg_agents::planner_candidates::planner_candidates(
            &target.world,
            &a("captain"),
            &target.events,
            &target.config.needs,
            &target.config.emotions,
            target.config.talk_radius,
            8,
            &target.sanctions,
        );
        assert!(
            !captain_candidates
                .iter()
                .any(|c| c.option.id.starts_with("sanction:")),
            "an adjudicated incident must not be re-proposed by generation"
        );
    }

    // An externally repeated proposal is rejected neutrally (authoritative
    // validation/dispatch side of the identity policy).
    let outcome = s.apply(&ActionProposal {
        actor: a("captain"),
        action: Action::Sanction {
            target: a("vandal"),
            kind: SanctionKind::Fine { amount: 10 },
            evidence: vec![FactId::new("f_assault")],
        },
        source: ProposalSource::Player,
        rationale: String::new(),
    });
    assert!(
        matches!(outcome, rpg_domain::action::ValidationOutcome::Rejected(_)),
        "a repeat sanction over unchanged evidence must be rejected"
    );
    assert_eq!(s.world.actor(&a("vandal")).unwrap().inventory.money, 90);
    assert!(
        !matches!(
            outcome,
            rpg_domain::action::ValidationOutcome::Pending { .. }
        ),
        "the rejection is neutral — no pending side state"
    );

    // A genuinely distinct incident (a new fact with its own origin event)
    // remains actionable: one more sanction, exactly once.
    apply(
        &mut s,
        "vandal",
        Action::Threaten {
            target: a("witness"),
            demand: "leave the docks again".into(),
        },
    );
    let threat_2 = last_event(&s, |k| matches!(k, EventKind::ThreatMade { .. }))
        .map(|e| e.id)
        .expect("second threat event");
    let (t2, m2) = (s.world.tick, s.world.minute_of_day);
    for w in ["witness", "witness2"] {
        record_provenance_evidence(
            &mut s.world,
            &mut s.events,
            t2,
            m2,
            &a(w),
            &FactId::new("f_assault_2"),
            true,
            &Provenance::Witnessed { event: threat_2 },
        );
        apply(
            &mut s,
            w,
            Action::ReportCrime {
                authority: a("captain"),
                evidence: vec![FactId::new("f_assault_2")],
            },
        );
    }
    let mut sanctioned_2 = false;
    for _ in 0..12 {
        s.step(Vec::new());
        if let Some(e) = s.events.all().iter().rev().find(|e| {
            matches!(
                &e.kind,
                EventKind::SanctionIssued { evidence, .. }
                    if evidence.contains(&FactId::new("f_assault_2"))
            )
        }) {
            match &e.kind {
                EventKind::SanctionIssued {
                    subject, evidence, ..
                } => {
                    assert_eq!(subject, &a("vandal"));
                    assert_eq!(evidence, &vec![FactId::new("f_assault_2")]);
                    sanctioned_2 = true;
                }
                _ => unreachable!(),
            }
            break;
        }
    }
    assert!(
        sanctioned_2,
        "a genuinely distinct incident must remain actionable"
    );
    assert_eq!(
        s.world.actor(&a("vandal")).unwrap().inventory.money,
        80,
        "exactly one fine per distinct incident"
    );
    // Conservation: fines moved money, they never minted or destroyed it.
    let total: i64 = s.world.actors.values().map(|a| a.inventory.money).sum();
    assert_eq!(
        total, money_before,
        "money is conserved modulo recorded transfers"
    );
}

// ---------------------------------------------------------------------------
// T3.9c (review F3): self-directed payments/fines are rejected neutrally —
// no money movement, no fictitious `NeedChanged`, no ledger entry — while a
// legitimate distinct-party transfer keeps its exactly-once effects.
// ---------------------------------------------------------------------------

#[test]
fn t3_9c_self_payment_and_self_sanction_are_neutrally_rejected() {
    let mut s = sim(world_fixture());
    add_fact(
        &mut s.world,
        "f_note",
        FactCategory::Observation,
        Sensitivity::Public,
    );
    s.world
        .actors
        .insert(a("captain").clone(), actor("captain", HALL, 6.0));
    s.world
        .actor_mut(&a("captain"))
        .unwrap()
        .capabilities
        .can_arrest = true;
    s.world.actor_mut(&a("captain")).unwrap().inventory.money = 100;
    seed_belief(&mut s.world, "captain", "f_note");

    let need_events_of = |sim: &Simulation, id: &str| {
        sim.events
            .all()
            .iter()
            .filter(|e| {
                matches!(
                    &e.kind,
                    EventKind::NeedChanged { actor, .. } if actor == &a(id)
                )
            })
            .count()
    };
    let needs_before = need_events_of(&s, "captain");

    // Self-payment: rejected; money unchanged; no payment event; no
    // need-movement event.
    let outcome = s.apply(&ActionProposal {
        actor: a("captain"),
        action: Action::Pay {
            target: a("captain"),
            amount: 50,
        },
        source: ProposalSource::Player,
        rationale: String::new(),
    });
    assert!(
        matches!(outcome, rpg_domain::action::ValidationOutcome::Rejected(_)),
        "self-payment must be rejected"
    );
    assert_eq!(
        s.world.actor(&a("captain")).unwrap().inventory.money,
        100,
        "a rejected self-payment moves no money"
    );
    assert!(
        s.events
            .all()
            .iter()
            .all(|e| !matches!(&e.kind, EventKind::MoneyPaid { from, to, .. } if from == to)),
        "no self-directed payment event"
    );
    assert_eq!(
        need_events_of(&s, "captain"),
        needs_before,
        "no fictitious NeedChanged from a self-payment"
    );

    // Self-fine: rejected; no sanction event; money unchanged; and the
    // rejected (never dispatched) act records no sanction identity.
    let outcome = s.apply(&ActionProposal {
        actor: a("captain"),
        action: Action::Sanction {
            target: a("captain"),
            kind: SanctionKind::Fine { amount: 10 },
            evidence: vec![FactId::new("f_note")],
        },
        source: ProposalSource::Player,
        rationale: String::new(),
    });
    assert!(
        matches!(outcome, rpg_domain::action::ValidationOutcome::Rejected(_)),
        "a self-fine must be rejected"
    );
    assert_eq!(s.world.actor(&a("captain")).unwrap().inventory.money, 100);
    assert!(
        last_event(&s, |k| matches!(k, EventKind::SanctionIssued { .. })).is_none(),
        "no sanction was issued"
    );
    assert_eq!(need_events_of(&s, "captain"), needs_before);
    assert!(
        s.sanctions.is_empty(),
        "only dispatched sanctions record identity"
    );

    // The shared settlement path cannot even be constructed with identical
    // parties: a self-deal has no counterparty and is validated out at
    // proposal time (so no settlement transfer is reachable with identical
    // parties; the shared transfer lowering additionally no-ops on them).
    let outcome = s.apply(&ActionProposal {
        actor: a("captain"),
        action: Action::ProposeDeal {
            target: a("captain"),
            terms: "self deal".into(),
            condition: QuestPredicate::HasMoney {
                actor: a("captain"),
                amount: 1,
            },
            performance: QuestPredicate::HasMoney {
                actor: a("captain"),
                amount: 1,
            },
            condition_deadline_minute: None,
            performance_deadline_minute: None,
            give: vec![],
            money: 0,
            note: String::new(),
        },
        source: ProposalSource::Player,
        rationale: String::new(),
    });
    assert!(
        matches!(outcome, rpg_domain::action::ValidationOutcome::Rejected(_)),
        "a self-deal must be rejected (no identical-party settlement is reachable)"
    );

    // Compatibility: a legitimate transfer keeps its exactly-once effects.
    s.world
        .actors
        .insert(a("payer").clone(), actor("payer", HALL, 5.0));
    s.world.actor_mut(&a("payer")).unwrap().inventory.money = 100;
    let outcome = s.apply(&ActionProposal {
        actor: a("payer"),
        action: Action::Pay {
            target: a("captain"),
            amount: 30,
        },
        source: ProposalSource::Player,
        rationale: String::new(),
    });
    assert!(
        !matches!(outcome, rpg_domain::action::ValidationOutcome::Rejected(_)),
        "a legitimate payment must apply"
    );
    assert_eq!(s.world.actor(&a("payer")).unwrap().inventory.money, 70);
    assert_eq!(s.world.actor(&a("captain")).unwrap().inventory.money, 130);
    assert_eq!(
        need_events_of(&s, "captain"),
        needs_before + 1,
        "a real transfer moves the recipient's wealth need exactly once"
    );
}

// ---------------------------------------------------------------------------
// T4.4b (review F4): relevant signals genuinely reconsider an active plan,
// the current intention's switch penalty is supplied in that pathway, and
// sustained signals never defeat the caps or the plan's persistence.
// ---------------------------------------------------------------------------

fn reconsideration_fixture(with_plan: bool) -> Simulation {
    let mut cfg = config();
    // Deterministic argmax: isolates the switch-penalty wiring from the
    // softmax draw (sampling is covered by T2.9b).
    cfg.utility_sampling.enabled = false;
    let mut s = Simulation::new(world_fixture(), 2026, cfg);
    s.world
        .actors
        .insert(a("bully").clone(), actor("bully", HALL, 4.0));
    // n2 already stands at its flee destination (flee candidate cost 0); n3
    // is the near source of its conflicted report (the query candidate).
    {
        let n2 = s.world.actor_mut(&a("n2")).unwrap();
        n2.location = LocationId::new(DOCKS);
        n2.pos = Vec2::new(33.0, 5.0);
        n2.relationship_mut(&a("bully")).fear = 0.9;
        n2.emotions.adjust(EmotionKind::Fear, 0.8, 1.0);
        let n3 = s.world.actor_mut(&a("n3")).unwrap();
        n3.location = LocationId::new(DOCKS);
        n3.pos = Vec2::new(33.0, 5.0);
    }
    add_fact(
        &mut s.world,
        "f_conflict",
        FactCategory::Observation,
        Sensitivity::Public,
    );
    record_provenance_evidence(
        &mut s.world,
        &mut s.events,
        0,
        480,
        &a("n2"),
        &FactId::new("f_conflict"),
        true,
        &Provenance::Told {
            by: a("n1"),
            event: None,
        },
    );
    record_provenance_evidence(
        &mut s.world,
        &mut s.events,
        0,
        480,
        &a("n2"),
        &FactId::new("f_conflict"),
        false,
        &Provenance::Told {
            by: a("n3"),
            event: None,
        },
    );
    // The trigger fact the claimer lawfully holds and may share.
    add_fact(
        &mut s.world,
        "f_noise",
        FactCategory::Observation,
        Sensitivity::Public,
    );
    seed_belief(&mut s.world, "n3", "f_noise");
    if with_plan {
        let plan = rpg_behavior::planner::adopt_plan(
            "get away from bully",
            "flee:bully:docks",
            Action::MoveTo {
                location: LocationId::new(DOCKS),
            },
            None,
            rpg_domain::behavior_types::AbandonmentCriteria {
                after_failed_attempts: None,
                predicate: None,
            },
            0,
            None,
        );
        s.planner.plans.insert(a("n2"), plan);
    }
    s
}

#[test]
fn t4_4b_relevant_signals_reconsider_an_active_plan_with_switch_costs() {
    // Fixture A: an active flee plan + a delivered claim (a relevant event).
    // The claim must trigger *genuine* reconsideration — but the query
    // challenger (raw utility above the incumbent) cannot beat the incumbent
    // through the §4.2 switch penalty, so the plan is KEPT, not re-adopted.
    let mut s = reconsideration_fixture(true);
    apply(
        &mut s,
        "n3",
        Action::ShareClaim {
            target: a("n2"),
            fact: FactId::new("f_noise"),
            assertive: true,
            deceptive: false,
            retold_chain: None,
        },
    );
    s.step(Vec::new());
    let activation = s
        .last_planner_pass()
        .activations
        .iter()
        .find(|act| act.actor == a("n2"))
        .expect("n2 must be activated");
    assert!(
        activation.replanned && activation.work > 0,
        "the relevant signal must trigger genuine reconsideration (candidates generated)"
    );
    let plan = s.planner.plans.get(&a("n2")).expect("plan persists");
    assert_eq!(
        plan.intention.option, "flee:bully:docks",
        "the challenger must not beat the incumbent through the switch penalty"
    );
    assert_eq!(
        plan.adopted_tick, 0,
        "a kept incumbent is never re-adopted (adopted tick preserved)"
    );
    assert_eq!(plan.intention.switches, 0, "no switch occurred");
    assert_eq!(plan.intention.failed_attempts, 0, "no failure recorded");

    // Sustained signal load (the fear gate re-signals every tick): the
    // reconsideration stays bounded (per-activation work cap) and never
    // defeats intention persistence — the plan is never thrashed away.
    for step in 1..6 {
        fear_of(&mut s, "n2", 0.8);
        s.step(Vec::new());
        let plan = s.planner.plans.get(&a("n2")).expect("plan persists");
        assert_eq!(plan.intention.option, "flee:bully:docks");
        assert_eq!(
            plan.adopted_tick, 0,
            "still never re-adopted at step {step}"
        );
        assert!(
            s.last_planner_pass().activations.len() <= 1,
            "budget is a hard cap under sustained load"
        );
        for act in &s.last_planner_pass().activations {
            assert!(
                act.work <= s.config.activation.max_candidates_per_activation,
                "per-activation work cap holds under sustained load"
            );
        }
    }

    // Fixture B (control): the same world without an active plan. The same
    // delivered claim leads a fresh deliberation — with no current intention
    // there is no switch penalty, and the challenger (the query) wins.
    let mut control = reconsideration_fixture(false);
    apply(
        &mut control,
        "n3",
        Action::ShareClaim {
            target: a("n2"),
            fact: FactId::new("f_noise"),
            assertive: true,
            deceptive: false,
            retold_chain: None,
        },
    );
    control.step(Vec::new());
    let control_plan = control
        .planner
        .plans
        .get(&a("n2"))
        .expect("the fresh deliberation must adopt a plan");
    assert!(
        control_plan.intention.option.starts_with("query:"),
        "without a current intention the challenger wins: {}",
        control_plan.intention.option
    );
}

// ---------------------------------------------------------------------------
// T2.9c (review finding): a proposal dropped at the at-post gate must take
// its sampled decision record with it — a pending decision never
// mis-attributes itself to the actor's next unrelated trace.
// ---------------------------------------------------------------------------

#[test]
fn t2_9c_dropped_at_post_proposal_leaves_no_stale_decision_record() {
    let mut s = sim(world_fixture());
    s.world
        .actors
        .insert(a("bully").clone(), actor("bully", HALL, 4.0));
    fear_of(&mut s, "n2", 0.8);
    s.world
        .actor_mut(&a("n2"))
        .unwrap()
        .relationship_mut(&a("bully"))
        .fear = 0.9;
    // n2 is not at its post: the schedule demands the docks while n2 stands
    // in the hall, so the planner's proposal for it is dropped at the gate.
    s.world
        .actor_mut(&a("n2"))
        .unwrap()
        .schedule
        .push(rpg_domain::world::ScheduleEntry {
            start_min: 0,
            end_min: 1440,
            location: LocationId::new(DOCKS),
            activity: "watch the docks".into(),
            public: false,
        });
    add_fact(
        &mut s.world,
        "f_other",
        FactCategory::Observation,
        Sensitivity::Public,
    );
    seed_belief(&mut s.world, "n1", "f_other");
    apply(
        &mut s,
        "n1",
        Action::ShareClaim {
            target: a("n2"),
            fact: FactId::new("f_other"),
            assertive: true,
            deceptive: false,
            retold_chain: None,
        },
    );

    let traces_before = s.causal_traces.len();
    s.step(Vec::new());
    // The planner genuinely sampled a decision for n2 this step...
    let activation = s
        .last_planner_pass()
        .activations
        .iter()
        .find(|act| act.actor == a("n2"))
        .expect("n2 must have been activated (fixture premise)");
    assert!(activation.replanned, "fixture: a candidate was considered");
    // ...but its proposal was dropped at the at-post gate: the only n2
    // proposal that actually applied is the schedule move, and its trace
    // must carry no decision (the dropped decision went with its proposal).
    let new_traces = &s.causal_traces[traces_before..];
    assert!(
        new_traces.iter().any(|t| t.actor == a("n2")),
        "the schedule move applied"
    );
    for t in new_traces {
        assert!(
            t.decision.is_none(),
            "a dropped proposal's decision must not mis-attribute to an unrelated trace"
        );
    }
}
