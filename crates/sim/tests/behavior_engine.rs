//! Behavior-engine Phase 0/1 acceptance tests at the authoritative-simulation
//! level: typed effect coverage for every action family (M1), rejection
//! mutates nothing, event-sourced replay of the evidence ledger (T1.7), the
//! shared report mechanic (claim separation), and inspectable causal traces.

use rpg_domain::action::{
    Action, ActionProposal, InvestigateTarget, Loudness, ProposalSource, QuestPredicate,
    RejectReason, ValidationOutcome,
};
use rpg_domain::emotions::{EmotionKind, EmotionsConfig};
use rpg_domain::events::EventKind;
use rpg_domain::geom::Rect;
use rpg_domain::ids::*;
use rpg_domain::needs::NeedsConfig;
use rpg_domain::world::{
    Actor, ActorKind, ActorStatus, Belief, Capabilities, Fact, FactCategory, Item, Location,
    LocationKind, NavGrid, Personality, Provenance, Sensitivity, StoryState, World,
};
use rpg_sim::{SimConfig, Simulation};
use std::collections::{BTreeMap, BTreeSet};

fn bare_actor(id: &str, x: f32, kind: ActorKind) -> Actor {
    Actor {
        id: ActorId::new(id),
        name: id.to_string(),
        kind,
        status: ActorStatus::Alive,
        location: LocationId::new("shop"),
        pos: rpg_domain::geom::Vec2::new(x, 1.0),
        facing: 0.0,
        personality: Personality::default(),
        values: Vec::new(),
        biography: String::new(),
        private_biography: String::new(),
        affiliations: vec![FactionId::new("watch")],
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

fn fixture() -> (World, ActorId, ActorId) {
    let nav = NavGrid::new(12, 12);
    // Fully open floor: every move is reachable.
    let mut locations = BTreeMap::new();
    locations.insert(
        LocationId::new("shop"),
        Location {
            id: LocationId::new("shop"),
            name: "Shop".into(),
            kind: LocationKind::Shop,
            bounds: Rect::new(0.0, 0.0, 10.0, 10.0),
            owner: None,
            sound_transmission: 1.0,
            private: false,
            tags: vec![],
        },
    );
    locations.insert(
        LocationId::new("backroom"),
        Location {
            id: LocationId::new("backroom"),
            name: "Backroom".into(),
            kind: LocationKind::Backroom,
            bounds: Rect::new(10.0, 10.0, 2.0, 2.0),
            owner: None,
            sound_transmission: 0.5,
            private: true,
            tags: vec![],
        },
    );
    let mut items = BTreeMap::new();
    items.insert(
        ItemId::new("coin"),
        Item {
            id: ItemId::new("coin"),
            name: "Coin".into(),
            description: String::new(),
            base_value: 10,
            tags: vec![],
            stackable: true,
        },
    );
    let mut facts = BTreeMap::new();
    facts.insert(
        FactId::new("f_x"),
        Fact {
            id: FactId::new("f_x"),
            statement: "x is so".into(),
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
    facts.insert(
        FactId::new("f_secret"),
        Fact {
            id: FactId::new("f_secret"),
            statement: "a secret".into(),
            truth: true,
            sensitivity: Sensitivity::InnerCircle,
            category: FactCategory::Organization,
            prerequisites: vec![],
            eligible_holders: vec![ActorId::new("b")],
            eligible_containers: vec![],
            committed: false,
            placed_at: None,
        },
    );
    let mut player = bare_actor(PLAYER, 1.0, ActorKind::Player);
    player.inventory.money = 100;
    player.beliefs.insert(
        FactId::new("f_x"),
        Belief {
            fact: FactId::new("f_x"),
            believes_true: true,
            confidence: 0.9,
            source: Provenance::Assumed,
            acquired_tick: 0,
        },
    );
    player.goals.push(rpg_domain::world::Goal {
        id: 1,
        description: "learn the town's secrets".into(),
        kind: rpg_domain::world::GoalKind::Investigate,
        priority: 5.0,
        need_boost: 0.0,
        status: rpg_domain::world::GoalStatus::Active,
        predicate: None,
        target_location: None,
        target_actor: None,
    });
    let mut b = bare_actor("b", 2.0, ActorKind::Npc);
    b.capabilities.can_arrest = true;
    b.flags.insert("authority".into());
    b.beliefs.insert(
        FactId::new("f_x"),
        Belief {
            fact: FactId::new("f_x"),
            believes_true: true,
            confidence: 0.9,
            source: Provenance::Assumed,
            acquired_tick: 0,
        },
    );
    let mut actors = BTreeMap::new();
    actors.insert(player.id.clone(), player);
    actors.insert(b.id.clone(), b);
    let mut factions = BTreeMap::new();
    factions.insert(
        FactionId::new("watch"),
        rpg_domain::world::Faction {
            id: FactionId::new("watch"),
            name: "The Watch".into(),
            kind: rpg_domain::world::FactionKind::Government,
            budget: 500,
            staff: 3,
            responsible_for: vec![LocationId::new("shop")],
            members: vec![ActorId::new("b"), ActorId::new(PLAYER)],
            priorities: vec![rpg_domain::world::FactionPriority {
                description: "patrol the shop".into(),
                weight: 1.0,
                cost: 50,
                predicate: Some(QuestPredicate::FactionPaid {
                    faction: FactionId::new("watch"),
                    priority: "patrol".into(),
                    amount: 50,
                }),
                resolved: false,
            }],
            legitimacy: 0.9,
            public: true,
        },
    );
    let world = World {
        tick: 0,
        day: 0,
        minute_of_day: 480,
        locations,
        actors,
        items,
        factions,
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
        damage_facts: BTreeMap::new(),
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
    (world, ActorId::new(PLAYER), ActorId::new("b"))
}

fn sim_config() -> SimConfig {
    SimConfig {
        enable_background: false,
        enable_director: false,
        enable_goal_autonomy: false,
        ..SimConfig::default()
    }
}

fn apply(sim: &mut Simulation, actor: &ActorId, action: Action) -> ValidationOutcome {
    sim.apply(&ActionProposal {
        actor: actor.clone(),
        action,
        source: ProposalSource::Player,
        rationale: String::new(),
    })
}

// ---------------------------------------------------------------------------
// Typed effect coverage (M1: every authoritative mutation is typed)
// ---------------------------------------------------------------------------

/// Every accepted action lowers into typed primitive effects; `Idle` mutates
/// nothing and legitimately carries none.
#[test]
fn every_accepted_action_produces_typed_effects() {
    let (world, player, b) = fixture();
    let mut sim = Simulation::new(world, 5, sim_config());

    // Setup state the actions need: an offer to accept, a held coin, warmth
    // for recruitment, damage for repair.
    let _ = apply(
        &mut sim,
        &b,
        Action::MakeOffer {
            target: player.clone(),
            give: vec![],
            want: vec![],
            condition: Some(QuestPredicate::Flag("deal_done".into())),
            disclose: None,
            note: "a deal".into(),
        },
    );
    let offer_id = *sim.world.offers.keys().next().unwrap();
    sim.world
        .actor_mut(&player)
        .unwrap()
        .inventory
        .add(ItemId::new("coin"), 3);
    sim.world
        .actor_mut(&b)
        .unwrap()
        .relationship_mut(&player)
        .trust = 1.0;
    sim.world.damage.insert(LocationId::new("shop"), 0.5);

    let cases: Vec<(Action, Vec<&'static str>)> = vec![
        (
            Action::Speak {
                target: Some(b.clone()),
                text: "hello".into(),
                loudness: Loudness::Normal,
            },
            // F2: hearing the utterance is a typed tie change (familiarity).
            vec!["observe", "change_tie", "create_event"],
        ),
        (
            Action::AskPrivately { target: b.clone() },
            vec!["request", "schedule"],
        ),
        (
            Action::MoveTo {
                location: LocationId::new("backroom"),
            },
            vec!["move"],
        ),
        (
            Action::MoveToPos {
                pos: rpg_domain::geom::Vec2::new(3.0, 3.0),
            },
            vec!["move"],
        ),
        (
            Action::MakeOffer {
                target: b.clone(),
                give: vec![],
                want: vec![],
                condition: None,
                disclose: None,
                note: "deal".into(),
            },
            vec!["offer"],
        ),
        (Action::AcceptOffer { offer: offer_id }, vec!["commit"]),
        (
            Action::Promise {
                target: b.clone(),
                terms: "help".into(),
                predicate: QuestPredicate::Flag("done".into()),
                deadline: None,
            },
            vec!["commit"],
        ),
        (
            Action::TransferItem {
                target: b.clone(),
                item: ItemId::new("coin"),
                qty: 1,
            },
            vec!["transfer", "change_tie"],
        ),
        (
            Action::Pay {
                target: b.clone(),
                amount: 5,
            },
            vec!["transfer", "change_tie"],
        ),
        (
            Action::ShareClaim {
                target: b.clone(),
                fact: FactId::new("f_x"),
                assertive: true,
                deceptive: false,
                retold_chain: None,
            },
            vec!["assert", "change_tie"],
        ),
        (
            Action::SpreadRumor {
                text: "rumor".into(),
                about: Some(FactId::new("f_x")),
            },
            vec!["assert", "create_event"],
        ),
        (
            // Placed before Investigate so the location is not yet inspected
            // and the narrative validator accepts the clue.
            Action::PlaceClue {
                fact: FactId::new("f_x"),
                kind: rpg_domain::world::ContainerKind::Document,
                location: Some(LocationId::new("shop")),
                holder: Some(b.clone()),
            },
            vec!["create_event"],
        ),
        (
            Action::Investigate {
                target: InvestigateTarget::Location(LocationId::new("shop")),
            },
            vec!["create_event"],
        ),
        (
            Action::ReportCrime {
                authority: b.clone(),
                evidence: vec![FactId::new("f_x")],
            },
            // F1 + review D1-b: the authority's reading of the report is a
            // typed, contextual appraisal; the tie change stays typed too.
            vec!["assert", "request", "change_tie", "appraise"],
        ),
        (
            Action::HideEvidence {
                item: ItemId::new("coin"),
                location: LocationId::new("shop"),
            },
            vec!["transform", "create_event"],
        ),
        (
            Action::AllocateBudget {
                faction: FactionId::new("watch"),
                amount: 50,
                purpose: "patrol".into(),
            },
            // The allocation resolves the patrol priority (predicate met),
            // completing the faction's goal. F3: the faction-budget
            // consumption is TRANSFORM (institutional pool consumption),
            // never a fabricated self-to-self TRANSFER.
            vec!["transform", "change_goal", "create_event"],
        ),
        (
            Action::Repair {
                location: LocationId::new("shop"),
            },
            // F3: location restore + budget consumption — two TRANSFORMs.
            vec!["transform"],
        ),
        (
            Action::Recruit { target: b.clone() },
            vec!["authorize", "change_tie"],
        ),
        (
            Action::ScheduleMeeting {
                target: b.clone(),
                location: LocationId::new("shop"),
                at_minute: 600,
            },
            vec!["schedule"],
        ),
        (
            Action::Threaten {
                target: b.clone(),
                demand: "the ledger".into(),
            },
            vec!["request", "appraise", "change_tie"],
        ),
        (
            Action::Comfort { target: b.clone() },
            vec!["appraise", "change_tie"],
        ),
        (
            // QUERY (I-5, G2): the question dispatches the Query primitive;
            // the ledger-derived answer is event-sourced.
            Action::Query {
                target: b.clone(),
                about: FactId::new("f_x"),
            },
            vec!["query", "change_tie"],
        ),
        (Action::Idle, vec![]),
    ];

    let mut covered: BTreeSet<String> = BTreeSet::new();
    for (action, expected_families) in cases {
        // Replay the fixture per action: state from earlier cases must not
        // invalidate later ones, so re-run everything up to this action? No —
        // the fixture above is ordered so each case stays valid; a failing
        // case names its action for diagnosis.
        let out = apply(&mut sim, &player, action.clone());
        let kind = action.kind_name();
        assert!(
            !matches!(out, ValidationOutcome::Rejected(_)),
            "{kind} was unexpectedly rejected: {out:?}"
        );
        let outcome = sim.outcomes.last().expect("outcome recorded");
        assert_eq!(outcome.action_kind, kind);
        let families: Vec<&str> = outcome.effects.iter().map(|e| e.family()).collect();
        for want in &expected_families {
            assert!(
                families.iter().any(|f| f == want),
                "{kind} must lower into {want} (got {families:?})"
            );
            covered.insert(want.to_string());
        }
        assert!(
            !outcome.events.is_empty()
                || matches!(out, ValidationOutcome::Pending { .. })
                || kind.starts_with("move_to")
                || kind == "idle",
            "{kind} must be event-sourced (movement commits a path; its \
             ActorMoved events are emitted during execution ticks)"
        );
        assert!(!outcome.trace.render().is_empty());
    }
    // SANCTION (I-5, G3): the institutional consequence path, applied by the
    // fixture's authority-capable actor (the loop's proposer has no
    // authority — that rejection class is covered by the dedicated test).
    let out = apply(
        &mut sim,
        &b,
        Action::Sanction {
            target: player.clone(),
            kind: rpg_domain::action::SanctionKind::Fine { amount: 5 },
            evidence: vec![FactId::new("f_x")],
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    let outcome = sim.outcomes.last().expect("outcome recorded");
    let families: Vec<&str> = outcome.effects.iter().map(|e| e.family()).collect();
    for want in ["sanction", "transfer", "appraise"] {
        assert!(
            families.contains(&want),
            "sanction must lower into {want} (got {families:?})"
        );
        covered.insert(want.to_string());
    }
    // All 16 primitive families now have action-level lowerings (plan I-5:
    // QUERY and SANCTION included); the vocabulary itself stays closed.
    let expected: BTreeSet<String> = [
        "move",
        "transfer",
        "transform",
        "observe",
        "assert",
        "query",
        "request",
        "offer",
        "commit",
        "authorize",
        "appraise",
        "sanction",
        "schedule",
        "change_goal",
        "change_tie",
        "create_event",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    assert_eq!(
        covered, expected,
        "the action vocabulary must lower into exactly these families"
    );
    assert_eq!(
        expected.len(),
        16,
        "the full vocabulary stays at 16 primitives"
    );
}

// ---------------------------------------------------------------------------
// Rejected actions mutate nothing
// ---------------------------------------------------------------------------

#[test]
fn rejected_actions_mutate_nothing() {
    let (world, player, _b) = fixture();
    let mut sim = Simulation::new(world, 5, sim_config());

    let attempts = vec![
        // Never heard of the fact: no knowledge, no disclosure.
        Action::ShareClaim {
            target: ActorId::new("b"),
            fact: FactId::new("f_secret"),
            assertive: true,
            deceptive: false,
            retold_chain: None,
        },
        Action::MoveTo {
            location: LocationId::new("nowhere"),
        },
        Action::Pay {
            target: ActorId::new("b"),
            amount: 999_999,
        },
        Action::TransferItem {
            target: ActorId::new("b"),
            item: ItemId::new("coin"),
            qty: 7,
        },
    ];

    for attempt in attempts {
        let before = serde_json::to_string(&sim.world).unwrap();
        let before_events = sim.events.len();
        let out = apply(&mut sim, &player, attempt.clone());
        assert!(
            matches!(out, ValidationOutcome::Rejected(_)),
            "{} should be rejected, got {out:?}",
            attempt.kind_name()
        );
        let after = serde_json::to_string(&sim.world).unwrap();
        assert_eq!(
            before,
            after,
            "a rejected {} must not mutate canonical state (including epistemics)",
            attempt.kind_name()
        );
        let outcome = sim.outcomes.last().unwrap();
        assert!(
            outcome.effects.is_empty(),
            "a rejected {} must produce no typed effects, got {:?}",
            attempt.kind_name(),
            outcome.effects
        );
        assert!(outcome.events.is_empty());
        // The rejection itself is recorded (observability), and the world is
        // untouched.
        assert_eq!(sim.events.len(), before_events + 1);
        assert!(matches!(
            sim.events.all().last().unwrap().kind,
            EventKind::ActionRejected { .. }
        ));
    }
}

/// The not-permitted variant needs the player to *know* the inner-circle
/// fact first; knowledge is not permission.
#[test]
fn knowledge_is_not_permission() {
    let (world, player, _b) = fixture();
    let mut sim = Simulation::new(world, 5, sim_config());
    let (tick, minute) = (sim.world.tick, sim.world.minute_of_day);
    rpg_social::record_belief(
        &mut sim.world,
        &mut sim.events,
        tick,
        minute,
        &player,
        &FactId::new("f_secret"),
        true,
        1.0,
        Provenance::Assumed,
    );
    let before = serde_json::to_string(&sim.world).unwrap();
    let out = apply(
        &mut sim,
        &player,
        Action::ShareClaim {
            target: ActorId::new("b"),
            fact: FactId::new("f_secret"),
            assertive: true,
            deceptive: false,
            retold_chain: None,
        },
    );
    assert!(matches!(
        out,
        ValidationOutcome::Rejected(RejectReason::NotPermittedToDisclose(_))
    ));
    assert_eq!(
        before,
        serde_json::to_string(&sim.world).unwrap(),
        "disclosure rejection must not mutate state"
    );
}

// ---------------------------------------------------------------------------
// T1.7 — event-sourced replay of the evidence ledger
// ---------------------------------------------------------------------------

#[test]
fn replay_reproduces_the_evidence_ledger_and_checksum() {
    let (world, player, _b) = fixture();
    let config = sim_config();
    let mut sim = Simulation::new(world, 5, config.clone());

    // A chain of claim sharing: player -> b, plus steps with routines.
    let _ = apply(
        &mut sim,
        &player,
        Action::ShareClaim {
            target: ActorId::new("b"),
            fact: FactId::new("f_x"),
            assertive: true,
            deceptive: false,
            retold_chain: None,
        },
    );
    let _ = apply(
        &mut sim,
        &player,
        Action::Speak {
            target: Some(ActorId::new("b")),
            text: "what do you know".into(),
            loudness: Loudness::Normal,
        },
    );
    for _ in 0..3 {
        let _ = sim.step(Vec::new());
    }

    assert!(
        !sim.world.epistemics.evidence.is_empty(),
        "the fixture must produce ledger entries"
    );

    let replayed = Simulation::replay_ops(sim.initial_world.clone(), sim.seed, config, &sim.ops);
    assert_eq!(
        serde_json::to_string(&sim.world).unwrap(),
        serde_json::to_string(&replayed.world).unwrap(),
        "proposal replay must reconstruct canonical state exactly, including epistemics"
    );
    assert_eq!(sim.events.checksum(), replayed.events.checksum());
    assert_eq!(
        serde_json::to_string(&sim.world.epistemics).unwrap(),
        serde_json::to_string(&replayed.world.epistemics).unwrap(),
        "the evidence ledger is event-sourced: replay reproduces it"
    );
}

// ---------------------------------------------------------------------------
// Shared report mechanic: claim separation at the authority
// ---------------------------------------------------------------------------

#[test]
fn report_crime_records_evidence_without_belief() {
    let (world, player, _b) = fixture();
    let mut sim = Simulation::new(world, 5, sim_config());
    // The authority has NOT heard the secret; the player reports it anyway.
    assert!(
        !sim.world
            .has_heard(&ActorId::new("b"), &FactId::new("f_secret")),
        "fixture precondition"
    );
    let out = apply(
        &mut sim,
        &player,
        Action::ReportCrime {
            authority: ActorId::new("b"),
            evidence: vec![FactId::new("f_secret")],
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted));
    // The authority *received* the report as provenance-bound evidence...
    let entries = sim
        .world
        .epistemics
        .evidence_about(&ActorId::new("b"), &FactId::new("f_secret"));
    assert_eq!(entries.len(), 1, "the report is on the authority's ledger");
    assert_eq!(
        entries[0].source_actor_id.as_ref().map(|a| a.as_str()),
        Some(PLAYER)
    );
    // ...but did not automatically come to believe it (claim separation).
    assert!(
        !sim.world
            .has_heard(&ActorId::new("b"), &FactId::new("f_secret")),
        "a report is not a belief: the authority's belief snapshot is untouched"
    );
    // The evidence write is event-sourced.
    assert!(sim.events.all().iter().any(|e| matches!(
        &e.kind,
        EventKind::EvidenceRecorded { observer, fact, .. }
            if observer.as_str() == "b" && fact.as_str() == "f_secret"
    )));
}

// ---------------------------------------------------------------------------
// Inspectable causal traces (Phase 0 deliverable)
// ---------------------------------------------------------------------------

#[test]
fn causal_traces_explain_goal_belief_event_cost() {
    let (world, player, _b) = fixture();
    let mut sim = Simulation::new(world, 5, sim_config());
    let out = apply(
        &mut sim,
        &player,
        Action::ShareClaim {
            target: ActorId::new("b"),
            fact: FactId::new("f_x"),
            assertive: true,
            deceptive: false,
            retold_chain: None,
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted));
    let trace = sim.causal_traces.last().unwrap();
    assert_eq!(trace.actor, player);
    assert_eq!(trace.action_kind, "share_claim");
    assert!(
        trace.because_goals.iter().any(|g| g.contains("secrets")),
        "the trace names the motivating goal: {trace:?}"
    );
    assert!(
        trace.because_beliefs.contains(&FactId::new("f_x")),
        "the trace names the belief behind the claim: {trace:?}"
    );
    // F4: a claim carries no resource cost, and none is invented — the cost
    // list stays empty instead of citing fabricated attention.
    assert!(
        trace.costs.is_empty(),
        "no fabricated cost for a costless action: {trace:?}"
    );
    assert!(
        trace.because_observations.is_empty(),
        "no perceived pre-action events in this fixture: {trace:?}"
    );
    assert!(
        !trace.effects.is_empty(),
        "the trace carries the produced effects"
    );
    let rendered = trace.render();
    assert!(rendered.contains("goal ["));
    assert!(rendered.contains("belief ["));
    assert!(rendered.contains("effects ["));
}

/// A real resource cost is still rendered: a payment costs coins.
#[test]
fn causal_trace_renders_real_transfer_costs() {
    let (world, player, _b) = fixture();
    let mut sim = Simulation::new(world, 5, sim_config());
    let _ = apply(
        &mut sim,
        &player,
        Action::Pay {
            target: ActorId::new("b"),
            amount: 25,
        },
    );
    let trace = sim.causal_traces.last().unwrap();
    assert_eq!(trace.costs, vec!["25 coins".to_string()], "{trace:?}");
    assert!(trace.render().contains("cost [25 coins]"));
}

/// Every validated proposal has exactly one outcome and one trace, in order,
/// including rejections (whose effects stay empty).
#[test]
fn one_outcome_and_trace_per_proposal() {
    let (world, player, _b) = fixture();
    let mut sim = Simulation::new(world, 5, sim_config());
    let _ = apply(&mut sim, &player, Action::Idle);
    let _ = apply(
        &mut sim,
        &player,
        Action::MoveTo {
            location: LocationId::new("nowhere"),
        },
    );
    let _ = apply(
        &mut sim,
        &player,
        Action::Pay {
            target: ActorId::new("b"),
            amount: 999_999,
        },
    );
    assert_eq!(sim.outcomes.len(), 3);
    assert_eq!(sim.causal_traces.len(), 3);
    assert_eq!(sim.outcomes[0].action_kind, "idle");
    assert_eq!(sim.outcomes[0].effects.len(), 0, "idle mutates nothing");
    assert!(matches!(
        sim.outcomes[1].validation,
        ValidationOutcome::Rejected(_)
    ));
    assert!(sim.outcomes[1].effects.is_empty());
    assert!(matches!(
        sim.outcomes[2].validation,
        ValidationOutcome::Rejected(_)
    ));
}

/// Rumor propagation through the ledger keeps the transmission chain (T1.2
/// integration with the legacy rumor path).
#[test]
fn rumor_propagation_keeps_a_provenance_chain() {
    use rpg_social as social;
    let (mut world, _player, _b) = fixture();
    let mut events = rpg_domain::events::EventLog::new();
    let mut rng = rpg_domain::rng::DeterministicRng::new(4);
    let (tick, minute) = (world.tick, world.minute_of_day);
    let speaker = ActorId::new(PLAYER);
    world.actor_mut(&speaker).unwrap().beliefs.insert(
        FactId::new("f_x"),
        Belief {
            fact: FactId::new("f_x"),
            believes_true: true,
            confidence: 0.9,
            source: Provenance::Assumed,
            acquired_tick: 0,
        },
    );
    let _ = social::share_claim(
        &mut world,
        &mut events,
        &mut rng,
        tick,
        minute,
        &speaker,
        &ActorId::new("holder"),
        &FactId::new("f_x"),
        true,
    );
    // The holder now has ledger evidence whose chain's originator is the
    // original speaker.
    let entries = world
        .epistemics
        .evidence_about(&ActorId::new("holder"), &FactId::new("f_x"));
    assert_eq!(entries.len(), 1);
    let chain = world
        .epistemics
        .chain(entries[0].source_chain_id)
        .cloned()
        .unwrap();
    assert_eq!(chain.originator.as_ref().map(|a| a.as_str()), Some(PLAYER));
    assert_eq!(chain.path, vec![speaker, ActorId::new("holder")]);
}

// ---------------------------------------------------------------------------
// Foundation repairs F1–F9 (bounded repair pass before Phase 2)
// ---------------------------------------------------------------------------

/// F1 + review D1-b: the authority's reading of a crime report is a directed
/// contextual appraisal — kernel-equality for the authority's own value
/// profile and prior edge, exactly once, citing the typed `CrimeReported`
/// event — never the legacy universal trust/respect delta. Directionality:
/// only the authority's edge moves.
#[test]
fn f1_report_crime_appraises_contextually_exactly_once_and_directional() {
    let (world, player, b) = fixture();
    let mut sim = Simulation::new(world, 5, sim_config());
    let (player_before, b_before) = (
        sim.world.actor(&player).unwrap().relationship(&b).clone(),
        sim.world.actor(&b).unwrap().relationship(&player).clone(),
    );
    let out = apply(
        &mut sim,
        &player,
        Action::ReportCrime {
            authority: b.clone(),
            evidence: vec![FactId::new("f_x")],
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    // Kernel equality: the authority's edge moved by EXACTLY the contextual
    // AccurateClaim appraisal the kernel computes for its own profile and
    // prior edge — no legacy universal delta may ride along.
    let authority = sim.world.actor(&b).unwrap();
    let profile = rpg_social::appraisal_profile(authority, &NeedsConfig::default());
    let b_edge_before = rpg_behavior::ContextualEdge::from_relationship(&b_before);
    let delta = rpg_behavior::appraise(
        &rpg_behavior::AppraisedEvent::AccurateClaim,
        &profile,
        &b_edge_before,
    );
    let expected = rpg_behavior::apply_delta(&b_edge_before, &delta);
    let rel = authority.relationship(&player);
    assert_eq!(
        rpg_behavior::ContextualEdge::from_relationship(&rel),
        expected,
        "production report-crime == the contextual kernel for the authority's profile"
    );
    // The legacy aggregate dims the removed fixed write used are untouched.
    assert_eq!(
        rel.trust, b_before.trust,
        "legacy universal trust delta is gone"
    );
    assert_eq!(rel.suspicion, b_before.suspicion, "no suspicion ride-along");
    assert_ne!(
        rel.trust_accuracy, b_before.trust_accuracy,
        "the appraisal actually moved the contextual edge"
    );
    // Directionality: the reporter's edge toward the authority is untouched.
    assert_eq!(
        sim.world.actor(&player).unwrap().relationship(&b),
        player_before,
        "Rel(reporter,authority) must not move: relationships are directed"
    );
    // Exactly one appraisal, citing the typed report as its causal origin.
    let report_event = sim
        .events
        .all()
        .iter()
        .find(|e| matches!(e.kind, EventKind::CrimeReported { .. }))
        .expect("the report is typed");
    let appraisals: Vec<_> = sim
        .events
        .all()
        .iter()
        .filter(|e| matches!(e.kind, EventKind::AppraisalRecorded { .. }))
        .collect();
    assert_eq!(appraisals.len(), 1, "exactly one appraisal per report");
    assert_eq!(
        appraisals[0].caused_by,
        vec![report_event.id],
        "the appraisal cites the real causal event"
    );
    // The mutation is typed (authority -> reporter) and appraised.
    let outcome = sim.outcomes.last().unwrap();
    assert!(outcome.effects.iter().any(|e| matches!(
        e,
        rpg_domain::behavior_types::Effect::ChangeTie { from, to, .. }
            if from == &b && to == &player
    )));
    assert!(outcome.effects.iter().any(|e| matches!(
        e,
        rpg_domain::behavior_types::Effect::Appraise { subject, .. } if subject == &b
    )));
}

/// F2: hearing an utterance builds familiarity with a typed tie change.
/// (Exposure-maintenance class — design §3.2 audit: directed, single-
/// dimension perceptual exposure from the real `SpeechEmitted` event, not a
/// valenced appraisal; classified background mutation pending M1/I-6.)
#[test]
fn f2_speak_builds_familiarity_with_a_change_tie_effect() {
    let (world, player, b) = fixture();
    let mut sim = Simulation::new(world, 5, sim_config());
    let out = apply(
        &mut sim,
        &player,
        Action::Speak {
            target: Some(b.clone()),
            text: "hello".into(),
            loudness: Loudness::Normal,
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted));
    let rel = sim.world.actor(&b).unwrap().relationship(&player);
    assert_eq!(rel.familiarity, 0.05, "exact familiarity delta");
    let outcome = sim.outcomes.last().unwrap();
    assert!(outcome.effects.iter().any(|e| matches!(
        e,
        rpg_domain::behavior_types::Effect::ChangeTie { from, to, .. }
            if from == &b && to == &player
    )));
}

/// F3: faction-budget consumption is honestly classified — never a
/// fabricated self-to-self transfer.
#[test]
fn f3_budget_actions_never_fabricate_a_self_transfer() {
    let (world, player, _b) = fixture();
    let mut sim = Simulation::new(world, 5, sim_config());
    let out = apply(
        &mut sim,
        &player,
        Action::AllocateBudget {
            faction: FactionId::new("watch"),
            amount: 50,
            purpose: "patrol".into(),
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted));
    let outcome = sim.outcomes.last().unwrap();
    for e in &outcome.effects {
        if let rpg_domain::behavior_types::Effect::Transfer { from, to, .. } = e {
            assert_ne!(
                from, to,
                "a budget action must not fabricate a self-to-self transfer"
            );
        }
    }
    // The consumption is typed as a Transform of the faction's budget object
    // with the real amount.
    assert!(outcome.effects.iter().any(|e| matches!(
        e,
        rpg_domain::behavior_types::Effect::Transform {
            consumed: 50,
            produced: 0,
            ..
        }
    )));
    assert_eq!(
        sim.world.faction(&FactionId::new("watch")).unwrap().budget,
        450,
        "the faction budget is truthfully deducted"
    );

    // Repair: same rule.
    sim.world.damage.insert(LocationId::new("shop"), 0.5);
    let out = apply(
        &mut sim,
        &player,
        Action::Repair {
            location: LocationId::new("shop"),
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted));
    let outcome = sim.outcomes.last().unwrap();
    for e in &outcome.effects {
        if let rpg_domain::behavior_types::Effect::Transfer { from, to, .. } = e {
            assert_ne!(
                from, to,
                "repair must not fabricate a self-to-self transfer"
            );
        }
    }
    // 0.5 damage -> cost 100; the worker (the player fixture stands in the
    // shop) performed the labor on site, so the paying faction also pays a
    // real wage of 25 (plan I-6 I6b §5.4): the budget is truthfully
    // deducted by cost + wage and the consumption is one Transform.
    let wage_paid = sim.events.all().iter().any(|e| {
        matches!(
            e.kind,
            rpg_domain::events::EventKind::WagePaid { amount: 25, .. }
        )
    });
    assert!(wage_paid, "on-site repair pays a real typed wage");
    assert_eq!(
        sim.world.actor(&player).unwrap().inventory.money,
        125,
        "the worker's custody gained exactly the wage"
    );
    assert_eq!(
        sim.world.faction(&FactionId::new("watch")).unwrap().budget,
        325
    );
    assert_eq!(
        outcome
            .effects
            .iter()
            .filter(|e| matches!(
                e,
                rpg_domain::behavior_types::Effect::Transform { consumed: 125, .. }
            ))
            .count(),
        1,
        "the budget consumption (repair + wage) is a Transform"
    );
    assert!(
        outcome.effects.iter().any(|e| matches!(
            e,
            rpg_domain::behavior_types::Effect::Transform {
                consumed: 0,
                produced: 1,
                ..
            }
        )),
        "the location restore is the other Transform"
    );
}

/// F4: traces cite only real causes — no most-recent-belief fallback, no
/// unperceived events, no fabricated attention cost.
#[test]
fn f4_trace_cites_only_real_causes() {
    let (world, player, _b) = fixture();
    let mut sim = Simulation::new(world, 5, sim_config());
    // An unrelated, just-acquired belief must not be cited for an action
    // about f_x.
    let (tick, minute) = (sim.world.tick, sim.world.minute_of_day);
    rpg_social::record_belief(
        &mut sim.world,
        &mut sim.events,
        tick,
        minute,
        &player,
        &FactId::new("f_unrelated"),
        true,
        0.9,
        Provenance::Assumed,
    );
    let out = apply(
        &mut sim,
        &player,
        Action::ShareClaim {
            target: ActorId::new("b"),
            fact: FactId::new("f_x"),
            assertive: true,
            deceptive: false,
            retold_chain: None,
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted));
    let trace = sim.causal_traces.last().unwrap();
    assert_eq!(
        trace.because_beliefs,
        vec![FactId::new("f_x")],
        "only the belief the action is about: {trace:?}"
    );
    assert!(trace.costs.is_empty(), "no fabricated attention: {trace:?}");
    assert!(trace.because_observations.is_empty());

    // Another actor's self-caused, unobserved event is not a cause the
    // player perceived: the player's next trace stays honest (empty).
    let _ = apply(
        &mut sim,
        &ActorId::new("b"),
        Action::ReportCrime {
            authority: ActorId::new("b"),
            evidence: vec![FactId::new("f_x")],
        },
    );
    let _ = apply(&mut sim, &player, Action::Idle);
    let trace = sim.causal_traces.last().unwrap();
    assert!(
        trace.because_observations.is_empty(),
        "events the actor never perceived are not causes: {trace:?}"
    );
    assert!(trace.costs.is_empty(), "idle carries no invented cost");
}

/// F7: a rumor's asserted polarity is the speaker's own belief, never the
/// world's canonical truth.
#[test]
fn f7_rumor_polarity_follows_speaker_belief_not_world_truth() {
    let (world, player, _b) = fixture();
    let mut sim = Simulation::new(world, 5, sim_config());
    // World truth for f_x is TRUE; the speaker believes it FALSE — they are
    // passing on a claim they themselves doubt.
    sim.world
        .actor_mut(&player)
        .unwrap()
        .beliefs
        .get_mut(&FactId::new("f_x"))
        .unwrap()
        .believes_true = false;
    let out = apply(
        &mut sim,
        &player,
        Action::SpreadRumor {
            text: "x is not so".into(),
            about: Some(FactId::new("f_x")),
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted));
    let outcome = sim.outcomes.last().unwrap();
    let asserted = outcome.effects.iter().find_map(|e| match e {
        rpg_domain::behavior_types::Effect::Assert {
            asserted_polarity, ..
        } => Some(*asserted_polarity),
        _ => None,
    });
    assert_eq!(
        asserted,
        Some(false),
        "the speaker's disbelief is what was asserted, not the world truth"
    );
}

/// F9: nonpositive payments are rejected before any mutation — full-world
/// neutrality, no money minting.
#[test]
fn f9_pay_rejects_nonpositive_values_without_minting() {
    let (world, player, _b) = fixture();
    let mut sim = Simulation::new(world, 5, sim_config());
    for amount in [0i64, -5] {
        let before = serde_json::to_string(&sim.world).unwrap();
        let before_events = sim.events.len();
        let player_money = sim.world.actor(&player).unwrap().inventory.money;
        let target_money = sim.world.actor(&ActorId::new("b")).unwrap().inventory.money;
        let out = apply(
            &mut sim,
            &player,
            Action::Pay {
                target: ActorId::new("b"),
                amount,
            },
        );
        assert!(
            matches!(out, ValidationOutcome::Rejected(RejectReason::Invalid(_))),
            "amount {amount} must be rejected, got {out:?}"
        );
        assert_eq!(
            before,
            serde_json::to_string(&sim.world).unwrap(),
            "a rejected nonpositive pay must not mutate canonical state \
             (full-world neutrality, amount {amount})"
        );
        assert_eq!(
            sim.world.actor(&player).unwrap().inventory.money,
            player_money,
            "no money minting"
        );
        assert_eq!(
            sim.world.actor(&ActorId::new("b")).unwrap().inventory.money,
            target_money
        );
        let outcome = sim.outcomes.last().unwrap();
        assert!(outcome.effects.is_empty());
        assert!(outcome.events.is_empty());
        assert_eq!(sim.events.len(), before_events + 1);
    }
    // A positive payment still works.
    let out = apply(
        &mut sim,
        &player,
        Action::Pay {
            target: ActorId::new("b"),
            amount: 7,
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted));
    assert_eq!(sim.world.actor(&player).unwrap().inventory.money, 93);
    assert_eq!(
        sim.world.actor(&ActorId::new("b")).unwrap().inventory.money,
        7
    );
}

// ---------------------------------------------------------------------------
// D1 repair: gift/payment relationships are directed contextual appraisals
// (plan I-4, review defect D1) — no legacy universal deltas, exactly once.
// ---------------------------------------------------------------------------

fn value_profiled_actor(id: &str, x: f32, tag: &str) -> Actor {
    let mut a = bare_actor(id, x, ActorKind::Npc);
    a.personality.tags.push(tag.to_string());
    a
}

/// A production payment/gift must produce EXACTLY the directed contextual
/// appraisal the kernel computes for the recipient's value profile — nothing
/// more (no legacy universal trust/affection/obligation/temptation delta),
/// nothing less. Directionality: only the recipient's edge moves. The same
/// event yields different outcomes for different profiles (context).
#[test]
fn d1_gift_and_payment_are_contextual_exactly_once_and_directional() {
    let (world, player, _b) = fixture();
    let mut sim = Simulation::new(world, 5, sim_config());
    let greedy = ActorId::new("greedy_npc");
    let proud = ActorId::new("proud_npc");
    sim.world.actors.insert(
        greedy.clone(),
        value_profiled_actor("greedy_npc", 2.5, "greedy"),
    );
    sim.world.actors.insert(
        proud.clone(),
        value_profiled_actor("proud_npc", 3.0, "proud"),
    );

    // Reference: the same GiftReceived appraisal applied to a fresh world —
    // the production lowering must equal it term for term.
    let reference = |tag: &str, value: f32| -> rpg_domain::world::Relationship {
        let (mut w2, _p, _b) = fixture();
        let donor = ActorId::new(PLAYER);
        let other = ActorId::new(tag);
        w2.actors.insert(
            other.clone(),
            value_profiled_actor(
                if tag == "greedy_npc" {
                    "greedy_npc"
                } else {
                    "proud_npc"
                },
                2.5,
                tag.strip_suffix("_npc").unwrap_or(tag),
            ),
        );
        rpg_social::apply_appraised_event(
            &mut w2,
            &other,
            &donor,
            rpg_behavior::AppraisedEvent::GiftReceived { value },
            None,
            &EmotionsConfig::default(),
            &NeedsConfig::default(),
        )
        .expect("both actors exist");
        w2.actor(&other).unwrap().relationship(&donor).clone()
    };

    // The greedy recipient is delighted; the proud one reads the same
    // payment as (partial) bribery — context-dependent outcomes.
    let out = apply(
        &mut sim,
        &player,
        Action::Pay {
            target: greedy.clone(),
            amount: 50,
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    let out = apply(
        &mut sim,
        &player,
        Action::Pay {
            target: proud.clone(),
            amount: 50,
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    let greedy_rel = sim
        .world
        .actor(&greedy)
        .unwrap()
        .relationship(&player)
        .clone();
    let proud_rel = sim
        .world
        .actor(&proud)
        .unwrap()
        .relationship(&player)
        .clone();
    assert_eq!(
        greedy_rel,
        reference("greedy_npc", 0.5),
        "production payment lowering == the contextual kernel for the greedy profile"
    );
    assert_eq!(
        proud_rel,
        reference("proud_npc", 0.5),
        "production payment lowering == the contextual kernel for the proud profile"
    );
    assert_ne!(
        greedy_rel.affection, proud_rel.affection,
        "the same event, different value profiles: context-dependent outcome"
    );
    // No legacy universal delta may ride along: the aggregate `trust` and
    // `temptation` dims are written only by the removed `apply_gift`.
    for rel in [&greedy_rel, &proud_rel] {
        assert_eq!(rel.trust, 0.0, "legacy universal trust delta is gone");
        assert_eq!(
            rel.temptation, 0.0,
            "legacy universal temptation delta is gone"
        );
    }
    // Directionality: the payer's edge toward the recipients is untouched.
    for target in [&greedy, &proud] {
        let payer_rel = sim.world.actor(&player).unwrap().relationship(target);
        assert_eq!(
            payer_rel,
            rpg_domain::world::Relationship::default(),
            "Rel(payer,recipient) must not move: relationships are directed"
        );
    }
    // Exactly one appraisal per payment.
    let appraisals = sim
        .events
        .all()
        .iter()
        .filter(|e| matches!(e.kind, EventKind::AppraisalRecorded { .. }))
        .count();
    assert_eq!(appraisals, 2, "exactly one appraisal per payment");

    // Item gifts take the same route (the former double-count: the legacy
    // delta plus the contextual appraisal).
    sim.world
        .actor_mut(&player)
        .unwrap()
        .inventory
        .add(ItemId::new("coin"), 1);
    let out = apply(
        &mut sim,
        &player,
        Action::TransferItem {
            target: greedy.clone(),
            item: ItemId::new("coin"),
            qty: 1,
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    let after = sim
        .world
        .actor(&greedy)
        .unwrap()
        .relationship(&player)
        .clone();
    // coin base_value 10 -> value_norm 0.1: the delta must equal exactly one
    // contextual GiftReceived(0.1) applied to the pre-gift edge. Direct
    // equality against the same kernel entry point is the strongest
    // exactly-once statement.
    let donor = ActorId::new(PLAYER);
    let other = ActorId::new("greedy_npc");
    let reference_after_gift = {
        let (mut w3, _p, _b) = fixture();
        w3.actors.insert(
            other.clone(),
            value_profiled_actor("greedy_npc", 2.5, "greedy"),
        );
        // Reproduce the production sequence: pay 50 then gift 1 coin.
        rpg_social::apply_appraised_event(
            &mut w3,
            &other,
            &donor,
            rpg_behavior::AppraisedEvent::GiftReceived { value: 0.5 },
            None,
            &EmotionsConfig::default(),
            &NeedsConfig::default(),
        );
        rpg_social::apply_appraised_event(
            &mut w3,
            &other,
            &donor,
            rpg_behavior::AppraisedEvent::GiftReceived { value: 0.1 },
            None,
            &EmotionsConfig::default(),
            &NeedsConfig::default(),
        );
        w3.actor(&other).unwrap().relationship(&donor).clone()
    };
    assert_eq!(
        after, reference_after_gift,
        "gift delta == exactly one additional contextual appraisal (no double count)"
    );
    // Exactly one appraisal per item gift as well.
    let appraisals = sim
        .events
        .all()
        .iter()
        .filter(|e| matches!(e.kind, EventKind::AppraisalRecorded { .. }))
        .count();
    assert_eq!(appraisals, 3, "pay, pay, gift: exactly one appraisal each");
}

/// Rejection neutrality for the gift/payment pathway: a rejected payment
/// writes no appraisal, no relationship change, and no events beyond the
/// typed rejection.
#[test]
fn d1_rejected_payment_is_appraisal_neutral() {
    let (world, player, _b) = fixture();
    let mut sim = Simulation::new(world, 5, sim_config());
    let before_world = serde_json::to_string(&sim.world).unwrap();
    let before_events = sim.events.len();
    let out = apply(
        &mut sim,
        &player,
        Action::Pay {
            target: ActorId::new("b"),
            amount: 999_999,
        },
    );
    assert!(matches!(out, ValidationOutcome::Rejected(_)), "{out:?}");
    assert_eq!(
        before_world,
        serde_json::to_string(&sim.world).unwrap(),
        "a rejected payment must not move a single relationship dimension"
    );
    assert_eq!(
        sim.events.len(),
        before_events + 1,
        "only the typed ActionRejected event is recorded"
    );
    let outcome = sim.outcomes.last().unwrap();
    assert!(outcome.effects.is_empty(), "rejection carries no effects");
}

// ---------------------------------------------------------------------------
// I-5 G1 (T1.9): canonical deceptive assertion, end to end
// ---------------------------------------------------------------------------

#[test]
fn t1_9_deceptive_assertion_canonical_end_to_end() {
    let (mut world, player, b) = fixture();
    // f_w: world truth TRUE. The player authored-believes TRUE; `b`
    // authored-believes FALSE (a knower who disbelieves — exactly the shape
    // of an intentional liar).
    world.facts.insert(
        FactId::new("f_w"),
        Fact {
            id: FactId::new("f_w"),
            statement: "the shipment was diverted".into(),
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
    world.actor_mut(&player).unwrap().beliefs.insert(
        FactId::new("f_w"),
        Belief {
            fact: FactId::new("f_w"),
            believes_true: true,
            confidence: 0.9,
            source: Provenance::Assumed,
            acquired_tick: 0,
        },
    );
    world.actor_mut(&b).unwrap().beliefs.insert(
        FactId::new("f_w"),
        Belief {
            fact: FactId::new("f_w"),
            believes_true: false,
            confidence: 0.8,
            source: Provenance::Assumed,
            acquired_tick: 0,
        },
    );
    // The player also illicitly learned the inner-circle secret (knowledge
    // is not permission).
    world.actor_mut(&player).unwrap().beliefs.insert(
        FactId::new("f_secret"),
        Belief {
            fact: FactId::new("f_secret"),
            believes_true: true,
            confidence: 0.9,
            source: Provenance::Assumed,
            acquired_tick: 0,
        },
    );
    // f_unheard exists but nobody holds it: the knowledge floor case.
    world.facts.insert(
        FactId::new("f_unheard"),
        Fact {
            id: FactId::new("f_unheard"),
            statement: "never spoken of".into(),
            truth: true,
            sensitivity: Sensitivity::Public,
            category: FactCategory::Observation,
            prerequisites: vec![],
            eligible_holders: vec![],
            eligible_containers: vec![],
            committed: false,
            placed_at: None,
        },
    );
    let mut sim = Simulation::new(world, 5, sim_config());

    // (a) The truthful path keeps the speaker-believes-true restriction: a
    // disbelieving speaker cannot smuggle a lie through it.
    let out = apply(
        &mut sim,
        &b,
        Action::ShareClaim {
            target: player.clone(),
            fact: FactId::new("f_w"),
            assertive: true,
            deceptive: false,
            retold_chain: None,
        },
    );
    assert!(
        matches!(
            out,
            ValidationOutcome::Rejected(RejectReason::LacksKnowledge(_))
        ),
        "the truthful path requires the speaker to believe the claim: {out:?}"
    );

    // (b) No world-truth leakage: the player believes TRUE and lies — the
    // asserted polarity is FALSE (the negation of the speaker's belief,
    // never the world truth), recorded faithfully on the recipient's ledger.
    let player_belief_at_dispatch = sim
        .world
        .actor(&player)
        .unwrap()
        .beliefs
        .get(&FactId::new("f_w"))
        .unwrap()
        .believes_true;
    assert!(player_belief_at_dispatch);
    let out = apply(
        &mut sim,
        &player,
        Action::ShareClaim {
            target: b.clone(),
            fact: FactId::new("f_w"),
            assertive: true,
            deceptive: true,
            retold_chain: None,
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    let delivered = sim.world.epistemics.evidence_about(&b, &FactId::new("f_w"));
    assert_eq!(delivered.len(), 1, "exactly one evidence item per delivery");
    assert!(
        !delivered[0].asserted_polarity,
        "the lie is recorded at the asserted polarity, not the speaker's belief"
    );
    assert_ne!(
        delivered[0].asserted_polarity,
        sim.world.facts.get(&FactId::new("f_w")).unwrap().truth,
        "the recorded content is the lie — the world's truth is never consulted"
    );
    let lie_event = sim
        .events
        .all()
        .iter()
        .find(|e| {
            matches!(&e.kind,
                EventKind::ClaimShared { from, deceptive: true, .. } if from == &player)
        })
        .expect("the M2 audit marker rides the ClaimShared delivery event");
    assert!(
        matches!(
            &lie_event.kind,
            EventKind::ClaimShared {
                deceptive: true,
                ..
            }
        ),
        "the intentional lie is marked in the event log"
    );

    // (c) The mirrored lie: `b` (who authored-disbelieved) asserts the
    // opposite polarity. The expectation is read at dispatch time — the
    // recipient-side belief snapshot is the recipient's own rolled state,
    // never an input to the speaker's polarity.
    let b_belief_at_dispatch = sim
        .world
        .actor(&b)
        .unwrap()
        .beliefs
        .get(&FactId::new("f_w"))
        .map(|v| v.believes_true);
    let out = apply(
        &mut sim,
        &b,
        Action::ShareClaim {
            target: player.clone(),
            fact: FactId::new("f_w"),
            assertive: true,
            deceptive: true,
            retold_chain: None,
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    let outcome = sim.outcomes.last().unwrap();
    let asserted = outcome.effects.iter().find_map(|e| match e {
        rpg_domain::behavior_types::Effect::Assert {
            asserted_polarity, ..
        } => Some(*asserted_polarity),
        _ => None,
    });
    assert_eq!(
        asserted,
        b_belief_at_dispatch.map(|v| !v),
        "the deceptive lowering asserts exactly the negation of the speaker's belief"
    );
    let delivered = sim
        .world
        .epistemics
        .evidence_about(&player, &FactId::new("f_w"));
    assert_eq!(delivered.len(), 1);
    assert_eq!(delivered[0].asserted_polarity, asserted.unwrap());
    assert!(
        sim.world.facts.get(&FactId::new("f_w")).unwrap().truth,
        "a lie is a proposition choice, never a truth mutation"
    );

    // (d) Knowledge is still the floor: nobody lies about what they have
    // never even heard of.
    let out = apply(
        &mut sim,
        &player,
        Action::ShareClaim {
            target: b.clone(),
            fact: FactId::new("f_unheard"),
            assertive: true,
            deceptive: true,
            retold_chain: None,
        },
    );
    assert!(
        matches!(
            out,
            ValidationOutcome::Rejected(RejectReason::LacksKnowledge(_))
        ),
        "deception does not bypass the knowledge boundary: {out:?}"
    );

    // (e) …and knowledge is not permission: lying about an inner-circle
    // secret is still a forbidden disclosure.
    let out = apply(
        &mut sim,
        &player,
        Action::ShareClaim {
            target: b.clone(),
            fact: FactId::new("f_secret"),
            assertive: true,
            deceptive: true,
            retold_chain: None,
        },
    );
    assert!(
        matches!(
            out,
            ValidationOutcome::Rejected(RejectReason::NotPermittedToDisclose(_))
        ),
        "deception does not relax the disclosure permission: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// I-5 G2 (T4.9): authoritative QUERY lowering — no omniscient answer
// ---------------------------------------------------------------------------

#[test]
fn t4_9_query_lowering_answers_from_ledger_never_world_truth() {
    let (mut world, player, b) = fixture();
    // f_u: committed world truth TRUE; nobody holds evidence about it.
    world.facts.insert(
        FactId::new("f_u"),
        Fact {
            id: FactId::new("f_u"),
            statement: "true but unobserved".into(),
            truth: true,
            sensitivity: Sensitivity::Public,
            category: FactCategory::Observation,
            prerequisites: vec![],
            eligible_holders: vec![],
            eligible_containers: vec![],
            committed: true,
            placed_at: None,
        },
    );
    // f_lie: committed world truth TRUE; `b` authored-DISbelieves it.
    world.facts.insert(
        FactId::new("f_lie"),
        Fact {
            id: FactId::new("f_lie"),
            statement: "true, doubted by b".into(),
            truth: true,
            sensitivity: Sensitivity::Public,
            category: FactCategory::Observation,
            prerequisites: vec![],
            eligible_holders: vec![],
            eligible_containers: vec![],
            committed: true,
            placed_at: None,
        },
    );
    world.actor_mut(&b).unwrap().beliefs.insert(
        FactId::new("f_lie"),
        Belief {
            fact: FactId::new("f_lie"),
            believes_true: false,
            confidence: 0.9,
            source: Provenance::Assumed,
            acquired_tick: 0,
        },
    );
    let mut sim = Simulation::new(world, 5, sim_config());

    // (1) A target with NO evidence answers Unknown even though the world's
    // truth says otherwise — no omniscient answer.
    let out = apply(
        &mut sim,
        &player,
        Action::Query {
            target: b.clone(),
            about: FactId::new("f_u"),
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    let answered = sim
        .events
        .all()
        .iter()
        .rev()
        .find(|e| matches!(&e.kind, EventKind::QueryAnswered { answerer, .. } if answerer == &b))
        .unwrap();
    assert!(matches!(
        &answered.kind,
        EventKind::QueryAnswered { unknown: true, polarities, .. } if polarities.is_empty()
    ));
    assert!(
        sim.world
            .epistemics
            .evidence_about(&player, &FactId::new("f_u"))
            .is_empty(),
        "an Unknown answer delivers no evidence"
    );
    assert!(
        !sim.world
            .actor(&player)
            .unwrap()
            .beliefs
            .contains_key(&FactId::new("f_u")),
        "a question never writes the asker's belief snapshot"
    );
    assert!(
        sim.world.facts.get(&FactId::new("f_u")).unwrap().truth,
        "world truth untouched"
    );

    // (2) The authored-knowledge fallback reports the target's OWN stance:
    // `b` disbelieves f_lie, so the answer polarity is FALSE — the world
    // truth (true) is never leaked through the answer.
    let out = apply(
        &mut sim,
        &player,
        Action::Query {
            target: b.clone(),
            about: FactId::new("f_lie"),
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    let answered = sim
        .events
        .all()
        .iter()
        .rev()
        .find(|e| {
            matches!(
                &e.kind,
                EventKind::QueryAnswered { answerer, about, .. }
                    if answerer == &b && *about == FactId::new("f_lie")
            )
        })
        .unwrap();
    assert!(matches!(
        &answered.kind,
        EventKind::QueryAnswered { unknown: false, polarities, .. } if polarities == &vec![false]
    ));
    // The answer was delivered: the asker now holds provenance-bound
    // evidence at the answered (false) polarity.
    let delivered = sim
        .world
        .epistemics
        .evidence_about(&player, &FactId::new("f_lie"));
    assert_eq!(delivered.len(), 1);
    assert!(!delivered[0].asserted_polarity);
    assert_eq!(
        delivered[0].source_actor_id.as_ref().map(|a| a.as_str()),
        Some("b")
    );
    assert!(sim.world.facts.get(&FactId::new("f_lie")).unwrap().truth);

    // (3) A non-empty ledger wins over the target's authored belief: plant a
    // FALSE-polarity ledger entry for f_x on `b` (who authored-believes
    // true) and the answer follows the LEDGER, not the belief and not the
    // world truth.
    let (tick, minute) = (sim.world.tick, sim.world.minute_of_day);
    rpg_behavior::record_provenance_evidence(
        &mut sim.world,
        &mut sim.events,
        tick,
        minute,
        &b,
        &FactId::new("f_x"),
        false,
        &Provenance::Told {
            by: player.clone(),
            event: None,
        },
    );
    let out = apply(
        &mut sim,
        &player,
        Action::Query {
            target: b.clone(),
            about: FactId::new("f_x"),
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    let delivered = sim
        .world
        .epistemics
        .evidence_about(&player, &FactId::new("f_x"));
    let last = delivered.last().unwrap();
    assert!(
        !last.asserted_polarity,
        "the answer follows the target's ledger, not the target's belief and not world truth"
    );
    assert!(sim.world.facts.get(&FactId::new("f_x")).unwrap().truth);

    // (4) An unknown proposition cannot be asked about at all.
    let out = apply(
        &mut sim,
        &player,
        Action::Query {
            target: b.clone(),
            about: FactId::new("f_nothing"),
        },
    );
    assert!(
        matches!(
            out,
            ValidationOutcome::Rejected(RejectReason::UnknownFact(_))
        ),
        "{out:?}"
    );
}

// ---------------------------------------------------------------------------
// I-5 G3 (T4.10): authoritative SANCTION lowering — authority, process,
// cost, evidence; not-permitted ≠ not-possible
// ---------------------------------------------------------------------------

#[test]
fn t4_10_sanction_requires_authority_process_cost_and_evidence() {
    let (world, player, b) = fixture();
    let mut sim = Simulation::new(world, 5, sim_config());

    // NOT-PERMITTED: a proposer with no institutional authority is rejected
    // as a capability failure — a different rejection class than the
    // not-possible cases below.
    let out = apply(
        &mut sim,
        &player,
        Action::Sanction {
            target: b.clone(),
            kind: rpg_domain::action::SanctionKind::Fine { amount: 10 },
            evidence: vec![FactId::new("f_x")],
        },
    );
    assert!(
        matches!(
            out,
            ValidationOutcome::Rejected(RejectReason::NoCapability(_))
        ),
        "unauthorized sanction: {out:?}"
    );
    assert_eq!(
        sim.world.actor(&b).unwrap().inventory.money,
        0,
        "nothing applied"
    );

    // NOT-POSSIBLE: the same act against an unknown subject is a different
    // rejection class entirely.
    let out = apply(
        &mut sim,
        &b,
        Action::Sanction {
            target: ActorId::new("nobody"),
            kind: rpg_domain::action::SanctionKind::Fine { amount: 10 },
            evidence: vec![FactId::new("f_x")],
        },
    );
    assert!(
        matches!(
            out,
            ValidationOutcome::Rejected(RejectReason::UnknownActor(_))
        ),
        "{out:?}"
    );

    // EVIDENCE prerequisite: the authority must hold what it cites.
    let out = apply(
        &mut sim,
        &b,
        Action::Sanction {
            target: player.clone(),
            kind: rpg_domain::action::SanctionKind::Fine { amount: 10 },
            evidence: vec![FactId::new("f_secret")],
        },
    );
    assert!(
        matches!(
            out,
            ValidationOutcome::Rejected(RejectReason::LacksKnowledge(_))
        ),
        "{out:?}"
    );
    // …and citing nothing at all is invalid.
    let out = apply(
        &mut sim,
        &b,
        Action::Sanction {
            target: player.clone(),
            kind: rpg_domain::action::SanctionKind::Fine { amount: 10 },
            evidence: vec![],
        },
    );
    assert!(
        matches!(out, ValidationOutcome::Rejected(RejectReason::Invalid(_))),
        "{out:?}"
    );

    // AUTHORIZED + EVIDENCED: the fine is a real transfer (the cost), the
    // process is the typed SanctionIssued event, and the subject's
    // contextual appraisal is recorded exactly once.
    let out = apply(
        &mut sim,
        &b,
        Action::Sanction {
            target: player.clone(),
            kind: rpg_domain::action::SanctionKind::Fine { amount: 40 },
            evidence: vec![FactId::new("f_x")],
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    assert_eq!(
        sim.world.actor(&player).unwrap().inventory.money,
        60,
        "the fine was paid"
    );
    assert_eq!(
        sim.world.actor(&b).unwrap().inventory.money,
        40,
        "conservation"
    );
    let outcome = sim.outcomes.last().unwrap();
    let families: Vec<&str> = outcome.effects.iter().map(|e| e.family()).collect();
    for want in ["sanction", "transfer", "appraise"] {
        assert!(
            families.contains(&want),
            "fine must lower into {want}: {families:?}"
        );
    }
    let issued = sim
        .events
        .all()
        .iter()
        .rev()
        .find(|e| matches!(&e.kind, EventKind::SanctionIssued { .. }))
        .unwrap();
    assert!(matches!(
        &issued.kind,
        EventKind::SanctionIssued { authority, subject, kind, evidence }
            if authority == &b && subject == &player && kind == "fine" && evidence.len() == 1
    ));
    let appraisals = sim
        .events
        .all()
        .iter()
        .filter(|e| matches!(e.kind, EventKind::AppraisalRecorded { .. }))
        .count();
    assert_eq!(appraisals, 1, "exactly one subject appraisal per sanction");

    // COST prerequisite: a fine the subject cannot pay is rejected before
    // any mutation (not-possible).
    let out = apply(
        &mut sim,
        &b,
        Action::Sanction {
            target: player.clone(),
            kind: rpg_domain::action::SanctionKind::Fine { amount: 999 },
            evidence: vec![FactId::new("f_x")],
        },
    );
    assert!(
        matches!(
            out,
            ValidationOutcome::Rejected(RejectReason::InsufficientFunds)
        ),
        "{out:?}"
    );
    assert_eq!(
        sim.world.actor(&player).unwrap().inventory.money,
        60,
        "no partial fine"
    );

    // Institutional praise: no cost, real appraisal.
    let out = apply(
        &mut sim,
        &b,
        Action::Sanction {
            target: player.clone(),
            kind: rpg_domain::action::SanctionKind::Praise,
            evidence: vec![FactId::new("f_x")],
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    assert_eq!(
        sim.world.actor(&player).unwrap().inventory.money,
        60,
        "praise moves no money"
    );
    let subject_emotions = sim.world.actor(&player).unwrap().emotions.clone();
    assert!(
        subject_emotions.value(EmotionKind::Pride) > 0.0,
        "institutional praise writes the pride emotion (mind-local, I-4b)"
    );
}

/// Arrest removes the subject from play as the sanction's custody effect.
#[test]
fn t4_10_arrest_takes_custody_through_the_authority_route() {
    let (world, player, b) = fixture();
    let mut sim = Simulation::new(world, 5, sim_config());
    let out = apply(
        &mut sim,
        &b,
        Action::Sanction {
            target: player.clone(),
            kind: rpg_domain::action::SanctionKind::Arrest,
            evidence: vec![FactId::new("f_x")],
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    assert_eq!(
        sim.world.actor(&player).unwrap().status,
        rpg_domain::world::ActorStatus::Departed,
        "the arrest took custody"
    );
    assert!(matches!(
        sim.outcomes.last().unwrap().effects.iter().find(|e| e.family() == "sanction"),
        Some(rpg_domain::behavior_types::Effect::Sanction { authority, subject, .. })
            if authority == &b && subject == &player
    ));
}

// ---------------------------------------------------------------------------
// Review D1-a: QUERY delivery is gated by the ShareClaim disclosure
// permission (design §3.1/§9.1.2) — refusals are truthful, typed, distinct
// from ignorance, and leak nothing.
// ---------------------------------------------------------------------------

/// A Private/Secret/InnerCircle fact whose canonical holder is the third
/// party `c`: `b` can be *taught* it without becoming authorized to disclose
/// it (knowledge is not permission).
fn sensitive_fact(id: &str, statement: &str, sensitivity: Sensitivity, holder: &str) -> Fact {
    Fact {
        id: FactId::new(id),
        statement: statement.into(),
        truth: true,
        sensitivity,
        category: FactCategory::Organization,
        prerequisites: vec![],
        eligible_holders: vec![ActorId::new(holder)],
        eligible_containers: vec![],
        committed: true,
        placed_at: None,
    }
}

/// Production QUERY dispatch (review D1-a; design §3.1 "protect secrets by
/// restricting disclosure choices", §9.1.2; plan I-5 G2): the ShareClaim
/// authorization gates EVERY answer delivery — ledger answers and the
/// authored-belief fallback alike — across all sensitive classes. Anyone may
/// ask (no asker-knowledge requirement); an unauthorized answerer refuses; an
/// authorized answerer still answers epistemically — never world truth.
#[test]
fn t4_9b_query_respects_disclosure_permission_and_refuses_truthfully() {
    let (mut world, player, b) = fixture();
    // Forbidden-to-answer facts: canonical holder is the third party `c`.
    world.facts.insert(
        FactId::new("f_private"),
        sensitive_fact("f_private", "a private matter", Sensitivity::Private, "c"),
    );
    world.facts.insert(
        FactId::new("f_secretfact"),
        sensitive_fact("f_secretfact", "a secret", Sensitivity::Secret, "c"),
    );
    world.facts.insert(
        FactId::new("f_inner2"),
        sensitive_fact(
            "f_inner2",
            "an inner-circle matter",
            Sensitivity::InnerCircle,
            "c",
        ),
    );
    // Authored-fallback facts (no ledger entries): `b` authored-believes the
    // allowed one FALSE while the world truth is TRUE — the delivered
    // polarity must follow the belief, never the truth.
    world.facts.insert(
        FactId::new("f_auth_priv"),
        sensitive_fact(
            "f_auth_priv",
            "a private authored belief",
            Sensitivity::Private,
            "c",
        ),
    );
    world.facts.insert(
        FactId::new("f_auth_priv_ok"),
        sensitive_fact(
            "f_auth_priv_ok",
            "an owned private authored belief",
            Sensitivity::Private,
            "b",
        ),
    );
    // Authorized-to-answer variants: `b` is the canonical holder.
    world.facts.insert(
        FactId::new("f_private_ok"),
        sensitive_fact(
            "f_private_ok",
            "an owned private fact",
            Sensitivity::Private,
            "b",
        ),
    );
    world.facts.insert(
        FactId::new("f_secret_ok"),
        sensitive_fact("f_secret_ok", "an owned secret", Sensitivity::Secret, "b"),
    );
    world.facts.insert(
        FactId::new("f_inner_ok"),
        sensitive_fact(
            "f_inner_ok",
            "an owned inner-circle fact",
            Sensitivity::InnerCircle,
            "b",
        ),
    );
    let mut sim = Simulation::new(world, 5, sim_config());

    // Teach `b` every ledger fact (evidence, not belief snapshots).
    let (tick, minute) = (sim.world.tick, sim.world.minute_of_day);
    for fact in [
        "f_private",
        "f_secretfact",
        "f_inner2",
        "f_private_ok",
        "f_secret_ok",
        "f_inner_ok",
    ] {
        rpg_behavior::record_provenance_evidence(
            &mut sim.world,
            &mut sim.events,
            tick,
            minute,
            &b,
            &FactId::new(fact),
            true,
            &Provenance::Assumed,
        );
    }
    // The authored fallback: `b` authored-believes both fallback facts FALSE
    // (no ledger entries — answer_query must take the belief path).
    for fact in ["f_auth_priv", "f_auth_priv_ok"] {
        sim.world.actor_mut(&b).unwrap().beliefs.insert(
            FactId::new(fact),
            Belief {
                fact: FactId::new(fact),
                believes_true: false,
                confidence: 0.9,
                source: Provenance::Assumed,
                acquired_tick: 0,
            },
        );
    }
    // The asker knows NOTHING about any of these facts: questions are
    // permitted without the asker knowing the fact (validated in-range and
    // fact-exists only).
    assert!(
        !sim.world
            .actor(&player)
            .unwrap()
            .beliefs
            .contains_key(&FactId::new("f_private")),
        "fixture precondition: the asker does not know the fact"
    );

    let refused_of = |sim: &Simulation, fact: &str| {
        sim.events
            .all()
            .iter()
            .rev()
            .find(|e| {
                matches!(
                    &e.kind,
                    EventKind::QueryAnswered { answerer, about, .. }
                        if answerer == &b && about.as_str() == fact
                )
            })
            .map(|e| e.kind.clone())
            .expect("an answer event was recorded")
    };
    let ask = |sim: &mut Simulation, fact: &str| -> ValidationOutcome {
        apply(
            sim,
            &player,
            Action::Query {
                target: b.clone(),
                about: FactId::new(fact),
            },
        )
    };

    // (1) FORBIDDEN: every sensitive class — ledger evidence in hand, the
    // unauthorized answerer refuses. The question itself is ACCEPTED
    // (asking needs no knowledge); the refusal transmits no polarity,
    // confidence or evidence and writes nothing into the asker's mind.
    for fact in ["f_private", "f_secretfact", "f_inner2"] {
        let b_evidence_before = sim
            .world
            .epistemics
            .evidence_about(&b, &FactId::new(fact))
            .len();
        let out = ask(&mut sim, fact);
        assert!(
            matches!(out, ValidationOutcome::Accepted),
            "the question about {fact} must be permitted: {out:?}"
        );
        assert!(
            matches!(
                refused_of(&sim, fact),
                EventKind::QueryAnswered { refused: true, unknown: false, polarities, confidence: 0.0, .. }
                    if polarities.is_empty()
            ),
            "{fact}: the unauthorized answerer refuses, distinctly from ignorance: {out:?}"
        );
        assert!(
            !player_knows_marker(&sim, fact),
            "{fact}: the refusal delivers no evidence and no belief"
        );
        assert_eq!(
            sim.world
                .epistemics
                .evidence_about(&b, &FactId::new(fact))
                .len(),
            b_evidence_before,
            "{fact}: the secret is not removed from the refusing answerer's memory"
        );
        assert!(
            sim.world.facts.get(&FactId::new(fact)).unwrap().truth,
            "{fact}: world truth untouched"
        );
        // The refusal moves no tie and shares no claim.
        let outcome = sim.outcomes.last().unwrap();
        assert!(
            !outcome
                .effects
                .iter()
                .any(|e| matches!(e, rpg_domain::behavior_types::Effect::ChangeTie { .. })),
            "{fact}: a refusal is not exposure: {outcome:?}"
        );
        assert!(
            !outcome
                .effects
                .iter()
                .any(|e| matches!(e, rpg_domain::behavior_types::Effect::Assert { .. })),
            "{fact}: a refusal asserts nothing: {outcome:?}"
        );
        assert!(
            !sim.events.all().iter().any(|e| matches!(
                &e.kind,
                EventKind::ClaimShared { fact: f, .. } if f.as_str() == fact
            )),
            "{fact}: no claim delivery happened"
        );
    }

    // (2) AUTHORIZED: the canonical holder answers the same kinds of
    // question (allowed recipient: the holder may tell anyone) — the answer
    // stays epistemically bounded (the holder's own polarity at ledger or
    // belief confidence), never world truth.
    let out = ask(&mut sim, "f_private_ok");
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    assert!(matches!(
        refused_of(&sim, "f_private_ok"),
        EventKind::QueryAnswered { refused: false, unknown: false, polarities, .. }
            if polarities == vec![true]
    ));
    assert!(
        player_knows_marker(&sim, "f_private_ok"),
        "the authorized answer was delivered as provenance-bound evidence"
    );
    let out = ask(&mut sim, "f_secret_ok");
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    assert!(matches!(
        refused_of(&sim, "f_secret_ok"),
        EventKind::QueryAnswered {
            refused: false,
            unknown: false,
            ..
        }
    ));
    assert!(player_knows_marker(&sim, "f_secret_ok"));
    // InnerCircle: the flag is required in ADDITION to holder status.
    let out = ask(&mut sim, "f_inner_ok");
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    assert!(
        matches!(
            refused_of(&sim, "f_inner_ok"),
            EventKind::QueryAnswered { refused: true, .. }
        ),
        "an inner-circle holder without the flag refuses"
    );
    assert!(!player_knows_marker(&sim, "f_inner_ok"));
    sim.world
        .actor_mut(&b)
        .unwrap()
        .flags
        .insert("inner_circle".into());
    let out = ask(&mut sim, "f_inner_ok");
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    assert!(matches!(
        refused_of(&sim, "f_inner_ok"),
        EventKind::QueryAnswered {
            refused: false,
            unknown: false,
            ..
        }
    ));
    assert!(player_knows_marker(&sim, "f_inner_ok"));

    // (3) AUTHORED FALLBACK: the same gate applies to belief-only answers.
    // Unauthorized: refused. Authorized: delivered at the BELIEF's polarity
    // (false) — the world truth (true) never leaks through the answer.
    let out = ask(&mut sim, "f_auth_priv");
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    assert!(matches!(
        refused_of(&sim, "f_auth_priv"),
        EventKind::QueryAnswered { refused: true, unknown: false, polarities, .. }
            if polarities.is_empty()
    ));
    assert!(!player_knows_marker(&sim, "f_auth_priv"));
    let out = ask(&mut sim, "f_auth_priv_ok");
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    assert!(
        matches!(
            refused_of(&sim, "f_auth_priv_ok"),
            EventKind::QueryAnswered { refused: false, unknown: false, polarities, .. }
                if polarities == vec![false]
        ),
        "the authored fallback reports the answerer's own stance, not world truth"
    );
    assert!(player_knows_marker(&sim, "f_auth_priv_ok"));
    assert!(
        sim.world
            .facts
            .get(&FactId::new("f_auth_priv_ok"))
            .unwrap()
            .truth,
        "world truth untouched by the delivered authored answer"
    );

    // (4) IGNORANCE preserved: a public fact nobody holds stays Unknown —
    // distinct from a refusal (refused=false there).
    sim.world.facts.insert(
        FactId::new("f_public_unknown"),
        sensitive_fact(
            "f_public_unknown",
            "unknown to all",
            Sensitivity::Public,
            "c",
        ),
    );
    let out = ask(&mut sim, "f_public_unknown");
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    assert!(matches!(
        refused_of(&sim, "f_public_unknown"),
        EventKind::QueryAnswered { refused: false, unknown: true, polarities, .. }
            if polarities.is_empty()
    ));
}

/// Helper for the t4_9b assertions: the asker holds provenance-bound evidence
/// about `fact` (delivery marker).
fn player_knows_marker(sim: &Simulation, fact: &str) -> bool {
    !sim.world
        .epistemics
        .evidence_about(&ActorId::new(PLAYER), &FactId::new(fact))
        .is_empty()
}

/// Review D1-a: a refusal is never conflated with ignorance — the same
/// answerer, in one session, gives both answer types and the typed events
/// differ in both fields (and neither leaks content).
#[test]
fn t4_9c_query_refusal_is_distinct_from_ignorance() {
    let (mut world, player, b) = fixture();
    world.facts.insert(
        FactId::new("f_pub"),
        Fact {
            id: FactId::new("f_pub"),
            statement: "public but unobserved".into(),
            truth: true,
            sensitivity: Sensitivity::Public,
            category: FactCategory::Observation,
            prerequisites: vec![],
            eligible_holders: vec![],
            eligible_containers: vec![],
            committed: true,
            placed_at: None,
        },
    );
    world.facts.insert(
        FactId::new("f_priv"),
        sensitive_fact("f_priv", "a private matter", Sensitivity::Private, "c"),
    );
    let mut sim = Simulation::new(world, 5, sim_config());
    let (tick, minute) = (sim.world.tick, sim.world.minute_of_day);
    rpg_behavior::record_provenance_evidence(
        &mut sim.world,
        &mut sim.events,
        tick,
        minute,
        &b,
        &FactId::new("f_priv"),
        true,
        &Provenance::Assumed,
    );
    let out = apply(
        &mut sim,
        &player,
        Action::Query {
            target: b.clone(),
            about: FactId::new("f_pub"),
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    let out = apply(
        &mut sim,
        &player,
        Action::Query {
            target: b.clone(),
            about: FactId::new("f_priv"),
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    let answers: Vec<_> = sim
        .events
        .all()
        .iter()
        .filter(|e| matches!(&e.kind, EventKind::QueryAnswered { answerer, .. } if answerer == &b))
        .collect();
    assert_eq!(answers.len(), 2);
    assert!(matches!(
        &answers[0].kind,
        EventKind::QueryAnswered {
            unknown: true,
            refused: false,
            ..
        }
    ));
    assert!(matches!(
        &answers[1].kind,
        EventKind::QueryAnswered {
            unknown: false,
            refused: true,
            ..
        }
    ));
    // Neither answer wrote anything into the asker's mind.
    assert!(sim
        .world
        .epistemics
        .evidence_about(&player, &FactId::new("f_pub"))
        .is_empty());
    assert!(sim
        .world
        .epistemics
        .evidence_about(&player, &FactId::new("f_priv"))
        .is_empty());
    assert!(!sim
        .world
        .actor(&player)
        .unwrap()
        .beliefs
        .contains_key(&FactId::new("f_priv")));
}

// ---------------------------------------------------------------------------
// Review D1-b: typed deal breach, broken commitment and report-crime are
// directed contextual appraisals (kernel-equality, exactly once, directional)
// — no universal deltas.
// ---------------------------------------------------------------------------

/// A typed deal breach is the creditor's contextual PromiseBroken appraisal:
/// kernel-equality for the creditor's own value profile, exactly once,
/// citing the typed `DealViolated` event — never the legacy universal
/// trust/anger/respect delta. Breach is dispatched through the production
/// route (the debtor walks away from a due deal: FSM
/// `(PerformanceDue, Cancel{Debtor}) ⇒ Violated`), with no maintenance pass
/// between the appraisal and the assertion. A hot-headed creditor appraises
/// the SAME breach differently (context), and only the creditor's edge moves
/// (directionality).
#[test]
fn d1b_deal_breach_appraises_contextually_exactly_once_and_directional() {
    let (world, player, b) = fixture();
    let mut sim = Simulation::new(world, 5, sim_config());
    // A second, hot-headed creditor: the same breach must read differently.
    let mut hothead = bare_actor("hothead", 4.0, ActorKind::Npc);
    hothead.personality.reciprocity = 1.0;
    sim.world.actors.insert(hothead.id.clone(), hothead);
    let hothead = ActorId::new("hothead");

    let propose = |sim: &mut Simulation, creditor: &ActorId| {
        let out = apply(
            sim,
            &player,
            Action::ProposeDeal {
                target: creditor.clone(),
                terms: "timely delivery".into(),
                condition: QuestPredicate::Flag("ordered".into()),
                performance: QuestPredicate::HasItem {
                    actor: creditor.clone(),
                    item: ItemId::new("coin"),
                    qty: 1,
                },
                condition_deadline_minute: None,
                performance_deadline_minute: None,
                give: vec![],
                money: 0,
                note: "deliver the medicine".into(),
            },
        );
        assert!(matches!(out, ValidationOutcome::Pending { .. }), "{out:?}");
        *sim.world.deals.keys().max().unwrap()
    };
    let deal_b = propose(&mut sim, &b);
    let deal_h = propose(&mut sim, &hothead);
    sim.world.set_flag("ordered");
    for (creditor, deal) in [(&b, deal_b), (&hothead, deal_h)] {
        let out = apply(&mut sim, creditor, Action::AcceptDeal { deal });
        assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    }
    let b_before = sim.world.actor(&b).unwrap().relationship(&player).clone();
    let h_before = sim
        .world
        .actor(&hothead)
        .unwrap()
        .relationship(&player)
        .clone();

    // One deterministic step: the condition holds, so both deals reach
    // `PerformanceDue` through the deal pipeline (no relationship mutation:
    // the edges were neutral, and decay skips zero edges).
    sim.step(Vec::new());
    assert_eq!(
        sim.world.deals.get(&deal_b).unwrap().state,
        rpg_domain::deal::DealState::PerformanceDue
    );
    assert_eq!(
        sim.world.deals.get(&deal_h).unwrap().state,
        rpg_domain::deal::DealState::PerformanceDue
    );
    // The debtor walks away from BOTH due deals: the FSM maps the unilateral
    // debtor cancellation onto the Violated path (production dispatch).
    for deal in [deal_b, deal_h] {
        let out = apply(&mut sim, &player, Action::CancelDeal { deal });
        assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
        assert_eq!(
            sim.world.deals.get(&deal).unwrap().state,
            rpg_domain::deal::DealState::Violated
        );
    }

    // Kernel equality per creditor profile: production breach == exactly one
    // contextual PromiseBroken appraisal through the creditor's own values.
    for (creditor, before) in [(&b, &b_before), (&hothead, &h_before)] {
        let observer = sim.world.actor(creditor).unwrap();
        let profile = rpg_social::appraisal_profile(observer, &NeedsConfig::default());
        let edge_before = rpg_behavior::ContextualEdge::from_relationship(before);
        let delta = rpg_behavior::appraise(
            &rpg_behavior::AppraisedEvent::PromiseBroken,
            &profile,
            &edge_before,
        );
        let expected = rpg_behavior::apply_delta(&edge_before, &delta);
        let rel = observer.relationship(&player);
        assert_eq!(
            rpg_behavior::ContextualEdge::from_relationship(&rel),
            expected,
            "production breach == the contextual kernel for {creditor}'s profile"
        );
        // No legacy universal delta may ride along: the removed fixed write
        // moved the aggregate `trust`/`respect` dims.
        assert_eq!(
            rel.trust, before.trust,
            "legacy universal trust delta is gone ({creditor})"
        );
        assert_ne!(
            rel.trust_benevolence, before.trust_benevolence,
            "the appraisal actually moved the contextual edge ({creditor})"
        );
        // Directionality: the debtor's edge toward the creditor is untouched.
        assert_eq!(
            sim.world.actor(&player).unwrap().relationship(creditor),
            rpg_domain::world::Relationship::default(),
            "Rel(debtor,creditor) must not move: relationships are directed"
        );
    }
    // Context: the same event, different value profiles — different deltas.
    let b_rel = sim.world.actor(&b).unwrap().relationship(&player).clone();
    let h_rel = sim
        .world
        .actor(&hothead)
        .unwrap()
        .relationship(&player)
        .clone();
    assert_ne!(
        b_rel.anger, h_rel.anger,
        "the hot-headed creditor angers more: appraisals are context-dependent"
    );

    // Exactly one appraisal per breach, each citing its real causal event.
    let violations = sim
        .events
        .all()
        .iter()
        .filter(|e| matches!(e.kind, EventKind::DealViolated { .. }))
        .count();
    assert_eq!(violations, 2, "one typed violation per breached deal");
    let appraisals: Vec<_> = sim
        .events
        .all()
        .iter()
        .filter(|e| matches!(e.kind, EventKind::AppraisalRecorded { .. }))
        .collect();
    assert_eq!(appraisals.len(), 2, "exactly one appraisal per breach");
    let violation_ids: std::collections::BTreeSet<_> = sim
        .events
        .all()
        .iter()
        .filter(|e| matches!(e.kind, EventKind::DealViolated { .. }))
        .map(|e| e.id)
        .collect();
    for a in &appraisals {
        assert_eq!(
            a.caused_by.len(),
            1,
            "each appraisal cites exactly its causal event"
        );
        assert!(
            violation_ids.contains(&a.caused_by[0]),
            "the appraisal's origin is the DealViolated event"
        );
    }
}

/// A broken commitment (production BreakPromise route) is the creditor's
/// contextual PromiseBroken appraisal — kernel equality for the creditor's
/// value profile, exactly once, citing the typed `CommitmentBroken` event —
/// never the legacy universal trust/anger/respect delta. Directionality: only
/// the creditor's edge moves.
#[test]
fn d1b_break_promise_appraises_contextually_exactly_once_and_directional() {
    let (world, player, b) = fixture();
    let mut sim = Simulation::new(world, 5, sim_config());
    let out = apply(
        &mut sim,
        &player,
        Action::Promise {
            target: b.clone(),
            terms: "help".into(),
            predicate: QuestPredicate::Flag("never_done".into()),
            deadline: None,
        },
    );
    let commitment = match &out {
        ValidationOutcome::Pending {
            commitment: Some(c),
            ..
        } => *c,
        other => panic!("expected a pending commitment, got {other:?}"),
    };
    let b_before = sim.world.actor(&b).unwrap().relationship(&player).clone();
    let player_before = sim.world.actor(&player).unwrap().relationship(&b).clone();

    let out = apply(&mut sim, &player, Action::BreakPromise { commitment });
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");

    // Kernel equality for the creditor's own profile.
    let observer = sim.world.actor(&b).unwrap();
    let profile = rpg_social::appraisal_profile(observer, &NeedsConfig::default());
    let edge_before = rpg_behavior::ContextualEdge::from_relationship(&b_before);
    let delta = rpg_behavior::appraise(
        &rpg_behavior::AppraisedEvent::PromiseBroken,
        &profile,
        &edge_before,
    );
    let expected = rpg_behavior::apply_delta(&edge_before, &delta);
    let rel = observer.relationship(&player);
    assert_eq!(
        rpg_behavior::ContextualEdge::from_relationship(&rel),
        expected,
        "production break-promise == the contextual kernel for the creditor's profile"
    );
    assert_eq!(
        rel.trust, b_before.trust,
        "legacy universal trust delta is gone"
    );
    assert_eq!(rel.suspicion, b_before.suspicion);
    assert_ne!(
        rel.trust_benevolence, b_before.trust_benevolence,
        "the appraisal actually moved the contextual edge"
    );
    // Directionality: the breaker's edge toward the creditor is untouched.
    assert_eq!(
        sim.world.actor(&player).unwrap().relationship(&b),
        player_before,
        "Rel(debtor,creditor) must not move: relationships are directed"
    );
    // The promise-break emotion is written mind-locally (I-4b).
    assert!(
        observer.emotions.value(EmotionKind::Anger) > 0.0,
        "the broken promise registered as anger (mind-local)"
    );
    // Exactly one appraisal citing the typed CommitmentBroken event.
    let broken_event = sim
        .events
        .all()
        .iter()
        .find(|e| matches!(e.kind, EventKind::CommitmentBroken { .. }))
        .expect("the break is typed");
    let appraisals: Vec<_> = sim
        .events
        .all()
        .iter()
        .filter(|e| matches!(e.kind, EventKind::AppraisalRecorded { .. }))
        .collect();
    assert_eq!(appraisals.len(), 1, "exactly one appraisal per break");
    assert_eq!(appraisals[0].caused_by, vec![broken_event.id]);
    // The typed effects: tie change + appraise.
    let outcome = sim.outcomes.last().unwrap();
    assert!(outcome.effects.iter().any(|e| matches!(
        e,
        rpg_domain::behavior_types::Effect::ChangeTie { from, to, .. }
            if from == &b && to == &player
    )));
    assert!(outcome.effects.iter().any(|e| matches!(
        e,
        rpg_domain::behavior_types::Effect::Appraise { subject, .. } if subject == &b
    )));
}
