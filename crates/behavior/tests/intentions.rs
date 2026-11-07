//! Behavior-engine Phase 2 kernel tests: persistent intentions (J) with the
//! fast path and switch penalty (T2.1), and the utility ranking kernel —
//! feasibility filter and lexicographic constraints (T2.8) plus seeded
//! softmax sampling with recorded samples (T2.9b-adjacent seam).
//!
//! The α_i·Δneeds term of design §4.2 is deliberately absent here: it belongs
//! to I-3b's completion after I-4a introduces needs state (plan §6). This
//! file pins the seam that term will extend.

use rpg_behavior::{
    consider_replan, rank, sample, Feasibility, Intention, Params, Replan, UtilityOption,
};
use rpg_domain::action::QuestPredicate;
use rpg_domain::behavior_types::AbandonmentCriteria;
use rpg_domain::rng::DeterministicRng;
use rpg_domain::world::{evaluate_predicate, World};
use std::collections::{BTreeMap, BTreeSet};

fn tiny_world() -> World {
    World {
        tick: 0,
        day: 0,
        minute_of_day: 480,
        locations: BTreeMap::new(),
        actors: BTreeMap::new(),
        items: BTreeMap::new(),
        factions: BTreeMap::new(),
        facts: BTreeMap::new(),
        nav: rpg_domain::world::NavGrid::new(2, 2),
        story: rpg_domain::world::StoryState {
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

fn intention(option: &str, satisfaction: Option<QuestPredicate>) -> Intention {
    Intention::adopt(
        "respond to the due obligation",
        option,
        satisfaction,
        AbandonmentCriteria::default(),
        0,
        None,
    )
}

// ---------------------------------------------------------------------------
// T2.1 — intention persistence and the fast path
// ---------------------------------------------------------------------------

/// An adopted plan survives ticks while its satisfaction predicate has not
/// begun to hold; the fast path continues without re-deliberation.
#[test]
fn t2_1_adopted_intention_survives_ticks_until_invalidated() {
    let mut world = tiny_world();
    let intent = intention(
        "honour_disclosure",
        Some(QuestPredicate::Flag("done".into())),
    );
    for tick in 1..=5 {
        world.tick = tick;
        assert_eq!(
            consider_replan(&intent, &world),
            Replan::Continue,
            "tick {tick}: the fast path keeps going"
        );
    }
    // The predicate begins to hold: the intention is invalidated.
    world.set_flag("done");
    assert_eq!(
        consider_replan(&intent, &world),
        Replan::Replan(rpg_behavior::ReplanReason::GoalSatisfied)
    );
}

/// Abandonment criteria: enough recorded failures end the intention
/// explicitly (declared-failure state, design §9.2.1).
#[test]
fn t2_1_abandonment_criteria_declare_failure() {
    let mut intent = Intention::adopt(
        "settle the delivery",
        "settle_delivery",
        None,
        AbandonmentCriteria {
            after_failed_attempts: Some(3),
            predicate: None,
        },
        0,
        None,
    );
    assert_eq!(consider_replan(&intent, &tiny_world()), Replan::Continue);
    intent.record_failure();
    intent.record_failure();
    assert_eq!(consider_replan(&intent, &tiny_world()), Replan::Continue);
    intent.record_failure();
    assert_eq!(
        consider_replan(&intent, &tiny_world()),
        Replan::Replan(rpg_behavior::ReplanReason::Abandoned),
        "the third failure abandons the intention"
    );
}

/// Switch bookkeeping: adopting a different option counts a switch; re-adopting
/// the same option does not.
#[test]
fn switch_count_tracks_option_changes() {
    let first = intention("honour_disclosure", None);
    assert_eq!(first.switches, 0);
    let same = Intention::adopt(
        "g",
        "honour_disclosure",
        None,
        Default::default(),
        1,
        Some(&first),
    );
    assert_eq!(
        same.switches, 0,
        "continuing the same option is not a switch"
    );
    let switched = Intention::adopt("g", "stall", None, Default::default(), 2, Some(&first));
    assert_eq!(switched.switches, 1, "changing option counts one switch");
}

// ---------------------------------------------------------------------------
// T2.8 — feasibility filter and lexicographic constraints
// ---------------------------------------------------------------------------

#[test]
fn t2_8_physically_impossible_options_are_filtered_before_scoring() {
    let options = vec![
        UtilityOption {
            id: "impossible_but_desirable".into(),
            goal_delta: 1000.0,
            cost: 0.0,
            feasibility: Feasibility::PhysicallyImpossible,
            needs_gain: Vec::new(),
        },
        UtilityOption {
            id: "mediocre_but_possible".into(),
            goal_delta: 0.1,
            cost: 0.0,
            feasibility: Feasibility::Possible,
            needs_gain: Vec::new(),
        },
    ];
    let r = rank(&options, None, Params::default());
    assert!(r.ranked.iter().all(|s| s.id != "impossible_but_desirable"));
    assert_eq!(r.impossible, vec!["impossible_but_desirable".to_string()]);
    assert_eq!(
        r.ranked.first().map(|s| s.id.as_str()),
        Some("mediocre_but_possible")
    );
    // Never sampled either: the sampler sees only the ranking.
    let mut rng = DeterministicRng::new(7);
    let picked = sample(&r, Params::default(), &mut rng).unwrap();
    assert_eq!(picked.id, "mediocre_but_possible");
}

#[test]
fn t2_8_lexicographic_constraints_cannot_be_traded_for_weights() {
    let options = vec![
        UtilityOption {
            id: "violates_principle".into(),
            goal_delta: 1000.0,
            cost: 0.0,
            feasibility: Feasibility::PersonallyUnacceptable,
            needs_gain: Vec::new(),
        },
        UtilityOption {
            id: "acceptable".into(),
            goal_delta: 0.1,
            cost: 0.0,
            feasibility: Feasibility::Possible,
            needs_gain: Vec::new(),
        },
    ];
    let r = rank(&options, None, Params::default());
    assert_eq!(r.ranked.first().map(|s| s.id.as_str()), Some("acceptable"));
    assert_eq!(r.unacceptable, vec!["violates_principle".to_string()]);

    // When *nothing* acceptable remains, the actor declines rather than
    // pick the least-bad violation.
    let only_bad = vec![UtilityOption {
        id: "violates_principle".into(),
        goal_delta: 1000.0,
        cost: 0.0,
        feasibility: Feasibility::PersonallyUnacceptable,
        needs_gain: Vec::new(),
    }];
    let r = rank(&only_bad, None, Params::default());
    assert!(r.ranked.is_empty(), "decline rather than violate");
    assert_eq!(r.unacceptable.len(), 1);
}

/// Physically impossible and personally unacceptable are distinct classes
/// (T2.8), reported separately by the ranking.
#[test]
fn t2_8_impossible_vs_unacceptable_are_distinguished() {
    let options = vec![
        UtilityOption {
            id: "cannot".into(),
            goal_delta: 5.0,
            cost: 0.0,
            feasibility: Feasibility::PhysicallyImpossible,
            needs_gain: Vec::new(),
        },
        UtilityOption {
            id: "will_not".into(),
            goal_delta: 5.0,
            cost: 0.0,
            feasibility: Feasibility::PersonallyUnacceptable,
            needs_gain: Vec::new(),
        },
        UtilityOption {
            id: "can_and_will".into(),
            goal_delta: 1.0,
            cost: 0.0,
            feasibility: Feasibility::Possible,
            needs_gain: Vec::new(),
        },
    ];
    let r = rank(&options, None, Params::default());
    assert_eq!(r.impossible, vec!["cannot".to_string()]);
    assert_eq!(r.unacceptable, vec!["will_not".to_string()]);
    assert_eq!(r.ranked.len(), 1);
}

/// The switch penalty: an option that differs from the standing intention pays
/// η_i·switch_i, so an equally-good continuation stays chosen (no oscillation).
#[test]
fn switch_penalty_deters_oscillation() {
    let options = vec![
        UtilityOption {
            id: "keep_doing".into(),
            goal_delta: 1.0,
            cost: 0.0,
            feasibility: Feasibility::Possible,
            needs_gain: Vec::new(),
        },
        UtilityOption {
            id: "switch_away".into(),
            goal_delta: 1.0,
            cost: 0.0,
            feasibility: Feasibility::Possible,
            needs_gain: Vec::new(),
        },
    ];
    // Ties break by id deterministically without an intention.
    let r = rank(&options, None, Params::default());
    assert_eq!(r.ranked[0].id, "keep_doing");
    // With "keep_doing" adopted, "switch_away" pays the penalty and stays
    // second; the ranking is stable across repeated calls.
    for _ in 0..3 {
        let r = rank(&options, Some("keep_doing"), Params::default());
        assert_eq!(r.ranked[0].id, "keep_doing");
        assert!(!r.ranked[0].switched);
        assert_eq!(r.ranked[1].id, "switch_away");
        assert!(r.ranked[1].switched);
    }
    // A large enough gain still justifies switching — the penalty is a
    // friction term, not a wall.
    let options = vec![
        UtilityOption {
            id: "keep_doing".into(),
            goal_delta: 1.0,
            cost: 0.0,
            feasibility: Feasibility::Possible,
            needs_gain: Vec::new(),
        },
        UtilityOption {
            id: "switch_away".into(),
            goal_delta: 5.0,
            cost: 0.0,
            feasibility: Feasibility::Possible,
            needs_gain: Vec::new(),
        },
    ];
    let r = rank(&options, Some("keep_doing"), Params::default());
    assert_eq!(r.ranked[0].id, "switch_away");
}

// ---------------------------------------------------------------------------
// T2.9b seam — seeded softmax sampling with recorded samples
// ---------------------------------------------------------------------------

#[test]
fn seeded_sampling_is_replayable_and_never_picks_invalid_options() {
    let options = vec![
        UtilityOption {
            id: "a".into(),
            goal_delta: 3.0,
            cost: 0.0,
            feasibility: Feasibility::Possible,
            needs_gain: Vec::new(),
        },
        UtilityOption {
            id: "impossible".into(),
            goal_delta: 100.0,
            cost: 0.0,
            feasibility: Feasibility::PhysicallyImpossible,
            needs_gain: Vec::new(),
        },
        UtilityOption {
            id: "b".into(),
            goal_delta: 1.0,
            cost: 0.0,
            feasibility: Feasibility::Possible,
            needs_gain: Vec::new(),
        },
    ];
    let r = rank(&options, None, Params::default());
    // The same seed reproduces the same choice and sample.
    let mut rng_a = DeterministicRng::new(42);
    let mut rng_b = DeterministicRng::new(42);
    let a = sample(&r, Params::default(), &mut rng_a).unwrap();
    let b = sample(&r, Params::default(), &mut rng_b).unwrap();
    assert_eq!(a, b, "same seed, same sample, same choice");
    assert!(a.id != "impossible", "invalid options are never sampled");
    assert!((0.0..=1.0).contains(&a.sample), "the sample is recorded");
    assert_eq!(r.ranked[a.index].id, a.id);
}

#[test]
fn empty_ranking_declines_instead_of_sampling() {
    let r = rank(&[], None, Params::default());
    assert!(r.ranked.is_empty());
    let mut rng = DeterministicRng::new(1);
    assert!(sample(&r, Params::default(), &mut rng).is_none());
}

/// The kernel reads world state only through the typed predicate evaluator —
/// a spot check that the satisfaction semantics are world-canonical, not
/// belief-local (that distinction lives in the caller's expectations).
#[test]
fn satisfaction_predicates_are_evaluated_against_the_world() {
    let mut world = tiny_world();
    assert!(!evaluate_predicate(
        &world,
        &QuestPredicate::Flag("x".into())
    ));
    world.set_flag("x");
    assert!(evaluate_predicate(
        &world,
        &QuestPredicate::Flag("x".into())
    ));
}
