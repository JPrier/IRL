//! Phase 1 epistemic microcore acceptance tests (test-spec §2): evidence
//! ledger storage, correlation protection, contradiction tolerance, no
//! automatic acceptance, no omniscience, source-chain preservation, and the
//! liar-vs-skeptic deliverable — all through the shared mechanics, with no
//! actor-id special cases.

use rpg_behavior::{
    accept_assertion, aggregate_about, answer_query, assert_proposition, knows_of, prior_strength,
    record_evidence, skeptic_seeks_evidence, Acceptance, AssertionDraft, QueryAnswer,
    RecipientAppraisal,
};
use rpg_domain::behavior_types::{AcquisitionChannel, Epistemics, SourceChainId};
use rpg_domain::events::EventLog;
use rpg_domain::ids::*;
use rpg_domain::rng::DeterministicRng;
use rpg_domain::world::{
    Actor, ActorKind, ActorStatus, Belief, Capabilities, NavGrid, Personality, Provenance,
    StoryState, World,
};
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
    let mut actors_map = BTreeMap::new();
    for id in actors {
        let a = bare_actor(id);
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
        epistemics: Epistemics::default(),
        lore: Default::default(),
    }
}

/// Author a belief (the way authored content does — directly on the actor,
/// with no ledger entry).
fn author_belief(world: &mut World, actor: &str, fact: &str, believes_true: bool, conf: f32) {
    world
        .actor_mut(&ActorId::new(actor))
        .unwrap()
        .beliefs
        .insert(
            FactId::new(fact),
            Belief {
                fact: FactId::new(fact),
                believes_true,
                confidence: conf,
                source: Provenance::Assumed,
                acquired_tick: 0,
            },
        );
}

fn draft(
    speaker: Option<&str>,
    fact: &str,
    polarity: bool,
    channel: AcquisitionChannel,
) -> AssertionDraft {
    AssertionDraft {
        speaker: speaker.map(ActorId::new),
        proposition: FactId::new(fact),
        asserted_polarity: polarity,
        channel,
        origin_event: None,
        reliability: 0.6,
        retold_chain: None,
        deceptive: false,
    }
}

fn appraisal(trust: f32, suspicion: f32, fear: f32, assertiveness: f32) -> RecipientAppraisal {
    RecipientAppraisal {
        trust,
        suspicion,
        fear,
        assertiveness,
    }
}

// ---------------------------------------------------------------------------
// T1.1 — evidence ledger storage (no overwrite)
// ---------------------------------------------------------------------------

/// The old `record_belief` kept one `Belief` per fact; the ledger keeps every
/// evidence item from distinct source chains.
#[test]
fn t1_1_evidence_ledger_keeps_every_entry_from_distinct_chains() {
    let mut epistemics = Epistemics::default();
    let observer = ActorId::new("observer");
    let fact = FactId::new("f_x");
    for i in 0..5u64 {
        let d = draft(
            Some(&format!("source{i}")),
            "f_x",
            true,
            AcquisitionChannel::Told,
        );
        let (ev, chain) = record_evidence(&mut epistemics, &observer, &d, i);
        assert_eq!(ev.source_chain_id, chain);
        assert!(epistemics.chain(chain).is_some(), "chain registered");
    }
    let entries = epistemics.evidence_about(&observer, &fact);
    assert_eq!(entries.len(), 5, "no overwrite: all five entries kept");
    // Distinct originators → distinct chains.
    let chains: BTreeSet<u64> = entries.iter().map(|e| e.source_chain_id.raw()).collect();
    assert_eq!(chains.len(), 5);
}

// ---------------------------------------------------------------------------
// T1.2 — correlation protection
// ---------------------------------------------------------------------------

/// Five retellings from one origin do not raise certainty like five
/// independent sources; the `independent_corrob_group` grouping is verified.
#[test]
fn t1_2_five_retellings_of_one_origin_are_not_five_confirmations() {
    let fact = FactId::new("f_y");
    let observer = ActorId::new("obs");

    // Correlated: one originator ("liar") separately tells five messengers;
    // each messenger then tells the observer.
    let mut correlated = world_with(&["liar", "m1", "m2", "m3", "m4", "m5", "obs"]);
    for m in ["m1", "m2", "m3", "m4", "m5"] {
        // Liar -> messenger: a fresh chain (the liar has no ledger entry; the
        // chain's originator is the liar).
        let d = draft(Some("liar"), "f_y", true, AcquisitionChannel::Told);
        let (_, chain) = record_evidence(&mut correlated.epistemics, &ActorId::new(m), &d, 1);
        // Messenger -> observer: the retelling inherits the same chain.
        let d2 = draft(Some(m), "f_y", true, AcquisitionChannel::Told);
        let (ev, chain2) = record_evidence(&mut correlated.epistemics, &observer, &d2, 2);
        assert_eq!(ev.source_chain_id, chain2, "retelling preserves the chain");
        assert_eq!(chain2, chain, "messenger's chain is the retelling's chain");
    }
    let entries = correlated.epistemics.evidence_about(&observer, &fact);
    assert_eq!(entries.len(), 5);
    let groups: BTreeSet<u64> = entries
        .iter()
        .map(|e| e.independent_corrob_group.unwrap())
        .collect();
    assert_eq!(
        groups.len(),
        1,
        "all five retellings share one correlation group (one originator)"
    );
    let correlated_agg = aggregate_about(&correlated.epistemics, &observer, &fact);
    assert_eq!(correlated_agg.independence_groups, 1);

    // Independent: five distinct originators with no shared origin.
    let mut independent = Epistemics::default();
    for i in 0..5u64 {
        let d = draft(
            Some(&format!("indep{i}")),
            "f_y",
            true,
            AcquisitionChannel::Told,
        );
        record_evidence(&mut independent, &observer, &d, i);
    }
    let independent_agg = aggregate_about(&independent, &observer, &fact);
    assert_eq!(independent_agg.independence_groups, 5);
    let groups: BTreeSet<u64> = independent
        .evidence_about(&observer, &fact)
        .iter()
        .map(|e| e.independent_corrob_group.unwrap())
        .collect();
    assert_eq!(groups.len(), 5, "independent origins get distinct groups");
    assert!(
        independent_agg.confidence > correlated_agg.confidence + 0.1,
        "five independent sources (conf {:.3}) must beat one origin retold \
         five times (conf {:.3})",
        independent_agg.confidence,
        correlated_agg.confidence
    );
}

/// Repeated assertion of the *same* chain (A tells B five times) also stays
/// well short of certainty.
#[test]
fn t1_2b_repeated_same_chain_assertion_does_not_yield_certainty() {
    let mut epistemics = Epistemics::default();
    let (speaker, observer) = (ActorId::new("a"), ActorId::new("b"));
    let fact = FactId::new("f_z");
    let mut first_chain: Option<SourceChainId> = None;
    for tick in 0..5u64 {
        let _ = speaker;
        let d = draft(Some("a"), "f_z", true, AcquisitionChannel::Told);
        let (ev, c) = record_evidence(&mut epistemics, &observer, &d, tick);
        if first_chain.is_none() {
            first_chain = Some(c);
        }
        assert_eq!(
            ev.source_chain_id,
            first_chain.unwrap(),
            "same chain across repeats"
        );
    }
    let agg = aggregate_about(&epistemics, &observer, &fact);
    assert_eq!(agg.independence_groups, 1);
    assert!(
        agg.confidence < 0.75,
        "five retellings of one chain must not approach certainty, got {:.3}",
        agg.confidence
    );
    assert!(prior_strength(&agg) < 0.6);
}

// ---------------------------------------------------------------------------
// T1.3 — contradiction tolerance
// ---------------------------------------------------------------------------

#[test]
fn t1_3_contradictory_reports_are_held_side_by_side() {
    let mut epistemics = Epistemics::default();
    let (observer, fact) = (ActorId::new("b"), FactId::new("f_w"));
    let d_true = draft(Some("a"), "f_w", true, AcquisitionChannel::Told);
    let d_false = draft(Some("c"), "f_w", false, AcquisitionChannel::Told);
    record_evidence(&mut epistemics, &observer, &d_true, 1);
    record_evidence(&mut epistemics, &observer, &d_false, 2);
    let entries = epistemics.evidence_about(&observer, &fact);
    assert_eq!(entries.len(), 2);
    assert!(entries.iter().any(|e| e.asserted_polarity));
    assert!(entries.iter().any(|e| !e.asserted_polarity));
    let agg = aggregate_about(&epistemics, &observer, &fact);
    assert!(agg.contradicted, "both polarities above threshold");
    assert!(
        (agg.confidence - 0.5).abs() < 0.2,
        "contradiction leaves the aggregate near neutral, got {:.3}",
        agg.confidence
    );
}

// ---------------------------------------------------------------------------
// T1.4 — no automatic acceptance
// ---------------------------------------------------------------------------

#[test]
fn t1_4_assertiveness_never_forces_belief() {
    // Maximum assertiveness + maximum speaker confidence must still leave
    // room for disbelief (probability clamped below 1) for a suspicious
    // recipient.
    let mut believed_any = false;
    let mut disbelieved_any = false;
    for seed in 0..64u64 {
        let mut world = world_with(&["a", "b"]);
        let mut events = EventLog::new();
        let mut rng = DeterministicRng::new(seed);
        let acc: Acceptance = accept_assertion(
            &mut world,
            &mut events,
            1,
            480,
            &ActorId::new("b"),
            draft(Some("a"), "f_k", true, AcquisitionChannel::Told),
            appraisal(0.0, 0.9, 0.0, 1.0),
            1.0,
            true,
            &mut rng,
        );
        assert!(
            acc.probability < 0.98,
            "an assertive claim must never approach certainty (p={})",
            acc.probability
        );
        // Delivery is guaranteed even when belief fails.
        assert_eq!(
            world
                .epistemics
                .evidence_about(&ActorId::new("b"), &FactId::new("f_k"))
                .len(),
            1,
            "the evidence item is recorded regardless of the roll (seed {seed})"
        );
        let typed_events = events
            .all()
            .iter()
            .filter(|e| {
                matches!(
                    e.kind,
                    rpg_domain::events::EventKind::EvidenceRecorded { .. }
                )
            })
            .count();
        assert_eq!(
            typed_events, 1,
            "one typed EvidenceRecorded event per delivery"
        );
        if acc.believed {
            believed_any = true;
        } else {
            disbelieved_any = true;
        }
    }
    assert!(
        believed_any && disbelieved_any,
        "acceptance stays probabilistic"
    );
}

// ---------------------------------------------------------------------------
// T1.5 — no omniscience / partial observability
// ---------------------------------------------------------------------------

#[test]
fn t1_5_query_answers_come_only_from_accessible_evidence() {
    let mut world = world_with(&["knower", "author", "blind"]);
    author_belief(&mut world, "author", "f_s", true, 0.8);

    // No evidence, no authored knowledge → cannot answer.
    assert_eq!(
        answer_query(&world, &ActorId::new("blind"), &FactId::new("f_s")),
        QueryAnswer::Unknown
    );
    // Authored belief is reportable at its authored confidence.
    match answer_query(&world, &ActorId::new("author"), &FactId::new("f_s")) {
        QueryAnswer::Heard {
            polarities,
            confidence,
            contradicted,
        } => {
            assert_eq!(polarities, vec![true]);
            assert!((confidence - 0.8).abs() < 1e-5);
            assert!(!contradicted);
        }
        other => panic!("expected Heard, got {other:?}"),
    }
    // Ledger evidence is reported with its polarities.
    let d = draft(Some("knower"), "f_s", false, AcquisitionChannel::Told);
    record_evidence(&mut world.epistemics, &ActorId::new("knower"), &d, 1);
    match answer_query(&world, &ActorId::new("knower"), &FactId::new("f_s")) {
        QueryAnswer::Heard { polarities, .. } => assert_eq!(polarities, vec![false]),
        other => panic!("expected Heard, got {other:?}"),
    }
}

/// Information noninterference smoke: changing an unreachable secret changes
/// nothing in an actor's observable epistemic behaviour under a fixed RNG.
#[test]
fn t1_5b_unreachable_secret_changes_nothing() {
    let build = |secret_truth: bool| -> World {
        let mut world = world_with(&["a", "b", "secret_keeper"]);
        // The secret: only the (absent-from-interaction) keeper knows it.
        author_belief(&mut world, "secret_keeper", "f_secret", secret_truth, 1.0);
        author_belief(&mut world, "a", "f_known", true, 0.8);
        world
    };
    let run = |world: &mut World| -> (bool, u64) {
        let mut events = EventLog::new();
        let mut rng = DeterministicRng::new(99);
        let acc = accept_assertion(
            world,
            &mut events,
            1,
            480,
            &ActorId::new("b"),
            draft(Some("a"), "f_known", true, AcquisitionChannel::Told),
            appraisal(0.0, 0.0, 0.0, 0.6),
            0.8,
            true,
            &mut rng,
        );
        (acc.believed, events.checksum())
    };
    let mut w_true = build(true);
    let mut w_false = build(false);
    let (believed_t, checksum_t) = run(&mut w_true);
    let (believed_f, checksum_f) = run(&mut w_false);
    assert_eq!(believed_t, believed_f);
    assert_eq!(
        checksum_t, checksum_f,
        "unreachable secret: identical observables"
    );
}

// ---------------------------------------------------------------------------
// T1.6 — liar vs skeptic (no special-case scripts)
// ---------------------------------------------------------------------------

#[test]
fn t1_6_liar_fools_a_trusting_npc_while_a_skeptic_requests_evidence() {
    let seeds = 64;
    let mut gullible_believed = 0;
    let mut skeptic_believed = 0;
    let mut skeptic_sought = 0;
    for seed in 0..seeds {
        // Two recipients of the SAME claim from the SAME speaker; the only
        // difference is the relationship the recipients hold.
        let mut w_gullible = world_with(&["liar", "gullible"]);
        let mut w_skeptic = world_with(&["liar", "skeptic"]);
        let mut events = EventLog::new();
        let mut rng = DeterministicRng::new(seed);
        let acc_g = accept_assertion(
            &mut w_gullible,
            &mut events,
            1,
            480,
            &ActorId::new("gullible"),
            draft(Some("liar"), "f_lie", true, AcquisitionChannel::Told),
            appraisal(0.9, 0.0, 0.0, 0.8),
            0.9,
            true,
            &mut rng,
        );
        let mut rng = DeterministicRng::new(seed);
        let acc_s = accept_assertion(
            &mut w_skeptic,
            &mut events,
            1,
            480,
            &ActorId::new("skeptic"),
            draft(Some("liar"), "f_lie", true, AcquisitionChannel::Told),
            appraisal(-0.9, 0.9, 0.0, 0.8),
            0.9,
            true,
            &mut rng,
        );
        if acc_g.believed {
            gullible_believed += 1;
        }
        if acc_s.believed {
            skeptic_believed += 1;
        }
        // The skeptic's response policy: an unbelieved weak claim triggers a
        // request for more evidence. Pure function of state, never of ids.
        if skeptic_seeks_evidence(&acc_s.aggregate, acc_s.believed) {
            skeptic_sought += 1;
        }
    }
    assert!(
        gullible_believed as f32 / seeds as f32 > 0.85,
        "a trusting NPC should almost always be fooled ({gullible_believed}/{seeds})"
    );
    assert!(
        (skeptic_believed as f32 / seeds as f32) < 0.35,
        "a distrusted, suspected speaker should rarely convince the skeptic \
         ({skeptic_believed}/{seeds})"
    );
    assert_eq!(
        skeptic_sought,
        seeds - skeptic_believed,
        "every unbelieved weak claim makes the skeptic seek more evidence"
    );
}

/// The deceptive path: `assert_proposition` lets an actor who *knows of* a
/// proposition assert a polarity contrary to their own belief (a lie is a
/// proposition choice, not a truth mutation) — and the recipient still only
/// believes per their own appraisal.
#[test]
fn t1_6b_a_lie_is_a_proposition_choice_not_a_truth_mutation() {
    let mut world = world_with(&["liar", "dupe"]);
    // The liar knows OF the fact but believes it FALSE (heard and doubted).
    author_belief(&mut world, "liar", "f_false_thing", false, 0.7);
    let mut events = EventLog::new();
    let mut rng = DeterministicRng::new(11);
    // A trusting dupe is told the false polarity asserted as true.
    world
        .actor_mut(&ActorId::new("dupe"))
        .unwrap()
        .relationship_mut(&ActorId::new("liar"))
        .trust = 0.9;
    let acc = assert_proposition(
        &mut world,
        &mut events,
        &mut rng,
        1,
        480,
        &ActorId::new("liar"),
        &ActorId::new("dupe"),
        &FactId::new("f_false_thing"),
        true,
        true,
    )
    .expect("the liar knows of the proposition");
    assert!(acc.believed, "the trusting dupe is fooled by the lie");
    // The lie did not mutate world truth: no `Fact` was created or altered.
    assert!(world.facts.is_empty());
    // The dupe holds evidence *as asserted* (polarity true, from the liar).
    let entries = world
        .epistemics
        .evidence_about(&ActorId::new("dupe"), &FactId::new("f_false_thing"));
    assert_eq!(entries.len(), 1);
    assert!(entries[0].asserted_polarity);
    assert_eq!(
        entries[0].source_actor_id.as_ref().map(|a| a.as_str()),
        Some("liar")
    );
    // A speaker who does NOT know of the proposition cannot assert it.
    let mut rng = DeterministicRng::new(12);
    assert!(assert_proposition(
        &mut world,
        &mut events,
        &mut rng,
        2,
        480,
        &ActorId::new("dupe"),
        &ActorId::new("liar"),
        &FactId::new("f_never_heard"),
        true,
        false,
    )
    .is_none());
}

// ---------------------------------------------------------------------------
// Source-chain preservation through the real acceptance path
// ---------------------------------------------------------------------------

#[test]
fn source_chain_preserved_across_a_transmission_path() {
    let mut world = world_with(&["origin", "b", "c"]);
    author_belief(&mut world, "origin", "f_chain", true, 0.9);
    let mut events = EventLog::new();
    let mut rng = DeterministicRng::new(3);

    // origin -> b
    let acc_b = accept_assertion(
        &mut world,
        &mut events,
        1,
        480,
        &ActorId::new("b"),
        draft(Some("origin"), "f_chain", true, AcquisitionChannel::Told),
        appraisal(0.0, 0.0, 0.0, 0.6),
        0.9,
        true,
        &mut rng,
    );
    let chain_b = acc_b.chain;
    let chain_rec = world.epistemics.chain(chain_b).cloned().unwrap();
    assert_eq!(
        chain_rec.originator.as_ref().map(|a| a.as_str()),
        Some("origin")
    );
    assert_eq!(
        chain_rec.path,
        vec![ActorId::new("origin"), ActorId::new("b")]
    );

    // b -> c: c inherits b's chain (the claim's origin chain is preserved).
    let acc_c = accept_assertion(
        &mut world,
        &mut events,
        2,
        480,
        &ActorId::new("c"),
        draft(Some("b"), "f_chain", true, AcquisitionChannel::Told),
        appraisal(0.0, 0.0, 0.0, 0.6),
        0.5,
        true,
        &mut rng,
    );
    assert_eq!(acc_c.chain, chain_b, "retelling shares the origin chain");
    let chain_rec = world.epistemics.chain(chain_b).cloned().unwrap();
    assert_eq!(
        chain_rec.path,
        vec![ActorId::new("origin"), ActorId::new("b"), ActorId::new("c")]
    );
    // Both entries share the correlation group.
    assert_eq!(
        acc_b.evidence.independent_corrob_group,
        acc_c.evidence.independent_corrob_group
    );
}

// ---------------------------------------------------------------------------
// knows_of: knowledge vs belief vs permission separation
// ---------------------------------------------------------------------------

#[test]
fn knows_of_distinguishes_heard_disbelieved_from_unknown() {
    let mut world = world_with(&["a", "b", "c"]);
    // b heard but disbelieved; c never heard.
    author_belief(&mut world, "b", "f_q", false, 0.4);
    assert!(knows_of(&world, &ActorId::new("b"), &FactId::new("f_q")));
    assert!(!knows_of(&world, &ActorId::new("c"), &FactId::new("f_q")));
    assert!(!knows_of(&world, &ActorId::new("a"), &FactId::new("f_q")));
}

// ---------------------------------------------------------------------------
// Legacy parity: an empty ledger reproduces the legacy probability exactly
// ---------------------------------------------------------------------------

#[test]
fn empty_ledger_reproduces_the_legacy_probability_formula() {
    let mut world = world_with(&["a", "b"]);
    let mut events = EventLog::new();
    let mut rng = DeterministicRng::new(0);
    let acc = accept_assertion(
        &mut world,
        &mut events,
        1,
        480,
        &ActorId::new("b"),
        draft(Some("a"), "f_p", true, AcquisitionChannel::Told),
        appraisal(0.0, 0.0, 0.0, 0.6),
        0.7,
        true,
        &mut rng,
    );
    // Legacy: p = 0.25 + 0.45*0.5 + 0.25*0.6 + 0.2*0.7 − 0.35*0.5 − 0.15*0
    //        = 0.59 (neutral-relationship suspicion midpoint included).
    assert!((acc.probability - 0.59).abs() < 1e-6);
    assert_eq!(acc.prior_strength, 0.0);
}

// ---------------------------------------------------------------------------
// F5 — explicit retold-chain selection across multiple chains per speaker
// ---------------------------------------------------------------------------

/// A speaker holding evidence on two chains for one proposition retells one
/// of them explicitly: the observer must join *that* chain, not the
/// speaker's latest one (the default inference).
#[test]
fn f5_retold_chain_selection_beats_latest_chain_inference() {
    let mut epistemics = Epistemics::default();
    let (speaker, observer, other) = (ActorId::new("a"), ActorId::new("d"), ActorId::new("e"));
    // The speaker holds chain1 (heard from b) and chain2 (heard from c) —
    // two distinct origins for the same proposition.
    let (_, chain1) = record_evidence(
        &mut epistemics,
        &speaker,
        &draft(Some("b"), "f_multi", true, AcquisitionChannel::Told),
        1,
    );
    let (_, chain2) = record_evidence(
        &mut epistemics,
        &speaker,
        &draft(Some("c"), "f_multi", true, AcquisitionChannel::Told),
        2,
    );
    assert_ne!(chain1, chain2, "distinct origins stay distinct chains");
    assert!(
        chain2.raw() > chain1.raw(),
        "chain2 is the speaker's latest"
    );

    // The speaker retells chain1 explicitly.
    let mut d = draft(Some("a"), "f_multi", true, AcquisitionChannel::Told);
    d.retold_chain = Some(chain1);
    let (ev, chain) = record_evidence(&mut epistemics, &observer, &d, 3);
    assert_eq!(
        chain, chain1,
        "the explicit selection wins over the latest-chain fallback"
    );
    assert_eq!(ev.source_chain_id, chain1);
    let rec = epistemics.chain(chain1).unwrap();
    assert!(
        rec.path.contains(&observer),
        "the retelling is recorded on the chain's path"
    );
    assert!(
        !epistemics.chain(chain2).unwrap().path.contains(&observer),
        "the unretold chain is untouched"
    );

    // Without an explicit selection the legacy default still applies for a
    // fresh observer: the speaker's latest chain (backward compatible).
    let d = draft(Some("a"), "f_multi", true, AcquisitionChannel::Told);
    let (_, chain_default) = record_evidence(&mut epistemics, &other, &d, 4);
    assert_eq!(chain_default, chain2);
}

/// An invalid retold-chain selection is ignored, never trusted: the legacy
/// inference applies and the chain registry stays consistent.
#[test]
fn f5_invalid_retold_chain_selection_falls_back_safely() {
    let mut epistemics = Epistemics::default();
    let (speaker, observer) = (ActorId::new("a"), ActorId::new("b"));
    // The speaker holds one chain for the proposition.
    let (_, held) = record_evidence(
        &mut epistemics,
        &speaker,
        &draft(Some("origin"), "f_own", true, AcquisitionChannel::Told),
        1,
    );
    // 1. An unregistered chain id.
    let mut d = draft(Some("a"), "f_own", true, AcquisitionChannel::Told);
    d.retold_chain = Some(SourceChainId(9999));
    let (_, chain) = record_evidence(&mut epistemics, &observer, &d, 2);
    assert_eq!(
        chain, held,
        "the bogus pointer is ignored; the speaker's chain is inherited"
    );
    // 2. A registered chain the speaker does not own for this proposition:
    //    a third party's unrelated chain.
    let (_, foreign) = record_evidence(
        &mut epistemics,
        &ActorId::new("someone_else"),
        &draft(Some("x"), "f_other", true, AcquisitionChannel::Told),
        3,
    );
    let mut d = draft(Some("a"), "f_own", true, AcquisitionChannel::Told);
    d.retold_chain = Some(foreign);
    let (_, chain) = record_evidence(&mut epistemics, &observer, &d, 4);
    assert_ne!(chain, foreign, "a foreign chain is never joined");
    assert!(
        epistemics.chain(chain).is_some(),
        "the fallback still registers a real chain"
    );
}

/// A chain's originator may retell their own claim even when they hold no
/// ledger entry for it (authored knowledge).
#[test]
fn f5_chain_originator_may_retell_their_own_chain() {
    let mut epistemics = Epistemics::default();
    let observer = ActorId::new("b");
    let (_, chain) = record_evidence(
        &mut epistemics,
        &ActorId::new("first_listener"),
        &draft(Some("origin"), "f_o", true, AcquisitionChannel::Told),
        1,
    );
    let mut d = draft(Some("origin"), "f_o", true, AcquisitionChannel::Told);
    d.retold_chain = Some(chain);
    let (_, retold) = record_evidence(&mut epistemics, &observer, &d, 2);
    assert_eq!(retold, chain, "the originator's own chain continues");
}

// ---------------------------------------------------------------------------
// F6 — told-path evidence links the emitted ClaimShared delivery event
// ---------------------------------------------------------------------------

#[test]
fn f6_told_path_evidence_links_the_claim_shared_event() {
    let mut world = world_with(&["a", "b"]);
    let mut events = EventLog::new();
    let mut rng = DeterministicRng::new(7);
    let acc = accept_assertion(
        &mut world,
        &mut events,
        1,
        480,
        &ActorId::new("b"),
        draft(Some("a"), "f_link", true, AcquisitionChannel::Told),
        appraisal(0.0, 0.0, 0.0, 0.6),
        0.8,
        true,
        &mut rng,
    );
    let claim = acc
        .claim_event
        .expect("a told-path delivery emits a ClaimShared event");
    let claim_ev = events.get(claim).unwrap();
    match &claim_ev.kind {
        rpg_domain::events::EventKind::ClaimShared {
            from,
            to,
            fact,
            assertive,
            believed,
            ..
        } => {
            assert_eq!(from.as_str(), "a");
            assert_eq!(to.as_str(), "b");
            assert_eq!(fact.as_str(), "f_link");
            assert!(*assertive);
            assert_eq!(
                *believed, acc.believed,
                "the delivery carries the rolled outcome"
            );
        }
        other => panic!("expected ClaimShared, got {other:?}"),
    }
    // The evidence write is causally linked to the delivery that caused it.
    let recorded = events
        .all()
        .iter()
        .find(|e| {
            matches!(
                e.kind,
                rpg_domain::events::EventKind::EvidenceRecorded { .. }
            )
        })
        .unwrap();
    assert!(
        recorded.caused_by.contains(&claim),
        "EvidenceRecorded.caused_by must reference the ClaimShared event"
    );
    // Delivery is ordered before the evidence it causes.
    assert!(claim.raw() < recorded.id.raw());
}

#[test]
fn f6_non_told_channels_emit_no_claim_shared_event() {
    let mut world = world_with(&["a", "b"]);
    let mut events = EventLog::new();
    let mut rng = DeterministicRng::new(9);
    let acc = accept_assertion(
        &mut world,
        &mut events,
        1,
        480,
        &ActorId::new("b"),
        draft(Some("a"), "f_seen", true, AcquisitionChannel::Witnessed),
        appraisal(0.0, 0.0, 0.0, 0.6),
        0.8,
        true,
        &mut rng,
    );
    assert!(
        acc.claim_event.is_none(),
        "a witnessed delivery is not a shared claim"
    );
    assert!(
        events
            .all()
            .iter()
            .all(|e| !matches!(e.kind, rpg_domain::events::EventKind::ClaimShared { .. })),
        "no ClaimShared event for a non-told channel"
    );
    assert!(
        events.all().iter().any(|e| matches!(
            e.kind,
            rpg_domain::events::EventKind::EvidenceRecorded { .. }
        )),
        "the evidence item is still recorded"
    );
}
