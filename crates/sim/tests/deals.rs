//! Behavior-engine Phase 2 acceptance tests (plan §4 contracts; test-spec
//! T2.2–T2.5e plus the Phase-2 production integration): the deal FSM at the
//! authoritative-simulation level — pinned lifecycle, ConditionMet ≠ benefit
//! delivered, false-claim neutrality, escrow/atomicity, partial obligations,
//! the cancellation authorization matrix, idempotent deadlines with the grace
//! policy, exactly-once settlement under replay, conservation, rejection
//! neutrality, and the NPC deal-response pipeline.

use rpg_domain::action::{
    Action, ActionProposal, ProposalSource, QuestPredicate, RejectReason, ValidationOutcome,
};
use rpg_domain::deal::DealState;
use rpg_domain::events::EventKind;
use rpg_domain::geom::{Rect, Vec2};
use rpg_domain::ids::DealId;
use rpg_domain::ids::*;
use rpg_domain::world::{
    Actor, ActorKind, ActorStatus, Belief, Capabilities, Fact, FactCategory, Item, Location,
    LocationKind, NavGrid, Personality, Provenance, Sensitivity, StoryState, World,
};
use rpg_sim::{SimConfig, Simulation};
use std::collections::{BTreeMap, BTreeSet};

const DEBTOR: &str = "debtor";
const CREDITOR: &str = "creditor";
const BYSTANDER: &str = "bystander";

fn actor(id: &str, x: f32) -> Actor {
    Actor {
        id: ActorId::new(id),
        name: id.to_string(),
        kind: ActorKind::Npc,
        status: ActorStatus::Alive,
        location: LocationId::new("shop"),
        pos: Vec2::new(x, 1.0),
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

/// Two traders plus a bystander in one open room: `debtor` believes `f_ledger`
/// and the fixture carries `medicine` and `coin` items.
fn fixture() -> World {
    let nav = NavGrid::new(12, 12);
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
    let mut items = BTreeMap::new();
    for (id, value) in [("medicine", 80), ("coin", 10)] {
        items.insert(
            ItemId::new(id),
            Item {
                id: ItemId::new(id),
                name: id.to_string(),
                description: String::new(),
                base_value: value,
                tags: vec![],
                stackable: true,
            },
        );
    }
    let mut facts = BTreeMap::new();
    facts.insert(
        FactId::new("f_ledger"),
        Fact {
            id: FactId::new("f_ledger"),
            statement: "the ledger records unusual payments".into(),
            truth: true,
            sensitivity: Sensitivity::Restricted,
            category: FactCategory::Ledger,
            prerequisites: vec![],
            eligible_holders: vec![],
            eligible_containers: vec![],
            committed: false,
            placed_at: None,
        },
    );
    let mut debtor = actor(DEBTOR, 1.0);
    debtor.beliefs.insert(
        FactId::new("f_ledger"),
        Belief {
            fact: FactId::new("f_ledger"),
            believes_true: true,
            confidence: 0.9,
            source: Provenance::Assumed,
            acquired_tick: 0,
        },
    );
    let mut actors = BTreeMap::new();
    actors.insert(debtor.id.clone(), debtor);
    actors.insert(actor(CREDITOR, 2.0).id.clone(), actor(CREDITOR, 2.0));
    actors.insert(actor(BYSTANDER, 3.0).id.clone(), actor(BYSTANDER, 3.0));
    World {
        tick: 0,
        day: 0,
        minute_of_day: 480,
        locations,
        actors,
        items,
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

fn config() -> SimConfig {
    SimConfig {
        enable_background: false,
        enable_director: false,
        enable_goal_autonomy: false,
        ..SimConfig::default()
    }
}

fn a(id: &str) -> ActorId {
    ActorId::new(id)
}

/// The bare fixture with the creditor already holding one medicine, so the
/// condition can be delivered by a *recorded* transfer (needed for
/// replay-faithful tests) instead of a test-side inventory add.
fn fixture_with_condition_stock() -> World {
    let mut w = fixture();
    w.actor_mut(&a(CREDITOR))
        .unwrap()
        .inventory
        .add(ItemId::new("medicine"), 1);
    w
}

fn apply(sim: &mut Simulation, actor: &ActorId, action: Action) -> ValidationOutcome {
    sim.apply(&ActionProposal {
        actor: actor.clone(),
        action,
        source: ProposalSource::Player,
        rationale: String::new(),
    })
}

/// Propose a disclosure deal: `debtor` promises to disclose `f_ledger` once
/// it physically holds `qty` medicine. Returns the new deal id.
fn propose_disclosure_deal(sim: &mut Simulation, qty: u32) -> DealId {
    let out = apply(
        sim,
        &a(DEBTOR),
        Action::ProposeDeal {
            target: a(CREDITOR),
            terms: "medicine for the ledger".into(),
            condition: QuestPredicate::HasItem {
                actor: a(DEBTOR),
                item: ItemId::new("medicine"),
                qty,
            },
            performance: QuestPredicate::HeardFact {
                actor: a(CREDITOR),
                fact: FactId::new("f_ledger"),
            },
            condition_deadline_minute: None,
            performance_deadline_minute: None,
            give: vec![],
            money: 0,
            note: "bring medicine and I will tell you".into(),
        },
    );
    assert!(matches!(out, ValidationOutcome::Pending { .. }), "{out:?}");
    *sim.world.deals.keys().max().unwrap()
}

/// Advance a `Proposed` deal to `Active` (creditor accepts).
fn accept(sim: &mut Simulation, deal: DealId) {
    let out = apply(sim, &a(CREDITOR), Action::AcceptDeal { deal });
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
}

/// Deliver the condition (plain transfer, not a settlement): the deal's
/// trigger predicate holds without any contractual settlement. Requires the
/// creditor to already hold the quantity (the fixture does for replay-safe
/// tests; other tests add inventory directly).
fn deliver_condition(sim: &mut Simulation, qty: u32) {
    let out = apply(
        sim,
        &a(CREDITOR),
        Action::TransferItem {
            target: a(DEBTOR),
            item: ItemId::new("medicine"),
            qty,
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
}

fn deal_state(sim: &Simulation, deal: DealId) -> DealState {
    sim.world.deals[&deal].state
}

fn count_events(sim: &Simulation, pred: impl Fn(&EventKind) -> bool) -> usize {
    sim.events.all().iter().filter(|e| pred(&e.kind)).count()
}

// ---------------------------------------------------------------------------
// T2.2 — pinned lifecycle and exactly-once terminal transitions
// ---------------------------------------------------------------------------

#[test]
fn t2_2_pinned_lifecycle_and_exactly_once_terminal() {
    let world = fixture_with_condition_stock();
    let mut sim = Simulation::new(world, 5, config());
    let deal = propose_disclosure_deal(&mut sim, 1);
    assert_eq!(deal_state(&sim, deal), DealState::Proposed);

    accept(&mut sim, deal);
    assert_eq!(deal_state(&sim, deal), DealState::Active);

    // Only the creditor may accept.
    let world2 = fixture();
    let mut sim2 = Simulation::new(world2, 5, config());
    let deal2 = propose_disclosure_deal(&mut sim2, 1);
    let out = apply(&mut sim2, &a(DEBTOR), Action::AcceptDeal { deal: deal2 });
    assert!(matches!(out, ValidationOutcome::Rejected(_)), "{out:?}");
    assert_eq!(deal_state(&sim2, deal2), DealState::Proposed);

    // Condition met: the obligation becomes due — in one deterministic step.
    deliver_condition(&mut sim, 1);
    sim.step(Vec::new());
    assert_eq!(deal_state(&sim, deal), DealState::PerformanceDue);
    assert_eq!(
        count_events(&sim, |k| matches!(k, EventKind::DealConditionMet { .. })),
        1,
        "ConditionMet is its own event"
    );
    assert_eq!(
        count_events(&sim, |k| matches!(k, EventKind::DealPerformanceDue { .. })),
        1,
        "PerformanceDue is a separate event"
    );

    // The debtor settles by disclosing: the creditor hears the fact.
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::ShareClaim {
            target: a(CREDITOR),
            fact: FactId::new("f_ledger"),
            assertive: true,
            deceptive: false,
            retold_chain: None,
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    sim.step(Vec::new());
    assert_eq!(deal_state(&sim, deal), DealState::Performed);
    assert_eq!(
        count_events(&sim, |k| matches!(k, EventKind::DealPerformed { .. })),
        1
    );

    // Exactly-once: a second terminal transition is rejected.
    let out = apply(&mut sim, &a(DEBTOR), Action::CancelDeal { deal });
    assert!(matches!(out, ValidationOutcome::Rejected(_)), "{out:?}");
    assert_eq!(deal_state(&sim, deal), DealState::Performed);
}

/// The FSM rejects illegal arrows even through authoritative actions.
#[test]
fn t2_2_illegal_transitions_rejected_by_validation() {
    let world = fixture();
    let mut sim = Simulation::new(world, 5, config());
    let deal = propose_disclosure_deal(&mut sim, 1);
    // Cancelling a proposal is legal; settling it is not (not yet due).
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::SettleDeal {
            deal,
            deliver: vec![],
            money: 0,
        },
    );
    assert!(matches!(out, ValidationOutcome::Rejected(_)), "{out:?}");
    assert_eq!(deal_state(&sim, deal), DealState::Proposed);
    // Unknown deals are unknown.
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::AcceptDeal { deal: DealId(999) },
    );
    assert!(matches!(out, ValidationOutcome::Rejected(_)), "{out:?}");
}

// ---------------------------------------------------------------------------
// T2.3 — ConditionMet ≠ benefit delivered
// ---------------------------------------------------------------------------

#[test]
fn t2_3_condition_met_is_not_benefit_delivered() {
    let world = fixture_with_condition_stock();
    let mut sim = Simulation::new(world, 5, config());
    let deal = propose_disclosure_deal(&mut sim, 1);
    accept(&mut sim, deal);

    // The medicine physically arrives (trigger predicate holds), but nothing
    // has been disclosed yet. The step that meets the condition runs the
    // pipeline *before* maintenance, so the obligation is only due now.
    deliver_condition(&mut sim, 1);
    sim.step(Vec::new());
    assert_eq!(deal_state(&sim, deal), DealState::PerformanceDue);
    assert!(
        !sim.world.has_heard(&a(CREDITOR), &FactId::new("f_ledger")),
        "the creditor has heard nothing yet"
    );
    assert!(
        count_events(&sim, |k| matches!(k, EventKind::DealPerformed { .. })) == 0,
        "a met condition must not perform the deal"
    );

    // The production pipeline answers the due obligation: the debtor
    // discloses, and the validated world predicate (heard) performs the deal.
    sim.step(Vec::new());
    assert!(
        sim.world.has_heard(&a(CREDITOR), &FactId::new("f_ledger")),
        "the due disclosure was delivered"
    );
    assert_eq!(deal_state(&sim, deal), DealState::Performed);
    assert_eq!(
        count_events(&sim, |k| matches!(k, EventKind::DealConditionMet { .. })),
        1
    );
    assert_eq!(
        count_events(&sim, |k| matches!(k, EventKind::DealPerformed { .. })),
        1,
        "distinct events: condition met vs benefit delivered"
    );
}

// ---------------------------------------------------------------------------
// T2.4 — a false fulfillment claim never moves the FSM
// ---------------------------------------------------------------------------

#[test]
fn t2_4_false_fulfillment_claim_does_not_move_fsm() {
    let world = fixture_with_condition_stock();
    let mut sim = Simulation::new(world, 5, config());
    let deal = propose_disclosure_deal(&mut sim, 1);
    accept(&mut sim, deal);
    deliver_condition(&mut sim, 1);
    sim.step(Vec::new());
    assert_eq!(deal_state(&sim, deal), DealState::PerformanceDue);

    // A debtor who cannot truthfully disclose (here: one who does not believe
    // the fact — knowledge boundary) asserts completion in speech. The claim
    // is recorded as a claim (Assert effect/event), the FSM does not move.
    sim.world
        .actor_mut(&a(DEBTOR))
        .unwrap()
        .beliefs
        .remove(&FactId::new("f_ledger"));
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::Speak {
            target: Some(a(CREDITOR)),
            text: "Consider the ledger disclosed, our deal is done.".into(),
            loudness: rpg_domain::action::Loudness::Normal,
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    sim.step(Vec::new());
    assert_eq!(
        deal_state(&sim, deal),
        DealState::PerformanceDue,
        "speech is not settlement"
    );
    assert_eq!(
        count_events(&sim, |k| matches!(k, EventKind::DealPerformed { .. })),
        0
    );
    // The claim itself was recorded as evidence (heard, not performed).
    assert!(
        count_events(&sim, |k| matches!(k, EventKind::SpeechEmitted { .. })) >= 1,
        "the claim is recorded evidence"
    );
}

// ---------------------------------------------------------------------------
// T2.5a — escrow/atomicity
// ---------------------------------------------------------------------------

#[test]
fn t2_5a_escrow_settlement_is_atomic() {
    let world = fixture();
    let mut sim = Simulation::new(world, 5, config());
    // A material deal: settle by delivering 2 medicine + 10 coins.
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::ProposeDeal {
            target: a(CREDITOR),
            terms: "work for medicine and coin".into(),
            condition: QuestPredicate::Flag("work_done".into()),
            performance: QuestPredicate::All(vec![
                QuestPredicate::HasItem {
                    actor: a(CREDITOR),
                    item: ItemId::new("medicine"),
                    qty: 2,
                },
                QuestPredicate::HasMoney {
                    actor: a(CREDITOR),
                    amount: 10,
                },
            ]),
            condition_deadline_minute: None,
            performance_deadline_minute: None,
            give: vec![],
            money: 0,
            note: "deliver both".into(),
        },
    );
    assert!(matches!(out, ValidationOutcome::Pending { .. }), "{out:?}");
    let deal = *sim.world.deals.keys().max().unwrap();
    // Condition met by flag, obligation due.
    sim.world.set_flag("work_done");
    accept(&mut sim, deal);
    sim.step(Vec::new());
    assert_eq!(deal_state(&sim, deal), DealState::PerformanceDue);

    // The debtor holds the medicine but not the coins: the bundle is invalid,
    // so NOTHING moves (escrow = all-or-nothing inside one validate_then_apply).
    sim.world
        .actor_mut(&a(DEBTOR))
        .unwrap()
        .inventory
        .add(ItemId::new("medicine"), 2);
    let before = sim
        .world
        .actor(&a(DEBTOR))
        .unwrap()
        .inventory
        .quantity(&ItemId::new("medicine"));
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::SettleDeal {
            deal,
            deliver: vec![rpg_domain::world::ItemStack {
                item: ItemId::new("medicine"),
                qty: 2,
            }],
            money: 10,
        },
    );
    assert!(matches!(out, ValidationOutcome::Rejected(_)), "{out:?}");
    assert_eq!(
        sim.world
            .actor(&a(DEBTOR))
            .unwrap()
            .inventory
            .quantity(&ItemId::new("medicine")),
        before,
        "rejection mutates nothing"
    );
    assert_eq!(
        sim.world
            .actor(&a(CREDITOR))
            .unwrap()
            .inventory
            .quantity(&ItemId::new("medicine")),
        0,
        "no partial transfer happened"
    );
    assert_eq!(deal_state(&sim, deal), DealState::PerformanceDue);

    // Now the full bundle is valid: every component moves in one call.
    sim.world.actor_mut(&a(DEBTOR)).unwrap().inventory.money = 10;
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::SettleDeal {
            deal,
            deliver: vec![rpg_domain::world::ItemStack {
                item: ItemId::new("medicine"),
                qty: 2,
            }],
            money: 10,
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    assert_eq!(
        sim.world
            .actor(&a(CREDITOR))
            .unwrap()
            .inventory
            .quantity(&ItemId::new("medicine")),
        2
    );
    assert_eq!(sim.world.actor(&a(CREDITOR)).unwrap().inventory.money, 10);
    assert_eq!(sim.world.actor(&a(DEBTOR)).unwrap().inventory.money, 0);
    assert_eq!(deal_state(&sim, deal), DealState::Performed);
}

/// Give-on-acceptance consideration moves atomically at `Accepted`, before
/// the FSM enters `Active` (§4.2). There is no v1 inventory lock: goods stay
/// with their holder and remain freely transferable while a settlement is
/// merely *proposed*.
#[test]
fn t2_5a_consideration_moves_at_acceptance_and_no_lock_exists() {
    let world = fixture();
    let mut sim = Simulation::new(world, 5, config());
    // Deal with give-on-acceptance consideration: 1 medicine from the debtor.
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::ProposeDeal {
            target: a(CREDITOR),
            terms: "consideration up front".into(),
            condition: QuestPredicate::Flag("done".into()),
            performance: QuestPredicate::HeardFact {
                actor: a(CREDITOR),
                fact: FactId::new("f_ledger"),
            },
            condition_deadline_minute: None,
            performance_deadline_minute: None,
            give: vec![rpg_domain::world::ItemStack {
                item: ItemId::new("medicine"),
                qty: 1,
            }],
            money: 0,
            note: "take this now".into(),
        },
    );
    assert!(matches!(out, ValidationOutcome::Pending { .. }), "{out:?}");
    let deal = *sim.world.deals.keys().max().unwrap();

    // No lock: the debtor can still give the medicine to the bystander while
    // the deal is only proposed. Custody is checked at settlement time.
    sim.world
        .actor_mut(&a(DEBTOR))
        .unwrap()
        .inventory
        .add(ItemId::new("medicine"), 1);
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::TransferItem {
            target: a(BYSTANDER),
            item: ItemId::new("medicine"),
            qty: 1,
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");

    // Accepting now fails the custody check for the whole bundle.
    let out = apply(&mut sim, &a(CREDITOR), Action::AcceptDeal { deal });
    assert!(matches!(out, ValidationOutcome::Rejected(_)), "{out:?}");
    assert_eq!(deal_state(&sim, deal), DealState::Proposed);
    assert_eq!(
        sim.world
            .actor(&a(BYSTANDER))
            .unwrap()
            .inventory
            .quantity(&ItemId::new("medicine")),
        1,
        "nothing moved on the rejected acceptance"
    );

    // With custody restored, acceptance moves the consideration atomically
    // and the deal is Active immediately.
    sim.world
        .actor_mut(&a(BYSTANDER))
        .unwrap()
        .inventory
        .remove(&ItemId::new("medicine"), 1);
    sim.world
        .actor_mut(&a(DEBTOR))
        .unwrap()
        .inventory
        .add(ItemId::new("medicine"), 1);
    accept(&mut sim, deal);
    assert_eq!(
        sim.world
            .actor(&a(CREDITOR))
            .unwrap()
            .inventory
            .quantity(&ItemId::new("medicine")),
        1,
        "consideration moved at Accepted"
    );
    assert_eq!(deal_state(&sim, deal), DealState::Active);
}

// ---------------------------------------------------------------------------
// T2.5b — partial obligations
// ---------------------------------------------------------------------------

#[test]
fn t2_5b_installments_settle_exactly_once() {
    let world = fixture();
    let mut sim = Simulation::new(world, 5, config());
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::ProposeDeal {
            target: a(CREDITOR),
            terms: "two installments".into(),
            condition: QuestPredicate::Flag("ordered".into()),
            performance: QuestPredicate::HasMoney {
                actor: a(CREDITOR),
                amount: 4,
            },
            condition_deadline_minute: None,
            performance_deadline_minute: None,
            give: vec![],
            money: 0,
            note: "coin by coin".into(),
        },
    );
    assert!(matches!(out, ValidationOutcome::Pending { .. }), "{out:?}");
    let deal = *sim.world.deals.keys().max().unwrap();
    // Split into two installments of 2 coins each (declared quantities).
    {
        use rpg_domain::deal::Installment;
        let d = sim.world.deals.get_mut(&deal).unwrap();
        d.installments = vec![
            Installment {
                performance: QuestPredicate::HasMoney {
                    actor: a(CREDITOR),
                    amount: 2,
                },
                expected_qty: 2,
                settled: false,
                settled_tick: None,
            },
            Installment {
                performance: QuestPredicate::HasMoney {
                    actor: a(CREDITOR),
                    amount: 4,
                },
                expected_qty: 2,
                settled: false,
                settled_tick: None,
            },
        ];
    }
    sim.world.set_flag("ordered");
    accept(&mut sim, deal);
    sim.step(Vec::new());
    assert_eq!(deal_state(&sim, deal), DealState::PerformanceDue);

    sim.world.actor_mut(&a(DEBTOR)).unwrap().inventory.money = 4;
    // First installment: 2 coins.
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::SettleDeal {
            deal,
            deliver: vec![],
            money: 2,
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    let d = &sim.world.deals[&deal];
    assert_eq!(d.performed_installments(), 1);
    assert_eq!(d.total_installments(), 2);
    assert_eq!(
        count_events(&sim, |k| matches!(
            k,
            EventKind::DealInstallmentSettled { .. }
        )),
        1
    );
    assert_eq!(deal_state(&sim, deal), DealState::PerformanceDue);

    // A second settlement of the *same* bundle would satisfy installment 2's
    // predicate (>= 2 more coins are not available, so the first unsettled
    // installment's predicate fails): the exactly-once guard for installment
    // 1 is exercised directly on the recorded state below. First, pay the
    // second installment and reach Performed only when all are settled.
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::SettleDeal {
            deal,
            deliver: vec![],
            money: 2,
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    assert_eq!(deal_state(&sim, deal), DealState::Performed);
    let d = &sim.world.deals[&deal];
    assert_eq!(d.performed_installments(), 2);
    assert_eq!(
        count_events(&sim, |k| matches!(k, EventKind::DealPerformed { .. })),
        1,
        "the deal performed exactly once, after the last installment"
    );

    // Exactly-once per installment: settling beyond the terminal deal is
    // rejected wholesale (the whole FSM is consumed).
    sim.world.actor_mut(&a(DEBTOR)).unwrap().inventory.money = 2;
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::SettleDeal {
            deal,
            deliver: vec![],
            money: 2,
        },
    );
    assert!(matches!(out, ValidationOutcome::Rejected(_)), "{out:?}");
}

/// An un-split obligation delivered short stays `PerformanceDue`, emits a
/// typed `DealPartialDelivery { got, expected }` event and mutates nothing
/// contractual; completing the delivery later performs the deal.
#[test]
fn t2_5b_partial_delivery_mutates_nothing_contractual() {
    let world = fixture();
    let mut sim = Simulation::new(world, 5, config());
    // Material deal: deliver 3 medicine to the creditor.
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::ProposeDeal {
            target: a(CREDITOR),
            terms: "three doses".into(),
            condition: QuestPredicate::Flag("ordered".into()),
            performance: QuestPredicate::HasItem {
                actor: a(CREDITOR),
                item: ItemId::new("medicine"),
                qty: 3,
            },
            condition_deadline_minute: None,
            performance_deadline_minute: None,
            give: vec![],
            money: 0,
            note: "three doses".into(),
        },
    );
    assert!(matches!(out, ValidationOutcome::Pending { .. }), "{out:?}");
    let deal = *sim.world.deals.keys().max().unwrap();
    sim.world.set_flag("ordered");
    accept(&mut sim, deal);
    sim.step(Vec::new());
    assert_eq!(deal_state(&sim, deal), DealState::PerformanceDue);

    // Short delivery: 1 of 3.
    sim.world
        .actor_mut(&a(DEBTOR))
        .unwrap()
        .inventory
        .add(ItemId::new("medicine"), 3);
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::SettleDeal {
            deal,
            deliver: vec![rpg_domain::world::ItemStack {
                item: ItemId::new("medicine"),
                qty: 1,
            }],
            money: 0,
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    assert_eq!(deal_state(&sim, deal), DealState::PerformanceDue);
    let partials: Vec<(u32, u32)> = sim
        .events
        .all()
        .iter()
        .filter_map(|e| match &e.kind {
            EventKind::DealPartialDelivery { got, expected, .. } => Some((*got, *expected)),
            _ => None,
        })
        .collect();
    assert_eq!(partials, vec![(1, 3)], "typed shortfall evidence");
    assert_eq!(
        count_events(&sim, |k| matches!(k, EventKind::DealPerformed { .. })),
        0
    );

    // The delivered dose is a real transfer (the creditor holds it), but the
    // contract did not move. The debtor delivers the rest; the contracted
    // predicate now holds and the deal performs exactly-once.
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::SettleDeal {
            deal,
            deliver: vec![rpg_domain::world::ItemStack {
                item: ItemId::new("medicine"),
                qty: 2,
            }],
            money: 0,
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    assert_eq!(deal_state(&sim, deal), DealState::Performed);
    assert_eq!(
        sim.world
            .actor(&a(CREDITOR))
            .unwrap()
            .inventory
            .quantity(&ItemId::new("medicine")),
        3,
        "conservation: the creditor received all three doses"
    );
    assert_eq!(
        count_events(&sim, |k| matches!(k, EventKind::DealPerformed { .. })),
        1
    );
}

// ---------------------------------------------------------------------------
// T2.5c — interruption/cancellation authorization matrix
// ---------------------------------------------------------------------------

#[test]
fn t2_5c_cancellation_authorization_matrix() {
    // Creditor cancels a due deal: Canceled.
    let world = fixture_with_condition_stock();
    let mut sim = Simulation::new(world, 5, config());
    let deal = propose_disclosure_deal(&mut sim, 1);
    accept(&mut sim, deal);
    deliver_condition(&mut sim, 1);
    sim.step(Vec::new());
    let out = apply(&mut sim, &a(CREDITOR), Action::CancelDeal { deal });
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    assert_eq!(deal_state(&sim, deal), DealState::Canceled);
    let events_before = sim.events.len();

    // A terminal deal is consumed: a second cancellation is rejected. Only
    // the typed rejection record is appended; no contractual event appears
    // and the history is otherwise intact.
    let out = apply(&mut sim, &a(DEBTOR), Action::CancelDeal { deal });
    assert!(matches!(out, ValidationOutcome::Rejected(_)), "{out:?}");
    assert_eq!(
        sim.events.len(),
        events_before + 1,
        "only the rejection record"
    );
    assert_eq!(
        count_events(&sim, |k| matches!(
            k,
            EventKind::DealCanceled { .. } | EventKind::DealViolated { .. }
        )),
        1,
        "the terminal deal produced no further lifecycle events"
    );

    // Debtor cancels a due deal unilaterally: the Violated path (§4.4).
    let world = fixture_with_condition_stock();
    let mut sim = Simulation::new(world, 5, config());
    let deal = propose_disclosure_deal(&mut sim, 1);
    accept(&mut sim, deal);
    deliver_condition(&mut sim, 1);
    sim.step(Vec::new());
    let out = apply(&mut sim, &a(DEBTOR), Action::CancelDeal { deal });
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    assert_eq!(
        deal_state(&sim, deal),
        DealState::Violated,
        "a debtor walking away past ConditionMet is a violation"
    );
    assert_eq!(
        count_events(&sim, |k| matches!(k, EventKind::DealViolated { .. })),
        1
    );

    // Debtor cancels *before* ConditionMet (Active): Canceled.
    let world = fixture_with_condition_stock();
    let mut sim = Simulation::new(world, 5, config());
    let deal = propose_disclosure_deal(&mut sim, 1);
    accept(&mut sim, deal);
    let out = apply(&mut sim, &a(DEBTOR), Action::CancelDeal { deal });
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    assert_eq!(deal_state(&sim, deal), DealState::Canceled);
}

// ---------------------------------------------------------------------------
// T2.5d — deadlines: idempotent ObligationDue + grace → Violated
// ---------------------------------------------------------------------------

#[test]
fn t2_5d_deadline_obligation_due_is_idempotent_then_violated() {
    let world = fixture();
    let mut cfg = config();
    cfg.deal_grace_minutes = 1;
    let mut sim = Simulation::new(world, 5, cfg);
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::ProposeDeal {
            target: a(CREDITOR),
            terms: "timely delivery".into(),
            condition: QuestPredicate::Flag("ordered".into()),
            performance: QuestPredicate::HasItem {
                actor: a(CREDITOR),
                item: ItemId::new("medicine"),
                qty: 1,
            },
            condition_deadline_minute: None,
            performance_deadline_minute: Some(481),
            give: vec![],
            money: 0,
            note: "deliver by minute 481".into(),
        },
    );
    assert!(matches!(out, ValidationOutcome::Pending { .. }), "{out:?}");
    let deal = *sim.world.deals.keys().max().unwrap();
    sim.world.set_flag("ordered");
    accept(&mut sim, deal);
    sim.step(Vec::new()); // now = 481: the performance deadline is crossed
    assert_eq!(deal_state(&sim, deal), DealState::PerformanceDue);
    assert_eq!(
        count_events(&sim, |k| matches!(k, EventKind::ObligationDue { .. })),
        1,
        "exactly one ObligationDue on the crossing tick"
    );

    // The grace window (one minute) still runs; the idempotent event does
    // not repeat and the contract state is untouched.
    sim.step(Vec::new()); // now = 482: grace runs until 482
    assert_eq!(
        count_events(&sim, |k| matches!(k, EventKind::ObligationDue { .. })),
        1,
        "idempotent by state check"
    );
    assert_eq!(deal_state(&sim, deal), DealState::PerformanceDue);

    // Grace elapsed: the deal is violated with a typed event.
    sim.step(Vec::new()); // now = 483 > 482
    assert_eq!(deal_state(&sim, deal), DealState::Violated);
    assert_eq!(
        count_events(&sim, |k| matches!(k, EventKind::DealViolated { .. })),
        1
    );
    // Deadline processing ran inside the deterministic step pipeline: no
    // out-of-band mutation happened (the violation is event-sourced).
    assert!(
        sim.events
            .all()
            .iter()
            .any(|e| matches!(e.kind, EventKind::DealViolated { .. })),
        "the violation is recorded in the event log"
    );
}

// ---------------------------------------------------------------------------
// T2.5e — exactly-once settlement under replay
// ---------------------------------------------------------------------------

#[test]
fn t2_5e_settlement_is_exactly_once_and_replay_safe() {
    let world = fixture_with_condition_stock();
    let initial_world = world.clone();
    let mut sim = Simulation::new(world, 5, config());
    let deal = propose_disclosure_deal(&mut sim, 1);
    accept(&mut sim, deal);
    deliver_condition(&mut sim, 1);
    sim.step(Vec::new());
    assert_eq!(deal_state(&sim, deal), DealState::PerformanceDue);

    // A second settlement attempt after performance is rejected.
    sim.step(Vec::new()); // pipeline honours the disclosure -> Performed
    assert_eq!(deal_state(&sim, deal), DealState::Performed);
    let checksum = sim.events.checksum();

    // Replay reproduces the full event log and the deal state exactly; the
    // recorded rejection (below) replays to the same rejection, so replaying
    // the log cannot double-settle.
    sim.world.actor_mut(&a(DEBTOR)).unwrap().inventory.money = 1;
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::SettleDeal {
            deal,
            deliver: vec![],
            money: 1,
        },
    );
    assert!(matches!(out, ValidationOutcome::Rejected(_)), "{out:?}");
    assert_eq!(sim.world.deals[&deal].state, DealState::Performed);

    let replay = Simulation::replay_ops(initial_world, 5, config(), &sim.ops);
    assert_eq!(replay.events.checksum(), sim.events.checksum());
    assert_eq!(replay.world.deals[&deal].state, DealState::Performed);
    assert_eq!(
        count_events(&replay, |k| matches!(k, EventKind::DealPerformed { .. })),
        1,
        "replay cannot double-settle"
    );
    assert_eq!(replay.world.deals[&deal], sim.world.deals[&deal]);
    let _ = checksum;
}

// ---------------------------------------------------------------------------
// Rejection neutrality, conservation, and the production pipeline
// ---------------------------------------------------------------------------

/// A rejected deal action mutates nothing: no effects, no events beyond the
/// typed rejection record, inventories intact.
#[test]
fn rejected_deal_actions_are_neutral() {
    let world = fixture();
    let mut sim = Simulation::new(world, 5, config());
    sim.world.actor_mut(&a(DEBTOR)).unwrap().inventory.money = 5;
    let deal = propose_disclosure_deal(&mut sim, 1);
    let before = sim.world.clone();
    let events_before = sim.events.len();
    let outcomes_before = sim.outcomes.len();
    // Settling a proposed (not due) deal is rejected.
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::SettleDeal {
            deal,
            deliver: vec![],
            money: 5,
        },
    );
    assert!(matches!(out, ValidationOutcome::Rejected(_)), "{out:?}");
    // No new world state beyond the rejection record itself.
    assert_eq!(sim.events.len(), events_before + 1);
    assert_eq!(sim.outcomes.len(), outcomes_before + 1);
    let outcome = sim.outcomes.last().unwrap();
    assert!(outcome.effects.is_empty(), "no effects on rejection");
    assert!(outcome.events.is_empty(), "no events on rejection");
    // Inventories identical.
    assert_eq!(
        sim.world.actor(&a(DEBTOR)).unwrap().inventory.money,
        before.actor(&a(DEBTOR)).unwrap().inventory.money
    );
    assert_eq!(deal_state(&sim, deal), DealState::Proposed);
}

/// Money and items are conserved through a settlement bundle (design §9.1.1
/// at the contract seam).
#[test]
fn settlement_conserves_resources() {
    let world = fixture();
    let mut sim = Simulation::new(world, 5, config());
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::ProposeDeal {
            target: a(CREDITOR),
            terms: "pay on delivery".into(),
            condition: QuestPredicate::Flag("ordered".into()),
            performance: QuestPredicate::HasMoney {
                actor: a(CREDITOR),
                amount: 25,
            },
            condition_deadline_minute: None,
            performance_deadline_minute: None,
            give: vec![],
            money: 0,
            note: "twenty five".into(),
        },
    );
    assert!(matches!(out, ValidationOutcome::Pending { .. }), "{out:?}");
    let deal = *sim.world.deals.keys().max().unwrap();
    sim.world.set_flag("ordered");
    accept(&mut sim, deal);
    sim.step(Vec::new());
    sim.world.actor_mut(&a(DEBTOR)).unwrap().inventory.money = 25;
    let total_before: i64 = sim.world.actors.values().map(|a| a.inventory.money).sum();
    let medicine_before: u32 = sim
        .world
        .actors
        .values()
        .map(|a| a.inventory.quantity(&ItemId::new("medicine")))
        .sum();
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::SettleDeal {
            deal,
            deliver: vec![],
            money: 25,
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    let total_after: i64 = sim.world.actors.values().map(|a| a.inventory.money).sum();
    let medicine_after: u32 = sim
        .world
        .actors
        .values()
        .map(|a| a.inventory.quantity(&ItemId::new("medicine")))
        .sum();
    assert_eq!(total_before, total_after, "money conserved");
    assert_eq!(medicine_before, medicine_after, "items conserved");
    assert_eq!(deal_state(&sim, deal), DealState::Performed);
}

/// Renegotiation is wired end-to-end: the creditor consents to the single
/// cycle, the deal re-enters `Active` with amended deadlines, and a second
/// renegotiation attempt is rejected (single-cycle policy, §4.1).
#[test]
fn renegotiation_wired_end_to_end_single_cycle() {
    let world = fixture();
    let mut sim = Simulation::new(world, 5, config());
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::ProposeDeal {
            target: a(CREDITOR),
            terms: "deliver later".into(),
            condition: QuestPredicate::Flag("ordered".into()),
            performance: QuestPredicate::HasItem {
                actor: a(CREDITOR),
                item: ItemId::new("medicine"),
                qty: 1,
            },
            condition_deadline_minute: None,
            performance_deadline_minute: Some(481),
            give: vec![],
            money: 0,
            note: "due soon".into(),
        },
    );
    assert!(matches!(out, ValidationOutcome::Pending { .. }), "{out:?}");
    let deal = *sim.world.deals.keys().max().unwrap();
    sim.world.set_flag("ordered");
    accept(&mut sim, deal);
    sim.step(Vec::new()); // deadline crossed -> ObligationDue, grace running
    assert_eq!(deal_state(&sim, deal), DealState::PerformanceDue);

    // Only the creditor may consent to the amendment.
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::RenegotiateDeal {
            deal,
            new_condition: None,
            new_performance: None,
            new_condition_deadline_minute: None,
            new_performance_deadline_minute: Some(600),
        },
    );
    assert!(matches!(out, ValidationOutcome::Rejected(_)), "{out:?}");

    // The creditor consents: the deal re-enters Active with the new
    // performance deadline, and the overdue state is reset.
    let out = apply(
        &mut sim,
        &a(CREDITOR),
        Action::RenegotiateDeal {
            deal,
            new_condition: None,
            new_performance: None,
            new_condition_deadline_minute: None,
            new_performance_deadline_minute: Some(600),
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    assert_eq!(deal_state(&sim, deal), DealState::Active);
    assert_eq!(
        sim.world.deals[&deal].performance_deadline_minute,
        Some(600)
    );
    assert_eq!(sim.world.deals[&deal].renegotiations, 1);
    assert!(count_events(&sim, |k| matches!(k, EventKind::DealRenegotiated { .. })) == 1);

    // The cycle is spent: a second renegotiation is rejected.
    let out = apply(
        &mut sim,
        &a(CREDITOR),
        Action::RenegotiateDeal {
            deal,
            new_condition: None,
            new_performance: None,
            new_condition_deadline_minute: None,
            new_performance_deadline_minute: Some(700),
        },
    );
    assert!(matches!(out, ValidationOutcome::Rejected(_)), "{out:?}");
}

/// The production deal-response pipeline: an NPC with a due, satisfiable
/// obligation proposes the settlement itself, adopts an intention (fast path
/// on the next tick), and the deal performs.
#[test]
fn deal_pipeline_ranks_and_adopts_intentions() {
    let world = fixture_with_condition_stock();
    let mut sim = Simulation::new(world, 5, config());
    let deal = propose_disclosure_deal(&mut sim, 1);
    accept(&mut sim, deal);
    deliver_condition(&mut sim, 1);
    // Step 1: the condition is met and the obligation becomes due (the
    // pipeline runs *before* maintenance, so it answers on the next tick).
    sim.step(Vec::new());
    assert_eq!(deal_state(&sim, deal), DealState::PerformanceDue);
    assert!(
        !sim.deal_intentions.contains_key(&a(DEBTOR)),
        "nothing to answer before the obligation was due"
    );
    // Step 2: the pipeline ranks the responses, adopts the best as an
    // intention, discloses, and the validated predicate performs the deal.
    sim.step(Vec::new());
    assert!(sim.world.has_heard(&a(CREDITOR), &FactId::new("f_ledger")));
    assert_eq!(deal_state(&sim, deal), DealState::Performed);
    assert_eq!(
        sim.deal_intentions
            .get(&a(DEBTOR))
            .map(|i| i.option.as_str()),
        Some("honour_disclosure"),
        "the ranked best response is adopted as an intention"
    );

    // Intention persistence across ticks (T2.1): the adopted plan survives
    // until invalidated; once the deal performed, the satisfaction predicate
    // holds and the intention is cleared on the next pass.
    sim.step(Vec::new());
    assert!(
        !sim.deal_intentions.contains_key(&a(DEBTOR)),
        "the satisfied intention is retired"
    );
}

/// When the creditor is out of range, delivery is not physically feasible
/// now: the ranking falls back to stalling and the contract state is
/// untouched until the parties are together.
#[test]
fn deal_pipeline_stalls_when_physically_infeasible() {
    let world = fixture();
    let mut sim = Simulation::new(world, 5, config());
    let deal = propose_disclosure_deal(&mut sim, 1);
    accept(&mut sim, deal);
    // Now the creditor walks to the far corner of the room (talk radius 3.5):
    // the obligation can no longer be delivered from across the room.
    if let Some(c) = sim.world.actors.get_mut(&a(CREDITOR)) {
        c.pos = Vec2::new(9.0, 9.0);
    }
    // The condition arrives from a nearby hand (no transfer needed: the
    // trigger predicate is about possession, not provenance).
    sim.world
        .actor_mut(&a(DEBTOR))
        .unwrap()
        .inventory
        .add(ItemId::new("medicine"), 1);
    sim.step(Vec::new());
    assert_eq!(deal_state(&sim, deal), DealState::PerformanceDue);
    assert!(
        !sim.world.has_heard(&a(CREDITOR), &FactId::new("f_ledger")),
        "nothing was delivered from across the room"
    );
    // The pipeline answers on the next tick: delivery is not physically
    // feasible now, so the ranking falls back to stalling.
    sim.step(Vec::new());
    let intent = sim.deal_intentions.get(&a(DEBTOR)).unwrap();
    assert_eq!(intent.option, "stall", "the ranked fallback is stalling");
    assert_eq!(deal_state(&sim, deal), DealState::PerformanceDue);
}

/// Every accepted deal action lowers into typed primitive effects and is
/// event-sourced (M1 invariant at the contract seam). A rejected deal action
/// produces neither (covered by `rejected_deal_actions_are_neutral`).
#[test]
fn deal_actions_produce_typed_effects_and_events() {
    let world = fixture();
    let mut sim = Simulation::new(world, 5, config());
    sim.world.actor_mut(&a(DEBTOR)).unwrap().inventory.money = 6;

    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::ProposeDeal {
            target: a(CREDITOR),
            terms: "settle in coin".into(),
            condition: QuestPredicate::Flag("ordered".into()),
            performance: QuestPredicate::HasMoney {
                actor: a(CREDITOR),
                amount: 6,
            },
            condition_deadline_minute: None,
            performance_deadline_minute: None,
            give: vec![],
            money: 0,
            note: "coin".into(),
        },
    );
    assert!(matches!(out, ValidationOutcome::Pending { .. }), "{out:?}");
    let outcome = sim.outcomes.last().unwrap();
    assert!(outcome.effects.iter().any(|e| e.family() == "create_event"));
    assert!(!outcome.events.is_empty());
    assert!(!outcome.trace.render().is_empty());
    let deal = *sim.world.deals.keys().max().unwrap();

    sim.world.set_flag("ordered");
    let out = apply(&mut sim, &a(CREDITOR), Action::AcceptDeal { deal });
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    let outcome = sim.outcomes.last().unwrap();
    assert!(outcome.effects.iter().any(|e| e.family() == "create_event"));
    assert!(!outcome.events.is_empty());

    sim.step(Vec::new());
    assert_eq!(deal_state(&sim, deal), DealState::PerformanceDue);
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::SettleDeal {
            deal,
            deliver: vec![],
            money: 6,
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    let outcome = sim.outcomes.last().unwrap();
    assert!(outcome.effects.iter().any(|e| e.family() == "transfer"));
    assert!(outcome.effects.iter().any(|e| e.family() == "create_event"));
    assert!(!outcome.events.is_empty());
    assert_eq!(deal_state(&sim, deal), DealState::Performed);

    // Cancel/renegotiate/reject paths are typed too: propose a second deal
    // and cancel it.
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::ProposeDeal {
            target: a(CREDITOR),
            terms: "void".into(),
            condition: QuestPredicate::Flag("never".into()),
            performance: QuestPredicate::HasMoney {
                actor: a(CREDITOR),
                amount: 1,
            },
            condition_deadline_minute: None,
            performance_deadline_minute: None,
            give: vec![],
            money: 0,
            note: "void".into(),
        },
    );
    assert!(matches!(out, ValidationOutcome::Pending { .. }), "{out:?}");
    let deal2 = *sim.world.deals.keys().max().unwrap();
    let out = apply(&mut sim, &a(CREDITOR), Action::CancelDeal { deal: deal2 });
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    let outcome = sim.outcomes.last().unwrap();
    assert!(outcome.effects.iter().any(|e| e.family() == "create_event"));
    assert_eq!(deal_state(&sim, deal2), DealState::Canceled);
}

/// A deal survives save/load exactly: the persisted world round-trips the
/// FSM state, and the restored simulation continues with the same
/// exactly-once guarantees.
#[test]
fn deal_survives_persistence_round_trip() {
    let world = fixture_with_condition_stock();
    let mut sim = Simulation::new(world, 5, config());
    let deal = propose_disclosure_deal(&mut sim, 1);
    accept(&mut sim, deal);
    deliver_condition(&mut sim, 1);
    sim.step(Vec::new());
    assert_eq!(deal_state(&sim, deal), DealState::PerformanceDue);

    let save = rpg_persistence::SaveFile::from_sim(&sim);
    let json = rpg_persistence::to_json_string(&save).unwrap();
    let loaded = rpg_persistence::from_json_str(&json).unwrap();
    assert_eq!(loaded.version, rpg_persistence::SAVE_VERSION);
    let restored = rpg_persistence::restore(loaded, config());
    assert_eq!(restored.world.deals[&deal].state, DealState::PerformanceDue);
    assert_eq!(restored.events.checksum(), sim.events.checksum());

    // Continuation on the restored simulation: the pipeline honours the due
    // disclosure exactly as the original would.
    let mut restored = restored;
    restored.step(Vec::new());
    assert_eq!(restored.world.deals[&deal].state, DealState::Performed);
    assert_eq!(restored.world.deals[&deal], {
        sim.step(Vec::new());
        sim.world.deals[&deal].clone()
    });
}

/// Repeated delivery attempts cannot double-settle a performed deal, and the
/// exactly-once guard holds through the whole recorded step sequence.
#[test]
fn repeated_settlement_attempts_cannot_double_settle() {
    let world = fixture_with_condition_stock();
    let mut sim = Simulation::new(world, 5, config());
    let deal = propose_disclosure_deal(&mut sim, 1);
    accept(&mut sim, deal);
    deliver_condition(&mut sim, 1);
    sim.step(Vec::new());
    assert_eq!(deal_state(&sim, deal), DealState::PerformanceDue);
    // The pipeline answers on the next tick and performs exactly once.
    sim.step(Vec::new());
    assert_eq!(deal_state(&sim, deal), DealState::Performed);
    // Extra ticks: no further lifecycle events, no double settlement.
    let performed_after_first =
        count_events(&sim, |k| matches!(k, EventKind::DealPerformed { .. }));
    sim.step(Vec::new());
    sim.step(Vec::new());
    assert_eq!(
        count_events(&sim, |k| matches!(k, EventKind::DealPerformed { .. })),
        performed_after_first
    );
}

// ---------------------------------------------------------------------------
// Repair tests — F1/F2/F5 bounded Phase-2 defects (review_recovered_phase2)
// ---------------------------------------------------------------------------

/// Propose a material deal: the debtor promises to deliver `qty` medicine to
/// the creditor once the `ordered` flag holds. Returns the new deal id.
fn propose_delivery_deal(sim: &mut Simulation, qty: u32) -> DealId {
    let out = apply(
        sim,
        &a(DEBTOR),
        Action::ProposeDeal {
            target: a(CREDITOR),
            terms: "material delivery".into(),
            condition: QuestPredicate::Flag("ordered".into()),
            performance: QuestPredicate::HasItem {
                actor: a(CREDITOR),
                item: ItemId::new("medicine"),
                qty,
            },
            condition_deadline_minute: None,
            performance_deadline_minute: None,
            give: vec![],
            money: 0,
            note: "deliver the doses".into(),
        },
    );
    assert!(matches!(out, ValidationOutcome::Pending { .. }), "{out:?}");
    *sim.world.deals.keys().max().unwrap()
}

fn medicine_total(sim: &Simulation) -> u32 {
    sim.world
        .actors
        .values()
        .map(|a| a.inventory.quantity(&ItemId::new("medicine")))
        .sum()
}

/// F1 (§4.2): consideration custody aggregates per item across duplicate
/// stacks with checked arithmetic. A bundle asking 20 against holdings of 10
/// is rejected wholesale on acceptance — the old per-stack check passed each
/// stack and the second one silently moved nothing — while an adequately
/// funded bundle moves atomically and conserves resources.
#[test]
fn t2_5a_duplicate_stack_consideration_is_validated_in_aggregate() {
    let world = fixture();
    let mut sim = Simulation::new(world, 5, config());
    sim.world
        .actor_mut(&a(DEBTOR))
        .unwrap()
        .inventory
        .add(ItemId::new("medicine"), 10);
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::ProposeDeal {
            target: a(CREDITOR),
            terms: "duplicate stacks".into(),
            condition: QuestPredicate::Flag("ordered".into()),
            performance: QuestPredicate::Flag("never".into()),
            condition_deadline_minute: None,
            performance_deadline_minute: None,
            give: vec![
                rpg_domain::world::ItemStack {
                    item: ItemId::new("medicine"),
                    qty: 10,
                },
                rpg_domain::world::ItemStack {
                    item: ItemId::new("medicine"),
                    qty: 10,
                },
            ],
            money: 0,
            note: "twenty doses up front".into(),
        },
    );
    assert!(matches!(out, ValidationOutcome::Pending { .. }), "{out:?}");
    let deal = *sim.world.deals.keys().max().unwrap();
    sim.world.set_flag("ordered");

    // Aggregate 20 > holdings 10: rejected, nothing mutates.
    let out = apply(&mut sim, &a(CREDITOR), Action::AcceptDeal { deal });
    assert!(matches!(out, ValidationOutcome::Rejected(_)), "{out:?}");
    assert_eq!(deal_state(&sim, deal), DealState::Proposed);
    assert_eq!(
        sim.world
            .actor(&a(DEBTOR))
            .unwrap()
            .inventory
            .quantity(&ItemId::new("medicine")),
        10,
        "nothing moved on the rejected acceptance"
    );
    assert_eq!(
        sim.world
            .actor(&a(CREDITOR))
            .unwrap()
            .inventory
            .quantity(&ItemId::new("medicine")),
        0
    );

    // Adequately funded: the whole bundle moves atomically at Accepted and
    // resources are conserved.
    sim.world
        .actor_mut(&a(DEBTOR))
        .unwrap()
        .inventory
        .add(ItemId::new("medicine"), 10);
    let total_before = medicine_total(&sim);
    let out = apply(&mut sim, &a(CREDITOR), Action::AcceptDeal { deal });
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    assert_eq!(deal_state(&sim, deal), DealState::Active);
    assert_eq!(
        sim.world
            .actor(&a(DEBTOR))
            .unwrap()
            .inventory
            .quantity(&ItemId::new("medicine")),
        0
    );
    assert_eq!(
        sim.world
            .actor(&a(CREDITOR))
            .unwrap()
            .inventory
            .quantity(&ItemId::new("medicine")),
        20
    );
    assert_eq!(medicine_total(&sim), total_before, "items conserved");
}

/// F1 (§4.2): the settlement route aggregates duplicate stacks the same way.
/// A short bundle is rejected wholesale with no `DealPartialDelivery`
/// evidence and nothing mutated; the fully funded aggregate performs.
#[test]
fn t2_5a_duplicate_stack_settlement_is_validated_in_aggregate() {
    let world = fixture();
    let mut sim = Simulation::new(world, 5, config());
    let deal = propose_delivery_deal(&mut sim, 20);
    sim.world.set_flag("ordered");
    accept(&mut sim, deal);
    sim.step(Vec::new());
    assert_eq!(deal_state(&sim, deal), DealState::PerformanceDue);
    sim.world
        .actor_mut(&a(DEBTOR))
        .unwrap()
        .inventory
        .add(ItemId::new("medicine"), 10);

    // Aggregate 20 > holdings 10: rejected wholesale, no shortfall evidence,
    // nothing mutated (the old code recorded got=20 from the requested sums
    // while only 10 actually moved).
    let total_before = medicine_total(&sim);
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::SettleDeal {
            deal,
            deliver: vec![
                rpg_domain::world::ItemStack {
                    item: ItemId::new("medicine"),
                    qty: 10,
                },
                rpg_domain::world::ItemStack {
                    item: ItemId::new("medicine"),
                    qty: 10,
                },
            ],
            money: 0,
        },
    );
    assert!(matches!(out, ValidationOutcome::Rejected(_)), "{out:?}");
    assert_eq!(deal_state(&sim, deal), DealState::PerformanceDue);
    assert_eq!(medicine_total(&sim), total_before, "rejection neutrality");
    assert_eq!(
        count_events(&sim, |k| matches!(
            k,
            EventKind::DealPartialDelivery { .. } | EventKind::DealPerformed { .. }
        )),
        0,
        "a rejected bundle produces no settlement evidence"
    );

    // Fully funded: the exact aggregate performs the deal exactly once.
    sim.world
        .actor_mut(&a(DEBTOR))
        .unwrap()
        .inventory
        .add(ItemId::new("medicine"), 10);
    let total_before = medicine_total(&sim);
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::SettleDeal {
            deal,
            deliver: vec![
                rpg_domain::world::ItemStack {
                    item: ItemId::new("medicine"),
                    qty: 10,
                },
                rpg_domain::world::ItemStack {
                    item: ItemId::new("medicine"),
                    qty: 10,
                },
            ],
            money: 0,
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    assert_eq!(deal_state(&sim, deal), DealState::Performed);
    assert_eq!(
        sim.world
            .actor(&a(CREDITOR))
            .unwrap()
            .inventory
            .quantity(&ItemId::new("medicine")),
        20
    );
    assert_eq!(medicine_total(&sim), total_before, "items conserved");
}

/// F1 (§4.3): partial-delivery evidence records *actual* transfers. A
/// genuine short delivery across duplicate stacks records `got` as the
/// quantity that really moved, and the delivered goods are real transfers.
#[test]
fn t2_5b_partial_delivery_evidence_records_actual_transfers() {
    let world = fixture();
    let mut sim = Simulation::new(world, 5, config());
    let deal = propose_delivery_deal(&mut sim, 10);
    sim.world.set_flag("ordered");
    accept(&mut sim, deal);
    sim.step(Vec::new());
    assert_eq!(deal_state(&sim, deal), DealState::PerformanceDue);
    sim.world
        .actor_mut(&a(DEBTOR))
        .unwrap()
        .inventory
        .add(ItemId::new("medicine"), 5);
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::SettleDeal {
            deal,
            deliver: vec![
                rpg_domain::world::ItemStack {
                    item: ItemId::new("medicine"),
                    qty: 2,
                },
                rpg_domain::world::ItemStack {
                    item: ItemId::new("medicine"),
                    qty: 3,
                },
            ],
            money: 0,
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    assert_eq!(deal_state(&sim, deal), DealState::PerformanceDue);
    let partials: Vec<(u32, u32)> = sim
        .events
        .all()
        .iter()
        .filter_map(|e| match &e.kind {
            EventKind::DealPartialDelivery { got, expected, .. } => Some((*got, *expected)),
            _ => None,
        })
        .collect();
    assert_eq!(
        partials,
        vec![(5, 10)],
        "evidence records the actual transfer"
    );
    assert_eq!(
        sim.world
            .actor(&a(CREDITOR))
            .unwrap()
            .inventory
            .quantity(&ItemId::new("medicine")),
        5,
        "the delivered goods are real transfers"
    );
}

/// F2 (§4.4): once consideration has exchanged at `Accepted` (§4.2), the
/// debtor has no `Canceled` arrow — the pinned text scopes debtor
/// cancellation to the no-consideration case and pins no redirect, so the
/// attempt is rejected neutrally: nothing mutates, the consideration is not
/// refunded, and the creditor's unilateral cancellation remains legal.
#[test]
fn t2_5c_debtor_cancellation_after_consideration_exchanged_is_rejected() {
    // Items consideration.
    let world = fixture();
    let mut sim = Simulation::new(world, 5, config());
    sim.world
        .actor_mut(&a(DEBTOR))
        .unwrap()
        .inventory
        .add(ItemId::new("medicine"), 1);
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::ProposeDeal {
            target: a(CREDITOR),
            terms: "consideration up front".into(),
            condition: QuestPredicate::Flag("ordered".into()),
            performance: QuestPredicate::Flag("never".into()),
            condition_deadline_minute: None,
            performance_deadline_minute: None,
            give: vec![rpg_domain::world::ItemStack {
                item: ItemId::new("medicine"),
                qty: 1,
            }],
            money: 0,
            note: "take this now".into(),
        },
    );
    assert!(matches!(out, ValidationOutcome::Pending { .. }), "{out:?}");
    let deal = *sim.world.deals.keys().max().unwrap();
    sim.world.set_flag("ordered");
    accept(&mut sim, deal);
    assert_eq!(deal_state(&sim, deal), DealState::Active);
    assert_eq!(
        sim.world
            .actor(&a(CREDITOR))
            .unwrap()
            .inventory
            .quantity(&ItemId::new("medicine")),
        1,
        "consideration moved at Accepted"
    );

    // The debtor's walk-away is rejected: no refund, no violation event,
    // nothing mutates.
    let out = apply(&mut sim, &a(DEBTOR), Action::CancelDeal { deal });
    assert!(matches!(out, ValidationOutcome::Rejected(_)), "{out:?}");
    assert_eq!(deal_state(&sim, deal), DealState::Active);
    assert_eq!(
        sim.world
            .actor(&a(DEBTOR))
            .unwrap()
            .inventory
            .quantity(&ItemId::new("medicine")),
        0,
        "no refund: the pinned text pins no refund policy"
    );
    assert_eq!(
        count_events(&sim, |k| matches!(
            k,
            EventKind::DealCanceled { .. } | EventKind::DealViolated { .. }
        )),
        0,
        "the attempt produced no lifecycle event"
    );

    // The creditor may still cancel unilaterally (§4.4).
    let out = apply(&mut sim, &a(CREDITOR), Action::CancelDeal { deal });
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    assert_eq!(deal_state(&sim, deal), DealState::Canceled);

    // Money consideration: the same restriction.
    let world = fixture();
    let mut sim = Simulation::new(world, 5, config());
    sim.world.actor_mut(&a(DEBTOR)).unwrap().inventory.money = 5;
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::ProposeDeal {
            target: a(CREDITOR),
            terms: "coin up front".into(),
            condition: QuestPredicate::Flag("ordered".into()),
            performance: QuestPredicate::Flag("never".into()),
            condition_deadline_minute: None,
            performance_deadline_minute: None,
            give: vec![],
            money: 5,
            note: "five coins now".into(),
        },
    );
    assert!(matches!(out, ValidationOutcome::Pending { .. }), "{out:?}");
    let deal = *sim.world.deals.keys().max().unwrap();
    sim.world.set_flag("ordered");
    accept(&mut sim, deal);
    assert_eq!(sim.world.actor(&a(CREDITOR)).unwrap().inventory.money, 5);
    let out = apply(&mut sim, &a(DEBTOR), Action::CancelDeal { deal });
    assert!(matches!(out, ValidationOutcome::Rejected(_)), "{out:?}");
    assert_eq!(deal_state(&sim, deal), DealState::Active);
    assert_eq!(
        sim.world.actor(&a(CREDITOR)).unwrap().inventory.money,
        5,
        "no money refund"
    );
    assert_eq!(sim.world.actor(&a(DEBTOR)).unwrap().inventory.money, 0);
}

/// F2 (§4.4): before consideration — a `Proposed` deal is cancellable by the
/// debtor even when consideration is declared, because nothing has moved
/// yet; the consideration stays where it was.
#[test]
fn t2_5c_debtor_cancellation_before_consideration_is_legal() {
    let world = fixture();
    let mut sim = Simulation::new(world, 5, config());
    sim.world
        .actor_mut(&a(DEBTOR))
        .unwrap()
        .inventory
        .add(ItemId::new("medicine"), 1);
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::ProposeDeal {
            target: a(CREDITOR),
            terms: "consideration declared, not yet moved".into(),
            condition: QuestPredicate::Flag("ordered".into()),
            performance: QuestPredicate::Flag("never".into()),
            condition_deadline_minute: None,
            performance_deadline_minute: None,
            give: vec![rpg_domain::world::ItemStack {
                item: ItemId::new("medicine"),
                qty: 1,
            }],
            money: 0,
            note: "take this on acceptance".into(),
        },
    );
    assert!(matches!(out, ValidationOutcome::Pending { .. }), "{out:?}");
    let deal = *sim.world.deals.keys().max().unwrap();
    let out = apply(&mut sim, &a(DEBTOR), Action::CancelDeal { deal });
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    assert_eq!(deal_state(&sim, deal), DealState::Canceled);
    assert_eq!(
        sim.world
            .actor(&a(DEBTOR))
            .unwrap()
            .inventory
            .quantity(&ItemId::new("medicine")),
        1,
        "nothing moved: no consideration had exchanged"
    );
    assert_eq!(
        count_events(&sim, |k| matches!(k, EventKind::DealViolated { .. })),
        0
    );
}

/// F5: a self-deal has no counterparty and no real exchange (design §5.2
/// presupposes distinct parties: proposal, acceptance, consideration). It is
/// rejected and the rejection is neutral — no deal is created, nothing
/// mutates.
#[test]
fn self_deals_are_rejected_neutrally() {
    let world = fixture();
    let mut sim = Simulation::new(world, 5, config());
    sim.world
        .actor_mut(&a(DEBTOR))
        .unwrap()
        .inventory
        .add(ItemId::new("medicine"), 1);
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::ProposeDeal {
            target: a(DEBTOR),
            terms: "self deal".into(),
            condition: QuestPredicate::Flag("ordered".into()),
            performance: QuestPredicate::Flag("done".into()),
            condition_deadline_minute: None,
            performance_deadline_minute: None,
            give: vec![],
            money: 0,
            note: "nonsense contract".into(),
        },
    );
    assert!(matches!(out, ValidationOutcome::Rejected(_)), "{out:?}");
    assert!(sim.world.deals.is_empty(), "no deal was created");
    assert_eq!(
        sim.world
            .actor(&a(DEBTOR))
            .unwrap()
            .inventory
            .quantity(&ItemId::new("medicine")),
        1
    );
    let outcome = sim.outcomes.last().unwrap();
    assert!(outcome.effects.is_empty(), "no effects on rejection");
    assert!(outcome.events.is_empty(), "no events on rejection");
}

/// F3/T2.1 end-to-end: rejected *attempts* of the adopted option feed the
/// abandonment accounting exactly once per attempt (not per proposal
/// generation); the configured failure threshold (3) declares abandonment;
/// and the pipeline does not re-adopt the same abandoned intention on later
/// ticks (the failure count never resets).
#[test]
fn t2_1_abandonment_threshold_declares_failure_and_is_not_readopted() {
    let world = fixture();
    let mut sim = Simulation::new(world, 5, config());
    let out = apply(
        &mut sim,
        &a(DEBTOR),
        Action::ProposeDeal {
            target: a(CREDITOR),
            terms: "disclose on demand".into(),
            condition: QuestPredicate::Flag("ordered".into()),
            performance: QuestPredicate::HeardFact {
                actor: a(CREDITOR),
                fact: FactId::new("f_ledger"),
            },
            condition_deadline_minute: None,
            performance_deadline_minute: None,
            give: vec![],
            money: 0,
            note: "the ledger".into(),
        },
    );
    assert!(matches!(out, ValidationOutcome::Pending { .. }), "{out:?}");
    let deal = *sim.world.deals.keys().max().unwrap();
    sim.world.set_flag("ordered");
    accept(&mut sim, deal);
    sim.step(Vec::new()); // the condition is met in-pipeline → PerformanceDue
    assert_eq!(deal_state(&sim, deal), DealState::PerformanceDue);

    // The creditor is gone: the honouring attempt stays ranked-possible
    // (knowledge + authorisation + proximity are unchanged) but every
    // dispatch fails on liveness — a real, repeatable attempt failure.
    sim.world.actors.get_mut(&a(CREDITOR)).unwrap().status = ActorStatus::Dead;

    // Three attempted rejections → exactly three failures.
    for expected in 1u32..=3 {
        sim.step(Vec::new());
        let intent = sim
            .deal_intentions
            .get(&a(DEBTOR))
            .unwrap_or_else(|| panic!("attempt {expected}: intention missing"));
        assert_eq!(intent.option, "honour_disclosure");
        assert_eq!(
            intent.failed_attempts, expected,
            "exactly one failure per attempted rejection"
        );
        assert_eq!(
            count_events(&sim, |k| matches!(k, EventKind::ActionRejected { .. })),
            expected as usize
        );
    }

    // The configured threshold declares abandonment.
    let intent = sim.deal_intentions.get(&a(DEBTOR)).unwrap();
    assert!(matches!(
        rpg_behavior::consider_replan(intent, &sim.world),
        rpg_behavior::Replan::Replan(rpg_behavior::ReplanReason::Abandoned)
    ));

    // Not re-adopted: later steps produce no further attempts — the failure
    // count does not reset and the option does not change.
    let rejections_before = count_events(&sim, |k| matches!(k, EventKind::ActionRejected { .. }));
    sim.step(Vec::new());
    sim.step(Vec::new());
    assert_eq!(
        count_events(&sim, |k| matches!(k, EventKind::ActionRejected { .. })),
        rejections_before,
        "the abandoned intention is not re-attempted"
    );
    let intent = sim.deal_intentions.get(&a(DEBTOR)).unwrap();
    assert_eq!(intent.option, "honour_disclosure", "not re-adopted");
    assert_eq!(
        intent.failed_attempts, 3,
        "re-adoption would have reset the count"
    );
    assert_eq!(deal_state(&sim, deal), DealState::PerformanceDue);
    // And directly: the pipeline no longer proposes an answer for the actor.
    assert!(sim.deal_response_proposals(&BTreeSet::new()).is_empty());
}

// ---------------------------------------------------------------------------
// T2.5c — authority-backed Excuse production route (plan §4.4; I-5)
// ---------------------------------------------------------------------------

/// From `PerformanceDue`, authority/institution evidence excuses the deal.
/// The route requires (a) a due deal (the FSM rejects every other state),
/// (b) an actually-authoritative proposer, and (c) cited evidence the
/// authority itself holds; the terminal `Excused` state is exactly-once and
/// the deal's event history stays intact.
#[test]
fn t2_5c_authority_excuse_route_requires_due_state_authority_and_evidence() {
    let mut world = fixture_with_condition_stock();
    // An authority that actually knows the cited fact (provenance-bound
    // grounds, like any sanction).
    let mut captain = actor("captain", 4.0);
    captain.capabilities.can_arrest = true;
    captain.flags.insert("authority".into());
    captain.beliefs.insert(
        FactId::new("f_ledger"),
        Belief {
            fact: FactId::new("f_ledger"),
            believes_true: true,
            confidence: 0.9,
            source: Provenance::Assumed,
            acquired_tick: 0,
        },
    );
    world.actors.insert(captain.id.clone(), captain);
    let captain = a("captain");
    let mut sim = Simulation::new(world, 5, config());
    let deal = propose_disclosure_deal(&mut sim, 1);
    accept(&mut sim, deal);

    // From `Active`, an excuse is an illegal FSM arrow: rejected, and
    // nothing mutates.
    let out = apply(
        &mut sim,
        &captain,
        Action::ExcuseDeal {
            deal,
            evidence: vec![FactId::new("f_ledger")],
        },
    );
    assert!(matches!(out, ValidationOutcome::Rejected(_)), "{out:?}");
    assert_eq!(deal_state(&sim, deal), DealState::Active);

    deliver_condition(&mut sim, 1);
    sim.step(Vec::new());
    assert_eq!(deal_state(&sim, deal), DealState::PerformanceDue);

    // A party without authority cannot excuse (not permitted, not applied).
    let out = apply(
        &mut sim,
        &a(CREDITOR),
        Action::ExcuseDeal {
            deal,
            evidence: vec![FactId::new("f_ledger")],
        },
    );
    assert!(
        matches!(
            out,
            ValidationOutcome::Rejected(RejectReason::NoCapability(_))
        ),
        "{out:?}"
    );
    assert_eq!(deal_state(&sim, deal), DealState::PerformanceDue);

    // The authority must hold what it cites: an unheard fact is unknown.
    let out = apply(
        &mut sim,
        &captain,
        Action::ExcuseDeal {
            deal,
            evidence: vec![FactId::new("f_unheard")],
        },
    );
    assert!(
        matches!(
            out,
            ValidationOutcome::Rejected(RejectReason::UnknownFact(_))
        ),
        "{out:?}"
    );
    assert_eq!(deal_state(&sim, deal), DealState::PerformanceDue);

    // Authorized + evidenced: the deal reaches `Excused` exactly-once with
    // its typed event carrying the cited evidence.
    let out = apply(
        &mut sim,
        &captain,
        Action::ExcuseDeal {
            deal,
            evidence: vec![FactId::new("f_ledger")],
        },
    );
    assert!(matches!(out, ValidationOutcome::Accepted), "{out:?}");
    assert_eq!(deal_state(&sim, deal), DealState::Excused);
    assert_eq!(
        count_events(&sim, |k| matches!(k, EventKind::DealExcused { .. })),
        1,
        "the typed excuse event is emitted exactly once"
    );

    // Exactly-once: a second terminal transition on the same id is rejected.
    let out = apply(
        &mut sim,
        &captain,
        Action::ExcuseDeal {
            deal,
            evidence: vec![FactId::new("f_ledger")],
        },
    );
    assert!(matches!(out, ValidationOutcome::Rejected(_)), "{out:?}");
    assert_eq!(deal_state(&sim, deal), DealState::Excused);
    // The event history stays intact for audit.
    assert!(count_events(&sim, |k| matches!(k, EventKind::DealProposed { .. })) == 1);
    assert!(count_events(&sim, |k| matches!(k, EventKind::DealConditionMet { .. })) == 1);
}
