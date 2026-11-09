//! Kani harnesses over production symbols (I-8 / T6.1-T6.8). Each harness's
//! declared bounds live in `crates/proof-kernels/limits.toml`; the CI proof
//! job (and `scripts/check-proof-audit.sh`) enforce the one-to-one
//! harness/limits mapping. Proof assumptions are stated inline and mirrored
//! in `AUDIT.md` ("bounded proof vs runtime evidence").
//!
//! Scaffold note: CBMC symex makes one `BTreeMap` insertion cost tens of
//! seconds and loops that *contain* map operations multiply that cost, so
//! the scaffolds here are deliberately minimal (two-actor world, unrolled
//! op sequences). The harnesses still call the exact production symbols;
//! the concrete authored town is exercised end-to-end by the unit,
//! scenario and property-test layers instead.

use kani::*;

use rpg_behavior::epistemics::{
    accept_assertion, acceptance_probability, aggregate_about, aggregate_from_groups, group_weight,
    prior_strength, AssertionDraft, GroupData, RecipientAppraisal,
};
use rpg_domain::action::{Action, ActionProposal, ProposalSource};
use rpg_domain::behavior_types::AcquisitionChannel;
use rpg_domain::deal::{transition, Deal, DealEvent, DealState, Party, ViolationReason};
use rpg_domain::events::EventLog;
use rpg_domain::geom::Vec2;
use rpg_domain::ids::{ActorId, DealId, EventId, FactId, ItemId, LocationId};
use rpg_domain::rng::DeterministicRng;
use rpg_domain::world::{
    Actor, ActorKind, ActorStatus, Capabilities, Inventory, ItemStack, MysteryMode, Personality,
    StoryState, World,
};
use rpg_sim::{split_money_transfer, SimConfig, SimOp, Simulation};

// ---------------------------------------------------------------------------
// Scaffolds
// ---------------------------------------------------------------------------

/// Minimal concrete actor (28 authored fields, everything empty/neutral).
fn minimal_actor(id: &str) -> Actor {
    Actor {
        id: ActorId::new(id),
        name: id.into(),
        kind: ActorKind::Npc,
        status: ActorStatus::Alive,
        location: LocationId::new("loc"),
        pos: Vec2 { x: 0.0, y: 0.0 },
        facing: 0.0,
        personality: Personality::default(),
        values: Vec::new(),
        biography: String::new(),
        private_biography: String::new(),
        affiliations: Vec::new(),
        capabilities: Capabilities::default(),
        inventory: Inventory::default(),
        goals: Vec::new(),
        schedule: Vec::new(),
        relationships: std::collections::BTreeMap::new(),
        beliefs: std::collections::BTreeMap::new(),
        inspected: std::collections::BTreeSet::new(),
        flags: std::collections::BTreeSet::new(),
        secret_allegiance: None,
        in_private_conversation_with: None,
        bargain: None,
        needs: rpg_domain::needs::Needs::default(),
        emotions: rpg_domain::emotions::Emotions::default(),
        rel_memory: std::collections::BTreeMap::new(),
    }
}

/// Minimal concrete world: two actors (observer + payee/speaker) in one
/// location, empty canonical state, default epistemics. Every harness that
/// exercises the epistemics/replay kernels runs against this scaffold - the
/// production symbols are identical to the dispatch path, only the concrete
/// world content is reduced to the minimum CBMC can verify.
fn minimal_world() -> World {
    let mut actors = std::collections::BTreeMap::new();
    let mut observer = minimal_actor("i8_observer");
    observer.inventory.money = 250;
    actors.insert(observer.id.clone(), observer);
    actors.insert(ActorId::new("i8_speaker"), minimal_actor("i8_speaker"));
    World {
        tick: 0,
        day: 0,
        minute_of_day: 0,
        locations: std::collections::BTreeMap::new(),
        actors,
        items: std::collections::BTreeMap::new(),
        factions: std::collections::BTreeMap::new(),
        facts: std::collections::BTreeMap::new(),
        nav: rpg_domain::world::NavGrid::new(4, 4),
        story: StoryState {
            mode: MysteryMode::FixedAnswer,
            anchors: Vec::new(),
            checkpoints: Vec::new(),
            fixed_answer: std::collections::BTreeMap::new(),
            committed_containers: std::collections::BTreeMap::new(),
        },
        commitments: std::collections::BTreeMap::new(),
        offers: std::collections::BTreeMap::new(),
        deals: std::collections::BTreeMap::new(),
        rumors: Vec::new(),
        damage: std::collections::BTreeMap::new(),
        damage_facts: std::collections::BTreeMap::new(),
        location_items: std::collections::BTreeMap::new(),
        flags: std::collections::BTreeSet::new(),
        ledger: std::collections::BTreeMap::new(),
        next_commitment_id: 0,
        next_offer_id: 0,
        next_deal_id: 0,
        next_rumor_id: 0,
        next_goal_id: 0,
        epistemics: rpg_domain::behavior_types::Epistemics::default(),
        lore: rpg_domain::world::WorldLore::default(),
    }
}

/// Bounded symbolic acceptance inputs: relationship/appraisal components in
/// their documented -1..1 / 0..1 ranges, NaN excluded (bounded f32 inputs,
/// P5). Mirrored in limits.toml.
fn bounded_appraisal() -> RecipientAppraisal {
    let trust: f32 = any();
    assume((-1.0..=1.0).contains(&trust) && !trust.is_nan());
    let suspicion: f32 = any();
    assume((-1.0..=1.0).contains(&suspicion) && !suspicion.is_nan());
    let fear: f32 = any();
    assume((-1.0..=1.0).contains(&fear) && !fear.is_nan());
    let assertiveness: f32 = any();
    assume((0.0..=1.0).contains(&assertiveness) && !assertiveness.is_nan());
    RecipientAppraisal {
        trust,
        suspicion,
        fear,
        assertiveness,
    }
}

fn bounded_confidence() -> f32 {
    let c: f32 = any();
    assume((0.0..=1.0).contains(&c) && !c.is_nan());
    c
}

// T6.1 - transfer/accounting conservation -----------------------------------

/// T6.1: the production money-transfer arithmetic kernel conserves total
/// money over the full `i64` space and rejects (never wraps) overflow.
/// No input bounds beyond the type: `kani::any::<i64>()` covers all values,
/// so the proof is complete for money. Production call sites (the only
/// conserved money movements): `Simulation::transfer_money` and
/// `Simulation::do_pay` (AUDIT rows T6.1a). No loops - unwind 1 is exact.
#[kani::proof]
#[kani::unwind(1)]
fn i8_t61_money_transfer_conservation() {
    let from_money: i64 = any();
    let to_money: i64 = any();
    let amount: i64 = any();
    match split_money_transfer(from_money, to_money, amount) {
        Ok((debited, credited)) => {
            // Conservation: the debit leaves `from` exactly and lands exactly
            // in `to`; verified in 128-bit space so the check itself cannot
            // overflow.
            assert_eq!(
                debited as i128 + credited as i128,
                from_money as i128 + to_money as i128
            );
            // The debit and credit are the exact checked results.
            assert_eq!(from_money as i128 - amount as i128, debited as i128);
            assert_eq!(to_money as i128 + amount as i128, credited as i128);
        }
        Err(err) => {
            // Overflow is rejected, never wrapped: the rejected side would
            // genuinely not fit in `i64`.
            match err {
                rpg_sim::MoneyTransferOverflow::Debit => {
                    let debit = from_money as i128 - amount as i128;
                    assert!(
                        debit < i64::MIN as i128 || debit > i64::MAX as i128,
                        "Debit rejected although it fits"
                    );
                }
                rpg_sim::MoneyTransferOverflow::Credit => {
                    let credit = to_money as i128 + amount as i128;
                    assert!(
                        credit < i64::MIN as i128 || credit > i64::MAX as i128,
                        "Credit rejected although it fits"
                    );
                }
            }
        }
    }
}

/// T6.1 (items): the production inventory kernels the sim's `move_item`
/// dispatch calls (`Inventory::remove` then `Inventory::add`) conserve the
/// transferred item and only ever move what was actually removed.
///
/// Declared input bounds (limits.toml `i8_t61_item_transfer_conservation`):
/// single stack per side, stock/qty <= 4 - covering the stack-fall branch
/// (`qty` > stock), the exact branch and the empty branch of `remove`,
/// while keeping `add` inside `u32` (the bounds themselves guarantee it).
/// The only loops are the `Vec<ItemStack>` scans with one element; unwind
/// 16 covers them plus the string byte-compare internals.
#[kani::proof]
#[kani::unwind(16)]
fn i8_t61_item_transfer_conservation() {
    const MAX_STOCK: u32 = 4;
    let stock_from: u32 = any();
    assume(stock_from <= MAX_STOCK);
    let stock_to: u32 = any();
    assume(stock_to <= MAX_STOCK);
    let qty: u32 = any();
    assume(qty <= MAX_STOCK);

    let item = ItemId::new("herb");
    let mut from = Inventory {
        items: vec![ItemStack {
            item: item.clone(),
            qty: stock_from,
        }],
        money: 0,
    };
    let mut to = Inventory {
        items: vec![ItemStack {
            item: item.clone(),
            qty: stock_to,
        }],
        money: 0,
    };
    let total_before = stock_from as u64 + stock_to as u64;

    // Exactly the `move_item` production sequence (sim dispatch):
    let removed = from.remove(&item, qty);
    assert!(removed <= qty);
    assert!(removed <= stock_from);
    to.add(item.clone(), removed);

    // Conservation + credit equals the actually-removed count (settlement
    // evidence must record the actual count).
    assert_eq!(to.quantity(&item) as u64, stock_to as u64 + removed as u64);
    assert_eq!(from.quantity(&item) as u64, (stock_from - removed) as u64);
    assert_eq!(
        from.quantity(&item) as u64 + to.quantity(&item) as u64,
        total_before
    );
}

// T6.2 - claim-acceptance boundary -------------------------------------------

/// T6.2: an assertion cannot write world truth through the production
/// `accept_assertion`. Acceptance writes exactly: one ledger item (the
/// asserted polarity), the typed `EvidenceRecorded` event (plus the typed
/// `ClaimShared` delivery event on the told path), and the rolled outcome -
/// nothing else. Facts, items, inventories, deals, tick and the observer's
/// beliefs are untouched (the belief *application* from the rolled outcome
/// is the sim layer's job, downstream of this kernel).
///
/// Bound scope (CBMC budget - documented in AUDIT.md): one concrete
/// validated delivery (told channel, positive polarity, neutral
/// relationship profile) on the minimal two-actor world. This verifies the
/// real production code path end-to-end - the boundary property is
/// structural (what the path writes) - while the *input space* generality
/// is carried by the numeric-core harnesses (T6.8 P1/P2/P5, symbolic), the
/// property tests, and the unit/integration layers; the symbolic-input
/// boundary harness is the disclosed residual (CBMC symex of symbolic
/// inputs through the acceptance heap path exceeds the machine/runner
/// budget).
#[kani::proof]
#[kani::unwind(30)]
fn i8_t62_claim_acceptance_boundary() {
    let mut world = minimal_world();
    let observer = ActorId::new("i8_observer");
    let speaker = ActorId::new("i8_speaker");
    // Truth snapshot (concrete).
    let money_observer = world.actor(&observer).unwrap().inventory.money;
    let money_speaker = world.actor(&speaker).unwrap().inventory.money;
    let items_observer = world.actor(&observer).unwrap().inventory.items.clone();
    let facts_before = world.facts.len();
    let items_world_before = world.items.len();
    let deals_before = world.deals.len();
    let beliefs_before = world.actor(&observer).unwrap().beliefs.len();
    let (tick, day, minute) = (world.tick, world.day, world.minute_of_day);

    let asserted_polarity = true;
    let told = true;
    // Concrete validated-delivery profile (bound scope - see the harness
    // doc): the float behavior is proven on the numeric core by T6.8
    // P1/P2/P5.
    let assertive = true;
    let deceptive = false;
    let reliability: f32 = 0.5;
    let appraisal = RecipientAppraisal {
        trust: 0.0,
        suspicion: 0.0,
        fear: 0.0,
        assertiveness: 0.5,
    };
    let speaker_confidence: f32 = 0.5;

    let fact = FactId::new("fact_i8_t62_probe");
    let draft = AssertionDraft {
        speaker: Some(speaker.clone()),
        proposition: fact.clone(),
        asserted_polarity,
        channel: if told {
            AcquisitionChannel::Told
        } else {
            AcquisitionChannel::Witnessed
        },
        origin_event: Some(EventId(4)),
        reliability,
        retold_chain: None,
        deceptive,
    };
    let mut events = EventLog::default();
    let mut rng = DeterministicRng::new(0xC0FFEE);
    let acceptance = accept_assertion(
        &mut world,
        &mut events,
        tick,
        minute,
        &observer,
        draft,
        appraisal,
        speaker_confidence,
        assertive,
        &mut rng,
    );

    // Ledger: exactly one item, with the asserted polarity, and the kernel-
    // computed perceived reliability (never the raw draft value).
    let ledger = world.epistemics.evidence_about(&observer, &fact);
    assert!(ledger.len() == 1);
    assert!(ledger[0].asserted_polarity == asserted_polarity);
    assert!(acceptance.evidence.proposition_id == fact);
    assert!(acceptance.chain.raw() == ledger[0].source_chain_id.raw());

    // WORLD TRUTH UNTOUCHED (the boundary property).
    assert!(world.actor(&observer).unwrap().inventory.money == money_observer);
    assert!(world.actor(&speaker).unwrap().inventory.money == money_speaker);
    assert!(world.actor(&observer).unwrap().inventory.items == items_observer);
    assert!(world.facts.len() == facts_before);
    assert!(world.items.len() == items_world_before);
    assert!(world.deals.len() == deals_before);
    // No belief write inside the kernel: the rolled outcome is returned, not
    // applied.
    assert!(world.actor(&observer).unwrap().beliefs.len() == beliefs_before);
    assert!((world.tick, world.day, world.minute_of_day) == (tick, day, minute));

    // Events: exactly the typed delivery + record events, nothing else.
    if told {
        assert!(events.len() == 2);
    } else {
        assert!(events.len() == 1);
    }
}

// T6.3 - deal lifecycle -------------------------------------------------------

/// The 11 pinned FSM states, indexed for symbolic selection (bounded
/// state/event space). All pure - no collections - so the unwind is the
/// driven-sequence bound, not a map bound.
const STATES: [DealState; 11] = [
    DealState::Proposed,
    DealState::Accepted,
    DealState::Active,
    DealState::ConditionMet,
    DealState::PerformanceDue,
    DealState::Performed,
    DealState::Violated,
    DealState::Excused,
    DealState::Rejected,
    DealState::Canceled,
    DealState::Renegotiated,
];

fn symbolic_event(choice: u8) -> DealEvent {
    match choice {
        0 => DealEvent::Accept,
        1 => DealEvent::Reject,
        2 => DealEvent::Activate,
        3 => DealEvent::ConditionMet,
        4 => DealEvent::ObligationDue,
        5 => DealEvent::Renegotiate,
        6 => DealEvent::Reactivate,
        7 => DealEvent::Perform,
        8 => DealEvent::Violate {
            reason: ViolationReason::DeadlineLapsed,
        },
        9 => DealEvent::Violate {
            reason: ViolationReason::DebtorReneged,
        },
        10 => DealEvent::Cancel { by: Party::Mutual },
        11 => DealEvent::Cancel {
            by: Party::Creditor,
        },
        _ => DealEvent::Cancel { by: Party::Debtor },
    }
}

/// T6.3: the production deal transition kernel rejects every arrow outside
/// the pinned set, never allows a terminal transition twice (exactly-once),
/// keeps `ConditionMet` distinct from `Performed`, and respects the 4.4
/// cancellation authorization matrix. `Ok` results are checked against the
/// pinned postconditions (an arrow is legal iff its pinned precondition
/// held); terminal rejections are total.
#[kani::proof]
#[kani::unwind(16)]
fn i8_t63_deal_lifecycle_invariants() {
    let state_choice: u8 = any();
    assume((state_choice as usize) < STATES.len());
    let state = STATES[state_choice as usize];
    let event_choice: u8 = any();
    assume(event_choice < 13);
    let event = symbolic_event(event_choice);
    let exchanged: bool = any();

    let result = transition(state, event, exchanged);

    // Exactly-once: a terminal state consumes the deal; every further event
    // is rejected.
    if state.is_terminal() {
        assert!(result.is_err());
    }
    // ConditionMet is a distinct transient state: the obligation must become
    // due before it can perform, and it is non-cancellable. The only legal
    // arrow out is ObligationDue.
    if state == DealState::ConditionMet && !matches!(event, DealEvent::ObligationDue) {
        assert!(result.is_err());
    }
    if let Ok(next) = result {
        // Pinned postconditions: an accepted arrow is legal iff its pinned
        // precondition held (the 4.1 arrow set + 4.4 cancellation matrix).
        let legal = match next {
            DealState::Accepted => state == DealState::Proposed && event == DealEvent::Accept,
            DealState::Rejected => state == DealState::Proposed && event == DealEvent::Reject,
            DealState::Active => {
                (state == DealState::Accepted && event == DealEvent::Activate)
                    || (state == DealState::Renegotiated && event == DealEvent::Reactivate)
            }
            DealState::ConditionMet => {
                state == DealState::Active && event == DealEvent::ConditionMet
            }
            DealState::PerformanceDue => {
                state == DealState::ConditionMet && event == DealEvent::ObligationDue
            }
            DealState::Performed => {
                state == DealState::PerformanceDue && event == DealEvent::Perform
            }
            DealState::Violated => {
                state == DealState::PerformanceDue
                    && matches!(
                        event,
                        DealEvent::Violate { .. } | DealEvent::Cancel { by: Party::Debtor }
                    )
            }
            DealState::Excused => state == DealState::PerformanceDue && event == DealEvent::Excuse,
            DealState::Renegotiated => {
                state == DealState::PerformanceDue && event == DealEvent::Renegotiate
            }
            DealState::Canceled => {
                state == DealState::Proposed
                    || ((state == DealState::Accepted || state == DealState::Active)
                        && !(matches!(event, DealEvent::Cancel { by: Party::Debtor }) && exchanged))
                    || (state == DealState::PerformanceDue
                        && matches!(
                            event,
                            DealEvent::Cancel {
                                by: Party::Mutual | Party::Creditor
                            }
                        ))
            }
            DealState::Proposed => false,
        };
        assert!(legal, "accepted an arrow outside the pinned set");
        // Terminal settlement is absorbing: the next terminal-producing state
        // is only reachable from the pinned pre-state, never re-entered from
        // a terminal one (checked by the `is_terminal` branch above).
        assert!(!state.is_terminal());
    }

    // Exactly-once over a bounded driven sequence: once `advance` reaches a
    // terminal state, every further advance is rejected - whatever the
    // events were.
    let mut deal = Deal::proposed(
        DealId(1),
        ActorId::new("i8_debtor"),
        ActorId::new("i8_creditor"),
        "i8",
        rpg_domain::action::QuestPredicate::Flag("condition".into()),
        rpg_domain::action::QuestPredicate::Flag("performance".into()),
        None,
        None,
        30,
        Vec::new(),
        0,
        Vec::new(),
        0,
    );
    let e1_choice: u8 = any();
    assume(e1_choice < 13);
    let e1 = symbolic_event(e1_choice);
    let _ = deal.advance(e1, 1);
    if deal.is_terminal() {
        let e2_choice: u8 = any();
        assume(e2_choice < 13);
        let e2 = symbolic_event(e2_choice);
        assert!(deal.advance(e2, 2).is_err());
    }

    // The pinned happy path still performs, and a second settlement is
    // rejected (concrete anchor for the symbolic proof above).
    let mut d2 = Deal::proposed(
        DealId(2),
        ActorId::new("i8_debtor"),
        ActorId::new("i8_creditor"),
        "i8",
        rpg_domain::action::QuestPredicate::Flag("condition".into()),
        rpg_domain::action::QuestPredicate::Flag("performance".into()),
        None,
        None,
        30,
        Vec::new(),
        0,
        Vec::new(),
        0,
    );
    assert!(d2.advance(DealEvent::Accept, 1).is_ok());
    assert!(d2.advance(DealEvent::Activate, 2).is_ok());
    assert!(d2.advance(DealEvent::ConditionMet, 3).is_ok());
    assert!(d2.advance(DealEvent::ObligationDue, 4).is_ok());
    assert_eq!(
        d2.advance(DealEvent::Perform, 5).unwrap(),
        DealState::Performed
    );
    assert!(d2.advance(DealEvent::Perform, 6).is_err());
}

// T6.4 - authoritative replay determinism ------------------------------------

/// T6.4: the production replay path (`Simulation::replay_ops`) reproduces
/// the canonical state of a recorded run over bounded op sequences: the
/// full event log (length + production checksum + every event via full
/// JSON), the full epistemics JSON, and the canonical per-actor state
/// (money, inventories, deals, clock) - never summaries.
///
/// Declared bounds (limits.toml `i8_t64_replay_determinism`): exactly 2
/// recorded `Apply` ops, payment amounts in 1..=3 (validated custody) over
/// the minimal two-actor world. The op application is unrolled in the
/// harness (a loop wrapping `apply` would multiply the verification cost of
/// the map operations inside the dispatch); the production `replay_ops`
/// loop over the recorded ops is itself exercised - that cost is inherent
/// to the replay under proof. Full world-JSON equality over arbitrary
/// sequences - including `Step` (autonomy) ops - is the T6.5b property
/// (`crates/eval/src/properties/replay.rs`, runtime evidence); the split is
/// documented in `AUDIT.md`.
#[kani::proof]
#[kani::unwind(2000)]
fn i8_t64_replay_determinism() {
    let amount1: i64 = any();
    assume((1..=3).contains(&amount1));
    let amount2: i64 = any();
    assume((1..=3).contains(&amount2));

    let initial = minimal_world();
    let you = ActorId::new("i8_observer");
    let grocer = ActorId::new("i8_speaker");
    let money_before: i64 = initial.actors.values().map(|a| a.inventory.money).sum();

    let mk_pay = |amount: i64, id: u64| {
        SimOp::Apply(Box::new(ActionProposal {
            actor: you.clone(),
            action: Action::Pay {
                target: grocer.clone(),
                amount,
            },
            source: ProposalSource::Scripted,
            rationale: format!("i8 replay op {id}"),
        }))
    };
    let op1 = mk_pay(amount1, 1);
    let op2 = mk_pay(amount2, 2);
    let ops = vec![op1.clone(), op2.clone()];

    // Recorded run: fresh session, apply the ops directly (unrolled).
    let mut recorded = Simulation::new(initial.clone(), 2026, SimConfig::default());
    match &op1 {
        SimOp::Apply(p) => {
            let _ = recorded.apply(p);
        }
        _ => unreachable!(),
    }
    match &op2 {
        SimOp::Apply(p) => {
            let _ = recorded.apply(p);
        }
        _ => unreachable!(),
    }
    // Authoritative replay from the same initial world.
    let replayed = Simulation::replay_ops(initial, 2026, SimConfig::default(), &ops);

    // Full event log: length, production checksum and every event (full
    // serialized log, not a summary).
    assert!(recorded.events.len() == replayed.events.len());
    assert!(recorded.events.checksum() == replayed.events.checksum());
    assert!(
        serde_json::to_string(&recorded.events).unwrap()
            == serde_json::to_string(&replayed.events).unwrap()
    );
    // Full epistemics JSON - never a summary.
    assert!(
        serde_json::to_string(&recorded.world.epistemics).unwrap()
            == serde_json::to_string(&replayed.world.epistemics).unwrap()
    );
    // Canonical world state on the exercised path: the replay reproduces
    // each actor's money and inventory, the clock, and the deals map
    // exactly (straight-line: no loop over the map, whose body would
    // multiply the CBMC cost).
    assert!(recorded.world.tick == replayed.world.tick);
    assert!(recorded.world.day == replayed.world.day);
    assert!(recorded.world.minute_of_day == replayed.world.minute_of_day);
    assert!(
        serde_json::to_string(&recorded.world.deals).unwrap()
            == serde_json::to_string(&replayed.world.deals).unwrap()
    );
    for who in [&you, &grocer] {
        let a = recorded.world.actor(who).expect("same actor set");
        let b = replayed.world.actor(who).expect("same actor set");
        assert!(a.inventory.money == b.inventory.money);
        assert!(
            serde_json::to_string(&a.inventory).unwrap()
                == serde_json::to_string(&b.inventory).unwrap()
        );
    }
    // Conservation on the exercised path: a validated payment moves money,
    // it never mints or burns it.
    assert_eq!(
        recorded
            .world
            .actors
            .values()
            .map(|a| a.inventory.money)
            .sum::<i64>(),
        money_before
    );
}

// T6.8 - numeric arbitration (P1-P5) -----------------------------------------

/// T6.8 P2/P5 (+ P1 on the empty ledger) over the production numeric core.
///
/// - P1 (empty-ledger case): the acceptance probability is in [0.05, 0.97]
///   for every bounded relationship input. The non-empty-ledger case (the
///   bounded prior term active) is the separate harness
///   `i8_t68_acceptance_p1_nonempty_ledger`.
/// - P2: an empty ledger reproduces the legacy relationship-only formula
///   exactly (the prior term contributes exactly 0.0) - checked against the
///   production `acceptance_probability` with the strength of the empty
///   production aggregate (`aggregate_from_groups(&[])` through
///   `prior_strength`), bit-for-bit in the same operation order.
/// - P5: all clamps hold - perceived reliability in [0.05, 0.98], prior
///   strength in [0, 1], aggregate confidence in [0, 1], no NaN anywhere.
///
/// The acceptance-probability core is the extracted pure production
/// function (`accept_assertion` calls it - single source of truth); these
/// numeric properties are pure float arithmetic over symbolic bounded
/// inputs, so the harness is map-light by construction. The full accept
/// path (record + events + roll) is exercised by T6.2 over the production
/// symbols; full-accept P1/P2 equality is additionally covered by the
/// T6.8a property tests at runtime scale.
#[kani::proof]
#[kani::unwind(16)]
fn i8_t68_acceptance_numeric_p1_p2_p5() {
    let appraisal = bounded_appraisal();
    let speaker_confidence = bounded_confidence();

    // P2: empty production aggregate -> strength exactly 0 -> the
    // probability reproduces the legacy formula exactly.
    let empty = aggregate_from_groups(&[]);
    let strength = prior_strength(&empty);
    assert!(empty.confidence == 0.0 && empty.independence_groups == 0);
    assert!(strength == 0.0);
    let (p, rel) = acceptance_probability(&appraisal, speaker_confidence, strength);
    // The legacy formula in the same operation order (bit-exact P2).
    let trust_01 = (1.0 + appraisal.trust) * 0.5;
    let suspicion_01 = (1.0 + appraisal.suspicion) * 0.5;
    let legacy = 0.25
        + 0.45 * trust_01
        + 0.25 * appraisal.assertiveness.clamp(0.0, 1.0)
        + 0.2 * speaker_confidence.clamp(0.0, 1.0)
        + 0.2 * strength
        - 0.35 * suspicion_01
        - 0.15 * appraisal.fear;
    assert!(p == legacy.clamp(0.05, 0.97), "P2 legacy-formula deviation");

    // P1 + P5: clamps and no-NaN over the bounded input space.
    assert!((0.05..=0.97).contains(&p) && !p.is_nan());
    assert!((0.05..=0.98).contains(&rel) && !rel.is_nan());
    assert!((0.0..=1.0).contains(&strength) && !strength.is_nan());
    assert!((0.0..=1.0).contains(&empty.confidence) && !empty.confidence.is_nan());
}

/// T6.8 P1 (non-empty ledger): with a pre-existing correlation group the
/// bounded prior term `0.2 * strength` is active inside the production
/// formula, and the acceptance probability still stays inside the clamp.
/// The "ledger" is one symbolic group through the production fold
/// (`aggregate_from_groups` -> `prior_strength`): the numeric content of a
/// single-item ledger without the map scaffold (loops wrapping map
/// operations multiply the CBMC symex cost; the map path is exercised by
/// T6.2 and the property tests).
#[kani::proof]
#[kani::unwind(16)]
fn i8_t68_acceptance_p1_nonempty_ledger() {
    let appraisal = bounded_appraisal();
    let speaker_confidence = bounded_confidence();
    let seeded_rel: f32 = any();
    assume((0.0..=1.0).contains(&seeded_rel) && !seeded_rel.is_nan());

    // One seeded item: one group, one chain, true polarity (production
    // fold, symbolic).
    let agg = aggregate_from_groups(&[GroupData {
        max_reliability: seeded_rel.clamp(0.0, 1.0),
        distinct_chains: 1,
        true_votes: 1,
    }]);
    let strength = prior_strength(&agg);
    assert!((0.0..=1.0).contains(&strength) && !strength.is_nan());
    assert!(agg.true_groups == 1 && agg.false_groups == 0);

    let (p, rel) = acceptance_probability(&appraisal, speaker_confidence, strength);
    assert!((0.05..=0.97).contains(&p) && !p.is_nan());
    assert!((0.05..=0.98).contains(&rel) && !rel.is_nan());
    assert!((0.0..=1.0).contains(&agg.confidence) && !agg.confidence.is_nan());
}
/// T6.8 P3/P4 over the production numeric-aggregation core.
///
/// - P3: retellings from one correlation group contribute at most
///   `max_reliability + 0.24` total weight - for **any** k distinct chains
///   up to the bound (k symbolic!) - so five retellings are never five
///   confirmations; and one group with the capped weight folds (through the
///   production `aggregate_from_groups`) to a confidence bounded by the
///   closed form cap/(cap+0.5), exactly one group.
/// - P4: `contradicted` iff both polarities weigh >= 0.25.
///
/// These are the numeric-arbitration properties, so the harness calls the
/// extracted pure production fold (`aggregate_from_groups`/`group_weight` -
/// `aggregate_about` calls both; the map-side grouping is audited at the
/// `aggregate_about` call sites). The ledger-write path
/// (`record_evidence` inside `accept_assertion`) is exercised end-to-end by
/// T6.2/T6.8a; full one-group aggregation over a real ledger is runtime
/// evidence (property tests + unit tests) - labeled honestly in AUDIT.md.
///
/// Input bounds: k <= 5 distinct chains, per-group reliability 0.05..=1.0,
/// NaN excluded. Pure arithmetic + one 1-2 element slice: no maps.
#[kani::proof]
#[kani::unwind(16)]
fn i8_t68_corroboration_p3_p4() {
    // P3: the corroboration cap, for any k <= 5 distinct chains.
    let r: f32 = any();
    assume((0.05..=1.0).contains(&r) && !r.is_nan());
    let k: u32 = any();
    assume((1..=5).contains(&k));
    let weight = group_weight(r, k as usize);
    // The cap itself: weight <= max_reliability + 0.24 for any k, the bonus
    // never negative, no NaN.
    assert!(weight <= r + 0.24, "corroboration cap violated");
    assert!(weight >= r);
    assert!(!weight.is_nan());
    // Through the production fold: one group with the capped weight and k
    // true retellings is exactly one group - not k confirmations - and its
    // confidence stays under the closed form of the capped weight.
    let agg = aggregate_from_groups(&[GroupData {
        max_reliability: r,
        distinct_chains: k,
        true_votes: k,
    }]);
    let capped = r + 0.24;
    let bound = capped / (capped + 0.5) + 1e-5;
    assert!(agg.confidence <= bound, "corroboration bound violated");
    assert!(agg.confidence >= 0.0 && !agg.confidence.is_nan());
    assert!(agg.independence_groups == 1);
    assert!(agg.true_groups == 1 && agg.false_groups == 0);
    assert!(!agg.contradicted);

    // P4: two independent groups of opposite polarity - `contradicted` iff
    // both sides weigh >= 0.25 (single item per group: weight = reliability).
    let r_true: f32 = any();
    assume((0.05..=1.0).contains(&r_true) && !r_true.is_nan());
    let r_false: f32 = any();
    assume((0.05..=1.0).contains(&r_false) && !r_false.is_nan());
    let agg2 = aggregate_from_groups(&[
        GroupData {
            max_reliability: r_true,
            distinct_chains: 1,
            true_votes: 1,
        },
        GroupData {
            max_reliability: r_false,
            distinct_chains: 1,
            true_votes: 0,
        },
    ]);
    assert!(agg2.true_groups == 1 && agg2.false_groups == 1);
    assert!(agg2.contradicted == (r_true >= 0.25 && r_false >= 0.25));
    assert!(agg2.independence_groups == 2);
}
