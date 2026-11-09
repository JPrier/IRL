//! Behavior-engine I6b (plan I-6, world-integration slice) acceptance tests,
//! exercised through the *real* `Simulation::step` production pipeline:
//!
//! * **T4.2 scarcity conservation** — money/items conserved modulo recorded
//!   sources/sinks (the typed `WagePaid` institutional disbursements) over a
//!   randomized soak through real dispatch;
//! * **I6b item 4** — wealth-driven work candidates: the wealth need is
//!   satisfied *only* through real validated resource gains (a faction wage
//!   for performed labor), never an artificial boost;
//! * **T4.3 faction institution behavior** — group decisions derive from
//!   member reports (member evidence ledgers, no truth access); a faction
//!   misbelieves when member reports are wrong;
//! * **I6b item 5** — one-shot completion hardening: outcomes match by the
//!   plan's full parameters (never actor+kind), a one-shot never re-executes
//!   when its outcome scrolls past the bounded retirement window, and
//!   same-step signals are spent on a real decision at retirement;
//! * **T4.6 offscreen catch-up** — bounded, deterministic, no duplicated
//!   outcomes, save/load + replay continuity;
//! * **I6b item 3** — autonomous retellings through the real production
//!   route carry an explicitly selected (validated) provenance chain and
//!   respect disclosure permissions;
//! * **I6b item 7** — sanction-evidence confidentiality (§9.1.12
//!   noninterference at the sanction seam);
//! * **M1** — every canonical mutation in a production soak is either
//!   event-sourced or an explicitly classified background exception.

use rpg_behavior::record_provenance_evidence;
use rpg_domain::action::{Action, ActionProposal, ProposalSource};
use rpg_domain::deal::DealState;
use rpg_domain::events::EventKind;
use rpg_domain::geom::{Rect, Vec2};
use rpg_domain::ids::*;
use rpg_domain::needs::NeedKind;
use rpg_domain::world::{
    Actor, ActorKind, ActorStatus, Belief, Capabilities, Fact, FactCategory, Item, Location,
    LocationKind, NavGrid, Provenance, Sensitivity, StoryState, World,
};
use rpg_sim::{SimConfig, Simulation, CATCH_UP_MAX_TICKS};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

const SHOP: &str = "shop";
const HALL: &str = "hall";
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
        personality: Default::default(),
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
        inspected: Default::default(),
        flags: Default::default(),
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

/// Two connected rooms plus a third; NPCs at room centers.
fn world_fixture() -> World {
    let nav = NavGrid::new(40, 12);
    let mut locations = BTreeMap::new();
    for (id, x) in [(SHOP, 0.0), (HALL, 14.0), (DOCKS, 28.0)] {
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
    actors.insert(a("m1").clone(), actor("m1", SHOP, 5.0));
    actors.insert(a("m2").clone(), actor("m2", SHOP, 6.0));
    actors.insert(a("out").clone(), actor("out", DOCKS, 33.0));
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
        flags: Default::default(),
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

fn background_config() -> SimConfig {
    SimConfig {
        enable_background: true,
        enable_director: false,
        enable_goal_autonomy: false,
        ..SimConfig::default()
    }
}

#[allow(dead_code)]
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
        "fixture action must apply: {outcome:?}"
    );
}

fn add_fact(w: &mut World, id: &str, sensitivity: Sensitivity, holders: Vec<&str>) {
    w.facts.insert(
        FactId::new(id),
        Fact {
            id: FactId::new(id),
            statement: format!("fact {id}"),
            truth: true,
            sensitivity,
            category: FactCategory::Observation,
            prerequisites: vec![],
            eligible_holders: holders.iter().map(|h| a(h)).collect(),
            eligible_containers: vec![],
            committed: false,
            placed_at: None,
        },
    );
}

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

/// Ledger evidence for `holder` under a fresh witnessed chain (the real
/// adapter the production observation path uses).
fn witness_evidence(
    w: &mut World,
    events: &mut rpg_domain::events::EventLog,
    tick: u64,
    minute: u32,
    holder: &str,
    fact: &str,
    origin_event: EventId,
) -> rpg_domain::behavior_types::SourceChainId {
    record_provenance_evidence(
        w,
        events,
        tick,
        minute,
        &a(holder),
        &FactId::new(fact),
        true,
        &Provenance::Witnessed {
            event: origin_event,
        },
    )
}

fn count_events(sim: &Simulation, pred: impl Fn(&EventKind) -> bool) -> usize {
    sim.events.all().iter().filter(|e| pred(&e.kind)).count()
}

/// Total money across all actors — the conservation quantity.
fn total_money(sim: &Simulation) -> i64 {
    sim.world.actors.values().map(|a| a.inventory.money).sum()
}

// ---------------------------------------------------------------------------
// T4.2 — scarcity conservation over a randomized soak (design §5.4, §9.1.1)
// ---------------------------------------------------------------------------

#[test]
fn t4_2_money_and_items_are_conserved_modulo_recorded_sources_over_a_soak() {
    let mut w = world_fixture();
    // Real money, items and a responsible faction (budget-only sink; no
    // members, so no wage sources fire in this soak).
    w.actor_mut(&a("m1")).unwrap().inventory.money = 500;
    w.actor_mut(&a("m1"))
        .unwrap()
        .inventory
        .add(ItemId::new("coin"), 7);
    w.actor_mut(&a("m2")).unwrap().inventory.money = 300;
    w.actor_mut(&a("m2"))
        .unwrap()
        .inventory
        .add(ItemId::new("coin"), 3);
    w.items.insert(
        ItemId::new("coin"),
        Item {
            id: ItemId::new("coin"),
            name: "Coin".into(),
            description: String::new(),
            base_value: 1,
            tags: vec![],
            stackable: true,
        },
    );
    w.factions.insert(
        FactionId::new("watch"),
        rpg_domain::world::Faction {
            id: FactionId::new("watch"),
            name: "Watch".into(),
            kind: rpg_domain::world::FactionKind::Guild,
            budget: 400,
            staff: 2,
            priorities: vec![],
            responsible_for: vec![LocationId::new(SHOP)],
            members: vec![],
            legitimacy: 1.0,
            public: true,
        },
    );
    let mut s = Simulation::new(w, 2026, background_config());
    let (m0, c0) = (total_money(&s), count_items(&s, "coin"));

    // A deterministic mixed soak: real validated dispatches plus background
    // ticks, 240 ticks total.
    let mut ops = 0usize;
    for t in 0..240u64 {
        let mut proposals = Vec::new();
        match t % 7 {
            0 => proposals.push(ActionProposal {
                actor: a("m1"),
                action: Action::Pay {
                    target: a("m2"),
                    amount: 3,
                },
                source: ProposalSource::Player,
                rationale: String::new(),
            }),
            1 => proposals.push(ActionProposal {
                actor: a("m2"),
                action: Action::TransferItem {
                    target: a("m1"),
                    item: ItemId::new("coin"),
                    qty: 1,
                },
                source: ProposalSource::Player,
                rationale: String::new(),
            }),
            _ => {}
        }
        if !proposals.is_empty() {
            ops += proposals.len();
        }
        s.step(proposals);
    }
    assert!(ops > 30, "fixture: the soak must include real dispatches");

    // Conservation: total money moves *only* through recorded institutional
    // sources (`WagePaid`), items only through validated transfers. No
    // members ⇒ no wages ⇒ both totals are exactly invariant, and no
    // quantity ever went negative.
    let wages: i64 = s
        .events
        .all()
        .iter()
        .filter_map(|e| match &e.kind {
            EventKind::WagePaid { amount, .. } => Some(*amount),
            _ => None,
        })
        .sum();
    assert_eq!(wages, 0);
    assert_eq!(total_money(&s), m0, "money conserved (no sources fired)");
    assert_eq!(count_items(&s, "coin"), c0, "items conserved");
    for actor in s.world.actors.values() {
        assert!(actor.inventory.money >= 0);
        assert!(actor.inventory.quantity(&ItemId::new("coin")) < u32::MAX);
    }
}

fn count_items(sim: &Simulation, item: &str) -> u32 {
    sim.world
        .actors
        .values()
        .map(|a| a.inventory.quantity(&ItemId::new(item)))
        .sum()
}

#[test]
fn t4_5a_competing_transfers_in_one_tick_resolve_deterministically() {
    // Two competing transfers of the same custody in one tick: the pipeline
    // orders proposals deterministically (recorded tick, actor, action kind)
    // and validates each optimistically against the current state — the
    // winner is stable across runs, and the loser is rejected honestly (no
    // minting, no negative custody).
    let build = || {
        let mut w = world_fixture();
        w.actor_mut(&a("m1")).unwrap().inventory.money = 10;
        Simulation::new(w, 2026, config())
    };
    let mut s = build();
    let proposals = vec![
        ActionProposal {
            actor: a("m1"),
            action: Action::Pay {
                target: a("m2"),
                amount: 10,
            },
            source: ProposalSource::Player,
            rationale: String::new(),
        },
        ActionProposal {
            actor: a("m1"),
            action: Action::Pay {
                target: a("out"),
                amount: 10,
            },
            source: ProposalSource::Player,
            rationale: String::new(),
        },
    ];
    s.step(proposals.clone());
    // Deterministic: the lexicographically later payee loses (out > m2 is
    // checked second and finds no funds); the same proposals replayed give
    // the identical outcome.
    let mut s2 = build();
    s2.step(proposals);
    assert_eq!(
        s.world.actor(&a("m2")).unwrap().inventory.money,
        s2.world.actor(&a("m2")).unwrap().inventory.money
    );
    assert_eq!(
        s.world.actor(&a("out")).unwrap().inventory.money,
        s2.world.actor(&a("out")).unwrap().inventory.money
    );
    assert_eq!(s.world.actor(&a("m1")).unwrap().inventory.money, 0);
    // Exactly one transfer succeeded; conservation holds.
    assert_eq!(total_money(&s), 10);
    assert_eq!(
        count_events(&s, |k| matches!(k, EventKind::MoneyPaid { .. })),
        1,
        "competing claims on one custody: one deterministic winner, one honest rejection"
    );
}

// ---------------------------------------------------------------------------
// I6b item 4 — wealth-driven work through real validated wage gains
// ---------------------------------------------------------------------------

/// Fixture: the faction's shop is damaged, the member `m1` stands there with
/// an urgent wealth need, and the faction's believed damage comes from the
/// member's own evidence (witnessed fact).
fn work_fixture() -> (Simulation, rpg_domain::behavior_types::SourceChainId) {
    let mut w = world_fixture();
    w.actor_mut(&a("m1"))
        .unwrap()
        .needs
        .set(NeedKind::Wealth, 0.2);
    w.actor_mut(&a("m1"))
        .unwrap()
        .affiliations
        .push(FactionId::new("watch"));
    w.factions.insert(
        FactionId::new("watch"),
        rpg_domain::world::Faction {
            id: FactionId::new("watch"),
            name: "Watch".into(),
            kind: rpg_domain::world::FactionKind::Guild,
            budget: 1000,
            staff: 1,
            priorities: vec![],
            responsible_for: vec![LocationId::new(SHOP)],
            members: vec![a("m1")],
            legitimacy: 1.0,
            public: true,
        },
    );
    w.damage.insert(LocationId::new(SHOP), 0.5);
    add_fact(&mut w, "f_damage", Sensitivity::Public, vec![]);
    w.damage_facts
        .insert(LocationId::new(SHOP), FactId::new("f_damage"));
    let mut s = Simulation::new(w, 2026, background_config());
    let (tick, minute) = (s.world.tick, s.world.minute_of_day);
    let origin = s.events.push(
        tick,
        minute,
        Some(a("m1")),
        EventKind::SpeechEmitted {
            speaker: a("m1"),
            text: "the shop is damaged".into(),
            addressee: None,
            loudness: rpg_domain::action::Loudness::Normal,
            location: LocationId::new(SHOP),
            private: false,
        },
        Vec::new(),
        Vec::new(),
    );
    let chain = witness_evidence(
        &mut s.world,
        &mut s.events,
        tick,
        minute,
        "m1",
        "f_damage",
        origin,
    );
    seed_belief(&mut s.world, "m1", "f_damage");
    (s, chain)
}

#[test]
fn t3_9d_wealth_need_is_satisfied_only_by_a_real_validated_wage() {
    let (mut s, _chain) = work_fixture();
    let money0 = s.world.actor(&a("m1")).unwrap().inventory.money;
    let budget0 = s.world.faction(&FactionId::new("watch")).unwrap().budget;
    let wealth0 = s
        .world
        .actor(&a("m1"))
        .unwrap()
        .needs
        .value(NeedKind::Wealth);

    // The group decision (member reports) funds the initiative; the planner
    // then proposes the real repair for the standing member.
    let mut saw_initiative = false;
    let mut saw_wage = false;
    let mut saw_plan = false;
    let mut plan_retired = false;
    for _ in 0..6 {
        s.step(Vec::new());
        saw_initiative |= count_events(
            &s,
            |k| matches!(k, EventKind::FactionRepairInitiated { faction, .. } if faction.as_str() == "watch"),
        ) > 0;
        saw_wage |= count_events(
            &s,
            |k| matches!(k, EventKind::WagePaid { worker, amount: 25, .. } if worker == &a("m1")),
        ) > 0;
        saw_plan |= s.planner.plans.contains_key(&a("m1"));
        plan_retired |= saw_plan && !s.planner.plans.contains_key(&a("m1"));
        if std::env::var("WI_DEBUG").is_ok() {
            for e in s.events.all() {
                if matches!(
                    e.kind,
                    EventKind::FactionRepairInitiated { .. }
                        | EventKind::WagePaid { .. }
                        | EventKind::RepairCompleted { .. }
                ) {
                    eprintln!(
                        "t{} {:?} cd={:?}",
                        e.tick,
                        e.kind,
                        s.transient().faction_repair_cooldown
                    );
                }
            }
            eprintln!(
                "t{} plan={:?} wealth={:.3} urgent={}",
                s.world.tick,
                s.planner
                    .plans
                    .get(&a("m1"))
                    .map(|p| p.intention.option.clone()),
                s.world
                    .actor(&a("m1"))
                    .unwrap()
                    .needs
                    .value(NeedKind::Wealth),
                s.world
                    .actor(&a("m1"))
                    .unwrap()
                    .needs
                    .is_urgent(NeedKind::Wealth)
            );
        }
        if saw_initiative && saw_wage && plan_retired {
            break;
        }
    }

    assert!(
        saw_initiative,
        "the faction funded the repair initiative on member reports"
    );
    assert!(saw_wage, "the performed repair paid the real typed wage");

    // The gain is exactly the validated wage: custody gains the wage, the
    // faction pays the initiative + the real repair cost + the wage, money
    // conservation holds, and the wealth need moved only through the
    // event-sourced delta of the real gain.
    let initiative_amount: i64 = s
        .events
        .all()
        .iter()
        .filter_map(|e| match &e.kind {
            EventKind::FactionRepairInitiated { amount, .. } => Some(*amount),
            _ => None,
        })
        .sum();
    let repair_cost: i64 = s
        .events
        .all()
        .iter()
        .filter_map(|e| match &e.kind {
            EventKind::RepairCompleted { cost, .. } => Some(*cost),
            _ => None,
        })
        .sum();
    let wages: i64 = s
        .events
        .all()
        .iter()
        .filter_map(|e| match &e.kind {
            EventKind::WagePaid { amount, .. } => Some(*amount),
            _ => None,
        })
        .sum();
    let money1 = s.world.actor(&a("m1")).unwrap().inventory.money;
    let budget1 = s.world.faction(&FactionId::new("watch")).unwrap().budget;
    let wealth1 = s
        .world
        .actor(&a("m1"))
        .unwrap()
        .needs
        .value(NeedKind::Wealth);
    assert_eq!(money1 - money0, wages, "custody gained exactly the wages");
    assert_eq!(wages, 25, "the on-site wage is cost/4 (0.51*200/4)");
    assert_eq!(
        budget0 - budget1,
        initiative_amount + repair_cost + wages,
        "faction paid initiative + repair cost + wage, nothing else"
    );
    // The event-sourced gain is the truthful wage norm; the need's net
    // movement is that gain minus the classified regular decay that ran in
    // the same steps (≤ 0.002/tick × ≤6 steps).
    assert!(
        (wealth1 - wealth0 - 0.25).abs() < 0.05,
        "wealth need moved by the truthful wage norm (25/100) minus regular decay: {wealth0} → {wealth1}"
    );
    assert_eq!(
        count_events(
            &s,
            |k| matches!(k, EventKind::NeedChanged { actor, need: NeedKind::Wealth, delta, .. }
            if actor == &a("m1") && *delta > 0.0)
        ),
        1,
        "the gain is event-sourced exactly once"
    );
    assert_eq!(
        s.world.damage_at(&LocationId::new(SHOP)),
        0.0,
        "the repair really happened"
    );
    assert!(
        !s.planner.plans.contains_key(&a("m1")),
        "the work plan retired on its satisfaction predicate"
    );
}

// ---------------------------------------------------------------------------
// T4.3 — faction group decisions derive from member reports (no truth access)
// ---------------------------------------------------------------------------

#[test]
fn t4_3_faction_funds_on_member_reports_and_misbelieves_when_reports_lie() {
    // (a) Misbelief: NO real damage, but a member was *told* (deceptively)
    // that the shop is damaged. The faction funds an initiative anyway —
    // its decision followed the (wrong) member report, never the truth.
    let mut w = world_fixture();
    w.factions.insert(
        FactionId::new("watch"),
        rpg_domain::world::Faction {
            id: FactionId::new("watch"),
            name: "Watch".into(),
            kind: rpg_domain::world::FactionKind::Guild,
            budget: 1000,
            staff: 1,
            priorities: vec![],
            responsible_for: vec![LocationId::new(SHOP)],
            members: vec![a("m1")],
            legitimacy: 1.0,
            public: true,
        },
    );
    add_fact(&mut w, "f_damage", Sensitivity::Public, vec![]);
    w.damage_facts
        .insert(LocationId::new(SHOP), FactId::new("f_damage"));
    // m1 believes the (false) damage report the way a real report arrives:
    // a told claim from m2 wrote both the legacy belief and the ledger
    // evidence (the production assert_claim path does exactly this).
    seed_belief(&mut w, "m1", "f_damage");
    let mut s = Simulation::new(w, 2026, background_config());
    let (tick, minute) = (s.world.tick, s.world.minute_of_day);
    let told_event = s.events.push(
        tick,
        minute,
        Some(a("m2")),
        EventKind::SpeechEmitted {
            speaker: a("m2"),
            text: "the shop is damaged".into(),
            addressee: Some(a("m1")),
            loudness: rpg_domain::action::Loudness::Normal,
            location: LocationId::new(SHOP),
            private: false,
        },
        Vec::new(),
        vec![a("m1")],
    );
    record_provenance_evidence(
        &mut s.world,
        &mut s.events,
        tick,
        minute,
        &a("m1"),
        &FactId::new("f_damage"),
        true,
        &Provenance::Told {
            by: a("m2"),
            event: Some(told_event),
        },
    );
    let budget0 = s.world.faction(&FactionId::new("watch")).unwrap().budget;
    for _ in 0..4 {
        s.step(Vec::new());
    }
    let initiated = count_events(&s, |k| {
        matches!(k, EventKind::FactionRepairInitiated { faction, believed_damage, .. }
            if faction.as_str() == "watch" && *believed_damage > 0.0)
    });
    assert_eq!(
        initiated, 1,
        "the faction funded the initiative from the member's (wrong) report"
    );
    assert_eq!(
        s.world.damage_at(&LocationId::new(SHOP)),
        0.0,
        "the truth never moved because of the misreport"
    );
    let budget1 = s.world.faction(&FactionId::new("watch")).unwrap().budget;
    assert!(
        budget1 < budget0,
        "the misbelief cost real institutional budget (overpaid on a wrong report)"
    );
    // The miscalculated amount is the *believed* severity, evented.
    let amount = s
        .events
        .all()
        .iter()
        .find_map(|e| match &e.kind {
            EventKind::FactionRepairInitiated { amount, .. } => Some(*amount),
            _ => None,
        })
        .unwrap();
    assert!(
        amount > 0 && amount <= 200,
        "believed-severity budget estimate"
    );

    // (b) Real damage with NO observers: nobody believes, nobody funds.
    let mut w = world_fixture();
    w.factions.insert(
        FactionId::new("watch"),
        rpg_domain::world::Faction {
            id: FactionId::new("watch"),
            name: "Watch".into(),
            kind: rpg_domain::world::FactionKind::Guild,
            budget: 1000,
            staff: 1,
            priorities: vec![],
            responsible_for: vec![LocationId::new(SHOP)],
            members: vec![a("m1")],
            legitimacy: 1.0,
            public: true,
        },
    );
    w.damage.insert(LocationId::new(SHOP), 0.5);
    add_fact(&mut w, "f_damage", Sensitivity::Public, vec![]);
    w.damage_facts
        .insert(LocationId::new(SHOP), FactId::new("f_damage"));
    let mut s = Simulation::new(w, 2026, background_config());
    for _ in 0..4 {
        s.step(Vec::new());
    }
    assert_eq!(
        count_events(&s, |k| matches!(
            k,
            EventKind::FactionRepairInitiated { .. }
        )),
        0,
        "no member report ⇒ no group belief ⇒ no initiative (no truth access)"
    );
}

// ---------------------------------------------------------------------------
// I6b item 5 — one-shot completion hardening
// ---------------------------------------------------------------------------

/// Fixture knowledge: `holder`'s ledger is *contradicted* about `fact`
/// (two polarities on two chains), so the epistemic QUERY candidate exists.
fn contradicted_fixture(fact: &str) -> (Simulation, EventId, EventId) {
    let mut w = world_fixture();
    w.actor_mut(&a("m1"))
        .unwrap()
        .needs
        .set(NeedKind::Safety, 0.2);
    add_fact(&mut w, fact, Sensitivity::Public, vec![]);
    add_fact(&mut w, "f_other", Sensitivity::Public, vec![]);
    let mut s = Simulation::new(w, 2026, config());
    let (tick, minute) = (s.world.tick, s.world.minute_of_day);
    let mut push_origin = |text: &str| {
        s.events.push(
            tick,
            minute,
            Some(a("m1")),
            EventKind::SpeechEmitted {
                speaker: a("m1"),
                text: text.to_string(),
                addressee: None,
                loudness: rpg_domain::action::Loudness::Normal,
                location: LocationId::new(SHOP),
                private: false,
            },
            Vec::new(),
            Vec::new(),
        )
    };
    let e1 = push_origin("first report");
    let e2 = push_origin("second report");
    witness_evidence(&mut s.world, &mut s.events, tick, minute, "m1", fact, e1);
    // The second item is a told claim asserting the *opposite* polarity: a
    // direct ledger write through the real adapter (opposite polarity via a
    // false asserted_polarity — the sim's deceptive path records exactly
    // this shape).
    {
        let draft = rpg_behavior::AssertionDraft {
            speaker: Some(a("m2")),
            proposition: FactId::new(fact),
            asserted_polarity: false,
            channel: rpg_domain::behavior_types::AcquisitionChannel::Told,
            origin_event: Some(e2),
            reliability: 0.6,
            retold_chain: None,
            deceptive: true,
        };
        let appraisal = rpg_behavior::RecipientAppraisal {
            trust: 0.0,
            suspicion: 0.0,
            fear: 0.0,
            assertiveness: 1.0,
        };
        rpg_behavior::accept_assertion(
            &mut s.world,
            &mut s.events,
            tick,
            minute,
            &a("m1"),
            draft,
            appraisal,
            0.9,
            true,
            &mut s.rng,
        );
    }
    (s, e1, e2)
}

#[test]
fn t4_5b_one_shot_outcomes_match_by_full_parameters_not_actor_kind() {
    // Parameter matching is a property of the retirement kernel itself:
    // a same-kind outcome with different parameters must NOT complete a
    // plan, whatever its route was.
    let mut probe = rpg_domain::events::EventLog::new();
    let (tick, minute) = (7u64, 480u32);
    probe.push(
        tick,
        minute,
        Some(a("m1")),
        EventKind::QuestionAsked {
            from: a("m1"),
            to: a("m2"),
            about: FactId::new("f_x"),
        },
        Vec::new(),
        Vec::new(),
    );
    let plan = Action::Query {
        target: a("m2"),
        about: FactId::new("f_x"),
    };
    let wrong_target = Action::Query {
        target: a("out"),
        about: FactId::new("f_x"),
    };
    let wrong_about = Action::Query {
        target: a("m2"),
        about: FactId::new("f_other"),
    };
    assert!(rpg_agents::planner_candidates::one_shot_completed(
        &plan,
        &a("m1"),
        &probe,
        0
    ));
    // A different target or a different `about` is a different outcome —
    // even from the same actor, even in the same window (the kind-only
    // matching this increment repairs would conflate both).
    assert!(!rpg_agents::planner_candidates::one_shot_completed(
        &wrong_target,
        &a("m1"),
        &probe,
        0
    ));
    assert!(!rpg_agents::planner_candidates::one_shot_completed(
        &wrong_about,
        &a("m1"),
        &probe,
        0
    ));
    // Sanction plans match subject+kind, not "any sanction by me".
    probe.push(
        tick,
        minute,
        Some(a("m1")),
        EventKind::SanctionIssued {
            authority: a("m1"),
            subject: a("m2"),
            kind: "fine".into(),
            evidence: vec![],
        },
        Vec::new(),
        Vec::new(),
    );
    let same = Action::Sanction {
        target: a("m2"),
        kind: rpg_domain::action::SanctionKind::Fine { amount: 10 },
        evidence: vec![],
    };
    let other = Action::Sanction {
        target: a("out"),
        kind: rpg_domain::action::SanctionKind::Fine { amount: 10 },
        evidence: vec![],
    };
    assert!(rpg_agents::planner_candidates::one_shot_completed(
        &same,
        &a("m1"),
        &probe,
        0
    ));
    assert!(!rpg_agents::planner_candidates::one_shot_completed(
        &other,
        &a("m1"),
        &probe,
        0
    ));
}

#[test]
fn t4_5c_a_one_shot_never_reexecutes_when_its_outcome_scrolls_past_the_window() {
    let (mut s, _e1, _e2) = contradicted_fixture("f_x");
    if std::env::var("WI_DEBUG").is_ok() {
        let c = rpg_agents::planner_candidates::planner_candidates(
            &s.world,
            &a("m1"),
            &s.events,
            &s.config.needs,
            &s.config.emotions,
            s.config.talk_radius,
            16,
            &s.sanctions,
        );
        for cand in &c {
            eprintln!("cand {} feas {:?}", cand.option.id, cand.option.feasibility);
        }
    }
    // The plan is adopted (the need's urgency latch engages at the first
    // maintenance tick) and its question is executed.
    for _ in 0..6 {
        s.step(Vec::new());
        let asked = count_events(&s, |k| {
            matches!(k, EventKind::QuestionAsked { from, about, .. }
                if from == &a("m1") && about == &FactId::new("f_x"))
        });
        if asked == 1 {
            break;
        }
    }
    let asked = count_events(&s, |k| {
        matches!(k, EventKind::QuestionAsked { from, about, .. }
            if from == &a("m1") && about == &FactId::new("f_x"))
    });
    assert_eq!(asked, 1, "fixture: the one-shot question was asked once");

    // One flood step pushes ≥70 unrelated events past the bounded (64-event)
    // retirement window, so the plan's outcome is no longer visible when the
    // actor's next activation checks it.
    let mut flood = Vec::new();
    for i in 0..70 {
        flood.push(ActionProposal {
            actor: a("m2"),
            action: Action::Speak {
                target: Some(a("m1")),
                text: format!("flood {i}"),
                loudness: rpg_domain::action::Loudness::Normal,
            },
            source: ProposalSource::Player,
            rationale: String::new(),
        });
    }
    s.step(flood);

    let asked = count_events(&s, |k| {
        matches!(k, EventKind::QuestionAsked { from, about, .. }
            if from == &a("m1") && about == &FactId::new("f_x"))
    });
    assert_eq!(
        asked, 1,
        "the one-shot plan must retire on its proposed-once record, not re-execute under event load (no duplicated outcomes)"
    );
    assert!(
        !s.planner.plans.contains_key(&a("m1")),
        "the stale plan is gone"
    );
}

#[test]
fn t4_5d_signals_arriving_with_completion_are_spent_on_a_real_decision() {
    let (mut s, _e1, _e2) = contradicted_fixture("f_x");
    // The activation that executes the question also answers it — the answer
    // is delivered as a QueryAnswered reconsideration signal in the same step.
    let mut signaled = false;
    for _ in 0..4 {
        s.step(Vec::new());
        signaled = s.planner.signals.get(&a("m1")).is_some_and(|s| {
            s.iter()
                .any(|sig| matches!(sig.trigger, rpg_behavior::ReplanTrigger::QueryAnswered))
        });
        if signaled {
            break;
        }
    }
    assert!(
        signaled,
        "fixture: the answer signaled the asker in the same step"
    );

    // Next activation: the plan retires (its outcome exists) AND the pending
    // signal is spent on a real re-deliberation in that same activation.
    s.step(Vec::new());
    let pass = s.last_planner_pass();
    let activation = pass
        .activations
        .iter()
        .find(|x| x.actor == a("m1"))
        .expect("the actor was activated");
    assert!(
        activation.replanned,
        "completion + pending signals ⇒ the activation re-deliberated (signals are not dropped)"
    );
    assert!(
        s.planner.signals.get(&a("m1")).is_none_or(|s| s.is_empty()),
        "the signal was consumed by a real decision, not lost"
    );
    assert!(
        !s.planner.plans.contains_key(&a("m1")),
        "the one-shot plan is retired"
    );
}

// ---------------------------------------------------------------------------
// T4.6 — bounded deterministic offscreen catch-up
// ---------------------------------------------------------------------------

fn catchup_world() -> World {
    let mut w = world_fixture();
    w.actor_mut(&a("m1")).unwrap().inventory.money = 100;
    w.actor_mut(&a("m2")).unwrap().inventory.money = 50;
    w.actor_mut(&a("m2"))
        .unwrap()
        .inventory
        .add(ItemId::new("coin"), 2);
    w.items.insert(
        ItemId::new("coin"),
        Item {
            id: ItemId::new("coin"),
            name: "Coin".into(),
            description: String::new(),
            base_value: 1,
            tags: vec![],
            stackable: true,
        },
    );
    w.factions.insert(
        FactionId::new("watch"),
        rpg_domain::world::Faction {
            id: FactionId::new("watch"),
            name: "Watch".into(),
            kind: rpg_domain::world::FactionKind::Guild,
            budget: 800,
            staff: 1,
            priorities: vec![rpg_domain::world::FactionPriority {
                description: "patrol raises".into(),
                weight: 1.0,
                cost: 50,
                predicate: None,
                resolved: false,
            }],
            responsible_for: vec![LocationId::new(SHOP)],
            members: vec![a("m1")],
            legitimacy: 1.0,
            public: true,
        },
    );
    w.damage.insert(LocationId::new(SHOP), 0.2);
    add_fact(&mut w, "f_damage", Sensitivity::Public, vec![]);
    w.damage_facts
        .insert(LocationId::new(SHOP), FactId::new("f_damage"));
    // A real deal: m1 owes m2 four coins (the fixture gives m2 two) within
    // 120 minutes; the condition is already true (flag set).
    w.deals.insert(
        DealId(1),
        rpg_domain::deal::Deal {
            id: DealId(1),
            debtor: a("m1"),
            creditor: a("m2"),
            terms: "four coins by nightfall".into(),
            condition: rpg_domain::action::QuestPredicate::Flag("always".into()),
            performance: rpg_domain::action::QuestPredicate::HasItem {
                actor: a("m2"),
                item: ItemId::new("coin"),
                qty: 4,
            },
            condition_deadline_minute: Some(500),
            performance_deadline_minute: Some(520),
            grace_minutes: 30,
            state: DealState::Active,
            give: vec![],
            money: 0,
            installments: vec![],
            renegotiations: 0,
            created_tick: 0,
            condition_met_tick: None,
            condition_overdue_emitted: false,
            performance_overdue_emitted: false,
            grace_until_minute: None,
            resolved_minute: None,
        },
    );
    w.set_flag("always");
    w
}

#[test]
fn t4_6a_catch_up_is_bounded_deterministic_and_continuous_across_save_load() {
    // Bounds: a huge skip truncates at the documented cap.
    let mut s = Simulation::new(catchup_world(), 2026, background_config());
    let report = s.catch_up(100_000);
    assert!(report.truncated);
    assert_eq!(report.ticks, CATCH_UP_MAX_TICKS);
    assert_eq!(s.world.tick, CATCH_UP_MAX_TICKS as u64);

    // Determinism + continuity: stepping minute-by-minute online (same seed)
    // must produce the identical world and event log as one catch-up call
    // of the same duration — the catch-up *is* the production pipeline.
    let mut online = Simulation::new(catchup_world(), 2026, background_config());
    for _ in 0..100 {
        online.step(Vec::new());
    }
    let mut off = Simulation::new(catchup_world(), 2026, background_config());
    let report = off.catch_up(100);
    assert!(!report.truncated);
    assert_eq!(report.ticks, 100);
    assert_eq!(
        serde_json::to_string(&off.world).unwrap(),
        serde_json::to_string(&online.world).unwrap(),
        "offscreen catch-up is causally equivalent to online stepping"
    );
    assert_eq!(off.events.checksum(), online.events.checksum());

    // Save/load mid-catch-up and continue identically (continuity).
    let json = serde_json::to_string(&off.transient()).unwrap();
    let transient: rpg_sim::SimTransient = serde_json::from_str(&json).unwrap();
    let mut restored = Simulation::restore_full(
        off.seed,
        off.world.clone(),
        off.events.clone(),
        off.rng.clone(),
        transient,
        off.ops.clone(),
        off.initial_world.clone(),
        off.config.clone(),
    );
    for _ in 0..40 {
        off.step(Vec::new());
        restored.step(Vec::new());
    }
    assert_eq!(
        serde_json::to_string(&off.world).unwrap(),
        serde_json::to_string(&restored.world).unwrap()
    );
    assert_eq!(off.events.checksum(), restored.events.checksum());

    // Replay continuity: the recorded ops (including the catch-up steps)
    // reproduce the state exactly.
    let replay = Simulation::replay_ops(
        off.initial_world.clone(),
        off.seed,
        off.config.clone(),
        &off.ops,
    );
    assert_eq!(
        serde_json::to_string(&replay.world).unwrap(),
        serde_json::to_string(&off.world).unwrap()
    );
    assert_eq!(replay.events.checksum(), off.events.checksum());

    // Debts: the catch-up ran the real deal FSM — the overdue deal resolved
    // (deadlines long past) through the production deadline pathway.
    let deal = off.world.deals.get(&DealId(1)).unwrap();
    assert!(
        matches!(
            deal.state,
            DealState::Performed | DealState::Violated | DealState::Excused
        ),
        "the skipped deal reached a resolution state: {:?}",
        deal.state
    );
}

// ---------------------------------------------------------------------------
// I6b item 3 — autonomous retellings with explicit validated chain selection
// ---------------------------------------------------------------------------

/// Fixture: `witness` holds TWO chains for one proposition (two separate
/// witnessed origins) and believes it; `listener` is a warm nearby contact
/// who does not know the proposition.
fn retell_fixture() -> (
    Simulation,
    rpg_domain::behavior_types::SourceChainId,
    rpg_domain::behavior_types::SourceChainId,
) {
    let mut w = world_fixture();
    w.actor_mut(&a("m1"))
        .unwrap()
        .needs
        .set(NeedKind::Safety, 0.2);
    // A real social edge: retellings traverse ties (§5.3).
    w.actor_mut(&a("m2"))
        .unwrap()
        .relationship_mut(&a("m1"))
        .trust = 1.0;
    add_fact(&mut w, "f_rumor", Sensitivity::Public, vec![]);
    let mut s = Simulation::new(w, 2026, config());
    let (tick, minute) = (s.world.tick, s.world.minute_of_day);
    let e1 = s.events.push(
        tick,
        minute,
        Some(a("m1")),
        EventKind::SpeechEmitted {
            speaker: a("m1"),
            text: "first sighting".into(),
            addressee: None,
            loudness: rpg_domain::action::Loudness::Normal,
            location: LocationId::new(SHOP),
            private: false,
        },
        Vec::new(),
        Vec::new(),
    );
    let e2 = s.events.push(
        tick,
        minute,
        Some(a("m1")),
        EventKind::SpeechEmitted {
            speaker: a("m1"),
            text: "second sighting".into(),
            addressee: None,
            loudness: rpg_domain::action::Loudness::Normal,
            location: LocationId::new(SHOP),
            private: false,
        },
        Vec::new(),
        Vec::new(),
    );
    let chain_old = witness_evidence(
        &mut s.world,
        &mut s.events,
        tick,
        minute,
        "m1",
        "f_rumor",
        e1,
    );
    let chain_new = witness_evidence(
        &mut s.world,
        &mut s.events,
        tick,
        minute,
        "m1",
        "f_rumor",
        e2,
    );
    assert_ne!(chain_old, chain_new, "two independent origins ⇒ two chains");
    seed_belief(&mut s.world, "m1", "f_rumor");
    (s, chain_old, chain_new)
}

#[test]
fn t4_6b_retell_carries_the_intended_nonlatest_chain_through_the_real_route() {
    let (mut s, chain_old, chain_new) = retell_fixture();

    // Candidate-level grounding: the retell option explicitly selects the
    // speaker's *oldest* held chain (an intended, non-latest selection).
    let candidates = rpg_agents::planner_candidates::planner_candidates(
        &s.world,
        &a("m1"),
        &s.events,
        &s.config.needs,
        &s.config.emotions,
        s.config.talk_radius,
        16,
        &s.sanctions,
    );
    let retell = candidates
        .iter()
        .find(|c| c.option.id.starts_with("retell:"))
        .expect("the believing actor generates a retell candidate");
    let Action::ShareClaim {
        target,
        fact,
        retold_chain,
        ..
    } = &retell.action
    else {
        panic!("retell candidate must be a ShareClaim");
    };
    assert_eq!(target, &a("m2"));
    assert_eq!(fact, &FactId::new("f_rumor"));
    assert_eq!(
        retold_chain.as_ref(),
        Some(&chain_old),
        "the intended (non-latest) chain is selected explicitly"
    );

    // Pipeline: the same candidate is adopted and dispatched through the
    // real production retelling route; the listener's evidence cites the
    // selected chain.
    for _ in 0..4 {
        s.step(Vec::new());
        if !s
            .world
            .epistemics
            .evidence_about(&a("m2"), &FactId::new("f_rumor"))
            .is_empty()
        {
            break;
        }
    }
    let delivered = s
        .world
        .epistemics
        .evidence_about(&a("m2"), &FactId::new("f_rumor"));
    assert_eq!(delivered.len(), 1, "one retelling, one evidence item");
    assert_eq!(
        delivered[0].source_chain_id,
        chain_old,
        "the delivered evidence continues the explicitly selected chain (not the latest {})",
        chain_new.raw()
    );
    assert_eq!(
        count_events(&s, |k| matches!(k, EventKind::ClaimShared { to, fact, .. }
            if to == &a("m2") && fact == &FactId::new("f_rumor"))),
        1,
        "the retelling went through the real ClaimShared route exactly once"
    );
}

#[test]
fn t4_6c_retells_respect_disclosure_permissions_and_stay_neutral_on_unknowns() {
    // A Secret the actor knows but is not a holder of: no retell candidate,
    // and the authoritative route rejects the attempt without delivering.
    let mut w = world_fixture();
    w.actor_mut(&a("m1"))
        .unwrap()
        .needs
        .set(NeedKind::Safety, 0.2);
    w.actor_mut(&a("m2"))
        .unwrap()
        .relationship_mut(&a("m1"))
        .trust = 1.0;
    add_fact(&mut w, "f_secret", Sensitivity::Secret, vec!["m2"]);
    let mut s = Simulation::new(w, 2026, config());
    let (tick, minute) = (s.world.tick, s.world.minute_of_day);
    let e1 = s.events.push(
        tick,
        minute,
        Some(a("m1")),
        EventKind::SpeechEmitted {
            speaker: a("m1"),
            text: "overheard".into(),
            addressee: None,
            loudness: rpg_domain::action::Loudness::Normal,
            location: LocationId::new(SHOP),
            private: false,
        },
        Vec::new(),
        Vec::new(),
    );
    witness_evidence(
        &mut s.world,
        &mut s.events,
        tick,
        minute,
        "m1",
        "f_secret",
        e1,
    );
    seed_belief(&mut s.world, "m1", "f_secret");

    let candidates = rpg_agents::planner_candidates::planner_candidates(
        &s.world,
        &a("m1"),
        &s.events,
        &s.config.needs,
        &s.config.emotions,
        s.config.talk_radius,
        16,
        &s.sanctions,
    );
    assert!(
        candidates
            .iter()
            .all(|c| !c.option.id.starts_with("retell:")),
        "a secret the actor may not disclose generates no retell candidate"
    );
    let outcome = s.apply(&ActionProposal {
        actor: a("m1"),
        action: Action::ShareClaim {
            target: a("m2"),
            fact: FactId::new("f_secret"),
            assertive: true,
            deceptive: false,
            retold_chain: None,
        },
        source: ProposalSource::Player,
        rationale: String::new(),
    });
    assert!(
        matches!(outcome, rpg_domain::action::ValidationOutcome::Rejected(_)),
        "the authoritative route still refuses the unauthorized disclosure"
    );
    assert!(
        s.world
            .epistemics
            .evidence_about(&a("m2"), &FactId::new("f_secret"))
            .is_empty(),
        "nothing was delivered"
    );

    // Unknown/unknown: retelling a fact the actor does not even know of is
    // rejected neutrally (LacksKnowledge), mutating nothing.
    add_fact(&mut s.world, "f_unknown", Sensitivity::Public, vec![]);
    let outcome = s.apply(&ActionProposal {
        actor: a("m1"),
        action: Action::ShareClaim {
            target: a("m2"),
            fact: FactId::new("f_unknown"),
            assertive: true,
            deceptive: false,
            retold_chain: None,
        },
        source: ProposalSource::Player,
        rationale: String::new(),
    });
    assert!(matches!(
        outcome,
        rpg_domain::action::ValidationOutcome::Rejected(
            rpg_domain::action::RejectReason::LacksKnowledge(_)
        )
    ));
}

#[test]
fn t4_6d_a_bogus_explicit_chain_falls_back_safely_in_the_production_route() {
    let (mut s, _old, _new) = retell_fixture();
    // An explicit chain the speaker does not own: the epistemic layer
    // validates the selection and falls back to the legacy inference —
    // delivery still happens, the chain registry is not corrupted, and the
    // listener receives real evidence.
    let bogus = rpg_domain::behavior_types::SourceChainId(99_999);
    let outcome = s.apply(&ActionProposal {
        actor: a("m1"),
        action: Action::ShareClaim {
            target: a("m2"),
            fact: FactId::new("f_rumor"),
            assertive: true,
            deceptive: false,
            retold_chain: Some(bogus),
        },
        source: ProposalSource::Player,
        rationale: String::new(),
    });
    assert!(matches!(
        outcome,
        rpg_domain::action::ValidationOutcome::Accepted
    ));
    let delivered = s
        .world
        .epistemics
        .evidence_about(&a("m2"), &FactId::new("f_rumor"));
    assert_eq!(delivered.len(), 1);
    assert_ne!(
        delivered[0].source_chain_id, bogus,
        "a bogus pointer never becomes the listener's provenance"
    );
    assert!(
        s.world.epistemics.chain(bogus).is_none(),
        "the registry was not corrupted by the bogus selection"
    );
}

// ---------------------------------------------------------------------------
// I6b item 7 — sanction-evidence confidentiality (§9.1.12 noninterference)
// ---------------------------------------------------------------------------

/// Fixture: an authority sanctions a culprit over a crime fact; `out` is a
/// bystander with no sensory or communication route to the evidence.
fn sanction_fixture(crime_fact: &str) -> Simulation {
    let mut w = world_fixture();
    let mut guard = actor("guard", HALL, 15.0);
    guard.capabilities.can_arrest = true;
    w.actors.insert(a("guard").clone(), guard);
    w.actor_mut(&a("m1")).unwrap().location = LocationId::new(HALL);
    w.actor_mut(&a("m1")).unwrap().pos = Vec2::new(15.5, 5.0);
    w.actor_mut(&a("m1")).unwrap().inventory.money = 50;
    add_fact(&mut w, crime_fact, Sensitivity::Public, vec!["guard"]);
    seed_belief(&mut w, "guard", crime_fact);
    let mut s = Simulation::new(w, 2026, config());
    // The bystander `out` is in the far room the whole time (no route).
    apply(
        &mut s,
        "guard",
        Action::Sanction {
            target: a("m1"),
            kind: rpg_domain::action::SanctionKind::Fine { amount: 10 },
            evidence: vec![FactId::new(crime_fact)],
        },
    );
    s
}

fn bystander_state(sim: &Simulation, crime_fact: &str) -> String {
    let actor = sim.world.actor(&a("out")).unwrap().clone();
    // Everything the bystander could possibly observe about itself: its own
    // mind (beliefs/needs/emotions/relationships/…) and its own evidence
    // ledger entries for the crime proposition.
    let ledger = sim
        .world
        .epistemics
        .evidence_about(&a("out"), &FactId::new(crime_fact));
    format!("{}|{:?}", serde_json::to_string(&actor).unwrap(), ledger)
}

#[test]
fn t4_10d_sanction_evidence_stays_confidential_to_unauthorized_observers() {
    let s = sanction_fixture("f_crime");
    // The bystander never learned the evidence.
    assert!(
        !rpg_behavior::knows_of(&s.world, &a("out"), &FactId::new("f_crime")),
        "no route ⇒ no evidence delivery"
    );
    // The sanction event observed only its subject; the bystander's planner
    // (which receives the event log) generates nothing grounded in the
    // protected evidence.
    let sanction = s
        .events
        .all()
        .iter()
        .find(|e| matches!(e.kind, EventKind::SanctionIssued { .. }))
        .unwrap();
    assert_eq!(sanction.observed_by, vec![a("m1")]);
    let candidates = rpg_agents::planner_candidates::planner_candidates(
        &s.world,
        &a("out"),
        &s.events,
        &s.config.needs,
        &s.config.emotions,
        s.config.talk_radius,
        16,
        &s.sanctions,
    );
    assert!(
        candidates.iter().all(|c| !c.summary.contains("f_crime")),
        "no candidate is grounded in evidence the actor cannot know"
    );
}

#[test]
fn t4_10e_changing_sanction_evidence_changes_nothing_the_bystander_can_observe() {
    // Noninterference at the sanction seam (design §9.1.12): the same
    // scenario with a *different* evidence fact id (and the subject
    // punished under a different identity) must leave the unauthorized
    // bystander's observable state identical under the same seed.
    let s_a = sanction_fixture("f_crime_a");
    let s_b = sanction_fixture("f_crime_b");
    assert_eq!(
        bystander_state(&s_a, "f_crime_a"),
        bystander_state(&s_b, "f_crime_b"),
        "the unauthorized bystander's observable state is identical under a changed sanction-evidence identity"
    );
    // The bystander's perceived event stream is identical: every event that
    // reached them is the same (descriptions never carry the evidence ids).
    let seen = |s: &Simulation| -> Vec<String> {
        s.events
            .all()
            .iter()
            .filter(|e| e.observed_by.iter().any(|o| o == &a("out")))
            .map(|e| e.kind.describe())
            .collect()
    };
    assert_eq!(seen(&s_a), seen(&s_b));
}

// ---------------------------------------------------------------------------
// M1 — executable mutation audit over a production soak
// ---------------------------------------------------------------------------

#[test]
fn m1_every_canonical_mutation_is_evented_or_classified() {
    let mut w = world_fixture();
    w.actor_mut(&a("m1")).unwrap().inventory.money = 200;
    w.actor_mut(&a("m2")).unwrap().inventory.money = 100;
    w.actor_mut(&a("m2"))
        .unwrap()
        .inventory
        .add(ItemId::new("coin"), 5);
    w.items.insert(
        ItemId::new("coin"),
        Item {
            id: ItemId::new("coin"),
            name: "Coin".into(),
            description: String::new(),
            base_value: 1,
            tags: vec![],
            stackable: true,
        },
    );
    add_fact(&mut w, "f_pub", Sensitivity::Public, vec![]);
    seed_belief(&mut w, "m1", "f_pub");
    // A responsible faction whose shop wears (background maintenance) — no
    // members, so no group decisions in this audit.
    w.factions.insert(
        FactionId::new("watch"),
        rpg_domain::world::Faction {
            id: FactionId::new("watch"),
            name: "Watch".into(),
            kind: rpg_domain::world::FactionKind::Guild,
            budget: 500,
            staff: 1,
            priorities: vec![],
            responsible_for: vec![LocationId::new(SHOP)],
            members: vec![],
            legitimacy: 1.0,
            public: true,
        },
    );
    w.damage.insert(LocationId::new(SHOP), 0.05);
    let mut s = Simulation::new(w, 2026, background_config());
    let money0 = total_money(&s);

    // Classified background-maintenance JSON paths (plan G4/M1): regular
    // needs/emotions/relationship/memory-salience decay + the clock.
    let classified_actor_keys = ["needs", "emotions", "rel_memory", "relationships"];
    let classified_world_keys = ["tick", "day", "minute_of_day"];

    let soak: Vec<Vec<ActionProposal>> = (0..90u64)
        .map(|t| {
            let mut v = Vec::new();
            match t % 9 {
                0 => v.push(ActionProposal {
                    actor: a("m1"),
                    action: Action::Pay {
                        target: a("m2"),
                        amount: 1,
                    },
                    source: ProposalSource::Player,
                    rationale: String::new(),
                }),
                1 => v.push(ActionProposal {
                    actor: a("m2"),
                    action: Action::TransferItem {
                        target: a("m1"),
                        item: ItemId::new("coin"),
                        qty: 1,
                    },
                    source: ProposalSource::Player,
                    rationale: String::new(),
                }),
                2 => v.push(ActionProposal {
                    actor: a("m1"),
                    action: Action::ShareClaim {
                        target: a("m2"),
                        fact: FactId::new("f_pub"),
                        assertive: true,
                        deceptive: false,
                        retold_chain: None,
                    },
                    source: ProposalSource::Player,
                    rationale: String::new(),
                }),
                3 => v.push(ActionProposal {
                    actor: a("m1"),
                    action: Action::MoveTo {
                        location: LocationId::new(HALL),
                    },
                    source: ProposalSource::Player,
                    rationale: String::new(),
                }),
                4 => v.push(ActionProposal {
                    actor: a("m1"),
                    action: Action::MoveTo {
                        location: LocationId::new(SHOP),
                    },
                    source: ProposalSource::Player,
                    rationale: String::new(),
                }),
                _ => {}
            }
            v
        })
        .collect();

    for proposals in soak {
        let before_events = s.events.all().len();
        let before = serde_json::to_value(&s.world).unwrap();
        s.step(proposals);
        let after_events = s.events.all().len();
        let after = serde_json::to_value(&s.world).unwrap();

        // Strip classified background maintenance from both snapshots.
        let mut b = before;
        let mut f = after;
        for key in classified_world_keys {
            b.as_object_mut().unwrap().remove(key);
            f.as_object_mut().unwrap().remove(key);
        }
        for side in [&mut b, &mut f] {
            if let Some(actors) = side.get_mut("actors").and_then(|v| v.as_object_mut()) {
                for (_, actor) in actors.iter_mut() {
                    if let Some(o) = actor.as_object_mut() {
                        for key in classified_actor_keys {
                            o.remove(key);
                        }
                    }
                }
            }
        }
        if b != f {
            assert!(
                after_events > before_events,
                "canonical state changed with no event in the same step (a silent mutation): diff = {}",
                serde_json::to_string(&json_diff_summary(&b, &f)).unwrap()
            );
        }
    }

    // Conservation still holds over the audited soak (no wages fired).
    let wages: i64 = s
        .events
        .all()
        .iter()
        .filter_map(|e| match &e.kind {
            EventKind::WagePaid { amount, .. } => Some(*amount),
            _ => None,
        })
        .sum();
    assert_eq!(total_money(&s), money0 + wages);
}

/// Compact top-level diff keys for the audit failure message.
fn json_diff_summary(a: &serde_json::Value, b: &serde_json::Value) -> serde_json::Value {
    let mut keys = Vec::new();
    if let (Some(x), Some(y)) = (a.as_object(), b.as_object()) {
        for k in x.keys() {
            if x.get(k) != y.get(k) {
                keys.push(serde_json::Value::String(k.clone()));
            }
        }
        for k in y.keys() {
            if !x.contains_key(k) {
                keys.push(serde_json::Value::String(k.clone()));
            }
        }
    }
    serde_json::Value::Array(keys)
}
