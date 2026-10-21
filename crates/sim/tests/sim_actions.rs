//! Integration tests for the authoritative simulation: action validation,
//! offers/commitments, perception routing, clue discovery, determinism and
//! save/load round-tripping.

use rpg_domain::action::{
    Action, ActionProposal, InvestigateTarget, Loudness, ProposalSource, QuestPredicate,
    RejectReason, ValidationOutcome,
};
use rpg_domain::events::EventKind;
use rpg_domain::geom::{GridPos, Rect, Vec2};
use rpg_domain::ids::*;
use rpg_domain::social::{CommitmentState, OfferState};
use rpg_domain::world::{
    Actor, ActorKind, ActorStatus, Capabilities, ContainerKind, Fact, FactCategory, Inventory,
    Item, Location, LocationKind, MysteryMode, NavGrid, Personality, Sensitivity, StoryAnchor,
    StoryState, Tile, World,
};
use rpg_narrative::NarrativeChange;
use rpg_sim::{SimConfig, Simulation};
use std::collections::{BTreeMap, BTreeSet};

fn loc(id: &str, name: &str, kind: LocationKind, x: f32, y: f32, private: bool) -> Location {
    Location {
        id: LocationId::new(id),
        name: name.into(),
        kind,
        bounds: Rect::new(x, y, 6.0, 6.0),
        owner: None,
        sound_transmission: 1.0,
        private,
        tags: vec![],
    }
}

fn actor(id: &str, kind: ActorKind, location: &str, pos: Vec2) -> Actor {
    Actor {
        id: ActorId::new(id),
        name: id.into(),
        kind,
        status: ActorStatus::Alive,
        location: LocationId::new(location),
        pos,
        facing: 0.0,
        personality: Personality::default(),
        values: vec![],
        affiliations: vec![],
        capabilities: Capabilities::default(),
        inventory: Inventory::default(),
        goals: vec![],
        schedule: vec![],
        relationships: BTreeMap::new(),
        beliefs: BTreeMap::new(),
        inspected: BTreeSet::new(),
        flags: BTreeSet::new(),
        secret_allegiance: None,
        in_private_conversation_with: None,
    }
}

pub fn test_world() -> World {
    let mut w = World {
        tick: 0,
        day: 0,
        minute_of_day: 480,
        locations: BTreeMap::new(),
        actors: BTreeMap::new(),
        items: BTreeMap::new(),
        factions: BTreeMap::new(),
        facts: BTreeMap::new(),
        nav: NavGrid::new(24, 16),
        story: StoryState {
            mode: MysteryMode::FixedAnswer,
            anchors: vec![StoryAnchor {
                id: "mafia".into(),
                statement: "A mafia exists".into(),
                immutable: true,
            }],
            checkpoints: vec![],
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
    };
    w.locations.insert(
        LocationId::new("shop"),
        loc("shop", "Grocer", LocationKind::Shop, 0.0, 0.0, false),
    );
    w.locations.insert(
        LocationId::new("backroom"),
        loc(
            "backroom",
            "Backroom",
            LocationKind::Backroom,
            0.0,
            6.0,
            true,
        ),
    );
    w.locations.insert(
        LocationId::new("office"),
        loc(
            "office",
            "Guard Office",
            LocationKind::GuardOffice,
            6.0,
            0.0,
            false,
        ),
    );
    w.locations.insert(
        LocationId::new("warehouse"),
        loc(
            "warehouse",
            "Warehouse",
            LocationKind::Warehouse,
            12.0,
            0.0,
            false,
        ),
    );
    for y in 0..5 {
        w.nav.set(GridPos::new(5, y), Tile::Wall);
    }

    let mut player = actor(PLAYER, ActorKind::Player, "shop", Vec2::new(2.0, 2.0));
    player.inventory.money = 500;
    w.actors.insert(player.id.clone(), player);

    let mut grocer = actor("grocer", ActorKind::Npc, "shop", Vec2::new(3.0, 2.0));
    grocer.capabilities.can_trade = true;
    grocer.affiliations.push(FactionId::new("merchants"));
    w.actors.insert(grocer.id.clone(), grocer);

    let mut guard = actor("guard", ActorKind::Npc, "office", Vec2::new(7.0, 2.0));
    guard.capabilities.can_arrest = true;
    guard.capabilities.can_repair = true;
    guard.flags.insert("authority".into());
    guard.affiliations.push(FactionId::new("watch"));
    w.actors.insert(guard.id.clone(), guard);

    w.items.insert(
        ItemId::new("ledger"),
        Item {
            id: ItemId::new("ledger"),
            name: "Ledger".into(),
            description: "Shipping ledger".into(),
            base_value: 50,
            tags: vec!["evidence".into()],
            stackable: false,
        },
    );
    w.items.insert(
        ItemId::new("medicine"),
        Item {
            id: ItemId::new("medicine"),
            name: "Medicine".into(),
            description: "Fever medicine".into(),
            base_value: 80,
            tags: vec!["favor".into()],
            stackable: true,
        },
    );
    w.facts.insert(
        FactId::new("shipment"),
        Fact {
            id: FactId::new("shipment"),
            statement: "Shipments are being diverted by the mafia".into(),
            truth: true,
            sensitivity: Sensitivity::Secret,
            category: FactCategory::Shipment,
            prerequisites: vec![],
            eligible_holders: vec![ActorId::new("grocer")],
            eligible_containers: vec![ContainerKind::Document, ContainerKind::Testimony],
            committed: false,
            placed_at: None,
        },
    );
    w
}

fn sim() -> Simulation {
    let config = SimConfig {
        enable_background: false,
        enable_director: false,
        minutes_per_tick: 1,
        ..SimConfig::default()
    };
    Simulation::new(test_world(), 7, config)
}

fn proposal(actor: &str, action: Action) -> ActionProposal {
    ActionProposal {
        actor: ActorId::new(actor),
        action,
        source: ProposalSource::Player,
        rationale: String::new(),
    }
}

#[test]
fn speak_routes_to_addressee() {
    let mut s = sim();
    let out = s.apply(&proposal(
        PLAYER,
        Action::Speak {
            target: Some(ActorId::new("grocer")),
            text: "Good morning.".into(),
            loudness: Loudness::Normal,
        },
    ));
    assert_eq!(out, ValidationOutcome::Accepted);
    let ev = s.events.all().last().unwrap();
    match &ev.kind {
        EventKind::SpeechEmitted { addressee, .. } => {
            assert_eq!(addressee.as_ref().unwrap().as_str(), "grocer")
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn out_of_range_speech_is_rejected() {
    let mut s = sim();
    let out = s.apply(&proposal(
        PLAYER,
        Action::Speak {
            target: Some(ActorId::new("guard")),
            text: "Hello?".into(),
            loudness: Loudness::Normal,
        },
    ));
    assert!(matches!(
        out,
        ValidationOutcome::Rejected(RejectReason::OutOfRange)
    ));
}

#[test]
fn offer_pending_then_transfers_on_accept() {
    let mut s = sim();
    if let Some(g) = s.world.actor_mut(&ActorId::new("grocer")) {
        g.inventory.add(ItemId::new("ledger"), 1);
    }
    let mut p = proposal(
        "grocer",
        Action::MakeOffer {
            target: ActorId::new(PLAYER),
            give: vec![rpg_domain::world::ItemStack {
                item: ItemId::new("ledger"),
                qty: 1,
            }],
            want: vec![],
            condition: None,
            note: "take the ledger".into(),
        },
    );
    p.actor = ActorId::new("grocer");
    let offer_id = match s.apply(&p) {
        ValidationOutcome::Pending {
            offer: Some(id), ..
        } => id,
        other => panic!("expected pending offer, got {other:?}"),
    };
    assert_eq!(
        s.world
            .actor(&ActorId::new(PLAYER))
            .unwrap()
            .inventory
            .quantity(&ItemId::new("ledger")),
        0
    );
    assert_eq!(
        s.apply(&proposal(PLAYER, Action::AcceptOffer { offer: offer_id })),
        ValidationOutcome::Accepted
    );
    assert_eq!(
        s.world
            .actor(&ActorId::new(PLAYER))
            .unwrap()
            .inventory
            .quantity(&ItemId::new("ledger")),
        1
    );
    assert_eq!(s.world.offers[&offer_id].state, OfferState::Accepted);
}

#[test]
fn promise_satisfied_by_predicate() {
    let mut s = sim();
    let id = match s.apply(&proposal(
        PLAYER,
        Action::Promise {
            target: ActorId::new("grocer"),
            terms: "I will protect your family".into(),
            predicate: QuestPredicate::Flag("family_safe".into()),
            deadline: None,
        },
    )) {
        ValidationOutcome::Pending {
            commitment: Some(id),
            ..
        } => id,
        other => panic!("expected commitment, got {other:?}"),
    };
    s.world.set_flag("family_safe");
    let report = s.step(vec![]);
    assert!(report.satisfied_commitments.contains(&id));
    assert_eq!(s.world.commitments[&id].state, CommitmentState::Satisfied);
}

#[test]
fn threat_raises_fear_and_anger() {
    let mut s = sim();
    assert_eq!(
        s.apply(&proposal(
            PLAYER,
            Action::Threaten {
                target: ActorId::new("grocer"),
                demand: "Give me the ledger".into(),
            }
        )),
        ValidationOutcome::Accepted
    );
    let rel = s
        .world
        .actor(&ActorId::new("grocer"))
        .unwrap()
        .relationship(&ActorId::new(PLAYER));
    assert!(rel.fear > 0.0 && rel.anger > 0.0);
}

#[test]
fn cannot_disclose_fact_you_do_not_believe() {
    let mut s = sim();
    assert!(matches!(
        s.apply(&proposal(
            PLAYER,
            Action::ShareClaim {
                target: ActorId::new("grocer"),
                fact: FactId::new("shipment"),
                assertive: true,
            }
        )),
        ValidationOutcome::Rejected(RejectReason::LacksKnowledge(_))
    ));
}

#[test]
fn investigate_reveals_placed_clue() {
    let mut s = sim();
    let mut events = rpg_domain::events::EventLog::new();
    let decision = rpg_narrative::commit_change(
        &mut s.world,
        &mut events,
        1,
        480,
        NarrativeChange::PlaceClue {
            fact: FactId::new("shipment"),
            kind: ContainerKind::Document,
            location: Some(LocationId::new("backroom")),
            holder: None,
        },
    );
    assert!(decision.is_accepted());
    // Searching is local: move into the room first.
    if let Some(p) = s.world.actor_mut(&ActorId::new(PLAYER)) {
        p.location = LocationId::new("backroom");
        p.pos = Vec2::new(3.0, 9.0);
    }
    assert_eq!(
        s.apply(&proposal(
            PLAYER,
            Action::Investigate {
                target: InvestigateTarget::Location(LocationId::new("backroom")),
            }
        )),
        ValidationOutcome::Accepted
    );
    assert!(s
        .world
        .believes(&ActorId::new(PLAYER), &FactId::new("shipment")));
}

#[test]
fn cannot_investigate_a_distant_location() {
    let mut s = sim();
    let out = s.apply(&proposal(
        PLAYER,
        Action::Investigate {
            target: InvestigateTarget::Location(LocationId::new("backroom")),
        },
    ));
    assert!(matches!(
        out,
        ValidationOutcome::Rejected(RejectReason::OutOfRange)
    ));
}

#[test]
fn restore_full_preserves_rng_and_movement_for_exact_continuation() {
    let mut s = sim();
    // Send the player on a multi-tick path, then snapshot mid-walk.
    let _ = s.apply(&proposal(
        PLAYER,
        Action::MoveTo {
            location: LocationId::new("warehouse"),
        },
    ));
    s.move_actors();
    let mut restored = Simulation::restore_full(
        s.seed,
        s.world.clone(),
        s.events.clone(),
        s.rng.clone(),
        s.transient(),
        s.ops.clone(),
        s.initial_world.clone(),
        s.config.clone(),
    );

    // Continue both simulations and compare canonical state exactly.
    for _ in 0..10 {
        s.step(vec![]);
        restored.step(vec![]);
    }
    assert_eq!(
        serde_json::to_string(&s.world).unwrap(),
        serde_json::to_string(&restored.world).unwrap(),
        "world diverged after restoring mid-movement"
    );
    assert_eq!(s.events.checksum(), restored.events.checksum());
}

#[test]
fn proposal_replay_reconstructs_identical_world() {
    let mut s = sim();
    let _ = s.apply(&proposal(
        PLAYER,
        Action::Speak {
            target: Some(ActorId::new("grocer")),
            text: "hello".into(),
            loudness: Loudness::Normal,
        },
    ));
    s.step(vec![]);
    let _ = s.apply(&proposal(
        PLAYER,
        Action::Threaten {
            target: ActorId::new("grocer"),
            demand: "ledger".into(),
        },
    ));
    for _ in 0..5 {
        s.step(vec![]);
    }
    let replayed =
        rpg_sim::Simulation::replay_ops(s.initial_world.clone(), s.seed, s.config.clone(), &s.ops);
    assert_eq!(
        serde_json::to_string(&s.world).unwrap(),
        serde_json::to_string(&replayed.world).unwrap(),
        "proposal replay did not reconstruct the world"
    );
    assert_eq!(s.events.checksum(), replayed.events.checksum());
}

#[test]
fn movement_advances_along_path() {
    let mut s = sim();
    assert_eq!(
        s.apply(&proposal(
            PLAYER,
            Action::MoveTo {
                location: LocationId::new("warehouse"),
            }
        )),
        ValidationOutcome::Accepted
    );
    let before = s.world.actor(&ActorId::new(PLAYER)).unwrap().pos;
    s.move_actors();
    let after = s.world.actor(&ActorId::new(PLAYER)).unwrap().pos;
    assert!(before.distance(after) > 0.0);
}

#[test]
fn deterministic_replay_for_same_seed_and_proposals() {
    let run = || {
        let mut s = sim();
        for i in 0..20 {
            let _ = s.apply(&proposal(
                PLAYER,
                Action::Speak {
                    target: Some(ActorId::new("grocer")),
                    text: format!("line {i}"),
                    loudness: Loudness::Normal,
                },
            ));
            let _ = s.step(vec![]);
        }
        s.events.checksum()
    };
    assert_eq!(run(), run());
}

#[test]
fn world_and_events_roundtrip_through_json() {
    let mut s = sim();
    let _ = s.apply(&proposal(
        PLAYER,
        Action::Threaten {
            target: ActorId::new("grocer"),
            demand: "ledger".into(),
        },
    ));
    let json = serde_json::to_string(&(&s.world, &s.events)).unwrap();
    let (w2, e2): (World, rpg_domain::events::EventLog) = serde_json::from_str(&json).unwrap();
    assert_eq!(
        serde_json::to_string(&w2).unwrap(),
        serde_json::to_string(&s.world).unwrap()
    );
    assert_eq!(e2.checksum(), s.events.checksum());
}
