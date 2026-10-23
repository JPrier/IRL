//! # rpg_content
//!
//! The handcrafted vertical-slice town for the **Hidden Mafia** scenario.
//!
//! Content is plain data. Building the world here (rather than in the sim)
//! keeps scenarios reproducible and lets the evaluation harness spin up the
//! same town with different seeds.

use rpg_domain::geom::{GridPos, Rect, Vec2};
use rpg_domain::ids::*;
use rpg_domain::world::{
    Actor, ActorKind, ActorStatus, Belief, Capabilities, Checkpoint, CheckpointState,
    ContainerKind, Fact, FactCategory, Faction, FactionKind, FactionPriority, Goal, GoalKind,
    GoalStatus, Inventory, Item, Location, LocationKind, MysteryMode, NavGrid, Personality,
    Provenance, ScheduleEntry, Sensitivity, StoryAnchor, StoryState, Tile, World, WorldLore,
};
use std::collections::{BTreeMap, BTreeSet};

pub const TOWN_WIDTH: i32 = 40;
pub const TOWN_HEIGHT: i32 = 24;

/// The secret leader key stored in `StoryState::fixed_answer`.
pub const LEADER_KEY: &str = "mafia_leader";

fn wall_rect(nav: &mut NavGrid, x: i32, y: i32, w: i32, h: i32, doors: &[GridPos]) {
    for gx in x..x + w {
        for gy in y..y + h {
            let on_edge = gx == x || gx == x + w - 1 || gy == y || gy == y + h - 1;
            let g = GridPos::new(gx, gy);
            if on_edge && !doors.contains(&g) {
                nav.set(g, Tile::Wall);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn npc(
    id: &str,
    name: &str,
    location: &str,
    pos: Vec2,
    personality: Personality,
    affiliations: &[&str],
    flags: &[&str],
    secret: Option<&str>,
    schedule: Vec<ScheduleEntry>,
) -> Actor {
    Actor {
        id: ActorId::new(id),
        name: name.into(),
        kind: ActorKind::Npc,
        status: ActorStatus::Alive,
        location: LocationId::new(location),
        pos,
        facing: 0.0,
        personality,
        values: vec![],
        biography: String::new(),
        affiliations: affiliations.iter().map(|a| FactionId::new(*a)).collect(),
        capabilities: Capabilities::default(),
        inventory: Inventory::default(),
        goals: vec![],
        schedule,
        relationships: BTreeMap::new(),
        beliefs: BTreeMap::new(),
        inspected: BTreeSet::new(),
        flags: flags.iter().map(|f| f.to_string()).collect(),
        secret_allegiance: secret.map(FactionId::new),
        in_private_conversation_with: None,
    }
}

/// Populate biographies and goals for the named NPCs. Called once when the
/// canonical town is built; goals remain persistent game-state entities.
fn populate_npc_lore(actors: &mut BTreeMap<ActorId, Actor>) {
    let goal = |id: u64, description: &str, kind: GoalKind, priority: f32| Goal {
        id,
        description: description.into(),
        kind,
        priority,
        status: GoalStatus::Active,
        predicate: None,
    };

    if let Some(a) = actors.get_mut(&ActorId::new("grocer")) {
        a.biography = "Marta Voss runs Voss & Daughter's, a modest grocer in the Market Plaza. A widow, she is fiercely protective of her sick daughter, Lena. Her shop is struggling, and she has recently been pressured by the mafia.".into();
        a.goals = vec![
            goal(1, "Keep Lena healthy and safe", GoalKind::Protect, 10.0),
            goal(2, "Keep the shop open and profitable", GoalKind::Wealth, 8.0),
            goal(3, "Avoid angering the mafia", GoalKind::Safety, 7.0),
            goal(4, "Protect the Voss family reputation", GoalKind::Status, 5.0),
        ];
    }

    if let Some(a) = actors.get_mut(&ActorId::new("grocer_daughter")) {
        a.biography = "Lena Voss is Marta's teenage daughter. She suffers from a chronic illness and rarely leaves the backroom of the grocer. She is observant and bright, but physically frail.".into();
        a.goals = vec![
            goal(1, "Recover from illness", GoalKind::Survival, 10.0),
            goal(2, "Keep mother from worrying too much", GoalKind::Affection, 7.0),
            goal(3, "Understand the strange visitors to the shop", GoalKind::Investigate, 4.0),
        ];
    }

    if let Some(a) = actors.get_mut(&ActorId::new("guard_captain")) {
        a.biography = "Captain Ilse Renn commands the local Watch. She is a stern, practical woman who has spent years trying to curb the mafia's influence. Recently, she has been frustrated by a lack of evidence and political interference.".into();
        a.goals = vec![
            goal(1, "Maintain public order", GoalKind::Duty, 9.0),
            goal(2, "Find evidence against the mafia", GoalKind::Investigate, 8.0),
            goal(3, "Keep her deputies safe", GoalKind::Safety, 7.0),
            goal(4, "Advance her career", GoalKind::Status, 6.0),
        ];
    }

    if let Some(a) = actors.get_mut(&ActorId::new("guard_rookie")) {
        a.biography = "Deputy Bram is a young, eager Watchman who serves under Captain Renn. He is idealistic but naive, often patrolling the plaza and hoping to make a difference.".into();
        a.goals = vec![
            goal(1, "Do a good job for the Captain", GoalKind::Duty, 8.0),
            goal(2, "Prove himself", GoalKind::Status, 7.0),
            goal(3, "Stay out of trouble", GoalKind::Safety, 5.0),
        ];
    }

    if let Some(a) = actors.get_mut(&ActorId::new("dockworker")) {
        a.biography = "Old Tarn has worked the docks for decades. He is gruff and secretive, having seen too much over the years. He is rumored to know things about the mafia's shipments.".into();
        a.goals = vec![
            goal(1, "Keep his head down", GoalKind::Survival, 9.0),
            goal(2, "Earn enough to retire", GoalKind::Wealth, 8.0),
            goal(3, "Hide what he knows about the mafia", GoalKind::Conceal, 7.0),
        ];
    }

    if let Some(a) = actors.get_mut(&ActorId::new("courier")) {
        a.biography = "Pell is a nervous courier who runs errands for various merchants. He secretly works for the mafia, carrying messages and small packages. He is constantly worried about being caught.".into();
        a.goals = vec![
            goal(1, "Avoid getting caught", GoalKind::Safety, 10.0),
            goal(2, "Hide his mafia ties", GoalKind::Conceal, 9.0),
            goal(3, "Earn enough to leave town", GoalKind::Wealth, 7.0),
        ];
    }

    if let Some(a) = actors.get_mut(&ActorId::new("merchant_head")) {
        a.biography = "Selene Voss is the public head of the merchants' guild and secretly the leader of the mafia. She is charming, ambitious, and utterly ruthless. She views the town as a game to be won.".into();
        a.goals = vec![
            goal(1, "Consolidate power over the town", GoalKind::Ambition, 10.0),
            goal(2, "Expand mafia operations", GoalKind::Wealth, 9.0),
            goal(3, "Protect her secret identity", GoalKind::Conceal, 8.0),
            goal(4, "Maintain her public reputation", GoalKind::Status, 7.0),
        ];
    }

    if let Some(a) = actors.get_mut(&ActorId::new("mafia_lieutenant")) {
        a.biography = "Halden Rook is the mafia's dockmaster and Selene's right hand. He is brutal, efficient, and fiercely loyal to Selene. He handles the smuggling operations.".into();
        a.goals = vec![
            goal(1, "Execute Selene's orders", GoalKind::Duty, 9.0),
            goal(2, "Keep the shipments running", GoalKind::Wealth, 8.0),
            goal(3, "Eliminate loose ends", GoalKind::Safety, 7.0),
        ];
    }

    if let Some(a) = actors.get_mut(&ActorId::new("accountant")) {
        a.biography = "Corvin Ashe is the mafia's bookkeeper. He is meticulous, anxious, and deeply involved in the mafia's financial crimes. He knows too much, making him both valuable and a liability.".into();
        a.goals = vec![
            goal(1, "Keep himself useful and alive", GoalKind::Safety, 9.0),
            goal(2, "Manage the mafia's finances", GoalKind::Wealth, 8.0),
            goal(3, "Hide the ledgers", GoalKind::Conceal, 7.0),
        ];
    }

    if let Some(a) = actors.get_mut(&ActorId::new("spy")) {
        a.biography = "Nessa Quill is a mafia spy who poses as a simple shopper and gossip. She is quiet, observant, and gathers intelligence for Selene. She has no loyalty to anyone but herself.".into();
        a.goals = vec![
            goal(1, "Gather information on town figures", GoalKind::Investigate, 9.0),
            goal(2, "Maintain her cover as a gossip", GoalKind::Conceal, 8.0),
            goal(3, "Earn Selene's favor", GoalKind::Status, 6.0),
        ];
    }

    if let Some(a) = actors.get_mut(&ActorId::new("priest")) {
        a.biography = "Father Alden is the town's elderly, kind-hearted priest. He tends to the sick and the poor, offering comfort and advice. He knows many secrets from confessions but keeps them sacred.".into();
        a.goals = vec![
            goal(1, "Comfort the sick and poor", GoalKind::Duty, 9.0),
            goal(2, "Protect his congregation", GoalKind::Safety, 8.0),
            goal(3, "Keep confessions secret", GoalKind::Conceal, 10.0),
        ];
    }

    if let Some(a) = actors.get_mut(&ActorId::new("tavern_keeper")) {
        a.biography = "Bess Marlow owns the local tavern. She is boisterous, sharp-witted, and knows everyone's business. Her tavern is the town's gossip hub.".into();
        a.goals = vec![
            goal(1, "Keep the tavern thriving", GoalKind::Wealth, 8.0),
            goal(2, "Be the center of town gossip", GoalKind::Status, 7.0),
            goal(3, "Avoid trouble from the mafia", GoalKind::Safety, 6.0),
        ];
    }

    if let Some(a) = actors.get_mut(&ActorId::new("neighbor")) {
        a.biography = "Ivo Prat is a quiet, middle-aged neighbor who minds his own business. He lives near the plaza and often observes the daily life of the town without interfering.".into();
        a.goals = vec![
            goal(1, "Live peacefully", GoalKind::Survival, 7.0),
            goal(2, "Stay out of mafia affairs", GoalKind::Safety, 8.0),
        ];
    }

    if let Some(a) = actors.get_mut(&ActorId::new("child")) {
        a.biography = "Pip is an energetic child who plays in the plaza. Curious and friendly, they see the town as a big playground and notice more than adults expect.".into();
        a.goals = vec![
            goal(1, "Make friends and have fun", GoalKind::Affection, 8.0),
            goal(2, "Discover interesting things", GoalKind::Investigate, 5.0),
        ];
    }
}

fn sched(start: u32, end: u32, loc: &str, act: &str) -> ScheduleEntry {
    ScheduleEntry {
        start_min: start,
        end_min: end,
        location: LocationId::new(loc),
        activity: act.into(),
        public: true,
    }
}

/// Build the canonical Hidden Mafia town. Deterministic.
pub fn build_town() -> World {
    let mut nav = NavGrid::new(TOWN_WIDTH, TOWN_HEIGHT);

    // --- Locations (rectangles) -------------------------------------
    let mut locations: BTreeMap<LocationId, Location> = BTreeMap::new();
    let mut add_loc = |id: &str,
                       name: &str,
                       kind: LocationKind,
                       r: Rect,
                       owner: Option<&str>,
                       transmission: f32,
                       private: bool| {
        locations.insert(
            LocationId::new(id),
            Location {
                id: LocationId::new(id),
                name: name.into(),
                kind,
                bounds: r,
                owner: owner.map(FactionId::new),
                sound_transmission: transmission,
                private,
                tags: vec![],
            },
        );
    };

    add_loc(
        "plaza",
        "Market Plaza",
        LocationKind::Plaza,
        Rect::new(15.0, 8.0, 12.0, 8.0),
        None,
        1.3,
        false,
    );
    add_loc(
        "grocer",
        "Voss & Daughter Grocers",
        LocationKind::Shop,
        Rect::new(3.0, 3.0, 7.0, 4.0),
        Some("merchants"),
        1.0,
        false,
    );
    add_loc(
        "grocer_backroom",
        "Grocer Backroom",
        LocationKind::Backroom,
        Rect::new(3.0, 11.0, 6.0, 3.0),
        Some("merchants"),
        0.4,
        true,
    );
    add_loc(
        "warehouse",
        "Dockside Warehouse",
        LocationKind::Warehouse,
        Rect::new(29.0, 3.0, 8.0, 4.0),
        Some("merchants"),
        0.9,
        false,
    );
    add_loc(
        "guard_office",
        "Town Watch Office",
        LocationKind::GuardOffice,
        Rect::new(3.0, 18.0, 9.0, 3.0),
        Some("watch"),
        1.0,
        false,
    );
    add_loc(
        "alley",
        "Fisher's Alley",
        LocationKind::Alley,
        Rect::new(15.0, 2.0, 7.0, 5.0),
        None,
        1.1,
        false,
    );
    add_loc(
        "residence",
        "Rowan Residence",
        LocationKind::Residence,
        Rect::new(29.0, 17.0, 8.0, 3.0),
        None,
        0.7,
        true,
    );
    add_loc(
        "mafia_hall",
        "Abandoned Customs House",
        LocationKind::FactionHall,
        Rect::new(29.0, 11.0, 8.0, 3.0),
        Some("mafia"),
        0.6,
        true,
    );

    // --- Walls with doorways ----------------------------------------
    // Grocer: bottom door (6,8) and right door (11,5).
    wall_rect(
        &mut nav,
        2,
        2,
        10,
        7,
        &[GridPos::new(6, 8), GridPos::new(11, 5)],
    );
    // Backroom: right door (9,12).
    wall_rect(&mut nav, 2, 10, 8, 5, &[GridPos::new(9, 12)]);
    // Warehouse: left door (28,5).
    wall_rect(&mut nav, 28, 2, 10, 7, &[GridPos::new(28, 5)]);
    // Guard office: top door (7,17).
    wall_rect(&mut nav, 2, 17, 11, 5, &[GridPos::new(7, 17)]);
    // Residence: top door (32,16).
    wall_rect(&mut nav, 28, 16, 10, 6, &[GridPos::new(32, 16)]);
    // Customs house: top door (32,10).
    wall_rect(&mut nav, 28, 10, 10, 5, &[GridPos::new(32, 10)]);
    // A couple of free-standing alley walls for occlusion tests.
    for y in 2..6 {
        nav.set(GridPos::new(18, y), Tile::Wall);
    }

    // --- Items -------------------------------------------------------
    let mut items: BTreeMap<ItemId, Item> = BTreeMap::new();
    let mut add_item = |id: &str, name: &str, desc: &str, value: i64, tags: &[&str]| {
        items.insert(
            ItemId::new(id),
            Item {
                id: ItemId::new(id),
                name: name.into(),
                description: desc.into(),
                base_value: value,
                tags: tags.iter().map(|t| t.to_string()).collect(),
                stackable: false,
            },
        );
    };
    add_item(
        "ledger",
        "Shipping Ledger",
        "A grocer's shipping ledger with odd entries.",
        60,
        &["evidence"],
    );
    add_item(
        "medicine",
        "Fever Medicine",
        "Bitter but effective.",
        80,
        &["favor"],
    );
    add_item(
        "manifest",
        "Dock Manifest",
        "A manifest listing 'special' containers.",
        90,
        &["evidence"],
    );
    add_item(
        "invoice",
        "Overwritten Invoice",
        "Numbers that do not match the manifest.",
        70,
        &["evidence"],
    );
    add_item(
        "locket",
        "Family Locket",
        "A small keepsake.",
        40,
        &["personal"],
    );

    // --- Facts / information carriers --------------------------------
    let mut facts: BTreeMap<FactId, Fact> = BTreeMap::new();
    let mut add_fact = |id: &str,
                        statement: &str,
                        sensitivity: Sensitivity,
                        category: FactCategory,
                        prereqs: &[&str],
                        holders: &[&str],
                        containers: &[ContainerKind]| {
        facts.insert(
            FactId::new(id),
            Fact {
                id: FactId::new(id),
                statement: statement.into(),
                truth: true,
                sensitivity,
                category,
                prerequisites: prereqs.iter().map(|p| FactId::new(*p)).collect(),
                eligible_holders: holders.iter().map(|h| ActorId::new(*h)).collect(),
                eligible_containers: containers.to_vec(),
                committed: false,
                placed_at: None,
            },
        );
    };

    add_fact(
        "f_mafia_exists",
        "An underground organization controls some town shipments.",
        Sensitivity::Restricted,
        FactCategory::Organization,
        &[],
        &["guard_captain", "merchant_head", "spy", "mafia_lieutenant"],
        &[
            ContainerKind::Testimony,
            ContainerKind::Rumor,
            ContainerKind::OverheardRemark,
        ],
    );
    add_fact(
        "f_shipments_diverted",
        "Certain shipments are diverted at the docks before they reach the market.",
        Sensitivity::Secret,
        FactCategory::Shipment,
        &["f_mafia_exists"],
        &["dockworker", "courier", "spy", "mafia_lieutenant"],
        &[ContainerKind::Document, ContainerKind::Testimony],
    );
    add_fact(
        "f_courier_route",
        "A courier runs sealed crates between the docks and the old customs house.",
        Sensitivity::Secret,
        FactCategory::Shipment,
        &["f_mafia_exists"],
        &["courier", "dockworker"],
        &[
            ContainerKind::OverheardRemark,
            ContainerKind::Testimony,
            ContainerKind::Document,
        ],
    );
    add_fact(
        "f_warehouse_stash",
        "The abandoned customs house hides a stockpile of contraband.",
        Sensitivity::Secret,
        FactCategory::Location,
        &["f_shipments_diverted"],
        &["dockworker", "mafia_lieutenant", "accountant"],
        &[
            ContainerKind::Document,
            ContainerKind::Environmental,
            ContainerKind::Testimony,
        ],
    );
    add_fact(
        "f_accountant_link",
        "A bookkeeper named Corvin keeps two sets of books for the organization.",
        Sensitivity::Secret,
        FactCategory::Ledger,
        &["f_shipments_diverted"],
        &["accountant", "merchant_head", "mafia_lieutenant"],
        &[ContainerKind::Document, ContainerKind::Testimony],
    );
    add_fact(
        "f_lieutenant_identity",
        "The quiet dockmaster, Halden Rook, is the organization's lieutenant.",
        Sensitivity::InnerCircle,
        FactCategory::Identity,
        &["f_warehouse_stash"],
        &["mafia_lieutenant", "accountant"],
        &[
            ContainerKind::Testimony,
            ContainerKind::Document,
            ContainerKind::OverheardRemark,
        ],
    );
    add_fact(
        "f_leader_identity",
        "The mafia's true leader is the merchant guild head, Selene Voss.",
        Sensitivity::InnerCircle,
        FactCategory::Identity,
        &["f_lieutenant_identity"],
        &["mafia_lieutenant", "accountant", "spy"],
        &[ContainerKind::Testimony, ContainerKind::Document],
    );
    add_fact(
        "f_grocer_ledger",
        "The grocer's ledger records unusual payments from the docks.",
        Sensitivity::Private,
        FactCategory::Ledger,
        &[],
        &["grocer"],
        &[ContainerKind::Document, ContainerKind::Testimony],
    );
    add_fact(
        "f_grocer_innocent",
        "The grocer is a victim of the racket, not a member of it.",
        Sensitivity::Private,
        FactCategory::Motive,
        &["f_grocer_ledger"],
        &["grocer", "grocer_daughter"],
        &[ContainerKind::Testimony, ContainerKind::Document],
    );
    add_fact(
        "f_daughter_ill",
        "The grocer's daughter is ill and needs medicine.",
        Sensitivity::Public,
        FactCategory::Motive,
        &[],
        &["grocer", "grocer_daughter", "priest"],
        &[ContainerKind::Testimony, ContainerKind::OverheardRemark],
    );

    // --- Story: anchors + flexible checkpoints ----------------------
    let anchors = vec![
        StoryAnchor {
            id: "mafia_exists".into(),
            statement: "A criminal organization exists in the town.".into(),
            immutable: true,
        },
        StoryAnchor {
            id: "mafia_goals".into(),
            statement: "It seeks to control shipments and evade the Watch.".into(),
            immutable: true,
        },
        StoryAnchor {
            id: "leader_seeded".into(),
            statement: "The leader's identity is fixed before evidence is generated.".into(),
            immutable: true,
        },
    ];
    let checkpoints = vec![
        Checkpoint {
            id: "cp_notice".into(),
            description: "The player learns the organization exists.".into(),
            requires: vec![FactId::new("f_mafia_exists")],
            predicate: None,
            state: CheckpointState::Dormant,
            candidate_facts: vec![
                FactId::new("f_shipments_diverted"),
                FactId::new("f_courier_route"),
            ],
        },
        Checkpoint {
            id: "cp_trace".into(),
            description: "The player traces the diversion to a place or a bookkeeper.".into(),
            requires: vec![FactId::new("f_shipments_diverted")],
            predicate: None,
            state: CheckpointState::Dormant,
            candidate_facts: vec![
                FactId::new("f_warehouse_stash"),
                FactId::new("f_accountant_link"),
            ],
        },
        Checkpoint {
            id: "cp_lieutenant".into(),
            description: "The player identifies a senior intermediary.".into(),
            requires: vec![FactId::new("f_warehouse_stash")],
            predicate: None,
            state: CheckpointState::Dormant,
            candidate_facts: vec![FactId::new("f_lieutenant_identity")],
        },
        Checkpoint {
            id: "cp_leader".into(),
            description: "The player identifies the leader.".into(),
            requires: vec![FactId::new("f_lieutenant_identity")],
            predicate: None,
            state: CheckpointState::Dormant,
            candidate_facts: vec![FactId::new("f_leader_identity")],
        },
    ];
    let mut fixed_answer = BTreeMap::new();
    fixed_answer.insert(LEADER_KEY.to_string(), "merchant_head".to_string());

    let story = StoryState {
        mode: MysteryMode::FixedAnswer,
        anchors,
        checkpoints,
        fixed_answer,
        committed_containers: BTreeMap::new(),
    };

    // --- Factions ----------------------------------------------------
    let mut factions: BTreeMap<FactionId, Faction> = BTreeMap::new();
    factions.insert(
        FactionId::new("watch"),
        Faction {
            id: FactionId::new("watch"),
            name: "Town Watch".into(),
            kind: FactionKind::Government,
            budget: 800,
            staff: 4,
            legitimacy: 0.9,
            public: true,
            members: vec![ActorId::new("guard_captain"), ActorId::new("guard_rookie")],
            responsible_for: vec![LocationId::new("guard_office"), LocationId::new("plaza")],
            priorities: vec![
                FactionPriority {
                    description: "repair the plaza fountain".into(),
                    weight: 2.0,
                    cost: 150,
                    predicate: None,
                    resolved: false,
                },
                FactionPriority {
                    description: "investigate the docks".into(),
                    weight: 5.0,
                    cost: 300,
                    predicate: None,
                    resolved: false,
                },
            ],
        },
    );
    factions.insert(
        FactionId::new("merchants"),
        Faction {
            id: FactionId::new("merchants"),
            name: "Merchants' Guild".into(),
            kind: FactionKind::Business,
            budget: 600,
            staff: 6,
            legitimacy: 0.8,
            public: true,
            members: vec![
                ActorId::new("merchant_head"),
                ActorId::new("grocer"),
                ActorId::new("dockworker"),
            ],
            responsible_for: vec![LocationId::new("warehouse"), LocationId::new("grocer")],
            priorities: vec![FactionPriority {
                description: "repair warehouse loading door".into(),
                weight: 3.0,
                cost: 120,
                predicate: None,
                resolved: false,
            }],
        },
    );
    factions.insert(
        FactionId::new("mafia"),
        Faction {
            id: FactionId::new("mafia"),
            name: "The Quiet Trade".into(),
            kind: FactionKind::Gang,
            budget: 900,
            staff: 5,
            legitimacy: 0.1,
            public: false,
            members: vec![
                ActorId::new("merchant_head"),
                ActorId::new("mafia_lieutenant"),
                ActorId::new("accountant"),
                ActorId::new("spy"),
            ],
            responsible_for: vec![LocationId::new("mafia_hall")],
            priorities: vec![FactionPriority {
                description: "relocate the contraband".into(),
                weight: 6.0,
                cost: 400,
                predicate: None,
                resolved: false,
            }],
        },
    );
    factions.insert(
        FactionId::new("temple"),
        Faction {
            id: FactionId::new("temple"),
            name: "Chapel of the Bell".into(),
            kind: FactionKind::Charity,
            budget: 200,
            staff: 2,
            legitimacy: 0.95,
            public: true,
            members: vec![ActorId::new("priest")],
            responsible_for: vec![],
            priorities: vec![],
        },
    );

    // --- Actors ------------------------------------------------------
    let mut actors: BTreeMap<ActorId, Actor> = BTreeMap::new();

    let mut player = npc(
        PLAYER,
        "You",
        "plaza",
        Vec2::new(20.0, 12.0),
        Personality::default(),
        &[],
        &["player"],
        None,
        vec![],
    );
    player.kind = ActorKind::Player;
    player.inventory.money = 250;
    actors.insert(player.id.clone(), player);

    let greedy = |mut p: Personality| {
        p.tags.push("greedy".into());
        p
    };
    let kind = |mut p: Personality| {
        p.tags.push("protective".into());
        p
    };

    actors.insert(ActorId::new("grocer"), {
        let mut a = npc(
            "grocer",
            "Marta Voss",
            "grocer",
            Vec2::new(4.5, 4.5),
            kind(Personality {
                agreeableness: 0.7,
                neuroticism: 0.6,
                openness: 0.4,
                ..Default::default()
            }),
            &["merchants"],
            &["shopkeeper"],
            None,
            vec![
                sched(360, 720, "grocer", "tending the shop"),
                sched(720, 780, "plaza", "buying supplies"),
                sched(780, 1080, "grocer", "tending the shop"),
            ],
        );
        a.capabilities.can_trade = true;
        a.inventory.add(ItemId::new("ledger"), 1);
        a
    });
    actors.insert(
        ActorId::new("grocer_daughter"),
        npc(
            "grocer_daughter",
            "Lena Voss",
            "grocer_backroom",
            Vec2::new(4.0, 12.0),
            Personality {
                agreeableness: 0.8,
                ..Default::default()
            },
            &["merchants"],
            &["sick"],
            None,
            vec![sched(0, 1440, "grocer_backroom", "resting")],
        ),
    );
    actors.insert(ActorId::new("guard_captain"), {
        let mut a = npc(
            "guard_captain",
            "Captain Ilse Renn",
            "guard_office",
            Vec2::new(7.0, 19.5),
            Personality {
                conscientiousness: 0.85,
                agreeableness: 0.5,
                ..Default::default()
            },
            &["watch"],
            &["authority", "guard"],
            None,
            vec![
                sched(360, 720, "guard_office", "on duty"),
                sched(720, 840, "plaza", "patrolling"),
                sched(840, 1200, "guard_office", "on duty"),
            ],
        );
        a.capabilities.can_arrest = true;
        a.capabilities.can_repair = true;
        a.capabilities.can_allocate_budget = true;
        a
    });
    actors.insert(ActorId::new("guard_rookie"), {
        let mut a = npc(
            "guard_rookie",
            "Deputy Bram",
            "plaza",
            Vec2::new(21.0, 12.0),
            Personality {
                agreeableness: 0.7,
                conscientiousness: 0.4,
                ..Default::default()
            },
            &["watch"],
            &["authority", "guard"],
            None,
            vec![sched(360, 1200, "plaza", "patrolling")],
        );
        a.capabilities.can_arrest = true;
        a
    });
    actors.insert(
        ActorId::new("dockworker"),
        npc(
            "dockworker",
            "Old Tarn",
            "warehouse",
            Vec2::new(31.0, 5.0),
            Personality {
                extraversion: 0.3,
                agreeableness: 0.6,
                ..Default::default()
            },
            &["merchants"],
            &["dockworker"],
            None,
            vec![
                sched(300, 800, "warehouse", "unloading"),
                sched(800, 1020, "plaza", "drinking"),
            ],
        ),
    );
    actors.insert(
        ActorId::new("courier"),
        npc(
            "courier",
            "Pell",
            "alley",
            Vec2::new(16.0, 4.0),
            Personality {
                extraversion: 0.4,
                neuroticism: 0.6,
                ..Default::default()
            },
            &["merchants"],
            &["courier", "runs_errands"],
            Some("mafia"),
            vec![sched(420, 900, "alley", "waiting for crates")],
        ),
    );
    actors.insert(ActorId::new("merchant_head"), {
        // The secret leader. Publicly a respectable guild head.
        let mut a = npc(
            "merchant_head",
            "Selene Voss",
            "plaza",
            Vec2::new(23.0, 10.0),
            greedy(Personality {
                extraversion: 0.8,
                agreeableness: 0.4,
                ..Default::default()
            }),
            &["merchants"],
            &["guild_head", "inner_circle", "mafia"],
            Some("mafia"),
            vec![
                sched(360, 700, "plaza", "holding court"),
                sched(700, 900, "warehouse", "inspecting cargo"),
                sched(900, 1100, "plaza", "holding court"),
            ],
        );
        a.capabilities.can_allocate_budget = true;
        a.capabilities.can_hire = true;
        a
    });
    actors.insert(ActorId::new("mafia_lieutenant"), {
        let mut a = npc(
            "mafia_lieutenant",
            "Halden Rook",
            "mafia_hall",
            Vec2::new(31.0, 13.0),
            Personality {
                conscientiousness: 0.7,
                agreeableness: 0.3,
                neuroticism: 0.3,
                ..Default::default()
            },
            &["merchants", "mafia"],
            &["dockmaster", "inner_circle", "mafia"],
            Some("mafia"),
            vec![sched(400, 1000, "mafia_hall", "overseeing")],
        );
        a.capabilities.can_allocate_budget = true;
        a
    });
    actors.insert(
        ActorId::new("accountant"),
        npc(
            "accountant",
            "Corvin Ashe",
            "mafia_hall",
            Vec2::new(29.0, 12.0),
            Personality {
                conscientiousness: 0.9,
                neuroticism: 0.7,
                ..Default::default()
            },
            &["merchants", "mafia"],
            &["bookkeeper", "inner_circle", "mafia"],
            Some("mafia"),
            vec![sched(420, 960, "mafia_hall", "keeping books")],
        ),
    );
    actors.insert(
        ActorId::new("spy"),
        npc(
            "spy",
            "Nessa Quill",
            "plaza",
            Vec2::new(19.0, 14.0),
            Personality {
                extraversion: 0.3,
                openness: 0.7,
                ..Default::default()
            },
            &["mafia"],
            &["spy", "mafia"],
            Some("mafia"),
            vec![
                sched(360, 720, "plaza", "watching the market"),
                sched(720, 960, "grocer", "shopping"),
                sched(960, 1200, "plaza", "watching the market"),
            ],
        ),
    );
    actors.insert(
        ActorId::new("priest"),
        npc(
            "priest",
            "Father Alden",
            "plaza",
            Vec2::new(17.0, 10.0),
            Personality {
                agreeableness: 0.9,
                openness: 0.7,
                ..Default::default()
            },
            &["temple"],
            &["clergy"],
            None,
            vec![sched(360, 1200, "plaza", "comforting the sick")],
        ),
    );
    actors.insert(
        ActorId::new("tavern_keeper"),
        npc(
            "tavern_keeper",
            "Bess Marlow",
            "plaza",
            Vec2::new(24.0, 14.0),
            Personality {
                extraversion: 0.9,
                agreeableness: 0.6,
                ..Default::default()
            },
            &["merchants"],
            &["tavern_keeper", "gossip"],
            None,
            vec![sched(300, 1200, "plaza", "running the tavern")],
        ),
    );
    actors.insert(
        ActorId::new("neighbor"),
        npc(
            "neighbor",
            "Ivo Prat",
            "residence",
            Vec2::new(32.0, 19.0),
            Personality {
                agreeableness: 0.5,
                neuroticism: 0.5,
                ..Default::default()
            },
            &[],
            &["neighbor"],
            None,
            vec![sched(360, 1080, "residence", "minding the house")],
        ),
    );
    actors.insert(
        ActorId::new("child"),
        npc(
            "child",
            "Pip",
            "plaza",
            Vec2::new(16.0, 15.0),
            Personality {
                openness: 0.9,
                extraversion: 0.7,
                ..Default::default()
            },
            &[],
            &["child"],
            None,
            vec![sched(360, 1080, "plaza", "playing")],
        ),
    );

    // Populate biographies and goals for named NPCs.
    populate_npc_lore(&mut actors);

    // --- Initial beliefs --------------------------------------------
    // Public facts everyone knows.
    for a in actors.values_mut() {
        a.beliefs.insert(
            FactId::new("f_daughter_ill"),
            Belief {
                fact: FactId::new("f_daughter_ill"),
                believes_true: true,
                confidence: 0.8,
                source: Provenance::Assumed,
                acquired_tick: 0,
            },
        );
    }
    // Organization members know their own secrets.
    let inner: Vec<ActorId> = vec![
        ActorId::new("merchant_head"),
        ActorId::new("mafia_lieutenant"),
        ActorId::new("accountant"),
        ActorId::new("spy"),
    ];
    for id in &inner {
        if let Some(a) = actors.get_mut(id) {
            for f in [
                "f_mafia_exists",
                "f_shipments_diverted",
                "f_courier_route",
                "f_warehouse_stash",
                "f_accountant_link",
                "f_lieutenant_identity",
                "f_leader_identity",
            ] {
                a.beliefs.insert(
                    FactId::new(f),
                    Belief {
                        fact: FactId::new(f),
                        believes_true: true,
                        confidence: 0.95,
                        source: Provenance::Assumed,
                        acquired_tick: 0,
                    },
                );
            }
        }
    }
    // The grocer privately knows their own ledger is suspicious.
    if let Some(a) = actors.get_mut(&ActorId::new("grocer")) {
        a.beliefs.insert(
            FactId::new("f_grocer_ledger"),
            Belief {
                fact: FactId::new("f_grocer_ledger"),
                believes_true: true,
                confidence: 0.9,
                source: Provenance::Assumed,
                acquired_tick: 0,
            },
        );
    }
    // The dockworker and courier know operational details.
    if let Some(a) = actors.get_mut(&ActorId::new("dockworker")) {
        for f in [
            "f_mafia_exists",
            "f_shipments_diverted",
            "f_courier_route",
            "f_warehouse_stash",
        ] {
            a.beliefs.insert(
                FactId::new(f),
                Belief {
                    fact: FactId::new(f),
                    believes_true: true,
                    confidence: 0.8,
                    source: Provenance::Assumed,
                    acquired_tick: 0,
                },
            );
        }
    }
    if let Some(a) = actors.get_mut(&ActorId::new("courier")) {
        for f in ["f_mafia_exists", "f_shipments_diverted", "f_courier_route"] {
            a.beliefs.insert(
                FactId::new(f),
                Belief {
                    fact: FactId::new(f),
                    believes_true: true,
                    confidence: 0.8,
                    source: Provenance::Assumed,
                    acquired_tick: 0,
                },
            );
        }
    }
    // The captain suspects something is wrong.
    if let Some(a) = actors.get_mut(&ActorId::new("guard_captain")) {
        a.beliefs.insert(
            FactId::new("f_mafia_exists"),
            Belief {
                fact: FactId::new("f_mafia_exists"),
                believes_true: true,
                confidence: 0.6,
                source: Provenance::Assumed,
                acquired_tick: 0,
            },
        );
    }

    // Physical items hidden in the town that a search can recover. This makes
    // the grocer's medicine-for-information quest completable in-world.
    let mut location_items: BTreeMap<LocationId, Vec<rpg_domain::world::ItemStack>> =
        BTreeMap::new();
    for loc in ["warehouse", "alley"] {
        location_items.insert(
            LocationId::new(loc),
            vec![rpg_domain::world::ItemStack {
                item: ItemId::new("medicine"),
                qty: 1,
            }],
        );
    }

    World {
        tick: 0,
        day: 0,
        minute_of_day: 8 * 60,
        locations,
        actors,
        items,
        factions,
        facts,
        nav,
        story,
        commitments: BTreeMap::new(),
        offers: BTreeMap::new(),
        rumors: Vec::new(),
        damage: BTreeMap::new(),
        location_items,
        flags: BTreeSet::new(),
        ledger: BTreeMap::new(),
        next_commitment_id: 1,
        next_offer_id: 1,
        next_rumor_id: 1,
        next_goal_id: 1,
        lore: WorldLore {
            settlement_name: "Ashwick".into(),
            geography: "A modest river town at the edge of the Northern Marches. The river bends around a rocky promontory, providing a natural harbor for small boats.".into(),
            history: "Founded as a trading post three centuries ago. The old fortifications are mostly ruins now, but the town has grown into a regional hub for grain and timber.".into(),
            culture: "Practical and tight-knit. People value hard work and discretion. Strangers are welcomed cautiously. Gossip travels fast, but overt curiosity is considered rude.".into(),
            landmarks: vec![
                "The Market Plaza, with its old stone well and weekly market".into(),
                "The Temple of the Weeping Saint, a modest stone building near the plaza".into(),
                "The Guard Office, a squat brick building by the north gate".into(),
                "The Docks, where river barges unload grain and timber".into(),
                "The Mafia Hall, a discreet building in the warehouse district".into(),
            ],
            economy: "Grain, timber, and river trade. The merchants' guild dominates local commerce. The mafia controls smuggling and protection rackets.".into(),
            institutions: "The Watch maintains order and answers to the town council. The Temple provides charity and spiritual guidance. The merchants' guild regulates trade and prices.".into(),
            recent_events: "A recent warehouse fire destroyed a shipment of grain. A courier went missing last month. The mafia is rumored to be expanding its operations.".into(),
            everyday_knowledge: "Bread is baked daily at the plaza bakery. The river is safe to cross at the ford, but not at the docks after dark. The Watch patrols the plaza and the main streets.".into(),
        },
    }
}

/// The seeded mafia leader, or `None` if the scenario is not seeded.
pub fn seeded_leader(world: &World) -> Option<ActorId> {
    world.story.fixed_answer.get(LEADER_KEY).map(ActorId::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn town_has_expected_scale() {
        let w = build_town();
        assert!(
            (6..=9).contains(&w.locations.len()),
            "locations: {}",
            w.locations.len()
        );
        let named = w.named_npcs().count();
        assert!((12..=16).contains(&named), "named npcs: {named}");
        assert!(w.factions.len() >= 3);
        assert!(w.facts.len() >= 8);
    }

    #[test]
    fn leader_is_seeded_and_inner_circle() {
        let w = build_town();
        let leader = seeded_leader(&w).expect("leader seeded");
        let a = w.actor(&leader).unwrap();
        assert!(a.has_flag("inner_circle"));
        assert!(a.secret_allegiance.is_some());
    }

    #[test]
    fn grocer_is_not_seeded_as_leader() {
        let w = build_town();
        let leader = seeded_leader(&w).unwrap();
        assert_ne!(leader.as_str(), "grocer");
    }

    #[test]
    fn all_locations_are_reachable_from_plaza() {
        let w = build_town();
        let start = rpg_domain::geom::to_grid(Vec2::new(20.0, 12.0));
        for (id, l) in &w.locations {
            let goal = rpg_domain::geom::to_grid(l.center());
            assert!(
                rpg_sim::pathfind::astar(&w.nav, start, goal).is_some(),
                "location {id} unreachable"
            );
        }
    }

    #[test]
    fn deterministic_builds_are_identical() {
        let a = build_town();
        let b = build_town();
        assert_eq!(a.actors.len(), b.actors.len());
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap()
        );
    }
}
