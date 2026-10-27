//! Regression tests for deterministic goal-driven autonomy (PR #7).
//!
//! NPCs must pursue their goals independently — walk to a goal target and
//! act on arrival — without player interaction and without any LLM calls.
//! These tests drive the real `Simulation::step` path, mirroring the
//! canonical town's layout (Guard Office → Dockside Warehouse) so the
//! pursuit deadlines are verified against actual travel time.

use rpg_domain::action::ActionProposal;
use rpg_domain::geom::{GridPos, Rect, Vec2};
use rpg_domain::ids::*;
use rpg_domain::world::{
    Actor, ActorKind, Capabilities, Goal, GoalKind, GoalStatus, Location, LocationKind, NavGrid,
    Personality, ScheduleEntry, StoryState, Tile, World,
};
use rpg_sim::{SimConfig, Simulation};
use std::collections::{BTreeMap, BTreeSet};

fn loc(id: &str, name: &str, kind: LocationKind, x: f32, y: f32, w: f32, h: f32) -> Location {
    Location {
        id: LocationId::new(id),
        name: name.into(),
        kind,
        bounds: Rect::new(x, y, w, h),
        owner: None,
        sound_transmission: 1.0,
        private: false,
        tags: vec![],
    }
}

fn npc(
    id: &str,
    location: &str,
    pos: Vec2,
    schedule: Vec<ScheduleEntry>,
    goals: Vec<Goal>,
) -> Actor {
    Actor {
        id: ActorId::new(id),
        name: id.into(),
        kind: ActorKind::Npc,
        status: rpg_domain::world::ActorStatus::Alive,
        location: LocationId::new(location),
        pos,
        facing: 0.0,
        personality: Personality::default(),
        values: vec![],
        biography: String::new(),
        private_biography: String::new(),
        affiliations: vec![],
        capabilities: Capabilities::default(),
        inventory: Default::default(),
        goals,
        schedule,
        relationships: BTreeMap::new(),
        beliefs: BTreeMap::new(),
        inspected: BTreeSet::new(),
        flags: BTreeSet::new(),
        secret_allegiance: None,
        in_private_conversation_with: None,
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

fn goal_t(
    id: u64,
    description: &str,
    kind: GoalKind,
    priority: f32,
    target_location: Option<&str>,
    target_actor: Option<&str>,
) -> Goal {
    Goal {
        id,
        description: description.into(),
        kind,
        priority,
        status: GoalStatus::Active,
        predicate: None,
        target_location: target_location.map(LocationId::new),
        target_actor: target_actor.map(ActorId::new),
    }
}

/// A test world with the canonical town's layout scale: the Guard Office is
/// ~30 world units from the Dockside Warehouse, so a pursuit must survive
/// several ticks of walking. `slow` widens the walk (default 1.0).
fn autonomy_world() -> World {
    let mut w = World {
        tick: 0,
        day: 0,
        minute_of_day: 480,
        locations: BTreeMap::new(),
        actors: BTreeMap::new(),
        items: BTreeMap::new(),
        factions: BTreeMap::new(),
        facts: BTreeMap::new(),
        nav: NavGrid::new(40, 26),
        story: StoryState {
            mode: rpg_domain::world::MysteryMode::FixedAnswer,
            anchors: vec![rpg_domain::world::StoryAnchor {
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
        lore: Default::default(),
    };
    w.locations.insert(
        LocationId::new("plaza"),
        loc(
            "plaza",
            "Market Plaza",
            LocationKind::Plaza,
            15.0,
            8.0,
            12.0,
            8.0,
        ),
    );
    w.locations.insert(
        LocationId::new("shop"),
        loc(
            "shop",
            "Voss & Daughter",
            LocationKind::Shop,
            3.0,
            3.0,
            7.0,
            4.0,
        ),
    );
    w.locations.insert(
        LocationId::new("backroom"),
        loc(
            "backroom",
            "Backroom",
            LocationKind::Backroom,
            3.0,
            11.0,
            6.0,
            3.0,
        ),
    );
    w.locations.insert(
        LocationId::new("warehouse"),
        loc(
            "warehouse",
            "Dockside Warehouse",
            LocationKind::Warehouse,
            29.0,
            3.0,
            8.0,
            4.0,
        ),
    );
    w.locations.insert(
        LocationId::new("office"),
        loc(
            "office",
            "Town Watch Office",
            LocationKind::GuardOffice,
            3.0,
            18.0,
            9.0,
            3.0,
        ),
    );
    w.actors.insert(
        ActorId::new(PLAYER),
        npc(PLAYER, "plaza", Vec2::new(21.0, 12.0), vec![], vec![]),
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
    Simulation::new(autonomy_world(), 7, config)
}

#[test]
fn captain_ren_pursues_investigate_goal_without_player_interaction() {
    let mut s = sim();
    let mut renn = npc(
        "guard_captain",
        "office",
        Vec2::new(7.0, 19.5),
        vec![sched(360, 720, "office", "on duty")],
        vec![goal_t(
            2,
            "Find evidence against the mafia",
            GoalKind::Investigate,
            8.0,
            Some("warehouse"),
            None,
        )],
    );
    renn.capabilities.speed = 1.5;
    s.world.actors.insert(renn.id.clone(), renn);

    // No player interaction at all: just step the simulation.
    let mut visited = false;
    let mut committed_tick = None;
    let commit_tick = s.world.tick;
    for _ in 0..120 {
        s.step(Vec::new());
        let a = s.world.actor(&ActorId::new("guard_captain")).unwrap();
        if a.location == LocationId::new("warehouse") {
            visited = true;
        }
        if a.inspected.contains("warehouse") && committed_tick.is_none() {
            committed_tick = Some(s.world.tick);
        }
        // After the investigation the pursuit expires, which must start the
        // cooldown; keep stepping until we observe it.
        if committed_tick.is_some() && s.autonomy.is_cooling(&a.id) {
            break;
        }
    }
    assert!(
        visited,
        "Renn should walk to the warehouse autonomously (no player interaction)"
    );
    let committed = committed_tick.expect("Renn should investigate the warehouse on arrival");
    let travel = committed - commit_tick;
    // Empirical deadline check: the travel-scaled pursuit deadline (path
    // length / budget + margin) must cover the whole walk — the ~30-unit
    // haul takes ~11 ticks at 3.0 units/tick, and the pursuit must never
    // expire mid-walk (no yo-yoing back to the scheduled post).
    assert!(
        travel < 20,
        "travel took {travel} ticks; deadline should track the walk"
    );
    assert!(
        s.autonomy.is_cooling(&ActorId::new("guard_captain")),
        "completed pursuit should be followed by a cooldown"
    );
}

#[test]
fn marta_pursues_protect_goal_toward_target_actor() {
    let mut s = sim();
    let mut marta = npc(
        "grocer",
        "shop",
        Vec2::new(4.5, 4.5),
        vec![sched(360, 720, "shop", "tending the shop")],
        vec![goal_t(
            1,
            "Keep Lena healthy and safe",
            GoalKind::Protect,
            10.0,
            None,
            Some("grocer_daughter"),
        )],
    );
    marta.capabilities.speed = 1.5;
    s.world.actors.insert(marta.id.clone(), marta);
    s.world.actors.insert(
        ActorId::new("grocer_daughter"),
        npc(
            "grocer_daughter",
            "backroom",
            Vec2::new(4.0, 12.0),
            vec![sched(0, 1440, "backroom", "resting")],
            vec![],
        ),
    );

    let mut visited = false;
    for _ in 0..40 {
        s.step(Vec::new());
        if s.world
            .actor(&ActorId::new("grocer"))
            .map(|a| a.location == LocationId::new("backroom"))
            .unwrap_or(false)
        {
            visited = true;
        }
    }
    assert!(
        visited,
        "Marta should walk to Lena's location autonomously (no player interaction)"
    );
}

#[test]
fn investigated_goal_is_not_re_pursued_yo_yo_prevention() {
    let mut s = sim();
    let mut renn = npc(
        "guard_captain",
        "office",
        Vec2::new(7.0, 19.5),
        vec![sched(360, 720, "office", "on duty")],
        vec![goal_t(
            2,
            "Find evidence against the mafia",
            GoalKind::Investigate,
            8.0,
            Some("warehouse"),
            None,
        )],
    );
    renn.capabilities.speed = 1.5;
    s.world.actors.insert(renn.id.clone(), renn);

    // Run past arrival, investigation, pursuit expiry and the cooldown.
    let mut investigated = false;
    for _ in 0..200 {
        s.step(Vec::new());
        if s.world
            .actor(&ActorId::new("guard_captain"))
            .map(|a| a.inspected.contains("warehouse"))
            .unwrap_or(false)
        {
            investigated = true;
        }
    }
    assert!(investigated, "Renn should have investigated the warehouse");
    let renn = s.world.actor(&ActorId::new("guard_captain")).unwrap();
    // Yo-yo prevention: the already-inspected goal is never re-selected, so
    // Renn settles back into her scheduled post instead of churning between
    // the office and the warehouse forever.
    assert!(
        !s.autonomy.is_pursuing(&renn.id),
        "a satisfied investigate goal must not be re-pursued"
    );
    assert!(
        renn.location == LocationId::new("office"),
        "Renn should have returned to her scheduled post, got {}",
        renn.location
    );
}

#[test]
fn autonomy_never_redirects_an_npc_near_the_player() {
    let mut s = sim();
    // Player stands within talk_radius of the NPC.
    if let Some(p) = s.world.actor_mut(&ActorId::new(PLAYER)) {
        p.pos = Vec2::new(8.0, 19.5);
        p.location = LocationId::new("office");
    }
    let mut renn = npc(
        "guard_captain",
        "office",
        Vec2::new(7.0, 19.5),
        vec![sched(360, 720, "office", "on duty")],
        vec![goal_t(
            2,
            "Find evidence against the mafia",
            GoalKind::Investigate,
            8.0,
            Some("warehouse"),
            None,
        )],
    );
    renn.capabilities.speed = 1.5;
    s.world.actors.insert(renn.id.clone(), renn);

    for _ in 0..40 {
        s.step(Vec::new());
    }
    let renn = s.world.actor(&ActorId::new("guard_captain")).unwrap();
    assert!(
        renn.location == LocationId::new("office"),
        "conversation preservation: an NPC within talk range must not be sent away, got {}",
        renn.location
    );
    assert!(!s.autonomy.is_pursuing(&renn.id));
}

#[test]
fn pursuit_expires_and_returns_the_actor_to_routine() {
    let mut s = sim();
    // A goal whose target is unreachable: pursuit is proposed but the walk
    // cannot complete; the deadline must expire it deterministically.
    let mut renn = npc(
        "guard_captain",
        "office",
        Vec2::new(7.0, 19.5),
        vec![sched(360, 720, "office", "on duty")],
        vec![goal_t(
            2,
            "Find evidence against the mafia",
            GoalKind::Investigate,
            8.0,
            Some("sealed_room"),
            None,
        )],
    );
    renn.capabilities.speed = 1.5;
    s.world.actors.insert(renn.id.clone(), renn);
    s.world.locations.insert(
        LocationId::new("sealed_room"),
        loc(
            "sealed_room",
            "Sealed",
            LocationKind::Backroom,
            30.0,
            18.0,
            4.0,
            4.0,
        ),
    );
    for y in 16..24 {
        s.world.nav.set(GridPos::new(28, y), Tile::Wall);
    }

    for _ in 0..100 {
        s.step(Vec::new());
    }
    let renn = s.world.actor(&ActorId::new("guard_captain")).unwrap();
    assert!(
        renn.location == LocationId::new("office"),
        "an unreachable pursuit must expire and return Renn to routine, got {}",
        renn.location
    );
    assert!(!s.autonomy.is_pursuing(&renn.id));
}

#[test]
fn autonomy_is_disabled_by_config() {
    let config = SimConfig {
        enable_background: false,
        enable_director: false,
        enable_goal_autonomy: false,
        minutes_per_tick: 1,
        ..SimConfig::default()
    };
    let mut s = Simulation::new(autonomy_world(), 7, config);
    let mut renn = npc(
        "guard_captain",
        "office",
        Vec2::new(7.0, 19.5),
        vec![sched(360, 720, "office", "on duty")],
        vec![goal_t(
            2,
            "Find evidence against the mafia",
            GoalKind::Investigate,
            8.0,
            Some("warehouse"),
            None,
        )],
    );
    renn.capabilities.speed = 1.5;
    s.world.actors.insert(renn.id.clone(), renn);

    for _ in 0..40 {
        s.step(Vec::new());
    }
    let renn = s.world.actor(&ActorId::new("guard_captain")).unwrap();
    assert!(
        renn.location == LocationId::new("office"),
        "with autonomy disabled, routines must not be goal-driven"
    );
    assert!(!s.autonomy.is_pursuing(&renn.id));
    assert_eq!(
        s.autonomy_proposals(&BTreeSet::new()),
        Vec::<ActionProposal>::new()
    );
}
