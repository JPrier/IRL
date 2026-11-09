//! Regression tests for deterministic goal-driven autonomy (PR #7).
//!
//! NPCs must pursue their goals independently — walk to a goal target and
//! act on arrival — without player interaction and without any LLM calls.
//! These tests drive the real `Simulation::step` path, mirroring the
//! canonical town's layout (Guard Office → Dockside Warehouse) so the
//! pursuit deadlines are verified against actual travel time.

use rpg_domain::geom::{GridPos, Rect, Vec2};
use rpg_domain::ids::*;
use rpg_domain::world::{
    Actor, ActorKind, ActorStatus, Capabilities, Goal, GoalKind, GoalStatus, Location,
    LocationKind, NavGrid, Personality, ScheduleEntry, StoryState, Tile, World,
};
use rpg_sim::autonomy::AutonomyProposal;
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
        bargain: None,
        needs: Default::default(),
        emotions: Default::default(),
        rel_memory: BTreeMap::new(),
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
        need_boost: 0.0,

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

/// The tended record (and any in-flight pursuit) is transient-but-persisted
/// state: a save taken mid-autonomy must restore to bit-identical
/// continuation, and a save written before the field existed must still load.
#[test]
fn autonomy_state_round_trips_and_old_saves_still_load() {
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

    // Run until the check-in has happened and the tend site is recorded.
    let mut tended = false;
    for _ in 0..60 {
        s.step(Vec::new());
        if s.autonomy
            .tended
            .get(&ActorId::new("grocer"))
            .is_some_and(|m| m.contains_key(&1))
        {
            tended = true;
            break;
        }
    }
    assert!(tended, "the check-in must record the tend site");

    // Round-trip the transient state through JSON and continue both
    // simulations: they must stay bit-identical.
    let json = serde_json::to_string(&s.transient()).unwrap();
    let transient: rpg_sim::SimTransient = serde_json::from_str(&json).unwrap();
    let mut restored = rpg_sim::Simulation::restore_full(
        s.seed,
        s.world.clone(),
        s.events.clone(),
        s.rng.clone(),
        transient,
        s.ops.clone(),
        s.initial_world.clone(),
        s.config.clone(),
    );
    for _ in 0..15 {
        s.step(Vec::new());
        restored.step(Vec::new());
    }
    assert_eq!(
        serde_json::to_string(&s.world).unwrap(),
        serde_json::to_string(&restored.world).unwrap(),
        "world diverged after restoring mid-autonomy"
    );
    assert_eq!(s.events.checksum(), restored.events.checksum());

    // A pre-`tended` save (only the original two maps) must still load.
    let old: rpg_sim::SimTransient =
        serde_json::from_str(r#"{"paths":{},"pending_private":[],"last_incident_minute":0,"autonomy":{"pursuing":{},"cooldowns":{}}}"#).unwrap();
    assert!(old.autonomy.tended.is_empty());
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
        Vec::<AutonomyProposal>::new()
    );
}

/// Regression for review finding #4: a fresh pursuit must be recorded only
/// after the at-post gate accepts it. Here the actor is *off* its post when
/// the goal first fires (the schedule proposes a `MoveTo`), so the pursuit
/// candidate must be dropped — recording it would suppress the schedule from
/// the next tick (the actor would divert to the goal instead of its post) and
/// its deadline, estimated before the schedule walk, could expire mid-walk.
#[test]
fn rejected_pursuit_does_not_suppress_the_schedule() {
    let mut s = sim();
    // Renn starts at the plaza while her schedule puts her on duty at the
    // office: routine proposes `MoveTo office`, so the gate must reject the
    // warehouse pursuit on this tick.
    let mut renn = npc(
        "guard_captain",
        "plaza",
        Vec2::new(21.0, 12.0),
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

    s.step(Vec::new());
    assert!(
        !s.autonomy.is_pursuing(&ActorId::new("guard_captain")),
        "a pursuit the at-post gate rejected must not be recorded"
    );

    // The schedule must keep owning the actor: she walks to her post, not to
    // the goal target. (Pre-fix, the recorded pursuit suppressed the routine
    // and she never went to the office at all.)
    let mut reached_post = false;
    for _ in 0..30 {
        s.step(Vec::new());
        if s.world
            .actor(&ActorId::new("guard_captain"))
            .map(|a| a.location == LocationId::new("office"))
            .unwrap_or(false)
        {
            reached_post = true;
            break;
        }
    }
    assert!(
        reached_post,
        "the schedule walk to the post must not be suppressed by a rejected pursuit"
    );

    // Once idling at her post, the same goal may legitimately fire; the
    // travel-scaled deadline (computed at acceptance, from the post) must
    // then carry the whole walk to the warehouse without expiring mid-walk.
    let mut committed_at: Option<u64> = None;
    let mut expired_mid_walk = false;
    let mut inspected = false;
    for _ in 0..150 {
        s.step(Vec::new());
        let pursuing = s.autonomy.is_pursuing(&ActorId::new("guard_captain"));
        let a = s.world.actor(&ActorId::new("guard_captain")).unwrap();
        if committed_at.is_none() {
            if pursuing {
                committed_at = Some(s.world.tick);
            }
        } else {
            if a.inspected.contains("warehouse") {
                inspected = true;
                break;
            }
            if !pursuing {
                // The pursuit ended before the goal action ever happened.
                expired_mid_walk = true;
                break;
            }
        }
    }
    assert!(committed_at.is_some(), "the goal should fire at the post");
    assert!(
        inspected,
        "after the gate accepts the pursuit at the post, the walk must complete and investigate"
    );
    assert!(
        !expired_mid_walk,
        "the pursuit deadline (computed at acceptance) must not expire mid-walk"
    );
}

/// Regression for review finding #6: Marta's Protect goal must produce
/// meaningful bounded behaviour instead of endless empty-shop loops — one
/// comfort check-in per visit, then she returns to tending the shop and does
/// not leave again while the ward stays put.
#[test]
fn protect_arrival_comforts_once_and_returns_to_post() {
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

    let mut comforted = 0usize;
    let mut ticks_at_backroom = 0usize;
    for _ in 0..240 {
        s.step(Vec::new());
        if s.world
            .actor(&ActorId::new("grocer"))
            .map(|a| a.location == LocationId::new("backroom"))
            .unwrap_or(false)
        {
            ticks_at_backroom += 1;
        }
        if s.events
            .all()
            .iter()
            .any(|e| matches!(e.kind, rpg_domain::events::EventKind::Comforted { .. }))
        {
            comforted += 1;
        }
    }
    assert!(
        comforted >= 1,
        "the Protect arrival must perform a real, observable check-in (Comforted event)"
    );
    // Bounded: the check-in happens once per ward position, not every tick
    // and not once per cooldown cycle. (Counting event occurrences keeps this
    // independent of exact arrival timing.)
    let distinct_events = s
        .events
        .all()
        .iter()
        .filter(|e| matches!(e.kind, rpg_domain::events::EventKind::Comforted { .. }))
        .count();
    assert_eq!(
        distinct_events, 1,
        "exactly one check-in while the ward stays put — no endless loop"
    );
    assert!(
        ticks_at_backroom <= 40,
        "Marta spent {ticks_at_backroom} ticks away from the shop over 240; \
         the check-in must be brief, not a second home"
    );
    let marta = s.world.actor(&ActorId::new("grocer")).unwrap();
    assert!(
        marta.location == LocationId::new("shop"),
        "after the check-in Marta must settle back at her scheduled post, got {}",
        marta.location
    );
    assert!(!s.autonomy.is_pursuing(&marta.id));
}

/// Regression for review finding #6 (movement variant): when the ward moves,
/// the settled Protect goal re-opens and Marta follows — meaningfully and
/// once per new position, not in a loop.
#[test]
fn protect_goal_reopens_only_when_the_ward_moves() {
    let mut s = sim();
    // Park the player away from both check-in sites: an NPC within talk
    // range of the player is never autonomy-driven (conversation
    // preservation), which must not confound this test.
    if let Some(p) = s.world.actor_mut(&ActorId::new(PLAYER)) {
        p.location = LocationId::new("office");
        p.pos = Vec2::new(7.0, 19.5);
    }
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

    // First check-in: settle the goal at the backroom.
    let mut first_visit = false;
    for _ in 0..60 {
        s.step(Vec::new());
        if s.world
            .actor(&ActorId::new("grocer"))
            .map(|a| a.location == LocationId::new("backroom"))
            .unwrap_or(false)
        {
            first_visit = true;
            break;
        }
    }
    assert!(first_visit, "Marta should check on Lena once");

    // While Lena stays put, Marta must not chase her again.
    for _ in 0..120 {
        s.step(Vec::new());
    }
    assert!(
        !s.autonomy.is_pursuing(&ActorId::new("grocer")),
        "a tended ward must not re-open the goal"
    );

    // Lena moves to the plaza for good (her schedule changes, e.g. she feels
    // well enough to watch the market): the goal re-opens exactly once and
    // Marta follows her there. Run the whole window — the check-in happens
    // on the tick after she reaches Lena, not when she first crosses the
    // plaza boundary. The ward must genuinely move; a bare teleport is
    // undone by her own routine (schedule) within a tick or two.
    if let Some(l) = s.world.actor_mut(&ActorId::new("grocer_daughter")) {
        l.location = LocationId::new("plaza");
        l.pos = Vec2::new(21.0, 12.0);
        l.schedule = vec![sched(0, 1440, "plaza", "watching the market")];
    }
    let mut followed = false;
    for _ in 0..120 {
        s.step(Vec::new());
        if s.world
            .actor(&ActorId::new("grocer"))
            .map(|a| a.location == LocationId::new("plaza"))
            .unwrap_or(false)
        {
            followed = true;
        }
    }
    assert!(followed, "a moved ward must re-open the protective pursuit");
    let comforted = s
        .events
        .all()
        .iter()
        .filter(|e| matches!(e.kind, rpg_domain::events::EventKind::Comforted { .. }))
        .count();
    assert_eq!(
        comforted, 2,
        "one check-in per ward position: backroom once, plaza once"
    );
}

/// Regression for the dead-ward edge: a ward who dies mid-pursuit must not
/// receive a phantom check-in. The arrival branch previously recorded a tend
/// site and emitted a `Comfort` the sim then rejected (`ActorNotAlive`) —
/// contradicting "the emitted proposal always applies" and silently settling
/// the goal with a tend record for a ward that can never move again.
#[test]
fn dead_ward_is_not_phantom_tended_on_arrival() {
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

    // Marta sets out while Lena is still alive.
    let mut pursuit_started = false;
    for _ in 0..20 {
        s.step(Vec::new());
        if s.autonomy.is_pursuing(&ActorId::new("grocer")) {
            pursuit_started = true;
            break;
        }
    }
    assert!(pursuit_started, "Marta should set out toward her ward");

    // Lena dies while Marta walks (authored story event, applied directly).
    s.world
        .actors
        .get_mut(&ActorId::new("grocer_daughter"))
        .unwrap()
        .status = ActorStatus::Dead;

    // Let the pursuit play out: the walk still completes (the arrival branch
    // is exercised with a dead ward), then the pursuit expires at its
    // deadline. She must never comfort the dead, and no tend site may be
    // recorded for a ward that can never move again.
    let mut visited = false;
    for _ in 0..400 {
        s.step(Vec::new());
        if s.world
            .actor(&ActorId::new("grocer"))
            .map(|a| a.location == LocationId::new("backroom"))
            .unwrap_or(false)
        {
            visited = true;
        }
        if !s.autonomy.is_pursuing(&ActorId::new("grocer")) {
            break;
        }
    }
    assert!(
        visited,
        "the walk must reach the ward's location despite the death"
    );

    let comforted = s
        .events
        .all()
        .iter()
        .filter(|e| matches!(e.kind, rpg_domain::events::EventKind::Comforted { .. }))
        .count();
    assert_eq!(
        comforted, 0,
        "a dead ward must never be comforted (the sim rejects Comfort on the dead)"
    );
    assert!(
        !s.autonomy
            .tended
            .get(&ActorId::new("grocer"))
            .is_some_and(|m| m.contains_key(&1)),
        "a comfort the sim would reject must not record a phantom tend site"
    );

    // The goal is spent (a dead ward can never be re-selected), and Marta
    // settles back at her scheduled post.
    for _ in 0..240 {
        s.step(Vec::new());
    }
    assert!(!s.autonomy.is_pursuing(&ActorId::new("grocer")));
    let marta = s.world.actor(&ActorId::new("grocer")).unwrap();
    assert_eq!(
        marta.location,
        LocationId::new("shop"),
        "Marta must settle back at her scheduled post, got {}",
        marta.location
    );
}
