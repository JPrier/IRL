//! Graphical, voice-native client for IRL.
//!
//! The authoritative simulation lives on the worker thread (see [`worker`]).
//! This binary is a *renderer and input device*: it sends [`Command`]s and
//! draws the immutable [`Snapshot`]s that come back. It never mutates world
//! state, and `rpg_sim` never depends on Bevy — the crate boundary is the
//! invariant the design asks for.
//!
//! Layout: the town is drawn as flat rectangles in world units, scaled by
//! [`PIXELS_PER_UNIT`] and centred on the bounding box of the locations. The
//! player is drawn green and on top; a nearby, talkable NPC is highlighted.

mod worker;

use std::collections::{HashMap, HashSet};

use bevy::prelude::*;
use rpg_domain::action::InvestigateTarget;
use rpg_domain::geom::Vec2;
use rpg_domain::ids::{LocationId, OfferId};
use worker::{
    default_save_path, spawn_worker, Command, KindView, LineKind, Snapshot, StatusView,
    WorkerConfig,
};

/// Pixels per world unit. The town spans ~34x18 units, so this fills a
/// 1280x800 window without scrolling.
const PIXELS_PER_UNIT: f32 = 20.0;
/// Draw size of an actor, in world units.
const ACTOR_SIZE: f32 = 0.9;
/// How far one WASD key press walks, in world units.
const MOVE_STEP: f32 = 3.0;
/// How many dialogue lines to show.
const DIALOGUE_LINES: usize = 7;

// ---------------------------------------------------------------------------
// Resources and components
// ---------------------------------------------------------------------------

#[derive(Resource)]
struct SimHandle {
    cmd_tx: std::sync::mpsc::Sender<Command>,
    // `mpsc::Receiver` is `Send` but not `Sync`, so it cannot live in a Bevy
    // `Resource` directly. A `Mutex` makes it `Sync` without changing the
    // worker-side API.
    snap_rx: std::sync::Mutex<std::sync::mpsc::Receiver<Snapshot>>,
    /// Shared with the worker so Escape/quit aborts an in-flight capture.
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    latest: Option<Snapshot>,
}

/// Entities spawned for world objects, keyed by simulation id, so updates
/// reuse the same entity instead of despawning and respawning every frame.
#[derive(Resource, Default)]
struct Spawned {
    locations: HashMap<String, Entity>,
    actors: HashMap<String, Entity>,
}

#[derive(Component)]
struct LocationSprite;

#[derive(Component)]
struct ActorSprite;

#[derive(Component)]
struct ClockText;

#[derive(Component)]
struct StatusText;

#[derive(Component)]
struct DialogueText;

#[derive(Component)]
struct HelpText;

#[derive(Component)]
struct OffersText;

#[derive(Component)]
struct InventoryText;

#[derive(Component)]
struct DiagnosticsText;

#[derive(Component)]
struct DiagnosticsPanel;

/// UI-only state: which offer is selected and whether the diagnostics overlay
/// is open. Never affects canonical simulation state.
#[derive(Resource, Default)]
struct UiState {
    /// Index into the player-addressed pending offers.
    selected_offer: usize,
    diagnostics: bool,
}

// Bevy query signatures get long quickly; naming them keeps clippy's
// `type_complexity` lint quiet and makes the systems below readable.
type LocationSpriteQuery<'w, 's> = Query<
    'w,
    's,
    (&'static mut Transform, &'static mut Sprite),
    (With<LocationSprite>, Without<ActorSprite>),
>;
type ActorSpriteQuery<'w, 's> = Query<
    'w,
    's,
    (&'static mut Transform, &'static mut Sprite),
    (With<ActorSprite>, Without<LocationSprite>),
>;
type HudTextQuery<'w, 's> = Query<
    'w,
    's,
    (
        &'static mut Text,
        Option<&'static ClockText>,
        Option<&'static StatusText>,
        Option<&'static DialogueText>,
        Option<&'static HelpText>,
        Option<&'static OffersText>,
        Option<&'static InventoryText>,
        Option<&'static DiagnosticsText>,
    ),
>;

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let cfg = parse_args();
    let handle = spawn_worker(cfg);

    App::new()
        .insert_resource(ClearColor(Color::srgb(0.055, 0.06, 0.08)))
        .insert_resource(SimHandle {
            cmd_tx: handle.cmd_tx,
            snap_rx: std::sync::Mutex::new(handle.snap_rx),
            cancel: handle.cancel,
            latest: None,
        })
        .init_resource::<Spawned>()
        .init_resource::<UiState>()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "IRL — Hidden Mafia".into(),
                ..default()
            }),
            ..default()
        }))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                poll_snapshot,
                handle_input,
                on_app_exit,
                sync_world,
                update_ui,
                update_offers,
                update_inventory,
                update_diagnostics,
                toggle_diagnostics,
            )
                .chain(),
        )
        .run();
}

fn parse_args() -> WorkerConfig {
    let mut seed = 2026u64;
    let mut model_path = None;
    let mut voice = "af_sarah".to_string();
    let mut no_voice = false;

    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--seed" => {
                i += 1;
                if let Some(v) = argv.get(i).and_then(|s| s.parse().ok()) {
                    seed = v;
                }
            }
            "--model" => {
                i += 1;
                model_path = argv.get(i).cloned();
            }
            "--voice" => {
                i += 1;
                if let Some(v) = argv.get(i) {
                    voice = v.clone();
                }
            }
            "--no-voice" => no_voice = true,
            "--doctor" | "--paths" => {
                println!("{}", worker::doctor_report());
                std::process::exit(0);
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    // With no explicit `--model`, fall back to a model bundled next to the
    // executable so a shipped build runs with no arguments.
    let model_path = model_path.or_else(worker::bundled_model);

    WorkerConfig {
        seed,
        model_path,
        voice,
        no_voice,
    }
}

fn print_help() {
    println!(
        "bevy-game — graphical, voice-native IRL client\n\
         \n\
         Usage: bevy-game [--seed N] [--model path.gguf] [--voice ID] [--no-voice]\n\
         \n\
         Controls:\n\
           WASD / arrows   walk\n\
           V               hold a voice conversation (mic -> ASR -> NPC -> TTS)\n\
           Esc             quit\n\
         \n\
         Voices (Kokoro): af_alloy af_bella af_nicole af_sarah af_sky am_adam\n\
                          am_michael bf_emma bf_isabella bm_george bm_lewis\n"
    );
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

fn setup(mut commands: Commands) {
    commands.spawn((Camera2d, Transform::from_xyz(0.0, 0.0, 1000.0)));

    let text_font = |size: f32| TextFont {
        font_size: size.into(),
        ..default()
    };

    commands.spawn((
        Text::new("IRL — Hidden Mafia"),
        text_font(24.0),
        TextColor(Color::srgb(0.92, 0.92, 0.96)),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(14.0),
            top: Val::Px(10.0),
            ..default()
        },
        ClockText,
    ));

    commands.spawn((
        Text::new(""),
        text_font(15.0),
        TextColor(Color::srgb(0.70, 0.74, 0.82)),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(14.0),
            top: Val::Px(42.0),
            ..default()
        },
        StatusText,
    ));

    commands.spawn((
        Text::new(""),
        text_font(16.0),
        TextColor(Color::srgb(0.86, 0.87, 0.90)),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(14.0),
            bottom: Val::Px(40.0),
            ..default()
        },
        DialogueText,
    ));

    commands.spawn((
        Text::new(
            "WASD move   V talk   Y accept   N reject   Tab next offer   E investigate\n\
             F5 save   F9 load   F3 diagnostics   , . nudge   Esc quit",
        ),
        text_font(13.0),
        TextColor(Color::srgb(0.45, 0.48, 0.55)),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(14.0),
            bottom: Val::Px(10.0),
            ..default()
        },
        HelpText,
    ));

    // Pending offers, right-aligned above the help line.
    commands.spawn((
        Text::new(""),
        text_font(14.0),
        TextColor(Color::srgb(0.98, 0.80, 0.25)),
        Node {
            position_type: PositionType::Absolute,
            right: Val::Px(14.0),
            bottom: Val::Px(64.0),
            ..default()
        },
        OffersText,
    ));

    // Inventory and money, right-aligned at the top.
    commands.spawn((
        Text::new(""),
        text_font(14.0),
        TextColor(Color::srgb(0.72, 0.86, 0.98)),
        Node {
            position_type: PositionType::Absolute,
            right: Val::Px(14.0),
            top: Val::Px(10.0),
            ..default()
        },
        InventoryText,
    ));

    // Developer diagnostics overlay (hidden by default; F3 toggles it).
    commands.spawn((
        Text::new(""),
        text_font(13.0),
        TextColor(Color::srgb(0.55, 0.95, 0.75)),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(340.0),
            top: Val::Px(10.0),
            ..default()
        },
        DiagnosticsText,
        DiagnosticsPanel,
    ));
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Drain every snapshot the worker published and keep only the newest.
fn poll_snapshot(mut sim: ResMut<SimHandle>) {
    // Drain under the lock, then release it before touching `latest`, so the
    // two disjoint fields are never borrowed at once through the `ResMut`.
    let mut newest = None;
    if let Ok(rx) = sim.snap_rx.get_mut() {
        while let Ok(snap) = rx.try_recv() {
            newest = Some(snap);
        }
    }
    if newest.is_some() {
        sim.latest = newest;
    }
}

/// Translate keyboard input into worker commands. World movement is only
/// *requested* here; the simulation validates and commits it.
fn handle_input(
    keys: Res<ButtonInput<KeyCode>>,
    sim: Res<SimHandle>,
    mut ui: ResMut<UiState>,
    mut exit: MessageWriter<AppExit>,
) {
    if keys.just_pressed(KeyCode::Escape) {
        // Route quitting through Bevy's own `AppExit` so the worker thread is
        // stopped by `on_app_exit` and audio/inference resources are released
        // instead of being killed with the process. Signal cancellation first
        // so a blocked capture returns immediately instead of holding the
        // worker for up to its full capture window.
        sim.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
        exit.write(AppExit::Success);
        return;
    }
    if keys.just_pressed(KeyCode::KeyV) {
        let _ = sim.cmd_tx.send(Command::VoiceTurn);
        return;
    }

    // --- Save / load ---------------------------------------------------
    if keys.just_pressed(KeyCode::F5) {
        let _ = sim.cmd_tx.send(Command::Save(default_save_path()));
        return;
    }
    if keys.just_pressed(KeyCode::F9) {
        let _ = sim.cmd_tx.send(Command::Load(default_save_path()));
        return;
    }

    let Some(snap) = sim.latest.as_ref() else {
        return;
    };
    if snap.busy {
        // A voice turn is in flight; ignore movement so commands do not pile up.
        return;
    }

    // --- Offers --------------------------------------------------------
    // Player-addressed offers are the only ones the player can accept.
    let pending: Vec<&worker::OfferView> = snap
        .offers
        .iter()
        .filter(|o| o.to_player && o.state == "pending")
        .collect();
    if keys.just_pressed(KeyCode::Tab) && !pending.is_empty() {
        ui.selected_offer = (ui.selected_offer + 1) % pending.len();
        return;
    }
    if keys.just_pressed(KeyCode::KeyY) {
        if let Some(offer) = pending.get(ui.selected_offer.min(pending.len().saturating_sub(1))) {
            let _ = sim.cmd_tx.send(Command::AcceptOffer(OfferId(offer.id)));
            ui.selected_offer = 0;
        }
        return;
    }
    if keys.just_pressed(KeyCode::KeyN) {
        if let Some(offer) = pending.get(ui.selected_offer.min(pending.len().saturating_sub(1))) {
            let _ = sim.cmd_tx.send(Command::RejectOffer(OfferId(offer.id)));
            ui.selected_offer = 0;
        }
        return;
    }

    // --- Investigate the current location ------------------------------
    if keys.just_pressed(KeyCode::KeyE) {
        let loc = LocationId::from(snap.player_location.as_str());
        let _ = sim
            .cmd_tx
            .send(Command::Investigate(InvestigateTarget::Location(loc)));
        return;
    }

    let mut dir = Vec2::ZERO;
    if keys.just_pressed(KeyCode::KeyW) || keys.just_pressed(KeyCode::ArrowUp) {
        dir.y += 1.0;
    }
    if keys.just_pressed(KeyCode::KeyS) || keys.just_pressed(KeyCode::ArrowDown) {
        dir.y -= 1.0;
    }
    if keys.just_pressed(KeyCode::KeyA) || keys.just_pressed(KeyCode::ArrowLeft) {
        dir.x -= 1.0;
    }
    if keys.just_pressed(KeyCode::KeyD) || keys.just_pressed(KeyCode::ArrowRight) {
        dir.x += 1.0;
    }
    if dir == Vec2::ZERO {
        return;
    }
    let target = Vec2::new(
        snap.player_pos.x + dir.x * MOVE_STEP,
        snap.player_pos.y + dir.y * MOVE_STEP,
    );
    let _ = sim.cmd_tx.send(Command::MoveTo(target));
}

/// Forward Bevy's `AppExit` to the worker so its audio/inference threads stop
/// cleanly instead of being torn down with the process.
fn on_app_exit(mut exits: MessageReader<AppExit>, sim: Res<SimHandle>) {
    for _ in exits.read() {
        sim.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
        let _ = sim.cmd_tx.send(Command::Quit);
    }
}

/// Spawn and update one sprite per location and per actor, so the rendered
/// world matches the authoritative snapshot exactly.
fn sync_world(
    mut commands: Commands,
    sim: Res<SimHandle>,
    mut spawned: ResMut<Spawned>,
    mut locations: LocationSpriteQuery,
    mut actors: ActorSpriteQuery,
) {
    let Some(snap) = sim.latest.as_ref() else {
        return;
    };

    // Centre the camera on the town's bounding box so the whole map is visible.
    let center = town_center(snap);

    // --- Locations ------------------------------------------------------
    let mut live_locations: HashSet<String> = HashSet::new();
    for loc in &snap.locations {
        live_locations.insert(loc.id.clone());
        let screen = Vec2::new(
            (loc.x + loc.w * 0.5 - center.x) * PIXELS_PER_UNIT,
            (loc.y + loc.h * 0.5 - center.y) * PIXELS_PER_UNIT,
        );
        // `size` feeds `Sprite`/`custom_size`, which are glam vectors; the
        // rest of the geometry stays in `rpg_domain::geom::Vec2`.
        let size = bevy::math::Vec2::new(loc.w * PIXELS_PER_UNIT, loc.h * PIXELS_PER_UNIT);
        let color = if loc.private {
            Color::srgb(0.20, 0.16, 0.26)
        } else {
            Color::srgb(0.15, 0.17, 0.21)
        };

        match spawned.locations.get(&loc.id).copied() {
            Some(entity) => {
                if let Ok((mut transform, mut sprite)) = locations.get_mut(entity) {
                    transform.translation.x = screen.x;
                    transform.translation.y = screen.y;
                    sprite.color = color;
                    sprite.custom_size = Some(size);
                }
            }
            None => {
                let entity = commands
                    .spawn((
                        Sprite::from_color(color, size),
                        Transform::from_xyz(screen.x, screen.y, 0.0),
                        LocationSprite,
                    ))
                    .id();
                spawned.locations.insert(loc.id.clone(), entity);
            }
        }
    }
    // Despawn locations that no longer exist in the snapshot.
    let stale: Vec<String> = spawned
        .locations
        .keys()
        .filter(|id| !live_locations.contains(*id))
        .cloned()
        .collect();
    for id in stale {
        if let Some(entity) = spawned.locations.remove(&id) {
            commands.entity(entity).despawn();
        }
    }

    // --- Actors ---------------------------------------------------------
    let mut live_actors: HashSet<String> = HashSet::new();
    for actor in &snap.actors {
        live_actors.insert(actor.id.clone());
        let screen = Vec2::new(
            (actor.pos.x - center.x) * PIXELS_PER_UNIT,
            (actor.pos.y - center.y) * PIXELS_PER_UNIT,
        );
        let color = actor_color(actor.kind, actor.status, actor.nearby);

        match spawned.actors.get(&actor.id).copied() {
            Some(entity) => {
                if let Ok((mut transform, mut sprite)) = actors.get_mut(entity) {
                    transform.translation.x = screen.x;
                    transform.translation.y = screen.y;
                    sprite.color = color;
                }
            }
            None => {
                let entity = commands
                    .spawn((
                        Sprite::from_color(
                            color,
                            bevy::math::Vec2::splat(ACTOR_SIZE * PIXELS_PER_UNIT),
                        ),
                        Transform::from_xyz(screen.x, screen.y, 10.0),
                        ActorSprite,
                    ))
                    .id();
                spawned.actors.insert(actor.id.clone(), entity);
            }
        }
    }
    let stale: Vec<String> = spawned
        .actors
        .keys()
        .filter(|id| !live_actors.contains(*id))
        .cloned()
        .collect();
    for id in stale {
        if let Some(entity) = spawned.actors.remove(&id) {
            commands.entity(entity).despawn();
        }
    }
}

/// Update the HUD text from the latest snapshot.
fn update_ui(sim: Res<SimHandle>, mut texts: HudTextQuery) {
    let Some(snap) = sim.latest.as_ref() else {
        return;
    };

    // Display names for dialogue speakers ("grocer" -> "Marta Voss").
    let mut names: HashMap<&str, &str> = HashMap::new();
    for actor in &snap.actors {
        names.insert(actor.id.as_str(), actor.name.as_str());
    }
    // Location display names ("market" -> "Market Square").
    let mut loc_names: HashMap<&str, &str> = HashMap::new();
    for loc in &snap.locations {
        loc_names.insert(loc.id.as_str(), loc.name.as_str());
    }

    for (mut text, clock, status, dialogue, _help, _offers, _inventory, _diagnostics) in &mut texts
    {
        if clock.is_some() {
            text.0 = format!(
                "IRL — Hidden Mafia          Day {}  {}",
                snap.day,
                snap.clock()
            );
        } else if status.is_some() {
            let here = loc_names
                .get(snap.player_location.as_str())
                .copied()
                .unwrap_or(snap.player_location.as_str());
            let nearby = match snap.nearby.as_deref() {
                Some(id) => {
                    let name = names.get(id).copied().unwrap_or(id);
                    let room = snap
                        .actors
                        .iter()
                        .find(|a| a.id == id)
                        .map(|a| a.location.as_str())
                        .and_then(|l| loc_names.get(l).copied())
                        .unwrap_or("");
                    if room.is_empty() {
                        name.to_string()
                    } else {
                        format!("{name} ({room})")
                    }
                }
                None => "nobody in earshot".to_string(),
            };
            let voice = match snap.voice_error.as_deref() {
                Some(err) => format!("   voice: {err}"),
                None => String::new(),
            };
            text.0 = format!(
                "{}  |  tick {}  |  at {}  |  near {}\n{} ({})  |  decisions: {} model / {} canonical / {} fallback{}",
                snap.status,
                snap.tick,
                here,
                nearby,
                snap.engine_name,
                snap.model_id,
                snap.model_decisions,
                snap.canonical_decisions,
                snap.fallback_decisions,
                voice,
            );
        } else if dialogue.is_some() {
            let start = snap.dialogue.len().saturating_sub(DIALOGUE_LINES);
            let lines: Vec<String> = snap.dialogue[start..]
                .iter()
                .map(|line| {
                    let who = match line.kind {
                        LineKind::Player => "You".to_string(),
                        LineKind::System => "—".to_string(),
                        LineKind::Npc => names
                            .get(line.speaker.as_str())
                            .copied()
                            .unwrap_or(line.speaker.as_str())
                            .to_string(),
                    };
                    format!("{who}: {}", line.text)
                })
                .collect();
            text.0 = lines.join("\n");
        }
    }
}

/// Render the player-addressed offers from the latest snapshot.
fn update_offers(sim: Res<SimHandle>, mut texts: Query<&mut Text, With<OffersText>>) {
    let Some(snap) = sim.latest.as_ref() else {
        return;
    };
    for mut text in &mut texts {
        let lines: Vec<String> = snap
            .offers
            .iter()
            .filter(|o| o.to_player && o.state == "pending")
            .map(|o| {
                format!(
                    "#{} from {} — give:{} want:{} ({})",
                    o.id, o.from, o.give, o.want, o.state
                )
            })
            .collect();
        text.0 = lines.join("\n");
    }
}

/// Render the player's inventory and money.
fn update_inventory(sim: Res<SimHandle>, mut texts: Query<&mut Text, With<InventoryText>>) {
    let Some(snap) = sim.latest.as_ref() else {
        return;
    };
    for mut text in &mut texts {
        let items: Vec<String> = snap
            .inventory
            .iter()
            .map(|i| format!("{} x{}", i.name, i.qty))
            .collect();
        text.0 = format!("Money: {}\n{}", snap.money, items.join("\n"));
    }
}

/// Developer diagnostics overlay; hidden unless F3 has been pressed.
fn update_diagnostics(
    sim: Res<SimHandle>,
    ui: Res<UiState>,
    mut texts: Query<&mut Text, With<DiagnosticsText>>,
) {
    let Some(snap) = sim.latest.as_ref() else {
        return;
    };
    for mut text in &mut texts {
        if !ui.diagnostics {
            text.0.clear();
            continue;
        }
        let commitments: Vec<String> = snap
            .commitments
            .iter()
            .map(|c| format!("  {} — {} ({})", c.kind, c.terms, c.state))
            .collect();
        let beliefs = if snap.nearby_beliefs.is_empty() {
            "—".to_string()
        } else {
            snap.nearby_beliefs.join("\n  ")
        };
        let events = if snap.recent_events.is_empty() {
            "—".to_string()
        } else {
            snap.recent_events.join("\n  ")
        };
        text.0 = format!(
            "DIAGNOSTICS\nengine: {} ({})\ntick: {}\nmodel decisions: {}\ncanonical: {}\nfallback: {}\nlast turn: {}\n\nPerception:\n  {}\n\nNearby relationship:\n  {}\n\nNearby beliefs (developer view):\n  {}\n\nCommitments:\n{}\n\nRecent events:\n  {}",
            snap.engine_name,
            snap.model_id,
            snap.tick,
            snap.model_decisions,
            snap.canonical_decisions,
            snap.fallback_decisions,
            snap.last_turn_timing.as_deref().unwrap_or("—"),
            snap.perception.join("\n  "),
            snap.nearby_relationship.as_deref().unwrap_or("—"),
            beliefs,
            commitments.join("\n"),
            events,
        );
    }
}

/// Toggle the diagnostics overlay. Kept as its own system so it never depends
/// on the ordering of `handle_input`.
fn toggle_diagnostics(keys: Res<ButtonInput<KeyCode>>, mut ui: ResMut<UiState>) {
    if keys.just_pressed(KeyCode::F3) {
        ui.diagnostics = !ui.diagnostics;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Centre of the bounding box of all locations (actors live inside it), so
/// the camera can be kept fixed while the whole town stays on screen.
fn town_center(snap: &Snapshot) -> Vec2 {
    if snap.locations.is_empty() {
        return Vec2::ZERO;
    }
    let mut min = Vec2::new(f32::INFINITY, f32::INFINITY);
    let mut max = Vec2::new(f32::NEG_INFINITY, f32::NEG_INFINITY);
    for loc in &snap.locations {
        min.x = min.x.min(loc.x);
        min.y = min.y.min(loc.y);
        max.x = max.x.max(loc.x + loc.w);
        max.y = max.y.max(loc.y + loc.h);
    }
    Vec2::new((min.x + max.x) * 0.5, (min.y + max.y) * 0.5)
}

/// Player, nearby NPC, other NPC, background, and dead/departed are all
/// visually distinct without needing any sprite assets.
fn actor_color(kind: KindView, status: StatusView, nearby: bool) -> Color {
    if status != StatusView::Alive {
        return Color::srgb(0.28, 0.28, 0.32);
    }
    match kind {
        KindView::Player => Color::srgb(0.30, 0.85, 0.45),
        KindView::Npc if nearby => Color::srgb(0.98, 0.80, 0.25),
        KindView::Npc => Color::srgb(0.90, 0.45, 0.35),
        KindView::Background => Color::srgb(0.55, 0.55, 0.62),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot_with(locations: Vec<worker::LocationView>) -> Snapshot {
        Snapshot {
            tick: 0,
            day: 1,
            minute_of_day: 0,
            locations,
            actors: vec![],
            player_pos: Vec2::ZERO,
            player_location: String::new(),
            nearby: None,
            dialogue: vec![],
            engine_name: String::new(),
            model_id: String::new(),
            model_decisions: 0,
            canonical_decisions: 0,
            fallback_decisions: 0,
            busy: false,
            status: String::new(),
            voice_error: None,
            recent_events: vec![],
            nearby_beliefs: vec![],
            nearby_relationship: None,
            perception: vec![],
            last_turn_timing: None,
            offers: vec![],
            commitments: vec![],
            inventory: vec![],
            money: 0,
        }
    }

    #[test]
    fn town_center_is_the_bounding_box_midpoint() {
        let snap = snapshot_with(vec![
            worker::LocationView {
                id: "a".into(),
                name: "A".into(),
                x: 0.0,
                y: 0.0,
                w: 10.0,
                h: 4.0,
                private: false,
            },
            worker::LocationView {
                id: "b".into(),
                name: "B".into(),
                x: 20.0,
                y: 8.0,
                w: 10.0,
                h: 4.0,
                private: false,
            },
        ]);
        let c = town_center(&snap);
        assert!((c.x - 15.0).abs() < 1e-6);
        assert!((c.y - 6.0).abs() < 1e-6);
    }

    #[test]
    fn town_center_of_empty_world_is_origin() {
        let snap = snapshot_with(vec![]);
        assert_eq!(town_center(&snap), Vec2::ZERO);
    }

    #[test]
    fn player_is_drawn_distinctly_from_npcs() {
        assert_ne!(
            actor_color(KindView::Player, StatusView::Alive, false),
            actor_color(KindView::Npc, StatusView::Alive, false)
        );
        assert_ne!(
            actor_color(KindView::Npc, StatusView::Alive, true),
            actor_color(KindView::Npc, StatusView::Alive, false)
        );
    }

    #[test]
    fn dead_actors_are_not_drawn_as_alive() {
        assert_ne!(
            actor_color(KindView::Npc, StatusView::Alive, false),
            actor_color(KindView::Npc, StatusView::Dead, false)
        );
    }
}
