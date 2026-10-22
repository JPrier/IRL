//! Graphical, voice-native client for IRL.
//!
//! The authoritative simulation runs in [`rpg_runtime`] on its own fixed-tick
//! thread, with the microphone, speech recognizer, language model and voice
//! on dedicated workers. This binary is a *renderer and input device*: it
//! sends [`Command`]s and draws the immutable [`Snapshot`]s that come back.
//! It never mutates world state and never waits on the simulation, and
//! `rpg_sim` never depends on Bevy — the crate boundary is the invariant the
//! design asks for.
//!
//! Layout: the town is drawn as flat rectangles in world units, scaled by
//! [`PIXELS_PER_UNIT`] and centred on the bounding box of the locations. The
//! player is drawn green and on top; a nearby, talkable NPC is highlighted.

mod headless;
mod paths;
mod voice;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;

use bevy::input::keyboard::KeyboardInput;
use bevy::input::ButtonState;
use bevy::prelude::*;
use rpg_domain::action::InvestigateTarget;
use rpg_domain::geom::Vec2;
use rpg_domain::ids::{LocationId, OfferId};
use rpg_runtime::{
    Command, KindView, LineKind, MicConfig, MicSelection, OfferView, Readiness, RuntimeConfig,
    RuntimeHandle, Snapshot, StatusView, VoiceStage,
};

use paths::default_save_path;
use voice::VoiceOptions;

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
    /// Owned so the runtime can be shut down cleanly on exit. `Receiver` is
    /// `Send` but not `Sync`, hence the `Mutex`; it is only ever locked by
    /// this thread, so the lock never contends.
    runtime: Mutex<Option<RuntimeHandle>>,
    commands: std::sync::mpsc::Sender<rpg_runtime::Inbox>,
    latest: Option<Snapshot>,
}

impl SimHandle {
    fn send(&self, cmd: Command) {
        let _ = self.commands.send(rpg_runtime::Inbox::Ui(cmd));
    }
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
struct VoiceText;

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

/// UI-only state. Never affects canonical simulation state.
#[derive(Resource, Default)]
struct UiState {
    /// Index into the player-addressed pending offers.
    selected_offer: usize,
    diagnostics: bool,
    /// `Some` while the player is typing a line of dialogue.
    typing: Option<String>,
    /// Push-to-talk is held (tracked locally so a release is always sent).
    ptt_held: bool,
    /// The text box closed this frame; its Enter/Esc must not also act as a
    /// gameplay key (reopen the box, or quit).
    typing_closed: bool,
    /// Smoothed renderer frame rate.
    fps: f32,
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
        Option<&'static VoiceText>,
        Option<&'static DialogueText>,
    ),
>;

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

struct Args {
    seed: u64,
    opts: VoiceOptions,
    auto_endpoint: bool,
    dump_utterance: bool,
}

fn main() {
    let args = parse_args();
    let mut session = rpg_eval::Session::new(args.seed);
    // The session's own engine is unused: decisions run on the runtime's
    // inference worker. A default placement keeps the first turn simple.
    session.teleport_player_to("grocer");
    let cfg = RuntimeConfig {
        seed: args.seed,
        mic: MicConfig {
            device: args.opts.mic_device.clone(),
            auto_endpoint: args.auto_endpoint.then_some(Duration::from_millis(900)),
            dump_utterance: args
                .dump_utterance
                .then(|| paths::home_root().join("last_utterance.wav")),
            ..MicConfig::default()
        },
        ..RuntimeConfig::default()
    };
    let runtime = rpg_runtime::spawn(cfg, session, voice::backends(&args.opts));
    let commands = runtime.inbox();

    App::new()
        .insert_resource(ClearColor(Color::srgb(0.055, 0.06, 0.08)))
        .insert_resource(SimHandle {
            runtime: Mutex::new(Some(runtime)),
            commands,
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
                measure_fps,
                handle_typing,
                handle_input,
                on_app_exit,
                sync_world,
                update_ui,
                update_offers,
                update_inventory,
                update_diagnostics,
            )
                .chain(),
        )
        .run();
}

fn parse_args() -> Args {
    let mut seed = 2026u64;
    let mut opts = VoiceOptions {
        voice: "af_sarah".into(),
        ..VoiceOptions::default()
    };
    let mut auto_endpoint = false;
    let mut dump_utterance = false;
    let mut selftest: Option<f32> = None;
    let mut voice_check = false;
    let mut near = "grocer".to_string();
    let mut turns = 1usize;
    let mut hold: Option<f32> = None;

    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    let next = |i: &mut usize| -> Option<String> {
        *i += 1;
        argv.get(*i).cloned()
    };
    while i < argv.len() {
        match argv[i].as_str() {
            "--seed" => {
                if let Some(v) = next(&mut i).and_then(|s| s.parse().ok()) {
                    seed = v;
                }
            }
            "--model" => opts.model_path = next(&mut i),
            "--voice" => {
                if let Some(v) = next(&mut i) {
                    opts.voice = v;
                }
            }
            "--no-voice" => opts.no_voice = true,
            "--mic" => opts.mic_device = next(&mut i),
            "--mic-wav" => opts.mic_wav = next(&mut i).map(PathBuf::from),
            "--auto-endpoint" => auto_endpoint = true,
            "--dump-utterance" => dump_utterance = true,
            "--list-devices" => {
                list_devices();
                std::process::exit(0);
            }
            "--mic-selftest" => {
                selftest = Some(
                    argv.get(i + 1)
                        .and_then(|s| s.parse().ok())
                        .inspect(|_| i += 1)
                        .unwrap_or(5.0),
                );
            }
            "--voice-check" => voice_check = true,
            "--near" => {
                if let Some(v) = next(&mut i) {
                    near = v;
                }
            }
            "--turns" => {
                if let Some(v) = next(&mut i).and_then(|s| s.parse().ok()) {
                    turns = v;
                }
            }
            "--hold" => hold = next(&mut i).and_then(|s| s.parse().ok()),
            "--doctor" | "--paths" => {
                println!("{}", paths::doctor_report());
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
    opts.model_path = opts.model_path.or_else(paths::bundled_model);

    if let Some(seconds) = selftest {
        let out = paths::home_root().join("mic_selftest.wav");
        std::process::exit(headless::mic_selftest(
            opts.mic_device.clone(),
            seconds,
            out,
        ));
    }
    if voice_check {
        // A WAV "microphone" needs the whole file plus a moment of silence.
        let hold = hold.unwrap_or_else(|| {
            opts.mic_wav
                .as_ref()
                .and_then(|p| rpg_speech::read_wav(p).ok())
                .map(|a| a.audio_seconds() + 1.0)
                .unwrap_or(5.0)
        });
        std::process::exit(headless::voice_check(headless::VoiceCheck {
            opts,
            seed,
            near,
            turns,
            hold,
        }));
    }

    Args {
        seed,
        opts,
        auto_endpoint,
        dump_utterance,
    }
}

fn list_devices() {
    use rpg_speech::audio::{AudioCapture, AudioPlayback};
    for (label, list) in [
        ("Input", AudioCapture::list_devices()),
        ("Output", AudioPlayback::list_devices()),
    ] {
        println!("{label} devices:");
        match list {
            Ok(list) => {
                for d in list {
                    let mark = if d.is_default { " (default)" } else { "" };
                    println!("  - {}{mark}", d.name);
                }
            }
            Err(e) => println!("  cannot enumerate: {e}"),
        }
    }
}

fn print_help() {
    println!(
        "bevy-game — graphical, voice-native IRL client\n\
         \n\
         Usage: bevy-game [--seed N] [--model path.gguf] [--voice ID] [--no-voice]\n\
         \x20                [--mic NAME] [--auto-endpoint] [--dump-utterance]\n\
         \n\
         Diagnostics (no window):\n\
           --list-devices            list audio input/output devices\n\
           --mic-selftest [SECS]     record raw microphone audio to mic_selftest.wav\n\
           --voice-check             run full voice turns through the real runtime and\n\
                                     report simulation ticks per voice stage\n\
             [--mic-wav FILE]        speak FILE into the pipeline instead of the mic\n\
             [--near NPC] [--turns N] [--hold SECS]\n\
           --doctor                  show where assets, models and saves are found\n\
         \n\
         Controls:\n\
           WASD / arrows   walk (always available, also while an NPC is thinking)\n\
           V (hold)        push-to-talk: speak while held, release to send\n\
           C               cancel the current conversation turn / interrupt speech\n\
           Enter           type a line instead of speaking (Enter sends, Esc cancels)\n\
           Y / N / Tab     accept / reject / cycle offers\n\
           E               investigate the current location\n\
           F5 / F9         quick save / load\n\
           F3              developer diagnostics\n\
           F7              switch microphone\n\
           F8              microphone self-test (writes mic_selftest.wav)\n\
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
    let at = |left: Option<f32>, right: Option<f32>, top: Option<f32>, bottom: Option<f32>| Node {
        position_type: PositionType::Absolute,
        left: left.map(Val::Px).unwrap_or(Val::Auto),
        right: right.map(Val::Px).unwrap_or(Val::Auto),
        top: top.map(Val::Px).unwrap_or(Val::Auto),
        bottom: bottom.map(Val::Px).unwrap_or(Val::Auto),
        ..default()
    };

    commands.spawn((
        Text::new("IRL — Hidden Mafia"),
        text_font(24.0),
        TextColor(Color::srgb(0.92, 0.92, 0.96)),
        at(Some(14.0), None, Some(10.0), None),
        ClockText,
    ));
    commands.spawn((
        Text::new(""),
        text_font(15.0),
        TextColor(Color::srgb(0.70, 0.74, 0.82)),
        at(Some(14.0), None, Some(42.0), None),
        StatusText,
    ));
    // Voice state: recording indicator, live mic level, transcript.
    commands.spawn((
        Text::new(""),
        text_font(16.0),
        TextColor(Color::srgb(0.98, 0.55, 0.45)),
        at(Some(14.0), None, None, Some(176.0)),
        VoiceText,
    ));
    commands.spawn((
        Text::new(""),
        text_font(16.0),
        TextColor(Color::srgb(0.86, 0.87, 0.90)),
        at(Some(14.0), None, None, Some(40.0)),
        DialogueText,
    ));
    commands.spawn((
        Text::new(
            "WASD move   hold V talk   C cancel   Enter type   Y accept   N reject   Tab next offer\n\
             E investigate   F5 save   F9 load   F3 diagnostics   F7 mic   F8 mic test   Esc quit",
        ),
        text_font(13.0),
        TextColor(Color::srgb(0.45, 0.48, 0.55)),
        at(Some(14.0), None, None, Some(4.0)),
        HelpText,
    ));
    // Pending offers, right-aligned above the help line.
    commands.spawn((
        Text::new(""),
        text_font(14.0),
        TextColor(Color::srgb(0.98, 0.80, 0.25)),
        at(None, Some(14.0), None, Some(64.0)),
        OffersText,
    ));
    // Inventory and money, right-aligned at the top.
    commands.spawn((
        Text::new(""),
        text_font(14.0),
        TextColor(Color::srgb(0.72, 0.86, 0.98)),
        at(None, Some(14.0), Some(10.0), None),
        InventoryText,
    ));
    // Developer diagnostics overlay (hidden by default; F3 toggles it).
    commands.spawn((
        Text::new(""),
        text_font(12.0),
        TextColor(Color::srgb(0.55, 0.95, 0.75)),
        at(Some(340.0), None, Some(10.0), None),
        DiagnosticsText,
    ));
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Drain every snapshot the simulation published and keep only the newest.
fn poll_snapshot(mut sim: ResMut<SimHandle>) {
    let mut newest = None;
    if let Ok(guard) = sim.runtime.get_mut() {
        if let Some(rt) = guard.as_ref() {
            while let Ok(snap) = rt.snapshots.try_recv() {
                newest = Some(snap);
            }
        }
    }
    if newest.is_some() {
        sim.latest = newest;
    }
}

fn measure_fps(time: Res<Time>, mut ui: ResMut<UiState>) {
    let dt = time.delta_secs();
    if dt > 0.0 {
        let fps = 1.0 / dt;
        ui.fps = if ui.fps == 0.0 {
            fps
        } else {
            ui.fps * 0.95 + fps * 0.05
        };
    }
}

/// Text entry for typed dialogue. While typing, keys produce text instead
/// of gameplay commands — the only time movement is disabled.
fn handle_typing(
    mut keys_in: MessageReader<KeyboardInput>,
    sim: Res<SimHandle>,
    mut ui: ResMut<UiState>,
) {
    let mut events: Vec<KeyboardInput> = keys_in.read().cloned().collect();
    ui.typing_closed = false;
    let Some(mut line) = ui.typing.take() else {
        return;
    };
    let mut open = true;
    for ev in events.drain(..) {
        if ev.state != ButtonState::Pressed {
            continue;
        }
        match ev.key_code {
            KeyCode::Enter | KeyCode::NumpadEnter => {
                let text = line.trim().to_string();
                if !text.is_empty() {
                    sim.send(Command::Say(text));
                }
                open = false;
                break;
            }
            KeyCode::Escape => {
                open = false;
                break;
            }
            KeyCode::Backspace => {
                line.pop();
            }
            _ => {
                if let Some(text) = &ev.text {
                    line.extend(text.chars().filter(|c| !c.is_control()));
                }
            }
        }
    }
    if open {
        ui.typing = Some(line);
    } else {
        ui.typing_closed = true;
    }
}

/// Translate keyboard input into simulation commands. World movement is
/// only *requested* here; the simulation validates and commits it. Nothing
/// is gated on the voice pipeline: the world and the player stay live while
/// an NPC listens, thinks or speaks.
fn handle_input(
    keys: Res<ButtonInput<KeyCode>>,
    sim: Res<SimHandle>,
    mut ui: ResMut<UiState>,
    mut exit: MessageWriter<AppExit>,
) {
    if keys.just_pressed(KeyCode::F3) {
        ui.diagnostics = !ui.diagnostics;
    }
    // Push-to-talk tracks the physical key so a release is never lost, even
    // if the key went up while typing started.
    let v_down = keys.pressed(KeyCode::KeyV) && ui.typing.is_none();
    if v_down != ui.ptt_held {
        ui.ptt_held = v_down;
        sim.send(Command::PushToTalk(v_down));
    }
    if ui.typing.is_some() || ui.typing_closed {
        return;
    }
    if keys.just_pressed(KeyCode::Escape) {
        exit.write(AppExit::Success);
        return;
    }
    if keys.just_pressed(KeyCode::Enter) || keys.just_pressed(KeyCode::NumpadEnter) {
        ui.typing = Some(String::new());
        return;
    }
    if keys.just_pressed(KeyCode::KeyC) {
        sim.send(Command::CancelVoice);
    }
    if keys.just_pressed(KeyCode::F7) {
        sim.send(Command::SelectMic(MicSelection::Next));
    }
    if keys.just_pressed(KeyCode::F8) {
        let path = paths::home_root().join("mic_selftest.wav");
        sim.send(Command::MicSelfTest {
            path: path.to_string_lossy().into_owned(),
            seconds: 4.0,
        });
    }

    // --- Save / load ---------------------------------------------------
    if keys.just_pressed(KeyCode::F5) {
        sim.send(Command::Save(default_save_path()));
    }
    if keys.just_pressed(KeyCode::F9) {
        sim.send(Command::Load(default_save_path()));
    }

    let Some(snap) = sim.latest.as_ref() else {
        return;
    };

    // --- Offers --------------------------------------------------------
    // Player-addressed offers are the only ones the player can accept.
    let pending: Vec<&OfferView> = snap
        .offers
        .iter()
        .filter(|o| o.to_player && o.state == "pending")
        .collect();
    if keys.just_pressed(KeyCode::Tab) && !pending.is_empty() {
        ui.selected_offer = (ui.selected_offer + 1) % pending.len();
    }
    let selected = pending
        .get(ui.selected_offer.min(pending.len().saturating_sub(1)))
        .map(|o| OfferId(o.id));
    if keys.just_pressed(KeyCode::KeyY) {
        if let Some(id) = selected {
            sim.send(Command::AcceptOffer(id));
            ui.selected_offer = 0;
        }
    }
    if keys.just_pressed(KeyCode::KeyN) {
        if let Some(id) = selected {
            sim.send(Command::RejectOffer(id));
            ui.selected_offer = 0;
        }
    }

    // --- Investigate the current location ------------------------------
    if keys.just_pressed(KeyCode::KeyE) {
        let loc = LocationId::from(snap.player_location.as_str());
        sim.send(Command::Investigate(InvestigateTarget::Location(loc)));
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
    sim.send(Command::MoveTo(target));
}

/// Stop the runtime when the app exits, so the microphone, speaker and model
/// are released instead of being killed with the process.
fn on_app_exit(mut exits: MessageReader<AppExit>, sim: Res<SimHandle>) {
    if exits.read().next().is_none() {
        return;
    }
    let runtime = sim.runtime.lock().ok().and_then(|mut g| g.take());
    if let Some(rt) = runtime {
        let report = rt.shutdown(Duration::from_millis(1500));
        if !report.detached.is_empty() {
            eprintln!("[shutdown] still busy at exit: {:?}", report.detached);
        }
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
fn update_ui(sim: Res<SimHandle>, ui: Res<UiState>, mut texts: HudTextQuery) {
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

    for (mut text, clock, status, voice, dialogue) in &mut texts {
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
                Some(id) => names.get(id).copied().unwrap_or(id).to_string(),
                None => "nobody in earshot".to_string(),
            };
            text.0 = format!(
                "{}  |  tick {}  |  at {}  |  near {}\n{} ({})  |  decisions: {} model / {} canonical / {} fallback",
                snap.status,
                snap.tick,
                here,
                nearby,
                snap.engine_name,
                snap.model_id,
                snap.model_decisions,
                snap.canonical_decisions,
                snap.fallback_decisions,
            );
        } else if voice.is_some() {
            text.0 = voice_hud(snap, &ui, &names);
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

/// The voice panel: what the pipeline is doing right now, a live input
/// meter (so "is the microphone hearing me?" is answerable at a glance) and
/// the last thing the recognizer understood.
fn voice_hud(snap: &Snapshot, ui: &UiState, names: &HashMap<&str, &str>) -> String {
    let v = &snap.voice;
    if let Some(line) = &ui.typing {
        return format!("SAY: {line}_   (Enter sends, Esc cancels)");
    }
    let stage = match v.stage {
        VoiceStage::Idle => "voice: ready (hold V)".to_string(),
        VoiceStage::Listening => format!("● REC {:.1}s — release V to send", v.stage_seconds),
        other => {
            let who = v
                .npc
                .as_deref()
                .map(|id| names.get(id).copied().unwrap_or(id))
                .unwrap_or("");
            format!(
                "{} {who} ({:.1}s) — C cancels",
                other.label(),
                v.stage_seconds
            )
        }
    };
    let meter = level_meter(v.mic_peak);
    let speech = if v.speech_detected { "  speech" } else { "" };
    let mic = match &v.mic {
        Readiness::Ready(_) => format!("mic {meter} peak {:.2}{speech}", v.mic_peak),
        Readiness::Loading => "mic: starting…".into(),
        Readiness::Unavailable(e) => format!("mic UNAVAILABLE: {e} (F7 switch, F8 test)"),
    };
    let heard = v
        .last_transcript
        .as_deref()
        .map(|t| format!("\nheard: \"{t}\""))
        .unwrap_or_default();
    let mut out = format!("{stage}\n{mic}{heard}");
    for (label, r) in [("speech recognition", &v.asr), ("voice", &v.tts)] {
        match r {
            Readiness::Loading => out.push_str(&format!("\n{label}: loading…")),
            Readiness::Unavailable(e) => out.push_str(&format!("\n{label} UNAVAILABLE: {e}")),
            Readiness::Ready(_) => {}
        }
    }
    out
}

/// A 16-step bar for a 0..1 peak level (log-scaled so speech is visible).
fn level_meter(peak: f32) -> String {
    let db = if peak <= 1e-4 {
        -80.0
    } else {
        20.0 * peak.log10()
    };
    let steps = (((db + 60.0) / 60.0).clamp(0.0, 1.0) * 16.0).round() as usize;
    format!("[{}{}]", "|".repeat(steps), ".".repeat(16 - steps))
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

/// Developer diagnostics overlay; hidden unless F3 has been pressed. Laid
/// out along the pipeline so a fault is easy to place: capture → VAD → ASR
/// → inference → simulation → playback.
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
        text.0 = diagnostics_text(snap, ui.fps);
    }
}

fn diagnostics_text(snap: &Snapshot, fps: f32) -> String {
    let d = &snap.diagnostics;
    let v = &snap.voice;
    let ms = |x: Option<f32>| {
        x.map(|m| format!("{m:.0} ms"))
            .unwrap_or_else(|| "—".into())
    };
    let id = |x: Option<u64>| x.map(|m| format!("#{m}")).unwrap_or_else(|| "—".into());
    let commitments: Vec<String> = snap
        .commitments
        .iter()
        .map(|c| format!("  {} — {} ({})", c.kind, c.terms, c.state))
        .collect();
    let list = |v: &[String]| {
        if v.is_empty() {
            "—".to_string()
        } else {
            v.join("\n  ")
        }
    };
    format!(
        "DIAGNOSTICS\n\
         SIMULATION  tick {}  {:.1}/{:.0} tps  delay {:.1} ms  max lag {:.1} ms  skipped {}  | render {:.0} fps\n\
         CAPTURE     {}\n\
         \x20           {:.0} samples/s  rms {:.4}  peak {:.4}  dropped {}  rec {:.1}s\n\
         VAD         {}\n\
         ASR         {}  last {}  heard {:?}\n\
         INFERENCE   {} ({})  conv {}  request {}  running {}  queue {}  wait {}  infer {}\n\
         TTS         {}  first audio {}  synth {}\n\
         PLAYBACK    {}  underruns {}\n\
         QUEUES      asr {}  llm {}  tts {}\n\
         STATE       {} ({:.1}s)  cancelled {}  stale {}  timeouts {}  refused V {}  regenerated {}  invalidated {}\n\
         \x20           recent: {}\n\
         decisions   {} model / {} canonical / {} fallback\n\n\
         Perception:\n  {}\n\nNearby relationship:\n  {}\n\nNearby beliefs (developer view):\n  {}\n\nCommitments:\n{}\n\nRecent events:\n  {}",
        snap.tick,
        d.ticks_per_second,
        d.target_ticks_per_second,
        d.tick_delay_ms,
        d.max_tick_lag_ms,
        d.skipped_ticks,
        fps,
        d.mic_device,
        d.mic_samples_per_second,
        d.mic_rms,
        d.mic_peak,
        d.mic_dropped_chunks,
        d.recording_seconds,
        if d.vad_speaking { "SPEECH" } else { "silence" },
        v.asr.describe(),
        ms(d.asr_latency_ms),
        d.last_transcript.as_deref().unwrap_or("—"),
        snap.engine_name,
        snap.model_id,
        id(d.conversation),
        id(d.request_id),
        id(d.active_inference),
        d.infer_queue,
        ms(d.llm_queue_ms),
        ms(d.llm_infer_ms),
        v.tts.describe(),
        ms(d.tts_first_audio_ms),
        ms(d.tts_synth_ms),
        d.playback,
        d.playback_underruns,
        d.asr_queue,
        d.infer_queue,
        d.tts_queue,
        v.stage.label(),
        v.stage_seconds,
        d.cancelled,
        d.stale_responses,
        d.timeouts,
        d.refused_ptt,
        d.regenerations,
        d.invalidated,
        d.transitions.join(", "),
        snap.model_decisions,
        snap.canonical_decisions,
        snap.fallback_decisions,
        snap.perception.join("\n  "),
        snap.nearby_relationship.as_deref().unwrap_or("—"),
        list(&snap.nearby_beliefs),
        commitments.join("\n"),
        list(&snap.recent_events),
    )
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
    use rpg_runtime::LocationView;

    fn snapshot_with(locations: Vec<LocationView>) -> Snapshot {
        let mut core = rpg_runtime::SimCore::new(
            rpg_eval::Session::new(2026),
            rpg_runtime::CoreConfig::default(),
            std::sync::Arc::new(rpg_runtime::Telemetry::new()),
        );
        core.tick(Duration::ZERO);
        let mut snap = core.snapshot();
        snap.locations = locations;
        snap
    }

    #[test]
    fn town_center_is_the_bounding_box_midpoint() {
        let snap = snapshot_with(vec![
            LocationView {
                id: "a".into(),
                name: "A".into(),
                x: 0.0,
                y: 0.0,
                w: 10.0,
                h: 4.0,
                private: false,
            },
            LocationView {
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

    #[test]
    fn level_meter_spans_silence_to_full_scale() {
        assert_eq!(level_meter(0.0), format!("[{}]", ".".repeat(16)));
        assert_eq!(level_meter(1.0), format!("[{}]", "|".repeat(16)));
        assert!(level_meter(0.05).contains('|'));
    }

    #[test]
    fn hud_shows_recording_and_unavailable_mic() {
        let mut snap = snapshot_with(vec![]);
        let ui = UiState::default();
        snap.voice.stage = VoiceStage::Listening;
        snap.voice.mic = Readiness::Ready("mic".into());
        let hud = voice_hud(&snap, &ui, &HashMap::new());
        assert!(hud.contains("REC"), "{hud}");
        snap.voice.mic = Readiness::Unavailable("no device".into());
        let hud = voice_hud(&snap, &ui, &HashMap::new());
        assert!(hud.contains("UNAVAILABLE"), "{hud}");
    }

    #[test]
    fn diagnostics_cover_every_pipeline_stage() {
        let snap = snapshot_with(vec![]);
        let text = diagnostics_text(&snap, 60.0);
        for section in [
            "SIMULATION",
            "CAPTURE",
            "VAD",
            "ASR",
            "INFERENCE",
            "TTS",
            "PLAYBACK",
            "QUEUES",
            "STATE",
        ] {
            assert!(text.contains(section), "missing {section}");
        }
    }
}
