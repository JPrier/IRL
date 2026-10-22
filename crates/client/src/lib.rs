//! # irl_client
//!
//! The one Bevy presentation of IRL, shared by the native (Windows/Linux)
//! client and the browser (WASM/WebGL2) client.
//!
//! The client is a *renderer and input device*: it sends
//! [`rpg_runtime::Command`]s through a platform [`Link`] and draws the
//! immutable [`rpg_runtime::Snapshot`]s that come back. It never mutates
//! world state and never waits on the simulation; `rpg_sim` never depends on
//! Bevy. The platform decides only how the authoritative `SimCore` is driven:
//!
//! * native — `rpg_runtime::spawn`: a fixed-tick simulation thread plus
//!   microphone/ASR/inference/TTS worker threads;
//! * browser — the same `SimCore`, ticked from the browser event loop, with
//!   inference as `fetch` futures.
//!
//! Layout: the town is drawn as flat rectangles in world units, scaled by
//! [`PIXELS_PER_UNIT`] and centred on the bounding box of the locations. The
//! player is drawn green and on top; a nearby, talkable NPC is highlighted.

use std::collections::{HashMap, HashSet};

use bevy::input::keyboard::KeyboardInput;
use bevy::input::ButtonState;
use bevy::prelude::*;
use rpg_domain::action::InvestigateTarget;
use rpg_domain::geom::Vec2;
use rpg_domain::ids::{LocationId, OfferId};
use rpg_runtime::{
    Command, KindView, LineKind, MicSelection, OfferView, Readiness, Snapshot, StatusView,
    VoiceStage,
};

/// Pixels per world unit. The town spans ~34x18 units, so this fills a
/// 1280x800 window without scrolling.
pub const PIXELS_PER_UNIT: f32 = 20.0;
/// Draw size of an actor, in world units.
const ACTOR_SIZE: f32 = 0.9;
/// How far one WASD key press walks, in world units.
pub const MOVE_STEP: f32 = 3.0;
/// How many dialogue lines to show.
const DIALOGUE_LINES: usize = 7;

// ---------------------------------------------------------------------------
// The platform seam
// ---------------------------------------------------------------------------

/// How the client reaches the authoritative simulation on this platform.
///
/// Implementations must never block: `send` enqueues, `poll` returns the
/// newest snapshot (driving the browser runtime's due ticks first, when
/// that is how the platform runs the simulation).
pub trait Link {
    fn send(&mut self, cmd: Command);
    /// The newest snapshot published since the last call, if any.
    fn poll(&mut self) -> Option<Snapshot>;
    /// Quick-save (native: a JSON file; browser: browser storage).
    fn quick_save(&mut self);
    /// Quick-load from wherever `quick_save` writes.
    fn quick_load(&mut self);
    /// Record a raw microphone self-test (native only; the platform picks
    /// where the WAV goes).
    fn mic_self_test(&mut self) {}
    /// Release devices and stop the runtime when the app exits.
    fn shutdown(&mut self) {}
}

/// The platform link, owned by the main thread (a non-send resource).
pub struct ClientLink(pub Box<dyn Link>);

/// The newest snapshot the renderer draws.
#[derive(Resource, Default)]
pub struct Latest(pub Option<Snapshot>);

/// Which platform the shared client is running on. It changes only key
/// bindings and help text — never rendering or gameplay.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClientPlatform {
    Native,
    Browser,
}

impl ClientPlatform {
    pub fn help(self) -> &'static str {
        match self {
            ClientPlatform::Native => {
                "WASD move   hold V talk   C cancel   Enter type   Y accept   N reject   Tab next offer\n\
                 E investigate   F5 save   F9 load   F3 diagnostics   F7 mic   F8 mic test   Esc quit"
            }
            ClientPlatform::Browser => {
                "WASD move   Enter type   hold V talk (if supported)   C cancel   Y accept   N reject   Tab next offer\n\
                 E investigate   F5 / K save   F9 / L load   F3 diagnostics"
            }
        }
    }

    fn quits_on_escape(self) -> bool {
        self == ClientPlatform::Native
    }

    fn has_mic_keys(self) -> bool {
        self == ClientPlatform::Native
    }
}

/// Resource form of the platform.
#[derive(Resource, Clone, Copy)]
pub struct Platform(pub ClientPlatform);

/// Adds the shared IRL presentation to an app that already has Bevy's
/// default plugins.
pub struct IrlClientPlugin {
    pub platform: ClientPlatform,
}

impl Plugin for IrlClientPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ClearColor(Color::srgb(0.055, 0.06, 0.08)))
            .insert_resource(Platform(self.platform))
            .init_resource::<Latest>()
            .init_resource::<Spawned>()
            .init_resource::<UiState>()
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
            );
    }
}

/// Build a complete IRL app around a platform link.
pub fn app(link: Box<dyn Link>, platform: ClientPlatform, window: Window) -> App {
    let mut app = App::new();
    app.insert_non_send(ClientLink(link))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(window),
            ..default()
        }))
        .add_plugins(IrlClientPlugin { platform });
    app
}

// ---------------------------------------------------------------------------
// Resources and components
// ---------------------------------------------------------------------------

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
pub struct UiState {
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
    pub fps: f32,
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
// Setup
// ---------------------------------------------------------------------------

fn setup(mut commands: Commands, platform: Res<Platform>) {
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
        Text::new("IRL - Hidden Mafia"),
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
        Text::new(platform.0.help()),
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

/// Take the newest snapshot the simulation published.
fn poll_snapshot(mut link: NonSendMut<ClientLink>, mut latest: ResMut<Latest>) {
    if let Some(snap) = link.0.poll() {
        latest.0 = Some(snap);
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
    mut link: NonSendMut<ClientLink>,
    mut ui: ResMut<UiState>,
) {
    let mut events: Vec<KeyboardInput> = keys_in.read().cloned().collect();
    ui.typing_closed = false;
    if ui.typing.is_none() {
        // Open the text box here when Enter arrives, so characters typed in
        // the same frame (slow frames, fast typists) land in it instead of
        // being dropped. `handle_input` sees the open box and stays out.
        let enter = events.iter().position(|ev| {
            ev.state == ButtonState::Pressed
                && matches!(ev.key_code, KeyCode::Enter | KeyCode::NumpadEnter)
        });
        match enter {
            Some(i) => {
                events.drain(..=i);
                ui.typing = Some(String::new());
            }
            None => return,
        }
    }
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
                    link.0.send(Command::Say(text));
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
/// is gated on the voice pipeline or on inference: the world and the player
/// stay live while an NPC listens, thinks or speaks.
fn handle_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut link: NonSendMut<ClientLink>,
    latest: Res<Latest>,
    platform: Res<Platform>,
    mut ui: ResMut<UiState>,
    mut exit: MessageWriter<AppExit>,
) {
    let platform = platform.0;
    let link = &mut link.0;
    if keys.just_pressed(KeyCode::F3) {
        ui.diagnostics = !ui.diagnostics;
    }
    // Push-to-talk tracks the physical key so a release is never lost, even
    // if the key went up while typing started.
    let v_down = keys.pressed(KeyCode::KeyV) && ui.typing.is_none();
    if v_down != ui.ptt_held {
        ui.ptt_held = v_down;
        link.send(Command::PushToTalk(v_down));
    }
    if ui.typing.is_some() || ui.typing_closed {
        return;
    }
    if keys.just_pressed(KeyCode::Escape) && platform.quits_on_escape() {
        exit.write(AppExit::Success);
        return;
    }
    if keys.just_pressed(KeyCode::Enter) || keys.just_pressed(KeyCode::NumpadEnter) {
        ui.typing = Some(String::new());
        return;
    }
    if keys.just_pressed(KeyCode::KeyC) {
        link.send(Command::CancelVoice);
    }
    if platform.has_mic_keys() {
        if keys.just_pressed(KeyCode::F7) {
            link.send(Command::SelectMic(MicSelection::Next));
        }
        if keys.just_pressed(KeyCode::F8) {
            link.mic_self_test();
        }
    }

    // --- Save / load ---------------------------------------------------
    let browser = platform == ClientPlatform::Browser;
    if keys.just_pressed(KeyCode::F5) || (browser && keys.just_pressed(KeyCode::KeyK)) {
        link.quick_save();
    }
    if keys.just_pressed(KeyCode::F9) || (browser && keys.just_pressed(KeyCode::KeyL)) {
        link.quick_load();
    }

    let Some(snap) = latest.0.as_ref() else {
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
            link.send(Command::AcceptOffer(id));
            ui.selected_offer = 0;
        }
    }
    if keys.just_pressed(KeyCode::KeyN) {
        if let Some(id) = selected {
            link.send(Command::RejectOffer(id));
            ui.selected_offer = 0;
        }
    }

    // --- Investigate the current location ------------------------------
    if keys.just_pressed(KeyCode::KeyE) {
        let loc = LocationId::from(snap.player_location.as_str());
        link.send(Command::Investigate(InvestigateTarget::Location(loc)));
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
    link.send(Command::MoveTo(step_target(snap, dir)));
}

/// Where one movement key press asks the player to walk.
pub fn step_target(snap: &Snapshot, dir: Vec2) -> Vec2 {
    Vec2::new(
        snap.player_pos.x + dir.x * MOVE_STEP,
        snap.player_pos.y + dir.y * MOVE_STEP,
    )
}

/// Stop the runtime when the app exits, so the microphone, speaker and model
/// are released instead of being killed with the process.
fn on_app_exit(mut exits: MessageReader<AppExit>, mut link: NonSendMut<ClientLink>) {
    if exits.read().next().is_none() {
        return;
    }
    link.0.shutdown();
}

/// Spawn and update one sprite per location and per actor, so the rendered
/// world matches the authoritative snapshot exactly.
fn sync_world(
    mut commands: Commands,
    latest: Res<Latest>,
    mut spawned: ResMut<Spawned>,
    mut locations: LocationSpriteQuery,
    mut actors: ActorSpriteQuery,
) {
    let Some(snap) = latest.0.as_ref() else {
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
fn update_ui(latest: Res<Latest>, ui: Res<UiState>, mut texts: HudTextQuery) {
    let Some(snap) = latest.0.as_ref() else {
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
                "IRL - Hidden Mafia          Day {}  {}",
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
            text.0 = dialogue_text(snap, &names);
        }
    }
}

/// The last few dialogue lines with display names.
pub fn dialogue_text(snap: &Snapshot, names: &HashMap<&str, &str>) -> String {
    let start = snap.dialogue.len().saturating_sub(DIALOGUE_LINES);
    snap.dialogue[start..]
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
        .collect::<Vec<_>>()
        .join("\n")
}

/// The voice panel: what the pipeline is doing right now, a live input
/// meter (so "is the microphone hearing me?" is answerable at a glance) and
/// the last thing the recognizer understood.
pub fn voice_hud(snap: &Snapshot, ui: &UiState, names: &HashMap<&str, &str>) -> String {
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
    // All voice components off for the same reason (`--no-voice`, browser
    // voice disabled): one line, not three.
    if let (Readiness::Unavailable(a), Readiness::Unavailable(b), Readiness::Unavailable(c)) =
        (&v.mic, &v.asr, &v.tts)
    {
        if a == b && b == c {
            return format!("voice off: {a}");
        }
    }
    let meter = level_meter(v.mic_peak);
    let speech = if v.speech_detected { "  speech" } else { "" };
    let mic = match &v.mic {
        Readiness::Ready(_) => format!("mic {meter} peak {:.2}{speech}", v.mic_peak),
        Readiness::Loading => "mic: starting…".into(),
        Readiness::Unavailable(e) => format!("mic UNAVAILABLE: {e} (Enter to type)"),
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
pub fn level_meter(peak: f32) -> String {
    let db = if peak <= 1e-4 {
        -80.0
    } else {
        20.0 * peak.log10()
    };
    let steps = (((db + 60.0) / 60.0).clamp(0.0, 1.0) * 16.0).round() as usize;
    format!("[{}{}]", "|".repeat(steps), ".".repeat(16 - steps))
}

/// Render the player-addressed offers from the latest snapshot.
fn update_offers(latest: Res<Latest>, mut texts: Query<&mut Text, With<OffersText>>) {
    let Some(snap) = latest.0.as_ref() else {
        return;
    };
    for mut text in &mut texts {
        text.0 = offers_text(snap);
    }
}

/// The pending offers addressed to the player.
pub fn offers_text(snap: &Snapshot) -> String {
    snap.offers
        .iter()
        .filter(|o| o.to_player && o.state == "pending")
        .map(|o| {
            format!(
                "#{} from {} — give:{} want:{} ({})",
                o.id, o.from, o.give, o.want, o.state
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Render the player's inventory and money.
fn update_inventory(latest: Res<Latest>, mut texts: Query<&mut Text, With<InventoryText>>) {
    let Some(snap) = latest.0.as_ref() else {
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
    latest: Res<Latest>,
    ui: Res<UiState>,
    mut texts: Query<&mut Text, With<DiagnosticsText>>,
) {
    let Some(snap) = latest.0.as_ref() else {
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

fn usd(x: Option<f64>) -> String {
    x.map(|c| format!("${c:.5}")).unwrap_or_else(|| "—".into())
}

/// The F3 overlay text. Never contains a credential: snapshots carry none.
pub fn diagnostics_text(snap: &Snapshot, fps: f32) -> String {
    let d = &snap.diagnostics;
    let v = &snap.voice;
    let inf = &d.inference;
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
         PLATFORM    {}  | provider {} ({})\n\
         SIMULATION  tick {}  {:.1}/{:.0} tps  delay {:.1} ms  max lag {:.1} ms  skipped {}  | render {:.0} fps\n\
         CAPTURE     {}\n\
         \x20           {:.0} samples/s  rms {:.4}  peak {:.4}  dropped {}  rec {:.1}s\n\
         VAD         {}\n\
         ASR         {}  last {}  heard {:?}\n\
         INFERENCE   {} ({})  conv {}  request {}  running {}  pending {}  queue {}  wait {}  infer {}  first token {}  streamed {}\n\
         REMOTE      requests {}  ok {}  errors {}  tokens {} in / {} out  cost reported {} / estimated {}  last {}\n\
         \x20           {}  last error: {}\n\
         TTS         {}  first audio {}  synth {}\n\
         PLAYBACK    {}  underruns {}\n\
         QUEUES      asr {}  llm {}  tts {}\n\
         STATE       {} ({:.1}s)  cancelled {}  stale {}  timeouts {}  refused V {}  regenerated {}  invalidated {}  rejected {}\n\
         \x20           recent: {}\n\
         decisions   {} model / {} canonical / {} fallback  | last: {}\n\n\
         Perception:\n  {}\n\nNearby relationship:\n  {}\n\nNearby beliefs (developer view):\n  {}\n\nCommitments:\n{}\n\nRecent events:\n  {}",
        if d.platform.is_empty() { "—" } else { &d.platform },
        snap.engine_name,
        if snap.model_id.is_empty() { "—" } else { &snap.model_id },
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
        d.pending_inference,
        d.infer_queue,
        ms(d.llm_queue_ms),
        ms(d.llm_infer_ms),
        ms(d.llm_first_token_ms),
        d.llm_stream_chunks
            .map(|c| c.to_string())
            .unwrap_or_else(|| "—".into()),
        inf.requests,
        inf.completed,
        inf.errors,
        inf.prompt_tokens,
        inf.completion_tokens,
        usd(inf.reported_cost_usd),
        usd(inf.estimated_cost_usd),
        ms(inf.last_latency_ms),
        inf.last_remote.as_deref().unwrap_or("—"),
        inf.last_error.as_deref().unwrap_or("—"),
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
        d.rejected_actions,
        d.transitions.join(", "),
        snap.model_decisions,
        snap.canonical_decisions,
        snap.fallback_decisions,
        d.last_decision.as_deref().unwrap_or("—"),
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
pub fn town_center(snap: &Snapshot) -> Vec2 {
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
pub fn actor_color(kind: KindView, status: StatusView, nearby: bool) -> Color {
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
    use std::time::Duration;

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
        let off = Readiness::Unavailable("disabled".into());
        snap.voice.stage = VoiceStage::Idle;
        (snap.voice.mic, snap.voice.asr, snap.voice.tts) = (off.clone(), off.clone(), off);
        assert_eq!(
            voice_hud(&snap, &ui, &HashMap::new()),
            "voice off: disabled"
        );
    }

    #[test]
    fn diagnostics_cover_every_pipeline_stage() {
        let snap = snapshot_with(vec![]);
        let text = diagnostics_text(&snap, 60.0);
        for section in [
            "PLATFORM",
            "SIMULATION",
            "CAPTURE",
            "VAD",
            "ASR",
            "INFERENCE",
            "REMOTE",
            "TTS",
            "PLAYBACK",
            "QUEUES",
            "STATE",
            "rejected",
            "decisions",
        ] {
            assert!(text.contains(section), "missing {section}");
        }
    }

    #[test]
    fn movement_steps_from_the_snapshot_position() {
        let mut snap = snapshot_with(vec![]);
        snap.player_pos = Vec2::new(10.0, 5.0);
        assert_eq!(
            step_target(&snap, Vec2::new(1.0, 0.0)),
            Vec2::new(10.0 + MOVE_STEP, 5.0)
        );
    }

    #[test]
    fn help_differs_only_in_platform_keys() {
        assert!(ClientPlatform::Native.help().contains("Esc quit"));
        assert!(!ClientPlatform::Browser.help().contains("Esc quit"));
        for p in [ClientPlatform::Native, ClientPlatform::Browser] {
            for key in ["WASD", "Enter type", "Y accept", "F3 diagnostics"] {
                assert!(p.help().contains(key), "{p:?} help lacks {key}");
            }
        }
    }
}
