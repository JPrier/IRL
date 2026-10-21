//! Simulation + voice worker for the graphical client.
//!
//! The authoritative [`Session`] lives here, on a dedicated worker thread. The
//! Bevy main thread never touches simulation state directly: it sends
//! [`Command`]s and renders the immutable [`Snapshot`]s published back. This
//! keeps the simulation independent of the renderer (a hard project rule) and
//! means a slow LLM turn never blocks the window.
//!
//! ```text
//!   Bevy main thread                     worker thread
//!   ----------------                     -------------
//!   input -> Command  ──(channel)──►  Session + SherpaTts/Asr + cpal
//!   render Snapshot  ◄──(channel)──  build_snapshot(&Session)
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use rpg_domain::action::{Action, InvestigateTarget, Loudness};
use rpg_domain::geom::Vec2;
use rpg_domain::ids::{ActorId, OfferId, PLAYER};
use rpg_domain::ports::{InferenceEngine, SpeechRecognizer, SpeechRequest, SpeechSynthesizer};
use rpg_domain::world::{ActorKind, ActorStatus};
use rpg_eval::Session;
use rpg_inference::{LlamaEngine, ModelProfile};
use rpg_speech::assets::SpeechAssets;
use rpg_speech::audio::{
    AudioCapture, AudioPlayback, CaptureConfig, LinearResampler, PlaybackConfig,
};
use rpg_speech::{SherpaAsr, SherpaTts, SpeechSegmentAudio, VadSegmenter};

/// Interaction radius, in world units. Normal speech carries ~6 units, but the
/// simulation also applies occlusion and location factors, so this stays
/// conservative: the UI never offers a target the validator would reject.
pub const TALK_RADIUS: f32 = 3.5;

/// How long a single push-to-talk capture may run before it gives up.
const MAX_CAPTURE_SECS: f32 = 6.0;

// ---------------------------------------------------------------------------
// View types (plain data; no simulation types leak into Bevy components)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KindView {
    Player,
    Npc,
    Background,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatusView {
    Alive,
    Dead,
    Departed,
}

#[derive(Clone, Debug)]
pub struct ActorView {
    pub id: String,
    pub name: String,
    pub pos: Vec2,
    pub kind: KindView,
    pub status: StatusView,
    pub location: String,
    pub nearby: bool,
}

#[derive(Clone, Debug)]
pub struct LocationView {
    pub id: String,
    pub name: String,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub private: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineKind {
    Player,
    Npc,
    System,
}

#[derive(Clone, Debug)]
pub struct DialogueLine {
    pub speaker: String,
    pub text: String,
    pub kind: LineKind,
}

/// One stack of items in the player's inventory, for display.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ItemView {
    pub name: String,
    pub qty: u32,
}

/// A pending or resolved offer between two actors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OfferView {
    pub id: u64,
    pub from: String,
    pub to: String,
    /// True when the player is the recipient, so the UI only lets the player
    /// accept/reject offers that were actually addressed to them.
    pub to_player: bool,
    pub give: String,
    pub want: String,
    pub price: i64,
    pub note: String,
    pub state: String,
}

/// A commitment (favour, debt, promise) tracked by the social layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitmentView {
    pub id: u64,
    pub kind: String,
    pub debtor: String,
    pub creditor: String,
    pub terms: String,
    pub state: String,
}

/// Everything the renderer needs for one frame.
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub tick: u64,
    pub day: u32,
    pub minute_of_day: u32,
    pub locations: Vec<LocationView>,
    pub actors: Vec<ActorView>,
    pub player_pos: Vec2,
    pub player_location: String,
    pub nearby: Option<String>,
    pub dialogue: Vec<DialogueLine>,
    pub engine_name: String,
    pub model_id: String,
    pub model_decisions: usize,
    pub canonical_decisions: usize,
    pub fallback_decisions: usize,
    pub busy: bool,
    pub status: String,
    pub voice_error: Option<String>,
    /// Developer-diagnostics data. Rendered only in the F3 overlay, never in
    /// the normal player UI (an NPC's beliefs are not player knowledge).
    pub recent_events: Vec<String>,
    pub nearby_beliefs: Vec<String>,
    pub nearby_relationship: Option<String>,
    pub perception: Vec<String>,
    pub last_turn_timing: Option<String>,
    pub offers: Vec<OfferView>,
    pub commitments: Vec<CommitmentView>,
    pub inventory: Vec<ItemView>,
    pub money: i64,
}

impl Snapshot {
    /// "HH:MM" from the simulation clock.
    pub fn clock(&self) -> String {
        format!(
            "{:02}:{:02}",
            self.minute_of_day / 60,
            self.minute_of_day % 60
        )
    }
}

/// Commands the UI sends to the worker.
#[derive(Clone, Debug)]
pub enum Command {
    /// Walk to a world position (validated by the simulation).
    MoveTo(Vec2),
    /// Capture the microphone, transcribe, decide, and speak the reply.
    VoiceTurn,
    /// Accept a pending offer from an NPC.
    AcceptOffer(OfferId),
    /// Reject a pending offer from an NPC.
    RejectOffer(OfferId),
    /// Investigate a location, actor, or item.
    Investigate(InvestigateTarget),
    /// Save the current canonical state to a JSON file.
    Save(String),
    /// Load canonical state from a JSON save file.
    Load(String),
    Quit,
}

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

pub struct WorkerConfig {
    pub seed: u64,
    pub model_path: Option<String>,
    pub voice: String,
    /// Disable neural voice entirely (rendering-only smoke test).
    pub no_voice: bool,
}

pub struct WorkerHandle {
    pub cmd_tx: Sender<Command>,
    pub snap_rx: Receiver<Snapshot>,
    /// Set by the UI to abort an in-flight capture or turn promptly.
    pub cancel: Arc<AtomicBool>,
}

/// Spawn the worker thread. It owns the session and all voice handles.
pub fn spawn_worker(cfg: WorkerConfig) -> WorkerHandle {
    let (cmd_tx, cmd_rx) = channel();
    let (snap_tx, snap_rx) = channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let worker_cancel = Arc::clone(&cancel);
    thread::Builder::new()
        .name("irl-sim-worker".into())
        .spawn(move || {
            let mut worker = Worker::new(cfg);
            worker.cancel = worker_cancel;
            worker.run(cmd_rx, snap_tx);
        })
        .expect("spawn sim worker");
    WorkerHandle {
        cmd_tx,
        snap_rx,
        cancel,
    }
}

struct Worker {
    session: Session,
    tts: Option<SherpaTts>,
    asr: Option<SherpaAsr>,
    vad: Option<VadSegmenter>,
    capture: Option<AudioCapture>,
    playback: Option<AudioPlayback>,
    voice: String,
    dialogue: Vec<DialogueLine>,
    status: String,
    /// True only while a voice turn owns the microphone/speaker. Movement is
    /// refused while busy so queued commands cannot fight the active turn.
    busy: bool,
    voice_error: Option<String>,
    cancel: Arc<AtomicBool>,
    /// Timing of the most recent voice turn, for the developer overlay.
    last_asr_ms: Option<u128>,
    last_decision_ms: Option<u128>,
    last_tts_ms: Option<u128>,
}

impl Worker {
    fn new(cfg: WorkerConfig) -> Self {
        let mut dialogue = Vec::new();
        let session = match build_session(&cfg) {
            Ok(s) => s,
            Err(e) => {
                dialogue.push(DialogueLine {
                    speaker: "system".into(),
                    text: format!("model unavailable: {e}"),
                    kind: LineKind::System,
                });
                Session::new(cfg.seed)
            }
        };
        let engine_name = session.engine_name();
        let model_id = session.model_id();
        dialogue.push(DialogueLine {
            speaker: "system".into(),
            text: format!("engine: {engine_name} ({model_id})"),
            kind: LineKind::System,
        });

        let mut worker = Self {
            session,
            tts: None,
            asr: None,
            vad: None,
            capture: None,
            playback: None,
            voice: cfg.voice.clone(),
            dialogue,
            status: String::new(),
            busy: false,
            voice_error: None,
            cancel: Arc::new(AtomicBool::new(false)),
            last_asr_ms: None,
            last_decision_ms: None,
            last_tts_ms: None,
        };
        if !cfg.no_voice {
            if let Err(e) = worker.load_voice() {
                worker.voice_error = Some(e.clone());
                worker.dialogue.push(DialogueLine {
                    speaker: "system".into(),
                    text: format!("voice unavailable: {e}"),
                    kind: LineKind::System,
                });
            }
        }
        worker
    }

    fn load_voice(&mut self) -> Result<(), String> {
        let manifest = workspace_manifest();
        let cache = workspace_cache();
        let assets = SpeechAssets::load(&cache, &manifest)
            .map_err(|e| format!("speech assets: {e} (run `voice-demo install`)"))?;
        let threads = default_threads();
        let mut asr = SherpaAsr::from_assets(&assets, rpg_speech::DEFAULT_ASR_BUNDLE, threads)
            .map_err(|e| e.to_string())?;
        asr.warm().map_err(|e| e.to_string())?;
        let mut tts = SherpaTts::from_assets(&assets, rpg_speech::DEFAULT_TTS_BUNDLE, threads)
            .map_err(|e| e.to_string())?;
        tts.warm(&self.voice).map_err(|e| e.to_string())?;
        let vad = VadSegmenter::from_assets(&assets, rpg_speech::DEFAULT_VAD_BUNDLE, 16_000)
            .map_err(|e| e.to_string())?;
        let capture = AudioCapture::open(&CaptureConfig::default()).map_err(|e| e.to_string())?;
        let playback = AudioPlayback::open(&PlaybackConfig::new(tts.sample_rate()))
            .map_err(|e| e.to_string())?;
        self.asr = Some(asr);
        self.tts = Some(tts);
        self.vad = Some(vad);
        self.capture = Some(capture);
        self.playback = Some(playback);
        Ok(())
    }

    fn run(&mut self, rx: Receiver<Command>, tx: Sender<Snapshot>) {
        self.status = "ready — press V to speak, WASD to walk".into();
        self.publish(&tx);
        loop {
            let mut quit = false;
            loop {
                match rx.try_recv() {
                    Ok(Command::Quit) => {
                        quit = true;
                        break;
                    }
                    Ok(Command::MoveTo(pos)) => self.move_to(pos),
                    Ok(Command::VoiceTurn) => self.voice_turn(&tx),
                    Ok(Command::AcceptOffer(id)) => self.accept_offer(id),
                    Ok(Command::RejectOffer(id)) => self.reject_offer(id),
                    Ok(Command::Investigate(target)) => self.investigate(target),
                    Ok(Command::Save(path)) => self.save(&path),
                    Ok(Command::Load(path)) => self.load(&path),
                    Err(_) => break,
                }
            }
            if quit {
                return;
            }
            // Advance the authoritative clock so schedules and background
            // actors progress even while the player stands still.
            let _ = self.session.tick();
            self.publish(&tx);
            thread::sleep(Duration::from_millis(120));
        }
    }

    fn move_to(&mut self, pos: Vec2) {
        let _ = self.session.player_do(Action::MoveToPos { pos });
    }

    fn accept_offer(&mut self, id: OfferId) {
        let _ = self.session.player_do(Action::AcceptOffer { offer: id });
    }

    fn reject_offer(&mut self, id: OfferId) {
        let _ = self.session.player_do(Action::RejectOffer { offer: id });
    }

    fn investigate(&mut self, target: InvestigateTarget) {
        let _ = self.session.player_do(Action::Investigate { target });
    }

    /// Persist canonical state. The save file is versioned and contains the
    /// simulation seed, world snapshot and event checksum; loading it must
    /// reproduce the same canonical outcomes.
    fn save(&mut self, path: &str) {
        let save = rpg_persistence::SaveFile::from_sim(&self.session.sim);
        let message = match rpg_persistence::save_json(&save, path) {
            Ok(()) => format!("saved to {path}"),
            Err(e) => format!("save failed: {e}"),
        };
        self.dialogue.push(DialogueLine {
            speaker: "system".into(),
            text: message,
            kind: LineKind::System,
        });
    }

    /// Restore canonical state from a JSON save file. The engine (resident
    /// model) is kept, so a reload does not pay the model load cost again.
    fn load(&mut self, path: &str) {
        match rpg_persistence::load_json(path) {
            Ok(save) => {
                let config = rpg_sim::SimConfig {
                    enable_background: true,
                    enable_director: true,
                    ..rpg_sim::SimConfig::default()
                };
                self.session.sim = rpg_persistence::restore(save, config);
                self.dialogue.clear();
                self.dialogue.push(DialogueLine {
                    speaker: "system".into(),
                    text: format!("loaded from {path}"),
                    kind: LineKind::System,
                });
            }
            Err(e) => {
                self.dialogue.push(DialogueLine {
                    speaker: "system".into(),
                    text: format!("load failed: {e}"),
                    kind: LineKind::System,
                });
            }
        }
    }

    fn talk(&mut self, text: &str) {
        self.dialogue.push(DialogueLine {
            speaker: "you".into(),
            text: text.to_string(),
            kind: LineKind::Player,
        });
        let Some(npc) = self.nearby() else {
            self.dialogue.push(DialogueLine {
                speaker: "system".into(),
                text: "nobody is close enough to hear you".into(),
                kind: LineKind::System,
            });
            return;
        };
        self.status = "thinking".into();
        let t0 = Instant::now();
        let reply = self.session.say(npc.as_str(), text);
        self.last_decision_ms = Some(t0.elapsed().as_millis());
        match reply {
            Some(line) => {
                self.dialogue.push(DialogueLine {
                    speaker: npc.as_str().to_string(),
                    text: line.clone(),
                    kind: LineKind::Npc,
                });
                self.speak(npc.as_str(), &line);
            }
            None => {
                self.dialogue.push(DialogueLine {
                    speaker: npc.as_str().to_string(),
                    text: "(no spoken reply; the accepted action was not `speak`)".into(),
                    kind: LineKind::System,
                });
            }
        }
        self.status = "ready".into();
    }

    fn voice_turn(&mut self, tx: &Sender<Snapshot>) {
        if self.asr.is_none() || self.capture.is_none() {
            self.dialogue.push(DialogueLine {
                speaker: "system".into(),
                text: "voice is unavailable on this machine".into(),
                kind: LineKind::System,
            });
            return;
        }
        self.busy = true;
        self.run_voice_turn(tx);
        self.busy = false;
        self.status = "ready".into();
        self.publish(tx);
    }

    /// The body of a voice turn. The caller owns `busy` and the final publish,
    /// so an early return can never leave the UI stuck in a busy state.
    fn run_voice_turn(&mut self, tx: &Sender<Snapshot>) {
        self.status = "listening — speak now".into();
        self.publish(tx);
        let seg = {
            let capture = self.capture.as_ref().expect("capture");
            let vad = self.vad.as_mut().expect("vad");
            capture_utterance(capture, vad, &self.cancel)
        };
        let Some(seg) = seg else {
            self.dialogue.push(DialogueLine {
                speaker: "system".into(),
                text: "did not catch that — press V and speak clearly".into(),
                kind: LineKind::System,
            });
            return;
        };

        self.status = "transcribing".into();
        self.publish(tx);
        let t_asr = Instant::now();
        let transcript = {
            let asr = self.asr.as_mut().expect("asr");
            match asr.transcribe(&seg.samples, seg.sample_rate) {
                Ok(t) => t.text.trim().to_string(),
                Err(e) => {
                    self.dialogue.push(DialogueLine {
                        speaker: "system".into(),
                        text: format!("transcription failed: {e}"),
                        kind: LineKind::System,
                    });
                    return;
                }
            }
        };
        self.last_asr_ms = Some(t_asr.elapsed().as_millis());
        if transcript.is_empty() {
            return;
        }
        // The transcript is what the player "said"; feed it through the same
        // grounded path as typed talk so the simulation stays authoritative.
        self.talk(&transcript);
    }

    /// Synthesize `text` and play it through the output device.
    ///
    /// Named NPCs receive a deterministic, distinct voice so the cast is
    /// audibly separable; `self.voice` (from `--voice`) is only the fallback
    /// for unnamed speakers. Voice choice never touches canonical state.
    fn speak(&mut self, speaker: &str, text: &str) {
        let (Some(tts), Some(playback)) = (self.tts.as_mut(), self.playback.as_ref()) else {
            return;
        };
        let voice_id = if speaker == "you" || speaker == "system" {
            self.voice.clone()
        } else {
            tts.voice_for_actor(speaker).id
        };
        let t0 = Instant::now();
        let audio = match tts.synthesize(&SpeechRequest {
            text: text.to_string(),
            voice_id,
            sample_rate: playback.sample_rate,
        }) {
            Ok(a) => a,
            Err(e) => {
                self.dialogue.push(DialogueLine {
                    speaker: "system".into(),
                    text: format!("tts failed: {e}"),
                    kind: LineKind::System,
                });
                return;
            }
        };
        self.last_tts_ms = Some(t0.elapsed().as_millis());
        // The neural model renders at its native rate; resample to the device
        // rate so playback speed is correct.
        let to_play = if audio.sample_rate == playback.sample_rate {
            audio.samples
        } else {
            LinearResampler::new(audio.sample_rate, playback.sample_rate).process(&audio.samples)
        };
        let seconds = to_play.len() as f32 / playback.sample_rate.max(1) as f32;
        if playback.queue(&to_play).is_ok() {
            let _ = playback.wait_until_drained(Duration::from_secs_f32(seconds + 2.0));
        }
    }

    /// Nearest living named NPC in the same room, within [`TALK_RADIUS`].
    fn nearby(&self) -> Option<ActorId> {
        let player = self.session.player();
        self.session
            .sim
            .world
            .actors
            .values()
            .filter(|a| a.id.0 != PLAYER)
            .filter(|a| a.kind == ActorKind::Npc)
            .filter(|a| a.status == ActorStatus::Alive)
            .filter(|a| a.location == player.location)
            .filter(|a| a.pos.distance(player.pos) <= TALK_RADIUS)
            .min_by(|a, b| {
                a.pos
                    .distance(player.pos)
                    .partial_cmp(&b.pos.distance(player.pos))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|a| a.id.clone())
    }

    fn publish(&self, tx: &Sender<Snapshot>) {
        let _ = tx.send(self.snapshot());
    }

    fn snapshot(&self) -> Snapshot {
        let world = &self.session.sim.world;
        let player = world.actor(&ActorId::new(PLAYER));
        let player_pos = player.map(|a| a.pos).unwrap_or(Vec2::ZERO);
        let player_location = player
            .map(|a| a.location.as_str().to_string())
            .unwrap_or_default();
        let nearby = self.nearby().map(|a| a.as_str().to_string());

        let locations = world
            .locations
            .values()
            .map(|l| LocationView {
                id: l.id.as_str().to_string(),
                name: l.name.clone(),
                x: l.bounds.x,
                y: l.bounds.y,
                w: l.bounds.w,
                h: l.bounds.h,
                private: l.private,
            })
            .collect();

        let actors = world
            .actors
            .values()
            .map(|a| ActorView {
                id: a.id.as_str().to_string(),
                name: a.name.clone(),
                pos: a.pos,
                kind: match a.kind {
                    ActorKind::Player => KindView::Player,
                    ActorKind::Npc => KindView::Npc,
                    ActorKind::Background => KindView::Background,
                },
                status: match a.status {
                    ActorStatus::Alive => StatusView::Alive,
                    ActorStatus::Dead => StatusView::Dead,
                    ActorStatus::Departed => StatusView::Departed,
                },
                location: a.location.as_str().to_string(),
                nearby: nearby.as_deref() == Some(a.id.as_str()),
            })
            .collect();

        let offers = world
            .offers
            .values()
            .map(|o| OfferView {
                id: o.id.raw(),
                from: actor_name(world, &o.from),
                to: actor_name(world, &o.to),
                to_player: o.to.0 == PLAYER,
                give: format_stacks(world, &o.give),
                want: format_stacks(world, &o.want),
                price: o.price,
                note: o.note.clone(),
                state: format!("{:?}", o.state).to_ascii_lowercase(),
            })
            .collect();

        let commitments = world
            .commitments
            .values()
            .map(|c| CommitmentView {
                id: c.id.raw(),
                kind: format!("{:?}", c.kind),
                debtor: actor_name(world, &c.debtor),
                creditor: actor_name(world, &c.creditor),
                terms: c.terms.clone(),
                state: format!("{:?}", c.state),
            })
            .collect();

        let player_inv = world.actor(&ActorId::new(PLAYER)).map(|a| &a.inventory);
        let inventory = player_inv
            .map(|inv| {
                inv.items
                    .iter()
                    .map(|s| ItemView {
                        name: world
                            .items
                            .get(&s.item)
                            .map(|i| i.name.clone())
                            .unwrap_or_else(|| s.item.as_str().to_string()),
                        qty: s.qty,
                    })
                    .collect()
            })
            .unwrap_or_default();
        let money = player_inv.map(|inv| inv.money).unwrap_or(0);

        // --- Developer diagnostics (F3 overlay only) ---------------------
        let nearby_beliefs = nearby
            .as_deref()
            .and_then(|id| world.actor(&ActorId::new(id)))
            .map(|a| {
                a.beliefs
                    .keys()
                    .filter_map(|f| world.facts.get(f))
                    .map(|f| format!("[{:?}] {}", f.sensitivity, f.statement).to_ascii_lowercase())
                    .collect()
            })
            .unwrap_or_default();
        let nearby_relationship = nearby
            .as_deref()
            .and_then(|id| world.actor(&ActorId::new(id)))
            .map(|a| {
                let r = a.relationship(&ActorId::new(PLAYER));
                format!(
                    "trust {:.2}  affection {:.2}  respect {:.2}  fear {:.2}  anger {:.2}  suspicion {:.2}  familiar {:.2}",
                    r.trust, r.affection, r.respect, r.fear, r.anger, r.suspicion, r.familiarity
                )
            });
        let perception = vec![
            format!("conversation radius {TALK_RADIUS}"),
            format!(
                "hearing: whisper {} / normal {} / loud {} / shout {}",
                Loudness::Whisper.radius(),
                Loudness::Normal.radius(),
                Loudness::Loud.radius(),
                Loudness::Shout.radius()
            ),
            "line of sight is wall-occluded (rpg_perception)".to_string(),
        ];
        let recent_events = self
            .session
            .sim
            .events
            .recent(8)
            .iter()
            .map(|e| format!("t{}  {}", e.tick, e.describe()))
            .collect();
        let last_turn_timing = match (self.last_asr_ms, self.last_decision_ms, self.last_tts_ms) {
            (None, None, None) => None,
            (a, d, t) => Some(format!(
                "asr {}  decision {}  tts {}",
                opt_ms(a),
                opt_ms(d),
                opt_ms(t)
            )),
        };

        Snapshot {
            tick: world.tick,
            day: world.day,
            minute_of_day: world.minute_of_day,
            locations,
            actors,
            player_pos,
            player_location,
            nearby,
            dialogue: self.dialogue.clone(),
            engine_name: self.session.engine_name(),
            model_id: self.session.model_id(),
            model_decisions: self.session.model_decisions(),
            canonical_decisions: self.session.canonical_decisions(),
            fallback_decisions: self.session.fallback_decisions(),
            busy: self.busy,
            status: self.status.clone(),
            voice_error: self.voice_error.clone(),
            recent_events,
            nearby_beliefs,
            nearby_relationship,
            perception,
            last_turn_timing,
            offers,
            commitments,
            inventory,
            money,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Capture one utterance from the microphone, using VAD to detect the end.
///
/// Returns `None` if no complete speech segment arrived within the window.
fn capture_utterance(
    capture: &AudioCapture,
    vad: &mut VadSegmenter,
    cancel: &AtomicBool,
) -> Option<SpeechSegmentAudio> {
    let mut resampler = LinearResampler::new(capture.native_sample_rate, 16_000);
    let start = Instant::now();
    while start.elapsed().as_secs_f32() < MAX_CAPTURE_SECS {
        if cancel.load(Ordering::Relaxed) {
            return None;
        }
        match capture.try_recv() {
            Some(chunk) => {
                let mono = deinterleave(&chunk, capture.native_channels);
                let mono16 = resampler.process(&mono);
                if let Some(seg) = vad.accept(&mono16).into_iter().next() {
                    return Some(seg);
                }
            }
            None => thread::sleep(Duration::from_millis(10)),
        }
    }
    // Flush whatever is buffered so a trailing phrase is not lost.
    vad.flush().into_iter().next()
}

/// Average interleaved channels down to mono.
fn deinterleave(interleaved: &[f32], channels: u16) -> Vec<f32> {
    let ch = channels.max(1) as usize;
    if ch == 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks(ch)
        .map(|frame| frame.iter().sum::<f32>() / ch as f32)
        .collect()
}

/// "123 ms" or an em dash when a stage did not run.
fn opt_ms(v: Option<u128>) -> String {
    v.map(|m| format!("{m} ms")).unwrap_or_else(|| "—".into())
}

/// Human-readable name for an actor id (falls back to the raw id).
fn actor_name(world: &rpg_domain::world::World, id: &ActorId) -> String {
    world
        .actor(id)
        .map(|a| a.name.clone())
        .unwrap_or_else(|| id.as_str().to_string())
}

/// "medicine x2, ledger" for display in offers.
fn format_stacks(
    world: &rpg_domain::world::World,
    stacks: &[rpg_domain::world::ItemStack],
) -> String {
    if stacks.is_empty() {
        return "—".into();
    }
    stacks
        .iter()
        .map(|s| {
            let name = world
                .items
                .get(&s.item)
                .map(|i| i.name.clone())
                .unwrap_or_else(|| s.item.as_str().to_string());
            if s.qty > 1 {
                format!("{name} x{}", s.qty)
            } else {
                name
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn default_threads() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
        .min(16)
}

fn build_session(cfg: &WorkerConfig) -> Result<Session, String> {
    let Some(path) = cfg.model_path.as_deref() else {
        return Ok(Session::new(cfg.seed));
    };
    if !std::path::Path::new(path).exists() {
        return Err(format!("model file not found: {path}"));
    }
    let path = path.replace('\\', "/");
    let json = format!(
        r#"{{"id":"bevy","model_path":"{path}","n_ctx":4096,"n_threads":{threads},"max_tokens":256,"temperature":0.4,"no_think":true,"deadline_ms":60000}}"#,
        threads = default_threads()
    );
    let profile = ModelProfile::from_json(&json).map_err(|e| e.to_string())?;
    let mut engine = LlamaEngine::load(profile).map_err(|e| e.to_string())?;
    engine.warm().map_err(|e| e.to_string())?;
    Ok(Session::with_engine(cfg.seed, Box::new(engine)))
}

fn workspace_cache() -> std::path::PathBuf {
    workspace_root().join(".cache/speech")
}

fn workspace_manifest() -> std::path::PathBuf {
    workspace_root().join("assets/speech/manifest.json")
}

fn workspace_root() -> std::path::PathBuf {
    // CARGO_MANIFEST_DIR is `apps/bevy-game`; the workspace is two levels up.
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."))
}

/// Default save slot used by the graphical client's quick-save keys.
///
/// Kept in the workspace root so the text client and graphical client can
/// share a save produced by the same authoritative simulation.
pub fn default_save_path() -> String {
    workspace_root()
        .join("irl_save.json")
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deinterleave_averages_stereo_to_mono() {
        assert_eq!(deinterleave(&[1.0, 3.0, 2.0, 4.0], 2), vec![2.0, 3.0]);
        assert_eq!(deinterleave(&[0.5, 1.0], 1), vec![0.5, 1.0]);
    }

    #[test]
    fn clock_formats_hh_mm() {
        let snap = Snapshot {
            tick: 0,
            day: 1,
            minute_of_day: 9 * 60 + 5,
            locations: vec![],
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
        };
        assert_eq!(snap.clock(), "09:05");
    }
}
