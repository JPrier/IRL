//! Browser entry point, the platform [`Link`], and the functions the page
//! (`web/index.html`) calls.
//!
//! The page's controls never touch game state directly: every export below
//! either enqueues an ordinary `rpg_runtime::Command` (the same commands the
//! keyboard sends) or changes the OpenRouter connection in
//! `rpg_web_runtime::browser`. The API key passes through `irl_connect` once
//! and then lives only in that module's memory.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use bevy::prelude::*;
use irl_client::{ClientPlatform, Link};
use rpg_domain::action::InvestigateTarget;
use rpg_domain::ids::{LocationId, OfferId};
use rpg_openrouter::web::now_ms;
use rpg_runtime::{Command, CoreConfig, Inbox, InferEvent, LineKind, Snapshot};
use rpg_web_runtime::browser::{self, BrowserDispatcher, ConnectSettings, SharedState};
use rpg_web_runtime::{InboxQueue, WebRuntime};
use serde_json::json;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

/// What the page-facing exports need to reach.
struct Handles {
    inbox: InboxQueue,
    shared: SharedState,
    latest: Rc<RefCell<Option<Snapshot>>>,
    frames: Rc<RefCell<VecDeque<f64>>>,
}

thread_local! {
    static HANDLES: RefCell<Option<Handles>> = const { RefCell::new(None) };
}

fn with<R>(f: impl FnOnce(&Handles) -> R) -> Option<R> {
    HANDLES.with(|h| h.borrow().as_ref().map(f))
}

fn send(cmd: Command) {
    with(|h| h.inbox.borrow_mut().push_back(Inbox::Ui(cmd)));
}

/// Owns the browser runtime. Pumped by every rendered frame *and* by a
/// timer, so the simulation keeps its fixed rate when frames are slow or the
/// tab is in the background (browsers pause animation frames there but keep
/// timers running, throttled).
struct Driver {
    rt: WebRuntime<BrowserDispatcher>,
    /// A snapshot newer than the one the renderer last took.
    fresh: bool,
}

impl Driver {
    fn pump(&mut self, latest: &RefCell<Option<Snapshot>>) {
        if let Some(snap) = self.rt.pump(now_ms()) {
            *latest.borrow_mut() = Some(snap);
            self.fresh = true;
        }
    }
}

/// The browser platform link.
struct WebLink {
    driver: Rc<RefCell<Driver>>,
    inbox: InboxQueue,
    latest: Rc<RefCell<Option<Snapshot>>>,
    frames: Rc<RefCell<VecDeque<f64>>>,
}

impl Link for WebLink {
    fn send(&mut self, cmd: Command) {
        self.inbox.borrow_mut().push_back(Inbox::Ui(cmd));
    }

    /// Called once per rendered frame: run due ticks, handle results that
    /// arrived, dispatch effects. Never waits.
    fn poll(&mut self) -> Option<Snapshot> {
        let now = now_ms();
        {
            let mut f = self.frames.borrow_mut();
            f.push_back(now);
            while f.front().is_some_and(|t| now - *t > 2000.0) {
                f.pop_front();
            }
        }
        let mut d = self.driver.borrow_mut();
        d.pump(&self.latest);
        if !std::mem::take(&mut d.fresh) {
            return None;
        }
        self.latest.borrow().clone()
    }

    fn quick_save(&mut self) {
        self.send(Command::ExportSave {
            slot: "quick".into(),
        });
    }

    fn quick_load(&mut self) {
        browser::load_slot(&self.inbox, "quick");
    }
}

/// Pump the simulation from a timer as well as from frames.
fn start_tick_timer(driver: Rc<RefCell<Driver>>, latest: Rc<RefCell<Option<Snapshot>>>) {
    let tick = Closure::<dyn FnMut()>::new(move || {
        // Skip if a frame is pumping right now (cannot happen on one thread,
        // but never panic on re-entrancy).
        if let Ok(mut d) = driver.try_borrow_mut() {
            d.pump(&latest);
        }
    });
    if let Some(w) = web_sys::window() {
        let period = (rpg_web_runtime::TICK_PERIOD_MS / 2.0) as i32;
        let _ = w.set_interval_with_callback_and_timeout_and_arguments_0(
            tick.as_ref().unchecked_ref(),
            period,
        );
    }
    // Lives for the page's lifetime.
    tick.forget();
}

fn query_param(name: &str) -> Option<String> {
    let search = web_sys::window()?.location().search().ok()?;
    web_sys::UrlSearchParams::new_with_str(&search)
        .ok()?
        .get(name)
}

fn report_fatal(msg: &str) {
    web_sys::console::error_1(&JsValue::from_str(msg));
    if let Some(w) = web_sys::window() {
        if let Ok(host) = js_sys::Reflect::get(&w, &JsValue::from_str("irlHost")) {
            if let Ok(f) = js_sys::Reflect::get(&host, &JsValue::from_str("fatal")) {
                if let Some(f) = f.dyn_ref::<js_sys::Function>() {
                    let _ = f.call1(&host, &JsValue::from_str(msg));
                }
            }
        }
    }
}

pub fn run() {
    std::panic::set_hook(Box::new(|info| {
        // Panic messages never contain the credential: it is never formatted.
        report_fatal(&format!("IRL stopped: {info}"));
    }));

    let seed = query_param("seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2026u64);
    let mut session = rpg_eval::Session::new(seed);
    session.teleport_player_to("grocer");

    let shared = browser::shared_state();
    let mut rt = WebRuntime::new(
        session,
        CoreConfig::default(),
        BrowserDispatcher::new(shared.clone()),
        now_ms(),
    );
    rt.core_mut().mark_unavailable(
        true,
        true,
        true,
        "voice is off - enable it in the panel (Enter always types)",
    );
    let inbox = rt.inbox();
    let inbox_handle = inbox.clone();
    inbox
        .borrow_mut()
        .push_back(Inbox::Infer(InferEvent::Ready {
            engine: "heuristic (no credentials)".into(),
            model_id: "deterministic policy".into(),
            is_real_model: false,
        }));
    let latest = Rc::new(RefCell::new(None));
    let frames = Rc::new(RefCell::new(VecDeque::new()));
    HANDLES.with(|h| {
        *h.borrow_mut() = Some(Handles {
            inbox,
            shared,
            latest: latest.clone(),
            frames: frames.clone(),
        })
    });

    let window = Window {
        title: "IRL - Hidden Mafia".into(),
        canvas: Some("#irl-canvas".into()),
        fit_canvas_to_parent: true,
        prevent_default_event_handling: true,
        ..default()
    };
    let driver = Rc::new(RefCell::new(Driver { rt, fresh: false }));
    start_tick_timer(driver.clone(), latest.clone());
    irl_client::app(
        Box::new(WebLink {
            driver,
            inbox: inbox_handle,
            latest,
            frames,
        }),
        ClientPlatform::Browser,
        window,
    )
    .run();
}

// ---------------------------------------------------------------------------
// Page API
// ---------------------------------------------------------------------------

fn handles_err() -> JsValue {
    JsValue::from_str("IRL is not running")
}

/// Check the key and model, then use OpenRouter for NPC decisions.
/// `settings` is JSON (`model`, `max_output_tokens`, `max_requests`,
/// `max_cost_usd`, `structured`, `stream`, `timeout_ms`). Resolves to a JSON
/// summary; rejects with a readable error. The key is never echoed.
#[wasm_bindgen]
pub async fn irl_connect(key: String, settings: String) -> Result<JsValue, JsValue> {
    let settings: ConnectSettings = if settings.trim().is_empty() {
        ConnectSettings::default()
    } else {
        serde_json::from_str(&settings).map_err(|e| JsValue::from_str(&format!("settings: {e}")))?
    };
    let (shared, inbox) = with(|h| (h.shared.clone(), h.inbox.clone())).ok_or_else(handles_err)?;
    match browser::connect(shared, inbox.clone(), key, settings).await {
        Ok(v) => Ok(JsValue::from_str(&v.to_string())),
        Err(e) => {
            inbox
                .borrow_mut()
                .push_back(Inbox::Ui(Command::Notice(format!("OpenRouter: {e}"))));
            Err(JsValue::from_str(&e))
        }
    }
}

/// Clear the key from memory, cancel outstanding requests, and return to the
/// deterministic policy.
#[wasm_bindgen]
pub fn irl_disconnect() {
    with(|h| browser::disconnect(&h.shared, &h.inbox));
}

/// Provider status and usage as JSON (never the key).
#[wasm_bindgen]
pub fn irl_status() -> String {
    with(|h| browser::status(&h.shared).to_string()).unwrap_or_else(|| "{}".into())
}

/// Structured-output-capable models from the public catalog, as JSON.
#[wasm_bindgen]
pub async fn irl_models() -> Result<JsValue, JsValue> {
    let models = browser::list_models()
        .await
        .map_err(|e| JsValue::from_str(&e))?;
    let list: Vec<_> = models
        .iter()
        .map(|m| {
            json!({
                "id": m.id,
                "name": m.name,
                "context_length": m.context_length,
                "prompt_price": m.prompt_price,
                "completion_price": m.completion_price,
                "seed": m.supports("seed"),
            })
        })
        .collect();
    Ok(JsValue::from_str(
        &serde_json::Value::from(list).to_string(),
    ))
}

/// Typed dialogue from the page's input box: the same command as typing in
/// the canvas, so the identical gameplay path.
#[wasm_bindgen]
pub fn irl_say(text: String) {
    let text = text.trim().to_string();
    if !text.is_empty() {
        send(Command::Say(text));
    }
}

/// Page buttons: `save`, `load`, `export`, `cancel`, `accept`, `reject`,
/// `investigate`. Returns whether the action was recognised.
#[wasm_bindgen]
pub fn irl_action(name: &str) -> bool {
    match name {
        "save" => send(Command::ExportSave {
            slot: "quick".into(),
        }),
        "load" => {
            with(|h| browser::load_slot(&h.inbox, "quick"));
        }
        "export" => send(Command::ExportSave {
            slot: "export".into(),
        }),
        "cancel" => send(Command::CancelVoice),
        "accept" | "reject" => {
            let offer = with(|h| {
                h.latest.borrow().as_ref().and_then(|s| {
                    s.offers
                        .iter()
                        .find(|o| o.to_player && o.state == "pending")
                        .map(|o| o.id)
                })
            })
            .flatten();
            match offer {
                Some(id) if name == "accept" => send(Command::AcceptOffer(OfferId(id))),
                Some(id) => send(Command::RejectOffer(OfferId(id))),
                None => send(Command::Notice("no pending offer".into())),
            }
        }
        "investigate" => {
            let loc = with(|h| {
                h.latest
                    .borrow()
                    .as_ref()
                    .map(|s| s.player_location.clone())
            })
            .flatten();
            if let Some(loc) = loc {
                send(Command::Investigate(InvestigateTarget::Location(
                    LocationId::from(loc.as_str()),
                )));
            }
        }
        _ => return false,
    }
    true
}

/// Load a save the player imported from a file. Validated (version, format)
/// exactly like a native save.
#[wasm_bindgen]
pub fn irl_import_save(json: String) {
    send(Command::ImportSave {
        label: "imported file".into(),
        json,
    });
}

/// Turn browser speech output (`speechSynthesis`) and push-to-talk input
/// (Web Speech recognition) on or off.
#[wasm_bindgen]
pub fn irl_set_voice(tts: bool, asr: bool) {
    with(|h| browser::set_voice(&h.shared, &h.inbox, tts, asr));
}

/// Speech callbacks from `web/host.js`.
#[wasm_bindgen]
pub fn irl_host_event(kind: String, conv: f64, line: f64, text: String) {
    with(|h| {
        browser::host_event(
            &h.shared,
            &h.inbox,
            &kind,
            conv.max(0.0) as u64,
            line.max(0.0) as u64,
            text,
        )
    });
}

/// A compact JSON view of the latest snapshot for the page, smoke tests and
/// performance probes. Contains canonical *view* data only — no credential,
/// no NPC-private beliefs.
#[wasm_bindgen]
pub fn irl_debug_state() -> String {
    with(|h| {
        let fps = {
            let f = h.frames.borrow();
            match (f.front(), f.back()) {
                (Some(a), Some(b)) if f.len() > 1 && b > a => {
                    (f.len() - 1) as f64 * 1000.0 / (b - a)
                }
                _ => 0.0,
            }
        };
        let latest = h.latest.borrow();
        let Some(s) = latest.as_ref() else {
            return json!({ "ready": false, "fps": fps }).to_string();
        };
        let d = &s.diagnostics;
        json!({
            "ready": true,
            "fps": fps,
            "tick": s.tick,
            "day": s.day,
            "clock": s.clock(),
            "world_epoch": s.world_epoch,
            "player": {"x": s.player_pos.x, "y": s.player_pos.y, "location": s.player_location},
            "nearby": s.nearby,
            "locations": s.locations.len(),
            "actors": s.actors.iter().map(|a| json!({
                "id": a.id, "name": a.name, "x": a.pos.x, "y": a.pos.y,
                "location": a.location, "kind": format!("{:?}", a.kind),
            })).collect::<Vec<_>>(),
            "dialogue": s.dialogue.iter().rev().take(30).rev().map(|l| json!({
                "speaker": l.speaker, "text": l.text,
                "kind": match l.kind { LineKind::Player => "player", LineKind::Npc => "npc", LineKind::System => "system" },
            })).collect::<Vec<_>>(),
            "offers": s.offers.iter().map(|o| json!({
                "id": o.id, "from": o.from, "to_player": o.to_player, "want": o.want,
                "give": o.give, "note": o.note, "state": o.state,
            })).collect::<Vec<_>>(),
            "commitments": s.commitments.iter().map(|c| json!({
                "kind": c.kind, "debtor": c.debtor, "terms": c.terms, "state": c.state,
            })).collect::<Vec<_>>(),
            "inventory": s.inventory.iter().map(|i| json!({"name": i.name, "qty": i.qty})).collect::<Vec<_>>(),
            "money": s.money,
            "engine": s.engine_name,
            "model": s.model_id,
            "decisions": {"model": s.model_decisions, "canonical": s.canonical_decisions, "fallback": s.fallback_decisions},
            "stage": s.voice.stage.label(),
            "status": s.status,
            "recent_events": s.recent_events,
            "diagnostics": {
                "platform": d.platform,
                "tps": d.ticks_per_second,
                "target_tps": d.target_ticks_per_second,
                "max_tick_lag_ms": d.max_tick_lag_ms,
                "skipped_ticks": d.skipped_ticks,
                "pending_inference": d.pending_inference,
                "llm_infer_ms": d.llm_infer_ms,
                "llm_first_token_ms": d.llm_first_token_ms,
                "inference": {
                    "requests": d.inference.requests,
                    "completed": d.inference.completed,
                    "errors": d.inference.errors,
                    "prompt_tokens": d.inference.prompt_tokens,
                    "completion_tokens": d.inference.completion_tokens,
                    "reported_cost_usd": d.inference.reported_cost_usd,
                    "estimated_cost_usd": d.inference.estimated_cost_usd,
                    "last_latency_ms": d.inference.last_latency_ms,
                    "last_remote": d.inference.last_remote,
                    "last_error": d.inference.last_error,
                },
                "last_decision": d.last_decision,
                "rejected_actions": d.rejected_actions,
                "stale": d.stale_responses,
                "cancelled": d.cancelled,
                "timeouts": d.timeouts,
                "invalidated": d.invalidated,
                "regenerations": d.regenerations,
            },
        })
        .to_string()
    })
    .unwrap_or_else(|| "{\"ready\":false}".into())
}
