//! The browser dispatcher, OpenRouter connection state and the page bridge
//! (`wasm32` only).
//!
//! * Inference: each decision request becomes a `fetch` future
//!   ([`rpg_openrouter::web`]) with its own `AbortController`. Cancel,
//!   disconnect, timeout and load all abort it; its late result (if any) is
//!   rejected by `SimCore` as stale.
//! * Credentials: the key lives only in [`Shared`] in this tab's memory. It
//!   is never logged, never part of a snapshot or save, never put in a URL
//!   and never written to `localStorage`/IndexedDB. Disconnect drops it
//!   (wiping the buffer) and cancels outstanding work.
//! * Speech and storage: small page-side functions on `window.irlHost`
//!   (`web/host.js`) wrap the Web Speech API, `speechSynthesis` and
//!   IndexedDB; they report back through [`host_event`]. They are platform
//!   I/O only — no gameplay logic lives in JavaScript.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use js_sys::Promise;
use rpg_domain::ports::InferenceError;
use rpg_openrouter::web::{self as or_web, now_ms, Abort};
use rpg_openrouter::{
    adapt_to_model, to_inference_response, ApiKey, Budget, KeyInfo, ModelInfo, OpenRouterConfig,
    StructuredMode, UsageMeter,
};
use rpg_runtime::{
    AsrEvent, AsrRequest, Command, Effect, Inbox, InferDone, InferEvent, InferRequest, MicEvent,
    MicInfo, MicRequest, TtsEvent, TtsRequest,
};
use serde::Deserialize;
use serde_json::json;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::{spawn_local, JsFuture};

use crate::{Dispatcher, InboxQueue};

// ---------------------------------------------------------------------------
// Page bridge (web/host.js)
// ---------------------------------------------------------------------------

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = irlHost, js_name = speechSupport)]
    fn host_speech_support() -> JsValue;
    #[wasm_bindgen(js_namespace = irlHost, js_name = recognitionStart, catch)]
    fn host_recognition_start(conv: f64) -> Result<(), JsValue>;
    #[wasm_bindgen(js_namespace = irlHost, js_name = recognitionStop, catch)]
    fn host_recognition_stop(conv: f64) -> Result<(), JsValue>;
    #[wasm_bindgen(js_namespace = irlHost, js_name = recognitionAbort, catch)]
    fn host_recognition_abort() -> Result<(), JsValue>;
    #[wasm_bindgen(js_namespace = irlHost, js_name = speak, catch)]
    fn host_speak(conv: f64, line: f64, speaker: &str, text: &str) -> Result<(), JsValue>;
    #[wasm_bindgen(js_namespace = irlHost, js_name = stopSpeaking, catch)]
    fn host_stop_speaking() -> Result<(), JsValue>;
    #[wasm_bindgen(js_namespace = irlHost, js_name = storagePut, catch)]
    fn host_storage_put(slot: &str, json: &str) -> Result<Promise, JsValue>;
    #[wasm_bindgen(js_namespace = irlHost, js_name = storageGet, catch)]
    fn host_storage_get(slot: &str) -> Result<Promise, JsValue>;
    #[wasm_bindgen(js_namespace = irlHost, js_name = downloadText, catch)]
    fn host_download_text(filename: &str, text: &str) -> Result<(), JsValue>;
}

fn js_text(v: &JsValue) -> String {
    if let Some(e) = v.dyn_ref::<js_sys::Error>() {
        return format!("{}", e.message());
    }
    v.as_string().unwrap_or_else(|| format!("{v:?}"))
}

fn post(inbox: &InboxQueue, msg: Inbox) {
    inbox.borrow_mut().push_back(msg);
}

fn notice(inbox: &InboxQueue, text: impl Into<String>) {
    post(inbox, Inbox::Ui(Command::Notice(text.into())));
}

// ---------------------------------------------------------------------------
// Connection state
// ---------------------------------------------------------------------------

/// An established OpenRouter connection. Holds the credential.
pub struct Connection {
    key: ApiKey,
    pub cfg: OpenRouterConfig,
    pub model: Option<ModelInfo>,
    pub key_info: Option<KeyInfo>,
}

pub enum Provider {
    /// Deterministic policy; no credential held.
    Heuristic,
    Connecting,
    OpenRouter(Box<Connection>),
}

#[derive(Clone, Copy, Debug, Default)]
pub struct VoiceSettings {
    pub tts: bool,
    pub asr: bool,
}

/// Everything the dispatcher and the page controls share. Main thread only.
pub struct Shared {
    pub provider: Provider,
    pub meter: UsageMeter,
    in_flight: BTreeMap<u64, Abort>,
    /// Bumped by every disconnect: an in-progress connect that finishes
    /// afterwards is discarded instead of re-installing a credential.
    connect_epoch: u64,
    pub last_error: Option<String>,
    pub voice: VoiceSettings,
    listening: Option<(u64, f64)>,
    speaking: BTreeMap<u64, f64>,
    last_progress_ms: f64,
}

pub type SharedState = Rc<RefCell<Shared>>;

pub fn shared_state() -> SharedState {
    Rc::new(RefCell::new(Shared {
        provider: Provider::Heuristic,
        meter: UsageMeter::default(),
        in_flight: BTreeMap::new(),
        connect_epoch: 0,
        last_error: None,
        voice: VoiceSettings::default(),
        listening: None,
        speaking: BTreeMap::new(),
        last_progress_ms: 0.0,
    }))
}

/// Settings the page sends with "Connect". All optional.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct ConnectSettings {
    pub model: Option<String>,
    pub max_output_tokens: Option<u32>,
    pub max_requests: Option<u32>,
    pub max_cost_usd: Option<f64>,
    pub structured: Option<String>,
    pub stream: Option<bool>,
    pub timeout_ms: Option<u64>,
    pub reasoning_effort: Option<String>,
}

impl ConnectSettings {
    fn config(&self) -> Result<OpenRouterConfig, String> {
        let mut cfg = OpenRouterConfig::default();
        if let Some(m) = self
            .model
            .as_deref()
            .map(str::trim)
            .filter(|m| !m.is_empty())
        {
            if m.contains(char::is_whitespace) || !m.contains('/') {
                return Err(format!(
                    "{m:?} is not an OpenRouter model slug (provider/model)"
                ));
            }
            cfg.model = m.to_string();
        }
        if let Some(n) = self.max_output_tokens {
            cfg.max_output_tokens = n.clamp(16, 4096);
        }
        cfg.budget = Budget {
            max_requests: self.max_requests.filter(|n| *n > 0),
            max_completion_tokens: None,
            max_cost_usd: self.max_cost_usd.filter(|c| c.is_finite() && *c > 0.0),
        };
        if let Some(s) = &self.structured {
            cfg.structured = StructuredMode::parse(s)
                .ok_or_else(|| format!("unknown structured-output mode {s:?}"))?;
        }
        if let Some(s) = self.stream {
            cfg.stream = s;
        }
        if let Some(t) = self.timeout_ms {
            cfg.timeout_ms = t.clamp(2_000, 120_000);
        }
        cfg.reasoning_effort = self
            .reasoning_effort
            .clone()
            .filter(|e| !e.trim().is_empty());
        Ok(cfg)
    }
}

/// Validate the key (free `GET /key`) and the model's structured-output
/// support, then switch NPC decisions to OpenRouter.
pub async fn connect(
    shared: SharedState,
    inbox: InboxQueue,
    raw_key: String,
    settings: ConnectSettings,
) -> Result<serde_json::Value, String> {
    let key = ApiKey::new(raw_key).map_err(|e| e.to_string())?;
    let mut cfg = settings.config()?;
    let epoch = {
        let mut s = shared.borrow_mut();
        abort_all(&mut s);
        // Each connect supersedes any earlier one still in flight (a manual
        // paste racing an OAuth return): only the newest may install a key.
        s.connect_epoch += 1;
        s.provider = Provider::Connecting;
        s.last_error = None;
        s.connect_epoch
    };
    let result = async {
        let abort = Abort::new().map_err(|e| e.to_string())?;
        let key_info = or_web::fetch_key_info(&cfg, &key, &abort)
            .await
            .map_err(|e| e.to_string())?;
        let models = or_web::fetch_models(&cfg, false, &abort)
            .await
            .map_err(|e| e.to_string())?;
        let model = models
            .into_iter()
            .find(|m| m.id == cfg.model)
            .ok_or_else(|| format!("model {:?} is not in the OpenRouter catalog", cfg.model))?;
        // Drop optional parameters the model does not list: with
        // require_parameters they would leave no endpoint to route to.
        adapt_to_model(&mut cfg, &model).map_err(|e| e.to_string())?;
        Ok::<_, String>((key_info, model))
    }
    .await;
    let mut s = shared.borrow_mut();
    if s.connect_epoch != epoch {
        return Err("superseded by a newer connect or a disconnect".into());
    }
    match result {
        Ok((key_info, model)) => {
            let model_id = cfg.model.clone();
            let summary = json!({
                "model": model_id,
                "model_name": model.name,
                "structured": cfg.structured.label(),
                "key_label": key_info.label,
                "limit_remaining": key_info.limit_remaining,
                "is_free_tier": key_info.is_free_tier,
            });
            s.provider = Provider::OpenRouter(Box::new(Connection {
                key,
                cfg,
                model: Some(model),
                key_info: Some(key_info),
            }));
            drop(s);
            post(
                &inbox,
                Inbox::Infer(InferEvent::Ready {
                    engine: "openrouter".into(),
                    model_id: model_id.clone(),
                    is_real_model: true,
                }),
            );
            notice(
                &inbox,
                format!("connected to OpenRouter - NPCs now use {model_id}"),
            );
            Ok(summary)
        }
        Err(e) => {
            s.provider = Provider::Heuristic;
            s.last_error = Some(e.clone());
            drop(s);
            // Whatever was connected before is gone: the core must stop
            // sending requests to it.
            post_heuristic(&inbox);
            Err(e)
        }
    }
}

fn post_heuristic(inbox: &InboxQueue) {
    post(
        inbox,
        Inbox::Infer(InferEvent::Ready {
            engine: "heuristic (no credentials)".into(),
            model_id: "deterministic policy".into(),
            is_real_model: false,
        }),
    );
}

fn abort_all(s: &mut Shared) {
    for (_, a) in std::mem::take(&mut s.in_flight) {
        a.cancel();
    }
}

/// Forget the credential, cancel outstanding work and return to the
/// deterministic policy.
pub fn disconnect(shared: &SharedState, inbox: &InboxQueue) {
    {
        let mut s = shared.borrow_mut();
        s.connect_epoch += 1;
        // The turn is cancelled first, so the aborted request's late result
        // is recognised as stale.
        post(inbox, Inbox::Ui(Command::CancelVoice));
        abort_all(&mut s);
        // Dropping the connection wipes the key.
        s.provider = Provider::Heuristic;
    }
    post(
        inbox,
        Inbox::Infer(InferEvent::Ready {
            engine: "heuristic (no credentials)".into(),
            model_id: "deterministic policy".into(),
            is_real_model: false,
        }),
    );
    notice(
        inbox,
        "disconnected from OpenRouter; the key was cleared from memory",
    );
}

/// Provider status for the page. Never includes the key.
pub fn status(shared: &SharedState) -> serde_json::Value {
    let s = shared.borrow();
    let (provider, model, structured, key_label, limit_remaining, budget) = match &s.provider {
        Provider::Heuristic => ("heuristic", None, None, None, None, None),
        Provider::Connecting => ("connecting", None, None, None, None, None),
        Provider::OpenRouter(c) => (
            "openrouter",
            Some(c.cfg.model.clone()),
            Some(c.cfg.structured.label()),
            c.key_info.as_ref().and_then(|k| k.label.clone()),
            c.key_info.as_ref().and_then(|k| k.limit_remaining),
            Some(c.cfg.budget.clone()),
        ),
    };
    json!({
        "provider": provider,
        "model": model,
        "structured": structured,
        "key_label": key_label,
        "key_limit_remaining_usd": limit_remaining,
        "budget": budget,
        "in_flight": s.in_flight.len(),
        "usage": s.meter,
        "last_error": s.last_error,
        "voice": {"tts": s.voice.tts, "asr": s.voice.asr},
    })
}

/// The structured-output-capable models, for the page's model picker.
pub async fn list_models() -> Result<Vec<ModelInfo>, String> {
    let abort = Abort::new().map_err(|e| e.to_string())?;
    let mut models = or_web::fetch_models(&OpenRouterConfig::default(), true, &abort)
        .await
        .map_err(|e| e.to_string())?;
    models.retain(ModelInfo::supports_structured);
    models.sort_by(|a, b| {
        let price = |m: &ModelInfo| m.completion_price.unwrap_or(f64::MAX);
        price(a)
            .partial_cmp(&price(b))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(models)
}

// ---------------------------------------------------------------------------
// Voice (Web Speech API via the page)
// ---------------------------------------------------------------------------

/// Which browser speech features exist (`{recognition, synthesis}`).
pub fn speech_support() -> (bool, bool) {
    let v = host_speech_support();
    let get = |k: &str| {
        js_sys::Reflect::get(&v, &JsValue::from_str(k))
            .ok()
            .and_then(|x| x.as_bool())
            .unwrap_or(false)
    };
    (get("recognition"), get("synthesis"))
}

/// Turn browser voice input/output on or off and report readiness to the
/// core. Typed dialogue always works regardless.
pub fn set_voice(shared: &SharedState, inbox: &InboxQueue, tts: bool, asr: bool) {
    let (rec, synth) = speech_support();
    let asr = asr && rec;
    let tts = tts && synth;
    shared.borrow_mut().voice = VoiceSettings { tts, asr };
    let off = |what: &str, supported: bool| {
        if supported {
            format!("{what} is off - enable it in the panel (Enter always types)")
        } else {
            format!("{what} is not supported by this browser (Enter types)")
        }
    };
    if asr {
        post(
            inbox,
            Inbox::Mic(MicEvent::DeviceReady(MicInfo {
                name: "browser speech recognition".into(),
                sample_rate: 16_000,
                channels: 1,
                sample_format: "Web Speech API - may use an online service".into(),
            })),
        );
        post(
            inbox,
            Inbox::Asr(AsrEvent::Ready {
                backend: "Web Speech API (browser-provided; may be online)".into(),
            }),
        );
    } else {
        post(
            inbox,
            Inbox::Ui(Command::VoiceUnavailable {
                mic: true,
                asr: true,
                tts: false,
                why: off("voice input", rec),
            }),
        );
    }
    if tts {
        post(
            inbox,
            Inbox::Tts(TtsEvent::Ready {
                backend: "speechSynthesis".into(),
                device: "browser".into(),
                sample_rate: 0,
            }),
        );
    } else {
        post(
            inbox,
            Inbox::Ui(Command::VoiceUnavailable {
                mic: false,
                asr: false,
                tts: true,
                why: off("spoken NPC lines", synth),
            }),
        );
    }
}

/// A callback from `web/host.js`.
pub fn host_event(
    shared: &SharedState,
    inbox: &InboxQueue,
    kind: &str,
    conv: u64,
    line: u64,
    text: String,
) {
    let mut s = shared.borrow_mut();
    let now = now_ms();
    let held = |s: &Shared| {
        s.listening
            .filter(|(c, _)| *c == conv)
            .map(|(_, t)| ((now - t) / 1000.0) as f32)
            .unwrap_or(0.0)
    };
    match kind {
        "recognition_result" | "recognition_nomatch" | "recognition_error" => {
            let seconds = held(&s);
            if s.listening.is_some_and(|(c, _)| c == conv) {
                s.listening = None;
            }
            drop(s);
            if kind == "recognition_error"
                && matches!(
                    text.as_str(),
                    "not-allowed" | "service-not-allowed" | "audio-capture"
                )
            {
                post(
                    inbox,
                    Inbox::Mic(MicEvent::DeviceError(format!(
                        "microphone unavailable ({text}) - allow microphone access, or press Enter to type"
                    ))),
                );
                return;
            }
            if kind == "recognition_error" && text == "aborted" {
                return;
            }
            // The browser recognizer delivers text, not audio: the turn
            // still passes through the same Captured → Transcript stages.
            post(
                inbox,
                Inbox::Mic(MicEvent::Captured {
                    conv,
                    samples: Vec::new(),
                    sample_rate: 16_000,
                    recorded_seconds: seconds,
                    speech_seconds: seconds,
                }),
            );
            let msg = match kind {
                "recognition_result" if !text.trim().is_empty() => {
                    Inbox::Asr(AsrEvent::Transcript {
                        conv,
                        text,
                        latency_ms: 0.0,
                    })
                }
                "recognition_error" if text == "network" => Inbox::Asr(AsrEvent::Failed {
                    conv,
                    error: "the browser's speech service is unreachable - type instead".into(),
                }),
                "recognition_error" => Inbox::Asr(AsrEvent::Failed {
                    conv,
                    error: format!("browser speech recognition: {text}"),
                }),
                _ => Inbox::Asr(AsrEvent::Failed {
                    conv,
                    error: "browser speech recognition heard no words".into(),
                }),
            };
            post(inbox, msg);
        }
        "tts_start" => {
            let started = s.speaking.get(&line).copied().unwrap_or(now);
            drop(s);
            post(
                inbox,
                Inbox::Tts(TtsEvent::Started {
                    conv,
                    line,
                    first_audio_ms: (now - started) as f32,
                }),
            );
        }
        "tts_end" => {
            let started = s.speaking.remove(&line).unwrap_or(now);
            drop(s);
            post(
                inbox,
                Inbox::Tts(TtsEvent::Finished {
                    conv,
                    line,
                    audio_seconds: 0.0,
                    synth_ms: (now - started) as f32,
                }),
            );
        }
        "tts_error" => {
            s.speaking.remove(&line);
            drop(s);
            let ev = if matches!(text.as_str(), "interrupted" | "canceled") {
                TtsEvent::Cancelled { conv, line }
            } else {
                TtsEvent::Failed {
                    conv,
                    line,
                    error: format!("speechSynthesis: {text}"),
                }
            };
            post(inbox, Inbox::Tts(ev));
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

/// Load a slot from browser storage into the simulation.
pub fn load_slot(inbox: &InboxQueue, slot: &str) {
    let inbox = inbox.clone();
    let slot = slot.to_string();
    spawn_local(async move {
        let result = match host_storage_get(&slot) {
            Ok(p) => JsFuture::from(p).await,
            Err(e) => Err(e),
        };
        match result {
            Ok(v) => match v.as_string() {
                Some(json) => post(
                    &inbox,
                    Inbox::Ui(Command::ImportSave {
                        label: format!("browser storage ({slot})"),
                        json,
                    }),
                ),
                None => notice(
                    &inbox,
                    "no saved game in browser storage yet (K or F5 saves)",
                ),
            },
            Err(e) => notice(
                &inbox,
                format!(
                    "browser storage unavailable ({}) - use Export save instead",
                    js_text(&e)
                ),
            ),
        }
    });
}

// ---------------------------------------------------------------------------
// The dispatcher
// ---------------------------------------------------------------------------

pub struct BrowserDispatcher {
    shared: SharedState,
}

impl BrowserDispatcher {
    pub fn new(shared: SharedState) -> Self {
        Self { shared }
    }

    fn decide(&mut self, job: rpg_runtime::InferJob, inbox: &InboxQueue) {
        let fail = |e: InferenceError| {
            post(
                inbox,
                Inbox::Infer(InferEvent::Done(InferDone {
                    conv: job.conv,
                    request_id: job.request_id,
                    world_epoch: job.world_epoch,
                    result: Err(e),
                    queue_ms: 0.0,
                    infer_ms: 0.0,
                })),
            )
        };
        let mut s = self.shared.borrow_mut();
        let (key, cfg, model) = match &s.provider {
            Provider::OpenRouter(c) => (c.key.clone(), c.cfg.clone(), c.model.clone()),
            _ => {
                drop(s);
                fail(InferenceError::NotLoaded(
                    "not connected to OpenRouter".into(),
                ));
                return;
            }
        };
        if let Err(e) = s.meter.check(&cfg.budget) {
            s.meter.last_error = Some(e.to_string());
            drop(s);
            fail(e.into_inference());
            return;
        }
        let abort = match Abort::new() {
            Ok(a) => a,
            Err(e) => {
                drop(s);
                fail(e.into_inference());
                return;
            }
        };
        s.in_flight.insert(job.request_id, abort.clone());
        drop(s);

        let shared = self.shared.clone();
        let inbox = inbox.clone();
        spawn_local(async move {
            let start = now_ms();
            let budget_shared = shared.clone();
            let budget = cfg.budget.clone();
            let mut before_attempt = move || {
                let mut s = budget_shared.borrow_mut();
                s.meter.check(&budget)?;
                s.meter.begin();
                Ok(())
            };
            let progress_inbox = inbox.clone();
            let progress_shared = shared.clone();
            let (conv, request_id) = (job.conv, job.request_id);
            let mut progress = move |chunks: u32| {
                // Count only, at most every 100 ms.
                let now = now_ms();
                let mut s = progress_shared.borrow_mut();
                if now - s.last_progress_ms >= 100.0 {
                    s.last_progress_ms = now;
                    post(
                        &progress_inbox,
                        Inbox::Infer(InferEvent::Progress {
                            conv,
                            request_id,
                            chunks,
                        }),
                    );
                }
            };
            let result = or_web::complete(
                &cfg,
                &key,
                &job.request,
                &abort,
                &mut progress,
                &mut before_attempt,
            )
            .await;
            drop(key);
            let latency = (now_ms() - start) as f32;
            let result = {
                let mut s = shared.borrow_mut();
                s.in_flight.remove(&job.request_id);
                match result {
                    Ok(c) => {
                        let est = c.usage.as_ref().and_then(|u| {
                            model
                                .as_ref()
                                .and_then(|m| m.estimate_cost(u.prompt_tokens, u.completion_tokens))
                        });
                        s.meter.record_ok(&c, est, latency);
                        Ok(to_inference_response(c, &cfg, &job.request, latency, est))
                    }
                    Err(e) => {
                        s.meter.record_err(&e, latency);
                        s.last_error = Some(e.to_string());
                        Err(e.into_inference())
                    }
                }
            };
            post(
                &inbox,
                Inbox::Infer(InferEvent::Done(InferDone {
                    conv: job.conv,
                    request_id: job.request_id,
                    world_epoch: job.world_epoch,
                    result,
                    queue_ms: 0.0,
                    infer_ms: latency,
                })),
            );
        });
    }
}

impl Dispatcher for BrowserDispatcher {
    fn dispatch(&mut self, effect: Effect, inbox: &InboxQueue) {
        match effect {
            Effect::Infer(InferRequest::Decide(job)) => self.decide(job, inbox),
            Effect::Infer(InferRequest::Cancel { request_id }) => {
                if let Some(a) = self.shared.borrow_mut().in_flight.remove(&request_id) {
                    a.cancel();
                }
            }
            Effect::Infer(InferRequest::Shutdown) => abort_all(&mut self.shared.borrow_mut()),
            Effect::Mic(MicRequest::Start { conv }) => {
                let enabled = self.shared.borrow().voice.asr;
                let started = if enabled {
                    host_recognition_start(conv as f64)
                } else {
                    Err(JsValue::from_str("voice input is off"))
                };
                match started {
                    Ok(()) => self.shared.borrow_mut().listening = Some((conv, now_ms())),
                    Err(e) => post(
                        inbox,
                        Inbox::Mic(MicEvent::DeviceError(format!(
                            "speech recognition could not start: {}",
                            js_text(&e)
                        ))),
                    ),
                }
            }
            Effect::Mic(MicRequest::Stop { conv }) => {
                if let Err(e) = host_recognition_stop(conv as f64) {
                    post(
                        inbox,
                        Inbox::Mic(MicEvent::DeviceError(format!(
                            "speech recognition failed: {}",
                            js_text(&e)
                        ))),
                    );
                }
            }
            Effect::Mic(MicRequest::Cancel { .. }) => {
                self.shared.borrow_mut().listening = None;
                let _ = host_recognition_abort();
            }
            Effect::Mic(MicRequest::SelfTest { .. }) => post(
                inbox,
                Inbox::Mic(MicEvent::SelfTest(Err(
                    "the raw microphone self-test is a native diagnostic".into(),
                ))),
            ),
            Effect::Mic(MicRequest::Select(_)) | Effect::Mic(MicRequest::Shutdown) => {}
            // The browser recognizer already produced text; the transcript
            // is queued right behind the `Captured` event.
            Effect::Asr(AsrRequest::Transcribe { .. })
            | Effect::Asr(AsrRequest::Cancel { .. })
            | Effect::Asr(AsrRequest::Shutdown) => {}
            Effect::Tts(TtsRequest::Speak(job)) => {
                self.shared.borrow_mut().speaking.insert(job.line, now_ms());
                if let Err(e) =
                    host_speak(job.conv as f64, job.line as f64, &job.speaker, &job.text)
                {
                    post(
                        inbox,
                        Inbox::Tts(TtsEvent::Failed {
                            conv: job.conv,
                            line: job.line,
                            error: js_text(&e),
                        }),
                    );
                }
            }
            Effect::Tts(TtsRequest::Shutdown) => {}
            // A new playback epoch: whatever is being spoken is stale.
            Effect::SetAudioEpoch(_) => {
                let _ = host_stop_speaking();
            }
            Effect::SaveData { slot, json } => {
                if slot == "export" {
                    let name = format!("irl-save-{}.json", js_sys::Date::now() as u64);
                    match host_download_text(&name, &json) {
                        Ok(()) => notice(inbox, format!("exported save as {name}")),
                        Err(e) => notice(inbox, format!("export failed: {}", js_text(&e))),
                    }
                    return;
                }
                let inbox = inbox.clone();
                spawn_local(async move {
                    let r = match host_storage_put(&slot, &json) {
                        Ok(p) => JsFuture::from(p).await.map(|_| ()),
                        Err(e) => Err(e),
                    };
                    match r {
                        Ok(()) => notice(&inbox, format!("saved to browser storage ({slot})")),
                        Err(e) => notice(
                            &inbox,
                            format!(
                                "browser storage unavailable ({}) - use Export save instead",
                                js_text(&e)
                            ),
                        ),
                    }
                });
            }
        }
    }
}
