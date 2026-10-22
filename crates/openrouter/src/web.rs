//! Browser OpenRouter transport (`wasm32`).
//!
//! Requests use the page's own `fetch`, so the browser's TLS, proxy and CORS
//! handling apply and nothing native is linked into the WASM build. Every
//! request is a future on the browser event loop — no blocking, no
//! synchronous XHR, no busy-waiting — and carries an [`Abort`] handle:
//! cancelling (or the request deadline) calls `AbortController.abort()`,
//! which rejects the pending `fetch`/stream read and closes the connection.
//!
//! Requests are ordinary CORS requests (`mode: "cors"`,
//! `credentials: "omit"`); `no-cors` is never used, since it would make the
//! authenticated response unreadable.

use std::cell::Cell;
use std::rc::Rc;

use js_sys::{Promise, Reflect, Uint8Array};
use rpg_domain::ports::InferenceRequest;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    AbortController, Headers, ReadableStreamDefaultReader, Request, RequestCredentials,
    RequestInit, RequestMode, Response,
};

use crate::sse::{SseParser, StreamAccumulator, StreamStep};
use crate::{
    chat_body, classify_http_error, parse_completion, parse_key_info, parse_models, ApiKey,
    Completion, KeyInfo, ModelInfo, OpenRouterConfig, OpenRouterError,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reason {
    None,
    Cancelled,
    Timeout,
}

/// Cancels one in-flight request. Cheap to clone; all clones abort the same
/// request.
#[derive(Clone)]
pub struct Abort {
    controller: AbortController,
    reason: Rc<Cell<Reason>>,
}

impl Abort {
    pub fn new() -> Result<Self, OpenRouterError> {
        Ok(Self {
            controller: AbortController::new()
                .map_err(|e| OpenRouterError::Network(js_error(&e)))?,
            reason: Rc::new(Cell::new(Reason::None)),
        })
    }

    /// Abort the request (player cancel, disconnect, superseded turn).
    pub fn cancel(&self) {
        if self.reason.get() == Reason::None {
            self.reason.set(Reason::Cancelled);
        }
        self.controller.abort();
    }

    fn time_out(&self) {
        if self.reason.get() == Reason::None {
            self.reason.set(Reason::Timeout);
        }
        self.controller.abort();
    }

    pub fn is_cancelled(&self) -> bool {
        self.reason.get() != Reason::None
    }

    fn error(&self, timeout_ms: u64, fallback: &JsValue) -> OpenRouterError {
        match self.reason.get() {
            Reason::Cancelled => OpenRouterError::Cancelled,
            Reason::Timeout => OpenRouterError::Timeout(timeout_ms),
            Reason::None => OpenRouterError::Network(format!(
                "fetch failed — network error or the request was blocked (CORS, \
                 extension, offline): {}",
                js_error(fallback)
            )),
        }
    }
}

/// A `setTimeout` that aborts on drop-safety: cleared when dropped.
struct Deadline {
    handle: i32,
    _closure: Closure<dyn FnMut()>,
}

impl Deadline {
    fn arm(abort: &Abort, ms: u64) -> Option<Self> {
        let window = web_sys::window()?;
        let a = abort.clone();
        let closure = Closure::<dyn FnMut()>::new(move || a.time_out());
        let handle = window
            .set_timeout_with_callback_and_timeout_and_arguments_0(
                closure.as_ref().unchecked_ref(),
                ms.min(i32::MAX as u64) as i32,
            )
            .ok()?;
        Some(Self {
            handle,
            _closure: closure,
        })
    }
}

impl Drop for Deadline {
    fn drop(&mut self) {
        if let Some(w) = web_sys::window() {
            w.clear_timeout_with_handle(self.handle);
        }
    }
}

fn js_error(v: &JsValue) -> String {
    if let Some(e) = v.dyn_ref::<js_sys::Error>() {
        return format!("{}: {}", e.name(), e.message());
    }
    v.as_string().unwrap_or_else(|| format!("{v:?}"))
}

/// Milliseconds on the page's monotonic clock.
pub fn now_ms() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or_else(js_sys::Date::now)
}

/// Resolve after `ms` without blocking the event loop.
pub async fn sleep_ms(ms: u64) {
    let promise = Promise::new(&mut |resolve, _| {
        if let Some(w) = web_sys::window() {
            let _ = w.set_timeout_with_callback_and_timeout_and_arguments_0(
                &resolve,
                ms.min(i32::MAX as u64) as i32,
            );
        }
    });
    let _ = JsFuture::from(promise).await;
}

async fn send(
    url: &str,
    method: &str,
    headers: &[(&'static str, String)],
    body: Option<&str>,
    abort: &Abort,
    timeout_ms: u64,
) -> Result<Response, OpenRouterError> {
    let init = RequestInit::new();
    init.set_method(method);
    init.set_mode(RequestMode::Cors);
    // Bearer auth only; never send cookies to the API.
    init.set_credentials(RequestCredentials::Omit);
    let h = Headers::new().map_err(|e| OpenRouterError::Network(js_error(&e)))?;
    for (k, v) in headers {
        h.set(k, v)
            .map_err(|_| OpenRouterError::Protocol(format!("invalid header {k}")))?;
    }
    init.set_headers_headers(&h);
    if let Some(b) = body {
        init.set_body(&JsValue::from_str(b));
    }
    init.set_signal(Some(&abort.controller.signal()));
    let request = Request::new_with_str_and_init(url, &init)
        .map_err(|e| OpenRouterError::Protocol(js_error(&e)))?;
    let window = web_sys::window().ok_or_else(|| OpenRouterError::Network("no window".into()))?;
    let resp = JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(|e| abort.error(timeout_ms, &e))?;
    resp.dyn_into::<Response>()
        .map_err(|_| OpenRouterError::Protocol("fetch did not return a Response".into()))
}

async fn text_of(
    resp: &Response,
    abort: &Abort,
    timeout_ms: u64,
) -> Result<String, OpenRouterError> {
    let p = resp
        .text()
        .map_err(|e| OpenRouterError::Network(js_error(&e)))?;
    let v = JsFuture::from(p)
        .await
        .map_err(|e| abort.error(timeout_ms, &e))?;
    Ok(v.as_string().unwrap_or_default())
}

async fn error_of(resp: &Response, abort: &Abort, timeout_ms: u64) -> OpenRouterError {
    let retry_after = resp.headers().get("Retry-After").ok().flatten();
    let body = text_of(resp, abort, timeout_ms).await.unwrap_or_default();
    classify_http_error(resp.status(), &body, retry_after.as_deref())
}

async fn get_json(
    url: &str,
    key: Option<&ApiKey>,
    abort: &Abort,
    timeout_ms: u64,
) -> Result<String, OpenRouterError> {
    let _deadline = Deadline::arm(abort, timeout_ms);
    let headers: Vec<(&'static str, String)> = key
        .map(|k| vec![("Authorization", k.bearer())])
        .unwrap_or_default();
    let resp = send(url, "GET", &headers, None, abort, timeout_ms).await?;
    if !resp.ok() {
        return Err(error_of(&resp, abort, timeout_ms).await);
    }
    text_of(&resp, abort, timeout_ms).await
}

/// The public model catalog (no credential is sent).
pub async fn fetch_models(
    cfg: &OpenRouterConfig,
    structured_only: bool,
    abort: &Abort,
) -> Result<Vec<ModelInfo>, OpenRouterError> {
    parse_models(&get_json(&cfg.models_url(structured_only), None, abort, 20_000).await?)
}

/// Validate the key without spending anything (`GET /key`).
pub async fn fetch_key_info(
    cfg: &OpenRouterConfig,
    key: &ApiKey,
    abort: &Abort,
) -> Result<KeyInfo, OpenRouterError> {
    let body = get_json(&cfg.key_url(), Some(key), abort, 20_000)
        .await
        .map_err(|e| e.redact_with(key))?;
    parse_key_info(&body)
}

/// One completion attempt. `progress` receives the number of streamed
/// content chunks so far — never the partial text.
pub async fn complete_once(
    cfg: &OpenRouterConfig,
    key: &ApiKey,
    req: &InferenceRequest,
    abort: &Abort,
    progress: &mut dyn FnMut(u32),
) -> Result<Completion, OpenRouterError> {
    let start = now_ms();
    let _deadline = Deadline::arm(abort, cfg.timeout_ms);
    let body = chat_body(cfg, req, cfg.stream).to_string();
    let headers = cfg.headers(key, true);
    let resp = send(
        &cfg.chat_url(),
        "POST",
        &headers,
        Some(&body),
        abort,
        cfg.timeout_ms,
    )
    .await?;
    if !resp.ok() {
        return Err(error_of(&resp, abort, cfg.timeout_ms).await);
    }
    let is_sse = resp
        .headers()
        .get("Content-Type")
        .ok()
        .flatten()
        .is_some_and(|t| t.contains("event-stream"));
    if !cfg.stream || !is_sse {
        return parse_completion(&text_of(&resp, abort, cfg.timeout_ms).await?);
    }
    let stream = resp
        .body()
        .ok_or_else(|| OpenRouterError::Protocol("stream response has no body".into()))?;
    let reader: ReadableStreamDefaultReader = stream.get_reader().unchecked_into();
    let mut parser = SseParser::new();
    let mut acc = StreamAccumulator::new();
    let mut chunks = 0u32;
    loop {
        let chunk = JsFuture::from(reader.read())
            .await
            .map_err(|e| abort.error(cfg.timeout_ms, &e))?;
        let done = Reflect::get(&chunk, &JsValue::from_str("done"))
            .ok()
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let events = if done {
            parser.finish()
        } else {
            let value = Reflect::get(&chunk, &JsValue::from_str("value"))
                .map_err(|_| OpenRouterError::Protocol("stream chunk without value".into()))?;
            parser.push(&Uint8Array::new(&value).to_vec())
        };
        let t = (now_ms() - start) as f32;
        for ev in &events {
            match acc.apply(ev, t)? {
                StreamStep::Done => {
                    let _ = reader.cancel();
                    return acc.finish();
                }
                StreamStep::Progress(_) => {
                    chunks += 1;
                    progress(chunks);
                }
                StreamStep::Continue => {}
            }
        }
        if done {
            return acc.finish();
        }
    }
}

/// [`complete_once`] with the bounded transient-error retry the native
/// engine uses. Never retries more than `cfg.max_retries` times, never waits
/// out a long `Retry-After`, never switches models.
pub async fn complete(
    cfg: &OpenRouterConfig,
    key: &ApiKey,
    req: &InferenceRequest,
    abort: &Abort,
    progress: &mut dyn FnMut(u32),
    attempts: &mut u32,
) -> Result<Completion, OpenRouterError> {
    let mut retries = 0;
    loop {
        *attempts += 1;
        match complete_once(cfg, key, req, abort, progress).await {
            Ok(c) => return Ok(c),
            Err(e) => {
                let e = e.redact_with(key);
                match (retries < cfg.max_retries)
                    .then(|| e.retry_delay_ms())
                    .flatten()
                {
                    Some(ms) if !abort.is_cancelled() => {
                        retries += 1;
                        sleep_ms(ms).await;
                        if abort.is_cancelled() {
                            return Err(OpenRouterError::Cancelled);
                        }
                    }
                    _ => return Err(e),
                }
            }
        }
    }
}
