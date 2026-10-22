//! # rpg_openrouter
//!
//! [OpenRouter](https://openrouter.ai/docs/quickstart) as an *optional*
//! inference provider behind [`rpg_domain::ports::InferenceEngine`].
//!
//! Everything that does not touch the network is shared by both transports:
//! request bodies (including the JSON-schema `response_format` built from the
//! shared [`rpg_domain::contract`]), response and error parsing, the SSE
//! stream parser, usage/cost accounting, request budgets and the model
//! catalog. The transports are thin:
//!
//! | target | module | HTTP | runs on |
//! |---|---|---|---|
//! | native | [`native`] | blocking `ureq` | the runtime's inference worker thread |
//! | `wasm32` | [`web`] | the page's `fetch` + `AbortController` | the browser event loop (futures) |
//!
//! A completion is only ever an *untrusted proposal*. Whatever the provider
//! claims (`strict: true` included), the answer is re-validated against the
//! action contract, resolved by the canonical NPC policy and validated by the
//! authoritative simulation before anything is committed. Nothing in this
//! crate can touch world state.
//!
//! ## Credentials
//!
//! [`ApiKey`] never prints its value (`Debug` is redacted, there is no
//! `Display`), is never serialized, and is wiped on drop. Error messages are
//! passed through [`redact`] before they leave the crate.

pub mod sse;

#[cfg(not(target_arch = "wasm32"))]
pub mod native;
#[cfg(target_arch = "wasm32")]
pub mod web;

use rpg_domain::contract::{strict_action_schema, ACTION_CONTRACT_ID, ACTION_SCHEMA_NAME};
use rpg_domain::ports::{InferenceError, InferenceRequest, InferenceResponse, RemoteProvenance};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use thiserror::Error;

pub use sse::{SseEvent, SseParser, StreamAccumulator};

/// Production API base.
pub const DEFAULT_API_BASE: &str = "https://openrouter.ai/api/v1";

/// The documented default model candidate.
///
/// Chosen on 2026-10-22 from
/// `GET /api/v1/models?supported_parameters=structured_outputs`, which lists
/// it with `structured_outputs`, `response_format` and `seed` support at a
/// low per-token price. Catalogs change: the model is always user-editable
/// (`--openrouter-model`, the browser's model field), and both clients check
/// the live catalog for structured-output support before relying on it.
pub const DEFAULT_MODEL: &str = "openai/gpt-6-luna";

/// Sent as `X-OpenRouter-Title` so requests are attributable on OpenRouter.
pub const APP_TITLE: &str = "IRL";

/// Environment variable the native client reads the key from.
pub const API_KEY_ENV: &str = "OPENROUTER_API_KEY";

// ---------------------------------------------------------------------------
// Credentials
// ---------------------------------------------------------------------------

/// An OpenRouter API key. Never printed, never serialized, wiped on drop.
#[derive(Clone, PartialEq, Eq)]
pub struct ApiKey(String);

impl ApiKey {
    /// Wrap a key. Surrounding whitespace (a pasted newline) is removed;
    /// an empty key is rejected.
    pub fn new(raw: impl Into<String>) -> Result<Self, OpenRouterError> {
        let raw = raw.into();
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(OpenRouterError::Auth("no API key provided".into()));
        }
        if trimmed.chars().any(|c| c.is_control() || c.is_whitespace()) {
            return Err(OpenRouterError::Auth(
                "the API key contains whitespace or control characters".into(),
            ));
        }
        Ok(Self(trimmed.to_string()))
    }

    /// The key value, for building the `Authorization` header only.
    pub fn expose(&self) -> &str {
        &self.0
    }

    /// `Bearer <key>`.
    pub fn bearer(&self) -> String {
        format!("Bearer {}", self.0)
    }
}

impl std::fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ApiKey(<redacted>)")
    }
}

impl Drop for ApiKey {
    fn drop(&mut self) {
        // Best-effort wipe of the heap copy we own.
        let mut bytes = std::mem::take(&mut self.0).into_bytes();
        for b in bytes.iter_mut() {
            // Volatile so the store is not optimised away.
            unsafe { std::ptr::write_volatile(b, 0) };
        }
    }
}

/// Replace every occurrence of the key (and its `Bearer` form) in `text`.
pub fn redact(text: &str, key: Option<&ApiKey>) -> String {
    match key {
        Some(k) if !k.0.is_empty() => text.replace(&k.0, "<redacted>"),
        _ => text.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// How the action contract is requested from the provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredMode {
    /// `response_format: json_schema` with `strict: true` and
    /// `provider.require_parameters: true`, so OpenRouter only routes to
    /// endpoints that support structured outputs. The default.
    JsonSchema,
    /// Explicit compatibility fallback for models without structured
    /// outputs: `response_format: json_object`. Responses are reported as
    /// *unstructured* — no schema guarantee is claimed — and still validated.
    JsonObject,
    /// No `response_format`; only the prompt asks for JSON. Reported as
    /// unstructured.
    PromptOnly,
}

impl StructuredMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "json_schema" | "schema" | "strict" => Some(Self::JsonSchema),
            "json_object" | "object" | "json" => Some(Self::JsonObject),
            "prompt" | "prompt_only" | "none" => Some(Self::PromptOnly),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::JsonSchema => "json_schema (strict)",
            Self::JsonObject => "json_object (compatibility; no schema guarantee)",
            Self::PromptOnly => "prompt only (no schema guarantee)",
        }
    }
}

/// Per-session spending guards for test runs. `None` means unlimited.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Budget {
    /// Maximum completion requests this session.
    pub max_requests: Option<u32>,
    /// Maximum completion tokens (reported) this session.
    pub max_completion_tokens: Option<u64>,
    /// Maximum *reported* cost in USD this session (falls back to the local
    /// estimate when the provider reports no cost).
    pub max_cost_usd: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OpenRouterConfig {
    pub api_base: String,
    /// OpenRouter model slug, e.g. `openai/gpt-6-luna`.
    pub model: String,
    /// Output-token cap per request (`max_tokens`).
    pub max_output_tokens: u32,
    /// Overrides the request's temperature when set.
    pub temperature: Option<f32>,
    /// Send the request seed. Providers may ignore it; it never makes remote
    /// output deterministic.
    pub send_seed: bool,
    /// Send `temperature`. Turned off for models that do not list it, since
    /// `require_parameters` would otherwise route to no endpoint.
    pub send_temperature: bool,
    /// Name of the output-token limit parameter the model accepts
    /// (`max_tokens`, or `max_completion_tokens` for models listing only it).
    pub max_tokens_param: String,
    pub structured: StructuredMode,
    /// Use SSE streaming (enables early cancellation on providers that
    /// support it, and progress feedback).
    pub stream: bool,
    /// Whole-request deadline.
    pub timeout_ms: u64,
    /// Optional `reasoning.effort` for reasoning models (e.g. `"low"`).
    pub reasoning_effort: Option<String>,
    /// Optional `HTTP-Referer` attribution header (native only; browsers set
    /// their own `Referer`).
    pub referer: Option<String>,
    pub title: String,
    pub budget: Budget,
    /// Bounded retries for transient failures (429 with a short
    /// `Retry-After`, 502, 503). Never more than this; never a model switch.
    pub max_retries: u32,
}

impl Default for OpenRouterConfig {
    fn default() -> Self {
        Self {
            api_base: DEFAULT_API_BASE.into(),
            model: DEFAULT_MODEL.into(),
            max_output_tokens: 400,
            temperature: None,
            send_seed: true,
            send_temperature: true,
            max_tokens_param: "max_tokens".into(),
            structured: StructuredMode::JsonSchema,
            stream: true,
            timeout_ms: 40_000,
            reasoning_effort: None,
            referer: None,
            title: APP_TITLE.into(),
            budget: Budget::default(),
            max_retries: 1,
        }
    }
}

impl OpenRouterConfig {
    pub fn chat_url(&self) -> String {
        format!("{}/chat/completions", self.api_base.trim_end_matches('/'))
    }

    /// The public model catalog, optionally filtered to models with
    /// structured-output support.
    pub fn models_url(&self, structured_only: bool) -> String {
        let base = format!("{}/models", self.api_base.trim_end_matches('/'));
        if structured_only {
            format!("{base}?supported_parameters=structured_outputs")
        } else {
            base
        }
    }

    /// `GET /key`: validates the key without spending anything.
    pub fn key_url(&self) -> String {
        format!("{}/key", self.api_base.trim_end_matches('/'))
    }

    /// Request headers, *including* the credential. The caller must never log
    /// these.
    pub fn headers(&self, key: &ApiKey, browser: bool) -> Vec<(&'static str, String)> {
        let mut h = vec![
            ("Authorization", key.bearer()),
            ("Content-Type", "application/json".to_string()),
            ("X-OpenRouter-Title", self.title.clone()),
        ];
        if !browser {
            if let Some(r) = &self.referer {
                h.push(("HTTP-Referer", r.clone()));
            }
        }
        h
    }
}

// ---------------------------------------------------------------------------
// Requests
// ---------------------------------------------------------------------------

/// Whether a request asks for the action contract.
fn wants_action_contract(req: &InferenceRequest) -> bool {
    req.response_contract.as_deref() == Some(ACTION_CONTRACT_ID)
}

/// Whether a response to `req` under `cfg` is schema-enforced by the
/// provider (and may be reported as `structured`).
pub fn is_structured(cfg: &OpenRouterConfig, req: &InferenceRequest) -> bool {
    wants_action_contract(req) && cfg.structured == StructuredMode::JsonSchema
}

/// The JSON body of `POST /chat/completions` for one NPC request. The
/// messages are exactly the NPC's scoped system prompt and rendered context —
/// the same input contract the local model receives.
pub fn chat_body(cfg: &OpenRouterConfig, req: &InferenceRequest, stream: bool) -> Value {
    let mut body = json!({
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": req.system},
            {"role": "user", "content": req.prompt},
        ],
        "stream": stream,
    });
    let obj = body.as_object_mut().expect("object");
    obj.insert(
        cfg.max_tokens_param.clone(),
        json!(cfg.max_output_tokens.max(16)),
    );
    if cfg.send_temperature {
        obj.insert(
            "temperature".into(),
            json!(cfg.temperature.unwrap_or(req.temperature).clamp(0.0, 2.0)),
        );
    }
    if cfg.send_seed {
        // Keep the seed in the positive 32-bit range every provider accepts.
        obj.insert("seed".into(), json!(req.seed & 0x7fff_ffff));
    }
    if let Some(effort) = &cfg.reasoning_effort {
        obj.insert("reasoning".into(), json!({ "effort": effort }));
    }
    if wants_action_contract(req) {
        match cfg.structured {
            StructuredMode::JsonSchema => {
                obj.insert(
                    "response_format".into(),
                    json!({
                        "type": "json_schema",
                        "json_schema": {
                            "name": ACTION_SCHEMA_NAME,
                            "strict": true,
                            "schema": strict_action_schema(),
                        }
                    }),
                );
                // Only route to endpoints that support every parameter we
                // sent, so an endpoint without structured outputs is never
                // silently used.
                obj.insert("provider".into(), json!({ "require_parameters": true }));
            }
            StructuredMode::JsonObject => {
                obj.insert("response_format".into(), json!({ "type": "json_object" }));
            }
            StructuredMode::PromptOnly => {}
        }
    }
    body
}

// ---------------------------------------------------------------------------
// Responses
// ---------------------------------------------------------------------------

/// Token and cost accounting as the provider reported it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub reasoning_tokens: Option<u32>,
    /// USD, as reported by OpenRouter (`usage.cost`).
    pub cost_usd: Option<f64>,
}

impl Usage {
    pub fn from_json(v: &Value) -> Option<Self> {
        let u = v.as_object()?;
        let n = |k: &str| u.get(k).and_then(Value::as_u64).map(|x| x as u32);
        Some(Self {
            prompt_tokens: n("prompt_tokens").unwrap_or(0),
            completion_tokens: n("completion_tokens").unwrap_or(0),
            reasoning_tokens: u
                .get("completion_tokens_details")
                .and_then(|d| d.get("reasoning_tokens"))
                .and_then(Value::as_u64)
                .map(|x| x as u32),
            cost_usd: u.get("cost").and_then(Value::as_f64),
        })
    }
}

/// One finished completion, transport-independent.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Completion {
    pub text: String,
    pub usage: Option<Usage>,
    /// Model the provider says answered.
    pub model: Option<String>,
    /// Upstream provider, when reported.
    pub provider: Option<String>,
    pub id: Option<String>,
    pub finish_reason: Option<String>,
    pub streamed: bool,
    /// Milliseconds from request start to the first content byte (streams).
    pub first_token_ms: Option<f32>,
}

/// Parse a non-streaming `chat.completion` body.
///
/// A `200` whose body holds only an `error` object is a failure, not an
/// empty answer.
pub fn parse_completion(body: &str) -> Result<Completion, OpenRouterError> {
    let v: Value = serde_json::from_str(body)
        .map_err(|e| OpenRouterError::Protocol(format!("response is not JSON: {e}")))?;
    if let Some(err) = v.get("error") {
        return Err(error_from_object(err));
    }
    let choice = v
        .get("choices")
        .and_then(|c| c.get(0))
        .ok_or_else(|| OpenRouterError::Protocol("response has no choices".into()))?;
    let finish = choice
        .get("finish_reason")
        .and_then(Value::as_str)
        .map(str::to_string);
    if finish.as_deref() == Some("error") {
        return Err(OpenRouterError::Provider(
            "provider reported finish_reason \"error\"".into(),
        ));
    }
    if finish.as_deref() == Some("length") {
        return Err(truncated());
    }
    let message = choice.get("message");
    let text = message
        .and_then(|m| m.get("content"))
        .map(content_text)
        .unwrap_or_default();
    if text.trim().is_empty() {
        let refusal = message
            .and_then(|m| m.get("refusal"))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty());
        return Err(match refusal {
            Some(r) => OpenRouterError::Provider(format!("model refused: {r}")),
            None => OpenRouterError::Protocol(format!(
                "empty completion (finish_reason {})",
                finish.as_deref().unwrap_or("none")
            )),
        });
    }
    Ok(Completion {
        text,
        usage: v.get("usage").and_then(Usage::from_json),
        model: v.get("model").and_then(Value::as_str).map(str::to_string),
        provider: v
            .get("provider")
            .and_then(Value::as_str)
            .map(str::to_string),
        id: v.get("id").and_then(Value::as_str).map(str::to_string),
        finish_reason: finish,
        streamed: false,
        first_token_ms: None,
    })
}

/// The answer hit the output-token cap before it was complete.
pub(crate) fn truncated() -> OpenRouterError {
    OpenRouterError::Protocol(
        "answer truncated at the output-token limit (raise max output tokens, or the model \
         spent them reasoning)"
            .into(),
    )
}

/// Message content is usually a string, but may be an array of parts.
fn content_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// Convert a completion into the domain response. `structured` is only true
/// when the provider was asked to enforce the schema *and* routing required
/// it; the answer is validated by the policy either way.
pub fn to_inference_response(
    c: Completion,
    cfg: &OpenRouterConfig,
    req: &InferenceRequest,
    latency_ms: f32,
    estimated_cost_usd: Option<f64>,
) -> InferenceResponse {
    let usage = c.usage.clone().unwrap_or_default();
    InferenceResponse {
        text: c.text,
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        first_token_ms: c.first_token_ms.unwrap_or(latency_ms),
        prefill_ms: c.first_token_ms.unwrap_or(0.0),
        decode_ms: latency_ms - c.first_token_ms.unwrap_or(0.0),
        structured: is_structured(cfg, req),
        remote: Some(RemoteProvenance {
            gateway: "openrouter".into(),
            requested_model: cfg.model.clone(),
            model: c.model,
            provider: c.provider,
            generation_id: c.id,
            reported_cost_usd: usage.cost_usd,
            estimated_cost_usd,
            streamed: c.streamed,
            latency_ms,
        }),
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Error, PartialEq)]
pub enum OpenRouterError {
    #[error("authentication failed: {0}")]
    Auth(String),
    #[error("insufficient credits: {0}")]
    InsufficientCredits(String),
    #[error("rate limited{}: {message}", retry_after_ms.map(|ms| format!(" (retry after {:.1}s)", ms as f64 / 1000.0)).unwrap_or_default())]
    RateLimited {
        message: String,
        retry_after_ms: Option<u64>,
    },
    #[error("forbidden (moderation or permissions): {0}")]
    Forbidden(String),
    #[error("structured output unsupported: {0}")]
    StructuredUnsupported(String),
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("provider unavailable (HTTP {status}): {message}")]
    Unavailable {
        status: u16,
        message: String,
        retry_after_ms: Option<u64>,
    },
    #[error("provider error: {0}")]
    Provider(String),
    #[error("HTTP {status}: {message}")]
    Http { status: u16, message: String },
    #[error("network error: {0}")]
    Network(String),
    #[error("request timed out after {0} ms")]
    Timeout(u64),
    #[error("request cancelled")]
    Cancelled,
    #[error("session budget exhausted: {0}")]
    Budget(String),
    #[error("protocol error: {0}")]
    Protocol(String),
}

impl OpenRouterError {
    /// How long to wait before the one permitted retry, if this error is
    /// transient. Long `Retry-After` values are not waited out.
    pub fn retry_delay_ms(&self) -> Option<u64> {
        const MAX_WAIT_MS: u64 = 3_000;
        match self {
            Self::RateLimited { retry_after_ms, .. } => match retry_after_ms {
                Some(ms) if *ms <= MAX_WAIT_MS => Some(*ms),
                Some(_) => None,
                None => Some(1_000),
            },
            Self::Unavailable { retry_after_ms, .. } => match retry_after_ms {
                Some(ms) if *ms <= MAX_WAIT_MS => Some(*ms),
                Some(_) => None,
                None => Some(500),
            },
            _ => None,
        }
    }

    /// Map onto the domain error the runtime understands.
    pub fn into_inference(self) -> InferenceError {
        match self {
            Self::Timeout(ms) => InferenceError::Timeout(ms),
            Self::Cancelled => InferenceError::Cancelled,
            Self::Auth(m) => InferenceError::NotLoaded(format!("OpenRouter authentication: {m}")),
            other => InferenceError::Backend(format!("OpenRouter {other}")),
        }
    }

    fn redacted(self, key: Option<&ApiKey>) -> Self {
        let r = |s: String| redact(&s, key);
        match self {
            Self::Auth(m) => Self::Auth(r(m)),
            Self::InsufficientCredits(m) => Self::InsufficientCredits(r(m)),
            Self::RateLimited {
                message,
                retry_after_ms,
            } => Self::RateLimited {
                message: r(message),
                retry_after_ms,
            },
            Self::Forbidden(m) => Self::Forbidden(r(m)),
            Self::StructuredUnsupported(m) => Self::StructuredUnsupported(r(m)),
            Self::BadRequest(m) => Self::BadRequest(r(m)),
            Self::Unavailable {
                status,
                message,
                retry_after_ms,
            } => Self::Unavailable {
                status,
                message: r(message),
                retry_after_ms,
            },
            Self::Provider(m) => Self::Provider(r(m)),
            Self::Http { status, message } => Self::Http {
                status,
                message: r(message),
            },
            Self::Network(m) => Self::Network(r(m)),
            Self::Protocol(m) => Self::Protocol(r(m)),
            other => other,
        }
    }

    /// Scrub the credential from any message text.
    pub fn redact_with(self, key: &ApiKey) -> Self {
        self.redacted(Some(key))
    }
}

/// Parse `Retry-After` (delta-seconds; HTTP dates are treated as unknown).
pub fn parse_retry_after(v: Option<&str>) -> Option<u64> {
    let s = v?.trim();
    s.parse::<f64>()
        .ok()
        .filter(|x| x.is_finite() && *x >= 0.0)
        .map(|x| (x * 1000.0) as u64)
}

fn truncate(s: &str, n: usize) -> String {
    let mut out: String = s.chars().take(n).collect();
    if s.chars().count() > n {
        out.push('…');
    }
    out
}

/// The status code and a readable message of an OpenRouter `error` object.
fn error_parts(err: &Value) -> (Option<u16>, String) {
    let code = err
        .get("code")
        .and_then(|c| {
            c.as_u64()
                .or_else(|| c.as_str().and_then(|s| s.parse().ok()))
        })
        .map(|c| c as u16);
    let mut message = err
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("unknown error")
        .to_string();
    let meta = err.get("metadata");
    if let Some(raw) = meta
        .and_then(|m| m.get("raw"))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        message = format!("{message} — {}", truncate(raw, 200));
    }
    if let Some(p) = meta
        .and_then(|m| m.get("provider_name"))
        .and_then(Value::as_str)
    {
        message = format!("{message} (provider {p})");
    }
    (code, message)
}

/// Build a typed error from an OpenRouter `error` object (in a 200 body or a
/// stream chunk).
pub(crate) fn error_from_object(err: &Value) -> OpenRouterError {
    let (code, message) = error_parts(err);
    classify(code.unwrap_or(0), message, None)
}

fn classify(status: u16, message: String, retry_after_ms: Option<u64>) -> OpenRouterError {
    let lower = message.to_ascii_lowercase();
    let structured_hint = lower.contains("structured")
        || lower.contains("response_format")
        || lower.contains("json_schema");
    match status {
        401 => OpenRouterError::Auth(message),
        402 => OpenRouterError::InsufficientCredits(message),
        403 => OpenRouterError::Forbidden(message),
        404 if structured_hint => OpenRouterError::StructuredUnsupported(message),
        400 if structured_hint => OpenRouterError::StructuredUnsupported(message),
        400 | 404 | 422 => OpenRouterError::BadRequest(message),
        408 => OpenRouterError::Timeout(0),
        429 => OpenRouterError::RateLimited {
            message,
            retry_after_ms,
        },
        502 | 503 => OpenRouterError::Unavailable {
            status,
            message,
            retry_after_ms,
        },
        0 => OpenRouterError::Provider(message),
        s => OpenRouterError::Http { status: s, message },
    }
}

/// Classify a non-2xx response. Never echoes the request (or its key).
pub fn classify_http_error(status: u16, body: &str, retry_after: Option<&str>) -> OpenRouterError {
    let parsed: Option<Value> = serde_json::from_str(body).ok();
    let message = match parsed.as_ref().and_then(|v| v.get("error")) {
        Some(e) => error_parts(e).1,
        None => {
            let t = body.trim();
            if t.is_empty() {
                format!("HTTP {status}")
            } else {
                truncate(t, 200)
            }
        }
    };
    classify(status, message, parse_retry_after(retry_after))
}

// ---------------------------------------------------------------------------
// Session accounting and budgets
// ---------------------------------------------------------------------------

/// Per-session request/token/cost accounting for one provider connection.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct UsageMeter {
    pub requests: u32,
    pub succeeded: u32,
    pub failed: u32,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// Sum of provider-reported cost, if any was reported.
    pub reported_cost_usd: Option<f64>,
    /// Sum of local estimates, if any were made.
    pub estimated_cost_usd: Option<f64>,
    pub last_latency_ms: Option<f32>,
    pub last_first_token_ms: Option<f32>,
    pub last_error: Option<String>,
}

impl UsageMeter {
    /// Refuse a new request that would exceed the budget.
    pub fn check(&self, budget: &Budget) -> Result<(), OpenRouterError> {
        if let Some(max) = budget.max_requests {
            if self.requests >= max {
                return Err(OpenRouterError::Budget(format!(
                    "{} of {max} requests used",
                    self.requests
                )));
            }
        }
        if let Some(max) = budget.max_completion_tokens {
            if self.completion_tokens >= max {
                return Err(OpenRouterError::Budget(format!(
                    "{} of {max} completion tokens used",
                    self.completion_tokens
                )));
            }
        }
        if let Some(max) = budget.max_cost_usd {
            let spent = self
                .reported_cost_usd
                .or(self.estimated_cost_usd)
                .unwrap_or(0.0);
            if spent >= max {
                return Err(OpenRouterError::Budget(format!(
                    "${spent:.4} of ${max:.4} spent"
                )));
            }
        }
        Ok(())
    }

    pub fn begin(&mut self) {
        self.requests += 1;
    }

    pub fn record_ok(&mut self, c: &Completion, estimated: Option<f64>, latency_ms: f32) {
        self.succeeded += 1;
        self.last_latency_ms = Some(latency_ms);
        self.last_first_token_ms = c.first_token_ms;
        self.last_error = None;
        if let Some(u) = &c.usage {
            self.prompt_tokens += u.prompt_tokens as u64;
            self.completion_tokens += u.completion_tokens as u64;
            if let Some(cost) = u.cost_usd {
                *self.reported_cost_usd.get_or_insert(0.0) += cost;
            }
        }
        if let Some(e) = estimated {
            *self.estimated_cost_usd.get_or_insert(0.0) += e;
        }
    }

    pub fn record_err(&mut self, e: &OpenRouterError, latency_ms: f32) {
        self.failed += 1;
        self.last_latency_ms = Some(latency_ms);
        self.last_error = Some(e.to_string());
    }
}

// ---------------------------------------------------------------------------
// Model catalog and key info
// ---------------------------------------------------------------------------

/// One entry of `GET /models`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub context_length: Option<u64>,
    /// USD per prompt token.
    pub prompt_price: Option<f64>,
    /// USD per completion token.
    pub completion_price: Option<f64>,
    pub supported_parameters: Vec<String>,
}

impl ModelInfo {
    pub fn supports(&self, param: &str) -> bool {
        self.supported_parameters.iter().any(|p| p == param)
    }

    /// Structured outputs need both the JSON-schema feature and the
    /// `response_format` parameter.
    pub fn supports_structured(&self) -> bool {
        self.supports("structured_outputs") && self.supports("response_format")
    }

    /// A *local estimate* from published per-token prices. Never presented as
    /// the provider's reported charge.
    pub fn estimate_cost(&self, prompt_tokens: u32, completion_tokens: u32) -> Option<f64> {
        Some(
            self.prompt_price? * prompt_tokens as f64
                + self.completion_price? * completion_tokens as f64,
        )
    }
}

fn price(v: Option<&Value>) -> Option<f64> {
    match v? {
        Value::String(s) => s.parse().ok(),
        Value::Number(n) => n.as_f64(),
        _ => None,
    }
    .filter(|p: &f64| p.is_finite() && *p >= 0.0)
}

/// Parse `GET /models` (`{"data": [...]}`).
pub fn parse_models(body: &str) -> Result<Vec<ModelInfo>, OpenRouterError> {
    let v: Value = serde_json::from_str(body)
        .map_err(|e| OpenRouterError::Protocol(format!("model list is not JSON: {e}")))?;
    let data = v
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| OpenRouterError::Protocol("model list has no data".into()))?;
    Ok(data
        .iter()
        .filter_map(|m| {
            let id = m.get("id")?.as_str()?.to_string();
            Some(ModelInfo {
                name: m
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or(&id)
                    .to_string(),
                context_length: m.get("context_length").and_then(Value::as_u64),
                prompt_price: price(m.get("pricing").and_then(|p| p.get("prompt"))),
                completion_price: price(m.get("pricing").and_then(|p| p.get("completion"))),
                supported_parameters: m
                    .get("supported_parameters")
                    .and_then(Value::as_array)
                    .map(|a| {
                        a.iter()
                            .filter_map(Value::as_str)
                            .map(str::to_string)
                            .collect()
                    })
                    .unwrap_or_default(),
                id,
            })
        })
        .collect())
}

/// Check that `model` can serve requests under `cfg`.
pub fn check_model(model: &ModelInfo, cfg: &OpenRouterConfig) -> Result<(), OpenRouterError> {
    if cfg.structured == StructuredMode::JsonSchema && !model.supports_structured() {
        return Err(OpenRouterError::StructuredUnsupported(format!(
            "{} does not list structured_outputs + response_format support; choose another \
             model, or explicitly select the json_object compatibility mode (no schema guarantee)",
            model.id
        )));
    }
    Ok(())
}

/// Check `model` against `cfg` and drop optional parameters the model does
/// not list, so `provider.require_parameters` never leaves no endpoint to
/// route to. Returns the names of the parameters that were dropped or
/// renamed. Structured outputs are never dropped silently: an unsupported
/// model is an error (see [`check_model`]).
pub fn adapt_to_model(
    cfg: &mut OpenRouterConfig,
    model: &ModelInfo,
) -> Result<Vec<String>, OpenRouterError> {
    check_model(model, cfg)?;
    let mut changed = Vec::new();
    if model.supported_parameters.is_empty() {
        return Ok(changed); // catalog says nothing; send the defaults
    }
    if cfg.send_seed && !model.supports("seed") {
        cfg.send_seed = false;
        changed.push("seed".into());
    }
    if cfg.send_temperature && !model.supports("temperature") {
        cfg.send_temperature = false;
        changed.push("temperature".into());
    }
    if cfg.reasoning_effort.is_some() && !model.supports("reasoning") {
        cfg.reasoning_effort = None;
        changed.push("reasoning".into());
    }
    if !model.supports(&cfg.max_tokens_param) {
        if let Some(alt) = ["max_tokens", "max_completion_tokens"]
            .into_iter()
            .find(|p| model.supports(p))
        {
            changed.push(format!("{} -> {alt}", cfg.max_tokens_param));
            cfg.max_tokens_param = alt.into();
        } else {
            return Err(OpenRouterError::BadRequest(format!(
                "{} lists no output-token limit parameter; IRL needs one to cap cost",
                model.id
            )));
        }
    }
    Ok(changed)
}

/// The relevant fields of `GET /key`.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct KeyInfo {
    pub label: Option<String>,
    /// Spending limit in USD, if the key has one.
    pub limit: Option<f64>,
    pub limit_remaining: Option<f64>,
    pub usage: Option<f64>,
    pub is_free_tier: bool,
}

pub fn parse_key_info(body: &str) -> Result<KeyInfo, OpenRouterError> {
    let v: Value = serde_json::from_str(body)
        .map_err(|e| OpenRouterError::Protocol(format!("key info is not JSON: {e}")))?;
    let d = v
        .get("data")
        .ok_or_else(|| OpenRouterError::Protocol("key info has no data".into()))?;
    Ok(KeyInfo {
        label: d.get("label").and_then(Value::as_str).map(str::to_string),
        limit: d.get("limit").and_then(Value::as_f64),
        limit_remaining: d.get("limit_remaining").and_then(Value::as_f64),
        usage: d.get("usage").and_then(Value::as_f64),
        is_free_tier: d
            .get("is_free_tier")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rpg_domain::ports::InferenceRole;

    fn req() -> InferenceRequest {
        InferenceRequest {
            id: 7,
            role: InferenceRole::NpcDecision,
            system: "sys".into(),
            prompt: "ctx".into(),
            max_tokens: 160,
            temperature: 0.0,
            seed: 7,
            response_contract: Some(ACTION_CONTRACT_ID.into()),
        }
    }

    #[test]
    fn api_key_is_never_printed() {
        let k = ApiKey::new("  sk-or-v1-secret\n").unwrap();
        assert_eq!(k.expose(), "sk-or-v1-secret");
        assert_eq!(format!("{k:?}"), "ApiKey(<redacted>)");
        assert!(ApiKey::new("   ").is_err());
        assert!(ApiKey::new("sk or").is_err());
        let msg = redact("Bearer sk-or-v1-secret rejected", Some(&k));
        assert!(!msg.contains("secret"), "{msg}");
    }

    #[test]
    fn body_requests_the_shared_contract_strictly() {
        let cfg = OpenRouterConfig::default();
        let body = chat_body(&cfg, &req(), false);
        assert_eq!(body["model"], DEFAULT_MODEL);
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][1]["content"], "ctx");
        assert_eq!(body["response_format"]["type"], "json_schema");
        assert_eq!(body["response_format"]["json_schema"]["strict"], true);
        assert_eq!(
            body["response_format"]["json_schema"]["name"],
            ACTION_SCHEMA_NAME
        );
        assert_eq!(
            body["response_format"]["json_schema"]["schema"],
            strict_action_schema()
        );
        assert_eq!(body["provider"]["require_parameters"], true);
        assert_eq!(body["seed"], 7);
        assert_eq!(body["stream"], false);
        assert!(is_structured(&cfg, &req()));
    }

    #[test]
    fn compatibility_modes_never_claim_structure() {
        for mode in [StructuredMode::JsonObject, StructuredMode::PromptOnly] {
            let cfg = OpenRouterConfig {
                structured: mode,
                ..OpenRouterConfig::default()
            };
            let body = chat_body(&cfg, &req(), true);
            assert!(body.get("provider").is_none());
            assert!(!is_structured(&cfg, &req()));
        }
        let mut plain = req();
        plain.response_contract = None;
        let body = chat_body(&OpenRouterConfig::default(), &plain, false);
        assert!(body.get("response_format").is_none());
    }

    #[test]
    fn parses_completion_usage_and_provenance() {
        let body = r#"{"id":"gen-1","model":"openai/gpt-6-luna","provider":"OpenAI",
            "choices":[{"finish_reason":"stop","message":{"role":"assistant",
            "content":"{\"action\":\"idle\"}"}}],
            "usage":{"prompt_tokens":120,"completion_tokens":9,"cost":0.00002,
            "completion_tokens_details":{"reasoning_tokens":3}}}"#;
        let c = parse_completion(body).unwrap();
        assert_eq!(c.text, r#"{"action":"idle"}"#);
        let u = c.usage.clone().unwrap();
        assert_eq!((u.prompt_tokens, u.completion_tokens), (120, 9));
        assert_eq!(u.reasoning_tokens, Some(3));
        assert_eq!(u.cost_usd, Some(0.00002));
        let r = to_inference_response(
            c,
            &OpenRouterConfig::default(),
            &req(),
            850.0,
            Some(0.00001),
        );
        let p = r.remote.unwrap();
        assert_eq!(p.provider.as_deref(), Some("OpenAI"));
        assert_eq!(p.reported_cost_usd, Some(0.00002));
        assert_eq!(p.estimated_cost_usd, Some(0.00001));
        assert!(r.structured);
    }

    #[test]
    fn error_bodies_are_failures_even_with_200() {
        let e = parse_completion(r#"{"error":{"code":502,"message":"upstream died"}}"#);
        assert!(
            matches!(e, Err(OpenRouterError::Unavailable { .. })),
            "{e:?}"
        );
        let e =
            parse_completion(r#"{"choices":[{"finish_reason":"error","message":{"content":""}}]}"#);
        assert!(matches!(e, Err(OpenRouterError::Provider(_))));
        let e = parse_completion(r#"{"choices":[{"message":{"content":null,"refusal":"no"}}]}"#);
        assert!(matches!(e, Err(OpenRouterError::Provider(_))));
        assert!(matches!(
            parse_completion("<html>"),
            Err(OpenRouterError::Protocol(_))
        ));
    }

    #[test]
    fn http_errors_are_classified() {
        let err = |s, b: &str, ra| classify_http_error(s, b, ra);
        assert!(matches!(
            err(
                401,
                r#"{"error":{"code":401,"message":"No auth credentials found"}}"#,
                None
            ),
            OpenRouterError::Auth(_)
        ));
        assert!(matches!(
            err(
                402,
                r#"{"error":{"code":402,"message":"Insufficient credits"}}"#,
                None
            ),
            OpenRouterError::InsufficientCredits(_)
        ));
        let rl = err(
            429,
            r#"{"error":{"code":429,"message":"slow down"}}"#,
            Some("2"),
        );
        assert_eq!(rl.retry_delay_ms(), Some(2000));
        let long = err(429, "{}", Some("60"));
        assert_eq!(long.retry_delay_ms(), None, "long waits are not retried");
        assert!(matches!(
            err(
                404,
                r#"{"error":{"code":404,"message":"No endpoints found that support the requested parameters: response_format"}}"#,
                None
            ),
            OpenRouterError::StructuredUnsupported(_)
        ));
        assert!(err(503, "", None).retry_delay_ms().is_some());
        assert!(err(400, "bad", None).retry_delay_ms().is_none());
        assert!(matches!(
            err(500, "oops", None),
            OpenRouterError::Http { .. }
        ));
    }

    #[test]
    fn budget_blocks_requests_tokens_and_cost() {
        let mut m = UsageMeter::default();
        let b = Budget {
            max_requests: Some(2),
            ..Budget::default()
        };
        assert!(m.check(&b).is_ok());
        m.begin();
        m.begin();
        assert!(matches!(m.check(&b), Err(OpenRouterError::Budget(_))));

        let mut m = UsageMeter::default();
        let c = Completion {
            usage: Some(Usage {
                prompt_tokens: 10,
                completion_tokens: 50,
                cost_usd: Some(0.02),
                ..Usage::default()
            }),
            ..Completion::default()
        };
        m.begin();
        m.record_ok(&c, Some(0.01), 100.0);
        assert_eq!(m.reported_cost_usd, Some(0.02));
        assert_eq!(m.estimated_cost_usd, Some(0.01));
        assert!(m
            .check(&Budget {
                max_completion_tokens: Some(50),
                ..Budget::default()
            })
            .is_err());
        assert!(m
            .check(&Budget {
                max_cost_usd: Some(0.02),
                ..Budget::default()
            })
            .is_err());
        assert!(m
            .check(&Budget {
                max_cost_usd: Some(0.05),
                ..Budget::default()
            })
            .is_ok());
    }

    #[test]
    fn model_catalog_drives_compatibility_and_estimates() {
        let body = r#"{"data":[
            {"id":"a/structured","name":"A","context_length":8192,
             "pricing":{"prompt":"0.000001","completion":"0.000002"},
             "supported_parameters":["response_format","structured_outputs","seed"]},
            {"id":"b/plain","pricing":{"prompt":"0","completion":"0"},
             "supported_parameters":["temperature"]}]}"#;
        let models = parse_models(body).unwrap();
        assert_eq!(models.len(), 2);
        assert!(models[0].supports_structured());
        assert!(!models[1].supports_structured());
        let cost = models[0].estimate_cost(1000, 100).unwrap();
        assert!((cost - 0.0012).abs() < 1e-12);
        let cfg = OpenRouterConfig::default();
        assert!(check_model(&models[0], &cfg).is_ok());
        assert!(matches!(
            check_model(&models[1], &cfg),
            Err(OpenRouterError::StructuredUnsupported(_))
        ));
        let compat = OpenRouterConfig {
            structured: StructuredMode::JsonObject,
            ..cfg
        };
        assert!(check_model(&models[1], &compat).is_ok());
    }

    #[test]
    fn unsupported_optional_parameters_are_dropped_not_sent() {
        let model = ModelInfo {
            id: "r/reasoner".into(),
            supported_parameters: vec![
                "response_format".into(),
                "structured_outputs".into(),
                "max_completion_tokens".into(),
            ],
            ..ModelInfo::default()
        };
        let mut cfg = OpenRouterConfig {
            reasoning_effort: Some("low".into()),
            ..OpenRouterConfig::default()
        };
        let changed = adapt_to_model(&mut cfg, &model).unwrap();
        assert_eq!(changed.len(), 4, "{changed:?}");
        let body = chat_body(&cfg, &req(), false);
        for absent in ["seed", "temperature", "reasoning", "max_tokens"] {
            assert!(body.get(absent).is_none(), "{absent} still sent");
        }
        assert_eq!(body["max_completion_tokens"], 400);
        // A generic routing 404 is not misreported as missing structured outputs.
        assert!(matches!(
            classify_http_error(
                404,
                r#"{"error":{"code":404,"message":"No endpoints found that support all the requested parameters"}}"#,
                None
            ),
            OpenRouterError::BadRequest(_)
        ));
    }

    #[test]
    fn truncated_answers_are_reported_as_such() {
        let e = parse_completion(
            r#"{"choices":[{"finish_reason":"length","message":{"content":"{\"action\":\"sp"}}]}"#,
        )
        .unwrap_err();
        assert!(e.to_string().contains("truncated"), "{e}");
    }

    #[test]
    fn key_info_parses() {
        let k = parse_key_info(
            r#"{"data":{"label":"irl-test","limit":1.0,"limit_remaining":0.75,"usage":0.25,"is_free_tier":false}}"#,
        )
        .unwrap();
        assert_eq!(k.label.as_deref(), Some("irl-test"));
        assert_eq!(k.limit_remaining, Some(0.75));
    }
}
