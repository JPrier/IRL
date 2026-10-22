//! Native OpenRouter transport.
//!
//! [`OpenRouterEngine`] is an ordinary resident [`InferenceEngine`]: the
//! native runtime runs it on its `irl-inference` worker thread, exactly like
//! the local llama.cpp engine, so a slow or stalled network can never delay a
//! simulation tick.
//!
//! Each HTTP request runs on a short-lived helper thread while the worker
//! waits on a channel, polling the runtime's interrupt flag and the request
//! deadline every few milliseconds. Cancelling therefore returns
//! immediately; the helper notices its own cancel flag between stream reads
//! and drops the connection, which stops generation (and billing) on
//! providers that support stream cancellation. Late output from a cancelled
//! helper goes nowhere: its channel has been dropped.

use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use rpg_domain::ports::{InferenceEngine, InferenceError, InferenceRequest, InferenceResponse};

use crate::sse::{SseParser, StreamAccumulator, StreamStep};
use crate::{
    chat_body, check_model, classify_http_error, parse_completion, parse_key_info, parse_models,
    to_inference_response, ApiKey, Completion, KeyInfo, ModelInfo, OpenRouterConfig,
    OpenRouterError, UsageMeter, API_KEY_ENV,
};

/// How often the waiting worker checks for cancellation and its deadline.
const POLL: Duration = Duration::from_millis(10);

enum Msg {
    Delta(String),
    Done(Result<Completion, OpenRouterError>),
}

/// OpenRouter as a resident inference engine for the native runtime.
pub struct OpenRouterEngine {
    cfg: OpenRouterConfig,
    key: ApiKey,
    agent: ureq::Agent,
    interrupt: Arc<AtomicBool>,
    meter: UsageMeter,
    model: Option<ModelInfo>,
    key_info: Option<KeyInfo>,
    /// Check the key and the model's structured-output support on `warm`.
    verify_on_warm: bool,
    warmed: bool,
}

impl OpenRouterEngine {
    pub fn new(cfg: OpenRouterConfig, key: ApiKey) -> Self {
        let mut builder = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(15))
            // Bounds a helper thread that is still reading after the worker
            // gave up on it; the request deadline itself is enforced by the
            // waiting worker.
            .timeout(Duration::from_millis(cfg.timeout_ms.saturating_add(5_000)))
            .user_agent(concat!(
                "IRL/",
                env!("CARGO_PKG_VERSION"),
                " rpg_openrouter"
            ));
        if let Some(proxy) = proxy_for(&cfg.api_base) {
            builder = builder.proxy(proxy);
        }
        let agent = builder.build();
        Self {
            cfg,
            key,
            agent,
            interrupt: Arc::new(AtomicBool::new(false)),
            meter: UsageMeter::default(),
            model: None,
            key_info: None,
            verify_on_warm: true,
            warmed: false,
        }
    }

    /// Read the key from `OPENROUTER_API_KEY`. The key is never accepted on
    /// the command line, where shell history and process listings keep it.
    pub fn from_env(cfg: OpenRouterConfig) -> Result<Self, OpenRouterError> {
        let raw = std::env::var(API_KEY_ENV).map_err(|_| {
            OpenRouterError::Auth(format!(
                "{API_KEY_ENV} is not set (export it, or use --openrouter-key-file)"
            ))
        })?;
        Ok(Self::new(cfg, ApiKey::new(raw)?))
    }

    /// Read the key from a file (first line).
    pub fn from_key_file(
        cfg: OpenRouterConfig,
        path: &std::path::Path,
    ) -> Result<Self, OpenRouterError> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            OpenRouterError::Auth(format!("cannot read key file {}: {e}", path.display()))
        })?;
        let first = text.lines().next().unwrap_or_default();
        Ok(Self::new(cfg, ApiKey::new(first)?))
    }

    /// Skip the network checks in `warm` (tests, or a catalog that is known
    /// to be unreachable).
    pub fn without_verification(mut self) -> Self {
        self.verify_on_warm = false;
        self
    }

    /// The flag the runtime sets to abort the in-flight request.
    pub fn interrupt_handle(&self) -> Arc<AtomicBool> {
        self.interrupt.clone()
    }

    pub fn config(&self) -> &OpenRouterConfig {
        &self.cfg
    }

    pub fn meter(&self) -> &UsageMeter {
        &self.meter
    }

    pub fn model_info(&self) -> Option<&ModelInfo> {
        self.model.as_ref()
    }

    pub fn key_info(&self) -> Option<&KeyInfo> {
        self.key_info.as_ref()
    }

    /// Validate the key (`GET /key`, free) and check that the configured
    /// model exists and supports the requested structured-output mode.
    pub fn verify(&mut self) -> Result<(), OpenRouterError> {
        let key_body = self.get(&self.cfg.key_url(), true)?;
        self.key_info = Some(parse_key_info(&key_body)?);
        let models = parse_models(&self.get(&self.cfg.models_url(false), false)?)?;
        let model = models
            .into_iter()
            .find(|m| m.id == self.cfg.model)
            .ok_or_else(|| {
                OpenRouterError::BadRequest(format!(
                    "model {:?} is not in the OpenRouter catalog",
                    self.cfg.model
                ))
            })?;
        check_model(&model, &self.cfg)?;
        if self.cfg.send_seed && !model.supports("seed") {
            // `require_parameters` would otherwise route to no endpoint.
            self.cfg.send_seed = false;
        }
        self.model = Some(model);
        Ok(())
    }

    fn get(&self, url: &str, auth: bool) -> Result<String, OpenRouterError> {
        let mut r = self.agent.get(url);
        if auth {
            r = r.set("Authorization", &self.key.bearer());
        }
        match r.call() {
            Ok(resp) => resp
                .into_string()
                .map_err(|e| OpenRouterError::Network(e.to_string())),
            Err(ureq::Error::Status(code, resp)) => {
                let ra = resp.header("Retry-After").map(str::to_string);
                let body = resp.into_string().unwrap_or_default();
                Err(classify_http_error(code, &body, ra.as_deref()))
            }
            Err(ureq::Error::Transport(t)) => Err(OpenRouterError::Network(t.to_string())),
        }
        .map_err(|e| e.redact_with(&self.key))
    }

    /// One HTTP attempt on a helper thread; returns as soon as it finishes,
    /// the deadline passes or the runtime interrupts.
    fn attempt(
        &self,
        req: &InferenceRequest,
        deadline: Instant,
        start: Instant,
        sink: &mut dyn FnMut(&str),
    ) -> Result<Completion, OpenRouterError> {
        let stream = self.cfg.stream;
        let body = chat_body(&self.cfg, req, stream).to_string();
        let headers = self.cfg.headers(&self.key, false);
        let url = self.cfg.chat_url();
        let agent = self.agent.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel::<Msg>();
        {
            let cancel = cancel.clone();
            thread::Builder::new()
                .name("irl-openrouter-http".into())
                .spawn(move || {
                    let r = run_http(&agent, &url, &headers, &body, stream, &cancel, &tx, start);
                    let _ = tx.send(Msg::Done(r));
                })
                .map_err(|e| OpenRouterError::Network(format!("cannot start request: {e}")))?;
        }
        loop {
            if self.interrupt.load(Ordering::SeqCst) {
                cancel.store(true, Ordering::SeqCst);
                return Err(OpenRouterError::Cancelled);
            }
            let now = Instant::now();
            if now >= deadline {
                cancel.store(true, Ordering::SeqCst);
                return Err(OpenRouterError::Timeout(self.cfg.timeout_ms));
            }
            match rx.recv_timeout(POLL.min(deadline - now)) {
                Ok(Msg::Delta(piece)) => sink(&piece),
                Ok(Msg::Done(r)) => return r,
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(OpenRouterError::Network("request thread exited".into()))
                }
            }
        }
    }

    /// Sleep before a retry, still honouring cancellation and the deadline.
    fn backoff(&self, ms: u64, deadline: Instant) -> Result<(), OpenRouterError> {
        let until = Instant::now() + Duration::from_millis(ms);
        if until >= deadline {
            return Err(OpenRouterError::Timeout(self.cfg.timeout_ms));
        }
        while Instant::now() < until {
            if self.interrupt.load(Ordering::SeqCst) {
                return Err(OpenRouterError::Cancelled);
            }
            thread::sleep(POLL);
        }
        Ok(())
    }
}

/// The proxy to use for `base`, honouring `HTTPS_PROXY`/`HTTP_PROXY`/
/// `ALL_PROXY` and `NO_PROXY` (which `ureq`'s own environment lookup does
/// not). Loopback addresses never use a proxy.
fn proxy_for(base: &str) -> Option<ureq::Proxy> {
    let (scheme, rest) = base.split_once("://")?;
    let host = rest
        .split(['/', '?'])
        .next()
        .unwrap_or_default()
        .rsplit_once(':')
        .map(|(h, _)| h)
        .unwrap_or(rest.split(['/', '?']).next().unwrap_or_default())
        .trim_matches(['[', ']'])
        .to_ascii_lowercase();
    if host == "localhost" || host.starts_with("127.") || host == "::1" {
        return None;
    }
    let env = |names: &[&str]| {
        names
            .iter()
            .find_map(|n| std::env::var(n).ok().filter(|v| !v.trim().is_empty()))
    };
    if let Some(no) = env(&["NO_PROXY", "no_proxy"]) {
        let bypass = no.split(',').map(str::trim).any(|p| {
            let p = p.trim_start_matches('.').to_ascii_lowercase();
            p == "*" || (!p.is_empty() && (host == p || host.ends_with(&format!(".{p}"))))
        });
        if bypass {
            return None;
        }
    }
    let url = if scheme.eq_ignore_ascii_case("https") {
        env(&["HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"])
    } else {
        env(&["HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"])
    }?;
    ureq::Proxy::new(url).ok()
}

#[allow(clippy::too_many_arguments)]
fn run_http(
    agent: &ureq::Agent,
    url: &str,
    headers: &[(&'static str, String)],
    body: &str,
    stream: bool,
    cancel: &AtomicBool,
    tx: &Sender<Msg>,
    start: Instant,
) -> Result<Completion, OpenRouterError> {
    let mut r = agent.post(url);
    for (k, v) in headers {
        r = r.set(k, v);
    }
    let resp = match r.send_string(body) {
        Ok(resp) => resp,
        Err(ureq::Error::Status(code, resp)) => {
            let ra = resp.header("Retry-After").map(str::to_string);
            let text = resp.into_string().unwrap_or_default();
            return Err(classify_http_error(code, &text, ra.as_deref()));
        }
        Err(ureq::Error::Transport(t)) => return Err(OpenRouterError::Network(t.to_string())),
    };
    let is_sse = resp.content_type().contains("event-stream");
    if !stream || !is_sse {
        // Non-streaming request, or an error/JSON body on a stream request.
        let text = resp
            .into_string()
            .map_err(|e| OpenRouterError::Network(e.to_string()))?;
        return parse_completion(&text);
    }
    let mut reader = resp.into_reader();
    let mut parser = SseParser::new();
    let mut acc = StreamAccumulator::new();
    let mut buf = [0u8; 8192];
    let ms = || start.elapsed().as_secs_f32() * 1000.0;
    loop {
        if cancel.load(Ordering::SeqCst) {
            // Dropping the reader closes the connection: providers that
            // support it stop generating (and billing) now.
            return Err(OpenRouterError::Cancelled);
        }
        let n = reader
            .read(&mut buf)
            .map_err(|e| OpenRouterError::Network(format!("stream read failed: {e}")))?;
        let events = if n == 0 {
            parser.finish()
        } else {
            parser.push(&buf[..n])
        };
        let before = acc.content_len();
        for ev in &events {
            if acc.apply(ev, ms())? == StreamStep::Done {
                return acc.finish();
            }
        }
        if acc.content_len() > before {
            // Progress only; the runtime never shows or acts on partial JSON.
            let _ = tx.send(Msg::Delta(String::new()));
        }
        if n == 0 {
            return acc.finish();
        }
    }
}

impl InferenceEngine for OpenRouterEngine {
    fn engine_name(&self) -> String {
        "openrouter".into()
    }

    fn model_id(&self) -> String {
        self.cfg.model.clone()
    }

    fn warm(&mut self) -> Result<(), InferenceError> {
        if self.warmed {
            return Ok(());
        }
        if self.verify_on_warm {
            self.verify()
                .map_err(|e| e.redact_with(&self.key).into_inference())?;
        }
        self.warmed = true;
        Ok(())
    }

    fn complete(&mut self, req: &InferenceRequest) -> Result<InferenceResponse, InferenceError> {
        let mut sink = |_: &str| {};
        self.complete_streaming(req, &mut sink)
    }

    fn complete_streaming(
        &mut self,
        req: &InferenceRequest,
        sink: &mut dyn FnMut(&str),
    ) -> Result<InferenceResponse, InferenceError> {
        if let Err(e) = self.meter.check(&self.cfg.budget) {
            self.meter.last_error = Some(e.to_string());
            return Err(e.into_inference());
        }
        let start = Instant::now();
        let deadline = start + Duration::from_millis(self.cfg.timeout_ms);
        let mut retries = 0;
        loop {
            self.meter.begin();
            let result = self.attempt(req, deadline, start, sink);
            let latency = start.elapsed().as_secs_f32() * 1000.0;
            match result {
                Ok(c) => {
                    let est = c.usage.as_ref().and_then(|u| {
                        self.model
                            .as_ref()
                            .and_then(|m| m.estimate_cost(u.prompt_tokens, u.completion_tokens))
                    });
                    self.meter.record_ok(&c, est, latency);
                    return Ok(to_inference_response(c, &self.cfg, req, latency, est));
                }
                Err(e) => {
                    let e = e.redact_with(&self.key);
                    let retry = (retries < self.cfg.max_retries)
                        .then(|| e.retry_delay_ms())
                        .flatten()
                        .filter(|_| self.meter.check(&self.cfg.budget).is_ok());
                    if let Some(ms) = retry {
                        retries += 1;
                        if let Err(stop) = self.backoff(ms, deadline) {
                            self.meter.record_err(&stop, latency);
                            return Err(stop.into_inference());
                        }
                        continue;
                    }
                    self.meter.record_err(&e, latency);
                    return Err(e.into_inference());
                }
            }
        }
    }

    fn cancel(&mut self, _id: u64) -> bool {
        self.interrupt.store(true, Ordering::SeqCst);
        true
    }
}
