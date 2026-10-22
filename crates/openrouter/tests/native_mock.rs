//! The native OpenRouter transport against a local mock server.
//!
//! Real sockets and real HTTP (via `ureq`), no network and no paid calls:
//! each test serves canned OpenRouter responses — JSON, SSE streams,
//! errors, slow and broken connections — and checks what the engine and the
//! shared NPC policy make of them.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use rpg_domain::ids::{ActorId, FactId, PLAYER};
use rpg_domain::ports::{InferenceEngine, InferenceError, InferenceRequest, InferenceRole};
use rpg_openrouter::native::OpenRouterEngine;
use rpg_openrouter::{ApiKey, Budget, OpenRouterConfig, StructuredMode};

const KEY: &str = "sk-or-v1-test-secret-0123456789";

/// One canned reply: status, extra headers, and body pieces sent with an
/// optional delay before each.
#[derive(Clone)]
struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    pieces: Vec<(Duration, Vec<u8>)>,
    /// Close without finishing a chunked body (simulated network drop).
    truncate: bool,
}

impl Reply {
    fn json(status: u16, body: &str) -> Self {
        Self {
            status,
            headers: vec![("Content-Type".into(), "application/json".into())],
            pieces: vec![(Duration::ZERO, body.as_bytes().to_vec())],
            truncate: false,
        }
    }

    fn sse(pieces: Vec<(Duration, &str)>) -> Self {
        Self {
            status: 200,
            headers: vec![("Content-Type".into(), "text/event-stream".into())],
            pieces: pieces
                .into_iter()
                .map(|(d, s)| (d, s.as_bytes().to_vec()))
                .collect(),
            truncate: false,
        }
    }

    fn header(mut self, k: &str, v: &str) -> Self {
        self.headers.push((k.into(), v.into()));
        self
    }
}

#[derive(Clone, Debug)]
struct Seen {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: String,
}

struct Mock {
    base: String,
    seen: Arc<Mutex<Vec<Seen>>>,
}

/// Serve `replies` for POST /chat/completions in order (the last one
/// repeats), plus fixed GET /key and GET /models answers.
fn mock(replies: Vec<Reply>, models: &str) -> Mock {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let models = models.to_string();
    {
        let seen = seen.clone();
        let counter = Arc::new(Mutex::new(0usize));
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { break };
                let (seen, models, replies, counter) = (
                    seen.clone(),
                    models.clone(),
                    replies.clone(),
                    counter.clone(),
                );
                // One thread per connection, so a slow or idle client can
                // never block the next one.
                thread::spawn(move || {
                    let Some(req) = read_request(&stream) else {
                        return;
                    };
                    seen.lock().unwrap().push(req.clone());
                    let reply = if req.method == "GET" && req.path.ends_with("/key") {
                        Reply::json(
                            200,
                            r#"{"data":{"label":"test","limit":1.0,"limit_remaining":1.0,"usage":0.0,"is_free_tier":false}}"#,
                        )
                    } else if req.method == "GET" && req.path.contains("/models") {
                        Reply::json(200, &models)
                    } else {
                        let mut n = counter.lock().unwrap();
                        let r = replies[(*n).min(replies.len() - 1)].clone();
                        *n += 1;
                        r
                    };
                    write_reply(stream, reply);
                });
            }
        });
    }
    Mock { base, seen }
}

fn read_request(stream: &TcpStream) -> Option<Seen> {
    let mut r = BufReader::new(stream.try_clone().ok()?);
    let mut line = String::new();
    r.read_line(&mut line).ok()?;
    let mut parts = line.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();
    let mut headers = Vec::new();
    let mut len = 0usize;
    loop {
        let mut h = String::new();
        r.read_line(&mut h).ok()?;
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            let (k, v) = (k.trim().to_string(), v.trim().to_string());
            if k.eq_ignore_ascii_case("content-length") {
                len = v.parse().unwrap_or(0);
            }
            headers.push((k, v));
        }
    }
    let mut body = vec![0u8; len];
    r.read_exact(&mut body).ok()?;
    Some(Seen {
        method,
        path,
        headers,
        body: String::from_utf8_lossy(&body).into_owned(),
    })
}

fn write_reply(mut s: TcpStream, reply: Reply) {
    let reason = match reply.status {
        200 => "OK",
        401 => "Unauthorized",
        429 => "Too Many Requests",
        _ => "Status",
    };
    let mut head = format!("HTTP/1.1 {} {reason}\r\n", reply.status);
    for (k, v) in &reply.headers {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n");
    if s.write_all(head.as_bytes()).is_err() {
        return;
    }
    for (delay, piece) in &reply.pieces {
        thread::sleep(*delay);
        let chunk = [format!("{:x}\r\n", piece.len()).as_bytes(), piece, b"\r\n"].concat();
        if s.write_all(&chunk).is_err() || s.flush().is_err() {
            return;
        }
    }
    if !reply.truncate {
        let _ = s.write_all(b"0\r\n\r\n");
    }
    let _ = s.flush();
}

const MODELS: &str = r#"{"data":[
  {"id":"test/structured","name":"Structured","pricing":{"prompt":"0.000001","completion":"0.000002"},
   "supported_parameters":["response_format","structured_outputs","seed","max_tokens","temperature"]},
  {"id":"test/plain","name":"Plain","pricing":{"prompt":"0","completion":"0"},
   "supported_parameters":["max_tokens","temperature"]}]}"#;

fn cfg(base: &str, stream: bool) -> OpenRouterConfig {
    OpenRouterConfig {
        api_base: base.into(),
        model: "test/structured".into(),
        stream,
        timeout_ms: 5_000,
        ..OpenRouterConfig::default()
    }
}

fn engine(c: OpenRouterConfig) -> OpenRouterEngine {
    OpenRouterEngine::new(c, ApiKey::new(KEY).unwrap())
}

fn request() -> InferenceRequest {
    InferenceRequest {
        id: 42,
        role: InferenceRole::NpcDecision,
        system: "You are a character.".into(),
        prompt: "The player says hello.".into(),
        max_tokens: 160,
        temperature: 0.0,
        seed: 42,
        response_contract: Some(rpg_domain::contract::ACTION_CONTRACT_ID.into()),
    }
}

fn completion_json(content: &str) -> String {
    serde_json::json!({
        "id": "gen-123",
        "model": "test/structured",
        "provider": "MockAI",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 12, "cost": 0.000124}
    })
    .to_string()
}

fn sse_chunk(content: &str) -> String {
    format!(
        "data: {}\n\n",
        serde_json::json!({"id":"gen-s","model":"test/structured",
            "choices":[{"index":0,"delta":{"content":content},"finish_reason":null}]})
    )
}

const SPEAK: &str = r#"{"action":"speak","target":"player","text":"Evening.","fact":null,"assertive":null,"give":null,"want":null,"note":null,"terms":null,"condition":null,"demand":null,"loudness":null}"#;

fn assert_no_key(text: &str) {
    assert!(!text.contains(KEY), "API key leaked into: {text}");
    assert!(!text.contains("test-secret"), "API key leaked into: {text}");
}

#[test]
fn non_streaming_request_carries_contract_auth_and_parses_usage() {
    let m = mock(vec![Reply::json(200, &completion_json(SPEAK))], MODELS);
    let mut e = engine(cfg(&m.base, false));
    e.warm().unwrap();
    let r = e.complete(&request()).unwrap();
    assert!(r.structured);
    assert_eq!(r.completion_tokens, 12);
    let p = r.remote.clone().unwrap();
    assert_eq!(p.gateway, "openrouter");
    assert_eq!(p.provider.as_deref(), Some("MockAI"));
    assert_eq!(p.reported_cost_usd, Some(0.000124));
    // Estimated from the catalog prices, and kept separate from the report.
    let est = p.estimated_cost_usd.unwrap();
    assert!((est - (100.0 * 0.000001 + 12.0 * 0.000002)).abs() < 1e-12);
    assert!(rpg_domain::contract::validate_action_json(&r.text).is_ok());

    let seen = m.seen.lock().unwrap();
    let post = seen.iter().find(|s| s.method == "POST").unwrap();
    assert_eq!(post.path, "/chat/completions");
    let auth = post
        .headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("authorization"))
        .map(|(_, v)| v.clone())
        .unwrap();
    assert_eq!(auth, format!("Bearer {KEY}"));
    let body: serde_json::Value = serde_json::from_str(&post.body).unwrap();
    assert_eq!(body["response_format"]["json_schema"]["strict"], true);
    assert_eq!(body["provider"]["require_parameters"], true);
    assert_eq!(body["seed"], 42);
    assert_eq!(body["stream"], false);
    // The catalog was consulted on warm.
    assert!(seen.iter().any(|s| s.path.ends_with("/key")));
    assert!(e.meter().succeeded == 1 && e.meter().requests == 1);
}

#[test]
fn streaming_handles_keepalives_split_chunks_usage_and_done() {
    let first = sse_chunk(r#"{"action":"speak","target":"player","#);
    let (a, b) = first.split_at(17); // split inside the JSON payload
    let usage = "data: {\"choices\":[{\"delta\":{\"content\":\"\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":90,\"completion_tokens\":20,\"cost\":0.0002}}\r\n\r\n";
    let rest = sse_chunk(r#""text":"Ca"#);
    let rest2 = sse_chunk(r#"fé?"}"#);
    let m = mock(
        vec![Reply::sse(vec![
            (Duration::ZERO, ": OPENROUTER PROCESSING\n\n"),
            (Duration::from_millis(30), a),
            (Duration::from_millis(30), b),
            (Duration::ZERO, &rest),
            (Duration::ZERO, &rest2),
            (Duration::ZERO, usage),
            (Duration::ZERO, "data: [DONE]\n\n"),
        ])],
        MODELS,
    );
    let mut e = engine(cfg(&m.base, true)).without_verification();
    let mut progress = 0;
    let r = e
        .complete_streaming(&request(), &mut |_| progress += 1)
        .unwrap();
    assert_eq!(
        r.text,
        r#"{"action":"speak","target":"player","text":"Café?"}"#
    );
    assert!(progress >= 2, "progress callbacks: {progress}");
    assert_eq!(r.prompt_tokens, 90);
    let p = r.remote.unwrap();
    assert!(p.streamed);
    assert_eq!(p.reported_cost_usd, Some(0.0002));
    assert!(r.first_token_ms > 0.0);
}

#[test]
fn auth_failure_is_reported_without_the_key() {
    let m = mock(
        vec![Reply::json(
            401,
            &format!(r#"{{"error":{{"code":401,"message":"User not found for key {KEY}"}}}}"#),
        )],
        MODELS,
    );
    let mut e = engine(cfg(&m.base, false)).without_verification();
    let err = e.complete(&request()).unwrap_err();
    let text = format!("{err} {err:?}");
    assert!(text.contains("authentication"), "{text}");
    assert_no_key(&text);
    assert_no_key(&format!("{:?}", e.meter()));
}

#[test]
fn short_rate_limit_is_retried_once_then_succeeds() {
    let m = mock(
        vec![
            Reply::json(
                429,
                r#"{"error":{"code":429,"message":"Rate limit exceeded"}}"#,
            )
            .header("Retry-After", "0.1"),
            Reply::json(200, &completion_json(SPEAK)),
        ],
        MODELS,
    );
    let mut e = engine(cfg(&m.base, false)).without_verification();
    assert!(e.complete(&request()).is_ok());
    assert_eq!(e.meter().requests, 2, "exactly one retry");
}

#[test]
fn persistent_rate_limit_and_long_retry_after_fail_without_looping() {
    let m = mock(
        vec![
            Reply::json(429, r#"{"error":{"code":429,"message":"slow down"}}"#)
                .header("Retry-After", "0"),
        ],
        MODELS,
    );
    let mut e = engine(cfg(&m.base, false)).without_verification();
    let err = e.complete(&request()).unwrap_err();
    assert!(err.to_string().contains("rate limited"), "{err}");
    assert_eq!(e.meter().requests, 2, "one retry, then give up");

    let m = mock(
        vec![
            Reply::json(429, r#"{"error":{"code":429,"message":"slow"}}"#)
                .header("Retry-After", "120"),
        ],
        MODELS,
    );
    let mut e = engine(cfg(&m.base, false)).without_verification();
    let t = Instant::now();
    assert!(e.complete(&request()).is_err());
    assert_eq!(
        e.meter().requests,
        1,
        "a long Retry-After is not waited out"
    );
    assert!(t.elapsed() < Duration::from_secs(2));
}

#[test]
fn mid_stream_error_and_dropped_connection_are_failures() {
    let partial = sse_chunk(r#"{"action":"#);
    let m = mock(
        vec![Reply::sse(vec![
            (Duration::ZERO, &partial),
            (
                Duration::ZERO,
                "data: {\"id\":\"x\",\"error\":{\"code\":502,\"message\":\"Provider disconnected\"},\"choices\":[{\"delta\":{\"content\":\"\"},\"finish_reason\":\"error\"}]}\n\n",
            ),
        ])],
        MODELS,
    );
    let mut e = engine(OpenRouterConfig {
        max_retries: 0,
        ..cfg(&m.base, true)
    })
    .without_verification();
    let err = e.complete(&request()).unwrap_err();
    assert!(err.to_string().contains("Provider disconnected"), "{err}");

    let mut broken = Reply::sse(vec![(Duration::ZERO, &partial)]);
    broken.truncate = true;
    let m = mock(vec![broken], MODELS);
    let mut e = engine(cfg(&m.base, true)).without_verification();
    let err = e.complete(&request()).unwrap_err();
    assert!(
        matches!(err, InferenceError::Backend(ref s) if s.contains("network") || s.contains("stream")),
        "{err:?}"
    );
}

#[test]
fn interrupt_cancels_promptly_while_the_provider_is_slow() {
    let m = mock(
        vec![Reply::sse(vec![
            (Duration::ZERO, ": OPENROUTER PROCESSING\n\n"),
            (Duration::from_secs(10), "data: [DONE]\n\n"),
        ])],
        MODELS,
    );
    let mut e = engine(OpenRouterConfig {
        timeout_ms: 30_000,
        ..cfg(&m.base, true)
    })
    .without_verification();
    let flag = e.interrupt_handle();
    let t = Instant::now();
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(200));
        flag.store(true, Ordering::SeqCst);
    });
    let err = e.complete(&request()).unwrap_err();
    assert!(matches!(err, InferenceError::Cancelled), "{err:?}");
    assert!(
        t.elapsed() < Duration::from_secs(2),
        "cancel took {:?}",
        t.elapsed()
    );
}

#[test]
fn deadline_bounds_a_stalled_provider() {
    let m = mock(
        vec![Reply::sse(vec![(
            Duration::from_secs(10),
            "data: [DONE]\n\n",
        )])],
        MODELS,
    );
    let mut e = engine(OpenRouterConfig {
        timeout_ms: 300,
        ..cfg(&m.base, true)
    })
    .without_verification();
    let t = Instant::now();
    let err = e.complete(&request()).unwrap_err();
    assert!(matches!(err, InferenceError::Timeout(300)), "{err:?}");
    assert!(t.elapsed() < Duration::from_secs(2));
}

#[test]
fn budget_refuses_requests_without_touching_the_network() {
    let m = mock(vec![Reply::json(200, &completion_json(SPEAK))], MODELS);
    let mut e = engine(OpenRouterConfig {
        budget: Budget {
            max_requests: Some(1),
            ..Budget::default()
        },
        ..cfg(&m.base, false)
    })
    .without_verification();
    assert!(e.complete(&request()).is_ok());
    let err = e.complete(&request()).unwrap_err();
    assert!(err.to_string().contains("budget"), "{err}");
    let posts = m
        .seen
        .lock()
        .unwrap()
        .iter()
        .filter(|s| s.method == "POST")
        .count();
    assert_eq!(posts, 1);
}

#[test]
fn warm_rejects_models_without_structured_outputs_unless_compat_is_explicit() {
    let m = mock(vec![Reply::json(200, &completion_json(SPEAK))], MODELS);
    let mut e = engine(OpenRouterConfig {
        model: "test/plain".into(),
        ..cfg(&m.base, false)
    });
    let err = e.warm().unwrap_err();
    assert!(
        err.to_string().contains("structured output unsupported"),
        "{err}"
    );

    let mut compat = engine(OpenRouterConfig {
        model: "test/plain".into(),
        structured: StructuredMode::JsonObject,
        ..cfg(&m.base, false)
    });
    compat.warm().unwrap();
    let r = compat.complete(&request()).unwrap();
    assert!(
        !r.structured,
        "compatibility mode must not claim schema enforcement"
    );
    // The unsupported seed parameter was dropped so routing still works.
    let seen = m.seen.lock().unwrap();
    let body: serde_json::Value =
        serde_json::from_str(&seen.iter().rev().find(|s| s.method == "POST").unwrap().body)
            .unwrap();
    assert!(body.get("seed").is_none());
    assert_eq!(body["response_format"]["type"], "json_object");
    drop(seen);

    let mut unknown = engine(OpenRouterConfig {
        model: "nobody/nothing".into(),
        ..cfg(&m.base, false)
    });
    let err = unknown.warm().unwrap_err().to_string();
    assert!(err.contains("catalog"), "{err}");
}

#[test]
fn ok_status_with_error_body_is_a_failure() {
    let m = mock(
        vec![Reply::json(
            200,
            r#"{"error":{"code":502,"message":"upstream returned garbage"}}"#,
        )],
        MODELS,
    );
    let mut e = engine(OpenRouterConfig {
        max_retries: 0,
        ..cfg(&m.base, false)
    })
    .without_verification();
    let err = e.complete(&request()).unwrap_err();
    assert!(
        err.to_string().contains("upstream returned garbage"),
        "{err}"
    );
}

// ---------------------------------------------------------------------------
// Through the shared gameplay path
// ---------------------------------------------------------------------------

/// A remote answer goes through exactly the canonical path a local answer
/// does: contract validation, canonical agenda/obligations, simulation
/// validation, commit — with provenance recorded.
#[test]
fn remote_decisions_use_the_canonical_commit_path() {
    // The model just chats; the grocer's designed agenda still turns it into
    // a conditional bargain, and that is recorded as canonical, not model.
    let m = mock(vec![Reply::json(200, &completion_json(SPEAK))], MODELS);
    let e = engine(cfg(&m.base, false)).without_verification();
    let mut s = rpg_eval::Session::with_engine(2026, Box::new(e));
    s.teleport_player_to("grocer");
    s.say("grocer", "I need to see your ledger.");
    let offer = s
        .sim
        .world
        .offers
        .values()
        .find(|o| o.from.as_str() == "grocer")
        .cloned()
        .expect("canonical agenda creates the offer");
    assert_eq!(
        offer.disclose.as_ref().map(|f| f.as_str()),
        Some("f_grocer_ledger")
    );
    assert_eq!(s.canonical_decisions(), 1);
    assert_eq!(s.model_decisions(), 0);

    // An ordinary line from the model is committed as the model's own.
    s.say("grocer", "Lovely weather.");
    assert_eq!(s.model_decisions(), 1);
    assert!(s
        .sim
        .events
        .all()
        .iter()
        .any(|e| matches!(&e.kind, rpg_domain::events::EventKind::SpeechEmitted { speaker, text, .. } if speaker.as_str() == "grocer" && text == "Evening.")));
}

/// A remote answer that breaks the contract (an invented field) or proposes
/// a disclosure the NPC cannot make is never committed as such.
#[test]
fn invalid_remote_proposals_never_mutate_the_world() {
    let bad_field = r#"{"action":"speak","text":"hi","teleport":"vault"}"#;
    let m = mock(vec![Reply::json(200, &completion_json(bad_field))], MODELS);
    let e = engine(cfg(&m.base, false)).without_verification();
    let mut s = rpg_eval::Session::with_engine(2026, Box::new(e));
    s.teleport_player_to("priest");
    s.say("priest", "Good evening, father.");
    assert_eq!(s.fallback_decisions(), 1, "contract violation falls back");
    let (_, src) = s.decisions.last().unwrap();
    assert!(
        src.label().contains("violated the action contract"),
        "{}",
        src.label()
    );

    // A schema-valid but unknowable disclosure is rejected by the simulation.
    let leak = r#"{"action":"share_claim","target":"player","fact":"f_leader_identity"}"#;
    let m = mock(vec![Reply::json(200, &completion_json(leak))], MODELS);
    let e = engine(cfg(&m.base, false)).without_verification();
    let mut s = rpg_eval::Session::with_engine(2026, Box::new(e));
    s.teleport_player_to("priest");
    let before = s.sim.events.checksum();
    s.say("priest", "Who leads them?");
    assert!(!s
        .sim
        .world
        .believes(&ActorId::new(PLAYER), &FactId::new("f_leader_identity")));
    assert!(
        s.trace.iter().any(|t| t.contains("Rejected")),
        "{:?}",
        s.trace
    );
    assert_ne!(
        before,
        s.sim.events.checksum(),
        "the rejection itself is logged"
    );
}
