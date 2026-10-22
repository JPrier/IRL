//! Server-sent events, as OpenRouter streams chat completions.
//!
//! [`SseParser`] turns arbitrary byte chunks into events and is correct for:
//! chunk boundaries anywhere (including inside a multi-byte UTF-8 sequence or
//! a `\r\n` pair), `\n`, `\r\n` and `\r` line endings, multi-line `data:`
//! fields (joined with `\n`), comment/keep-alive lines (`: OPENROUTER
//! PROCESSING`), and a final event without a trailing blank line.
//!
//! [`StreamAccumulator`] applies events to one completion: content deltas,
//! the usage/accounting chunk, `[DONE]`, and mid-stream errors (a top-level
//! `error` object and/or `finish_reason: "error"` on an HTTP 200 stream).
//! Accumulated text is *never* handed to gameplay before the stream finishes
//! and the whole answer is validated; progress callers only see a length.

use crate::{error_from_object, Completion, OpenRouterError, Usage};
use serde_json::Value;

/// One dispatched SSE event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SseEvent {
    /// A `data` payload (multi-line data joined with `\n`).
    Data(String),
    /// A comment line, e.g. OpenRouter's keep-alive. Carries the text after
    /// the colon.
    Comment(String),
}

/// Incremental SSE decoder.
#[derive(Debug, Default)]
pub struct SseParser {
    /// Bytes of the current, not yet terminated line.
    line: Vec<u8>,
    /// `data` lines of the event being built.
    data: Vec<String>,
    /// The previous chunk ended in `\r`; a leading `\n` belongs to it.
    pending_cr: bool,
}

impl SseParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed bytes; returns the events completed by them.
    pub fn push(&mut self, bytes: &[u8]) -> Vec<SseEvent> {
        let mut out = Vec::new();
        for &b in bytes {
            if self.pending_cr {
                self.pending_cr = false;
                if b == b'\n' {
                    continue;
                }
            }
            match b {
                b'\n' => self.end_line(&mut out),
                b'\r' => {
                    self.end_line(&mut out);
                    self.pending_cr = true;
                }
                _ => self.line.push(b),
            }
        }
        out
    }

    /// The stream ended: flush a final event that had no blank line.
    pub fn finish(&mut self) -> Vec<SseEvent> {
        let mut out = Vec::new();
        if !self.line.is_empty() {
            self.end_line(&mut out);
        }
        self.dispatch(&mut out);
        out
    }

    fn end_line(&mut self, out: &mut Vec<SseEvent>) {
        let line = String::from_utf8_lossy(&std::mem::take(&mut self.line)).into_owned();
        if line.is_empty() {
            self.dispatch(out);
            return;
        }
        if let Some(comment) = line.strip_prefix(':') {
            out.push(SseEvent::Comment(comment.trim().to_string()));
            return;
        }
        let (field, value) = match line.split_once(':') {
            Some((f, v)) => (f, v.strip_prefix(' ').unwrap_or(v)),
            None => (line.as_str(), ""),
        };
        // `event`, `id` and `retry` carry nothing OpenRouter's chat stream
        // needs; unknown fields are ignored per the SSE spec.
        if field == "data" {
            self.data.push(value.to_string());
        }
    }

    fn dispatch(&mut self, out: &mut Vec<SseEvent>) {
        if !self.data.is_empty() {
            out.push(SseEvent::Data(std::mem::take(&mut self.data).join("\n")));
        }
    }
}

/// What applying one event did.
#[derive(Clone, Debug, PartialEq)]
pub enum StreamStep {
    /// Nothing of note (comment, accounting chunk, empty delta).
    Continue,
    /// New content arrived; carries the total content length so far.
    Progress(usize),
    /// `[DONE]`: the stream is complete.
    Done,
}

/// Builds one [`Completion`] from stream events.
#[derive(Debug, Default)]
pub struct StreamAccumulator {
    text: String,
    usage: Option<Usage>,
    model: Option<String>,
    provider: Option<String>,
    id: Option<String>,
    finish_reason: Option<String>,
    done: bool,
    keepalives: u32,
    first_token_ms: Option<f32>,
}

impl StreamAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Keep-alive comments seen (drives a "connected, waiting" indicator).
    pub fn keepalives(&self) -> u32 {
        self.keepalives
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    pub fn content_len(&self) -> usize {
        self.text.len()
    }

    /// Apply an event. `now_ms` (time since the request started) stamps the
    /// first content delta.
    pub fn apply(&mut self, ev: &SseEvent, now_ms: f32) -> Result<StreamStep, OpenRouterError> {
        let data = match ev {
            SseEvent::Comment(_) => {
                self.keepalives += 1;
                return Ok(StreamStep::Continue);
            }
            SseEvent::Data(d) => d.trim(),
        };
        if data == "[DONE]" {
            self.done = true;
            return Ok(StreamStep::Done);
        }
        if self.done {
            // Nothing is valid after [DONE]; ignore rather than corrupt.
            return Ok(StreamStep::Continue);
        }
        let v: Value = serde_json::from_str(data)
            .map_err(|e| OpenRouterError::Protocol(format!("stream chunk is not JSON: {e}")))?;
        if let Some(err) = v.get("error") {
            return Err(error_from_object(err));
        }
        if self.id.is_none() {
            self.id = v.get("id").and_then(Value::as_str).map(str::to_string);
        }
        if let Some(m) = v.get("model").and_then(Value::as_str) {
            self.model = Some(m.to_string());
        }
        if let Some(p) = v.get("provider").and_then(Value::as_str) {
            self.provider = Some(p.to_string());
        }
        if let Some(u) = v.get("usage").and_then(Usage::from_json) {
            // The accounting chunk repeats the finish_reason; it is data, not
            // a second terminal event.
            self.usage = Some(u);
        }
        let mut step = StreamStep::Continue;
        if let Some(choice) = v.get("choices").and_then(|c| c.get(0)) {
            if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                if reason == "error" {
                    return Err(OpenRouterError::Provider(
                        "stream ended with finish_reason \"error\"".into(),
                    ));
                }
                self.finish_reason = Some(reason.to_string());
            }
            if let Some(delta) = choice
                .get("delta")
                .and_then(|d| d.get("content"))
                .and_then(Value::as_str)
            {
                if !delta.is_empty() {
                    if self.first_token_ms.is_none() {
                        self.first_token_ms = Some(now_ms);
                    }
                    self.text.push_str(delta);
                    step = StreamStep::Progress(self.text.len());
                }
            }
        }
        Ok(step)
    }

    /// The finished completion. A stream that stopped without `[DONE]` *and*
    /// without a finish reason was interrupted; one with no content failed.
    pub fn finish(self) -> Result<Completion, OpenRouterError> {
        if !self.done && self.finish_reason.is_none() {
            return Err(OpenRouterError::Network(
                "stream ended before completion (connection interrupted)".into(),
            ));
        }
        if self.finish_reason.as_deref() == Some("length") {
            return Err(crate::truncated());
        }
        if self.text.trim().is_empty() {
            return Err(OpenRouterError::Protocol(format!(
                "stream finished without content (finish_reason {})",
                self.finish_reason.as_deref().unwrap_or("none")
            )));
        }
        Ok(Completion {
            text: self.text,
            usage: self.usage,
            model: self.model,
            provider: self.provider,
            id: self.id,
            finish_reason: self.finish_reason,
            streamed: true,
            first_token_ms: self.first_token_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(s: &str) -> SseEvent {
        SseEvent::Data(s.into())
    }

    #[test]
    fn handles_every_line_ending_and_any_chunking() {
        let stream = b": OPENROUTER PROCESSING\r\n\r\ndata: {\"a\":1}\n\ndata: x\rdata: y\r\rdata: [DONE]\r\n\r\n";
        let whole = SseParser::new().push(stream);
        assert_eq!(
            whole,
            vec![
                SseEvent::Comment("OPENROUTER PROCESSING".into()),
                data("{\"a\":1}"),
                data("x\ny"),
                data("[DONE]"),
            ]
        );
        // Byte-at-a-time gives the identical result.
        let mut p = SseParser::new();
        let mut split = Vec::new();
        for b in stream.iter() {
            split.extend(p.push(std::slice::from_ref(b)));
        }
        split.extend(p.finish());
        assert_eq!(split, whole);
    }

    #[test]
    fn utf8_split_across_chunks_survives() {
        let text = "data: {\"c\":\"café — ☕\"}\n\n".as_bytes();
        for cut in 1..text.len() {
            let mut p = SseParser::new();
            let mut ev = p.push(&text[..cut]);
            ev.extend(p.push(&text[cut..]));
            assert_eq!(ev, vec![data("{\"c\":\"café — ☕\"}")], "cut at {cut}");
        }
    }

    #[test]
    fn final_event_without_blank_line_is_flushed() {
        let mut p = SseParser::new();
        assert!(p.push(b"data: [DONE]").is_empty());
        assert_eq!(p.finish(), vec![data("[DONE]")]);
    }

    fn chunk(content: &str) -> SseEvent {
        data(&format!(
            r#"{{"id":"gen-9","model":"m/x","choices":[{{"index":0,"delta":{{"content":{}}},"finish_reason":null}}]}}"#,
            serde_json::to_string(content).unwrap()
        ))
    }

    #[test]
    fn accumulates_content_usage_and_done() {
        let mut acc = StreamAccumulator::new();
        assert_eq!(
            acc.apply(&SseEvent::Comment("OPENROUTER PROCESSING".into()), 1.0)
                .unwrap(),
            StreamStep::Continue
        );
        assert_eq!(
            acc.apply(&chunk("{\"action\":"), 120.0).unwrap(),
            StreamStep::Progress(10)
        );
        acc.apply(&chunk("\"idle\"}"), 130.0).unwrap();
        acc.apply(
            &data(r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#),
            131.0,
        )
        .unwrap();
        // Accounting chunk repeats finish_reason and carries usage.
        acc.apply(
            &data(r#"{"choices":[{"delta":{"content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":50,"completion_tokens":6,"cost":0.0001}}"#),
            132.0,
        )
        .unwrap();
        assert_eq!(acc.apply(&data("[DONE]"), 133.0).unwrap(), StreamStep::Done);
        assert_eq!(acc.keepalives(), 1);
        let c = acc.finish().unwrap();
        assert_eq!(c.text, r#"{"action":"idle"}"#);
        assert_eq!(c.usage.unwrap().completion_tokens, 6);
        assert_eq!(c.first_token_ms, Some(120.0));
        assert_eq!(c.id.as_deref(), Some("gen-9"));
        assert!(c.streamed);
    }

    #[test]
    fn mid_stream_errors_fail_the_request() {
        let mut acc = StreamAccumulator::new();
        acc.apply(&chunk("{\"act"), 1.0).unwrap();
        let e = acc
            .apply(
                &data(r#"{"id":"x","error":{"code":502,"message":"Provider disconnected"},"choices":[{"delta":{"content":""},"finish_reason":"error"}]}"#),
                2.0,
            )
            .unwrap_err();
        assert!(matches!(e, OpenRouterError::Unavailable { .. }), "{e:?}");

        let mut acc = StreamAccumulator::new();
        let e = acc
            .apply(
                &data(r#"{"choices":[{"delta":{},"finish_reason":"error"}]}"#),
                1.0,
            )
            .unwrap_err();
        assert!(matches!(e, OpenRouterError::Provider(_)));
    }

    #[test]
    fn interrupted_and_empty_streams_are_errors() {
        let mut acc = StreamAccumulator::new();
        acc.apply(&chunk("{\"action\":\"id"), 1.0).unwrap();
        assert!(matches!(acc.finish(), Err(OpenRouterError::Network(_))));

        let mut acc = StreamAccumulator::new();
        acc.apply(&data("[DONE]"), 1.0).unwrap();
        assert!(matches!(acc.finish(), Err(OpenRouterError::Protocol(_))));

        let mut acc = StreamAccumulator::new();
        assert!(matches!(
            acc.apply(&data("{not json"), 1.0),
            Err(OpenRouterError::Protocol(_))
        ));
    }
}
