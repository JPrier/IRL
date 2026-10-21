//! # rpg_inference
//!
//! Resident local inference via `llama-cpp-2` (llama.cpp). This is the
//! production-intended backend behind `rpg_domain::ports::InferenceEngine`.
//!
//! Design properties:
//! * weights are loaded once and reused for every NPC;
//! * the chat template baked into the GGUF is used when available;
//! * tool decisions are constrained to JSON with a grammar sampler;
//! * tokens are streamed to a sink as they are produced;
//! * deadlines and cooperative cancellation are checked per token;
//! * CPU is the default; `n_gpu_layers` enables GPU offload when the build
//!   supports it.
//!
//! The model file path and all runtime knobs come from a JSON [`ModelProfile`],
//! so models can be swapped without recompiling.

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use rpg_domain::ports::{InferenceEngine, InferenceError, InferenceRequest, InferenceResponse};
use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

/// ChatML markers built from escapes so the literal control tokens never
/// appear in source (some tooling sanitises them).
const IM_START: &str = "\u{3c}\u{7c}im_start\u{7c}\u{3e}";
const IM_END: &str = "\u{3c}\u{7c}im_end\u{7c}\u{3e}";
const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

/// JSON schema used for constrained tool decisions. The `action` enum is the
/// only mandatory field; any additional fields are action-specific.
pub const ACTION_SCHEMA: &str = r#"
{
  "type": "object",
  "properties": {
    "action": {"type": "string", "enum": ["speak", "share_claim", "offer", "promise", "threaten", "idle"]},
    "target": {"type": "string"},
    "text": {"type": "string"},
    "fact": {"type": "string"},
    "assertive": {"type": "boolean"},
    "give": {"type": "array", "items": {"type": "string"}},
    "want": {"type": "array", "items": {"type": "string"}},
    "note": {"type": "string"},
    "terms": {"type": "string"},
    "condition": {"type": "string"},
    "demand": {"type": "string"},
    "loudness": {"type": "string", "enum": ["whisper", "normal", "loud", "shout"]}
  },
  "required": ["action"],
  "additionalProperties": false
}
"#;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelProfile {
    pub id: String,
    pub model_path: String,
    #[serde(default = "default_n_ctx")]
    pub n_ctx: u32,
    #[serde(default = "default_threads")]
    pub n_threads: i32,
    #[serde(default)]
    pub n_gpu_layers: u32,
    #[serde(default = "default_temp")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub seed: u32,
    /// `"action-json-v1"`, raw GBNF, or `None` for unconstrained text.
    #[serde(default)]
    pub grammar: Option<String>,
    #[serde(default)]
    pub stop: Vec<String>,
    #[serde(default = "default_deadline")]
    pub deadline_ms: u64,
    /// Ask the model to skip its reasoning block (family-specific marker).
    #[serde(default)]
    pub no_think: bool,
    #[serde(default)]
    pub chat_format: Option<String>,
}

fn default_n_ctx() -> u32 {
    4096
}
fn default_threads() -> i32 {
    4
}
fn default_temp() -> f32 {
    0.4
}
fn default_top_p() -> f32 {
    0.9
}
fn default_max_tokens() -> u32 {
    192
}
fn default_deadline() -> u64 {
    30_000
}

impl ModelProfile {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, InferenceError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| InferenceError::MissingAsset(e.to_string()))?;
        serde_json::from_str(&text).map_err(|e| InferenceError::Backend(e.to_string()))
    }

    pub fn from_json(text: &str) -> Result<Self, InferenceError> {
        serde_json::from_str(text).map_err(|e| InferenceError::Backend(e.to_string()))
    }

    /// A conservative estimate of resident weight memory from the file size.
    pub fn weight_bytes(&self) -> u64 {
        std::fs::metadata(&self.model_path)
            .map(|m| m.len())
            .unwrap_or(0)
    }
}

fn shared_backend() -> Result<&'static LlamaBackend, InferenceError> {
    static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
    if BACKEND.get().is_none() {
        let backend =
            LlamaBackend::init().map_err(|e| InferenceError::Backend(format!("{e:?}")))?;
        // If another thread won the race, keep the existing backend.
        let _ = BACKEND.set(backend);
    }
    BACKEND
        .get()
        .ok_or_else(|| InferenceError::Backend("llama backend unavailable".into()))
}

/// Buffers partial UTF-8 sequences so streamed text never contains replaced
/// bytes mid-codepoint.
#[derive(Default)]
struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let out = s.to_string();
                self.pending.clear();
                out
            }
            Err(err) => {
                let valid = err.valid_up_to();
                let out = String::from_utf8_lossy(&self.pending[..valid]).to_string();
                self.pending.drain(..valid);
                out
            }
        }
    }

    fn finish(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.pending).to_string();
        self.pending.clear();
        out
    }
}

pub struct LlamaEngine {
    profile: ModelProfile,
    model: LlamaModel,
    cancel: Arc<AtomicBool>,
    warmed: bool,
}

impl LlamaEngine {
    pub fn load(profile: ModelProfile) -> Result<Self, InferenceError> {
        if !Path::new(&profile.model_path).exists() {
            return Err(InferenceError::MissingAsset(profile.model_path.clone()));
        }
        let backend = shared_backend()?;
        let params = LlamaModelParams::default().with_n_gpu_layers(profile.n_gpu_layers);
        let model = LlamaModel::load_from_file(backend, &profile.model_path, &params)
            .map_err(|e| InferenceError::Backend(format!("{e:?}")))?;
        Ok(Self {
            profile,
            model,
            cancel: Arc::new(AtomicBool::new(false)),
            warmed: false,
        })
    }

    pub fn profile(&self) -> &ModelProfile {
        &self.profile
    }

    /// A handle another thread can use to request cancellation.
    pub fn cancel_handle(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }

    pub fn model_path(&self) -> PathBuf {
        PathBuf::from(&self.profile.model_path)
    }

    fn build_prompt(&self, req: &InferenceRequest) -> String {
        let mut prompt = self.render_chat(req);
        if self.profile.no_think {
            apply_no_think_prefix(&mut prompt);
        }
        prompt
    }

    /// Render the chat template (baked GGUF template, named override, or the
    /// ChatML fallback). `llama-cpp-2` 0.1.160 does not expose template kwargs
    /// such as `enable_thinking`, so non-thinking behaviour is normalised in
    /// [`apply_no_think_prefix`].
    fn render_chat(&self, req: &InferenceRequest) -> String {
        let system = req.system.clone();
        let tmpl = match &self.profile.chat_format {
            Some(name) => self.model.chat_template(Some(name)).ok(),
            None => self.model.chat_template(None).ok(),
        };
        if let Some(tmpl) = tmpl {
            if let (Ok(sys), Ok(user)) = (
                LlamaChatMessage::new("system".into(), system.clone()),
                LlamaChatMessage::new("user".into(), req.prompt.clone()),
            ) {
                if let Ok(prompt) = self.model.apply_chat_template(&tmpl, &[sys, user], true) {
                    return prompt;
                }
            }
        }
        // Fallback: Qwen ChatML with a pre-closed empty reasoning block.
        format!(
            "{start}system\n{system}{end}\n{start}user\n{prompt}{end}\n{start}assistant\n{think_open}\n\n{think_close}\n\n",
            start = IM_START,
            end = IM_END,
            think_open = THINK_OPEN,
            think_close = THINK_CLOSE,
            prompt = req.prompt
        )
    }

    fn grammar_for(&self, req: &InferenceRequest) -> Option<String> {
        let spec = req
            .response_contract
            .as_deref()
            .or(self.profile.grammar.as_deref())?;
        if spec == "action-json-v1" {
            llama_cpp_2::json_schema_to_grammar(ACTION_SCHEMA).ok()
        } else {
            Some(spec.to_string())
        }
    }
}

/// Rewrite a bare or open-ended assistant prefix into a pre-closed empty
/// reasoning block. Needed because `llama-cpp-2` 0.1.160 cannot forward
/// `enable_thinking=false` to the GGUF chat template.
fn apply_no_think_prefix(prompt: &mut String) {
    let nl = char::from(10u8);
    let open = format!("{THINK_OPEN}{nl}");
    let empty = format!("{THINK_OPEN}{nl}{nl}{THINK_CLOSE}{nl}{nl}");
    if prompt.ends_with(&empty) {
        return;
    }
    if prompt.ends_with(&open) {
        prompt.truncate(prompt.len() - open.len());
        prompt.push_str(&empty);
        return;
    }
    let assistant_nl = format!("assistant{nl}");
    if prompt.ends_with(&assistant_nl) || prompt.ends_with("assistant") {
        prompt.push_str(&empty);
    }
}
/// Remove a leading reasoning block (including a pre-filled empty one) so
/// callers receive only the answer. Text without a complete block is
/// returned unchanged.
fn strip_reasoning_block(text: String) -> String {
    let trimmed = text.trim_start();
    if !trimmed.starts_with(THINK_OPEN) {
        return text;
    }
    if let Some(end) = trimmed.find(THINK_CLOSE) {
        let rest = &trimmed[end + THINK_CLOSE.len()..];
        let nl = char::from(10u8);
        let cr = char::from(13u8);
        return rest.trim_start_matches([nl, cr]).to_string();
    }
    text
}

/// True when `text` is exactly one complete JSON object. Used to stop
/// constrained decoding at the grammar's terminal state.
fn json_value_complete(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.len() < 2 || !trimmed.starts_with('{') || !trimmed.ends_with('}') {
        return false;
    }
    serde_json::from_str::<serde_json::Value>(trimmed).is_ok()
}

impl InferenceEngine for LlamaEngine {
    fn engine_name(&self) -> String {
        "llama.cpp".into()
    }

    fn model_id(&self) -> String {
        self.profile.id.clone()
    }

    fn warm(&mut self) -> Result<(), InferenceError> {
        if self.warmed {
            return Ok(());
        }
        let req = InferenceRequest {
            id: 0,
            role: rpg_domain::ports::InferenceRole::NpcDecision,
            system: "Warm up.".into(),
            prompt: "Reply with {\"action\":\"idle\"}.".into(),
            max_tokens: 8,
            temperature: 0.0,
            seed: 0,
            response_contract: None,
        };
        let mut sink = |_: &str| {};
        let _ = self.complete_streaming(&req, &mut sink)?;
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
        self.cancel.store(false, Ordering::SeqCst);
        let backend = shared_backend()?;
        let prompt = self.build_prompt(req);
        if std::env::var("RPG_DEBUG_PROMPT").is_ok() {
            eprintln!(
                "--- prompt ---
{prompt}
--- end prompt ---"
            );
        }
        let prompt_tokens = self.model.vocab().tokenize(prompt.as_bytes(), true, true);
        let max_new = req.max_tokens.min(self.profile.max_tokens).max(1) as usize;

        // Keep the prompt within the context window.
        let ctx_size = self.profile.n_ctx as usize;
        let keep = ctx_size.saturating_sub(max_new + 8).max(1);
        let prompt_tokens: Vec<_> = if prompt_tokens.len() > keep {
            prompt_tokens[prompt_tokens.len() - keep..].to_vec()
        } else {
            prompt_tokens
        };

        let n_ctx = NonZeroU32::new(self.profile.n_ctx.max(256));
        let params = LlamaContextParams::default()
            .with_n_ctx(n_ctx)
            .with_n_threads(self.profile.n_threads)
            .with_n_threads_batch(self.profile.n_threads)
            .with_n_batch(prompt_tokens.len().max(1) as u32);
        let mut ctx = self
            .model
            .new_context(backend, params)
            .map_err(|e| InferenceError::Backend(format!("{e:?}")))?;

        let prefill_start = Instant::now();
        let mut batch = LlamaBatch::new(prompt_tokens.len().max(1), 1);
        batch
            .add_sequence(&prompt_tokens, 0, false)
            .map_err(|e| InferenceError::Backend(format!("{e:?}")))?;
        ctx.decode(&mut batch)
            .map_err(|e| InferenceError::Backend(format!("{e:?}")))?;
        let prefill_ms = prefill_start.elapsed().as_secs_f32() * 1000.0;

        let grammar = self.grammar_for(req);
        let structured = grammar.is_some();
        let mut samplers: Vec<LlamaSampler> = Vec::new();
        if let Some(grammar) = &grammar {
            // A malformed grammar must be a loud, recoverable error. The old
            // code silently dropped it and then claimed `structured: true`.
            let g = LlamaSampler::grammar(&self.model, grammar, "root").map_err(|e| {
                InferenceError::Backend(format!("grammar sampler init failed: {e:?}"))
            })?;
            samplers.push(g);
        }
        samplers.push(LlamaSampler::temp(req.temperature.max(0.0)));
        samplers.push(LlamaSampler::top_p(self.profile.top_p, 1));
        samplers.push(LlamaSampler::dist(req.seed as u32));
        let mut sampler = LlamaSampler::chain_simple(samplers);

        let decode_start = Instant::now();
        let mut first_token_ms = 0.0f32;
        let mut text = String::new();
        let mut utf8 = Utf8Stream::default();
        let mut decoder = LlamaBatch::new(1, 1);
        let mut n_cur = prompt_tokens.len();
        let mut produced = 0u32;

        while produced < max_new as u32 {
            if self.cancel.load(Ordering::SeqCst) {
                return Err(InferenceError::Timeout(0));
            }
            if decode_start.elapsed().as_millis() as u64 > self.profile.deadline_ms {
                return Err(InferenceError::Timeout(self.profile.deadline_ms));
            }
            // `LlamaSampler::sample` already accepts the selected token into
            // the sampler chain ("Sample and accept a token..."). Calling
            // `accept` again was the root cause of the grammar abort: it
            // advanced the grammar twice per token and left the stack empty,
            // so the next `sample()` hit
            // `GGML_ASSERT(!stacks.empty())` in `llama_grammar_reject_candidates`.
            let token = sampler.sample(&ctx, -1);
            if self.model.vocab().is_eog(token) {
                break;
            }
            let bytes = self.model.vocab().token_to_piece(token, false, None);
            let piece = utf8.push(&bytes);
            if !piece.is_empty() {
                if produced == 0 {
                    first_token_ms = decode_start.elapsed().as_secs_f32() * 1000.0;
                }
                text.push_str(&piece);
                sink(&piece);
            }
            produced += 1;

            // A JSON-schema value is complete once it parses. Stop before the
            // next `sample()` so no trailing prose is generated. The grammar
            // sampler permits EOG at this point, so `is_eog` would also stop
            // us, but not reliably on every model.
            if structured && json_value_complete(&text) {
                break;
            }
            if self
                .profile
                .stop
                .iter()
                .any(|s| !s.is_empty() && text.contains(s))
            {
                break;
            }

            decoder.clear();
            decoder
                .add(token, n_cur as i32, &[0], true)
                .map_err(|e| InferenceError::Backend(format!("{e:?}")))?;
            n_cur += 1;
            ctx.decode(&mut decoder)
                .map_err(|e| InferenceError::Backend(format!("{e:?}")))?;
        }
        let tail = utf8.finish();
        if !tail.is_empty() {
            text.push_str(&tail);
            sink(&tail);
        }
        let decode_ms = decode_start.elapsed().as_secs_f32() * 1000.0;

        Ok(InferenceResponse {
            text: strip_reasoning_block(text),
            prompt_tokens: prompt_tokens.len() as u32,
            completion_tokens: produced,
            first_token_ms,
            prefill_ms,
            decode_ms,
            structured,
        })
    }

    fn cancel(&mut self, _id: u64) -> bool {
        self.cancel.store(true, Ordering::SeqCst);
        true
    }
}

/// One-line description of the llama.cpp build for the benchmark report.
pub fn backend_info() -> String {
    format!(
        "llama.cpp via llama-cpp-2 {} (build: {})",
        env!("CARGO_PKG_VERSION"),
        if cfg!(feature = "cuda") {
            "cuda"
        } else if cfg!(target_os = "macos") {
            "metal"
        } else {
            "cpu"
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_grammar_is_valid_gbnf_source() {
        let g = llama_cpp_2::json_schema_to_grammar(ACTION_SCHEMA);
        assert!(g.is_ok(), "schema to grammar failed: {g:?}");
        assert!(g.unwrap().contains("root"));
    }

    #[test]
    fn action_grammar_has_expected_root_and_enum() {
        let g = llama_cpp_2::json_schema_to_grammar(ACTION_SCHEMA).unwrap();
        assert!(g.contains("root ::="), "root rule missing:\n{g}");
        for action in [
            "speak",
            "share_claim",
            "offer",
            "promise",
            "threaten",
            "idle",
        ] {
            assert!(g.contains(action), "action enum missing {action}");
        }
    }

    #[test]
    fn json_value_complete_only_accepts_full_objects() {
        assert!(!json_value_complete(""));
        assert!(!json_value_complete("{"));
        assert!(!json_value_complete("{\"action\":"));
        assert!(!json_value_complete("{\"action\":\"idle\""));
        assert!(!json_value_complete("[1,2]"));
        assert!(json_value_complete("{\"action\":\"idle\"}"));
        assert!(json_value_complete("  {\"action\":\"idle\"}\n"));
    }

    #[test]
    fn strip_reasoning_block_removes_only_complete_blocks() {
        let nl = char::from(10u8);
        let empty = format!("{THINK_OPEN}{nl}{nl}{THINK_CLOSE}{nl}{nl}");
        let filled = format!("{THINK_OPEN}{nl}reasoning{nl}{THINK_CLOSE}{nl}{nl}answer");
        assert_eq!(strip_reasoning_block(filled), "answer");
        assert_eq!(strip_reasoning_block(empty.clone()), "");
        assert_eq!(strip_reasoning_block("plain answer".into()), "plain answer");
        // An unterminated block is left alone.
        let dangling = format!("{THINK_OPEN}{nl}still thinking");
        assert_eq!(strip_reasoning_block(dangling.clone()), dangling);
    }

    #[test]
    fn no_think_prefix_is_normalised_and_idempotent() {
        let nl = char::from(10u8);
        let empty = format!("{THINK_OPEN}{nl}{nl}{THINK_CLOSE}{nl}{nl}");
        let mut p = format!("{IM_START}assistant{nl}{THINK_OPEN}{nl}");
        apply_no_think_prefix(&mut p);
        assert!(p.ends_with(&empty), "got {p:?}");
        let mut again = p.clone();
        apply_no_think_prefix(&mut again);
        assert_eq!(p, again, "apply_no_think_prefix must be idempotent");

        let mut bare = format!("{IM_START}assistant{nl}");
        apply_no_think_prefix(&mut bare);
        assert!(bare.ends_with(&empty), "got {bare:?}");

        // An already explicitly-closed reasoning block is left untouched.
        let mut closed = format!("{IM_START}assistant{nl}{empty}");
        let before = closed.clone();
        apply_no_think_prefix(&mut closed);
        assert_eq!(closed, before);
    }
    #[test]
    fn profile_parses_defaults() {
        let p =
            ModelProfile::from_json(r#"{"id":"test","model_path":"/tmp/missing.gguf"}"#).unwrap();
        assert_eq!(p.n_ctx, 4096);
        assert_eq!(p.max_tokens, 192);
        assert_eq!(p.n_threads, 4);
    }

    #[test]
    fn utf8_stream_handles_split_codepoints() {
        let mut s = Utf8Stream::default();
        // "é" is 0xC3 0xA9.
        assert_eq!(s.push(&[0xC3]), "");
        assert_eq!(s.push(&[0xA9]), "é");
        assert_eq!(s.finish(), "");
    }

    #[test]
    fn missing_model_fails_loudly() {
        let p = ModelProfile::from_json(
            r#"{"id":"missing","model_path":"./definitely-not-here.gguf"}"#,
        )
        .unwrap();
        assert!(matches!(
            LlamaEngine::load(p),
            Err(InferenceError::MissingAsset(_))
        ));
    }
}
