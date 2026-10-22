//! The NPC action contract: the one structured-output schema every inference
//! backend is asked to produce.
//!
//! Local llama.cpp turns [`ACTION_SCHEMA`] into a GBNF grammar; remote
//! providers (OpenRouter) receive [`strict_action_schema`] as a JSON-schema
//! `response_format`. Both renderings describe the *same* contract, and every
//! answer — whatever produced it and whatever the provider promised — is
//! re-checked here by [`validate_action_json`] before it is parsed into a
//! typed [`crate::action::Action`]. A provider's `strict: true` is a request,
//! not a guarantee.
//!
//! The validated object is still only a *proposal*: canonical grounding,
//! policy resolution and world validation happen afterwards, and only the
//! authoritative simulation commits anything.

use serde_json::{Map, Value};

/// Contract identifier carried in `InferenceRequest::response_contract`.
pub const ACTION_CONTRACT_ID: &str = "action-json-v1";

/// Name used for the JSON-schema `response_format` on remote providers.
pub const ACTION_SCHEMA_NAME: &str = "irl_npc_action";

/// The actions a model may propose.
pub const ACTIONS: [&str; 6] = [
    "speak",
    "share_claim",
    "offer",
    "promise",
    "threaten",
    "idle",
];

/// Loudness values a `speak` action may carry.
pub const LOUDNESS: [&str; 4] = ["whisper", "normal", "loud", "shout"];

/// The canonical JSON schema. `action` is the only mandatory field; any
/// additional fields are action-specific.
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

/// Field kinds in the contract, in schema order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FieldKind {
    Action,
    Str,
    Bool,
    StrArray,
    Loudness,
}

const FIELDS: [(&str, FieldKind); 12] = [
    ("action", FieldKind::Action),
    ("target", FieldKind::Str),
    ("text", FieldKind::Str),
    ("fact", FieldKind::Str),
    ("assertive", FieldKind::Bool),
    ("give", FieldKind::StrArray),
    ("want", FieldKind::StrArray),
    ("note", FieldKind::Str),
    ("terms", FieldKind::Str),
    ("condition", FieldKind::Str),
    ("demand", FieldKind::Str),
    ("loudness", FieldKind::Loudness),
];

/// The canonical schema as a JSON value.
pub fn action_schema() -> Value {
    serde_json::from_str(ACTION_SCHEMA).expect("ACTION_SCHEMA is valid JSON")
}

/// The same contract rendered for providers with an OpenAI-style *strict*
/// structured-output mode.
///
/// Strict modes require every property to be listed in `required` and
/// `additionalProperties: false`, so optional fields are expressed as
/// nullable (`["string", "null"]`) instead of omitted. Only keywords that
/// strict modes broadly accept are used (`type`, `enum`, `items`,
/// `properties`, `required`, `additionalProperties`, `description`).
/// [`validate_action_json`] accepts both renderings: an explicit `null` means
/// the same as an absent field.
pub fn strict_action_schema() -> Value {
    let mut props = Map::new();
    for (name, kind) in FIELDS {
        let v = match kind {
            FieldKind::Action => serde_json::json!({
                "type": "string",
                "enum": ACTIONS,
                "description": "The single action you take."
            }),
            FieldKind::Str => serde_json::json!({"type": ["string", "null"]}),
            FieldKind::Bool => serde_json::json!({"type": ["boolean", "null"]}),
            FieldKind::StrArray => serde_json::json!({
                "type": ["array", "null"],
                "items": {"type": "string"}
            }),
            FieldKind::Loudness => {
                let mut values: Vec<Value> = LOUDNESS.iter().map(|s| Value::from(*s)).collect();
                values.push(Value::Null);
                serde_json::json!({"type": ["string", "null"], "enum": values})
            }
        };
        props.insert(name.to_string(), v);
    }
    serde_json::json!({
        "type": "object",
        "properties": Value::Object(props),
        "required": FIELDS.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
        "additionalProperties": false
    })
}

/// Extract the JSON object a model answered with.
///
/// Structured backends answer with exactly one object; unconstrained text may
/// wrap it in prose or a code fence. Returns the slice from the first `{` to
/// the last `}`.
pub fn extract_json_object(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    (end >= start).then(|| &text[start..=end])
}

/// Validate a model answer against the action contract.
///
/// Returns the parsed object on success. Checks: the text holds a JSON
/// object; `action` is present and one of [`ACTIONS`]; every other key is a
/// contract field of the right type (`null` is accepted for optional fields,
/// as the strict rendering requires); `loudness` is one of [`LOUDNESS`]; no
/// unknown keys (`additionalProperties: false`).
pub fn validate_action_json(text: &str) -> Result<Map<String, Value>, String> {
    let raw = extract_json_object(text).ok_or_else(|| "no JSON object in output".to_string())?;
    let value: Value = serde_json::from_str(raw).map_err(|e| format!("invalid JSON: {e}"))?;
    validate_action_value(&value)?;
    match value {
        Value::Object(map) => Ok(map),
        _ => unreachable!("validated as an object"),
    }
}

/// [`validate_action_json`] for an already-parsed value.
pub fn validate_action_value(value: &Value) -> Result<(), String> {
    let obj = value
        .as_object()
        .ok_or_else(|| "top-level value is not an object".to_string())?;
    match obj.get("action") {
        Some(Value::String(a)) if ACTIONS.contains(&a.as_str()) => {}
        Some(Value::String(a)) => return Err(format!("unknown action {a:?}")),
        Some(other) => return Err(format!("`action` must be a string, got {other}")),
        None => return Err("missing required field `action`".into()),
    }
    for (key, v) in obj {
        let Some((_, kind)) = FIELDS.iter().find(|(n, _)| n == key) else {
            return Err(format!("unexpected field `{key}`"));
        };
        if v.is_null() {
            if *kind == FieldKind::Action {
                return Err("`action` must not be null".into());
            }
            continue;
        }
        let ok = match kind {
            FieldKind::Action => true,
            FieldKind::Str => v.is_string(),
            FieldKind::Bool => v.is_boolean(),
            FieldKind::StrArray => v.as_array().is_some_and(|a| a.iter().all(Value::is_string)),
            FieldKind::Loudness => v.as_str().is_some_and(|s| LOUDNESS.contains(&s)),
        };
        if !ok {
            return Err(format!("field `{key}` has the wrong type or value: {v}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_schema_parses_and_lists_every_field() {
        let s = action_schema();
        let props = s["properties"].as_object().unwrap();
        assert_eq!(props.len(), FIELDS.len());
        for (name, _) in FIELDS {
            assert!(props.contains_key(name), "missing {name}");
        }
        assert_eq!(s["required"], serde_json::json!(["action"]));
    }

    #[test]
    fn strict_schema_requires_every_field_and_is_closed() {
        let s = strict_action_schema();
        assert_eq!(s["additionalProperties"], Value::Bool(false));
        let required: Vec<&str> = s["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(required.len(), FIELDS.len());
        // Optional fields are nullable, `action` is not.
        assert_eq!(s["properties"]["action"]["type"], "string");
        assert_eq!(
            s["properties"]["text"]["type"],
            serde_json::json!(["string", "null"])
        );
        assert!(s["properties"]["loudness"]["enum"]
            .as_array()
            .unwrap()
            .contains(&Value::Null));
        // Same field set and action enum as the canonical schema.
        let canon = action_schema();
        assert_eq!(
            canon["properties"]["action"]["enum"],
            s["properties"]["action"]["enum"]
        );
    }

    #[test]
    fn validator_accepts_both_renderings() {
        assert!(validate_action_json(r#"{"action":"idle"}"#).is_ok());
        assert!(validate_action_json(
            r#"{"action":"speak","target":"player","text":"Hi","fact":null,"assertive":null,
               "give":null,"want":null,"note":null,"terms":null,"condition":null,"demand":null,
               "loudness":null}"#
        )
        .is_ok());
        assert!(validate_action_json("Sure! ```json\n{\"action\":\"idle\"}\n```").is_ok());
    }

    #[test]
    fn validator_rejects_contract_violations() {
        for (bad, why) in [
            ("no json here", "no JSON"),
            (r#"{"text":"hi"}"#, "missing"),
            (r#"{"action":"steal"}"#, "unknown action"),
            (r#"{"action":null}"#, "string"),
            (r#"{"action":"speak","text":3}"#, "wrong type"),
            (r#"{"action":"offer","give":["a",1]}"#, "wrong type"),
            (r#"{"action":"speak","loudness":"scream"}"#, "wrong type"),
            (
                r#"{"action":"idle","mutate_world":true}"#,
                "unexpected field",
            ),
            (
                r#"{"action":"idle"} and {"action":"speak"}"#,
                "invalid JSON",
            ),
            (r#"{"action":"idle""#, "no JSON"),
        ] {
            let err = validate_action_json(bad).expect_err(bad);
            assert!(err.contains(why), "{bad}: {err}");
        }
    }
}
