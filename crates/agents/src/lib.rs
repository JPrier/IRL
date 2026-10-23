//! # rpg_agents
//!
//! Logical agents built on top of the canonical world:
//!
//! * [`Scheduler`] — priority queue for shared inference work.
//! * [`ScriptedEngine`] / [`HeuristicEngine`] — deterministic inference
//!   backends used by tests and as no-LLM baselines.
//! * [`NpcPolicy`] — builds a minimal scoped context and turns a model (or
//!   deterministic fallback) decision into a typed [`ActionProposal`].
//! * [`DirectorAgent`] — proposes narrative changes for the controller to
//!   validate.
//! * [`SimpleMemoryRetriever`] — actor-scoped, provenance-aware retrieval.
//!
//! Agents never mutate world state; they only propose.

use rpg_domain::action::{Action, ActionProposal, ProposalSource, QuestPredicate};
use rpg_domain::ids::*;
use rpg_domain::ports::{
    InferenceEngine, InferenceError, InferenceRequest, InferenceResponse, InferenceRole,
    MemoryRetriever, RetrievedMemory,
};
use rpg_domain::social::CommitmentState;
use rpg_domain::world::{ActorKind, Sensitivity, World};
use rpg_narrative::{Decision, NarrativeChange};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Queued {
    priority: u8,
    seq: u64,
    request: InferenceRequest,
}

impl PartialEq for Queued {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.seq == other.seq
    }
}
impl Eq for Queued {}

impl PartialOrd for Queued {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Queued {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap on priority, then FIFO within a priority.
        other
            .priority
            .cmp(&self.priority)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

/// Priority scheduler for a single shared model. P0 player-facing, P1 local,
/// P2 background. Coalescing/cancellation hooks are intentionally explicit.
#[derive(Default)]
pub struct Scheduler {
    queue: BinaryHeap<Queued>,
    next_seq: u64,
    pub dropped: u64,
    pub completed: u64,
}

impl Scheduler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, request: InferenceRequest) {
        let priority = request.role.priority();
        self.next_seq += 1;
        self.queue.push(Queued {
            priority,
            seq: self.next_seq,
            request,
        });
    }

    pub fn pop(&mut self) -> Option<InferenceRequest> {
        self.queue.pop().map(|q| q.request)
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Drop all queued work below `min_priority` (e.g. under load).
    pub fn drop_below(&mut self, min_priority: u8) {
        let keep: Vec<Queued> = self
            .queue
            .drain()
            .filter(|q| {
                let keep = q.priority <= min_priority;
                if !keep {
                    self.dropped += 1;
                }
                keep
            })
            .collect();
        self.queue.extend(keep);
    }
}

// ---------------------------------------------------------------------------
// Deterministic test / baseline engines
// ---------------------------------------------------------------------------

/// Returns pre-registered responses keyed by request id. Unmatched requests
/// return `default` (or an empty string). Every request is recorded so tests
/// can assert on prompt scoping.
#[derive(Default)]
pub struct ScriptedEngine {
    responses: BTreeMap<u64, String>,
    default: String,
    pub requests: Vec<InferenceRequest>,
    warmed: bool,
}

impl ScriptedEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_default(mut self, response: impl Into<String>) -> Self {
        self.default = response.into();
        self
    }

    pub fn register(&mut self, id: u64, response: impl Into<String>) {
        self.responses.insert(id, response.into());
    }

    pub fn last_prompt(&self) -> Option<&str> {
        self.requests.last().map(|r| r.prompt.as_str())
    }
}

impl InferenceEngine for ScriptedEngine {
    fn engine_name(&self) -> String {
        "scripted".into()
    }
    fn model_id(&self) -> String {
        "scripted-v1".into()
    }
    fn warm(&mut self) -> Result<(), InferenceError> {
        self.warmed = true;
        Ok(())
    }
    fn complete(&mut self, req: &InferenceRequest) -> Result<InferenceResponse, InferenceError> {
        if !self.warmed {
            self.warm()?;
        }
        self.requests.push(req.clone());
        let text = self
            .responses
            .get(&req.id)
            .cloned()
            .unwrap_or_else(|| self.default.clone());
        Ok(InferenceResponse {
            completion_tokens: text.split_whitespace().count() as u32,
            prompt_tokens: (req.prompt.len() / 4) as u32,
            first_token_ms: 0.0,
            prefill_ms: 0.0,
            decode_ms: 0.0,
            structured: false,
            text,
            remote: None,
        })
    }
}

/// A no-LLM baseline that always returns an unparseable empty object, causing
/// the policy to use its deterministic fallback. Useful as a latency floor in
/// benchmarks: it costs nothing and must not be mistaken for intelligence.
#[derive(Default)]
pub struct HeuristicEngine {
    warmed: bool,
}

impl HeuristicEngine {
    pub fn new() -> Self {
        Self::default()
    }
}

impl InferenceEngine for HeuristicEngine {
    fn engine_name(&self) -> String {
        "heuristic".into()
    }
    fn model_id(&self) -> String {
        "heuristic-v1".into()
    }
    fn is_real_model(&self) -> bool {
        false
    }
    fn warm(&mut self) -> Result<(), InferenceError> {
        self.warmed = true;
        Ok(())
    }
    fn complete(&mut self, _req: &InferenceRequest) -> Result<InferenceResponse, InferenceError> {
        Ok(InferenceResponse::default())
    }
}

// ---------------------------------------------------------------------------
// Context construction
// ---------------------------------------------------------------------------

/// Build the *minimal sufficient* prompt for one NPC response. This never
/// includes facts the NPC does not personally believe, nor any director-only
/// state.
pub fn render_npc_context(
    world: &World,
    npc: &ActorId,
    speaker: &ActorId,
    utterance: &str,
    history: &[(ActorId, String)],
) -> String {
    let Some(a) = world.actor(npc) else {
        return String::new();
    };
    let mut out = String::new();
    out.push_str(&format!(
        "You are {} ({}). Traits: {}.\n",
        a.name,
        a.id,
        if a.personality.tags.is_empty() {
            "ordinary".to_string()
        } else {
            a.personality.tags.join(", ")
        }
    ));
    if !a.affiliations.is_empty() {
        out.push_str(&format!(
            "Affiliations: {}.\n",
            a.affiliations
                .iter()
                .map(|f| f.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    let loc = world
        .location(&a.location)
        .map(|l| l.name.as_str())
        .unwrap_or("somewhere");
    out.push_str(&format!("You are at {loc}.\n"));

    if !a.biography.is_empty() {
        out.push_str(&format!("Your background: {}\n", a.biography));
    }

    // Shared world lore available to every resident.
    if !world.lore.settlement_name.is_empty() {
        out.push_str(&format!(
            "About {}: {}\n",
            world.lore.settlement_name, world.lore.geography
        ));
        out.push_str(&format!("History: {}\n", world.lore.history));
        out.push_str(&format!("Culture: {}\n", world.lore.culture));
        out.push_str(&format!("Economy: {}\n", world.lore.economy));
        out.push_str(&format!("Institutions: {}\n", world.lore.institutions));
        out.push_str(&format!("Recent events: {}\n", world.lore.recent_events));
        out.push_str(&format!(
            "Everyday knowledge: {}\n",
            world.lore.everyday_knowledge
        ));
        if !world.lore.landmarks.is_empty() {
            out.push_str(&format!("Landmarks: {}\n", world.lore.landmarks.join("; ")));
        }
    }

    let rel = a.relationship(speaker);
    out.push_str(&format!(
        "Toward {}: trust={:.2} affection={:.2} fear={:.2} anger={:.2} suspicion={:.2} obligation={:.2}.\n",
        speaker, rel.trust, rel.affection, rel.fear, rel.anger, rel.suspicion, rel.obligation
    ));

    // Only the NPC's own beliefs, and never inner-circle facts for outsiders.
    out.push_str("Things you believe:\n");
    for (fid, b) in &a.beliefs {
        if !b.believes_true {
            continue;
        }
        if let Some(f) = world.fact(fid) {
            if f.sensitivity == Sensitivity::InnerCircle && !a.has_flag("inner_circle") {
                continue;
            }
            out.push_str(&format!("- {}: {}\n", fid, f.statement));
        }
    }

    if !a.goals.is_empty() {
        out.push_str("Your goals:\n");
        for g in &a.goals {
            out.push_str(&format!("- {}\n", g.description));
        }
    }

    // Deals the NPC owes. A satisfied deal must be honoured: the model is told
    // to disclose the promised fact instead of stalling with another offer.
    let deals: Vec<String> = world
        .commitments
        .values()
        .filter(|c| c.debtor == *npc)
        .map(|c| {
            let fact = c
                .promised_fact
                .as_ref()
                .map(|f| format!(" (promised fact {f})"))
                .unwrap_or_default();
            format!(
                "#{} to {}: {} [{:?}]{}",
                c.id, c.creditor, c.terms, c.state, fact
            )
        })
        .collect();
    if !deals.is_empty() {
        out.push_str("Deals you owe:\n");
        for d in &deals {
            out.push_str(&format!("- {d}\n"));
        }
        out.push_str(
            "If one of your deals is Satisfied, honour it now: share the promised \
             fact with {\"action\":\"share_claim\",\"fact\":\"<fact_id>\"}.\n",
        );
    }

    if !history.is_empty() {
        out.push_str("\nRecent conversation:\n");
        for (who, said) in history {
            out.push_str(&format!("{}: \"{}\"\n", who, said));
        }
    }

    out.push_str(&format!("\n{} says: \"{}\"\n", speaker, utterance));
    out.push_str(
        "Rules:\n\
         - You are a person with goals, memories, and relationships. Respond naturally as your character would.\n\
         - Use the recent conversation to understand references like 'here', 'that', or 'him'.\n\
         - Never invent facts you do not know. If you do not know, say so or express uncertainty.\n\
         - You may share facts you believe, but only if you are willing to. You can refuse, deflect, lie, or bargain.\n\
         - Ordinary greetings, small talk, and questions about the town or your work should get a natural spoken reply.\n\
         - Only make an offer, promise, or threat when it fits the conversation and your goals.\n\
         - Keep replies brief unless the player asks for substantial information.\n",
    );
    out.push_str(
        r#"Respond with ONE JSON object, no prose. Allowed shapes:
{"action":"speak","target":"<id>","text":"..."}
{"action":"share_claim","target":"<id>","fact":"<fact_id>","assertive":true}
{"action":"offer","target":"<id>","condition":"has_item:<item>","fact":"<fact_id you will share>","note":"what you promise in return"}
{"action":"promise","target":"<id>","terms":"...","condition":"has_item:<item>"}
{"action":"threaten","target":"<id>","demand":"..."}
{"action":"idle"}
If you need something before you help, make a conditional offer: condition "has_item:<item>".
"#,
    );
    out
}

/// Deserialize an optional list where an explicit `null` (the strict
/// structured-output rendering of an absent field) means empty.
fn null_as_empty<'de, D>(de: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<Vec<String>>::deserialize(de)?.unwrap_or_default())
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct DecisionJson {
    #[serde(default)]
    action: String,
    #[serde(default)]
    target: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    fact: Option<String>,
    #[serde(default)]
    assertive: Option<bool>,
    #[serde(default, deserialize_with = "null_as_empty")]
    give: Vec<String>,
    #[serde(default, deserialize_with = "null_as_empty")]
    want: Vec<String>,
    #[serde(default)]
    note: Option<String>,
    #[serde(default)]
    terms: Option<String>,
    #[serde(default)]
    condition: Option<String>,
    #[serde(default)]
    demand: Option<String>,
    #[serde(default)]
    loudness: Option<String>,
}

fn parse_decision(
    text: &str,
    world: Option<&World>,
    npc: &ActorId,
    speaker: &ActorId,
) -> Option<Action> {
    let raw = rpg_domain::contract::extract_json_object(text)?;
    let d: DecisionJson = serde_json::from_str(raw).ok()?;
    // Ground the target: a hallucinated actor id is repaired to the person the
    // NPC is actually addressing. The simulation still validates every field.
    let target = match d.target.as_deref() {
        Some(t)
            if world
                .map(|w| w.actors.contains_key(&ActorId::new(t)))
                .unwrap_or(true) =>
        {
            Some(ActorId::new(t))
        }
        _ => Some(speaker.clone()),
    };
    let item = |s: &String| rpg_domain::world::ItemStack {
        item: ItemId::new(s),
        qty: 1,
    };
    match d.action.as_str() {
        "speak" => Some(Action::Speak {
            target,
            text: d.text.unwrap_or_default(),
            loudness: match d.loudness.as_deref() {
                Some("whisper") => rpg_domain::action::Loudness::Whisper,
                Some("loud") => rpg_domain::action::Loudness::Loud,
                Some("shout") => rpg_domain::action::Loudness::Shout,
                _ => rpg_domain::action::Loudness::Normal,
            },
        }),
        "share_claim" => Some(Action::ShareClaim {
            target: target?,
            fact: FactId::new(d.fact?),
            assertive: d.assertive.unwrap_or(true),
        }),
        "offer" => {
            let condition = d
                .condition
                .as_deref()
                .and_then(|c| parse_offer_condition(c, world, npc))
                .or_else(|| {
                    d.want.first().map(|w| QuestPredicate::HasItem {
                        actor: npc.clone(),
                        item: ItemId::new(w),
                        qty: 1,
                    })
                });
            Some(Action::MakeOffer {
                target: target?,
                give: d.give.iter().map(item).collect(),
                want: d.want.iter().map(item).collect(),
                condition,
                disclose: d.fact.map(FactId::new),
                note: d.note.unwrap_or_default(),
            })
        }
        "promise" => Some(Action::Promise {
            target: target?,
            terms: d.terms.unwrap_or_default(),
            predicate: d
                .condition
                .as_deref()
                .and_then(|c| parse_offer_condition(c, world, npc))
                .unwrap_or_else(|| QuestPredicate::Flag(d.condition.unwrap_or_default())),
            deadline: None,
        }),
        "threaten" => Some(Action::Threaten {
            target: target?,
            demand: d.demand.unwrap_or_default(),
        }),
        "idle" | "" => Some(Action::Idle),
        _ => None,
    }
}

/// Ground a model-provided condition string into a canonical predicate.
///
/// The schema exposes only a free-form `condition` string, so the model may say
/// `has_item:medicine`, `flag:met_before`, or just `medicine`. Bare tokens are
/// resolved as item ids when the world defines that item, otherwise as flags.
fn parse_offer_condition(
    raw: &str,
    world: Option<&World>,
    npc: &ActorId,
) -> Option<QuestPredicate> {
    let s = raw.trim();
    if s.is_empty() {
        return None;
    }
    let (kind, value) = match s.split_once(':') {
        Some((k, v)) => (k.trim().to_ascii_lowercase(), v.trim().to_string()),
        None => (String::new(), s.to_string()),
    };
    if value.is_empty() {
        return None;
    }
    match kind.as_str() {
        "has_item" | "hasitem" | "item" => Some(QuestPredicate::HasItem {
            actor: npc.clone(),
            item: ItemId::new(&value),
            qty: 1,
        }),
        "flag" => Some(QuestPredicate::Flag(value)),
        _ => {
            let known_item = world
                .map(|w| w.items.contains_key(&ItemId::new(&value)))
                .unwrap_or(false);
            if known_item {
                Some(QuestPredicate::HasItem {
                    actor: npc.clone(),
                    item: ItemId::new(&value),
                    qty: 1,
                })
            } else {
                Some(QuestPredicate::Flag(value))
            }
        }
    }
}

/// Deterministic fallback used when the model is unavailable or returns an
/// invalid action. This is intentionally simple and *not* a substitute for
/// real conversation; it keeps a scenario runnable under hard failure.
pub fn heuristic_decision(
    world: &World,
    npc: &ActorId,
    speaker: &ActorId,
    utterance: &str,
) -> Action {
    let Some(a) = world.actor(npc) else {
        return Action::Idle;
    };
    let u = utterance.to_lowercase();
    let rel = a.relationship(speaker);
    let wants_ledger = u.contains("ledger") || u.contains("record") || u.contains("book");
    let wants_secret = u.contains("mafia") || u.contains("shipment") || u.contains("smuggl");
    let hostile = u.contains("threat") || u.contains("or else") || u.contains("hurt");

    if hostile {
        return Action::Speak {
            target: Some(speaker.clone()),
            text: "Careful. Threats have a way of being remembered.".into(),
            loudness: rpg_domain::action::Loudness::Normal,
        };
    }

    // The grocer's designed conditional bargain. If they already received the
    // medicine they needed, they honour it; otherwise they ask for it.
    if a.id.as_str() == "grocer" && wants_ledger {
        let has_medicine = a.inventory.quantity(&ItemId::new("medicine")) > 0;
        if has_medicine || rel.trust > 0.45 {
            return Action::ShareClaim {
                target: speaker.clone(),
                fact: FactId::new("f_grocer_ledger"),
                assertive: true,
            };
        }
        return Action::MakeOffer {
            target: speaker.clone(),
            give: vec![],
            want: vec![],
            condition: Some(QuestPredicate::HasItem {
                actor: a.id.clone(),
                item: ItemId::new("medicine"),
                qty: 1,
            }),
            disclose: Some(FactId::new("f_grocer_ledger")),
            note: "Bring medicine for my daughter, and I will show you the ledger.".into(),
        };
    }

    // Clue holders share only accessible secrets, and only if they trust you.
    if wants_secret && rel.trust > 0.25 {
        let candidate = a
            .beliefs
            .iter()
            .filter(|(_, b)| b.believes_true)
            .filter_map(|(fid, _)| world.fact(fid).map(|f| (fid, f)))
            .find(|(_, f)| {
                f.sensitivity != Sensitivity::InnerCircle
                    && f.category != rpg_domain::world::FactCategory::Motive
            })
            .map(|(fid, _)| fid.clone());
        if let Some(fact) = candidate {
            return Action::ShareClaim {
                target: speaker.clone(),
                fact,
                assertive: true,
            };
        }
    }

    // Default in-character line.
    let line = match a.id.as_str() {
        "grocer" => "The shop is open, but I have little to say about my suppliers.",
        "guard_captain" => "The Watch is listening. Say what you came to say.",
        "priest" => "Peace to you. What weighs on your heart?",
        "dockworker" => "Docks are busy. Best not linger.",
        "courier" => "I only carry what I'm paid to carry.",
        _ => "Hm? What do you want?",
    };
    Action::Speak {
        target: Some(speaker.clone()),
        text: line.into(),
        loudness: rpg_domain::action::Loudness::Normal,
    }
}

// ---------------------------------------------------------------------------
// NPC policy
// ---------------------------------------------------------------------------

/// How an NPC action was produced. Callers must surface this so a heuristic
/// fallback is never presented as a real model interaction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecisionSource {
    /// A real local model produced a schema-valid action.
    Model,
    /// The model was consulted, but a canonical simulation rule (a satisfied
    /// obligation) determined the final action.
    CanonicalObligation { reason: String },
    /// The model was consulted, but the NPC's designed agenda (an unmet need)
    /// determined the final action.
    CanonicalAgenda { reason: String },
    /// A real model was configured but failed; the deterministic policy answered.
    HeuristicFallback { reason: String },
    /// No real model is configured (explicit heuristic mode).
    HeuristicEngine,
}

impl DecisionSource {
    pub fn is_model(&self) -> bool {
        matches!(self, DecisionSource::Model)
    }

    pub fn label(&self) -> String {
        match self {
            DecisionSource::Model => "model".into(),
            DecisionSource::CanonicalObligation { reason } => {
                format!("canonical-obligation ({reason})")
            }
            DecisionSource::CanonicalAgenda { reason } => {
                format!("canonical-agenda ({reason})")
            }
            DecisionSource::HeuristicFallback { reason } => {
                format!("heuristic-fallback ({reason})")
            }
            DecisionSource::HeuristicEngine => "heuristic-engine".into(),
        }
    }
}

/// One NPC decision plus provenance, so the caller can validate and report it.
#[derive(Clone, Debug)]
pub struct NpcDecision {
    pub proposal: ActionProposal,
    pub source: DecisionSource,
    pub prompt: String,
    pub raw: Option<InferenceResponse>,
}

pub struct NpcPolicy;

impl NpcPolicy {
    /// Ask the engine for a decision and report how it was produced. A real
    /// model failure falls back to the deterministic policy *and says so*.
    ///
    /// This is the synchronous composition of [`NpcPolicy::request`] and
    /// [`NpcPolicy::resolve`]. Realtime clients run those two halves on
    /// different threads so the simulation never waits for the model.
    pub fn decide(
        &self,
        world: &World,
        npc: &ActorId,
        speaker: &ActorId,
        utterance: &str,
        engine: &mut dyn InferenceEngine,
        request_id: u64,
        history: &[(ActorId, String)],
    ) -> NpcDecision {
        if !engine.is_real_model() {
            return self.heuristic(world, npc, speaker, utterance, None);
        }
        let req = self.request(world, npc, speaker, utterance, request_id, history);
        let result = engine.complete(&req);
        let model_id = engine.model_id();
        self.resolve(world, npc, speaker, utterance, req, result, &model_id)
    }

    /// Build the model request for one NPC decision.
    ///
    /// The request carries only the NPC's scoped, rendered context — never a
    /// reference to the mutable world — so it can be handed to an inference
    /// worker while the simulation keeps running.
    pub fn request(
        &self,
        world: &World,
        npc: &ActorId,
        speaker: &ActorId,
        utterance: &str,
        request_id: u64,
        history: &[(ActorId, String)],
    ) -> InferenceRequest {
        InferenceRequest {
            id: request_id,
            role: InferenceRole::NpcDecision,
            system: "You are a character in a grounded social RPG. Stay in character. Never invent facts you were not told. Reply with one JSON object only.".into(),
            prompt: render_npc_context(world, npc, speaker, utterance, history),
            max_tokens: 160,
            // Tool decisions are deterministic: a small model asked at higher
            // temperature drifts between valid but unrelated actions.
            temperature: 0.0,
            seed: request_id,
            response_contract: Some(rpg_domain::contract::ACTION_CONTRACT_ID.into()),
        }
    }

    /// The deterministic policy's decision. `reason` is `Some` when a real
    /// model was configured but could not be used, so the provenance records
    /// a fallback rather than an explicit heuristic engine.
    pub fn heuristic(
        &self,
        world: &World,
        npc: &ActorId,
        speaker: &ActorId,
        utterance: &str,
        reason: Option<String>,
    ) -> NpcDecision {
        let label = reason
            .clone()
            .unwrap_or_else(|| "deterministic policy".into());
        NpcDecision {
            proposal: ActionProposal {
                actor: npc.clone(),
                action: heuristic_decision(world, npc, speaker, utterance),
                source: ProposalSource::NpcPolicy,
                rationale: label,
            },
            source: match reason {
                Some(r) => DecisionSource::HeuristicFallback { reason: r },
                None => DecisionSource::HeuristicEngine,
            },
            prompt: render_npc_context(world, npc, speaker, utterance, &[]),
            raw: None,
        }
    }

    /// Turn a model result into a decision.
    ///
    /// `world` is the world *at resolution time*, which may be later than the
    /// world `req` was rendered from: parsing, the canonical agenda and the
    /// canonical obligations are all checked against the current state, and
    /// the caller must still validate the proposal before committing it.
    #[allow(clippy::too_many_arguments)]
    pub fn resolve(
        &self,
        world: &World,
        npc: &ActorId,
        speaker: &ActorId,
        utterance: &str,
        req: InferenceRequest,
        result: Result<InferenceResponse, InferenceError>,
        model_id: &str,
    ) -> NpcDecision {
        let fallback = |reason: String| {
            let mut d = self.heuristic(world, npc, speaker, utterance, Some(reason));
            d.prompt = req.prompt.clone();
            d
        };
        let resp = match result {
            Ok(resp) => resp,
            Err(e) => return fallback(format!("inference error: {e}")),
        };
        if std::env::var("RPG_DEBUG_DECISION").is_ok() {
            eprintln!("--- decision raw ---\n{}\n--- end decision ---", resp.text);
        }
        // The answer is untrusted whatever the backend promised: check it
        // against the shared action contract before interpreting any field.
        if let Err(why) = rpg_domain::contract::validate_action_json(&resp.text) {
            return fallback(format!(
                "model output violated the action contract ({why}): {}",
                preview(&resp.text)
            ));
        }
        let Some(mut action) = parse_decision(&resp.text, Some(world), npc, speaker) else {
            return fallback(format!(
                "model output was not a valid action: {}",
                preview(&resp.text)
            ));
        };
        // Canonical agenda: an NPC with an unmet item need should not give up
        // a private fact for free. If the model did not initiate the designed
        // bargain, the policy does.
        if let Some((fact, item)) = canonical_bargain(world, npc, speaker, utterance) {
            match &mut action {
                Action::MakeOffer {
                    condition,
                    disclose,
                    ..
                } => {
                    *condition = Some(QuestPredicate::HasItem {
                        actor: npc.clone(),
                        item: item.clone(),
                        qty: 1,
                    });
                    *disclose = Some(fact.clone());
                }
                Action::ShareClaim { .. } => {}
                _ => {
                    return NpcDecision {
                        proposal: ActionProposal {
                            actor: npc.clone(),
                            action: Action::MakeOffer {
                                target: speaker.clone(),
                                give: vec![],
                                want: vec![],
                                condition: Some(QuestPredicate::HasItem {
                                    actor: npc.clone(),
                                    item: item.clone(),
                                    qty: 1,
                                }),
                                disclose: Some(fact.clone()),
                                note: format!("Bring me {item} and I will tell you what I know."),
                            },
                            source: ProposalSource::NpcPolicy,
                            rationale: format!("designed bargain ({item} for {fact})"),
                        },
                        source: DecisionSource::CanonicalAgenda {
                            reason: format!("needs {item} before sharing {fact}"),
                        },
                        prompt: req.prompt,
                        raw: Some(resp),
                    };
                }
            }
        }
        // Canonical obligation: a satisfied deal the NPC owes the player must
        // be honoured. A small model often stalls with another offer here; the
        // simulation authority does not.
        if let Some(fact) = owed_disclosure(world, npc, speaker, utterance) {
            if !matches!(action, Action::ShareClaim { .. }) {
                return NpcDecision {
                    proposal: ActionProposal {
                        actor: npc.clone(),
                        action: Action::ShareClaim {
                            target: speaker.clone(),
                            fact: fact.clone(),
                            assertive: true,
                        },
                        source: ProposalSource::NpcPolicy,
                        rationale: format!("honouring a satisfied deal ({fact})"),
                    },
                    source: DecisionSource::CanonicalObligation {
                        reason: format!("satisfied deal requires {fact}"),
                    },
                    prompt: req.prompt,
                    raw: Some(resp),
                };
            }
        }
        NpcDecision {
            proposal: ActionProposal {
                actor: npc.clone(),
                action,
                source: ProposalSource::NpcModel,
                rationale: format!("{model_id} completion_tokens={}", resp.completion_tokens),
            },
            source: DecisionSource::Model,
            prompt: req.prompt,
            raw: Some(resp),
        }
    }

    /// Backwards-compatible convenience used by older callers and tests.
    pub fn respond(
        &self,
        world: &World,
        npc: &ActorId,
        speaker: &ActorId,
        utterance: &str,
        engine: &mut dyn InferenceEngine,
        request_id: u64,
    ) -> ActionProposal {
        self.decide(world, npc, speaker, utterance, engine, request_id, &[])
            .proposal
    }
}

/// Short, sanitised preview of raw model output for provenance records.
fn preview(text: &str) -> String {
    let trimmed = text.trim();
    let mut s: String = trimmed.chars().take(120).collect();
    if trimmed.chars().count() > 120 {
        s.push_str("...");
    }
    s
}

/// A designed bargain the NPC should drive: it withholds a private fact the
/// player is asking about until it receives a needed item it lacks.
fn canonical_bargain(
    world: &World,
    npc: &ActorId,
    speaker: &ActorId,
    utterance: &str,
) -> Option<(FactId, ItemId)> {
    let a = world.actor(npc)?;
    let fact = relevant_believed_fact(world, npc, utterance)?;
    let f = world.fact(&fact)?;
    if f.sensitivity == Sensitivity::Public || world.believes(speaker, &fact) {
        return None;
    }
    for (fid, b) in &a.beliefs {
        if !b.believes_true {
            continue;
        }
        let Some(bf) = world.fact(fid) else {
            continue;
        };
        let stmt = bf.statement.to_lowercase();
        for item in world.items.keys() {
            if a.inventory.quantity(item) > 0 {
                continue;
            }
            if stmt.contains(item.as_str()) {
                return Some((fact.clone(), item.clone()));
            }
        }
    }
    None
}

/// If the NPC owes the player a satisfied obligation, return the fact it should
/// now disclose: the promised fact, else the fact the player is asking about.
fn owed_disclosure(
    world: &World,
    npc: &ActorId,
    speaker: &ActorId,
    utterance: &str,
) -> Option<FactId> {
    let debt = world.commitments.values().find(|c| {
        c.debtor == *npc && c.creditor == *speaker && c.state == CommitmentState::Satisfied
    })?;
    if let Some(fact) = &debt.promised_fact {
        if !world.believes(speaker, fact) {
            return Some(fact.clone());
        }
    }
    let fact = relevant_believed_fact(world, npc, utterance)?;
    if world.believes(speaker, &fact) {
        return None;
    }
    Some(fact)
}

/// The believed fact whose id or statement best matches the utterance. Used as
/// a fallback when an offer did not name the promised fact explicitly.
fn relevant_believed_fact(world: &World, npc: &ActorId, utterance: &str) -> Option<FactId> {
    let a = world.actor(npc)?;
    let words: Vec<String> = utterance
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 4)
        .map(str::to_string)
        .collect();
    if words.is_empty() {
        return None;
    }
    let mut best: Option<(usize, FactId)> = None;
    for (fid, b) in &a.beliefs {
        if !b.believes_true {
            continue;
        }
        let Some(f) = world.fact(fid) else {
            continue;
        };
        if f.sensitivity == Sensitivity::InnerCircle && !a.has_flag("inner_circle") {
            continue;
        }
        let id = fid.as_str().to_lowercase();
        let statement = f.statement.to_lowercase();
        let statement_words: Vec<&str> = statement
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .collect();
        let mut score = 0usize;
        for w in &words {
            if id.contains(w.as_str()) {
                score += 2;
            } else if statement_words.contains(&w.as_str()) {
                score += 1;
            }
        }
        if score > 0 && best.as_ref().map(|(s, _)| score > *s).unwrap_or(true) {
            best = Some((score, fid.clone()));
        }
    }
    best.map(|(_, f)| f)
}

// ---------------------------------------------------------------------------
// Director
// ---------------------------------------------------------------------------

pub struct DirectorAgent;

impl DirectorAgent {
    /// Propose promoting a player-invested NPC into a story role, if one is
    /// eligible and does not already have a role. The narrative controller
    /// validates the proposal; this function does not mutate anything.
    pub fn propose_role(world: &World) -> Option<NarrativeChange> {
        let player = world.actor(&ActorId::new(PLAYER))?;
        let mut best: Option<(f32, &ActorId)> = None;
        for (id, rel) in &player.relationships {
            if id.as_str() == PLAYER {
                continue;
            }
            let Some(a) = world.actor(id) else {
                continue;
            };
            if a.kind != ActorKind::Npc || !a.is_alive() {
                continue;
            }
            if a.flags.iter().any(|f| f.starts_with("role:")) {
                continue;
            }
            if a.affiliations.is_empty() {
                continue;
            }
            let score = rel.familiarity + rel.affection + rel.trust;
            if best.as_ref().map(|(s, _)| score > *s).unwrap_or(true) {
                best = Some((score, id));
            }
        }
        best.map(|(_, id)| NarrativeChange::PromoteRole {
            actor: id.clone(),
            role: "person_of_interest".into(),
        })
    }

    /// Convenience: propose and let the controller decide.
    pub fn propose_role_validated(world: &World) -> Option<(NarrativeChange, Decision)> {
        let change = Self::propose_role(world)?;
        let decision = rpg_narrative::validate_change(world, &change);
        Some((change, decision))
    }
}

// ---------------------------------------------------------------------------
// Memory retrieval
// ---------------------------------------------------------------------------

/// A tiny actor-scoped retriever over the NPC's own beliefs and relationships.
/// Production retrieval would search the persisted event store; this keeps the
/// provenance boundary intact and is enough for the prototype.
pub struct SimpleMemoryRetriever;

impl MemoryRetriever for SimpleMemoryRetriever {
    fn retrieve(&self, _actor: &ActorId, _query: &str, _limit: usize) -> Vec<RetrievedMemory> {
        // The world is not held here; retrieval over the live world is exposed
        // as a free function so callers can pass the canonical world.
        Vec::new()
    }
}

/// Retrieve the NPC's most relevant believed facts for a query.
pub fn retrieve_beliefs(
    world: &World,
    actor: &ActorId,
    query: &str,
    limit: usize,
) -> Vec<RetrievedMemory> {
    let Some(a) = world.actor(actor) else {
        return Vec::new();
    };
    let q = query.to_lowercase();
    let mut out: Vec<RetrievedMemory> = a
        .beliefs
        .iter()
        .filter(|(_, b)| b.believes_true)
        .filter_map(|(fid, b)| {
            let f = world.fact(fid)?;
            let text = f.statement.clone();
            let relevant = text.to_lowercase().contains(&q) || q.is_empty();
            let score = if relevant {
                0.5 + b.confidence * 0.5
            } else {
                b.confidence * 0.2
            };
            Some(RetrievedMemory {
                text,
                score,
                source_event: None,
                about: None,
            })
        })
        .collect();
    out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    out.truncate(limit);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rpg_domain::world::{
        Actor, ActorStatus, Belief, Capabilities, Inventory, Personality, Provenance, WorldLore,
    };
    use std::collections::{BTreeMap, BTreeSet};

    fn world_with_grocer() -> World {
        let mut w = rpg_content_for_test();
        let mut grocer = Actor {
            id: ActorId::new("grocer"),
            name: "Marta".into(),
            kind: ActorKind::Npc,
            status: ActorStatus::Alive,
            location: LocationId::new("shop"),
            pos: rpg_domain::geom::Vec2::ZERO,
            facing: 0.0,
            personality: Personality::default(),
            values: vec![],
            biography: String::new(),
            affiliations: vec![FactionId::new("merchants")],
            capabilities: Capabilities::default(),
            inventory: Inventory::default(),
            goals: vec![],
            schedule: vec![],
            relationships: BTreeMap::new(),
            beliefs: BTreeMap::new(),
            inspected: BTreeSet::new(),
            flags: BTreeSet::new(),
            secret_allegiance: None,
            in_private_conversation_with: None,
        };
        grocer.beliefs.insert(
            FactId::new("f_grocer_ledger"),
            Belief {
                fact: FactId::new("f_grocer_ledger"),
                believes_true: true,
                confidence: 0.9,
                source: Provenance::Assumed,
                acquired_tick: 0,
            },
        );
        w.actors.insert(grocer.id.clone(), grocer);
        w
    }

    // Minimal world for agent tests; the real town is in rpg_content.
    fn rpg_content_for_test() -> World {
        use rpg_domain::world::{
            Fact, FactCategory, Location, LocationKind, MysteryMode, NavGrid, StoryState,
        };
        use std::collections::BTreeMap;
        let mut w = World {
            tick: 0,
            day: 0,
            minute_of_day: 480,
            locations: BTreeMap::new(),
            actors: BTreeMap::new(),
            items: BTreeMap::new(),
            factions: BTreeMap::new(),
            facts: BTreeMap::new(),
            nav: NavGrid::new(8, 8),
            story: StoryState {
                mode: MysteryMode::FixedAnswer,
                anchors: vec![],
                checkpoints: vec![],
                fixed_answer: BTreeMap::new(),
                committed_containers: BTreeMap::new(),
            },
            commitments: BTreeMap::new(),
            offers: BTreeMap::new(),
            rumors: vec![],
            damage: BTreeMap::new(),
            location_items: BTreeMap::new(),
            flags: BTreeSet::new(),
            ledger: BTreeMap::new(),
            next_commitment_id: 1,
            next_offer_id: 1,
            next_rumor_id: 1,
            next_goal_id: 1,
            lore: WorldLore::default(),
        };
        w.locations.insert(
            LocationId::new("shop"),
            Location {
                id: LocationId::new("shop"),
                name: "Shop".into(),
                kind: LocationKind::Shop,
                bounds: rpg_domain::geom::Rect::new(0.0, 0.0, 8.0, 8.0),
                owner: None,
                sound_transmission: 1.0,
                private: false,
                tags: vec![],
            },
        );
        w.actors.insert(
            ActorId::new(PLAYER),
            Actor {
                id: ActorId::new(PLAYER),
                name: "You".into(),
                kind: ActorKind::Player,
                status: ActorStatus::Alive,
                location: LocationId::new("shop"),
                pos: rpg_domain::geom::Vec2::ZERO,
                facing: 0.0,
                personality: Personality::default(),
                values: vec![],
                biography: String::new(),
                affiliations: vec![],
                capabilities: Capabilities::default(),
                inventory: Inventory::default(),
                goals: vec![],
                schedule: vec![],
                relationships: BTreeMap::new(),
                beliefs: BTreeMap::new(),
                inspected: BTreeSet::new(),
                flags: BTreeSet::new(),
                secret_allegiance: None,
                in_private_conversation_with: None,
            },
        );
        w.facts.insert(
            FactId::new("f_grocer_ledger"),
            Fact {
                id: FactId::new("f_grocer_ledger"),
                statement: "The ledger records unusual payments.".into(),
                truth: true,
                sensitivity: Sensitivity::Private,
                category: FactCategory::Ledger,
                prerequisites: vec![],
                eligible_holders: vec![ActorId::new("grocer")],
                eligible_containers: vec![],
                committed: false,
                placed_at: None,
            },
        );
        w
    }

    #[test]
    fn scripted_engine_returns_registered_and_default() {
        let mut e = ScriptedEngine::new().with_default("{}");
        e.register(1, "{\"action\":\"idle\"}");
        let req = InferenceRequest {
            id: 1,
            role: InferenceRole::NpcDecision,
            system: String::new(),
            prompt: String::new(),
            max_tokens: 10,
            temperature: 0.0,
            seed: 0,
            response_contract: None,
        };
        assert_eq!(e.complete(&req).unwrap().text, "{\"action\":\"idle\"}");
        let req2 = InferenceRequest { id: 2, ..req };
        assert_eq!(e.complete(&req2).unwrap().text, "{}");
    }

    #[test]
    fn parse_decision_maps_offer_and_speak() {
        let speaker = ActorId::new(PLAYER);
        let a = parse_decision(
            "Sure: {\"action\":\"speak\",\"text\":\"hi\"}",
            None,
            &speaker,
            &speaker,
        )
        .unwrap();
        assert!(matches!(a, Action::Speak { .. }));
        let b = parse_decision(
            "{\"action\":\"offer\",\"give\":[\"ledger\"],\"want\":[\"medicine\"],\"note\":\"deal\"}",
            None,
            &speaker,
            &speaker,
        )
        .unwrap();
        match b {
            Action::MakeOffer {
                give,
                want,
                condition,
                ..
            } => {
                assert_eq!(give[0].item.as_str(), "ledger");
                assert_eq!(want[0].item.as_str(), "medicine");
                // No world is supplied, so the bare `want` item is grounded
                // as a canonical HasItem predicate on the proposing actor.
                match condition {
                    Some(QuestPredicate::HasItem { item, .. }) => {
                        assert_eq!(item.as_str(), "medicine");
                    }
                    other => panic!("expected medicine condition, got {other:?}"),
                }
            }
            other => panic!("expected offer, got {other:?}"),
        }
    }

    #[test]
    fn parse_offer_condition_grounds_items_and_flags() {
        let npc = ActorId::new("grocer");
        let item = parse_offer_condition("has_item:medicine", None, &npc).unwrap();
        assert!(matches!(item, QuestPredicate::HasItem { .. }));
        let flag = parse_offer_condition("flag:met_before", None, &npc).unwrap();
        assert!(matches!(flag, QuestPredicate::Flag(_)));
        // A bare token with no world to resolve it is treated as a flag.
        let bare = parse_offer_condition("medicine", None, &npc).unwrap();
        assert!(matches!(bare, QuestPredicate::Flag(_)));
        assert!(parse_offer_condition("  ", None, &npc).is_none());
    }
    #[test]
    fn grocer_asks_for_medicine_when_low_trust() {
        let w = world_with_grocer();
        let a = heuristic_decision(
            &w,
            &ActorId::new("grocer"),
            &ActorId::new(PLAYER),
            "Show me the ledger.",
        );
        match a {
            Action::MakeOffer { condition, .. } => match condition {
                Some(QuestPredicate::HasItem { item, qty, .. }) => {
                    assert_eq!(item.as_str(), "medicine");
                    assert_eq!(qty, 1);
                }
                other => panic!("expected medicine condition, got {other:?}"),
            },
            other => panic!("expected conditional offer, got {other:?}"),
        }
    }

    #[test]
    fn context_does_not_leak_inner_circle_facts() {
        let mut w = world_with_grocer();
        w.facts.insert(
            FactId::new("f_secret"),
            rpg_domain::world::Fact {
                id: FactId::new("f_secret"),
                statement: "THE LEADER IS SELENE".into(),
                truth: true,
                sensitivity: Sensitivity::InnerCircle,
                category: rpg_domain::world::FactCategory::Identity,
                prerequisites: vec![],
                eligible_holders: vec![],
                eligible_containers: vec![],
                committed: false,
                placed_at: None,
            },
        );
        w.actor_mut(&ActorId::new("grocer"))
            .unwrap()
            .beliefs
            .insert(
                FactId::new("f_secret"),
                Belief {
                    fact: FactId::new("f_secret"),
                    believes_true: true,
                    confidence: 1.0,
                    source: Provenance::Assumed,
                    acquired_tick: 0,
                },
            );
        let prompt = render_npc_context(
            &w,
            &ActorId::new("grocer"),
            &ActorId::new(PLAYER),
            "who leads it?",
            &[],
        );
        assert!(!prompt.contains("SELENE"));
    }

    #[test]
    fn scheduler_orders_by_priority_then_fifo() {
        let mut s = Scheduler::new();
        let mk = |id, role| InferenceRequest {
            id,
            role,
            system: String::new(),
            prompt: String::new(),
            max_tokens: 1,
            temperature: 0.0,
            seed: 0,
            response_contract: None,
        };
        s.push(mk(1, InferenceRole::Memory));
        s.push(mk(2, InferenceRole::NpcDecision));
        s.push(mk(3, InferenceRole::Director));
        s.push(mk(4, InferenceRole::NpcDecision));
        assert_eq!(s.pop().unwrap().id, 2);
        assert_eq!(s.pop().unwrap().id, 4);
        // Remaining P2 tasks pop FIFO: Memory(id 1) before Director(id 3).
        assert_eq!(s.pop().unwrap().id, 1);
        assert_eq!(s.pop().unwrap().id, 3);
    }

    #[test]
    fn test_prompt_includes_dialogue_history() {
        let w = world_with_grocer();
        let npc = ActorId::new("grocer");
        let player = ActorId::new(PLAYER);
        let history = vec![
            (player.clone(), "Hello there!".to_string()),
            (npc.clone(), "Good day to you.".to_string()),
            (player.clone(), "I was wondering about the ledger.".to_string()),
        ];
        let prompt = render_npc_context(
            &w,
            &npc,
            &player,
            "What do you think about it?",
            &history,
        );
        assert!(prompt.contains("Recent conversation:"));
        assert!(prompt.contains("Hello there!"));
        assert!(prompt.contains("Good day to you."));
        assert!(prompt.contains("I was wondering about the ledger."));
    }

    #[test]
    fn test_prompt_includes_world_knowledge_and_biography() {
        let mut w = world_with_grocer();
        w.actor_mut(&ActorId::new("grocer")).unwrap().biography =
            "Marta has run the shop for twenty years and knows everyone's business.".into();
        w.lore = WorldLore {
            settlement_name: "Ashwick".into(),
            geography: "A river town at the edge of the Northern Marches.".into(),
            history: "Founded as a trading post three centuries ago.".into(),
            culture: "Practical and tight-knit.".into(),
            landmarks: vec!["The Market Plaza".into()],
            economy: "Grain and timber.".into(),
            institutions: "The Watch and the Temple.".into(),
            recent_events: "A recent warehouse fire.".into(),
            everyday_knowledge: "Bread is baked daily at the plaza bakery.".into(),
        };
        let prompt = render_npc_context(
            &w,
            &ActorId::new("grocer"),
            &ActorId::new(PLAYER),
            "What do you know about this place?",
            &[],
        );
        assert!(prompt.contains("Marta has run the shop for twenty years"));
        assert!(prompt.contains("Ashwick"));
        assert!(prompt.contains("A river town at the edge of the Northern Marches"));
        assert!(prompt.contains("Founded as a trading post three centuries ago"));
        assert!(prompt.contains("The Market Plaza"));
    }
}
