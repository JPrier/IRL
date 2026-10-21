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

    out.push_str(&format!("\n{} says: \"{}\"\n", speaker, utterance));
    out.push_str(
        "Respond with ONE JSON object, no prose. Allowed shapes:\n\
         {\"action\":\"speak\",\"target\":\"<id>\",\"text\":\"...\"}\n\
         {\"action\":\"share_claim\",\"target\":\"<id>\",\"fact\":\"<fact_id>\",\"assertive\":true}\n\
         {\"action\":\"offer\",\"target\":\"<id>\",\"give\":[\"<item>\"],\"want\":[\"<item>\"],\"note\":\"...\"}\n\
         {\"action\":\"promise\",\"target\":\"<id>\",\"terms\":\"...\",\"condition\":\"<flag>\"}\n\
         {\"action\":\"threaten\",\"target\":\"<id>\",\"demand\":\"...\"}\n\
         {\"action\":\"idle\"}\n",
    );
    out
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
    #[serde(default)]
    give: Vec<String>,
    #[serde(default)]
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

fn parse_decision(text: &str, speaker: &ActorId) -> Option<Action> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end < start {
        return None;
    }
    let d: DecisionJson = serde_json::from_str(&text[start..=end]).ok()?;
    let target = d
        .target
        .as_ref()
        .map(ActorId::new)
        .or_else(|| Some(speaker.clone()));
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
        "offer" => Some(Action::MakeOffer {
            target: target?,
            give: d.give.iter().map(item).collect(),
            want: d.want.iter().map(item).collect(),
            condition: d.condition.map(QuestPredicate::Flag),
            note: d.note.unwrap_or_default(),
        }),
        "promise" => Some(Action::Promise {
            target: target?,
            terms: d.terms.unwrap_or_default(),
            predicate: QuestPredicate::Flag(d.condition.unwrap_or_default()),
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

pub struct NpcPolicy;

impl NpcPolicy {
    /// Ask the engine for a decision; fall back to the deterministic policy if
    /// the model fails or returns an invalid/unparseable action.
    pub fn respond(
        &self,
        world: &World,
        npc: &ActorId,
        speaker: &ActorId,
        utterance: &str,
        engine: &mut dyn InferenceEngine,
        request_id: u64,
    ) -> ActionProposal {
        let prompt = render_npc_context(world, npc, speaker, utterance);
        let req = InferenceRequest {
            id: request_id,
            role: InferenceRole::NpcDecision,
            system: "You are a character in a grounded social RPG. Stay in character. \
                     Never invent facts you were not told."
                .into(),
            prompt,
            max_tokens: 160,
            temperature: 0.4,
            seed: request_id,
            response_contract: Some("action-json-v1".into()),
        };
        let action = engine
            .complete(&req)
            .ok()
            .and_then(|r| parse_decision(&r.text, speaker))
            .unwrap_or_else(|| heuristic_decision(world, npc, speaker, utterance));
        ActionProposal {
            actor: npc.clone(),
            action,
            source: ProposalSource::NpcModel,
            rationale: "policy.respond".into(),
        }
    }
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
        Actor, ActorStatus, Belief, Capabilities, Inventory, Personality, Provenance,
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
        let a = parse_decision("Sure: {\"action\":\"speak\",\"text\":\"hi\"}", &speaker).unwrap();
        assert!(matches!(a, Action::Speak { .. }));
        let b = parse_decision(
            "{\"action\":\"offer\",\"give\":[\"ledger\"],\"want\":[\"medicine\"],\"note\":\"deal\"}",
            &speaker,
        )
        .unwrap();
        match b {
            Action::MakeOffer { give, want, .. } => {
                assert_eq!(give[0].item.as_str(), "ledger");
                assert_eq!(want[0].item.as_str(), "medicine");
            }
            other => panic!("expected offer, got {other:?}"),
        }
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
}
