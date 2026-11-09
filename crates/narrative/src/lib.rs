//! # rpg_narrative
//!
//! Deterministic story controller: immutable anchors, flexible checkpoints,
//! information carriers, and validation of director proposals.
//!
//! The director (an LLM, in production) only ever produces
//! [`NarrativeChange`] *proposals*. This module decides whether a proposal is
//! consistent with already-observed facts and, if so, commits it. It is the
//! story equivalent of the action validator.

use rpg_domain::action::QuestPredicate;
use rpg_domain::events::{EventKind, EventLog};
use rpg_domain::ids::*;
use rpg_domain::rng::DeterministicRng;
use rpg_domain::social::CommitmentState;
use rpg_domain::world::{
    evaluate_predicate, Actor, ActorKind, CheckpointState, ContainerCommit, ContainerKind,
    GoalKind, GoalStatus, Sensitivity, World,
};

/// A proposed narrative change. Never applied directly.
#[derive(Clone, Debug, PartialEq)]
pub enum NarrativeChange {
    /// Introduce a fact into the world as a physical/social clue.
    PlaceClue {
        fact: FactId,
        kind: ContainerKind,
        location: Option<LocationId>,
        holder: Option<ActorId>,
    },
    /// Promote an NPC into a story role (informant, witness, ...).
    PromoteRole { actor: ActorId, role: String },
    /// Activate a dormant checkpoint.
    ActivateCheckpoint { checkpoint: String },
    /// Record a narrative hook that may be realised later.
    RecordHook { description: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    Accepted,
    Rejected(String),
}

impl Decision {
    pub fn is_accepted(&self) -> bool {
        matches!(self, Decision::Accepted)
    }
}

/// The player actor, if present.
pub fn player(world: &World) -> Option<&Actor> {
    world.actor(&ActorId::new(PLAYER))
}

/// Does the player believe `fact`?
pub fn player_knows(world: &World, fact: &FactId) -> bool {
    world.believes(&ActorId::new(PLAYER), fact)
}

/// Locations the player has *not* already thoroughly inspected.
pub fn uninspected_locations(world: &World) -> Vec<LocationId> {
    let inspected = player(world)
        .map(|p| p.inspected.clone())
        .unwrap_or_default();
    world
        .locations
        .keys()
        .filter(|l| !inspected.contains(l.as_str()))
        .cloned()
        .collect()
}

/// Validate a proposed change against immutable anchors, committed facts and
/// the player's observation history. This is the consistency gate.
pub fn validate_change(world: &World, change: &NarrativeChange) -> Decision {
    match change {
        NarrativeChange::PlaceClue {
            fact,
            kind,
            location,
            holder,
        } => {
            let Some(f) = world.fact(fact) else {
                return Decision::Rejected(format!("unknown fact {fact}"));
            };
            if f.committed {
                return Decision::Rejected(format!("fact {fact} is already committed"));
            }
            // Prerequisites must be discoverable/known first.
            for pre in &f.prerequisites {
                if !player_knows(world, pre)
                    && !world.fact(pre).map(|p| p.committed).unwrap_or(false)
                {
                    return Decision::Rejected(format!(
                        "prerequisite {pre} not yet established for {fact}"
                    ));
                }
            }
            // Eligible container kinds.
            if !f.eligible_containers.is_empty() && !f.eligible_containers.contains(kind) {
                return Decision::Rejected(format!(
                    "container kind {kind:?} not eligible for {fact}"
                ));
            }
            // Location must exist and not have been already inspected.
            if let Some(loc) = location {
                if world.location(loc).is_none() {
                    return Decision::Rejected(format!("unknown location {loc}"));
                }
                let inspected = player(world)
                    .map(|p| p.inspected.contains(loc.as_str()))
                    .unwrap_or(false);
                if inspected {
                    return Decision::Rejected(format!(
                        "cannot plant {fact} in already-inspected {loc}"
                    ));
                }
            }
            // Holder must exist, be alive, and be a plausible source.
            if let Some(h) = holder {
                let Some(a) = world.actor(h) else {
                    return Decision::Rejected(format!("unknown holder {h}"));
                };
                if !a.is_alive() {
                    return Decision::Rejected(format!("holder {h} is not alive"));
                }
                if a.kind == ActorKind::Player {
                    return Decision::Rejected("cannot make the player a clue holder".into());
                }
                if !f.eligible_holders.is_empty() && !f.eligible_holders.contains(h) {
                    return Decision::Rejected(format!("{h} is not an eligible holder for {fact}"));
                }
            }
            if location.is_none() && holder.is_none() {
                return Decision::Rejected("clue needs a location or a holder".into());
            }
            Decision::Accepted
        }
        NarrativeChange::PromoteRole { actor, role } => {
            let Some(a) = world.actor(actor) else {
                return Decision::Rejected(format!("unknown actor {actor}"));
            };
            if !a.is_alive() {
                return Decision::Rejected(format!("{actor} is not alive"));
            }
            if a.kind != ActorKind::Npc {
                return Decision::Rejected(format!("{actor} is not a named NPC"));
            }
            // Role promotion must fit an established connection: the NPC needs
            // at least one affiliation or a relevant flag. We never rewrite
            // identity to force a role.
            if a.affiliations.is_empty() && a.flags.is_empty() {
                return Decision::Rejected(format!(
                    "{actor} has no established affiliation to justify role '{role}'"
                ));
            }
            if role.trim().is_empty() {
                return Decision::Rejected("empty role".into());
            }
            Decision::Accepted
        }
        NarrativeChange::ActivateCheckpoint { checkpoint } => {
            let Some(c) = world.story.checkpoints.iter().find(|c| &c.id == checkpoint) else {
                return Decision::Rejected(format!("unknown checkpoint {checkpoint}"));
            };
            if c.state != CheckpointState::Dormant {
                return Decision::Rejected(format!("checkpoint {checkpoint} is not dormant"));
            }
            if !c.requires.iter().all(|f| player_knows(world, f)) {
                return Decision::Rejected(format!(
                    "checkpoint {checkpoint} prerequisites not yet known"
                ));
            }
            Decision::Accepted
        }
        NarrativeChange::RecordHook { description } => {
            if description.trim().is_empty() {
                Decision::Rejected("empty hook".into())
            } else {
                Decision::Accepted
            }
        }
    }
}

/// Commit a validated change. Returns the decision (rejections are no-ops).
pub fn commit_change(
    world: &mut World,
    events: &mut EventLog,
    tick: u64,
    minute: u32,
    change: NarrativeChange,
) -> Decision {
    let decision = validate_change(world, &change);
    if let Decision::Rejected(reason) = &decision {
        return Decision::Rejected(reason.clone());
    }
    match change {
        NarrativeChange::PlaceClue {
            fact,
            kind,
            location,
            holder,
        } => {
            let loc = location.clone();
            if let Some(f) = world.facts.get_mut(&fact) {
                f.committed = true;
                f.placed_at = loc.clone();
            }
            world.story.committed_containers.insert(
                fact.clone(),
                ContainerCommit {
                    fact: fact.clone(),
                    kind,
                    location: loc.clone(),
                    holder: holder.clone(),
                    tick,
                },
            );
            events.push(
                tick,
                minute,
                None,
                EventKind::CluePlaced {
                    fact,
                    kind,
                    location: loc,
                    holder,
                },
                Vec::new(),
                Vec::new(),
            );
        }
        NarrativeChange::PromoteRole { actor, role } => {
            if let Some(a) = world.actor_mut(&actor) {
                a.flags.insert(format!("role:{role}"));
            }
            events.push(
                tick,
                minute,
                Some(actor),
                EventKind::CheckpointChanged {
                    checkpoint: format!("role:{role}"),
                    state: "promoted".into(),
                },
                Vec::new(),
                Vec::new(),
            );
        }
        NarrativeChange::ActivateCheckpoint { checkpoint } => {
            if let Some(c) = world
                .story
                .checkpoints
                .iter_mut()
                .find(|c| c.id == checkpoint)
            {
                c.state = CheckpointState::Active;
            }
            events.push(
                tick,
                minute,
                None,
                EventKind::CheckpointChanged {
                    checkpoint,
                    state: "active".into(),
                },
                Vec::new(),
                Vec::new(),
            );
        }
        NarrativeChange::RecordHook { description } => {
            world.set_flag(format!("hook:{description}"));
        }
    }
    Decision::Accepted
}

/// Activate every dormant checkpoint whose prerequisites the player now knows.
/// Returns the ids activated.
pub fn activate_ready_checkpoints(
    world: &mut World,
    events: &mut EventLog,
    tick: u64,
    minute: u32,
) -> Vec<String> {
    let ready: Vec<String> = world
        .story
        .checkpoints
        .iter()
        .filter(|c| c.state == CheckpointState::Dormant)
        .filter(|c| c.requires.iter().all(|f| player_knows(world, f)))
        .filter(|c| {
            c.predicate
                .as_ref()
                .map(|p| evaluate_predicate(world, p))
                .unwrap_or(true)
        })
        .map(|c| c.id.clone())
        .collect();
    for id in &ready {
        commit_change(
            world,
            events,
            tick,
            minute,
            NarrativeChange::ActivateCheckpoint {
                checkpoint: id.clone(),
            },
        );
    }
    ready
}

/// Pick the next eligible clue (an uncommitted fact from an active checkpoint)
/// and commit a placement for it in a location the player has not inspected.
///
/// Returns the fact that was placed, if any. Deterministic for a given RNG.
pub fn place_next_clue(
    world: &mut World,
    events: &mut EventLog,
    rng: &mut DeterministicRng,
    tick: u64,
    minute: u32,
) -> Option<FactId> {
    // Candidate facts come from active checkpoints.
    let mut candidates: Vec<FactId> = Vec::new();
    for cp in &world.story.checkpoints {
        if cp.state == CheckpointState::Active {
            for f in &cp.candidate_facts {
                if world.fact(f).map(|x| !x.committed).unwrap_or(false) {
                    candidates.push(f.clone());
                }
            }
        }
    }
    if candidates.is_empty() {
        return None;
    }
    // Deterministic shuffle-ish selection.
    let idx = rng.pick_index(candidates.len());
    let fact = candidates.swap_remove(idx);

    // Prefer a living eligible holder in an uninspected location.
    let uninspected = uninspected_locations(world);
    let f = world.fact(&fact).cloned()?;

    // Try holders first (people are richer clue carriers than objects).
    let mut holder_choice: Option<ActorId> = None;
    for h in &f.eligible_holders {
        if let Some(a) = world.actor(h) {
            if a.is_alive() && uninspected.contains(&a.location) {
                holder_choice = Some(h.clone());
                break;
            }
        }
    }

    let kind = f
        .eligible_containers
        .first()
        .copied()
        .unwrap_or(ContainerKind::Document);

    let location_choice = if holder_choice.is_none() {
        // Prefer a location that is not owned by the player and is uninspected.
        uninspected.first().cloned()
    } else {
        None
    };

    let change = NarrativeChange::PlaceClue {
        fact: fact.clone(),
        kind,
        location: location_choice,
        holder: holder_choice,
    };
    match commit_change(world, events, tick, minute, change) {
        Decision::Accepted => Some(fact),
        Decision::Rejected(_) => None,
    }
}

/// Count the distinct committed containers through which a fact can be
/// discovered. Used to assert the "three clue rule" for critical revelations.
pub fn independent_route_count(world: &World, fact: &FactId) -> usize {
    world
        .story
        .committed_containers
        .values()
        .filter(|c| &c.fact == fact)
        .count()
}

// ---------------------------------------------------------------------------
// Emergent quests (design §6.2/§6.3, plan Phase 5 / I-7)
//
// A quest is *recognized*, never authored: Q = (beneficiary, goal, blocking
// predicates, achievable pathways, stake, contract?). Every field is read
// from world truth — existing goals, content-authored bargains, physical
// item placement and accepted commitments/deals. The recognizer invents
// nothing: if a needed resource exists nowhere in the world, no pathway
// exists and no opportunity is surfaced. The director is constrained: it
// may only surface *currently recognized* opportunities, it commits no
// world truth, and it cannot treat a proposed quest as an established
// contract (a contract is reported only when an accepted commitment or a
// non-terminal deal actually matches the goal predicate).
//
// Ordering is fully deterministic (BTreeMap iteration, no RNG) so quests
// participate in replay; surfacing appends a `DirectorProposed` provenance
// event and mutates nothing else, keeping save/load and checksum behavior.
// ---------------------------------------------------------------------------

/// Bounded work: at most this many pathways are reported per opportunity
/// (truncated in deterministic scan order).
pub const MAX_QUEST_PATHWAYS: usize = 8;
/// Bounded work: at most this many opportunities per recognition pass
/// (truncated in deterministic actor/goal order).
pub const MAX_QUEST_OPPORTUNITIES: usize = 16;

/// Where a recognized quest's goal came from. Both are real, persisted
/// actor state — the recognizer never fabricates a goal.
#[derive(Clone, Debug, PartialEq)]
pub enum QuestGoalSource {
    /// A persistent actor goal carrying a stable satisfaction predicate
    /// (`Goal::predicate`) that is currently unmet.
    Goal { id: u64 },
    /// The content-authored conditional bargain (`Actor::bargain`): the
    /// typed world condition the actor needs satisfied in exchange for a
    /// promised disclosure.
    Bargain,
}

/// An achievable pathway *observed in the world*. Both variants cite where
/// the needed resource physically already is; the director may not teleport
/// resources, so surfacing can only ever reference real placement. Ordered
/// for deterministic canonical sort (placement before held, then by ids).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum QuestPathway {
    /// The needed item is physically present at a location.
    ItemAvailableAt {
        item: ItemId,
        location: LocationId,
        qty: u32,
    },
    /// A living actor other than the beneficiary holds the needed item.
    HeldByActor {
        item: ItemId,
        holder: ActorId,
        qty: u32,
    },
}

/// What is at stake for the beneficiary. Read from the goal/bargain and any
/// matching accepted contract — never invented.
#[derive(Clone, Debug, PartialEq)]
pub struct QuestStake {
    /// The beneficiary's goal class, when the quest derives from a `Goal`.
    pub goal_kind: Option<GoalKind>,
    /// The beneficiary's goal priority, when the quest derives from a `Goal`.
    pub goal_priority: Option<f32>,
    /// The fact the beneficiary promises to disclose once the goal state
    /// holds (the bargain's stake, or a matching commitment's promise).
    pub promised_fact: Option<FactId>,
}

/// An established contract that already binds this quest's goal state.
/// Only ever populated from a real accepted commitment or a non-terminal
/// deal; a *proposed* quest is never reported as contracted (T5.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuestContract {
    /// An accepted, still-standing commitment whose predicate matches.
    Commitment(CommitmentId),
    /// A non-terminal deal whose condition matches.
    Deal(DealId),
}

/// A recognized quest opportunity: design §6.2
/// `Q = (beneficiary, goal, blocking predicates, achievable pathways,
/// stake, contract?)`.
#[derive(Clone, Debug, PartialEq)]
pub struct QuestOpportunity {
    pub beneficiary: ActorId,
    pub goal: QuestGoalSource,
    /// Human-readable goal description taken from the actor's own goal, or
    /// composed mechanically from the bargain's typed condition.
    pub description: String,
    /// The intended state. Quest resolution is *this predicate's
    /// satisfaction* — never one authored solution (T5.2 substitute
    /// satisfaction).
    pub goal_predicate: QuestPredicate,
    /// The currently-unsatisfied blocking conditions (sub-predicates that
    /// do not yet hold in the world).
    pub blockers: Vec<QuestPredicate>,
    /// Achievable pathways observed in world truth (bounded, deterministic
    /// order). Privileged observation — never presented to an actor as a
    /// witnessed event.
    pub pathways: Vec<QuestPathway>,
    pub stake: QuestStake,
    pub contract: Option<QuestContract>,
}

/// A typed factual assertion a director surface makes. Claims are the
/// anchor-validation surface (T5.4): each must reference an *existing,
/// committed* fact and assert its true polarity. Anything else is rejected
/// as invented witnessed evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuestClaim {
    pub fact: FactId,
    pub polarity: bool,
}

/// A director proposal (design §6.3/§8). Surfacing is presentation only.
#[derive(Clone, Debug, PartialEq)]
pub enum DirectorProposal {
    /// Surface a recognized quest opportunity to an actor (e.g. through
    /// dialogue). Commits no world truth: no facts, goals, deals, flags or
    /// story state change — only the `DirectorProposed` provenance event.
    SurfaceQuest {
        opportunity: QuestOpportunity,
        presented_to: ActorId,
        claims: Vec<QuestClaim>,
    },
}

/// The epistemically *permitted* actor-facing projection of a surfaced
/// quest: who needs what, what they promise, whether a real contract
/// exists, and whether a pathway is known to exist at all. It deliberately
/// carries **no** pathway locations and no world-truth details — those are
/// privileged recognizer output, not things an actor was shown
/// ([`QuestOpportunity`] is the privileged form; keep it out of dialogue).
#[derive(Clone, Debug, PartialEq)]
pub struct QuestSurface {
    pub beneficiary: ActorId,
    pub beneficiary_name: String,
    pub description: String,
    pub promised_fact: Option<FactId>,
    /// True only when an accepted commitment or non-terminal deal actually
    /// binds this goal state (never for a merely proposed quest).
    pub has_contract: bool,
    /// Whether any genuine pathway exists — existence only, no locations.
    pub has_known_pathway: bool,
}

impl QuestSurface {
    /// The permitted projection of a recognized opportunity.
    pub fn of(world: &World, opportunity: &QuestOpportunity) -> QuestSurface {
        QuestSurface {
            beneficiary: opportunity.beneficiary.clone(),
            beneficiary_name: world
                .actor(&opportunity.beneficiary)
                .map(|a| a.name.clone())
                .unwrap_or_default(),
            description: opportunity.description.clone(),
            promised_fact: opportunity.stake.promised_fact.clone(),
            has_contract: opportunity.contract.is_some(),
            has_known_pathway: !opportunity.pathways.is_empty(),
        }
    }
}

/// Blocking conditions: the leaves of `predicate` that do not currently
/// hold. For `All`/`Any` children are walked individually; `Not(inner)` is
/// reported as written (the negated condition itself is blocking). Every
/// reported blocker is verifiably false right now — the recognizer cannot
/// invent an obstacle.
fn unsatisfied_leaves(world: &World, predicate: &QuestPredicate, out: &mut Vec<QuestPredicate>) {
    if evaluate_predicate(world, predicate) {
        return;
    }
    match predicate {
        QuestPredicate::All(children) | QuestPredicate::Any(children) => {
            for c in children {
                unsatisfied_leaves(world, c, out);
            }
        }
        _ => out.push(predicate.clone()),
    }
}

/// Mechanically describe a bargain condition from world data (item names,
/// quantities) — no authored prose, no actor literals.
fn describe_condition(world: &World, predicate: &QuestPredicate) -> String {
    match predicate {
        QuestPredicate::HasItem { actor, item, qty } => {
            let who = world
                .actor(actor)
                .map(|a| a.name.as_str())
                .unwrap_or(actor.as_str());
            let what = world
                .items
                .get(item)
                .map(|i| i.name.as_str())
                .unwrap_or(item.as_str());
            if *qty == 1 {
                format!("{who} needs {what}")
            } else {
                format!("{who} needs {what} ×{qty}")
            }
        }
        other => format!("{other:?}"),
    }
}

/// Pathways for one unsatisfied `HasItem` blocker, derived from where the
/// item physically is right now: static location placement first, then
/// living actors' inventories (beneficiary excluded — their lack of the
/// item is the blocker). Deterministic scan order; bounded by
/// [`MAX_QUEST_PATHWAYS`].
fn item_pathways(
    world: &World,
    beneficiary: &ActorId,
    item: &ItemId,
    qty: u32,
) -> Vec<QuestPathway> {
    let mut out = Vec::new();
    for (loc, stacks) in &world.location_items {
        for stack in stacks {
            if &stack.item == item && stack.qty > 0 {
                out.push(QuestPathway::ItemAvailableAt {
                    item: item.clone(),
                    location: loc.clone(),
                    qty: stack.qty.min(qty),
                });
            }
        }
    }
    for (id, actor) in &world.actors {
        if id == beneficiary || !actor.is_alive() {
            continue;
        }
        let held = actor.inventory.quantity(item);
        if held > 0 {
            out.push(QuestPathway::HeldByActor {
                item: item.clone(),
                holder: id.clone(),
                qty: held.min(qty),
            });
        }
    }
    out.truncate(MAX_QUEST_PATHWAYS);
    out
}

/// The established contract binding `predicate`, if any: an accepted
/// (still pending) commitment or a non-terminal deal whose condition is
/// exactly this predicate. Commitments are scanned before deals; both maps
/// are ordered, so the earliest matching id wins deterministically.
fn matching_contract(world: &World, predicate: &QuestPredicate) -> Option<QuestContract> {
    for (id, cm) in &world.commitments {
        if cm.state == CommitmentState::Pending && &cm.predicate == predicate {
            return Some(QuestContract::Commitment(*id));
        }
    }
    for (id, d) in &world.deals {
        if !d.is_terminal() && &d.condition == predicate {
            return Some(QuestContract::Deal(*id));
        }
    }
    None
}

/// Build an opportunity from a blocked goal/bargain predicate, or `None`
/// when no achievable pathway exists in the world (the recognizer never
/// invents resources, so a quest with no real lead is not an opportunity).
fn build_opportunity(
    world: &World,
    beneficiary: ActorId,
    goal: QuestGoalSource,
    description: String,
    predicate: QuestPredicate,
    stake: QuestStake,
) -> Option<QuestOpportunity> {
    let mut blockers = Vec::new();
    unsatisfied_leaves(world, &predicate, &mut blockers);
    let mut pathways = Vec::new();
    for blocker in &blockers {
        if let QuestPredicate::HasItem {
            actor, item, qty, ..
        } = blocker
        {
            if actor == &beneficiary {
                pathways.extend(item_pathways(world, &beneficiary, item, *qty));
            }
        }
    }
    if pathways.is_empty() {
        return None;
    }
    pathways.sort();
    pathways.dedup();
    pathways.truncate(MAX_QUEST_PATHWAYS);
    let contract = matching_contract(world, &predicate);
    Some(QuestOpportunity {
        beneficiary,
        goal,
        description,
        goal_predicate: predicate,
        blockers,
        pathways,
        stake,
        contract,
    })
}

/// Recognize quest opportunities from real world state (design §6.2).
///
/// Deterministic: actors in map order, each actor's goals in list order,
/// then the actor's content-authored bargain. The pass is bounded by
/// [`MAX_QUEST_OPPORTUNITIES`]. Recognition reads only:
///
/// * active goals with unmet stable predicates ([`Goal::predicate`]),
/// * content-authored bargains whose typed condition is unmet (with the
///   same knowledge guards the bargain pathway uses: the actor must
///   actually believe the promised fact, the fact must not be public, and
///   the player must not already know it — no stake otherwise),
/// * physical item placement and living inventories for pathways,
/// * accepted commitments / non-terminal deals for `contract?`.
///
/// The recognizer is a reader: it never mutates the world, never marks a
/// goal satisfied, never creates a contract, and its output contains no
/// actor-id branches.
pub fn recognize_quests(world: &World) -> Vec<QuestOpportunity> {
    let mut out: Vec<QuestOpportunity> = Vec::new();
    'actors: for (id, actor) in &world.actors {
        if actor.kind != ActorKind::Npc || !actor.is_alive() {
            continue;
        }
        for g in &actor.goals {
            if g.status != GoalStatus::Active {
                continue;
            }
            let Some(predicate) = &g.predicate else {
                continue;
            };
            if evaluate_predicate(world, predicate) {
                continue; // already satisfied: not a quest
            }
            let stake = QuestStake {
                goal_kind: Some(g.kind),
                goal_priority: Some(g.priority),
                promised_fact: matching_commitment_promise(world, predicate),
            };
            if let Some(q) = build_opportunity(
                world,
                id.clone(),
                QuestGoalSource::Goal { id: g.id },
                g.description.clone(),
                predicate.clone(),
                stake,
            ) {
                out.push(q);
                if out.len() >= MAX_QUEST_OPPORTUNITIES {
                    break 'actors;
                }
            }
        }
        // Content-authored bargain: the typed world condition this actor
        // needs satisfied in exchange for a promised disclosure.
        if let Some(b) = &actor.bargain {
            let disclosed = &b.disclose;
            let f_ok = world
                .fact(disclosed)
                .map(|f| f.sensitivity != Sensitivity::Public)
                .unwrap_or(false)
                && world.believes(id, disclosed)
                && !world
                    .actors
                    .values()
                    .any(|a| a.kind == ActorKind::Player && world.believes(&a.id, disclosed));
            if f_ok && !evaluate_predicate(world, &b.condition) {
                let stake = QuestStake {
                    goal_kind: None,
                    goal_priority: None,
                    promised_fact: Some(disclosed.clone()),
                };
                if let Some(q) = build_opportunity(
                    world,
                    id.clone(),
                    QuestGoalSource::Bargain,
                    describe_condition(world, &b.condition),
                    b.condition.clone(),
                    stake,
                ) {
                    out.push(q);
                    if out.len() >= MAX_QUEST_OPPORTUNITIES {
                        break 'actors;
                    }
                }
            }
        }
    }
    out
}

/// The promised fact of an accepted commitment matching `predicate`, if any.
fn matching_commitment_promise(world: &World, predicate: &QuestPredicate) -> Option<FactId> {
    world
        .commitments
        .values()
        .find(|cm| cm.state == CommitmentState::Pending && &cm.predicate == predicate)
        .and_then(|cm| cm.promised_fact.clone())
}

/// Quest resolution = satisfaction of the goal predicate — nothing else.
/// Any route that makes the intended state true resolves the quest
/// (design §6.2 substitute satisfaction: never require one authored
/// solution). This is the *recognition* of resolution only; the narrative
/// layer has no authority to write goal status, deal state or world truth.
pub fn quest_satisfied(world: &World, opportunity: &QuestOpportunity) -> bool {
    evaluate_predicate(world, &opportunity.goal_predicate)
}

/// Validate a constrained director proposal (design §6.3/§8):
///
/// 1. **Recognition gate** — the proposal's opportunity must be
///    *structurally identical* to a currently recognized opportunity. The
///    director cannot invent a beneficiary, goal, predicate, stake,
///    pathway or contract: any tampered field fails the match.
/// 2. **Presentation target** — must exist and be alive.
/// 3. **Claims (anchor validation, T5.4)** — every claim must reference an
///    existing *committed* fact and assert its true polarity. Claims on
///    unknown facts (invention) or uncommitted facts (invented witnessed
///    evidence) and inverted polarities (retroactive contradiction of
///    committed truth / authored anchors, design §9.1.7) are rejected.
pub fn validate_director_proposal(world: &World, proposal: &DirectorProposal) -> Decision {
    let DirectorProposal::SurfaceQuest {
        opportunity,
        presented_to,
        claims,
    } = proposal;
    let Some(target) = world.actor(presented_to) else {
        return Decision::Rejected(format!("unknown presentation target {presented_to}"));
    };
    if !target.is_alive() {
        return Decision::Rejected(format!("presentation target {presented_to} is not alive"));
    }
    let recognized = recognize_quests(world);
    if !recognized.iter().any(|q| q == opportunity) {
        return Decision::Rejected(
            "opportunity is not currently recognized — the director cannot invent \
             beneficiaries, goals, pathways, stakes or contracts"
                .into(),
        );
    }
    for claim in claims {
        let Some(f) = world.fact(&claim.fact) else {
            return Decision::Rejected(format!(
                "claim references unknown fact {} — the director cannot invent facts",
                claim.fact
            ));
        };
        if !f.committed {
            return Decision::Rejected(format!(
                "claim on uncommitted fact {} would invent witnessed evidence",
                claim.fact
            ));
        }
        if claim.polarity != f.truth {
            return Decision::Rejected(format!(
                "claim contradicts the committed truth of {} (no retroactive \
                 narrative contradiction)",
                claim.fact
            ));
        }
    }
    Decision::Accepted
}

/// Commit a validated director proposal. Surfacing appends only the
/// `DirectorProposed` provenance event (a record that the presentation
/// happened, like a speech event — not a mutation of world truth); it
/// writes no facts, goals, deals, flags or story state. Rejections are
/// no-ops.
pub fn commit_director_proposal(
    world: &World,
    events: &mut EventLog,
    tick: u64,
    minute: u32,
    proposal: DirectorProposal,
) -> Decision {
    let decision = validate_director_proposal(world, &proposal);
    if let Decision::Rejected(reason) = &decision {
        return Decision::Rejected(reason.clone());
    }
    let DirectorProposal::SurfaceQuest { opportunity, .. } = &proposal;
    events.push(
        tick,
        minute,
        None,
        EventKind::DirectorProposed {
            proposal: format!(
                "surface_quest: {} — {}",
                opportunity.beneficiary, opportunity.description
            ),
            accepted: true,
            reason: String::new(),
        },
        Vec::new(),
        Vec::new(),
    );
    Decision::Accepted
}

#[cfg(test)]
mod tests {
    use super::*;
    use rpg_domain::geom::{Rect, Vec2};
    use rpg_domain::world::{
        Actor, ActorKind, ActorStatus, Capabilities, Fact, FactCategory, Location, LocationKind,
        MysteryMode, NavGrid, Personality, Sensitivity, StoryAnchor, StoryState, WorldLore,
    };
    use std::collections::{BTreeMap, BTreeSet};

    fn actor(id: &str, kind: ActorKind) -> Actor {
        Actor {
            id: ActorId::new(id),
            name: id.into(),
            kind,
            status: ActorStatus::Alive,
            location: LocationId::new("plaza"),
            pos: Vec2::ZERO,
            facing: 0.0,
            personality: Personality::default(),
            values: vec![],
            biography: String::new(),
            private_biography: String::new(),
            affiliations: vec![],
            capabilities: Capabilities::default(),
            inventory: Default::default(),
            goals: vec![],
            schedule: vec![],
            relationships: BTreeMap::new(),
            beliefs: BTreeMap::new(),
            inspected: BTreeSet::new(),
            flags: BTreeSet::new(),
            secret_allegiance: None,
            in_private_conversation_with: None,
            bargain: None,
            needs: Default::default(),
            emotions: Default::default(),
            rel_memory: BTreeMap::new(),
        }
    }

    fn base_world() -> World {
        let mut w = World {
            tick: 0,
            day: 0,
            minute_of_day: 480,
            locations: BTreeMap::new(),
            actors: BTreeMap::new(),
            items: BTreeMap::new(),
            factions: BTreeMap::new(),
            facts: BTreeMap::new(),
            nav: NavGrid::new(16, 16),
            story: StoryState {
                mode: MysteryMode::FixedAnswer,
                anchors: vec![StoryAnchor {
                    id: "mafia_exists".into(),
                    statement: "A mafia exists".into(),
                    immutable: true,
                }],
                checkpoints: vec![],
                fixed_answer: BTreeMap::new(),
                committed_containers: BTreeMap::new(),
            },
            commitments: BTreeMap::new(),
            offers: BTreeMap::new(),
            rumors: Vec::new(),
            damage: BTreeMap::new(),
            damage_facts: BTreeMap::new(),
            location_items: BTreeMap::new(),
            flags: BTreeSet::new(),
            ledger: BTreeMap::new(),
            next_commitment_id: 1,
            next_offer_id: 1,
            deals: BTreeMap::new(),
            next_deal_id: 1,
            next_rumor_id: 1,
            next_goal_id: 1,
            epistemics: Default::default(),
            lore: WorldLore::default(),
        };
        w.locations.insert(
            LocationId::new("plaza"),
            Location {
                id: LocationId::new("plaza"),
                name: "Plaza".into(),
                kind: LocationKind::Plaza,
                bounds: Rect::new(0.0, 0.0, 8.0, 8.0),
                owner: None,
                sound_transmission: 1.2,
                private: false,
                tags: vec![],
            },
        );
        w.locations.insert(
            LocationId::new("warehouse"),
            Location {
                id: LocationId::new("warehouse"),
                name: "Warehouse".into(),
                kind: LocationKind::Warehouse,
                bounds: Rect::new(8.0, 0.0, 8.0, 8.0),
                owner: None,
                sound_transmission: 0.8,
                private: false,
                tags: vec![],
            },
        );
        w.actors
            .insert(ActorId::new(PLAYER), actor(PLAYER, ActorKind::Player));
        w.actors
            .insert(ActorId::new("clerk"), actor("clerk", ActorKind::Npc));
        w
    }

    fn fact(id: &str, holder: Option<&str>) -> Fact {
        Fact {
            id: FactId::new(id),
            statement: format!("statement {id}"),
            truth: true,
            sensitivity: Sensitivity::Secret,
            category: FactCategory::Shipment,
            prerequisites: vec![],
            eligible_holders: holder.map(|h| vec![ActorId::new(h)]).unwrap_or_default(),
            eligible_containers: vec![ContainerKind::Testimony, ContainerKind::Document],
            committed: false,
            placed_at: None,
        }
    }

    #[test]
    fn cannot_plant_clue_in_inspected_location() {
        let mut w = base_world();
        w.facts.insert(FactId::new("f1"), fact("f1", None));
        w.actor_mut(&ActorId::new(PLAYER))
            .unwrap()
            .inspected
            .insert("warehouse".into());
        let change = NarrativeChange::PlaceClue {
            fact: FactId::new("f1"),
            kind: ContainerKind::Document,
            location: Some(LocationId::new("warehouse")),
            holder: None,
        };
        assert!(matches!(
            validate_change(&w, &change),
            Decision::Rejected(_)
        ));
    }

    #[test]
    fn placing_in_fresh_location_is_allowed_and_commits() {
        let mut w = base_world();
        w.facts.insert(FactId::new("f1"), fact("f1", None));
        let mut events = EventLog::new();
        let change = NarrativeChange::PlaceClue {
            fact: FactId::new("f1"),
            kind: ContainerKind::Document,
            location: Some(LocationId::new("warehouse")),
            holder: None,
        };
        assert!(commit_change(&mut w, &mut events, 1, 480, change).is_accepted());
        assert!(w.fact(&FactId::new("f1")).unwrap().committed);
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn role_promotion_requires_affiliation() {
        let mut w = base_world();
        let change = NarrativeChange::PromoteRole {
            actor: ActorId::new("clerk"),
            role: "informant".into(),
        };
        assert!(matches!(
            validate_change(&w, &change),
            Decision::Rejected(_)
        ));
        w.actor_mut(&ActorId::new("clerk"))
            .unwrap()
            .affiliations
            .push(FactionId::new("merchants_guild"));
        assert!(validate_change(&w, &change).is_accepted());
    }

    #[test]
    fn checkpoint_activates_when_prerequisites_known() {
        let mut w = base_world();
        w.facts.insert(FactId::new("lead"), fact("lead", None));
        w.story.checkpoints.push(rpg_domain::world::Checkpoint {
            id: "cp1".into(),
            description: "first lead".into(),
            requires: vec![FactId::new("lead")],
            predicate: None,
            state: CheckpointState::Dormant,
            candidate_facts: vec![FactId::new("f1")],
        });
        let mut events = EventLog::new();
        assert!(activate_ready_checkpoints(&mut w, &mut events, 1, 480).is_empty());
        w.actor_mut(&ActorId::new(PLAYER)).unwrap().beliefs.insert(
            FactId::new("lead"),
            rpg_domain::world::Belief {
                fact: FactId::new("lead"),
                believes_true: true,
                confidence: 1.0,
                source: rpg_domain::world::Provenance::Assumed,
                acquired_tick: 0,
            },
        );
        let activated = activate_ready_checkpoints(&mut w, &mut events, 2, 481);
        assert_eq!(activated, vec!["cp1".to_string()]);
    }
}
