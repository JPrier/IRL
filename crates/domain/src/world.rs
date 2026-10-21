//! Canonical world data model.
//!
//! Everything here is plain serializable data. The *simulation* crate owns the
//! logic that mutates it; this crate only defines what state *is*. This keeps
//! the domain independent of Bevy, model runtimes and storage.

use crate::geom::{GridPos, Rect, Vec2};
use crate::ids::*;
use crate::social::{Commitment, Offer, Rumor};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

// ---------------------------------------------------------------------------
// Actors
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActorKind {
    Player,
    /// Individually modelled, persistent named character.
    Npc,
    /// Cheap routine actor; not eligible for story roles by default.
    Background,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActorStatus {
    Alive,
    Dead,
    Departed,
}

/// Big-five-ish personality plus a few game-relevant dials. Values are 0..1.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Personality {
    pub openness: f32,
    pub conscientiousness: f32,
    pub extraversion: f32,
    pub agreeableness: f32,
    pub neuroticism: f32,
    /// How readily the actor takes risks.
    pub risk_tolerance: f32,
    /// How strongly the actor reciprocates kindness / cruelty.
    pub reciprocity: f32,
    /// How strongly the actor weighs its own safety.
    pub self_preservation: f32,
    /// Freeform traits, e.g. "greedy", "devout", "protective".
    pub tags: Vec<String>,
}

impl Default for Personality {
    fn default() -> Self {
        Self {
            openness: 0.5,
            conscientiousness: 0.5,
            extraversion: 0.5,
            agreeableness: 0.5,
            neuroticism: 0.5,
            risk_tolerance: 0.5,
            reciprocity: 0.5,
            self_preservation: 0.5,
            tags: Vec::new(),
        }
    }
}

impl Personality {
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }
}

/// Multi-dimensional relationship from the perspective of the owning actor.
/// All fields are in `-1.0 ..= 1.0` unless noted, and must be clamped by the
/// mutation helpers in `social`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Relationship {
    /// Belief that the other will act in good faith.
    pub trust: f32,
    /// Warmth / fondness.
    pub affection: f32,
    /// Esteem for competence or character.
    pub respect: f32,
    /// Fear of the other.
    pub fear: f32,
    /// Active anger toward the other.
    pub anger: f32,
    /// Grief / sorrow associated with the other.
    pub sadness: f32,
    /// Desire for what the other can offer.
    pub temptation: f32,
    /// Belief the other may be hiding something or acting against interests.
    pub suspicion: f32,
    /// How much the owner feels they owe the other (0..1).
    pub obligation: f32,
    /// How well the owner knows the other (0..1).
    pub familiarity: f32,
}

impl Default for Relationship {
    fn default() -> Self {
        Self {
            trust: 0.0,
            affection: 0.0,
            respect: 0.0,
            fear: 0.0,
            anger: 0.0,
            sadness: 0.0,
            temptation: 0.0,
            suspicion: 0.0,
            obligation: 0.0,
            familiarity: 0.0,
        }
    }
}

impl Relationship {
    pub fn clamp_all(&mut self) {
        for v in [
            &mut self.trust,
            &mut self.affection,
            &mut self.respect,
            &mut self.fear,
            &mut self.anger,
            &mut self.sadness,
            &mut self.temptation,
            &mut self.suspicion,
        ] {
            *v = v.clamp(-1.0, 1.0);
        }
        self.obligation = self.obligation.clamp(0.0, 1.0);
        self.familiarity = self.familiarity.clamp(0.0, 1.0);
    }

    /// A single scalar used only for coarse ordering, never for gameplay.
    pub fn warmth(&self) -> f32 {
        self.trust * 0.4 + self.affection * 0.3 + self.respect * 0.2 - self.fear * 0.1
    }
}

/// What an actor is currently trying to achieve.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Goal {
    pub id: u64,
    pub description: String,
    pub kind: GoalKind,
    /// Higher runs first when the actor must choose.
    pub priority: f32,
    pub status: GoalStatus,
    /// Optional stable predicate that marks the goal satisfied.
    #[serde(default)]
    pub predicate: Option<crate::action::QuestPredicate>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoalKind {
    Survival,
    Wealth,
    Safety,
    Affection,
    Status,
    Duty,
    Ideology,
    Ambition,
    Protect,
    Revenge,
    Investigate,
    Conceal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoalStatus {
    Active,
    Satisfied,
    Abandoned,
    Failed,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ScheduleEntry {
    /// Inclusive start minute-of-day (0..1440).
    pub start_min: u32,
    /// Exclusive end minute-of-day (0..1440).
    pub end_min: u32,
    pub location: LocationId,
    pub activity: String,
    /// Whether this activity is perceptible/useful to spies.
    #[serde(default)]
    pub public: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ItemStack {
    pub item: ItemId,
    pub qty: u32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Inventory {
    pub items: Vec<ItemStack>,
    pub money: i64,
}

impl Inventory {
    pub fn quantity(&self, item: &ItemId) -> u32 {
        self.items
            .iter()
            .find(|s| &s.item == item)
            .map(|s| s.qty)
            .unwrap_or(0)
    }

    pub fn add(&mut self, item: ItemId, qty: u32) {
        if let Some(s) = self.items.iter_mut().find(|s| s.item == item) {
            s.qty += qty;
        } else {
            self.items.push(ItemStack { item, qty });
        }
    }

    /// Remove up to `qty`; returns how many were actually removed.
    pub fn remove(&mut self, item: &ItemId, qty: u32) -> u32 {
        let mut removed = 0;
        if let Some(pos) = self.items.iter().position(|s| &s.item == item) {
            let take = self.items[pos].qty.min(qty);
            self.items[pos].qty -= take;
            removed = take;
            if self.items[pos].qty == 0 {
                self.items.remove(pos);
            }
        }
        removed
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Actor {
    pub id: ActorId,
    pub name: String,
    pub kind: ActorKind,
    pub status: ActorStatus,
    pub location: LocationId,
    pub pos: Vec2,
    /// Facing in radians; 0 = +x.
    pub facing: f32,
    pub personality: Personality,
    pub values: Vec<String>,
    pub affiliations: Vec<FactionId>,
    pub capabilities: Capabilities,
    pub inventory: Inventory,
    pub goals: Vec<Goal>,
    pub schedule: Vec<ScheduleEntry>,
    pub relationships: BTreeMap<ActorId, Relationship>,
    /// Facts the actor believes to be true (whether or not they are).
    pub beliefs: BTreeMap<FactId, Belief>,
    /// Items/locations the actor has already inspected (for clue placement).
    pub inspected: BTreeSet<String>,
    /// Free-form flags used by policies, e.g. "is_mafia", "is_spy".
    pub flags: BTreeSet<String>,
    /// Optional faction the actor secretly serves.
    #[serde(default)]
    pub secret_allegiance: Option<FactionId>,
    /// Whether the actor is currently in a private conversation.
    #[serde(default)]
    pub in_private_conversation_with: Option<ActorId>,
}

impl Actor {
    pub fn is_alive(&self) -> bool {
        self.status == ActorStatus::Alive
    }

    pub fn has_flag(&self, f: &str) -> bool {
        self.flags.contains(f)
    }

    pub fn relationship(&self, other: &ActorId) -> Relationship {
        self.relationships.get(other).cloned().unwrap_or_default()
    }

    pub fn relationship_mut(&mut self, other: &ActorId) -> &mut Relationship {
        self.relationships.entry(other.clone()).or_default()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Capabilities {
    pub can_trade: bool,
    pub can_arrest: bool,
    pub can_repair: bool,
    pub can_hire: bool,
    pub can_allocate_budget: bool,
    /// Movement speed in world units per simulation second.
    pub speed: f32,
}

impl Default for Capabilities {
    fn default() -> Self {
        Self {
            can_trade: false,
            can_arrest: false,
            can_repair: false,
            can_hire: false,
            can_allocate_budget: false,
            speed: 1.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Locations & navigation
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LocationKind {
    Shop,
    Backroom,
    Warehouse,
    GuardOffice,
    Plaza,
    Street,
    Alley,
    Residence,
    FactionHall,
    Office,
    Temple,
    Tavern,
    Other,
}

/// A tile on the navigation grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tile {
    Floor,
    Wall,
    /// Walkable, but blocks line of sight weakly (e.g. a curtain). Kept simple:
    /// opaque when `Opaque`, otherwise see-through.
    Opaque,
}

impl Tile {
    pub fn walkable(self) -> bool {
        matches!(self, Tile::Floor)
    }
    pub fn blocks_sight(self) -> bool {
        matches!(self, Tile::Wall | Tile::Opaque)
    }
    pub fn blocks_sound(self) -> bool {
        matches!(self, Tile::Wall)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Location {
    pub id: LocationId,
    pub name: String,
    pub kind: LocationKind,
    /// Area on the navigation grid, in world units.
    pub bounds: Rect,
    /// Faction that owns/controls the space, if any.
    pub owner: Option<FactionId>,
    /// Acoustic privacy multiplier: lower = sound carries less far.
    #[serde(default = "one")]
    pub sound_transmission: f32,
    /// Whether this counts as a "quiet/private" space for privacy affordances.
    #[serde(default)]
    pub private: bool,
    pub tags: Vec<String>,
}

fn one() -> f32 {
    1.0
}

impl Location {
    pub fn contains(&self, p: Vec2) -> bool {
        self.bounds.contains(p)
    }
    pub fn center(&self) -> Vec2 {
        self.bounds.center()
    }
}

/// The tile grid for the whole town.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NavGrid {
    pub width: i32,
    pub height: i32,
    pub tiles: Vec<Tile>,
}

impl NavGrid {
    pub fn new(width: i32, height: i32) -> Self {
        Self {
            width,
            height,
            tiles: vec![Tile::Floor; (width * height) as usize],
        }
    }

    pub fn in_bounds(&self, g: GridPos) -> bool {
        g.x >= 0 && g.y >= 0 && g.x < self.width && g.y < self.height
    }

    pub fn tile(&self, g: GridPos) -> Option<Tile> {
        if self.in_bounds(g) {
            Some(self.tiles[(g.y * self.width + g.x) as usize])
        } else {
            None
        }
    }

    pub fn set(&mut self, g: GridPos, t: Tile) {
        if self.in_bounds(g) {
            self.tiles[(g.y * self.width + g.x) as usize] = t;
        }
    }

    pub fn set_rect(&mut self, r: Rect, t: Tile) {
        for y in r.y.floor() as i32..=(r.y + r.h).ceil() as i32 {
            for x in r.x.floor() as i32..=(r.x + r.w).ceil() as i32 {
                self.set(GridPos::new(x, y), t);
            }
        }
    }

    pub fn walkable(&self, g: GridPos) -> bool {
        self.tile(g).map(Tile::walkable).unwrap_or(false)
    }
}

// ---------------------------------------------------------------------------
// Items & factions
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Item {
    pub id: ItemId,
    pub name: String,
    pub description: String,
    pub base_value: i64,
    pub tags: Vec<String>,
    pub stackable: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FactionKind {
    Government,
    Business,
    Guild,
    Gang,
    Cult,
    Clan,
    Family,
    Charity,
}

/// One thing a faction wants to spend on.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FactionPriority {
    pub description: String,
    /// Weight relative to other priorities; higher = more urgent.
    pub weight: f32,
    /// Estimated cost to address.
    pub cost: i64,
    /// A stable predicate that marks this priority resolved.
    #[serde(default)]
    pub predicate: Option<crate::action::QuestPredicate>,
    pub resolved: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Faction {
    pub id: FactionId,
    pub name: String,
    pub kind: FactionKind,
    pub budget: i64,
    pub staff: u32,
    pub priorities: Vec<FactionPriority>,
    /// Territory / assets the faction is responsible for.
    pub responsible_for: Vec<LocationId>,
    pub members: Vec<ActorId>,
    /// Public legitimacy 0..1; affects whether it can act openly.
    pub legitimacy: f32,
    /// Whether the faction's existence is publicly known.
    pub public: bool,
}

impl Faction {
    /// Pick the highest expected-value unresolved priority the faction can
    /// afford. Returns `None` when nothing is worth doing.
    pub fn best_affordable_priority(&self) -> Option<usize> {
        self.priorities
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.resolved && p.cost <= self.budget)
            .max_by(|(_, a), (_, b)| {
                let av = a.weight / (1.0 + a.cost as f32);
                let bv = b.weight / (1.0 + b.cost as f32);
                av.partial_cmp(&bv).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
    }
}

// ---------------------------------------------------------------------------
// Facts, beliefs, provenance
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Sensitivity {
    Public,
    Restricted,
    Private,
    Secret,
    /// Only those inside the consiracy may plausibly know.
    InnerCircle,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Fact {
    pub id: FactId,
    /// Human-readable canonical statement.
    pub statement: String,
    pub truth: bool,
    pub sensitivity: Sensitivity,
    pub category: FactCategory,
    /// Facts that must be known before this one is plausible to discover.
    #[serde(default)]
    pub prerequisites: Vec<FactId>,
    /// Actors who could plausibly already know this fact.
    #[serde(default)]
    pub eligible_holders: Vec<ActorId>,
    /// Container kinds this fact may manifest as.
    #[serde(default)]
    pub eligible_containers: Vec<ContainerKind>,
    /// True once in-world evidence exists / it has been discovered.
    #[serde(default)]
    pub committed: bool,
    /// Location the fact (as a physical clue) currently lives in, if placed.
    #[serde(default)]
    pub placed_at: Option<LocationId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FactCategory {
    Organization,
    Identity,
    Crime,
    Shipment,
    Ledger,
    Relationship,
    Location,
    Motive,
    Observation,
    Rumor,
}

/// The physical/social form a fact can take when it enters play.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContainerKind {
    Document,
    Object,
    OverheardRemark,
    Testimony,
    Transaction,
    Environmental,
    Rumor,
}

/// Where a belief came from. Provenance is what stops "omniscient gossip".
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Provenance {
    /// Directly witnessed by the believer.
    Witnessed { event: EventId },
    /// Told by another actor.
    Told { by: ActorId, event: Option<EventId> },
    /// Read from a document/object.
    Read { item: ItemId },
    /// Inferred.
    Inferred { from: Vec<FactId> },
    /// Unknown / from the model's own reasoning (kept distinct on purpose).
    Assumed,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Belief {
    pub fact: FactId,
    /// Whether the actor thinks the statement is true.
    pub believes_true: bool,
    pub confidence: f32,
    pub source: Provenance,
    pub acquired_tick: u64,
}

// ---------------------------------------------------------------------------
// Story / narrative state (canonical side)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MysteryMode {
    /// The answer is fixed before evidence is generated (supports deduction).
    FixedAnswer,
    /// Some answer detail is chosen among still-viable candidates, then
    /// permanently committed.
    CollaborativelyResolved,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoryAnchor {
    pub id: String,
    pub statement: String,
    /// Immutable anchors can never be changed by the director.
    pub immutable: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Checkpoint {
    pub id: String,
    pub description: String,
    /// Facts that must be known for the checkpoint to become active.
    pub requires: Vec<FactId>,
    /// Machine-readable predicate for alternate completion.
    #[serde(default)]
    pub predicate: Option<crate::action::QuestPredicate>,
    pub state: CheckpointState,
    /// Fact packages eligible to be placed when this activates.
    #[serde(default)]
    pub candidate_facts: Vec<FactId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckpointState {
    Dormant,
    Active,
    Committed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoryState {
    pub mode: MysteryMode,
    pub anchors: Vec<StoryAnchor>,
    pub checkpoints: Vec<Checkpoint>,
    /// The fixed answer for `FixedAnswer` mysteries (e.g. the mafia leader).
    #[serde(default)]
    pub fixed_answer: BTreeMap<String, String>,
    /// Committed clue placements: fact -> (holder/container description).
    #[serde(default)]
    pub committed_containers: BTreeMap<FactId, ContainerCommit>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContainerCommit {
    pub fact: FactId,
    pub kind: ContainerKind,
    pub location: Option<LocationId>,
    pub holder: Option<ActorId>,
    pub tick: u64,
}

// ---------------------------------------------------------------------------
// The canonical world container
// ---------------------------------------------------------------------------

/// All canonical, authoritative game state. The `sim` crate owns the logic
/// that mutates this; every other layer only reads it (or proposes changes).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct World {
    pub tick: u64,
    pub day: u32,
    pub minute_of_day: u32,

    pub locations: BTreeMap<LocationId, Location>,
    pub actors: BTreeMap<ActorId, Actor>,
    pub items: BTreeMap<ItemId, Item>,
    pub factions: BTreeMap<FactionId, Faction>,
    pub facts: BTreeMap<FactId, Fact>,
    pub nav: NavGrid,
    pub story: StoryState,

    pub commitments: BTreeMap<CommitmentId, Commitment>,
    pub offers: BTreeMap<OfferId, Offer>,
    pub rumors: Vec<Rumor>,

    /// Structural damage 0..1 per location (1 = destroyed).
    pub damage: BTreeMap<LocationId, f32>,
    /// Global scenario flags used for explicit conditions.
    pub flags: BTreeSet<String>,
    /// Arbitrary numeric scenario counters keyed by string, used by
    /// `QuestPredicate::FactionPaid` and similar conditions.
    #[serde(default)]
    pub ledger: BTreeMap<String, i64>,

    pub next_commitment_id: u64,
    pub next_offer_id: u64,
    pub next_rumor_id: u64,
    pub next_goal_id: u64,
}

impl World {
    pub fn actor(&self, id: &ActorId) -> Option<&Actor> {
        self.actors.get(id)
    }

    pub fn actor_mut(&mut self, id: &ActorId) -> Option<&mut Actor> {
        self.actors.get_mut(id)
    }

    pub fn location(&self, id: &LocationId) -> Option<&Location> {
        self.locations.get(id)
    }

    pub fn fact(&self, id: &FactId) -> Option<&Fact> {
        self.facts.get(id)
    }

    /// Which named location contains this point (last match wins for overlaps).
    pub fn location_at(&self, p: Vec2) -> Option<&LocationId> {
        self.locations
            .iter()
            .find(|(_, l)| l.contains(p))
            .map(|(id, _)| id)
    }

    pub fn damage_at(&self, id: &LocationId) -> f32 {
        self.damage.get(id).copied().unwrap_or(0.0)
    }

    /// Does `actor` currently believe `fact` to be true?
    pub fn believes(&self, actor: &ActorId, fact: &FactId) -> bool {
        self.actors
            .get(actor)
            .and_then(|a| a.beliefs.get(fact))
            .map(|b| b.believes_true)
            .unwrap_or(false)
    }

    /// All living named NPCs (excludes player and background actors).
    pub fn named_npcs(&self) -> impl Iterator<Item = &Actor> {
        self.actors
            .values()
            .filter(|a| a.kind == ActorKind::Npc && a.is_alive())
    }

    pub fn faction(&self, id: &FactionId) -> Option<&Faction> {
        self.factions.get(id)
    }

    pub fn faction_mut(&mut self, id: &FactionId) -> Option<&mut Faction> {
        self.factions.get_mut(id)
    }

    pub fn set_flag(&mut self, flag: impl Into<String>) {
        self.flags.insert(flag.into());
    }

    pub fn has_flag(&self, flag: &str) -> bool {
        self.flags.contains(flag)
    }

    /// Minute-of-day formatted as HH:MM, useful for logs and UIs.
    pub fn clock_string(&self) -> String {
        format!(
            "{:02}:{:02}",
            self.minute_of_day / 60,
            self.minute_of_day % 60
        )
    }
}

/// Deterministically evaluate a world condition. This is the single source of
/// truth for quests, commitments, faction priorities and checkpoints.
pub fn evaluate_predicate(world: &World, pred: &crate::action::QuestPredicate) -> bool {
    use crate::action::QuestPredicate as P;
    match pred {
        P::HasItem { actor, item, qty } => world
            .actor(actor)
            .map(|a| a.inventory.quantity(item) >= *qty)
            .unwrap_or(false),
        P::ActorAlive { actor } => world.actor(actor).map(|a| a.is_alive()).unwrap_or(false),
        P::ActorDead { actor } => world.actor(actor).map(|a| !a.is_alive()).unwrap_or(false),
        P::ActorAt { actor, location } => world
            .actor(actor)
            .map(|a| a.location == *location)
            .unwrap_or(false),
        P::BelievesFact { actor, fact } => world.believes(actor, fact),
        P::FactCommitted { fact } => world.fact(fact).map(|f| f.committed).unwrap_or(false),
        P::LocationRepaired { location } => world.damage_at(location) <= 0.0,
        P::FactionPaid {
            faction,
            priority,
            amount,
        } => {
            world
                .ledger
                .get(&format!("{faction}:{priority}"))
                .copied()
                .unwrap_or(0)
                >= *amount
        }
        P::HasMoney { actor, amount } => world
            .actor(actor)
            .map(|a| a.inventory.money >= *amount)
            .unwrap_or(false),
        P::All(items) => items.iter().all(|p| evaluate_predicate(world, p)),
        P::Any(items) => items.iter().any(|p| evaluate_predicate(world, p)),
        P::Not(inner) => !evaluate_predicate(world, inner),
        P::Flag(flag) => world.flags.contains(flag),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::QuestPredicate;

    fn empty_world() -> World {
        World {
            tick: 0,
            day: 0,
            minute_of_day: 0,
            locations: BTreeMap::new(),
            actors: BTreeMap::new(),
            items: BTreeMap::new(),
            factions: BTreeMap::new(),
            facts: BTreeMap::new(),
            nav: NavGrid::new(4, 4),
            story: StoryState {
                mode: MysteryMode::FixedAnswer,
                anchors: Vec::new(),
                checkpoints: Vec::new(),
                fixed_answer: BTreeMap::new(),
                committed_containers: BTreeMap::new(),
            },
            commitments: BTreeMap::new(),
            offers: BTreeMap::new(),
            rumors: Vec::new(),
            damage: BTreeMap::new(),
            flags: BTreeSet::new(),
            ledger: BTreeMap::new(),
            next_commitment_id: 1,
            next_offer_id: 1,
            next_rumor_id: 1,
            next_goal_id: 1,
        }
    }

    #[test]
    fn flag_predicate_reflects_world_flags() {
        let mut w = empty_world();
        assert!(!evaluate_predicate(&w, &QuestPredicate::Flag("x".into())));
        w.set_flag("x");
        assert!(evaluate_predicate(&w, &QuestPredicate::Flag("x".into())));
    }

    #[test]
    fn all_and_any_compose() {
        let mut w = empty_world();
        w.set_flag("a");
        let all = QuestPredicate::All(vec![
            QuestPredicate::Flag("a".into()),
            QuestPredicate::Flag("b".into()),
        ]);
        let any = QuestPredicate::Any(vec![
            QuestPredicate::Flag("a".into()),
            QuestPredicate::Flag("b".into()),
        ]);
        assert!(!evaluate_predicate(&w, &all));
        assert!(evaluate_predicate(&w, &any));
    }

    #[test]
    fn damage_zero_means_repaired() {
        let w = empty_world();
        assert!(evaluate_predicate(
            &w,
            &QuestPredicate::LocationRepaired {
                location: LocationId::new("nowhere")
            }
        ));
    }
}
