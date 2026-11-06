//! # rpg_narrative
//!
//! Deterministic story controller: immutable anchors, flexible checkpoints,
//! information carriers, and validation of director proposals.
//!
//! The director (an LLM, in production) only ever produces
//! [`NarrativeChange`] *proposals*. This module decides whether a proposal is
//! consistent with already-observed facts and, if so, commits it. It is the
//! story equivalent of the action validator.

use rpg_domain::events::{EventKind, EventLog};
use rpg_domain::ids::*;
use rpg_domain::rng::DeterministicRng;
use rpg_domain::world::{
    evaluate_predicate, Actor, ActorKind, CheckpointState, ContainerCommit, ContainerKind, World,
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
