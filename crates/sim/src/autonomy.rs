//! Deterministic, goal-driven autonomy for NPCs.
//!
//! Authored goals carry an optional pursuit target (a location or a person).
//! When an idle actor stands at its scheduled post, it may pursue the
//! highest-priority target-bearing goal: walk there, act on arrival, dwell
//! briefly, then cool down and return to its routine. Selection is purely
//! deterministic and local — no model calls, no per-tick inference.
//!
//! Conversation preservation is a hard rule: an actor within talk range of
//! the player, in a private conversation, or already walking is never
//! redirected by autonomy.

use rpg_domain::action::{Action, ActionProposal, InvestigateTarget, ProposalSource};
use rpg_domain::ids::{ActorId, LocationId, PLAYER};
use rpg_domain::world::{evaluate_predicate, Actor, ActorKind, Goal, GoalKind, GoalStatus};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::pathfind;
use crate::Simulation;

/// A commitment to walk to `location` in service of `goal_id`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Pursuit {
    pub goal_id: u64,
    pub location: LocationId,
    pub until_tick: u64,
}

/// Transient autonomy state. Persisted for exact continuation but carries no
/// canonical world truth: it only schedules and paces autonomous pursuits.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GoalAutonomy {
    /// ActorId -> the pursuit currently being carried out.
    pub pursuing: BTreeMap<ActorId, Pursuit>,
    /// ActorId -> tick until which goal autonomy is paused.
    pub cooldowns: BTreeMap<ActorId, u64>,
}

impl GoalAutonomy {
    /// Whether `actor` is currently pursuing a goal target.
    pub fn is_pursuing(&self, actor: &ActorId) -> bool {
        self.pursuing.contains_key(actor)
    }

    /// Whether `actor` is on a cooldown after completing a pursuit.
    pub fn is_cooling(&self, actor: &ActorId) -> bool {
        self.cooldowns.contains_key(actor)
    }
}

impl Simulation {
    /// Deterministic goal proposals for actors that would otherwise idle.
    ///
    /// Called from `step_impl` after routine proposals; it *replaces* the
    /// routine `Idle` for an actor when it has something better to do. This
    /// method mutates the pursuit/cooldown state (expiring stale pursuits,
    /// committing to new ones) as a side effect.
    pub fn autonomy_proposals(&mut self, already_acted: &BTreeSet<ActorId>) -> Vec<ActionProposal> {
        if !self.config.enable_goal_autonomy {
            return Vec::new();
        }
        let tick = self.world.tick;
        let margin_ticks = self.config.goal_pursuit_margin_ticks as u64;
        let cooldown_ticks = self.config.goal_cooldown_ticks as u64;

        // 1. Expire stale pursuits and cooldowns.
        let expired: Vec<ActorId> = self
            .autonomy
            .pursuing
            .iter()
            .filter(|(_, p)| p.until_tick <= tick)
            .map(|(id, _)| id.clone())
            .collect();
        for id in expired {
            self.autonomy.pursuing.remove(&id);
            self.autonomy.cooldowns.insert(id, tick + cooldown_ticks);
        }
        let cooled: Vec<ActorId> = self
            .autonomy
            .cooldowns
            .iter()
            .filter(|(_, t)| **t <= tick)
            .map(|(id, _)| id.clone())
            .collect();
        for id in cooled {
            self.autonomy.cooldowns.remove(&id);
        }

        let mut out = Vec::new();
        for actor in self.world.actors.values() {
            if !self.autonomy_eligible(actor, already_acted) {
                continue;
            }
            if let Some(pursuit) = self.autonomy.pursuing.get(&actor.id).cloned() {
                // Already pursuing: walk, act, or dwell until the deadline.
                if actor.location == pursuit.location {
                    // Arrived. Perform the goal action once, then dwell. Only
                    // a location goal aimed at this exact site can be acted on
                    // here; an actor-goal at a shared location stays silent.
                    if let Some(goal) = actor.goals.iter().find(|g| g.id == pursuit.goal_id) {
                        let aims_here = goal.target_location.as_ref() == Some(&pursuit.location);
                        if goal.kind == GoalKind::Investigate
                            && aims_here
                            && !actor.inspected.contains(pursuit.location.as_str())
                        {
                            out.push(ActionProposal {
                                actor: actor.id.clone(),
                                action: Action::Investigate {
                                    target: InvestigateTarget::Location(pursuit.location.clone()),
                                },
                                source: ProposalSource::NpcPolicy,
                                rationale: format!("goal: {}", goal.description),
                            });
                        }
                    }
                    continue;
                }
                // Not arrived yet: keep walking (path may have been cut).
                let rationale = actor
                    .goals
                    .iter()
                    .find(|g| g.id == pursuit.goal_id)
                    .map(|g| g.description.clone())
                    .unwrap_or_else(|| "pursuit".into());
                out.push(ActionProposal {
                    actor: actor.id.clone(),
                    action: Action::MoveTo {
                        location: pursuit.location.clone(),
                    },
                    source: ProposalSource::NpcPolicy,
                    rationale: format!("goal: {rationale}"),
                });
                continue;
            }
            // Fresh pursuit? Only the highest-priority target-bearing goal.
            let Some((goal, target, travel_ticks)) = self.select_goal_target(actor) else {
                continue;
            };
            if actor.location == target {
                // Already at the target: nothing observable to do. Cool down
                // to avoid re-selecting the same satisfied situation.
                self.autonomy
                    .cooldowns
                    .insert(actor.id.clone(), tick + cooldown_ticks);
                continue;
            }
            out.push(ActionProposal {
                actor: actor.id.clone(),
                action: Action::MoveTo {
                    location: target.clone(),
                },
                source: ProposalSource::NpcPolicy,
                rationale: format!("goal: {}", goal.description),
            });
            self.autonomy.pursuing.insert(
                actor.id.clone(),
                Pursuit {
                    goal_id: goal.id,
                    location: target,
                    // Travel-scaled deadline: enough ticks to walk the planned
                    // path at the actor's speed, plus a pacing margin. Long
                    // hauls are not cut short; nothing lingers forever.
                    until_tick: tick + travel_ticks + margin_ticks,
                },
            );
        }
        out
    }

    /// Standard eligibility guards for autonomous action. Mirrors the
    /// routine guards, plus conversation preservation: an actor near the
    /// player is never redirected, so conversations cannot be broken.
    fn autonomy_eligible(&self, actor: &Actor, already_acted: &BTreeSet<ActorId>) -> bool {
        if !actor.is_alive() || actor.kind == ActorKind::Player {
            return false;
        }
        if already_acted.contains(&actor.id)
            || self.paths.contains_key(&actor.id)
            || actor.in_private_conversation_with.is_some()
            || self
                .pending_private
                .iter()
                .any(|p| p.a == actor.id || p.b == actor.id)
        {
            return false;
        }
        // Conversation preservation: don't walk away from the player.
        if let Some(player) = self.world.actor(&ActorId::new(PLAYER)) {
            if player.pos.distance(actor.pos) <= self.config.talk_radius {
                return false;
            }
        }
        true
    }

    /// The highest-priority active goal with a reachable pursuit target.
    /// Ties break by goal id for determinism. Predicate-satisfied goals are
    /// skipped; unreachable targets are skipped. Returns the estimated
    /// travel ticks for the planned path at the actor's speed, used for the
    /// pursuit deadline.
    fn select_goal_target(&self, actor: &Actor) -> Option<(Goal, LocationId, u64)> {
        let dt = self.config.move_units_per_tick;
        let mut best: Option<(f32, u64, LocationId, u64)> = None;
        for g in &actor.goals {
            if g.status != GoalStatus::Active {
                continue;
            }
            if let Some(pred) = &g.predicate {
                if evaluate_predicate(&self.world, pred) {
                    continue;
                }
            }
            let target = if let Some(loc) = &g.target_location {
                loc.clone()
            } else if let Some(other) = &g.target_actor {
                match self.world.actor(other) {
                    Some(o) if o.is_alive() => o.location.clone(),
                    _ => continue,
                }
            } else {
                continue;
            };
            if actor.location == target {
                continue;
            }
            // Yo-yo prevention: an Investigate goal whose location was already
            // examined is satisfied for this pass; re-selecting it would make
            // the actor churn between its routine and the target.
            if g.kind == GoalKind::Investigate
                && g.target_location.is_some()
                && actor.inspected.contains(target.as_str())
            {
                continue;
            }
            let dest = self
                .world
                .location(&target)
                .map(|l| l.center())
                .unwrap_or_default();
            let Some(path) = pathfind::plan_world_path(&self.world.nav, actor.pos, dest) else {
                continue;
            };
            // Path length in world units, including the initial gap from the
            // actor's position to the first waypoint.
            let mut pts = Vec::with_capacity(path.len() + 1);
            pts.push(actor.pos);
            pts.extend(path);
            let path_len: f32 = pts.windows(2).map(|p| p[0].distance(p[1])).sum();
            let budget = dt * actor.capabilities.speed.max(0.1);
            let travel_ticks = (path_len / budget).ceil().max(1.0) as u64;
            let score = (g.priority, std::cmp::Reverse(g.id));
            let better = match &best {
                None => true,
                Some((p, id, _, _)) => score > (*p, std::cmp::Reverse(*id)),
            };
            if better {
                best = Some((g.priority, g.id, target, travel_ticks));
            }
        }
        let (_priority, id, target, travel) = best?;
        let goal = actor.goals.iter().find(|g| g.id == id)?.clone();
        Some((goal, target, travel))
    }
}
