//! Bounded event-driven planning and fair activation (design §4.3, §8;
//! plan I-6, I6a slice).
//!
//! Two pieces live here as pure, serializable kernels:
//!
//! * **Fair deterministic activation** ([`ActivationState`]): round-robin
//!   rotation with a stable actor order — every eligible NPC receives at
//!   least one activation slot per window (design §8
//!   `pick_active_agents_with_budget`), with explicit per-step budget and
//!   per-activation work caps (design §9.2.4, M6). Relevant events
//!   (signals) activate sooner — but a signal claim may jump the rotation
//!   queue at most once per full rotation window, so sustained signal load
//!   cannot starve the rotation; nothing here lets one actor monopolize
//!   activations or lets an unrelated event spawn work.
//! * **Persistent plans** ([`Plan`]): the adopted plan with its concrete
//!   next step, its satisfaction predicate and abandonment criteria
//!   (design §3 `J`). Plans survive ticks and save/load; the fast path
//!   continues them without re-deliberation; relevant-event signals
//!   ([`ReplanSignal`]) trigger bounded replanning.
//!
//! The kernels are world-truth-free by construction: they hold no world
//! reference. Candidate generation (actor-local, beliefs-only) lives in
//! `rpg_agents`; pipeline orchestration lives in `rpg_sim`. Planner state
//! is mind-local — like the deal intentions it never mutates canonical
//! world state and carries no events (M1 classification: mind-local
//! bookkeeping, not an authoritative world mutation).

use rpg_domain::action::Action;
use rpg_domain::behavior_types::AbandonmentCriteria;
use rpg_domain::emotions::EmotionResponse;
use rpg_domain::ids::ActorId;
use rpg_domain::ids::EventId;
use rpg_domain::needs::NeedKind;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Why an actor's reconsideration was scheduled (design §8:
/// `schedule_reconsideration(npc, priority_for(observed_events))`). Only
/// genuinely relevant events become signals: an event that is not about the
/// actor, its deals, its appraisals, its needs or its emotion gates never
/// produces one — so unrelated events cannot trigger unbounded replanning.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ReplanTrigger {
    /// A mind-local appraisal was recorded (threat, gift, sanction, …).
    Appraisal { interpretation: String },
    /// A question the actor asked was answered (new evidence may change the
    /// plan).
    QueryAnswered,
    /// A contract the actor is a party to changed state.
    DealChanged { state: String },
    /// A need crossed its urgency threshold (latched urgent).
    NeedUrgent { need: NeedKind },
    /// An emotion response class became gated (flee/retaliate/…).
    EmotionGate { response: EmotionResponse },
    /// An offer was made to the actor.
    OfferReceived,
    /// A claim was delivered to the actor (new evidence may change the plan).
    ClaimReceived { fact: rpg_domain::ids::FactId },
}

/// One scheduled reconsideration. Bounded per actor: the queue keeps at
/// most `max_signals_per_actor` entries (oldest dropped).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReplanSignal {
    pub tick: u64,
    pub trigger: ReplanTrigger,
    /// The originating event, when the trigger is event-sourced (trace
    /// provenance; never fabricated).
    pub event: Option<EventId>,
}

/// A persistent adopted plan (design §3 `J`): the intention plus its
/// concrete next step. Survives ticks and save/load; the fast path re-proposes
/// the step without re-deliberation while the intention stays valid.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    /// What the plan is for (trace description; never authoritative).
    pub summary: String,
    /// The adopted intention (option, satisfaction, abandonment criteria).
    pub intention: crate::intentions::Intention,
    /// The concrete next step: the typed action the plan proposes (the sim
    /// validates it like any other proposal — adoption is never application).
    pub action: Action,
    /// Tick the plan was adopted.
    pub adopted_tick: u64,
    /// The last tick the plan's own step was proposed for execution (plan
    /// I-6 I6b one-shot hardening): a one-shot plan proposes its step at
    /// most once — if the parameter-matched outcome event has scrolled past
    /// the retirement window by the next activation, the plan retires on
    /// this record instead of re-executing (no duplicate outcomes).
    /// Rejections do not set it (a rejected attempt is failure bookkeeping,
    /// not a completion), so retry-until-abandonment semantics are intact.
    #[serde(default)]
    pub last_proposed_tick: Option<u64>,
}

/// Which need/actor-local condition currently gives an actor a claim on an
/// activation slot (design §4.3: deliberation is event-driven — an actor
/// with nothing relevant pending is not deliberated).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActivationReason {
    /// A relevant event was recorded since the last activation.
    Signal,
    /// A plan exists: continue or reassess it (cheap fast-path work).
    Plan,
    /// A need is latched urgent (design §6.4: satiation/thresholds).
    NeedUrgent,
    /// An emotion response class is gated.
    EmotionGate,
}

/// Fair, deterministic activation state (mind-local; persisted).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ActivationState {
    /// Stable rotation order (sorted actor ids; `ensure_order` maintains
    /// it as actors come and go without reshuffling).
    pub order: Vec<ActorId>,
    /// Round-robin cursor into `order`.
    pub cursor: usize,
    /// Tick of each actor's most recent activation. A priority (signal)
    /// claim may jump the rotation queue at most once per full rotation
    /// window (`order.len()` ticks) — the bound that keeps sustained
    /// signal load from starving the rotation (design §8: every eligible
    /// NPC receives at least one activation slot per window even when
    /// another actor is re-signaled every tick).
    #[serde(default)]
    pub last_activated: BTreeMap<ActorId, u64>,
}

impl ActivationState {
    /// Keep `order` a stable, sorted snapshot of `eligible`: retain eligible
    /// actors in their existing relative order, append newcomers sorted.
    pub fn ensure_order(&mut self, eligible: &[ActorId]) {
        let set: BTreeSet<&ActorId> = eligible.iter().collect();
        self.order.retain(|a| set.contains(a));
        let known: BTreeSet<&ActorId> = self.order.iter().collect();
        let mut newcomers: Vec<&ActorId> =
            eligible.iter().filter(|a| !known.contains(*a)).collect();
        newcomers.sort();
        self.order.extend(newcomers.into_iter().cloned());
        if !self.order.is_empty() {
            self.cursor %= self.order.len();
        } else {
            self.cursor = 0;
        }
    }

    /// Pick up to `budget` actors to activate this step: priority actors
    /// (relevant signals) first — deduplicated, in the given order — then
    /// round-robin from the cursor over the rest, each picked actor
    /// advancing the cursor past itself. Deterministic; no actor can be
    /// activated twice in one step; the budget is a hard cap.
    ///
    /// Non-starvation under sustained signal load (design §8: every
    /// eligible NPC receives at least one activation slot per window): a
    /// priority actor may *jump* the queue at most once per full rotation
    /// window (`order.len()` ticks since its last activation of any kind).
    /// Within the window a re-signaled actor waits its ordinary rotation
    /// turn — so a persistently re-signaled actor cannot monopolize a
    /// budget shared with other eligible actors, while a *freshly*
    /// relevant actor still activates sooner.
    pub fn pick(
        &mut self,
        eligible: &BTreeSet<ActorId>,
        priority: &[ActorId],
        budget: usize,
        now: u64,
    ) -> Vec<ActorId> {
        let mut picked: Vec<ActorId> = Vec::new();
        if budget == 0 || self.order.is_empty() {
            return picked;
        }
        let len = self.order.len();
        let window = len.max(1) as u64;
        for a in priority {
            if picked.len() >= budget {
                return picked;
            }
            if !eligible.contains(a) || picked.contains(a) {
                continue;
            }
            // Jump only outside this actor's fairness window; inside it the
            // signal holder waits for its ordinary rotation slot below.
            let may_jump = self
                .last_activated
                .get(a)
                .is_none_or(|t| now.saturating_sub(*t) >= window);
            if may_jump {
                if let Some(idx) = self.order.iter().position(|o| o == a) {
                    self.cursor = (idx + 1) % len;
                }
                self.last_activated.insert(a.clone(), now);
                picked.push(a.clone());
            }
        }
        if picked.len() >= budget {
            return picked;
        }
        for i in 0..len {
            let idx = (self.cursor + i) % len;
            let a = &self.order[idx];
            if eligible.contains(a) && !picked.contains(a) {
                picked.push(a.clone());
                self.cursor = (idx + 1) % len;
                self.last_activated.insert(a.clone(), now);
                if picked.len() >= budget {
                    break;
                }
            }
        }
        picked
    }
}

/// The planner's persistent state (mind-local; persisted in the sim's
/// transient save state for exact continuation). Everything is bounded.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PlannerState {
    /// Adopted plans per actor (design §3 `J`).
    pub plans: BTreeMap<ActorId, Plan>,
    /// Scheduled reconsiderations per actor (bounded per actor).
    pub signals: BTreeMap<ActorId, Vec<ReplanSignal>>,
    /// Options abandoned recently, with the tick until which re-adoption is
    /// suppressed (hysteresis against re-adopting a just-abandoned plan —
    /// the same loop-prevention the deal pathway applies to its intentions).
    pub recently_abandoned: BTreeMap<ActorId, BTreeMap<String, u64>>,
    /// Fair-activation rotation state.
    pub activation: ActivationState,
}

impl PlannerState {
    /// Record a reconsideration signal for `actor`, bounded at
    /// `max_per_actor` entries (oldest dropped). A signal is only recorded
    /// when the trigger is genuinely about the actor — call sites decide
    /// relevance; this kernel only enforces the bound.
    pub fn record_signal(&mut self, actor: &ActorId, signal: ReplanSignal, max_per_actor: usize) {
        let queue = self.signals.entry(actor.clone()).or_default();
        // Collapse a run of identical triggers: two appraisals in one tick
        // are one reconsideration, not two (bounded work, M6).
        if queue
            .last()
            .is_some_and(|s| s.trigger == signal.trigger && s.tick == signal.tick)
        {
            return;
        }
        queue.push(signal);
        if queue.len() > max_per_actor {
            let overflow = queue.len() - max_per_actor;
            queue.drain(0..overflow);
        }
    }

    /// Drain (take) all pending signals for `actor`.
    pub fn take_signals(&mut self, actor: &ActorId) -> Vec<ReplanSignal> {
        self.signals.remove(actor).unwrap_or_default()
    }

    /// Whether `option` was abandoned by `actor` recently (before `until`).
    pub fn recently_abandoned(&self, actor: &ActorId, option: &str, now: u64) -> bool {
        self.recently_abandoned
            .get(actor)
            .and_then(|m| m.get(option))
            .is_some_and(|until| *until > now)
    }

    /// Record that `actor` abandoned `option`; suppressed from re-adoption
    /// until `until_tick`. Bounded: an actor's suppressed set is pruned of
    /// expired entries on write.
    pub fn record_abandoned(&mut self, actor: &ActorId, option: &str, until_tick: u64) {
        let entry = self.recently_abandoned.entry(actor.clone()).or_default();
        entry.retain(|_, until| *until > until_tick.saturating_sub(1));
        entry.insert(option.to_string(), until_tick);
    }

    /// Retire expired suppression entries (deterministic bookkeeping).
    pub fn prune_abandoned(&mut self, now: u64) {
        for entry in self.recently_abandoned.values_mut() {
            entry.retain(|_, until| *until > now);
        }
        self.recently_abandoned.retain(|_, e| !e.is_empty());
    }
}

/// Adoption bundle the sim uses to create a plan. Kept explicit so the sim
/// cannot accidentally build a plan without its abandonment criteria.
/// `previous` is the intention the actor is switching away from (when it has
/// one): `Intention::adopt` counts the switch (§4.2 η_i·switch_i
/// bookkeeping), so the trace shows oscillation honestly.
pub fn adopt_plan(
    summary: &str,
    option: impl Into<String>,
    action: Action,
    satisfaction: Option<rpg_domain::action::QuestPredicate>,
    abandonment: AbandonmentCriteria,
    adopted_tick: u64,
    previous: Option<&crate::intentions::Intention>,
) -> Plan {
    Plan {
        summary: summary.to_string(),
        intention: crate::intentions::Intention::adopt(
            summary,
            option,
            satisfaction,
            abandonment,
            adopted_tick,
            previous,
        ),
        action,
        adopted_tick,
        last_proposed_tick: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids() -> Vec<ActorId> {
        ["a", "b", "c", "d"]
            .iter()
            .map(|s| ActorId::new(*s))
            .collect()
    }

    #[test]
    fn fair_rotation_gives_every_eligible_actor_a_slot_per_window() {
        let ids = ids();
        let mut state = ActivationState::default();
        state.ensure_order(&ids);
        let eligible: BTreeSet<ActorId> = ids.iter().cloned().collect();
        let mut seen: BTreeSet<ActorId> = BTreeSet::new();
        // Budget 1: four ticks cover all four actors.
        for tick in 0..4 {
            let picked = state.pick(&eligible, &[], 1, tick);
            assert_eq!(picked.len(), 1);
            let first = picked[0].clone();
            seen.insert(first);
        }
        assert_eq!(seen, eligible);
    }

    #[test]
    fn priority_actors_go_first_without_breaking_fairness() {
        let ids = ids();
        let mut state = ActivationState::default();
        state.ensure_order(&ids);
        let eligible: BTreeSet<ActorId> = ids.iter().cloned().collect();
        let picked = state.pick(&eligible, &[ActorId::new("c")], 2, 0);
        assert_eq!(picked[0], ActorId::new("c"));
        // The rotation continues fairly from after the priority actor (the
        // jump advanced the cursor past it).
        assert_eq!(picked[1], ActorId::new("d"));
    }

    #[test]
    fn budget_is_a_hard_cap() {
        let ids = ids();
        let mut state = ActivationState::default();
        state.ensure_order(&ids);
        let eligible: BTreeSet<ActorId> = ids.iter().cloned().collect();
        let picked = state.pick(&eligible, &[], 2, 0);
        assert_eq!(picked.len(), 2);
        // Not eligible actors are skipped and do not consume the budget.
        let mut half: BTreeSet<ActorId> = BTreeSet::new();
        half.insert(ActorId::new("b"));
        let picked = state.pick(&half, &[], 3, 1);
        assert_eq!(picked, vec![ActorId::new("b")]);
    }

    #[test]
    fn persistent_signal_holder_cannot_starve_the_rotation() {
        // Adversarial regression (review F1): one actor is re-signaled on
        // every tick while two others are eligible through persistent plans;
        // budget 1 must still rotate through all three.
        let x = ActorId::new("x");
        let y = ActorId::new("y");
        let z = ActorId::new("z");
        let mut state = ActivationState::default();
        state.ensure_order(&[x.clone(), y.clone(), z.clone()]);
        let eligible: BTreeSet<ActorId> = [x.clone(), y.clone(), z.clone()].into();
        let mut seen: BTreeMap<ActorId, usize> = BTreeMap::new();
        for tick in 0..24u64 {
            let picked = state.pick(&eligible, std::slice::from_ref(&x), 1, tick);
            assert_eq!(picked.len(), 1, "budget is a hard cap at tick {tick}");
            *seen.entry(picked[0].clone()).or_default() += 1;
        }
        for id in [&x, &y, &z] {
            let count = seen.get(id).copied().unwrap_or(0);
            assert!(count >= 1, "{id} starved under sustained signal load");
        }
        // The window bound also keeps the signal holder from monopolizing:
        // with three actors it cannot be activated more than once per
        // window, i.e. at most every other slot here.
        assert!(
            seen[&x] <= seen[&y] + 2,
            "signal holder monopolized activations: {seen:?}"
        );
    }

    #[test]
    fn a_fresh_signal_jumps_immediately_but_only_once_per_window() {
        let x = ActorId::new("x");
        let y = ActorId::new("y");
        let mut state = ActivationState::default();
        state.ensure_order(&[y.clone(), x.clone()]);
        let eligible: BTreeSet<ActorId> = [x.clone(), y.clone()].into();
        // Tick 0: the fresh signal jumps ahead of the rotation.
        assert_eq!(
            state.pick(&eligible, std::slice::from_ref(&x), 1, 0),
            vec![x.clone()]
        );
        // Tick 1: the re-signal does not jump again within the window (the
        // other eligible actor takes its slot).
        assert_eq!(
            state.pick(&eligible, std::slice::from_ref(&x), 1, 1),
            vec![y.clone()]
        );
        // Tick 2: the window (len = 2) has elapsed — the signal holder may
        // jump again.
        assert_eq!(
            state.pick(&eligible, std::slice::from_ref(&x), 1, 2),
            vec![x.clone()]
        );
    }

    #[test]
    fn signals_are_bounded_and_deduplicated_per_tick() {
        let mut state = PlannerState::default();
        let a = ActorId::new("a");
        for i in 0..20 {
            state.record_signal(
                &a,
                ReplanSignal {
                    tick: i / 2,
                    trigger: ReplanTrigger::Appraisal {
                        interpretation: "felt threatened".into(),
                    },
                    event: None,
                },
                8,
            );
        }
        assert_eq!(state.signals.get(&a).map(Vec::len), Some(8));
    }

    #[test]
    fn abandoned_options_are_suppressed_only_temporarily() {
        let mut state = PlannerState::default();
        let a = ActorId::new("a");
        state.record_abandoned(&a, "sell", 10);
        assert!(state.recently_abandoned(&a, "sell", 5));
        assert!(!state.recently_abandoned(&a, "sell", 10));
        state.prune_abandoned(10);
        assert!(state.recently_abandoned.is_empty());
    }
}
