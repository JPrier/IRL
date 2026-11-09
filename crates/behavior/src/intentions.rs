//! Persistent intentions (design §3 `J`, §4.3 fast path; plan I-3).
//!
//! An intention is the actor's adopted plan: once adopted it survives world
//! ticks — the fast path continues it without re-deliberation — until it is
//! invalidated by its own satisfaction predicate or until its abandonment
//! criteria are met (design §3: abandonment criteria on goals).
//!
//! This kernel is pure: it reads canonical world state through
//! [`evaluate_predicate`] and never mutates anything. It is the production
//! fast-path/continuation check the new decision pathway calls; it carries no
//! actor-id special cases (T2.6) and no world-truth access beyond the typed
//! predicate evaluator.
//!
//! The needs-urgency mapping into intention *selection* is deliberately not
//! here: `Actor` has no needs state yet (plan I-4a, Phase 3). Adding a
//! placeholder needs term before that state exists would be a silent stub —
//! I-3b completes that work when I-4a lands (see `super::utility`).

use rpg_domain::action::QuestPredicate;
use rpg_domain::behavior_types::AbandonmentCriteria;
use rpg_domain::world::{evaluate_predicate, World};
use serde::{Deserialize, Serialize};

/// An adopted plan with continuation state (design §3 `J`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Intention {
    /// What the intention is for (trace description; never authoritative).
    pub goal: String,
    /// The chosen option's stable id (the action/plan family adopted).
    pub option: String,
    /// Predicate that marks the intention satisfied; `None` = open-ended.
    #[serde(default)]
    pub satisfaction: Option<QuestPredicate>,
    #[serde(default)]
    pub abandonment: AbandonmentCriteria,
    pub adopted_tick: u64,
    /// Consecutive failed attempts of the adopted option (rejections count).
    #[serde(default)]
    pub failed_attempts: u32,
    /// How many times the adopted option changed (switch-penalty bookkeeping;
    /// η_i·switch_i in design §4.2 — deters oscillation).
    #[serde(default)]
    pub switches: u32,
}

/// Why the fast path should stop continuing an intention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplanReason {
    /// The satisfaction predicate now holds: the goal is reached.
    GoalSatisfied,
    /// The abandonment criteria are met: give the goal up explicitly.
    Abandoned,
}

/// Fast-path decision: continue the adopted intention, or replan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Replan {
    /// Keep going: the adopted plan survives this tick without
    /// re-deliberation (§4.3 fast path).
    Continue,
    /// The intention is invalidated; re-deliberation is required.
    Replan(ReplanReason),
}

impl Intention {
    /// Adopt an option. Replacing an intention whose option differs counts a
    /// switch (the penalty that deters oscillation is applied by the caller
    /// in the utility ranking).
    pub fn adopt(
        goal: impl Into<String>,
        option: impl Into<String>,
        satisfaction: Option<QuestPredicate>,
        abandonment: AbandonmentCriteria,
        adopted_tick: u64,
        previous: Option<&Intention>,
    ) -> Self {
        let option = option.into();
        let switches = previous
            .map(|p| {
                // `switches` is u32 in the persisted schema.
                p.switches.saturating_add(u32::from(p.option != option))
            })
            .unwrap_or(0);
        Intention {
            goal: goal.into(),
            option,
            satisfaction,
            abandonment,
            adopted_tick,
            failed_attempts: 0,
            switches,
        }
    }

    /// Record one failed attempt of the adopted option (e.g. a rejection).
    pub fn record_failure(&mut self) {
        self.failed_attempts = self.failed_attempts.saturating_add(1);
    }
}

/// T2.1: an adopted plan survives world ticks unless invalidated; the fast
/// path continues without re-deliberation. Only the intention's own
/// satisfaction predicate or its abandonment criteria stop it.
pub fn consider_replan(intention: &Intention, world: &World) -> Replan {
    if let Some(pred) = &intention.satisfaction {
        if evaluate_predicate(world, pred) {
            return Replan::Replan(ReplanReason::GoalSatisfied);
        }
    }
    if let Some(n) = intention.abandonment.after_failed_attempts {
        if intention.failed_attempts >= n {
            return Replan::Replan(ReplanReason::Abandoned);
        }
    }
    Replan::Continue
}
