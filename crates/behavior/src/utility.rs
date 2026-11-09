//! Utility ranking of candidate options (design §4.2; plan I-3b).
//!
//! Expectations are computed by the **caller from the actor's beliefs** —
//! this kernel scores what it is given and never reads world truth, so a
//! wrong belief yields a wrong (but explainable) expectation, which is the
//! design intent. Ranking is bounded and deterministic:
//!
//! 1. **Feasibility filter first** (design §4.1 stage 5, T2.8): physically
//!    impossible options are removed *before* any scoring; they can never be
//!    selected, no matter their desirability.
//! 2. **Lexicographic hard constraints** (T2.8): personally unacceptable
//!    options cannot be traded for weights. While any acceptable option
//!    exists, unacceptable ones are excluded from the ranking; when *no*
//!    acceptable option remains, the ranking returns empty (the actor
//!    declines rather than violates). This is distinct from physical
//!    impossibility and is reported as such.
//! 3. **Needs differential** (α_i·Δneeds_i, plan I-3b + I-4a): an option's
//!    expected need changes are weighted by the actor's *current* need
//!    state — the same gain is worth more when the need is deprived
//!    (marginal utility), so two actors with different need profiles rank
//!    the same option set differently (T3.9).
//! 4. **Switch penalty** (η_i·switch_i): an option that differs from the
//!    currently adopted intention's option pays the caller-provided penalty,
//!    deterring oscillation between otherwise-equal options.

use rpg_domain::needs::{NeedKind, Needs, NeedsConfig};
use rpg_domain::rng::DeterministicRng;
use serde::{Deserialize, Serialize};

/// Whether an option is possible, and if it is, whether the actor may
/// choose it. Physically impossible and personally unacceptable are distinct
/// classes (T2.8).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Feasibility {
    /// Physically possible and not barred by a lexicographic constraint.
    Possible,
    /// The world cannot realize this option (guards/paths/resources).
    /// Filtered before scoring; never sampled.
    PhysicallyImpossible,
    /// Physically possible but barred by a lexicographic personal
    /// constraint (e.g. "refuses betrayal except under severe coercion").
    /// Excluded while any clean option exists; never traded for weights.
    PersonallyUnacceptable,
}

/// One candidate option with the caller-computed expectation terms
/// (beliefs-only; never world truth).
#[derive(Clone, Debug, PartialEq)]
pub struct UtilityOption {
    /// Stable option id (also the intention `option` value).
    pub id: String,
    /// Expected goal progress Δgoal (belief-based), unbounded scale.
    pub goal_delta: f32,
    /// Expected cost c_i(a) (items, money, risk, attention), ≥ 0 scale.
    pub cost: f32,
    pub feasibility: Feasibility,
    /// Expected need changes Δneeds_i(a) (design §4.2 α_i·Δneeds_i term):
    /// `(kind, gain)` pairs — positive gains satisfy a need, negative ones
    /// deplete it. Empty means the option has no needs expectation.
    pub needs_gain: Vec<(NeedKind, f32)>,
}

impl UtilityOption {
    /// A convenience constructor for options without needs expectations.
    pub fn new(
        id: impl Into<String>,
        goal_delta: f32,
        cost: f32,
        feasibility: Feasibility,
    ) -> Self {
        UtilityOption {
            id: id.into(),
            goal_delta,
            cost,
            feasibility,
            needs_gain: Vec::new(),
        }
    }
}

/// A scored option: what the kernel ranked and why.
#[derive(Clone, Debug, PartialEq)]
pub struct Scored {
    pub id: String,
    pub utility: f32,
    /// True when the switch penalty was applied (option ≠ current
    /// intention's option).
    pub switched: bool,
}

/// The outcome of ranking: what survived the filters.
#[derive(Clone, Debug, PartialEq)]
pub struct Ranking {
    /// Acceptable options, best first. Empty when nothing acceptable
    /// survived (the actor declines rather than violates a constraint).
    pub ranked: Vec<Scored>,
    /// Physically impossible options, removed before scoring (T2.8).
    pub impossible: Vec<String>,
    /// Personally unacceptable options, excluded lexicographically (T2.8).
    pub unacceptable: Vec<String>,
}

/// A sampled choice: replayable because both the RNG sample and the
/// resulting index are recorded (design §4.2: "include the RNG seed and
/// sample in the replay record").
#[derive(Clone, Debug, PartialEq)]
pub struct Sampled {
    pub id: String,
    pub index: usize,
    /// The uniform sample drawn (0..1) for the softmax walk.
    pub sample: f32,
}

/// Ranking parameters. Weights are modeling assumptions (plan §9), kept in
/// one struct so call sites stay honest about what they chose.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// η_i: penalty for switching away from the current intention's option.
    pub switch_penalty: f32,
    /// Softmax temperature τ_i (low → near-deterministic argmax).
    pub tau: f32,
    /// α_i: weight of the needs-differential term (design §4.2). 0 disables
    /// it (a needs-blind actor).
    pub needs_alpha: f32,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            switch_penalty: 2.0,
            tau: 0.2,
            needs_alpha: 1.0,
        }
    }
}

/// Marginal utility weight of one need unit for the actor's *current* need
/// state (T3.9 needs differential): a positive gain is weighted by the
/// remaining deprivation `1 − value` (diminishing toward satiation — gains
/// above the satiation point register nothing), a loss by 1.0 (deprivation
/// hurts at full weight). Returns 0 for an empty gain.
fn needs_marginal(kind: NeedKind, gain: f32, needs: &Needs, config: &NeedsConfig) -> f32 {
    if gain == 0.0 {
        return 0.0;
    }
    let value = needs.value(kind);
    if gain > 0.0 {
        if value >= config.tuning(kind).satiation {
            return 0.0;
        }
        (1.0 - value).clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// The needs-differential term Δneeds_i for one option, given the actor's
/// current needs state and tuning: `Σ gain_k · marginal_k(value_k)`.
pub fn needs_term(option: &UtilityOption, needs: &Needs, config: &NeedsConfig) -> f32 {
    option
        .needs_gain
        .iter()
        .map(|(kind, gain)| gain * needs_marginal(*kind, *gain, needs, config))
        .sum()
}

/// Rank feasible options: filter physically impossible first, exclude
/// lexicographically unacceptable ones while a clean option exists, apply
/// the switch penalty, and sort best-first. Ties break by option id for
/// determinism.
///
/// Needs-blind variant: no α_i·Δneeds term is applied (equivalent to an
/// actor with α_i = 0).
pub fn rank(options: &[UtilityOption], current: Option<&str>, params: Params) -> Ranking {
    rank_impl(options, current, params, None, None)
}

/// Rank with the **needs differential** (design §4.2 α_i·Δneeds_i; plan
/// I-3b): each option's `needs_gain` is weighted by the actor's current
/// needs state and tuning, so deprived needs make their gains worth more —
/// two actors with different need profiles rank the same options
/// differently (T3.9).
pub fn rank_with_needs(
    options: &[UtilityOption],
    current: Option<&str>,
    params: Params,
    needs: &Needs,
    config: &NeedsConfig,
) -> Ranking {
    rank_impl(options, current, params, Some(needs), Some(config))
}

fn rank_impl(
    options: &[UtilityOption],
    current: Option<&str>,
    params: Params,
    needs: Option<&Needs>,
    config: Option<&NeedsConfig>,
) -> Ranking {
    let mut impossible = Vec::new();
    let mut unacceptable = Vec::new();
    let mut clean: Vec<(&UtilityOption, f32, bool)> = Vec::new();
    for opt in options {
        match opt.feasibility {
            Feasibility::PhysicallyImpossible => {
                impossible.push(opt.id.clone());
            }
            Feasibility::PersonallyUnacceptable => {
                unacceptable.push(opt.id.clone());
            }
            Feasibility::Possible => {
                let switched = current.is_some_and(|c| c != opt.id.as_str());
                let needs_term = match (needs, config) {
                    (Some(n), Some(c)) => needs_term(opt, n, c),
                    _ => 0.0,
                };
                let utility = opt.goal_delta - opt.cost + params.needs_alpha * needs_term
                    - if switched { params.switch_penalty } else { 0.0 };
                clean.push((opt, utility, switched));
            }
        }
    }
    // Lexicographic: hard constraints are not tradable. If nothing clean
    // remains, decline entirely rather than pick the least-bad violation.
    if clean.is_empty() {
        return Ranking {
            ranked: Vec::new(),
            impossible,
            unacceptable,
        };
    }
    clean.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.id.cmp(&b.0.id))
    });
    Ranking {
        ranked: clean
            .into_iter()
            .map(|(opt, utility, switched)| Scored {
                id: opt.id.clone(),
                utility,
                switched,
            })
            .collect(),
        impossible,
        unacceptable,
    }
}

/// Seeded softmax sampling over a ranking (design §4.2): `P_i(a) ∝
/// exp(U(a)/τ)`. The sample and the resulting index are both returned, so
/// replay reproduces the choice from the recorded seed. Empty rankings
/// sample to `None` (never an invalid option).
pub fn sample(ranking: &Ranking, params: Params, rng: &mut DeterministicRng) -> Option<Sampled> {
    if ranking.ranked.is_empty() {
        return None;
    }
    let tau = if params.tau.abs() < 1e-6 {
        1e-6
    } else {
        params.tau
    };
    let sample = rng.next_f32();
    // Stable exp-sum (subtract the max to avoid overflow; ties break by id,
    // which `rank` already guaranteed through its ordering).
    let max = ranking
        .ranked
        .iter()
        .map(|s| s.utility)
        .fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = ranking
        .ranked
        .iter()
        .map(|s| ((s.utility - max) / tau).exp())
        .collect();
    let total: f32 = weights.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        // Degenerate numerics: fall back to the deterministic best.
        let best = &ranking.ranked[0];
        return Some(Sampled {
            id: best.id.clone(),
            index: 0,
            sample,
        });
    }
    let mut acc = 0.0f32;
    for (i, w) in weights.iter().enumerate() {
        acc += w / total;
        if sample <= acc {
            return Some(Sampled {
                id: ranking.ranked[i].id.clone(),
                index: i,
                sample,
            });
        }
    }
    // Floating-point tail: the walk fell just short of 1.0.
    let best = &ranking.ranked[0];
    Some(Sampled {
        id: best.id.clone(),
        index: 0,
        sample,
    })
}
