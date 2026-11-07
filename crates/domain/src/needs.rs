//! Needs state `N` (design §3, plan I-4a): the physical/social needs of an
//! actor — safety, belonging, wealth, health, status — as plain serializable
//! state with configured per-need decay rates, satiation thresholds and
//! urgency mapping.
//!
//! Modelling choices (design §6.4 stability control, plan §9):
//! * Needs decay linearly toward 0 and are satisfied toward 1.
//! * **Satiation**: above the configured satiation point each further gain
//!   has diminishing effect (T3.8), so repeated feeding never overshoots.
//! * **Urgency with hysteresis**: a need whose value falls below its
//!   configured urgency threshold *latches* urgent; the latch releases only
//!   when the value recovers above the (higher) release threshold. Urgency
//!   while latched is bounded by a cap, so escalation is bounded and a
//!   need sitting just under the threshold cannot ping-pong the urgency
//!   every tick (T3.8 hysteresis check).
//! * Regular decay is classified as a **non-evented background mutation**
//!   (plan G4/M1); threshold crossings are event-sourced by the sim as
//!   typed `NeedThresholdCrossed` events, and the mapped goal-urgency raise
//!   they trigger is event-sourced as `GoalUrgencyRaised`.

use serde::{Deserialize, Serialize};

/// The five design needs (§3 `N`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum NeedKind {
    Safety,
    Belonging,
    Wealth,
    Health,
    Status,
}

pub const NEED_KINDS: [NeedKind; 5] = [
    NeedKind::Safety,
    NeedKind::Belonging,
    NeedKind::Wealth,
    NeedKind::Health,
    NeedKind::Status,
];

impl NeedKind {
    pub fn name(self) -> &'static str {
        match self {
            NeedKind::Safety => "safety",
            NeedKind::Belonging => "belonging",
            NeedKind::Wealth => "wealth",
            NeedKind::Health => "health",
            NeedKind::Status => "status",
        }
    }

    pub fn index(self) -> usize {
        match self {
            NeedKind::Safety => 0,
            NeedKind::Belonging => 1,
            NeedKind::Wealth => 2,
            NeedKind::Health => 3,
            NeedKind::Status => 4,
        }
    }

    pub fn from_index(i: usize) -> Option<NeedKind> {
        NEED_KINDS.get(i).copied()
    }
}

/// Tuning for one need (design choices, kept explicit per plan §9).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NeedTuning {
    /// Value lost per simulation tick (regular, non-evented decay).
    pub decay_per_tick: f32,
    /// Above this value further gains have diminishing returns.
    pub satiation: f32,
    /// A crossing below this value latches urgency (typed event).
    pub urgent_below: f32,
    /// The latch releases only above this (higher) value — hysteresis.
    pub urgent_release: f32,
    /// Maximum mapped urgency (bounds escalation; T3.8).
    pub urgency_cap: f32,
}

impl Default for NeedTuning {
    fn default() -> Self {
        NeedTuning {
            decay_per_tick: 0.002,
            satiation: 0.85,
            urgent_below: 0.35,
            urgent_release: 0.55,
            urgency_cap: 3.0,
        }
    }
}

/// Configuration for all five needs (part of the sim configuration).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NeedsConfig {
    pub tuning: [NeedTuning; 5],
}

impl NeedsConfig {
    pub fn tuning(&self, kind: NeedKind) -> &NeedTuning {
        &self.tuning[kind.index()]
    }
}

/// A threshold crossing reported by [`Needs::decay`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThresholdDirection {
    /// The need fell below its urgency threshold.
    FellBelow,
    /// The need recovered above the release threshold (hysteresis release).
    Recovered,
}

/// A need-threshold crossing detected while decaying.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NeedCrossing {
    pub kind: NeedKind,
    pub direction: ThresholdDirection,
    /// The value at the crossing (recorded for the typed event).
    pub value: f32,
}

/// Per-actor needs state (design §3 `N`). Values are `0.0..=1.0`; the
/// latched urgency flags are part of the state so the crossing events and
/// the urgency mapping are deterministic under replay.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Needs {
    values: [f32; 5],
    /// Latched urgent flags with hysteresis (see module docs).
    urgent_latched: [bool; 5],
}

impl Default for Needs {
    fn default() -> Self {
        Needs {
            values: [0.5; 5],
            urgent_latched: [false; 5],
        }
    }
}

impl Needs {
    pub fn value(&self, kind: NeedKind) -> f32 {
        self.values[kind.index()]
    }

    /// Direct set (authoritative maintenance; clamped).
    pub fn set(&mut self, kind: NeedKind, value: f32) {
        self.values[kind.index()] = value.clamp(0.0, 1.0);
    }

    /// Whether the need's urgency latch is currently engaged.
    pub fn is_urgent(&self, kind: NeedKind) -> bool {
        self.urgent_latched[kind.index()]
    }

    /// Diminishing-returns satisfaction (T3.8): each unit gained counts
    /// fully below the satiation point, then with a linearly shrinking
    /// factor toward the 1.0 ceiling.
    pub fn satisfy(&mut self, kind: NeedKind, amount: f32, config: &NeedsConfig) -> f32 {
        let t = config.tuning(kind).satiation.clamp(0.0, 1.0);
        let v = &mut self.values[kind.index()];
        let headroom_below_satiation = (t - *v).max(0.0);
        let gained_below = amount.min(headroom_below_satiation);
        let above = (amount - gained_below).max(0.0);
        let room_above = (1.0 - *v - gained_below).max(0.0);
        let gained_above = if t >= 1.0 {
            0.0
        } else {
            above * (room_above / (1.0 - t)).min(1.0)
        };
        *v = (*v + gained_below + gained_above).clamp(0.0, 1.0);
        gained_below + gained_above
    }

    /// One tick of regular decay. Returns the threshold crossings detected
    /// this tick (hysteresis: a crossing fires exactly once per excursion).
    /// The decay itself is the classified non-evented mutation; the caller
    /// event-sources the crossings.
    pub fn decay(&mut self, config: &NeedsConfig) -> Vec<NeedCrossing> {
        let mut crossings = Vec::new();
        for (i, kind) in NEED_KINDS.iter().enumerate() {
            let t = config.tuning(*kind);
            self.values[i] = (self.values[i] - t.decay_per_tick).max(0.0);
            let v = self.values[i];
            if !self.urgent_latched[i] && v < t.urgent_below {
                self.urgent_latched[i] = true;
                crossings.push(NeedCrossing {
                    kind: *kind,
                    direction: ThresholdDirection::FellBelow,
                    value: v,
                });
            } else if self.urgent_latched[i] && v >= t.urgent_release {
                self.urgent_latched[i] = false;
                crossings.push(NeedCrossing {
                    kind: *kind,
                    direction: ThresholdDirection::Recovered,
                    value: v,
                });
            }
        }
        crossings
    }

    /// The urgency currently mapped from this need onto goal selection
    /// (T3.8): 0 while not latched; while latched, a bounded ramp from the
    /// urgency threshold down to 0, capped by the tuning's urgency cap.
    pub fn urgency(&self, kind: NeedKind, config: &NeedsConfig) -> f32 {
        if !self.urgent_latched[kind.index()] {
            return 0.0;
        }
        let t = config.tuning(kind);
        let below = (t.urgent_below - self.values[kind.index()]).max(0.0);
        (below / t.urgent_below.max(f32::EPSILON) * t.urgency_cap).min(t.urgency_cap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decay_fires_crossing_once_with_hysteresis() {
        let config = NeedsConfig::default();
        let mut n = Needs::default();
        n.set(NeedKind::Wealth, 0.4);
        // Decay below the urgency threshold: exactly one FellBelow crossing.
        n.set(NeedKind::Wealth, 0.3);
        let c = n.decay(&config);
        assert_eq!(
            c,
            vec![NeedCrossing {
                kind: NeedKind::Wealth,
                direction: ThresholdDirection::FellBelow,
                value: 0.3 - config.tuning(NeedKind::Wealth).decay_per_tick,
            }]
        );
        // Latched: further decay fires nothing (no ping-pong while low).
        for _ in 0..5 {
            assert!(n.decay(&config).is_empty());
        }
        assert!(n.is_urgent(NeedKind::Wealth));
        // Recover above the release threshold: exactly one Recovered crossing.
        n.set(NeedKind::Wealth, 0.56);
        let c = n.decay(&config);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].direction, ThresholdDirection::Recovered);
        assert!(!n.is_urgent(NeedKind::Wealth));
    }

    #[test]
    fn urgency_is_bounded_and_zero_when_not_latched() {
        let config = NeedsConfig::default();
        let mut n = Needs::default();
        assert_eq!(n.urgency(NeedKind::Safety, &config), 0.0);
        n.set(NeedKind::Safety, 0.0);
        let _ = n.decay(&config);
        let u = n.urgency(NeedKind::Safety, &config);
        assert!(u > 0.0 && u <= config.tuning(NeedKind::Safety).urgency_cap);
    }

    #[test]
    fn satiation_has_diminishing_returns() {
        let config = NeedsConfig::default();
        let mut n = Needs::default();
        n.set(NeedKind::Belonging, 0.0);
        let first = n.satisfy(NeedKind::Belonging, 0.1, &config);
        n.set(NeedKind::Belonging, 0.9);
        let above = n.satisfy(NeedKind::Belonging, 0.1, &config);
        assert!((first - 0.1).abs() < 1e-6, "below satiation: full gain");
        assert!(
            above < first,
            "above satiation: diminishing ({above} vs {first})"
        );
    }
}
