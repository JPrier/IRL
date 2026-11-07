//! Emotions state `E` (design §3, plan I-4b): the actor's **temporary
//! appraisal variables** — fear, anger, joy, grief, pride, shame — as plain
//! serializable state with configured decay rates, response thresholds and
//! per-kind sensitivities.
//!
//! Separation (design §3 separation 7, T3.12): emotions are *mind-local*.
//! They are written by `APPRAISE` effects and decay over time; they never
//! mutate world truth, and an emotion spike does not by itself move a
//! relationship edge (and vice versa). Threshold crossings **gate response
//! classes** (flee / retaliate / seek comfort / reconcile) deterministically
//! instead of hardcoded reactions (T3.11); the response classes are what a
//! decision layer consumes.

use serde::{Deserialize, Serialize};

/// The six design emotions (§3 `E`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EmotionKind {
    Fear,
    Anger,
    Joy,
    Grief,
    Pride,
    Shame,
}

pub const EMOTION_KINDS: [EmotionKind; 6] = [
    EmotionKind::Fear,
    EmotionKind::Anger,
    EmotionKind::Joy,
    EmotionKind::Grief,
    EmotionKind::Pride,
    EmotionKind::Shame,
];

impl EmotionKind {
    pub fn name(self) -> &'static str {
        match self {
            EmotionKind::Fear => "fear",
            EmotionKind::Anger => "anger",
            EmotionKind::Joy => "joy",
            EmotionKind::Grief => "grief",
            EmotionKind::Pride => "pride",
            EmotionKind::Shame => "shame",
        }
    }

    pub fn index(self) -> usize {
        match self {
            EmotionKind::Fear => 0,
            EmotionKind::Anger => 1,
            EmotionKind::Joy => 2,
            EmotionKind::Grief => 3,
            EmotionKind::Pride => 4,
            EmotionKind::Shame => 5,
        }
    }

    pub fn from_index(i: usize) -> Option<EmotionKind> {
        EMOTION_KINDS.get(i).copied()
    }
}

/// Deterministically gated response classes (design §11 Phase 3: NPCs can
/// flee, reject, retaliate or reconcile *according to state*).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmotionResponse {
    /// Fear crossed its threshold: flee/avoid is available.
    Flee,
    /// Anger crossed its threshold: retaliate is available.
    Retaliate,
    /// Grief crossed its threshold: seek/accept comfort is available.
    SeekComfort,
    /// Shame crossed its threshold: reconcile/make amends is available.
    Reconcile,
}

/// Response thresholds (design choices, explicit per plan §9).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmotionThresholds {
    pub flee_fear: f32,
    pub retaliate_anger: f32,
    pub comfort_grief: f32,
    pub reconcile_shame: f32,
}

impl Default for EmotionThresholds {
    fn default() -> Self {
        EmotionThresholds {
            flee_fear: 0.7,
            retaliate_anger: 0.7,
            comfort_grief: 0.6,
            reconcile_shame: 0.6,
        }
    }
}

/// Configuration for the emotions state (part of the sim configuration):
/// per-kind decay rates and base sensitivities (a sensitivity scales the
/// deltas an appraisal produces for this actor, which is how per-actor
/// sensitivity alters where thresholds effectively cross — T3.11).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmotionsConfig {
    pub decay_per_tick: [f32; 6],
    pub sensitivity: [f32; 6],
    pub thresholds: EmotionThresholds,
}

impl Default for EmotionsConfig {
    fn default() -> Self {
        EmotionsConfig {
            decay_per_tick: [0.01; 6],
            sensitivity: [1.0; 6],
            thresholds: EmotionThresholds::default(),
        }
    }
}

/// Per-actor emotions state (design §3 `E`). Values are `0.0..=1.0` and
/// decay toward 0.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Emotions {
    values: [f32; 6],
}

impl Default for Emotions {
    fn default() -> Self {
        Emotions { values: [0.0; 6] }
    }
}

impl Emotions {
    pub fn value(&self, kind: EmotionKind) -> f32 {
        self.values[kind.index()]
    }

    /// Write an appraised delta, scaled by the actor's sensitivity for the
    /// kind (base config × per-actor scale, supplied by the caller) and
    /// clamped to `0.0..=1.0`. Mind-local by construction: this touches no
    /// world state.
    pub fn adjust(&mut self, kind: EmotionKind, delta: f32, sensitivity: f32) {
        let v = &mut self.values[kind.index()];
        *v = (*v + delta * sensitivity.max(0.0)).clamp(0.0, 1.0);
    }

    /// One tick of decay (classified non-evented background mutation).
    pub fn decay(&mut self, config: &EmotionsConfig) {
        for (i, v) in self.values.iter_mut().enumerate() {
            *v = (*v - config.decay_per_tick[i]).max(0.0);
        }
    }

    /// The response classes whose thresholds this state currently gates.
    /// Deterministic, ordered by kind index.
    pub fn responses(&self, config: &EmotionsConfig) -> Vec<EmotionResponse> {
        let t = &config.thresholds;
        let mut out = Vec::new();
        if self.value(EmotionKind::Fear) >= t.flee_fear {
            out.push(EmotionResponse::Flee);
        }
        if self.value(EmotionKind::Anger) >= t.retaliate_anger {
            out.push(EmotionResponse::Retaliate);
        }
        if self.value(EmotionKind::Grief) >= t.comfort_grief {
            out.push(EmotionResponse::SeekComfort);
        }
        if self.value(EmotionKind::Shame) >= t.reconcile_shame {
            out.push(EmotionResponse::Reconcile);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjust_scales_by_sensitivity_and_clamps() {
        let mut e = Emotions::default();
        e.adjust(EmotionKind::Fear, 0.4, 1.0);
        assert!((e.value(EmotionKind::Fear) - 0.4).abs() < 1e-6);
        e.adjust(EmotionKind::Fear, 0.4, 1.5);
        assert!(
            (e.value(EmotionKind::Fear) - 1.0).abs() < 1e-6,
            "clamped at 1"
        );
    }

    #[test]
    fn responses_gate_deterministically() {
        let config = EmotionsConfig::default();
        let mut e = Emotions::default();
        assert!(e.responses(&config).is_empty());
        e.adjust(EmotionKind::Fear, 0.75, 1.0);
        e.adjust(EmotionKind::Shame, 0.65, 1.0);
        assert_eq!(
            e.responses(&config),
            vec![EmotionResponse::Flee, EmotionResponse::Reconcile]
        );
    }

    #[test]
    fn decay_returns_below_threshold() {
        let config = EmotionsConfig {
            decay_per_tick: [0.1; 6],
            ..Default::default()
        };
        let mut e = Emotions::default();
        e.adjust(EmotionKind::Fear, 0.8, 1.0);
        assert!(e.responses(&config).contains(&EmotionResponse::Flee));
        for _ in 0..3 {
            e.decay(&config);
        }
        assert!(
            !e.responses(&config).contains(&EmotionResponse::Flee),
            "fear decayed back below the flee threshold"
        );
    }
}
