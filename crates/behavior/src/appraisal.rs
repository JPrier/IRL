//! Appraisal kernels (design §3.2, §11 Phase 3; plan I-4 + I-4b).
//!
//! The directed, contextual relationship update: for event `e`,
//!
//! ```text
//! ΔR_ij = f(e, V_i, B_i, goals_i, R_ij, context)
//! ```
//!
//! **No universal deltas.** Every delta is a function of the observer's
//! value profile `V_i` (extracted from authored personality/values by the
//! caller), the current edge `R_ij` (saturating), and the appraisal context
//! (including goal urgency from the needs state). Two actors perceiving the
//! same event produce different deltas (T3.3).
//!
//! Dimensions (design §3.2): trust-in-accuracy, trust-in-benevolence,
//! affection, respect, fear, anger, obligation, familiarity, perceived
//! power. Context separation (T3.2): an accuracy-relevant event moves
//! trust-in-accuracy without touching trust-in-benevolence, and vice versa.
//!
//! Updates are **bounded and saturating** (T3.4): a delta shrinks with the
//! remaining headroom toward ±1, so repeated identical events asymptote
//! instead of escalating. Unused dimensions **decay** toward 0 at
//! configured per-dimension rates.
//!
//! Forgiveness/retaliation (T3.5) is a deterministic policy over the
//! appraisal state (anger, affection, benevolence trust, profile), not a
//! hardcoded reaction; both paths stay reachable depending on state.
//!
//! Emotions (I-4b) are written by the same appraised events through
//! [`emotion_deltas`], but they are a separate mind-local state: an emotion
//! spike never moves the edge and an edge delta never writes an emotion by
//! itself (T3.12 separation).
//!
//! This kernel is pure: it never reads world truth, carries no actor-id
//! special cases, and mutates nothing — the social layer owns application
//! and provenance (T3.6).

use rpg_domain::emotions::EmotionKind;
use rpg_domain::world::Relationship;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// The appraised event (context, not world truth)
// ---------------------------------------------------------------------------

/// An event as the observer interpreted it (design §3.2: the *interpretation*
/// is the observer's own, and may differ from intent). Severity/value inputs
/// are `0.0..=1.0` scales supplied by the caller.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum AppraisedEvent {
    /// Received a gift or payment of `value`.
    GiftReceived { value: f32 },
    /// A claim of the other was verified accurate.
    AccurateClaim,
    /// The other's deception was exposed (to me).
    DeceptionExposed,
    /// Was threatened with severity.
    ThreatReceived { severity: f32 },
    /// Was comforted with perceived sincerity.
    ComfortReceived { sincerity: f32 },
    /// Was harmed with severity (culprit known).
    HarmInflicted { severity: f32 },
    /// A promise to me was kept.
    PromiseKept,
    /// A promise to me was broken.
    PromiseBroken,
    /// Was insulted (identity-relevant speech) with severity.
    InsultReceived { severity: f32 },
    /// Was praised.
    PraiseReceived,
    /// Was sanctioned by an institution (fined, arrested, boycotted) with
    /// severity — the SANCTION primitive's mind-local reading (plan I-5).
    Sanctioned { severity: f32 },
}

// ---------------------------------------------------------------------------
// The observer's value profile V_i
// ---------------------------------------------------------------------------

/// The observer's value profile (design §3 `V_i`). The social layer extracts
/// this from authored personality and value tags; the kernel only consumes
/// it, so the same event yields different deltas for different profiles.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppraisalProfile {
    /// How strongly material gain registers (greed).
    pub greed: f32,
    /// How strongly unsolicited gifts read as humiliation/bribery (pride).
    pub pride: f32,
    /// How much the threat of harm is discounted (bravery, inverse fear).
    pub bravery: f32,
    /// How strongly kindness/cruelty is repaid in kind.
    pub reciprocity: f32,
    /// How fast anger cools (forgiveness disposition).
    pub forgiveness: f32,
    /// Current safety-goal urgency from the needs state `goals_i` input:
    /// a deprived safety need amplifies fear appraisals.
    pub safety_urgency: f32,
}

impl Default for AppraisalProfile {
    fn default() -> Self {
        AppraisalProfile {
            greed: 0.5,
            pride: 0.5,
            bravery: 0.5,
            reciprocity: 0.5,
            forgiveness: 0.5,
            safety_urgency: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// The directed contextual edge R_ij
// ---------------------------------------------------------------------------

/// The contextual dimensions of `R_ij` (design §3.2). This view reads the
/// shared dims off the canonical [`Relationship`] plus the three
/// `serde(default)` contextual fields; it is a snapshot for scoring, not
/// owned state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContextualEdge {
    pub trust_accuracy: f32,
    pub trust_benevolence: f32,
    pub affection: f32,
    pub respect: f32,
    pub fear: f32,
    pub anger: f32,
    pub obligation: f32,
    pub familiarity: f32,
    pub perceived_power: f32,
}

impl ContextualEdge {
    pub fn from_relationship(rel: &Relationship) -> Self {
        ContextualEdge {
            trust_accuracy: rel.trust_accuracy,
            trust_benevolence: rel.trust_benevolence,
            affection: rel.affection,
            respect: rel.respect,
            fear: rel.fear,
            anger: rel.anger,
            obligation: rel.obligation,
            familiarity: rel.familiarity,
            perceived_power: rel.perceived_power,
        }
    }
}

/// One appraisal's directed deltas over the contextual dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct DirectedDelta {
    pub trust_accuracy: f32,
    pub trust_benevolence: f32,
    pub affection: f32,
    pub respect: f32,
    pub fear: f32,
    pub anger: f32,
    pub obligation: f32,
    pub familiarity: f32,
    pub perceived_power: f32,
}

impl DirectedDelta {
    /// True when the delta would not change the edge at all.
    pub fn is_zero(&self) -> bool {
        *self == DirectedDelta::default()
    }
}

/// Bounded, saturating accumulation (T3.4): a positive delta shrinks with
/// the headroom above the current value (and applies in full while moving
/// back toward 0 from the negative side); symmetric for negative deltas.
pub fn saturating_add(current: f32, delta: f32) -> f32 {
    if delta >= 0.0 {
        let headroom = (1.0 - current.max(0.0)).max(0.0);
        current + delta * headroom
    } else {
        let headroom = (1.0 + current.min(0.0)).max(0.0);
        current + delta * headroom
    }
}

/// Apply one appraisal's deltas with saturation and clamping.
pub fn apply_delta(edge: &ContextualEdge, delta: &DirectedDelta) -> ContextualEdge {
    ContextualEdge {
        trust_accuracy: saturating_add(edge.trust_accuracy, delta.trust_accuracy).clamp(-1.0, 1.0),
        trust_benevolence: saturating_add(edge.trust_benevolence, delta.trust_benevolence)
            .clamp(-1.0, 1.0),
        affection: saturating_add(edge.affection, delta.affection).clamp(-1.0, 1.0),
        respect: saturating_add(edge.respect, delta.respect).clamp(-1.0, 1.0),
        fear: saturating_add(edge.fear, delta.fear).clamp(0.0, 1.0),
        anger: saturating_add(edge.anger, delta.anger).clamp(0.0, 1.0),
        obligation: saturating_add(edge.obligation, delta.obligation).clamp(0.0, 1.0),
        familiarity: saturating_add(edge.familiarity, delta.familiarity).clamp(0.0, 1.0),
        perceived_power: saturating_add(edge.perceived_power, delta.perceived_power)
            .clamp(0.0, 1.0),
    }
}

/// Per-dimension decay rates toward 0 (design §3.2: "decay at variable
/// rates"). A dimension only decays when it is not being exercised, so the
/// caller applies decay in maintenance; an appraisal first, then decay keeps
/// the ordering explicit.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecayRates {
    pub trust_accuracy: f32,
    pub trust_benevolence: f32,
    pub affection: f32,
    pub respect: f32,
    pub fear: f32,
    pub anger: f32,
    pub obligation: f32,
    pub familiarity: f32,
    pub perceived_power: f32,
}

impl Default for DecayRates {
    fn default() -> Self {
        // Modeling assumptions (plan §9): feelings cool faster than facts.
        DecayRates {
            trust_accuracy: 0.001,
            trust_benevolence: 0.001,
            affection: 0.002,
            respect: 0.001,
            fear: 0.004,
            anger: 0.006,
            obligation: 0.0015,
            familiarity: 0.0,
            perceived_power: 0.0,
        }
    }
}

impl DecayRates {
    fn rate(&self, dim: usize) -> f32 {
        match dim {
            0 => self.trust_accuracy,
            1 => self.trust_benevolence,
            2 => self.affection,
            3 => self.respect,
            4 => self.fear,
            5 => self.anger,
            6 => self.obligation,
            7 => self.familiarity,
            _ => self.perceived_power,
        }
    }
}

/// Decay one edge's dimensions toward 0 at the configured rates (T3.4b).
/// Never flips a sign: a decaying feeling cools to neutral, not to its
/// opposite.
pub fn decay_edge(edge: &mut ContextualEdge, rates: &DecayRates) {
    let dims: [&mut f32; 9] = [
        &mut edge.trust_accuracy,
        &mut edge.trust_benevolence,
        &mut edge.affection,
        &mut edge.respect,
        &mut edge.fear,
        &mut edge.anger,
        &mut edge.obligation,
        &mut edge.familiarity,
        &mut edge.perceived_power,
    ];
    for (i, v) in dims.into_iter().enumerate() {
        let rate = rates.rate(i);
        if rate <= 0.0 || v.abs() <= rate {
            continue;
        }
        *v -= v.signum() * rate;
    }
}

// ---------------------------------------------------------------------------
// The appraisal function: ΔR_ij = f(e, V_i, R_ij, context)
// ---------------------------------------------------------------------------

/// Appraise one event for the observer: compute the directed deltas from the
/// observer's profile (`V_i`), the current edge (`R_ij` conditions fear and
/// anger — a feared/powerful aggressor frightens more, a loved one's affront
/// angers less), and the event itself. Pure, deterministic, and explicitly
/// *not* universal (T3.3): profiles with different greed/pride/bravery/
/// reciprocity produce different deltas for the same event.
pub fn appraise(
    event: &AppraisedEvent,
    profile: &AppraisalProfile,
    edge: &ContextualEdge,
) -> DirectedDelta {
    let mut d = DirectedDelta::default();
    match event {
        AppraisedEvent::GiftReceived { value } => {
            let v = value.clamp(0.0, 1.0);
            // Gratitude scales with greed; pride discounts it (and can invert
            // it into humiliation/bribery appraisal: respect and affection
            // fall while obligation — the hook of a bribe — still rises).
            d.affection = (0.25 + 0.35 * profile.greed - 0.45 * profile.pride) * v;
            d.respect = -0.15 * profile.pride * v;
            d.obligation = (0.25 + 0.2 * profile.greed) * v;
            d.trust_benevolence = 0.1 * v * (1.0 - profile.pride);
        }
        AppraisedEvent::AccurateClaim => {
            // Accuracy is *contextual*: this is trust-in-accuracy's event.
            // It says nothing about benevolence (T3.2 separation).
            d.trust_accuracy = 0.2;
            d.respect = 0.05;
            d.familiarity = 0.02;
        }
        AppraisedEvent::DeceptionExposed => {
            // A deliberate deception is a benevolence betrayal; anger scales
            // with reciprocity. Accuracy drops less than benevolence: the
            // claim was *false* here, but the dimension most betrayed is
            // good faith.
            d.trust_benevolence = -0.4;
            d.trust_accuracy = -0.15;
            d.anger = 0.15 + 0.25 * profile.reciprocity;
            d.respect = -0.2;
            d.affection = -0.2;
        }
        AppraisedEvent::ThreatReceived { severity } => {
            let s = severity.clamp(0.0, 1.0);
            // Fear scales inversely with bravery and with the observer's
            // current safety urgency (needs state feeds appraisal).
            d.fear = (0.3 + 0.35 * (1.0 - profile.bravery) + 0.3 * profile.safety_urgency) * s;
            d.anger = (0.1 + 0.2 * profile.reciprocity * (1.0 - profile.bravery * 0.5)) * s;
            d.trust_benevolence = -0.2 * s;
            d.perceived_power = 0.1 * s;
            d.familiarity = 0.02 * s;
        }
        AppraisedEvent::ComfortReceived { sincerity } => {
            let s = sincerity.clamp(0.0, 1.0);
            d.affection = (0.1 + 0.15 * profile.reciprocity) * s;
            d.anger = -0.15 * s;
            d.fear = -0.1 * s;
            d.trust_benevolence = 0.1 * s;
            d.obligation = 0.05 * s;
        }
        AppraisedEvent::HarmInflicted { severity } => {
            let s = severity.clamp(0.0, 1.0);
            d.fear = (0.25 + 0.3 * (1.0 - profile.bravery) + 0.3 * profile.safety_urgency) * s;
            d.anger = (0.25 + 0.35 * profile.reciprocity) * s;
            d.trust_benevolence = -0.5 * s;
            d.affection = -0.3 * s;
            d.respect = -0.15 * s;
            d.perceived_power = 0.15 * s;
        }
        AppraisedEvent::PromiseKept => {
            d.trust_benevolence = 0.15;
            d.trust_accuracy = 0.05;
            d.respect = 0.05;
        }
        AppraisedEvent::PromiseBroken => {
            d.trust_benevolence = -0.25;
            d.anger = 0.1 + 0.2 * profile.reciprocity;
            d.respect = -0.1;
            d.obligation = -0.1;
        }
        AppraisedEvent::InsultReceived { severity } => {
            let s = severity.clamp(0.0, 1.0);
            d.anger = (0.2 + 0.25 * profile.reciprocity) * s;
            d.respect = -0.2 * s;
            d.affection = -0.15 * s;
        }
        AppraisedEvent::PraiseReceived => {
            d.affection = 0.1;
            d.respect = 0.05;
        }
        AppraisedEvent::Sanctioned { severity } => {
            let s = severity.clamp(0.0, 1.0);
            // Being sanctioned by a real authority: fear scales with
            // vulnerability (bravery, safety urgency); anger with
            // reciprocity; good faith in the sanctioner drops a little.
            d.fear = (0.2 + 0.2 * (1.0 - profile.bravery) + 0.2 * profile.safety_urgency) * s;
            d.anger = (0.1 + 0.15 * profile.reciprocity) * s;
            d.trust_benevolence = -0.1 * s;
            d.respect = -0.1 * s;
            d.obligation = 0.1 * s;
        }
    }
    // R_ij context: a feared, powerful actor frightens more; a loved one's
    // affront stirs less cold anger. Fear-of-them amplifies fear; warmth
    // moderates it.
    if d.fear != 0.0 {
        d.fear *= 1.0 + 0.3 * edge.perceived_power;
    }
    if d.anger > 0.0 {
        d.anger *= 1.0 - 0.25 * edge.affection.max(0.0);
    }
    // Interaction itself deepens familiarity regardless of valence.
    d.familiarity += 0.02;
    d
}

// ---------------------------------------------------------------------------
// Emotion deltas (I-4b): the APPRAISE effect's mind-local writes
// ---------------------------------------------------------------------------

/// Emotion deltas for one appraised event: what the *experience* did to the
/// observer's temporary emotion state. This is a separate output from the
/// relationship deltas by design (T3.12 separation): either can be nonzero
/// without the other (e.g. `AccurateClaim` moves the edge's accuracy trust
/// but no listed emotion; `PraiseReceived` raises pride — emotion — with
/// only mild edge movement).
pub fn emotion_deltas(
    event: &AppraisedEvent,
    profile: &AppraisalProfile,
) -> [(EmotionKind, f32); 3] {
    match event {
        AppraisedEvent::GiftReceived { value } => {
            let v = value.clamp(0.0, 1.0);
            [
                (
                    EmotionKind::Joy,
                    (0.2 + 0.2 * profile.greed - 0.2 * profile.pride) * v,
                ),
                (EmotionKind::Pride, -0.1 * profile.pride * v),
                (EmotionKind::Shame, 0.05 * profile.pride * v),
            ]
        }
        AppraisedEvent::AccurateClaim => {
            // Neutral information: an edge delta with no emotion spike —
            // the separation of T3.12 in the other direction.
            [
                (EmotionKind::Joy, 0.0),
                (EmotionKind::Grief, 0.0),
                (EmotionKind::Fear, 0.0),
            ]
        }
        AppraisedEvent::DeceptionExposed => [
            (EmotionKind::Anger, 0.2 + 0.2 * profile.reciprocity),
            (EmotionKind::Grief, 0.1),
            (EmotionKind::Fear, 0.05),
        ],
        AppraisedEvent::ThreatReceived { severity } => {
            let s = severity.clamp(0.0, 1.0);
            [
                (
                    EmotionKind::Fear,
                    (0.35 + 0.35 * (1.0 - profile.bravery) + 0.3 * profile.safety_urgency) * s,
                ),
                (
                    EmotionKind::Anger,
                    (0.15 + 0.2 * profile.reciprocity * (1.0 - profile.bravery * 0.5)) * s,
                ),
                (EmotionKind::Joy, -0.1 * s),
            ]
        }
        AppraisedEvent::ComfortReceived { sincerity } => {
            let s = sincerity.clamp(0.0, 1.0);
            [
                (EmotionKind::Joy, 0.15 * s),
                (EmotionKind::Grief, -0.2 * s),
                (EmotionKind::Fear, -0.1 * s),
            ]
        }
        AppraisedEvent::HarmInflicted { severity } => {
            let s = severity.clamp(0.0, 1.0);
            [
                (
                    EmotionKind::Fear,
                    (0.3 + 0.3 * (1.0 - profile.bravery) + 0.3 * profile.safety_urgency) * s,
                ),
                (EmotionKind::Anger, (0.3 + 0.3 * profile.reciprocity) * s),
                (EmotionKind::Grief, 0.2 * s),
            ]
        }
        AppraisedEvent::PromiseKept => [
            (EmotionKind::Joy, 0.1),
            (EmotionKind::Grief, 0.0),
            (EmotionKind::Fear, 0.0),
        ],
        AppraisedEvent::PromiseBroken => [
            (EmotionKind::Anger, 0.15 + 0.15 * profile.reciprocity),
            (EmotionKind::Grief, 0.1),
            (EmotionKind::Joy, -0.1),
        ],
        AppraisedEvent::InsultReceived { severity } => {
            let s = severity.clamp(0.0, 1.0);
            [
                (EmotionKind::Anger, (0.25 + 0.2 * profile.reciprocity) * s),
                (EmotionKind::Shame, 0.3 * s * (1.0 - profile.pride * 0.5)),
                (EmotionKind::Joy, -0.1 * s),
            ]
        }
        AppraisedEvent::PraiseReceived => [
            (EmotionKind::Pride, 0.3),
            (EmotionKind::Joy, 0.2),
            (EmotionKind::Shame, -0.1),
        ],
        AppraisedEvent::Sanctioned { severity } => {
            let s = severity.clamp(0.0, 1.0);
            [
                (
                    EmotionKind::Fear,
                    (0.25 + 0.25 * (1.0 - profile.bravery)) * s,
                ),
                (EmotionKind::Anger, (0.15 + 0.15 * profile.reciprocity) * s),
                (EmotionKind::Shame, 0.2 * s),
            ]
        }
    }
}

// ---------------------------------------------------------------------------
// Forgiveness / retaliation (T3.5)
// ---------------------------------------------------------------------------

/// What the observer's current state inclines them toward after an
/// affront/harm: forgive (repair the tie), retaliate, or hold the grievance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Aftermath {
    /// Anger has cooled enough and the tie was warm: the forgiveness path
    /// (accept apologies/reparations, anger decays faster).
    Forgive,
    /// Anger is high and the profile reciprocates harm (or the other was
    /// never trusted as benevolent): the retaliation path.
    Retaliate,
    /// Neither: hold the grievance (no repair, no strike).
    Hold,
}

/// Thresholds for the aftermath policy (modeling assumptions, explicit).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AftermathThresholds {
    /// Anger at/above this may trigger retaliation.
    pub retaliate_anger: f32,
    /// Anger below this (and a warm tie) permits forgiveness.
    pub forgive_anger: f32,
    /// Affection at/above this makes forgiveness available.
    pub forgive_affection: f32,
}

impl Default for AftermathThresholds {
    fn default() -> Self {
        AftermathThresholds {
            retaliate_anger: 0.55,
            forgive_anger: 0.25,
            forgive_affection: 0.2,
        }
    }
}

/// Deterministic aftermath policy over the appraisal state (T3.5): both
/// paths are reachable by state — high anger + reciprocating profile ⇒
/// retaliate; cooled anger + warm affection ⇒ forgive; else hold.
pub fn aftermath(
    edge: &ContextualEdge,
    profile: &AppraisalProfile,
    thresholds: &AftermathThresholds,
) -> Aftermath {
    if edge.anger >= thresholds.retaliate_anger
        && (profile.reciprocity >= 0.5 || edge.trust_benevolence < 0.0)
    {
        return Aftermath::Retaliate;
    }
    if edge.anger < thresholds.forgive_anger && edge.affection >= thresholds.forgive_affection {
        return Aftermath::Forgive;
    }
    Aftermath::Hold
}

/// The forgiveness path accelerates anger decay (disposition-scaled);
/// retaliation does not. Returns the decay rate to use this tick.
pub fn aftermath_anger_decay(
    base_rate: f32,
    outcome: Aftermath,
    profile: &AppraisalProfile,
) -> f32 {
    match outcome {
        Aftermath::Forgive => base_rate * (1.0 + profile.forgiveness),
        _ => base_rate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge() -> ContextualEdge {
        ContextualEdge {
            trust_accuracy: 0.0,
            trust_benevolence: 0.0,
            affection: 0.0,
            respect: 0.0,
            fear: 0.0,
            anger: 0.0,
            obligation: 0.0,
            familiarity: 0.0,
            perceived_power: 0.0,
        }
    }

    #[test]
    fn saturating_reaches_no_further_than_one() {
        let mut e = edge();
        let delta = DirectedDelta {
            affection: 0.3,
            ..Default::default()
        };
        for _ in 0..50 {
            e = apply_delta(&e, &delta);
        }
        assert!(
            e.affection <= 1.0 && e.affection > 0.99,
            "asymptote: {}",
            e.affection
        );
    }

    #[test]
    fn decay_does_not_flip_sign() {
        let rates = DecayRates::default();
        let mut e = edge();
        e.affection = 0.01;
        decay_edge(&mut e, &rates);
        assert!(e.affection >= 0.0);
        e.affection = -0.01;
        decay_edge(&mut e, &rates);
        assert!(e.affection <= 0.0);
    }
}
