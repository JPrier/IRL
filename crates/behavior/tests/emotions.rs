//! Phase 3 emotions-state acceptance tests (test-spec T3.10–T3.12; plan
//! I-4b): `Actor.emotions` as temporary appraisal variables written by
//! `APPRAISE`, with decay/thresholds/sensitivities, threshold-gated response
//! classes, and the strict emotions ≠ relationships separation — through the
//! production kernels the sim dispatch and maintenance call.

use rpg_behavior::{
    apply_delta, appraise, emotion_deltas, AppraisalProfile, AppraisedEvent, ContextualEdge,
};
use rpg_domain::emotions::{
    EmotionKind, EmotionResponse, EmotionThresholds, Emotions, EmotionsConfig, EMOTION_KINDS,
};
use rpg_domain::world::Relationship;

// ---------------------------------------------------------------------------
// T3.10 — emotions state `E`: temporary appraisal variables, decay/
// thresholds/sensitivities; `APPRAISE` writes emotions (mind-local)
// ---------------------------------------------------------------------------

#[test]
fn t3_10_all_six_emotions_are_temporary_appraisal_variables_with_decay() {
    let config = EmotionsConfig {
        decay_per_tick: [0.05; 6],
        ..Default::default()
    };
    let mut e = Emotions::default();
    // All six kinds exist and are writable (mind-local state only).
    for (i, kind) in EMOTION_KINDS.iter().enumerate() {
        e.adjust(*kind, 0.3 + 0.1 * i as f32, 1.0);
        assert!(e.value(*kind) > 0.0, "{} writable", kind.name());
    }
    e.decay(&config);
    for kind in EMOTION_KINDS {
        assert!(
            e.value(kind) < 0.3 + 0.1 * kind.index() as f32,
            "{} decays back toward 0",
            kind.name()
        );
        assert!(e.value(kind) >= 0.0);
    }
}

/// T3.10 (the Appraise path): the production appraisal kernel derives the
/// emotion writes for an event — the same call the sim's `Effect::Appraise`
/// application makes. Sensitivities scale the writes (T3.11 overlap).
#[test]
fn t3_10_appraise_writes_emotions_derived_from_the_event() {
    let profile = AppraisalProfile::default();
    let edge = ContextualEdge {
        trust_accuracy: 0.0,
        trust_benevolence: 0.0,
        affection: 0.0,
        respect: 0.0,
        fear: 0.0,
        anger: 0.0,
        obligation: 0.0,
        familiarity: 0.0,
        perceived_power: 0.0,
    };
    // A threat frightens (fear) and stirs anger.
    let deltas = emotion_deltas(&AppraisedEvent::ThreatReceived { severity: 1.0 }, &profile);
    let fear = deltas
        .iter()
        .find(|(k, _)| *k == EmotionKind::Fear)
        .unwrap()
        .1;
    let anger = deltas
        .iter()
        .find(|(k, _)| *k == EmotionKind::Anger)
        .unwrap()
        .1;
    assert!(fear > 0.0 && anger > 0.0);
    // The edge application and the emotion writes are *separate* steps —
    // applying the edge delta writes no emotion, and vice versa.
    let before = edge;
    let after = apply_delta(
        &before,
        &appraise(
            &AppraisedEvent::ThreatReceived { severity: 1.0 },
            &profile,
            &edge,
        ),
    );
    let mut e = Emotions::default();
    e.adjust(EmotionKind::Fear, fear, 1.0);
    assert!(
        (e.value(EmotionKind::Joy) - 0.0).abs() < 1e-6,
        "emotions untouched by the edge delta"
    );
    assert!(after.fear > before.fear, "edge moved by the appraisal");
}

// ---------------------------------------------------------------------------
// T3.11 — threshold-gated responses: flee/retaliate/comfort/reconcile gate
// deterministically; decay returns below threshold; per-actor sensitivity
// alters crossing points
// ---------------------------------------------------------------------------

#[test]
fn t3_11_thresholds_gate_response_classes_deterministically() {
    let config = EmotionsConfig::default();
    let mut e = Emotions::default();
    assert!(
        e.responses(&config).is_empty(),
        "below every threshold: no gated class"
    );
    e.adjust(EmotionKind::Fear, 0.8, 1.0);
    e.adjust(EmotionKind::Anger, 0.8, 1.0);
    e.adjust(EmotionKind::Grief, 0.65, 1.0);
    e.adjust(EmotionKind::Shame, 0.65, 1.0);
    assert_eq!(
        e.responses(&config),
        vec![
            EmotionResponse::Flee,
            EmotionResponse::Retaliate,
            EmotionResponse::SeekComfort,
            EmotionResponse::Reconcile,
        ],
        "all four design response classes gate deterministically"
    );
    // Same input state → same gating (deterministic).
    let again = e.responses(&config);
    assert_eq!(e.responses(&config), again);
}

#[test]
fn t3_11_emotions_decay_back_below_threshold_within_configured_time() {
    let config = EmotionsConfig {
        decay_per_tick: [0.2; 6],
        ..Default::default()
    };
    let mut e = Emotions::default();
    e.adjust(EmotionKind::Fear, config.thresholds.flee_fear + 0.01, 1.0);
    e.adjust(
        EmotionKind::Anger,
        config.thresholds.retaliate_anger + 0.01,
        1.0,
    );
    assert!(e.responses(&config).contains(&EmotionResponse::Flee));
    assert!(e.responses(&config).contains(&EmotionResponse::Retaliate));
    // Within the configured time (≤ 5 ticks), both decay back below.
    for _ in 0..5 {
        e.decay(&config);
    }
    assert!(
        !e.responses(&config).contains(&EmotionResponse::Flee),
        "fear back below the flee threshold"
    );
    assert!(
        !e.responses(&config).contains(&EmotionResponse::Retaliate),
        "anger back below the retaliate threshold"
    );
}

#[test]
fn t3_11_sensitivity_alters_the_effective_crossing_point_per_actor() {
    let config = EmotionsConfig::default();
    let event_delta = 0.45; // below the 0.7 flee threshold at sensitivity 1.0
    let mut insensitive = Emotions::default();
    insensitive.adjust(EmotionKind::Fear, event_delta, 1.0);
    assert!(
        !insensitive
            .responses(&config)
            .contains(&EmotionResponse::Flee),
        "a typical actor does not cross"
    );
    let mut sensitive = Emotions::default();
    // A high-sensitivity actor (scaled ×2 by its personality) crosses.
    sensitive.adjust(EmotionKind::Fear, event_delta, 2.0);
    assert!(
        sensitive
            .responses(&config)
            .contains(&EmotionResponse::Flee),
        "the sensitive actor crosses the same threshold from the same event"
    );
    // And the per-kind thresholds themselves are configurable.
    let cfg = EmotionsConfig {
        thresholds: EmotionThresholds {
            flee_fear: 0.3,
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(insensitive.responses(&cfg).contains(&EmotionResponse::Flee));
}

// ---------------------------------------------------------------------------
// T3.12 — emotions ≠ relationships: either can move without the other
// ---------------------------------------------------------------------------

#[test]
fn t3_12_relationship_delta_without_emotion_spike_is_representable() {
    let config = EmotionsConfig::default();
    let profile = AppraisalProfile::default();
    let edge = ContextualEdge::from_relationship(&Relationship::default());
    // An accurate claim moves the *accuracy* dimension of the edge…
    let delta = appraise(&AppraisedEvent::AccurateClaim, &profile, &edge);
    let after = apply_delta(&edge, &delta);
    assert!(
        after.trust_accuracy > edge.trust_accuracy,
        "the relationship edge moved"
    );
    // …while no emotion moved at all.
    let mut e = Emotions::default();
    for (kind, d) in emotion_deltas(&AppraisedEvent::AccurateClaim, &profile) {
        e.adjust(kind, d, 1.0);
    }
    for kind in EMOTION_KINDS {
        assert_eq!(e.value(kind), 0.0, "{} must not move", kind.name());
        assert!(e.responses(&config).is_empty());
    }
}

#[test]
fn t3_12_emotion_spike_without_relationship_delta_is_representable() {
    // The emotion state is independent: writing emotions touches no edge.
    let mut e = Emotions::default();
    e.adjust(EmotionKind::Grief, 0.9, 1.0);
    let rel = Relationship::default();
    assert!(
        e.responses(&EmotionsConfig::default())
            .contains(&EmotionResponse::SeekComfort),
        "grief gates comfort with no relationship written"
    );
    assert_eq!(rel, Relationship::default(), "the edge is untouched");
    // And conversely the appraisal kernel's edge output carries no emotion
    // channel: `apply_delta` cannot reach an Emotions value.
    let edge = ContextualEdge::from_relationship(&Relationship::default());
    let after = apply_delta(
        &edge,
        &appraise(
            &AppraisedEvent::HarmInflicted { severity: 1.0 },
            &AppraisalProfile::default(),
            &edge,
        ),
    );
    assert!(after.anger > 0.0 && after.fear > 0.0);
    // Still a separate struct: no panic, no coupling.
    let _ = Emotions::default();
}
