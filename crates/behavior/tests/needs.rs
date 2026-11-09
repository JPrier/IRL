//! Phase 3 needs-state acceptance tests (test-spec T3.7, T3.8; plan I-4a):
//! the five design needs with configured decay rates, satiation thresholds,
//! urgency mapping into goal selection with bounded escalation (hysteresis),
//! and the needs-differential utility term (T3.9 overlap) — all through the
//! production kernels the sim maintenance and the utility ranking call.

use rpg_behavior::{rank_with_needs, Feasibility, Params, UtilityOption};
use rpg_domain::needs::{NeedKind, NeedTuning, Needs, NeedsConfig, ThresholdDirection};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn config() -> NeedsConfig {
    NeedsConfig::default()
}

// ---------------------------------------------------------------------------
// T3.7 — needs state `N`: five needs, configured per-need decay rates
// ---------------------------------------------------------------------------

#[test]
fn t3_7_actor_needs_decay_at_configured_per_need_rates() {
    let mut config = config();
    // Distinct decay rates: health rots twice as fast as safety.
    let health_tuning = NeedTuning {
        decay_per_tick: 2.0 * config.tuning(NeedKind::Safety).decay_per_tick,
        ..Default::default()
    };
    config.tuning[NeedKind::Health.index()] = health_tuning;

    let mut needs = Needs::default();
    let s0 = needs.value(NeedKind::Safety);
    let h0 = needs.value(NeedKind::Health);
    assert!(
        (s0 - h0).abs() < 1e-6,
        "both needs start at the default level"
    );

    needs.decay(&config);
    let s1 = needs.value(NeedKind::Safety);
    let h1 = needs.value(NeedKind::Health);
    assert!(
        (h0 - h1) > (s0 - s1) * 1.5,
        "health must decay faster than safety: {s0}→{s1} vs {h0}→{h1}"
    );
    // All five design needs are present and decayable.
    for kind in [
        NeedKind::Safety,
        NeedKind::Belonging,
        NeedKind::Wealth,
        NeedKind::Health,
        NeedKind::Status,
    ] {
        let v0 = needs.value(kind);
        let _ = needs.decay(&config);
        assert!(
            needs.value(kind) <= v0,
            "{} decays monotonically toward 0",
            kind.name()
        );
    }
}

/// T3.7 (event-sourcing side): the crossings the decay reports are exactly
/// the typed `NeedThresholdCrossed` payloads the sim event-sources; regular
/// decay produces none.
#[test]
fn t3_7_threshold_crossings_are_reported_exactly_once_per_excursion() {
    let config = config();
    let mut needs = Needs::default();
    needs.set(NeedKind::Status, 0.4); // above the 0.35 urgency threshold
                                      // Regular decay above the threshold: no crossings (declared non-evented).
    assert!(needs.decay(&config).is_empty());
    // Crossing below fires exactly one FellBelow crossing…
    needs.set(NeedKind::Status, 0.2);
    let crossings = needs.decay(&config);
    assert_eq!(crossings.len(), 1);
    assert_eq!(crossings[0].kind, NeedKind::Status);
    assert_eq!(crossings[0].direction, ThresholdDirection::FellBelow);
    // …and while it stays low, nothing further fires (no ping-pong).
    for _ in 0..20 {
        assert!(needs.decay(&config).is_empty());
    }
}

// ---------------------------------------------------------------------------
// T3.8 — satiation/thresholds: diminishing returns, bounded urgency with
// hysteresis, needs-biased goal selection
// ---------------------------------------------------------------------------

#[test]
fn t3_8_satiation_has_diminishing_returns_above_threshold() {
    let config = config();
    let mut needs = Needs::default();
    needs.set(NeedKind::Wealth, 0.0);
    let hungry_gain = needs.satisfy(NeedKind::Wealth, 0.2, &config);
    needs.set(NeedKind::Wealth, 0.9); // above the 0.85 satiation threshold
    let satiated_gain = needs.satisfy(NeedKind::Wealth, 0.2, &config);
    assert!(
        hungry_gain > satiated_gain,
        "the same 0.2 gain registers more when deprived: {hungry_gain} vs {satiated_gain}"
    );
    needs.set(NeedKind::Wealth, 0.99);
    let very_satiated = needs.satisfy(NeedKind::Wealth, 0.2, &config);
    assert!(
        very_satiated < satiated_gain && very_satiated > 0.0,
        "diminishing continues toward the ceiling: {very_satiated}"
    );
    assert!(needs.value(NeedKind::Wealth) <= 1.0, "never overshoots");
}

#[test]
fn t3_8_urgency_is_bounded_and_releases_with_hysteresis() {
    let config = config();
    let mut needs = Needs::default();
    // Not latched: no urgency regardless of value wobbling around the
    // threshold from above.
    needs.set(NeedKind::Safety, 0.36);
    needs.decay(&config);
    assert_eq!(needs.urgency(NeedKind::Safety, &config), 0.0);
    // Latched: urgency bounded by the cap, rising as the need starves.
    needs.set(NeedKind::Safety, 0.3);
    let _ = needs.decay(&config);
    let u = needs.urgency(NeedKind::Safety, &config);
    assert!(u > 0.0, "crossed needs map urgency");
    needs.set(NeedKind::Safety, 0.0);
    let u_max = needs.urgency(NeedKind::Safety, &config);
    assert!(
        u_max <= config.tuning(NeedKind::Safety).urgency_cap,
        "bounded"
    );
    assert!(u_max > u, "deeper deprivation maps more urgency");
    // Hysteresis: wobble below the urgent threshold does NOT re-fire or
    // release; only crossing the release threshold does.
    needs.set(NeedKind::Safety, 0.34);
    assert!(needs.decay(&config).is_empty());
    needs.set(NeedKind::Safety, 0.3);
    assert!(needs.decay(&config).is_empty());
    needs.set(NeedKind::Safety, 0.6); // above the 0.55 release threshold
    let crossings = needs.decay(&config);
    assert_eq!(crossings.len(), 1);
    assert_eq!(crossings[0].direction, ThresholdDirection::Recovered);
    assert_eq!(needs.urgency(NeedKind::Safety, &config), 0.0);
}

/// T3.8/T3.9: a need below its urgency threshold biases goal selection —
/// the deprived actor ranks the wealth-gaining option above the one a
/// satiated actor prefers. Two actors, same options, different ranking.
#[test]
fn t3_8_wealth_below_threshold_biases_toward_economic_options() {
    let config = NeedsConfig::default();
    let params = Params::default();
    // Option A: economic (wealth gain). Option B: comfortable (belonging).
    let options = [
        UtilityOption {
            id: "work_for_pay".into(),
            goal_delta: 1.0,
            cost: 0.0,
            feasibility: Feasibility::Possible,
            needs_gain: vec![(NeedKind::Wealth, 1.0)],
        },
        UtilityOption {
            id: "stay_with_friends".into(),
            goal_delta: 1.0,
            cost: 0.0,
            feasibility: Feasibility::Possible,
            needs_gain: vec![(NeedKind::Belonging, 1.0)],
        },
    ];
    let mut deprived = Needs::default();
    deprived.set(NeedKind::Wealth, 0.05); // wealth-starved
    let mut satiated = Needs::default();
    satiated.set(NeedKind::Wealth, 0.95); // wealth-satiated
    let ranked_deprived = rank_with_needs(&options, None, params, &deprived, &config);
    let ranked_satiated = rank_with_needs(&options, None, params, &satiated, &config);
    assert_eq!(ranked_deprived.ranked[0].id, "work_for_pay");
    assert_eq!(
        ranked_satiated.ranked[0].id, "stay_with_friends",
        "with wealth satiated, the economic option loses its edge"
    );
}

// ---------------------------------------------------------------------------
// T3.9 — the needs-differential utility term: same options, different need
// profiles ⇒ different ranking; lexicographic constraints still dominate;
// switch penalty still deters oscillation.
// ---------------------------------------------------------------------------

#[test]
fn t3_9_differential_ranking_actors_with_different_need_profiles() {
    let config = NeedsConfig::default();
    let params = Params::default();
    let options = [
        UtilityOption {
            id: "earn_money".into(),
            goal_delta: 1.0,
            cost: 0.0,
            feasibility: Feasibility::Possible,
            needs_gain: vec![(NeedKind::Wealth, 1.0)],
        },
        UtilityOption {
            id: "seek_company".into(),
            goal_delta: 1.0,
            cost: 0.0,
            feasibility: Feasibility::Possible,
            needs_gain: vec![(NeedKind::Belonging, 1.0)],
        },
    ];
    let mut lonely = Needs::default();
    lonely.set(NeedKind::Belonging, 0.0);
    let mut broke = Needs::default();
    broke.set(NeedKind::Wealth, 0.0);
    let lonely_rank = rank_with_needs(&options, None, params, &lonely, &config);
    let broke_rank = rank_with_needs(&options, None, params, &broke, &config);
    assert_eq!(lonely_rank.ranked[0].id, "seek_company");
    assert_eq!(broke_rank.ranked[0].id, "earn_money");
    // The *same* actor grows less greedy about wealth as it satiates:
    let mut changing = Needs::default();
    changing.set(NeedKind::Wealth, 0.0);
    let before = rank_with_needs(&options, None, params, &changing, &config).ranked[0].utility;
    changing.set(NeedKind::Wealth, 1.0);
    let after = rank_with_needs(&options, None, params, &changing, &config).ranked;
    let earn_after = after.iter().find(|s| s.id == "earn_money").unwrap().utility;
    assert!(
        before > earn_after,
        "satiating wealth drops the option's utility ({before} → {earn_after})"
    );
}

#[test]
fn t3_9_lexicographic_constraints_still_dominate_the_needs_term() {
    let config = NeedsConfig::default();
    let params = Params::default();
    let options = [
        UtilityOption {
            id: "hugely_rewarding_betrayal".into(),
            goal_delta: 1000.0,
            cost: 0.0,
            feasibility: Feasibility::PersonallyUnacceptable,
            needs_gain: vec![(NeedKind::Wealth, 1.0)],
        },
        UtilityOption {
            id: "honest_and_humble".into(),
            goal_delta: 0.1,
            cost: 0.0,
            feasibility: Feasibility::Possible,
            needs_gain: vec![(NeedKind::Wealth, -0.2)],
        },
    ];
    let mut starving = Needs::default();
    starving.set(NeedKind::Wealth, 0.0);
    let ranked = rank_with_needs(&options, None, params, &starving, &config);
    // Needs pressure cannot buy the lexicographic violation back.
    assert_eq!(ranked.ranked.len(), 1);
    assert_eq!(ranked.ranked[0].id, "honest_and_humble");
    assert_eq!(
        ranked.unacceptable,
        vec!["hugely_rewarding_betrayal".to_string()]
    );
}

#[test]
fn t3_9_switch_penalty_deters_oscillation_despite_needs_pressure() {
    let config = NeedsConfig::default();
    let params = Params {
        switch_penalty: 10.0,
        ..Default::default()
    };
    let options = [
        UtilityOption {
            id: "keep_working".into(),
            goal_delta: 1.0,
            cost: 0.0,
            feasibility: Feasibility::Possible,
            needs_gain: vec![(NeedKind::Wealth, 1.0)],
        },
        UtilityOption {
            id: "chase_status".into(),
            goal_delta: 1.2,
            cost: 0.0,
            feasibility: Feasibility::Possible,
            needs_gain: vec![(NeedKind::Status, 1.0)],
        },
    ];
    let mut needs = Needs::default();
    needs.set(NeedKind::Status, 0.0); // status gain is worth more…
    needs.set(NeedKind::Wealth, 0.0); // …but so is wealth.
    let current = Some("keep_working");
    let ranked = rank_with_needs(&options, current, params, &needs, &config);
    // The marginal-status gain (+0.2 raw) cannot overcome the switch
    // penalty: the standing intention continues.
    assert_eq!(ranked.ranked[0].id, "keep_working");
    assert!(!ranked.ranked[0].switched);
    assert!(ranked.ranked[1].switched);
}
