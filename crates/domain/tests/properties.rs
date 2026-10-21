//! Property-based invariants for the shared domain model.

use proptest::prelude::*;
use rpg_domain::geom::Vec2;
use rpg_domain::rng::DeterministicRng;
use rpg_domain::world::Relationship;

proptest! {
    #[test]
    fn rng_f32_always_in_unit_range(seed in any::<u64>(), n in 1usize..500) {
        let mut rng = DeterministicRng::new(seed);
        for _ in 0..n {
            let v = rng.next_f32();
            prop_assert!((0.0..1.0).contains(&v), "out of range: {v}");
        }
    }

    #[test]
    fn rng_bounded_respects_bound(seed in any::<u64>(), bound in 1u32..10_000) {
        let mut rng = DeterministicRng::new(seed);
        for _ in 0..100 {
            prop_assert!(rng.next_bounded(bound) < bound);
        }
    }

    #[test]
    fn relationship_stays_clamped_under_arbitrary_deltas(
        deltas in prop::collection::vec((-5.0f32..5.0, -5.0f32..5.0), 1..50)
    ) {
        let mut r = Relationship::default();
        for (a, b) in deltas {
            r.trust += a;
            r.affection += b;
            r.fear += b;
            r.suspicion += a;
            r.obligation += a;
            r.clamp_all();
            prop_assert!((-1.0..=1.0).contains(&r.trust));
            prop_assert!((-1.0..=1.0).contains(&r.affection));
            prop_assert!((-1.0..=1.0).contains(&r.fear));
            prop_assert!((-1.0..=1.0).contains(&r.suspicion));
            prop_assert!((0.0..=1.0).contains(&r.obligation));
        }
    }

    #[test]
    fn vec2_distance_is_symmetric_and_non_negative(
        x1 in -1000.0f32..1000.0,
        y1 in -1000.0f32..1000.0,
        x2 in -1000.0f32..1000.0,
        y2 in -1000.0f32..1000.0,
    ) {
        let a = Vec2::new(x1, y1);
        let b = Vec2::new(x2, y2);
        let d1 = a.distance(b);
        let d2 = b.distance(a);
        prop_assert!(d1 >= 0.0);
        prop_assert!((d1 - d2).abs() < 1e-3);
    }
}
