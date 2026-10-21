//! Deterministic, seedable pseudo random number generator.
//!
//! We deliberately do **not** depend on a third-party RNG for simulation, so
//! that "same seed + same recorded model proposals ⇒ identical canonical
//! events" is a property of our own, auditable code.

use serde::{Deserialize, Serialize};

/// SplitMix64 — tiny, fast, well-distributed, trivially serializable.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeterministicRng {
    state: u64,
    /// How many values have been drawn; useful for replay diagnostics.
    draws: u64,
}

impl DeterministicRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
            draws: 0,
        }
    }

    pub fn seed(&self) -> u64 {
        self.state
    }

    pub fn draws(&self) -> u64 {
        self.draws
    }

    pub fn next_u64(&mut self) -> u64 {
        self.draws += 1;
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        // Use the top 24 bits for a clean f32 mantissa.
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }

    /// Uniform integer in `0..bound` (bound > 0).
    pub fn next_bounded(&mut self, bound: u32) -> u32 {
        assert!(bound > 0, "bound must be positive");
        // Rejection sampling keeps the distribution unbiased.
        let zone = u32::MAX - (u32::MAX % bound);
        loop {
            let v = (self.next_u64() >> 32) as u32;
            if v < zone {
                return v % bound;
            }
        }
    }

    pub fn chance(&mut self, p: f32) -> bool {
        self.next_f32() < p
    }

    /// Pick an index for a slice; returns 0 for empty slices.
    pub fn pick_index(&mut self, len: usize) -> usize {
        if len == 0 {
            0
        } else {
            self.next_bounded(len as u32) as usize
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_sequence() {
        let mut a = DeterministicRng::new(42);
        let mut b = DeterministicRng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seed_differs() {
        let mut a = DeterministicRng::new(1);
        let mut b = DeterministicRng::new(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn f32_in_unit_range() {
        let mut r = DeterministicRng::new(7);
        for _ in 0..10_000 {
            let v = r.next_f32();
            assert!((0.0..1.0).contains(&v), "out of range: {v}");
        }
    }

    #[test]
    fn bounded_is_in_range() {
        let mut r = DeterministicRng::new(9);
        for _ in 0..10_000 {
            assert!(r.next_bounded(7) < 7);
        }
    }
}
