//! Production-bound proof kernels (plan I-8; test-spec T6.1-T6.8).
//!
//! Every harness in [`harnesses`] calls the *exact production symbol* the
//! simulation dispatch pathway calls - never a reimplemented toy model. The
//! one-to-one mapping harness -> production symbol(s) -> production call
//! sites is enforced by `crates/proof-kernels/AUDIT.md` +
//! `scripts/check-proof-audit.sh` (CI gate), and every harness's unwind /
//! input bounds are documented in `crates/proof-kernels/limits.toml`
//! (missing bounds entry or orphaned entry fails CI).
//!
//! Harness inventory (I-8 checkpoint):
//!
//! | Harness (limits.toml key) | Spec | Production symbol(s) |
//! |---|---|---|
//! | `i8_t61_money_transfer_conservation` | T6.1 | `rpg_sim::split_money_transfer` |
//! | `i8_t61_item_transfer_conservation` | T6.1 | `rpg_domain::world::Inventory::{remove, add}` (the `rpg_sim` `move_item` path) |
//! | `i8_t62_claim_acceptance_boundary` | T6.2 | `rpg_behavior::epistemics::accept_assertion` |
//! | `i8_t63_deal_lifecycle_invariants` | T6.3 | `rpg_domain::deal::{transition, Deal::advance}` |
//! | `i8_t64_replay_determinism` | T6.4 | `rpg_sim::{Simulation::new, Simulation::replay_ops}` |
//! | `i8_t68_acceptance_numeric_p1_p2_p5` | T6.8 | `accept_assertion`, `aggregate_about`, `prior_strength` |
//! | `i8_t68_corroboration_p3_p4` | T6.8 | `record_evidence`, `aggregate_about` |
//!
//! Harnesses live behind `#[cfg(kani)]`: normal builds (stable toolchain,
//! clippy, cargo test, WASM) are unaffected. Run them with
//! `cargo kani -p rpg_proof_kernels` on Linux (or via the CI `proof` job);
//! Kani is not installed on the Windows dev host (unsupported platform).

#[cfg(kani)]
mod harnesses;
