//! Property-based checks (T6.5/T6.5b, plan I-8). Runtime evidence over
//! generated bounded inputs through the real dispatch — explicitly labeled
//! evidence, never formal proof (the [PROOF] layer is
//! `crates/proof-kernels` + the CI Kani job).

pub mod replay;
pub mod safety;

use proptest::prelude::*;

/// One generated operation for the replay/conservation generators:
/// a validated-domain payment, a validated-domain item transfer, or a world
/// step (empty external proposals; NPC autonomy still runs inside `step`).
#[derive(Clone, Debug, PartialEq)]
pub enum ArbOp {
    Pay(i64),
    Transfer(u32),
    Step,
}

/// Typed generator over the action families exercised by this checkpoint:
/// payments (1..=5 coins), herb transfers (1..=3 units), world steps. All
/// in the validated domain of the prepped town (player money 250, 6 herbs,
/// payee in talk range), so every op lands in the event log.
pub fn arb_ops() -> impl Strategy<Value = ArbOp> {
    prop_oneof![
        (1i64..=5).prop_map(ArbOp::Pay),
        (1u32..=3).prop_map(ArbOp::Transfer),
        (0u8..1).prop_map(|_| ArbOp::Step),
    ]
}

/// Payments only (used for the money-conservation property).
pub fn arb_pay_ops(size: core::ops::Range<usize>) -> impl Strategy<Value = Vec<ArbOp>> {
    proptest::collection::vec(
        prop_oneof![
            (1i64..=5).prop_map(ArbOp::Pay),
            (0u8..1).prop_map(|_| ArbOp::Step)
        ],
        size,
    )
}
