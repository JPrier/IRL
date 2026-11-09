//! T6.5b — full event-log replay property (proptest). Generated bounded
//! action-proposal sequences through the real dispatch; `replay_ops` must
//! reproduce the **full event log** (length + checksum + every event), the
//! **full world JSON** and the **full epistemics JSON** — never summaries.
//! Includes `Step` ops, so the step/autonomy replay path is covered at
//! scale here (the Kani T6.4 harness bounds itself to recorded `Apply`
//! ops; see `crates/proof-kernels/AUDIT.md`).

use super::{arb_ops, ArbOp};
use proptest::prelude::*;
use rpg_content::build_town;
use rpg_domain::action::ActionProposal;
use rpg_domain::ids::{ActorId, ItemId};
use rpg_sim::{SimConfig, SimOp, Simulation};

/// Prep the generated world: payee in talk range and the player holding
/// transferable stock (the generators only emit *validated-domain* ops, so
/// every op lands in the event log - rejected proposals emit no events and
/// would make the property vacuous).
fn prepped_world() -> rpg_domain::world::World {
    let mut world = build_town();
    let player = ActorId::new(rpg_domain::ids::PLAYER);
    let grocer = ActorId::new("grocer");
    if let Some(a) = world.actor_mut(&grocer) {
        a.pos = rpg_domain::geom::Vec2::new(21.0, 12.0);
        a.location = rpg_domain::ids::LocationId::new("plaza");
        // Station the payee (schedule-driven walking would drift them out of
        // talk range across generated Steps).
        a.schedule = Vec::new();
    }
    if let Some(a) = world.actor_mut(&player) {
        a.inventory.add(ItemId::new("herb"), 99);
    }
    world
}

fn ops_from(cases: Vec<ArbOp>) -> Vec<SimOp> {
    let player = ActorId::new(rpg_domain::ids::PLAYER);
    let grocer = ActorId::new("grocer");
    let herb = ItemId::new("herb");
    cases
        .into_iter()
        .map(|c| match c {
            ArbOp::Pay(amount) => SimOp::Apply(Box::new(ActionProposal {
                actor: player.clone(),
                action: rpg_domain::action::Action::Pay {
                    target: grocer.clone(),
                    amount,
                },
                source: rpg_domain::action::ProposalSource::Scripted,
                rationale: String::new(),
            })),
            ArbOp::Transfer(qty) => SimOp::Apply(Box::new(ActionProposal {
                actor: player.clone(),
                action: rpg_domain::action::Action::TransferItem {
                    target: grocer.clone(),
                    item: herb.clone(),
                    qty,
                },
                source: rpg_domain::action::ProposalSource::Scripted,
                rationale: String::new(),
            })),
            ArbOp::Step => SimOp::Step(Vec::new()),
        })
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// T6.5b: replay reproduces the full canonical state for arbitrary
    /// bounded op sequences (payments, item transfers, world steps).
    #[test]
    fn replay_reproduces_full_canonical_state(cases in proptest::collection::vec(arb_ops(), 0..8)) {
        let initial = prepped_world();
        let ops = ops_from(cases);

        let mut recorded = Simulation::new(initial.clone(), 2026, SimConfig::default());
        for op in &ops {
            match op {
                SimOp::Apply(p) => {
                    let _ = recorded.apply(p);
                }
                SimOp::Step(props) => {
                    let _ = recorded.step(props.clone());
                }
            }
        }
        let replayed = Simulation::replay_ops(initial, 2026, SimConfig::default(), &ops);

        // Full event log: length + production checksum + every event (full
        // serialized log, not a summary).
        prop_assert_eq!(recorded.events.len(), replayed.events.len());
        prop_assert_eq!(recorded.events.checksum(), replayed.events.checksum());
        prop_assert_eq!(
            serde_json::to_string(&recorded.events).unwrap(),
            serde_json::to_string(&replayed.events).unwrap()
        );
        // Full world JSON and full epistemics JSON - never summaries.
        prop_assert_eq!(
            serde_json::to_string(&recorded.world).unwrap(),
            serde_json::to_string(&replayed.world).unwrap()
        );
        prop_assert_eq!(
            serde_json::to_string(&recorded.world.epistemics).unwrap(),
            serde_json::to_string(&replayed.world.epistemics).unwrap()
        );
    }

    /// Conservation modulo recorded sources/sinks (§5.4): with background
    /// income disabled, the only money/item movements are the generated
    /// transfers - totals are exactly conserved across arbitrary sequences.
    #[test]
    fn conservation_modulo_recorded_sources_sinks(cases in super::arb_pay_ops(0..8)) {
        let initial = prepped_world();
        let ops = ops_from(cases);
        let money_before: i64 = initial.actors.values().map(|a| a.inventory.money).sum();
        let herb = ItemId::new("herb");
        let herbs_before: u64 = initial
            .actors
            .values()
            .map(|a| a.inventory.quantity(&herb) as u64)
            .sum();

        let config = SimConfig {
            enable_background: false,
            enable_director: false,
            enable_goal_autonomy: false,
            ..SimConfig::default()
        };
        let mut sim = Simulation::new(initial, 2026, config);
        for op in &ops {
            match op {
                SimOp::Apply(p) => {
                    let outcome = sim.apply(p);
                    prop_assert!(!matches!(
                        outcome,
                        rpg_domain::action::ValidationOutcome::Rejected(..)
                    ));
                }
                SimOp::Step(props) => {
                    let _ = sim.step(props.clone());
                }
            }
        }
        let money_after: i64 = sim.world.actors.values().map(|a| a.inventory.money).sum();
        let herbs_after: u64 = sim
            .world
            .actors
            .values()
            .map(|a| a.inventory.quantity(&herb) as u64)
            .sum();
        prop_assert_eq!(money_after, money_before);
        prop_assert_eq!(herbs_after, herbs_before);
    }
}
