//! T6.5 — property-based safety tests (proptest): arbitrary bounded
//! actions/inventories through the real dispatch satisfy the design §9.1
//! invariants this checkpoint covers. Which §9.1 clause each property
//! checks is documented per-test and in `crates/proof-kernels/README.md`
//! (T6.6 checklist). Runtime evidence — never reported as formal proof.

use super::{arb_ops, ArbOp};
use proptest::prelude::*;
use rpg_content::build_town;
use rpg_domain::action::ActionProposal;
use rpg_domain::ids::{ActorId, ItemId};
use rpg_sim::{SimConfig, SimOp, Simulation};

/// Build the prepped world (payee in range; player holds transferable
/// stock) and run the generated ops with background mutation disabled, so
/// the only resource movements are the generated transfers.
fn run_generated(cases: Vec<ArbOp>) -> (Simulation, Vec<SimOp>, rpg_domain::world::World) {
    let mut world = build_town();
    let player = ActorId::new(rpg_domain::ids::PLAYER);
    let grocer = ActorId::new("grocer");
    if let Some(a) = world.actor_mut(&grocer) {
        a.pos = rpg_domain::geom::Vec2::new(21.0, 12.0);
        a.location = rpg_domain::ids::LocationId::new("plaza");
        // Station the payee: schedule-driven walking would drift them out of
        // talk range across generated Steps (the generators only emit
        // validated-domain ops).
        a.schedule = Vec::new();
    }
    if let Some(a) = world.actor_mut(&player) {
        a.inventory.add(ItemId::new("herb"), 99);
    }
    let herb = ItemId::new("herb");
    let ops = cases
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
        .collect::<Vec<_>>();
    let config = SimConfig {
        enable_background: false,
        enable_director: false,
        enable_goal_autonomy: false,
        ..SimConfig::default()
    };
    let sim = Simulation::new(world.clone(), 2026, config);
    (sim, ops, world)
}

fn world_totals(world: &rpg_domain::world::World, herb: &ItemId) -> (i64, u64) {
    (
        world.actors.values().map(|a| a.inventory.money).sum(),
        world
            .actors
            .values()
            .map(|a| a.inventory.quantity(herb) as u64)
            .sum(),
    )
}

fn execute(sim: &mut Simulation, ops: &[SimOp]) -> bool {
    let mut all_applied = true;
    for op in ops {
        match op {
            SimOp::Apply(p) => {
                if matches!(
                    sim.apply(p),
                    rpg_domain::action::ValidationOutcome::Rejected(..)
                ) {
                    all_applied = false;
                }
            }
            SimOp::Step(props) => {
                let _ = sim.step(props.clone());
            }
        }
    }
    all_applied
}

proptest! {
    #![proptest_config(proptest::test_runner::Config {
        cases: 64,
        ..proptest::test_runner::Config::default()
    })]

    /// §9.1.1 conservation: money and item totals are exactly conserved
    /// across arbitrary transfer/step sequences (no background sources
    /// generated; §5.4 conservation modulo recorded sources/sinks).
    #[test]
    fn transfers_conserve_resources(cases in proptest::collection::vec(arb_ops(), 0..8)) {
        let herb = ItemId::new("herb");
        let (mut sim, ops, initial) = run_generated(cases);
        let (money_before, herbs_before) = world_totals(&initial, &herb);
        prop_assert!(
            execute(&mut sim, &ops),
            "generated op rejected: generator left the validated domain"
        );
        let (money_after, herbs_after) = world_totals(&sim.world, &herb);
        prop_assert_eq!(money_after, money_before);
        prop_assert_eq!(herbs_after, herbs_before);
    }

    /// §9.1.3 claim separation (transfer/step side): authoritative world
    /// truth (the facts ledger) is never written by the transfer/step
    /// dispatch - only speech-act acceptance touches epistemics, and the
    /// authoritative facts map is untouched by any generated op.
    #[test]
    fn transfer_step_never_writes_truth(cases in proptest::collection::vec(arb_ops(), 0..8)) {
        let (mut sim, ops, initial) = run_generated(cases);
        let facts_before = serde_json::to_string(&initial.facts).unwrap();
        prop_assert!(execute(&mut sim, &ops));
        let facts_after = serde_json::to_string(&sim.world.facts).unwrap();
        prop_assert_eq!(facts_after, facts_before);
    }

    /// §9.1.11 temporal consistency: event ticks are non-decreasing across
    /// the full recorded log for arbitrary op sequences.
    #[test]
    fn event_ticks_monotonic(cases in proptest::collection::vec(arb_ops(), 0..8)) {
        let (mut sim, ops, _initial) = run_generated(cases);
        prop_assert!(execute(&mut sim, &ops));
        let mut last_tick = 0u64;
        for event in sim.events.all().iter() {
            prop_assert!(event.tick >= last_tick);
            last_tick = event.tick;
        }
    }

    /// §9.1.10 no dangling references (perception side): every event's
    /// observers exist in the world (no phantom observer ids).
    #[test]
    fn no_dangling_observers(cases in proptest::collection::vec(arb_ops(), 0..8)) {
        let (mut sim, ops, _initial) = run_generated(cases);
        prop_assert!(execute(&mut sim, &ops));
        for event in sim.events.all().iter() {
            for observer in &event.observed_by {
                prop_assert!(sim.world.actor(observer).is_some(), "phantom observer {observer:?}");
            }
        }
    }
}
