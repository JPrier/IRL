//! Behavior-engine Phase 0 baseline (plan §3, Phase 0): time per step,
//! NPC decisions per simulated minute, and replay stability, measured on the
//! real town through the real session commit path. The printed numbers are
//! the baseline artifact; the replay-stability assertion is the gate.
//!
//! Everything that changes canonical state must flow through recorded ops
//! (`apply`/`step`) for the replay to reconstruct it — the fixture walks
//! instead of teleporting.

use rpg_eval::Session;
use rpg_sim::Simulation;
use std::time::Instant;

#[test]
fn phase0_baseline_step_time_and_replay_stability() {
    const TICKS: u32 = 300;

    let start = Instant::now();
    let mut s = Session::new(2026);
    s.goto("grocer");
    s.say(
        "grocer",
        "I need to see your ledger. The one with the dock payments.",
    );
    for i in 0..TICKS {
        // One scripted player conversation every ~60 simulated minutes so
        // the decision pathway is exercised at a realistic cadence.
        if i % 60 == 0 {
            s.say("grocer", "Any news about the shipments?");
        }
        let _ = s.tick();
    }
    let elapsed = start.elapsed();

    let per_tick_us = elapsed.as_micros() as f64 / f64::from(TICKS);
    let simulated_minutes = f64::from(TICKS);
    println!(
        "baseline: {:.0} us/step, {:.4} NPC decisions/simulated-minute \
         ({} decisions over {} ticks, {} events)",
        per_tick_us,
        s.decisions.len() as f64 / simulated_minutes,
        s.decisions.len(),
        TICKS,
        s.sim.events.len()
    );

    // Replay stability: the recorded proposal log must reconstruct the exact
    // canonical state, including the evidence ledger.
    let replayed = Simulation::replay_ops(
        s.sim.initial_world.clone(),
        s.sim.seed,
        s.sim.config.clone(),
        &s.sim.ops,
    );
    assert_eq!(
        s.checksum(),
        replayed.events.checksum(),
        "replay must reproduce the canonical event stream"
    );
    assert_eq!(
        serde_json::to_string(&s.sim.world.epistemics).unwrap(),
        serde_json::to_string(&replayed.world.epistemics).unwrap(),
        "replay must reproduce the evidence ledger exactly"
    );
    assert_eq!(
        serde_json::to_string(&s.sim.world).unwrap(),
        serde_json::to_string(&replayed.world).unwrap(),
        "replay must reconstruct canonical world state exactly"
    );
}
