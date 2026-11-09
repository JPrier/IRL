# rpg_proof_kernels — production-bound Kani proof harnesses (I-8)

Kani harnesses over **production symbols only** (plan §5 Phase 6 corrected
rule). Run on Linux: `cargo kani -p rpg_proof_kernels` (or the CI `proof`
job). Harness↔limits mapping: `limits.toml` (CI-enforced). Harness→symbol→
call-site mapping: `AUDIT.md` (CI-enforced by `scripts/check-proof-audit.sh`).

## T6.6 — inductive coverage audit (checklist; CI-checked)

Every primitive lowering used by the sim's mutation dispatch has either a
preservation-lemma reference (a Kani harness proving it) or reuses a
verified effect it is written in terms of. Rows marked **deferred** are
honestly out of this checkpoint's bounded scope and are covered by runtime
evidence layers instead (unit/integration tests, M1 audit, property tests) —
never claimed as proof.

| Primitive lowering (dispatch → kernel) | Preservation reference | Status |
|---|---|---|
| money transfer (`do_pay`, settlement, fines) | `split_money_transfer` → harness `i8_t61_money_transfer_conservation` | **[PROOF] T6.1 (verified)** |
| item transfer (`do_transfer`, settlement delivery) | `Inventory::{remove, add}` → harness `i8_t61_item_transfer_conservation` | **[PROOF] T6.1 (verified)** |
| acceptance probability + perceived reliability (numeric core) | `acceptance_probability` → harnesses `i8_t68_acceptance_numeric_p1_p2_p5`, `i8_t68_acceptance_p1_nonempty_ledger` | **[PROOF] T6.8 P1/P2/P5 (verified)** |
| corroboration weight + numeric fold (numeric core) | `group_weight`/`aggregate_from_groups` → harness `i8_t68_corroboration_p3_p4` | **[PROOF] T6.8 P3/P4 (verified)** |
| assertion delivery (claim → ledger, full accept path) | `accept_assertion` → harness `i8_t62_claim_acceptance_boundary` | **[BLOCKED AT BOUNDS]**: harness exists and targets the production symbol; CBMC 0.68 symex exceeds budget (see AUDIT "Known limitations"). Runtime evidence: T1.x tests, property tests, scenarios. |
| evidence recording (all ledger writes) | `record_evidence` (inside `accept_assertion`, exercised by T6.2 attempt) | runtime evidence (T1.x/property tests) |
| aggregation map-side grouping (ledger → groups) | `aggregate_about` bucketing (calls the proven fold core) | runtime evidence (T1.x/property tests) |
| deal lifecycle (proposals → settlement) | `deal::transition`/`Deal::advance` → harness `i8_t63_deal_lifecycle_invariants` | **[PROOF] T6.3 (verified)** |
| authoritative replay (recorded ops → canonical state) | `Simulation::replay_ops` → harness `i8_t64_replay_determinism` | **[BLOCKED AT BOUNDS]** (same cause); runtime evidence: T6.5b property (full event log + world + epistemics JSON equality, incl. Steps), scenario `deterministic_replay` checksum, `save_reload_continuity` |
| belief application (rolled outcome → `Actor.beliefs`) | sim layer downstream of `accept_assertion` (writes only from the returned roll); covered by `crates/sim/tests` acceptance tests + T6.5 safety property | **deferred (runtime evidence)** |
| movement/pathfinding (`step` movement lowering) | grid BFS; conservation-free (no authoritative resource movement); covered by scenario determinism checks (seed 2026) | **deferred (runtime evidence)** |
| needs/emotions decay (background typed mutation, G4) | classified non-evented decay; M1 audit + T3.x tests | **deferred (runtime evidence)** |
| faction member-report aggregation | aggregate over reported evidence via `aggregate_about` (fold core proved above); report *collection* path covered by I6b R1/R5 tests | **[PROOF] fold core; report path runtime evidence** |
| appraisal/relationship updates (mind-local) | no authoritative resource movement; covered by T3.x tests | **deferred (runtime evidence)** |
| offer/commitment bookkeeping (`offers`, `commitments`) | covered by T2.x acceptance tests | **deferred (runtime evidence)** |
| perception channel routing (observed_by) | covered by R7/T4.x noninterference tests + T6.5 perception property | **deferred (runtime evidence)** |
| sanction identity ledger (exactly-once punishment) | covered by T4.10 identity tests (I6a F-series) | **deferred (runtime evidence)** |

## Layers (design §9.3)

1. unit/integration tests (`cargo test --workspace`) — behavior contracts;
2. **Kani bounded model checking (this crate + CI `proof` job)** — the
   [PROOF] layer, bounds in `limits.toml`;
3. property tests (`crates/eval/src/properties/`) — runtime evidence at
   generated scale (T6.5/T6.5b), labels kept separate from [PROOF];
4. Verus/Lean/TLA+ — not used (plan: optional/deferred).

## Running

- Local (WSL2 Ubuntu): `cargo kani -p rpg_proof_kernels --output-format terse`
- CI: the `proof` job in `.github/workflows/ci.yml`
  (`model-checking/kani-github-action@v1`, kani 0.68.0, ubuntu-24.04) runs
  every harness at its `limits.toml` bounds plus the audit/limits checks.
- Negative gates: `scripts/check-proof-audit.sh` (limits 1:1, audit
  call-sites, bypass greps); the deliberate-violation Kani negative test
  (T6.7) is documented in AUDIT.md and run once in an isolated worktree.
