# Proof-boundary audit (plan I-8; test-spec §8 / T6.6). Enforced by
# `scripts/check-proof-audit.sh` in the CI `proof` job: a harness
# referencing a non-production symbol, a stale call-site entry, or a
# production call bypassing the audited kernel fails CI. Bounds for each
# harness live in `limits.toml` (same CI gate).

## Production-kernel rule

Every harness below calls the exact production `pub fn` the simulation
dispatch pathway calls. No harness reimplements a kernel. The single
extraction done for harnessability (`rpg_sim::split_money_transfer`) was
*moved into* the sim dispatch: both production transfer lowering sites call
it (see T6.1a call sites) - it is not duplicated.

## Harness -> production symbol -> production call sites

| Harness (limits.toml key) | Spec | Production symbol(s) | Production call sites (file#fn) |
|---|---|---|---|
| `i8_t61_money_transfer_conservation` | T6.1 | `rpg_sim::split_money_transfer` (`crates/sim/src/lib.rs#split_money_transfer`) | `crates/sim/src/lib.rs#transfer_money` (deal settlement + sanction fines), `crates/sim/src/lib.rs#do_pay` (player/NPC payments) |
| `i8_t61_item_transfer_conservation` | T6.1 | `rpg_domain::world::Inventory::remove` + `Inventory::add` (`crates/domain/src/world.rs`) | `crates/sim/src/lib.rs#move_item` (the `do_transfer`/settlement item path; calls `remove` then `add` with the *actual* removed count) |
| `i8_t62_claim_acceptance_boundary` | T6.2 | `rpg_behavior::epistemics::accept_assertion` (`crates/behavior/src/epistemics.rs`) | `crates/sim/src/lib.rs#do_share_claim` (the only assertion delivery path). **[BLOCKED AT BOUNDS]** the harness exists and runs against the production symbol, but CBMC 0.68 symex of the acceptance heap path (BTreeMap/Vec/String machinery - even fully concrete) exceeds the local/runner budget; see "Known limitations". Behavior coverage at runtime scale: T1.x unit/integration tests + eval property tests + the numeric cores (T6.8). |
| `i8_t63_deal_lifecycle_invariants` | T6.3 | `rpg_domain::deal::transition` + `Deal::advance` (`crates/domain/src/deal.rs`) | `crates/domain/src/deal.rs#Deal::advance` is the guarded settlement entry; production callers: `crates/sim/src/lib.rs` deal settlement/dispatch (proposals, deadlines, renegotiation, excuse) route every lifecycle event through `advance` |
| `i8_t64_replay_determinism` | T6.4 | `rpg_sim::Simulation::new` + `Simulation::replay_ops` (`crates/sim/src/lib.rs`) | `crates/sim/src/lib.rs#replay_ops` (authoritative continuation path; `replay_ops` re-runs recorded `Apply`/`Step` ops through the same `apply_impl`/`step_impl` the recorded session used). **[BLOCKED AT BOUNDS]** same CBMC budget cause as T6.2; replay equality at scale is runtime evidence via T6.5b. |
| `i8_t68_acceptance_numeric_p1_p2_p5` | T6.8 | `rpg_behavior::epistemics::{accept_assertion, aggregate_about, prior_strength}` | same as T6.2 (`do_share_claim` acceptance); `aggregate_about`/`prior_strength` are additionally called by `crates/behavior/src/mechanics.rs` and `crates/sim/src/lib.rs` (faction member reports, planner evidence value) |
| `i8_t68_acceptance_p1_nonempty_ledger` | T6.8 | `rpg_behavior::epistemics::{acceptance_probability, aggregate_from_groups, prior_strength}` | the numeric core `accept_assertion` calls; the "non-empty ledger" is one symbolic correlation group through the production fold (the P1 prior term active), map-light by construction |
| `i8_t68_corroboration_p3_p4` | T6.8 | `rpg_behavior::epistemics::{group_weight, aggregate_from_groups}` (the extracted pure numeric fold; `aggregate_about` calls both) | `crates/behavior/src/epistemics.rs#aggregate_about` (the production fold route: bucket by origin identity, then fold via `aggregate_from_groups`) |

### Sanctioned extraction (behavior-preserving, I-8)

Two extractions were moved *into* production (single source of truth, never
duplicated), both behavior-preserving and exercised by their production
callers:

1. `rpg_sim::split_money_transfer` - the money-transfer arithmetic behind
   `transfer_money`/`do_pay` (see T6.1a row).
2. `rpg_behavior::epistemics::{group_weight, aggregate_from_groups,
   acceptance_probability}` - the numeric fold that was inline in
   `aggregate_about` and the probability/reliability core that was inline
   in `accept_assertion`. Both production functions now call the extracted
   pure fns. Accumulation order is preserved (`into_values` ascending
   group id -> Vec preserves order; the probability expression is verbatim),
   so the f32 results are bit-identical - confirmed by the byte-identical
   scenario-runner output (seed 2026) and the full workspace test suite.
   The T6.8 numeric properties are proven on these cores; the full ledger
   path (records + grouping + fold) runs through T6.2 and the eval
   property tests.

## Bypass boundary (what the audit script enforces)

- The only production conserved money movement is via
  `split_money_transfer` (call sites: `transfer_money`, `do_pay`). A direct
  debit (`inventory.money -=`) anywhere in `crates/*/src` is a bypass and
  fails CI. Documented non-transfer credit exception (a source, not a
  transfer - no debit side, conservation not applicable):
  `crates/sim/src/lib.rs#do_repair` wage credit
  (`a.inventory.money += wage`, authored repair income).
- Epistemic ledger writes stay inside `crates/behavior/src/epistemics.rs`
  (`record_evidence`); no production module outside it may push into
  `Epistemics.evidence` directly (grep-checked).
- The aggregation numeric fold is `aggregate_from_groups`/`group_weight`
  (called by `aggregate_about`); a production caller reimplementing the
  fold inline would be a bypass (the audit checks the fold symbols exist
  and `aggregate_about` routes through them).
- Every `crates/{sim,behavior,domain,social}/src` production call of the
  audited kernels must go through the symbols above; the script greps for
  raw arithmetic bypasses around each audited site.

## Bounded proof vs runtime evidence (labeling rule, T-L never conflates with T6)

- **[PROOF] T6.1a/T6.1b/T6.2/T6.3/T6.4/T6.8 (this crate)**: Kani bounded
  model checking over the declared bounds in `limits.toml`. What is proven
  is *the kernel over the bounded input space*, with the explicit
  assumptions stated in each harness (concrete scaffold world for the
  epistemics harnesses; scalar inputs within documented ranges, NaN
  excluded; inventory stacks bounded by `limits.toml`). T6.4 proves replay
  equality for recorded `Apply` sequences over the authored town; it does
  not bound the step/autonomy state space.
- **[EVIDENCE] T6.5/T6.5b (`crates/eval/src/properties/`)**: proptest
  property tests - arbitrary bounded worlds/op sequences through the real
  dispatch. Strong empirical coverage including the step/autonomy replay
  path, but *not* formal proof; reports must keep the labels separate.
- **Mutation-interface assumptions remaining**: world truth is mutated only
  through the sim's typed dispatch (`Session::apply`/`step`) - asserted at
  runtime by the M1 audit (I-6/I-8 evidence layer) and by T6.2 for the
  acceptance kernel specifically; there is no static guarantee that a
  future `&mut World` caller bypasses dispatch. That residual is a
  documented runtime-evidence assumption, not a proven boundary.

## Known limitations of the current checkpoint (honest scope)

- **T6.2 (accept-acceptance boundary) and T6.4 (replay determinism) are
  BLOCKED at their declared bounds**: the harnesses exist and call the
  production symbols, but CBMC 0.68 symex makes a single `BTreeMap`
  insertion cost tens of seconds and symbolic or heap-heavy paths through
  the acceptance/dispatch machinery (nested maps, event structs, fmt)
  exceed both the local budget (WSL2, 31 GiB; >15 GiB and growing at
  ~30 min) and hosted-runner memory (7 GiB). Verified (smaller) pieces:
  T6.1/T6.3/T6.8 all pass (see logs); the *numeric* behavior of the
  acceptance formula is proven symbolically on the extracted core (T6.8
  P1/P2/P5); the *structural* accept/replay behavior is runtime evidence
  (T1.x tests, eval property tests incl. T6.5b full-log equality,
  scenario replay checksums). The attempt logs are the blocker evidence;
  the harnesses stay in the source (excluded from the CI matrix until the
  bound is tractable).
- T6.8 P3/P4 prove the numeric fold core (`group_weight`,
  `aggregate_from_groups`) with symbolic k/reliability; the map-side
  grouping in `aggregate_about` over real ledgers is covered by T6.2
  runtime evidence (tests/properties).
- Kani runs on Linux only (official support matrix); local proof execution
  uses WSL2 Ubuntu (`cargo kani -p rpg_proof_kernels`); CI runs the
  verified harnesses via `model-checking/kani-github-action@v1` (kani
  0.68.0, ubuntu-24.04, one harness per matrix leg).
