# Compositional Behavior Engine — Acceptance Test Specification

Base: `origin/main` @ `a4ee23b`. Companion to `compositional_behavior_engine_implementation_plan.md` (revision 2 — its §4 pins the contract settlement semantics T2.5 tests).
All tests run with `cargo test --workspace` and `cargo run -q -p scenario-runner -- --seed 2026` unless noted. Formal harnesses run in CI (Kani, Linux) — not available locally (blocker reported).

Evidence-vs-proof convention: tests marked **[EVIDENCE]** are bounded monitored checks under stated assumptions (they can demonstrate violations; passing them does not prove the property unboundedly). Only the Kani harnesses (T6.1–T6.4) are **[PROOF]** artifacts over their documented bounds (`crates/proof-kernels/limits.toml`).

## 1. Regression baseline (must hold at every phase boundary)

- T0.1 All 10 existing scenarios PASS at seed 2026 (list in plan §1).
- T0.2 `cargo fmt --all --check` exit 0.
- T0.3 `cargo clippy --workspace --all-targets -- -D warnings` exit 0 (CI parity; locally verified on 8 core crates at baseline).
- T0.4 WASM, directly built: `cargo build -p rpg_behavior --target wasm32-unknown-unknown` succeeds (the behavior crate is pure std) **and** `cargo build -p web-game --target wasm32-unknown-unknown` succeeds (consumes behavior transitively via rpg_eval→rpg_sim). Both at every phase boundary.
- T0.5 Deterministic replay: `save_reload_continuity` + `deterministic_replay` scenarios pass; identical save + seed + inputs ⇒ identical event checksum.
- T0.6 Benchmark acceptance (recorded): `cargo test -p rpg_eval --test behavior_baseline -- --nocapture` passes its replay-stability assertions (event checksum + world JSON + epistemics JSON) — hard gate; prints us/step, recorded baseline **93 us/step** (300 ticks, seed 2026); soft regression budget ≤ 2× baseline (≤ 186 us/step) until recalibrated (plan §8.6).

## 2. Phase 1 — Epistemic microcore (**DELIVERED**; regression-locked)

- T1.1 Evidence ledger storage: a proposition receives N evidence entries from distinct source chains; `EvidenceStore`/`Epistemics` returns them all (no overwrite like the old `record_belief`). ✅ `crates/behavior/tests/epistemic_ledger.rs::t1_1`
- T1.2 Correlation protection: 5 retellings from one origin chain do not raise certainty like 5 independent sources; `independent_corrob_group` grouping verified; repeated same-chain assertion never yields certainty. ✅ `t1_2`, `t1_2b`
- T1.3 Contradiction tolerance: NPC holds contradictory reports; queries expose both, not a forced boolean. ✅ `t1_3`
- T1.4 No automatic acceptance: an assertion with low perceived reliability does not flip belief deterministically; the `assertive` flag weights odds but never forces belief (regression vs `share_claim` semantics). ✅ `t1_4`
- T1.5 No omniscience: NPC answers a QUERY only from accessible evidence; **[EVIDENCE]** information-noninterference smoke: unreachable secret changes no observable action with fixed RNG. ✅ `t1_5`, `t1_5b`
- T1.6 Liar-vs-skeptic deliverable scenario: a liar fools one NPC; a skeptic NPC requests additional evidence — both emerge without actor-id special cases. ✅ `t1_6`
- T1.7 Replay: evidence insertions recorded via SimOp; replay reproduces belief states and the full evidence ledger. ✅ `crates/sim/tests/behavior_engine.rs::replay_reproduces_the_evidence_ledger_and_checksum`
- T1.8 Backward compat: old saves (SAVE_VERSION 2) load via migration; `share_claim`/`do_share_claim`/`do_warn_contact` callers compile and behave via adapter. ✅ persistence `test_v2_save_migrates_to_v3_with_empty_epistemics`
- T1.8b A lie is a proposition choice, not a truth mutation; knowing-of ≠ believing (either polarity) — `knows_of` and authored-polarity tests. ✅ `t1_6b`, `knows_of_distinguishes_heard_disbelieved_from_unknown`, `empty_ledger_reproduces_the_legacy_probability_formula`
- T1.9 **[NEW, G1]** Canonical deceptive assertion end-to-end (I-5): a proposal carrying `polarity ≠ speaker's own belief` (deceptive) passes validation only when the speaker `knows_of` the proposition; the authoritative lowering records the asserted polarity (not the speaker's belief) in evidence + events; the M2 provenance audit marks it as an intentional lie; world truth is untouched.

## 3. Phase 2 — Goals, intentions, commitments

- T2.1 Intention persistence: adopted plan survives world ticks unless invalidated; fast path continues without re-deliberation.
- T2.1b Commitment survival vs dialogue (guardrail §12.7): agreements survive rendering; speech never settles a contract (T2.4 also covers the negative side).
- T2.2 Deal FSM (plan §4.1): transitions Proposed→Accepted→Active→ConditionMet→PerformanceDue→{Performed, Violated, Excused} and Rejected/Canceled/Renegotiated; each terminal transition exactly-once; illegal transitions rejected.
- T2.3 Condition-met ≠ benefit-delivered: delivering medicine activates the disclosure obligation (ConditionMet) separately from the promised disclosure (Performed) — distinct events, distinct states.
- T2.4 False fulfillment claim: a claim of completion without world predicate/evidence does NOT move the FSM; the claim itself is only recorded evidence.
- **T2.5 Partial delivery / interruption — semantics are PINNED in plan §4 (this spec tests that text, not an implementation-defined reading):**
  - T2.5a Escrow/atomicity (§4.2): a settlement bundle applied in one `validate_then_apply` either transfers every component or nothing (rejection mutates nothing); give-on-acceptance consideration moves atomically at `Accepted`; no inventory lock/reservation mechanic exists in v1.
  - T2.5b Partial obligations (§4.3): an installment deal settles each installment exactly-once and reaches `Performed` only when all are settled (partial state observable); an un-split obligation short-delivered stays `PerformanceDue`, emits typed `DealPartialDelivery {got, expected}`, mutates nothing contractual, and enables creditor renegotiate/excuse/violate paths.
  - T2.5c Interruption/cancellation (§4.4): cancellation authorization matrix — mutual/creditor/debtor-before-ConditionMet-with-no-consideration-exchanged ⇒ Canceled; debtor-after-ConditionMet ⇒ Violated path; from PerformanceDue: creditor consent ⇒ Canceled, authority evidence ⇒ Excused, else deadline ⇒ Violated; event history intact afterwards.
  - T2.5d Deadlines (§4.5): crossing emits exactly one idempotent `ObligationDue` event; grace policy then Violated with typed event; deadline processing runs in the deterministic step pipeline (no out-of-band mutation).
  - T2.5e Exactly-once settlement (§4.6): replaying the recorded settlement events cannot double-settle (state-guarded transition); property-level coverage in T6.5, proof in T6.3.
- T2.6 Hardcoded-grocer removal, **scoped to the production new pathway** (two enforceable parts):
  (a) grep audit over the new-pathway code only — `crates/behavior/**` plus the new-pathway dispatch region in `crates/agents` — for literal actor-id conditionals (`id.as_str() == "…"`); the legacy `heuristic_decision` fallback and scenario content fixtures (`f_grocer_*`) are explicitly exempt until I-11;
  (b) **rename-equivalence test**: run the new pathway on a world whose actor ids are permuted (grocer↔baker, structure preserved) and assert decisions are isomorphic — the strongest check that no literal id gates a decision.
- T2.7 Grocer scenario regression: `grocer_conditional_bargain` PASS with the bargain generated from goal + resource obstacle + offer via the new pathway.
- T2.8 Utility ranking: feasibility filter removes physically impossible options before scoring; lexicographic hard constraints cannot be traded for weights; physically-impossible vs personally-unacceptable are distinguished.
- T2.9 Deadline handling: overdue obligation emits `ObligationDue`; NPC reacts (renegotiate/perform/abandon) per §4.5.
- T2.9b Seeded softmax sampling: sampling uses recorded seed + sample; invalid actions are never sampled; replay reproduces the choice.

## 4. Phase 3 — Social graph, needs, emotions

- T3.1 Directed edges: Rel(i,j) ≠ Rel(j,i) after one-sided event.
- T3.2 Contextual trust: trusting a mechanic ≠ trusting with a secret (dimension separation).
- T3.3 No universal deltas: same event e yields different ΔR for different V_i (two value profiles).
- T3.4 Saturation + decay: repeated identical events asymptote; unused dimensions decay at configured rates.
- T3.5 Forgiveness/retaliation: both paths reachable depending on appraisal state.
- T3.6 Appraisal provenance: each relationship delta references event id + interpretation.
- **T3.7 [NEW, I-4a] Needs state `N` (design §3)**: `Actor.needs` (safety, belonging, wealth, health, status) with configured decay rates; needs decay is a classified background mutation (event-sourced threshold crossings via typed `NeedThresholdCrossed`, regular decay declared non-evented in the M1 audit).
- **T3.8 [NEW, I-4a] Satiation/thresholds**: diminishing returns above satiation threshold; crossing a threshold raises the mapped goal urgency (e.g. wealth below threshold biases toward economic options); no unbounded urgency escalation (hysteresis check).
- **T3.9 [NEW, I-3b] Utility needs integration (§4.2 α_i·Δneeds)**: two actors with different need profiles rank the same option set differently; lexicographic constraints still dominate; `switch_i` penalty deters oscillation.
- **T3.10 [NEW, I-4b] Emotions state `E`**: `Actor.emotions` temporary appraisal variables with decay/thresholds/sensitivities; `Effect::Appraise` writes emotions (mind-local — never world truth).
- **T3.11 [NEW, I-4b] Threshold-gated responses**: crossing fear/anger thresholds gates response classes (flee/retaliate/comfort/reconcile) deterministically; emotions decay back below threshold within configured time; sensitivity parameters alter crossing points per actor.
- **T3.12 [NEW, I-4b] Emotions ≠ relationships**: an emotion spike without a relationship delta (and vice versa) is representable; appraisal provenance recorded for both.

## 5. Phase 4 — Autonomous world

- T4.1 Offscreen autonomy: with no player input, world advances — schedules execute, contracts resolve, faction accounts change.
- T4.2 Scarcity conservation: money/items conserved modulo recorded sources/sinks over a 1000-tick randomized run (property test).
- T4.3 Faction institution behavior: group decision derived from member reports (no truth access); a faction misbelieves when member reports are wrong.
- T4.4 Deliberation budgets: LLM requests per world step ≤ configured cap; planner nodes bounded; cap enforcement test (heavy tier fires only when expected value warrants).
- T4.5 Conflict resolution: two competing transfers on the same resource in one tick are ordered deterministically (tick, priority, sequence) — no nondeterministic winner.
- T4.6 Catch-up: unloaded area produces bounded catch-up events from schedules/ledgers; debts/deaths/clues/resources preserved exactly; incidental animation NOT exactly simulated (classification list asserted).
- T4.7 Multi-rate scheduling: tick categories fire at declared rates (smoke test on scheduler).
- T4.8 Novelty handling: unknown intent yields clarification/fallback, never a state mutation.
- **T4.9 [NEW, G2/I-5] QUERY lowering (§1.1 "no omniscient answer")**: an authoritative question path dispatches `Effect::Query`; the answer is derived from the target's ledger (`answer_query`) and emitted as typed events; a target with no evidence answers Unknown even when world truth says otherwise.
- **T4.10 [NEW, G3/I-5] SANCTION lowering (§1.1 "authority, process, cost, evidence")**: a sanction action lowers to `Effect::Sanction` only when the authority check, process (typed event), cost, and evidence prerequisites hold; unauthorized sanction is rejected (not-permitted ≠ not-possible distinction asserted).
- **T4.11 [NEW] Not-possible vs not-permitted vs not-believed vs attempt-failed**: the reject-reason vocabulary distinguishes all four; a physically infeasible act and a normatively prohibited act produce different rejection classes.
- **T4.12 [NEW, §1.2] Typed speech-act pipeline**: free-form input parses into typed candidate speech acts with confidence; parser text can never directly set a belief/resource/quest/world fact (structural test: no authoritative path reads raw utterance text into state mutations).
- **T4.13 [NEW, §4.3] Tier gating**: fast path continues routine without inference; medium path reacts deterministically; heavy path (LLM/bounded search) fires only when the expected-value gate opens; tier choice is recorded in the causal trace.

## 6. Phase 5 — Emergent quest and story recognition

- T5.1 Quest recognition: grocer's medicine goal + obstacle + player pathway recognized as quest opportunity without scripted quest giver.
- T5.2 Substitute satisfaction: an alternative satisfying the goal predicate is accepted as quest resolution (no single authored solution).
- T5.3 Director constraints: director cannot mutate committed truth, invent witnessed events, teleport resources, or promote repeatedly-investigated innocents (negative tests).
- T5.4 Anchor validation: surfaced opportunities checked against authored anchors (existing narrative validation reused).

## 7. Liveness under a bounded fair scheduler (**[EVIDENCE]** — plan I-10; design §9.2)

All T-L tests use the same harness: a **fair activation scheduler** (round-robin with aging over active NPCs; every active NPC gets ≥1 proposal slot per window W; budget caps per NPC per step) and fixed seeds. Each test states its fairness assumption, its bound (ticks/actors), and is explicitly **bounded monitored evidence — not a proof**. The proof artifacts for the underlying mechanisms are the Kani kernels (T6.x); the two categories must never be conflated in reports.

- T-L1 No permanent task deadlock (§9.2.1) **[EVIDENCE]**: under the fair scheduler over N ticks with M actors and bounded resources, every critical narrative goal either reaches a resolution state or an explicit declared-failure state (abandonment criteria met) — no goal remains both active and permanently blocked. Assumptions recorded: resource availability, actor liveness, fair scheduling.
- T-L2 Eventual resolution (§9.2.2) **[EVIDENCE]**: every accepted, due deal reaches {Performed, Violated, Excused, Renegotiated} within the bound; the test does **not** require NPCs to honor promises (design note) — it requires the *system* to resolve them. Settlement exactly-once mechanism itself is proven in T6.3.
- T-L3 Offscreen autonomy (§9.2.3) **[EVIDENCE]**: with zero player proposals, world advances under the fair scheduler (cross-ref T4.1); bounded catch-up keeps debts/deaths/clues/resources causally equivalent (T4.6).
- T-L4 Budget boundedness (§9.2.4) **[EVIDENCE]**: inference calls and planner nodes per world step ≤ caps under adversarial load (many simultaneous deadlines); caps enforced, not advisory (cross-ref T4.4/M6).
- T-L5 Narrative access (§9.2.5) **[EVIDENCE]**: while scenario assumptions hold, critical revelations retain ≥2 independent evidence routes across the seeded battery (extends scenario `three_independent_routes` to generated variants).

## 8. Phase 6 — Formal proof kernels (CI; Kani not installed locally)

**Production-kernel rule (review correction): every harness calls the actual production function the dispatch pathway calls — no reimplemented toy kernels.** `crates/proof-kernels/AUDIT.md` maps harness → production symbol → production call sites; CI script `scripts/check-proof-audit.sh` fails on: harness referencing a non-production symbol, a stale call-site entry, or a production call bypassing the audited symbol. Kani bounds live in `crates/proof-kernels/limits.toml`; the CI proof job runs each harness at its declared bounds **and** validates the artifact against the harnesses (missing bounds entry or orphaned entry ⇒ CI failure) — an ongoing gate, updated in the same PR as harness changes, not a one-time document.

- T6.1 **[PROOF]** Kani harness: transfer/accounting conservation over bounded exhaustive inputs, calling the production `rpg_sim` transfer validation/apply path (I-8 extracts a pure helper *into* sim dispatch if needed — single source of truth). `limits.toml` entry: max inventory stacks, quantities, actors.
- T6.2 **[PROOF]** Kani harness: claim-acceptance boundary — an assertion (any polarity, any assertiveness) cannot write world truth; acceptance only writes ledger + belief via the production `accept_assertion`.
- T6.3 **[PROOF]** Kani harness: deal lifecycle (production `rpg_domain::deal` transition fn) — exactly-once terminal settlement, no illegal transitions, ConditionMet ≠ Performed, over bounded deal/step spaces.
- T6.4 **[PROOF]** Kani harness: authoritative replay determinism over bounded step sequences (production replay path).
- T6.5 Property-based tests (`crates/eval/properties/safety.rs`, proptest): arbitrary bounded actors/actions/contracts/inventories satisfy design §9.1.1–12 (conservation, disclosure integrity, claim separation, perception boundaries, authoritative action, commitment integrity, no retroactive contradiction, rights scoping, temporal consistency, no dangling references, replay, noninterference).
- T6.5b **[NEW] Full event-log replay property test** (`crates/eval/properties/replay.rs`): generated bounded action sequences; `replay_ops` reproduces the **full event log** (length + checksum + every event), the full world JSON, and the full epistemics JSON — never summaries.
- T6.6 Inductive coverage audit: every primitive lowering has a preservation-lemma reference or reuses a verified effect (checklist in proof-kernels README; CI check).
- T6.7 CI gate: `make verify` includes property tests + Kani job; bounded model-check limits documented in `limits.toml`; a deliberately introduced conservation violation fails CI (negative test, run once manually then reverted).
- T6.8 **[NEW, numeric arbitration proof, §9.1-adjacent]** **[PROOF]** Kani harness over the **production** `rpg_behavior::epistemics::{aggregate_about, prior_strength}` and the acceptance-probability clamp in `accept_assertion`, properties over bounded inputs: (P1) probability ∈ [0.05, 0.97]; (P2) empty ledger ⇒ exactly the legacy formula; (P3) k retellings from one correlation group contribute weight ≤ max_reliability + 0.24 for any k (bounded corroboration); (P4) `contradicted` iff both polarities weigh ≥ 0.25; (P5) clamps hold — no NaN/overflow over bounded f32 inputs.

## 9. Micro-world generated evaluation (design §13; plan I-9)

Generator: parameterized three-plus-actor micro-world (player, grocer, guard, optional spy) varying, per seeded variant: the §13 action axes (ask for ledger / offer medicine-money-protection-misinformation / withhold or deliver a promise / assert conflicting evidence / broadcast vs private speech / observe or miss an event / threaten or help / guard duty + corruption) × actor values/trust parameters/needs profiles/positions/channel reliabilities. Deterministic: seed → identical world, identical replay. **Hundreds of variants: ≥ 300 per full run** (nightly CI runs a fixed subset; the phase gate runs all).

- TM.1 **[EVIDENCE]** M3 gate (0 omniscient disclosures): automated per-variant check — every ASSERT's proposition content is derivable from the speaker's evidence ledger + authored knowledge (`knows_of`); any assertion with zero ledger evidence and no authored belief fails the variant; aggregate over ≥300 variants must be 0 violations. (Extends T1.5 to generated scale.)
- TM.2 **[EVIDENCE]** M7 gates, numeric: (a) ≥3 distinct terminal outcome classes across the battery (e.g. fooled / skeptic-seeks-evidence / refuse-conceal / renegotiated); (b) 100% of NPC decisions carry a causal trace naming goal + belief-or-observation + cost; (c) 100% of variants replay exactly (checksum + full JSON); (d) T2.6 checks pass (no literal actor-id branches in the production new pathway) — the "no `if actor_id == "grocer"`, no hand-written utterance-keyword hacks" acceptance from §13.
- TM.3 **[EVIDENCE]** All design hard invariants hold in every variant (conservation, claim separation, perception boundaries — reuse T6.5 property checks per variant tail).
- TM.4 **[EVIDENCE]** Causal intelligibility sampling: for a fixed random subset of variants, the causal trace renders a coherent "did X because of…" line whose referenced goal/belief/observation exist in that variant's world (spot-check harness, documented sample size).
- TM.5 Determinism: same seed ⇒ byte-identical event stream and world (regression-lock for the battery itself).

## 10. Metrics verification

- M1 100% authoritative mutations typed + event-logged, **including classified background mutations** (regular needs/salience decay declared non-evented in the audit; threshold crossings event-sourced) — audit assertion test (I-6/I-8).
- M2 100% NPC claims carry provenance **including intentional lies** (unblocked by I-5/T1.9) — audit test.
- M3 0 omniscient disclosures in adversarial generated scenarios — **measured by TM.1 over ≥300 seeded variants**, not a hand-written scenario.
- M4 0 stale/invalid contract completions in bounded harnesses.
- M5 100% deterministic replay for identical recorded inputs + versions (full event-log property test T6.5b).
- M6 Planning/LLM budget caps configurable and enforced (config test, T4.4/T-L4).
- M7 Qualitative → **measurable gates per TM.2** (≥3 outcome classes; 100% trace coverage; 100% variant replay; no literal-id branches in the new pathway); qualitative review remains on the rendered traces (TM.4).

## 11. Coverage matrix (requirement → owner → test)

Owners per plan §6/§7. Every full-design requirement row has an owner; **no unowned requirements**.

| Design requirement | Owner | Tests |
|---|---|---|
| §1 interaction definition + rejection classes | I-5, I-3b | T4.9–T4.11 |
| §1.1 16 primitives (effect algebra + lowering) | I-1 ✅ / I-5 / I-6 | plan §3 table, T6.6, sim `every_accepted_action_produces_typed_effects`, T4.9–T4.10 |
| §1.2 typed speech acts + parser + fallback | I-5 | T4.12, T4.8 |
| §2 matrix rows / extension mechanism | I-5 (+ guardrail §12.1) | T4.8, T1.6 |
| §3 separations 1–8 | I-2 ✅ / I-3 / I-5 / I-6 / I-4b / I-7 | T1.3–T1.6, T2.3, T4.10, T4.3, T3.10, T5.3 |
| §3.1 evidence ledger, chains, correlation, secrets | I-2 ✅ | T1.1–T1.4, T1.8b |
| §3 `N` needs | I-4a | T3.7–T3.8 |
| §3 `E` emotions | I-4b | T3.10–T3.12 |
| §3.2 directed contextual relationships | I-4 | T3.1–T3.4 |
| §4.1 six stages | I-3b/I-6 (+ I-2 ✅ perceive/interpret) | T2.8, T2.1, T4.4 |
| §4.2 utility, lexicographic, softmax sampling | I-3b | T2.8, T2.9b, T3.9 |
| §4.3 deliberation tiers | I-6 | T4.4, T4.13 |
| §5.1 dyadic history | I-4 | T3.1, T3.5 |
| §5.2 contract FSM + partial/interruption semantics | plan §4 + I-3 | T2.2–T2.5e, T6.3 |
| §5.3 knowledge network | I-2 ✅ / I-6 | T1.2, T1.7, T4.6 |
| §5.4 resources/work, conservation | I-6 / I-8 | T4.2, T6.1 |
| §5.5 factions as institutions | I-6 | T4.3 |
| §6.1 macro structures | I-6 / I-9 | TM.1–TM.3 |
| §6.2 quests, substitutes | I-7 | T5.1–T5.2 |
| §6.3 director constraints | I-7 | T5.3 |
| §6.4 stability/feedback control | I-4a/I-4b/I-6 | T3.8, T3.11, T4.2, T-L1 |
| §7 worked chain | scenario gate | T0.1 (grocer + spy scenarios), trace test |
| §8 macro algorithm (multi-rate, ordering, catch-up) | I-6 | T4.5–T4.7, T4.4 |
| §9.1.1–12 safety properties | I-2 ✅ / I-3 / I-5 / I-6 / I-7 / I-8 | T6.1, T6.5, T1.5, T1.4, T4.5, T2.3, T5.3, T2.2, T6.5, T0.5, T1.5b, T6.5b |
| §9.2.1–5 liveness | I-10 (+ I-3/I-6/I-7) | T-L1–T-L5 **[EVIDENCE]**; proofs where applicable T6.3/T6.4 |
| §9.3 verification layers | I-8 | T6.1–T6.8 |
| §9.4 inductive strategy + audit + CI gate | I-5/I-8 | T6.6, T6.7, M1 |
| §11 Phase 0–6 deliverables | I-1..I-8 | per-phase sections above + T0.1 |
| §11 metrics M1–M7 | I-5/I-6/I-8/I-9 | M1–M7 rows above |
| §12.1–12 guardrails | mapped owners per plan §7 | T4.8, T1.5, T3.2, TM.2, T1.6b/T1.9, T4.4, T2.1b, T4.12, T5.3, T6.6, T-L1/T4.2, T0.5/T6.5b |
| §13 micro-world experiment | I-9 | TM.1–TM.5, T2.6 |
| Backward compat / migration | I-1 ✅ / I-11 | T0.5, T1.8, persistence migration test |
