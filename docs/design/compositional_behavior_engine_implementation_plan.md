# Compositional Behavior Engine — Implementation Plan

Authoritative design: `docs/design/compositional_behavior_engine.md` (merged via PR #8, 581 lines).
Base: `origin/main` @ `a4ee23bbec97ed2d5aa6a155f17488a34f54b2a0` ("docs: add compositional behavior engine design (#8)").
Branch: `feat/compositional-behavior-engine` (created from origin/main; old `feat/believable-npcs-persistent-conversations` branch untouched).
Revision 2 of this plan: addresses the plan review (production-kernel formal harnesses, micro-world generated evaluation, needs/emotions increments, fair-scheduler liveness evidence, pinned contract settlement semantics, scoped grocer checks, direct WASM build, ongoing proof-limits CI check, recorded benchmark acceptance).

## 1. Baseline (verified, not assumed)

| Check | Command | Result |
|---|---|---|
| Toolchain | `rustup show active-toolchain` | 1.98.1-x86_64-pc-windows-msvc (pinned via `rust-toolchain.toml` MSRV 1.91 / Bevy 1.97.1 note) |
| Working tree | `git status` | clean at base a4ee23b **plus the delivered Phase 0–1 work below (uncommitted)** |
| fmt | `cargo fmt --all --check` | exit 0 |
| clippy (8 core crates, `-D warnings`) | `cargo clippy -p rpg_domain ... -p rpg_eval --all-targets` | exit 0 |
| Tests | `cargo test` on domain/perception/social/narrative/agents/sim/persistence/behavior | all pass: domain 14(+5 behavior_types), perception 4(+doc 5 ut), narrative 4, social 7, agents 28+11(autonomy)+18(sim_actions), sim 4+8(behavior_engine)+11(autonomy), persistence 10+1(v2→v3 migration), **behavior 12** |
| Scenario gate | `cargo run -q -p scenario-runner -- --seed 2026` | 10/10 PASS (friendly_investigation, grocer_conditional_bargain, private_conversation_and_spy, blackmail_has_consequences, ignore_plot_world_progresses, save_reload_continuity, deterministic_replay, knowledge_boundaries, three_independent_routes, innocent_grocer_stays_innocent) |
| Benchmark acceptance | `cargo test -p rpg_eval --test behavior_baseline -- --nocapture` | **baseline 93 us/step** (300 ticks, seed 2026, 3 decisions, 83 events), replay reproduces event checksum + full world JSON + full epistemics JSON. See §8. |
| Formal tools | `Get-Command kani/verus/lean` | NOT installed locally — Phase 6 local verification blocked; Kani must run in CI (Linux job) or be installed later. Reported as blocker, not silently skipped. |

No `AGENTS.md` at repo root. `.omx/` holds tracked mirrors of the two planning artifacts (`compositional_behavior_engine_implementation_plan.md`, `compositional_behavior_engine_test_spec.md`); canonical artifacts live in tracked `docs/design/`.

## 2. Delivered work (Phase 0–1) and remaining integration gaps

The working tree already contains delivered Phase 0 and Phase 1 work. **It is preserved; nothing below re-does it.** Verification evidence: 12 behavior tests, 8 sim `behavior_engine` tests, 3 eval `behavior_phase1` tests, 1 eval `behavior_baseline` test, persistence v2→v3 migration test — all green at revision time.

### 2.1 Delivered — Phase 0 (instrumentation and shared schema)
- `crates/domain/src/behavior_types.rs`: the 16-primitive `Effect` algebra (`Effect::family()` covers exactly the design §1.1 vocabulary, unit-tested), `Intent`/`IntentKind`, `SpeechAct` (incl. `Assert { deceptive }` and `Unknown` fallback), `Evidence`, `SourceChainId`/`SourceChain`, canonical `Epistemics` state, `AcquisitionChannel` (with default reliabilities), `GoalPredicate`/`AbandonmentCriteria`, `ActionOutcome` (rejected ⇒ no effects, no events, by construction), `CausalTrace` with `render()` ("did X because of goal Y, belief Z, observed event E, cost C").
- `World::epistemics` canonical field (`serde(default)`), `EventKind::{EvidenceRecorded, RequestMade, EvidenceHidden}` + renderings.
- `Simulation`: `outcomes` + `causal_traces` buffers, `effects_buf` drained per dispatch, `build_trace`, rejection-mutates-nothing invariant.
- `rpg_persistence`: `SAVE_VERSION` 2→3 with forward migration (`PRE_EPISTEMICS_SAVE_VERSION`) + regression test (v2 save loads; event checksum untouched; restored sim replays identically).
- Benchmark baseline artifact: `crates/eval/tests/behavior_baseline.rs` (step time, decisions/min, full replay stability).

### 2.2 Delivered — Phase 1 (epistemic microcore)
- `crates/behavior` (new crate, pure `rpg_domain` + serde; leaf layer):
  - `epistemics.rs`: `record_evidence` (chain inheritance across retellings), `aggregate_about` (correlation-protected fold: group weight = max reliability + 0.08/distinct extra chain, capped at +0.24; neutral mass 0.5), `prior_strength`, `accept_assertion` (evidence always recorded + typed `EvidenceRecorded` event; belief rolled, never forced; empty ledger reproduces the legacy probability exactly), `knows_of` (knowing-of ≠ believing ≠ permitted-to-disclose).
  - `mechanics.rs`: `answer_query` (ledger → authored belief → Unknown; never world truth), `assert_proposition` (any polarity; `deceptive` parameter: lying is a proposition choice, never a truth mutation; requires only knows-of), `make_request`, `record_refusal`, `skeptic_seeks_evidence` policy.
  - `adapters.rs`: provenance→draft/record bridges.
- `rpg_social`: `share_claim`/`record_belief` rewritten on the ledger (legacy snapshot + evidence item; compatibility preserved for `do_share_claim`/`do_warn_contact`).
- `rpg_sim`: `do_report_crime` records each reported fact as provenance-bound evidence (claim separation at the authority boundary).
- Test coverage: T1.1–T1.8 equivalents in `crates/behavior/tests/epistemic_ledger.rs` (12), `crates/sim/tests/behavior_engine.rs` (8), `crates/eval/tests/behavior_phase1.rs` (3).

### 2.3 Remaining integration gaps (must not be silently deferred)
| # | Gap | Design anchor | Detail |
|---|---|---|---|
| G1 | **Canonical deceptive assertions end-to-end** | §1.1 ASSERT, §3.1 lying, §11 metrics ("including explicitly intentional lies") | `assert_proposition(deceptive)` exists, and `SpeechAct::Assert { deceptive }` exists, but the authoritative sim lowering (`do_share_claim`) only ever records `asserted_polarity: true` and no `Action`/proposal pathway carries a polarity-inverted intentional lie through validation, effects, and the M2 audit. Lying must be representable *and auditable* without bypassing knowledge boundaries (`knows_of`). Owner: I-5. |
| G2 | **QUERY lowering** | §1.1 QUERY ("no omniscient answer") | `answer_query` exists in behavior, but no authoritative sim path dispatches a question into `Effect::Query` + a ledger-derived answer + typed events; NPC "answers" today flow through speech rendering only. Owner: I-5. |
| G3 | **SANCTION lowering** | §1.1 SANCTION ("authority, process, cost and evidence") | No action lowers into `Effect::Sanction`; threat/blackmail consequences exist, but fine/arrest/boycott/praise as institutional consequences with authority+evidence checks are unlowered. Owner: I-5. |
| G4 | **Background typed mutations** | §8 algorithm, §9.1.5 ("all state changes route through typed validated transitions"), M1 | Needs decay (once I-4a lands), goal-urgency updates, memory salience decay, offscreen catch-up (I-6) mutate actor state inside `step_impl`/autonomy without typed effects or events. Every background mutation must be classified: event-sourced typed effect, or explicitly declared non-canonical (with the M1 audit enforcing the classification). Owner: I-6 (+ I-4a/I-4b for their own decays). |
| G5 | **Needs/emotions state absent** | §3 `N`, `E`; §4.2 α_i·Δneeds term | `Actor` has no needs or temporary-emotion fields; no decay/thresholds/sensitivities; utility has no needs term. Owners: I-4a, I-4b, I-3b. |
| G6 | **Contract FSM absent** | §5.2 | `domain::social::Commitment` (Pending/Satisfied/Broken/Expired) is not the design FSM: no Proposed→Accepted→Active→ConditionMet→PerformanceDue→{Performed/Violated/Excused} + Rejected/Canceled lifecycle, no condition-met ≠ benefit-delivered separation, no partial-delivery/interruption semantics, no exactly-once settlement guard. Semantics pinned in §4 below *before* Phase 2 implementation. Owner: I-3. |
| G7 | **Fair-scheduler liveness evidence** | §9.2 | The five liveness properties have no bounded fair-scheduler tests; nothing distinguishes monitored bounded evidence from proof artifacts. Owner: I-10. |
| G8 | **Micro-world generated evaluation absent** | §13 | No seeded-variant micro-world battery; M3/M7 are unmeasured. Owner: I-9. |

## 3. The 16 primitives (design §1.1) — ownership and lowering status

`MOVE, TRANSFER, TRANSFORM, OBSERVE, ASSERT, QUERY, REQUEST, OFFER, COMMIT, AUTHORIZE, APPRAISE, SANCTION, SCHEDULE, CHANGE_GOAL, CHANGE_TIE, CREATE_EVENT`

Every authoritative world mutation lowers into zero or more of these. Status: **lowered** = a production dispatch path already produces the effect through `validate_then_apply`; **gap** = no production lowering yet (owners in §6).

| Primitive family | Lowering + validation | Owner crate | Status / seam |
|---|---|---|---|
| `MOVE` | path/speed guard | `rpg_sim` | **lowered** (`do_move` → `Effect::Move` on committed path) |
| `TRANSFER` | quantity/ownership ledger integrity | `rpg_sim` | **lowered** (give/pay/search-recover/fund/repair paths); conservation invariant formal kernel in I-8 |
| `TRANSFORM` | input/output accounting | `rpg_sim` | **lowered** (hide→conceal, repair) |
| `OBSERVE` | channel delivery, no route → no record | `rpg_perception` + `rpg_sim` | **lowered** (speak to listeners, investigate clue finds) |
| `ASSERT` | claim record, no auto-belief | `rpg_domain` (types) + `rpg_behavior` (acceptance) + `rpg_social`/`rpg_sim` (dispatch) | **lowered for truthful path**; **G1: deceptive polarity unlowered** (I-5) |
| `QUERY` | no omniscient answer | `rpg_behavior` + `rpg_sim` + `rpg_perception` | **G2: not lowered** (I-5); mechanics exist (`answer_query`) |
| `REQUEST` | request ≠ compliance | `rpg_behavior` + `rpg_sim` | **lowered** (`make_request` events; private-talk demand; report-crime demand; threat demand) |
| `OFFER` | acceptance ≠ fulfillment | `rpg_sim` (offers) + `rpg_domain::deal` (FSM, I-3) | **lowered** (make/accept/reject offer + `Effect::Offer`); FSM acceptance→settlement is G6 |
| `COMMIT` | typed predicate lifecycle | `rpg_domain::deal` (I-3) + `rpg_persistence` | **partially lowered** (`Effect::Commit` on promise/accept); lifecycle states are G6 |
| `AUTHORIZE` | scoped capability grants | `rpg_domain` + `rpg_sim` | **lowered** (recruit → scoped crew grant) |
| `APPRAISE` | actor-local subjective eval | `rpg_behavior` + `rpg_social` | **lowered as recorded appraisal events** (threat/comfort); full emotions model is G5 (I-4b) |
| `SANCTION` | authority, process, cost, evidence | `rpg_sim` + `rpg_social` | **G3: not lowered** (I-5) |
| `SCHEDULE` | clock/conflict handling | `rpg_sim` | **lowered** (private-talk reservation, `do_schedule`) |
| `CHANGE_GOAL` | no illicit world-truth mutation | `rpg_behavior` + `rpg_sim` | **lowered** (faction funding completion); full goal lifecycle with urgency in I-3b/I-6 |
| `CHANGE_TIE` | directed, contextual edge update | `rpg_social` | **lowered** (gift/threat/comfort/claim/join ties) |
| `CREATE_EVENT` | immutable event id + ordering | `rpg_sim` event log + `rpg_persistence` | **lowered** (speech, offers, investigations, concealment, funding) |

## 4. Contract settlement semantics — PINNED before any Phase 2 implementation

This section is the **normative specification** for design §5.2 ("Partial delivery or interruption needs defined semantics"). The FSM, its transitions, and exactly these settlement rules are implemented in I-3; test-spec T2.2–T2.5x test *this* text, not an implementation-defined reading. Naming: the FSM lives in a **new module `rpg_domain::deal`** — `rpg_domain::contract` is already the LLM action JSON contract (`ACTION_SCHEMA`/`validate_action_json`) and must not be overloaded.

### 4.1 Deal FSM
```
Proposed -> Accepted -> Active -> ConditionMet -> PerformanceDue -> Performed | Violated | Excused
    |            |          |             |                |
    v           v          v             v                v
 Rejected    Canceled   Canceled      (obligation      Renegotiated -> Active
                                      becomes due)     (single renegotiation cycle;
                                                       creditor must consent)
```
- Terminal states: `Rejected`, `Performed`, `Violated`, `Excused`, `Canceled`. Exactly-once: every terminal transition consumes the deal; a second terminal transition on the same id is rejected by validation (T2.2, Kani T6.3).
- **ConditionMet ≠ benefit delivered**: `ConditionMet` records that the trigger predicate holds (e.g. grocer physically received the medicine); only `PerformanceDue` → `Performed` records the promised benefit (e.g. the ledger disclosure). Separate events, separate state.
- A fulfillment claim (speech) never moves the FSM: only validated world predicates or settled `Effect::Transfer`/`Transform` evidence do (T2.4).

### 4.2 Escrow and atomicity
- **Escrow = atomic settlement inside one validated proposal, not an inventory lock.** A settlement proposal (the `TRANSFER`/`TRANSFORM` bundle that discharges a due obligation) is applied by a single `validate_then_apply` call: either every transfer in the bundle succeeds or nothing mutates (rejection mutates nothing — invariant already enforced by `apply_impl`).
- The debtor commits only resources it controls; the sim validates custody at settlement time. There is **no v1 reservation/lock mechanic** (no third-party inventory holds): this is a documented design choice, revisited only if concurrency is introduced (design §8).
- Give-on-acceptance consideration (`Offer.give`) moves atomically at `Accepted`, before the FSM enters `Active`.

### 4.3 Partial obligations
- A deal may declare **installments**: an ordered list of typed predicates/quantities, each with its own settlement. Each installment settlement is its own exactly-once transition; the deal reaches `Performed` only when all installments are settled (partial state is observable: `performed_installments < total`).
- An **un-split** obligation accepts no partial settlement: a delivery short of the contracted quantity leaves the deal in `PerformanceDue`, records a typed `DealPartialDelivery { deal, got, expected }` event (evidence for creditor renegotiation/appraisal), and mutates nothing contractual. The creditor may then: (a) propose `Renegotiated` (creditor consent required; one cycle), (b) `Excuse` via authority/institution evidence, or (c) let the deadline lapse into `Violated`.
- Interruption of a multi-step performance (e.g. actor midway through a delivery route): in-flight progress is recorded as events (per-tick `MOVE`/`OBSERVE`), the obligation remains `PerformanceDue` (or `Active` if the condition was met by someone else); no automatic compensation — creditor response is a decision, not a mechanic.

### 4.4 Cancellation and interruption
- `Canceled` is legal from {`Proposed`, `Accepted`, `Active`} by: mutual consent, the creditor unilaterally, or the debtor **before** `ConditionMet` with no consideration exchanged. Debtor cancellation after `ConditionMet` is not `Canceled` — it is the `Violated` path (recorded, with appraisal/sanction consequences).
- From `PerformanceDue`: creditor consent ⇒ `Canceled`; authority/institution evidence ⇒ `Excused`; otherwise deadline lapse ⇒ `Violated`.
- A violated or canceled deal leaves its event history intact (audit); apologies/reparations are new deals or `SANCTION`-family consequences, not FSM edits.

### 4.5 Deadlines
- `deadline_minute` applies to `Active` (condition deadline) and `PerformanceDue` (performance deadline) independently.
- Crossing a deadline emits exactly one typed `ObligationDue { deal }` event (idempotent by state check) and starts the configured grace policy: renegotiation window, then `Violated` with a typed `DealViolated` event. Deadline processing is part of the deterministic `step` pipeline (§8 ordering), never an out-of-band mutation (G4 applies).

### 4.6 Exactly-once settlement
- Settlement is a guarded FSM transition: `validate_then_apply` checks current state; the settlement event carries the deal id; replaying the event log cannot double-settle because the state check is part of the recorded, deterministic transition sequence. Kani harness T6.3 proves exactly-once over bounded inputs; property test T6.5 exercises it over generated sequences.

## 5. Phase plan (design §11), mapped to work items

### Phase 0 — Instrumentation and shared schema — **DELIVERED** (§2.1)
### Phase 1 — Epistemic microcore — **DELIVERED** (§2.2)
### Phase 2 — Goals, intentions, commitments — **OPEN (G6 + G1)**
- Precondition: §4 semantics pinned (done in this revision). No Phase 2 code lands before this spec is merged.
- I-3 deal FSM per §4 (new `rpg_domain::deal` module), persistent intentions (`J`), utility-based option comparison (I-3b).
- **Grocer migration, scoped to the production new pathway**: the hardcoded branch at `crates/agents/src/lib.rs:982` (`if a.id.as_str() == "grocer" && wants_ledger`, feeding `f_grocer_ledger` at 987/1000) and the per-actor literal-id idle lines at `agents/src/lib.rs:1028` must leave the *new decision pathway* only. The legacy `heuristic_decision` fallback and scenario content fixtures (`f_grocer_*`) are explicitly allowed to retain ids until I-11 completes. Enforcement is the two-part check in test-spec T2.6 (scoped grep + rename-equivalence), not a repo-wide grep.
- Deliverable: grocer scenario still passes; bargain emerges from goal+obstacle+economics via the new pathway.
- **File ownership**: `crates/behavior` (intentions, utility), `crates/domain/src/deal.rs` (FSM), `crates/sim` (settlement validation, exactly-once guard), `crates/persistence` (deal serde + version bump only if needed), tests `crates/sim/tests/deals.rs`, `crates/eval`.

### Phase 3 — Social graph, needs, emotions — **OPEN (G5)**
- I-4: directed contextual relationship edges (trust-in-accuracy, trust-in-benevolence, affection, respect, fear, anger, obligation, familiarity, perceived power), bounded saturating updates, per-dimension decay, context separation; appraisal ΔR_ij = f(e, V_i, B_i, goals_i, R_ij, context) — no universal deltas; memory salience + forgiveness/retaliation.
- I-4a **needs state** (design §3 `N`): `Actor.needs` (safety, belonging, wealth, health, status) with per-need decay rates, satiation thresholds, and urgency mapping into goal selection; needs decay is a **typed background mutation** (G4): declared effect classification + event-sourced threshold crossings (`NeedThresholdCrossed`), regular decay non-evented by explicit classification in the M1 audit.
- I-4b **emotions state** (design §3 `E`): `Actor.emotions` (temporary appraisal variables: fear, anger, joy, grief, pride, shame) with decay, thresholds, and sensitivities; `Effect::Appraise` writes emotions (mind-local); threshold crossings gate response classes (flee/retaliate/comfort/reconcile) instead of hardcoded reactions.
- **File ownership**: `crates/domain` (Needs/Emotions types + config structs), `crates/behavior` (appraisal + decay/threshold kernels), `crates/social` (relationship edges), `crates/sim` (tick pipeline integration), tests `crates/social/tests/`, `crates/behavior/tests/{needs,emotions}.rs`, `crates/eval`.

### Phase 4 — Autonomous world — **OPEN (G4, G2, G3 remainder)**
- I-6: bounded event-driven plans (fast/medium/heavy deliberation tiers; heavy = LLM or bounded search, gated on expected value); scarcity accounting (money `i64` fixed-count — never floats; conservation modulo recorded sources/sinks); factions as institutions (assets, officers, policies; group beliefs are aggregates of member reports, never truth access); offscreen catch-up (bounded events from schedules/ledgers, expand around player; explicit causal-equivalence classification list: debts, deaths, clues, resources exact — incidental animation approximate); deterministic conflict resolution (tick, priority class, stable sequence; optimistic validation + rejection + replan); **background typed mutation audit** (G4) with the M1 gate.
- **File ownership**: `crates/behavior` (planner), `crates/agents` (LLM policy + scheduler; keep `temperature 0.0`, `seed = request_id`; `dist(seed)` LlamaEngine), `crates/sim` (conflict resolution + background mutation classification), `crates/content` (faction content), tests `crates/sim/tests/autonomy.rs`, `crates/eval`.

### Phase 5 — Emergent quest and story recognition — **OPEN**
- I-7: quest recognizer Q = (beneficiary, goal, blocking predicates, achievable pathways, stake, contract?) — recognizes rather than authors; validates against causal prerequisites + authored anchors (reuse `narrative` anchor validation). Director constrained: proposes only from uncommitted plausible opportunities; no truth mutation, no witnessed-event invention, no teleporting resources, no repeated-investigation-to-culprit promotion.
- **File ownership**: `crates/narrative`, tests `crates/narrative/tests/`, `crates/eval` scenario extensions.

### Phase 6 — Formal proof coverage — **OPEN, corrected**
Correction of the review finding: harnesses must exercise the **actual production kernels**, never reimplemented toys; production pathways must call those same kernels; the audit must be enforceable; Kani bounds must be a documented, CI-checked artifact; numeric arbitration gets a proof; replay gets full event-log property tests.

- **Production-kernel rule**: every Kani harness calls the production `pub fn` directly (same symbol the sim/social/behavior dispatch calls). Kernels and their production call sites:
  1. **Transfer/accounting conservation**: `rpg_sim` inventory transfer validation + apply path used by `do_give`/`do_pay` (if extraction of a pure helper is needed for harnessability, the helper is *moved into* the sim dispatch so production calls it — never duplicated).
  2. **Numeric arbitration (epistemic)**: `rpg_behavior::epistemics::{aggregate_about, prior_strength}` and the acceptance-probability clamp inside `accept_assertion`. Proof obligations (bounded inputs): (P1) probability ∈ [0.05, 0.97]; (P2) empty ledger reproduces the legacy formula exactly; (P3) k retellings from one correlation group contribute weight ≤ max_reliability + 0.24 for any k (bounded corroboration — five retellings ≠ five confirmations); (P4) `contradicted` iff both polarities weigh ≥ 0.25; (P5) weight/reliability clamps hold (no NaN/overflow over bounded f32 inputs).
  3. **Deal lifecycle**: `rpg_domain::deal` transition function — exactly-once terminal settlement, no illegal transitions, ConditionMet ≠ Performed.
  4. **Authoritative replay**: replay of recorded ops over bounded step sequences reproduces the event log exactly.
- **Enforceable audit**: `crates/proof-kernels/AUDIT.md` maps each harness → production symbol(s) → production call sites (file:fn). CI script `scripts/check-proof-audit.sh` (new, part of I-8) fails when: a harness references a symbol absent from production, a listed production call site no longer exists, or a kernel is called in production *not* through the audited symbol (grep-level check over `crates/{sim,social,behavior}/src`). Deliberately-introduced violation (toy reimplementation or unaudited call) must fail CI (negative test, run once manually then reverted).
- **Proof-limits artifact, ongoing CI check**: `crates/proof-kernels/limits.toml` documents per-harness bounds (unwind depth, max actors/items/inventory stacks, max ticks, max ledger entries). The CI `proof` job (Linux, Kani) runs every harness with its declared bounds **and** a limits check: a harness whose bounds are missing from `limits.toml`, or an artifact entry with no matching harness, fails CI. The artifact is updated in the same PR as any harness change — an ongoing gate, not a one-time document.
- **Full event-log replay property tests**: `crates/eval/properties/replay.rs` (proptest): generated bounded action-proposal sequences (typed generator over the 16 families); `replay_ops` must reproduce the **full event log** (length, checksum, every event), the **full world JSON**, and the **full epistemics JSON** — not summaries.
- **Property-based safety tests**: `crates/eval/properties/safety.rs`: arbitrary bounded actors/actions/contracts/inventories satisfy design §9.1.1–12.
- Layers kept: unit/integration (exists), Kani (above), Verus-verified transition functions (optional/stretch), Lean 4 abstract proofs (stretch, clearly marked separate model), TLA+ (deferred unless concurrency added).
- **Blocker**: kani/verus/lean not installed locally; Kani is Linux CI-only. Property-test equivalents run locally. Do not claim local formal verification without the tool.
- **File ownership**: `crates/proof-kernels` (new), `crates/eval/properties/`, `.github/workflows/ci.yml` (proof job + limits/audit checks), `scripts/check-proof-audit.sh`, `Makefile` (`make verify` extended).

## 6. Independent implementation partition & integration order

Each work item is separately landable; shared interfaces land first. Status and concrete ownership:

| Item | Scope | Owner files | Depends on | Gate (test-spec refs) |
|---|---|---|---|---|
| **I-1** ✅ | domain behavior_types + persistence v3 | `crates/domain/src/behavior_types.rs`, `crates/persistence/src/lib.rs` | — | T1.8, persistence migration test — **met** |
| **I-2** ✅ | behavior crate: evidence ledger, source chains, acceptance, shared mechanics; social/sim adapters | `crates/behavior/src/*`, `crates/social/src/lib.rs`, `crates/sim/src/lib.rs` (report path) | I-1 | T1.1–T1.8 — **met** |
| **I-3** | deal FSM per §4 + intentions `J` + exactly-once settlement | `crates/domain/src/deal.rs` (new), `crates/sim/src/lib.rs` (settlement), `crates/persistence` | I-1, §4 (this doc) | T2.1–T2.5x, T6.3 |
| **I-3b** | utility ranking: feasibility filter, lexicographic hard constraints, α_i·Δneeds term, switch penalty | `crates/behavior/src/utility.rs` (new), `crates/agents` (dispatch), `crates/domain` (Needs read) | I-2, I-4a | T2.8, T3.9 (needs-differential ranking) |
| **I-4** | directed contextual relationships + appraisal + forgiveness/retaliation | `crates/social/src/lib.rs`, `crates/behavior/src/appraisal.rs` (new) | I-2 | T3.1–T3.6 |
| **I-4a** | needs state `N`: types, decay, satiation/thresholds, typed background mutation, urgency mapping | `crates/domain/src/needs.rs` (new), `crates/sim/src/lib.rs` (tick), `crates/behavior` (urgency) | I-1 | T3.7–T3.8, part of T-L1 evidence |
| **I-4b** | emotions state `E`: types, decay/thresholds/sensitivities, APPRAISE writes emotions, response gating | `crates/domain/src/emotions.rs` (new), `crates/behavior/src/appraisal.rs`, `crates/sim/src/lib.rs` | I-4a | T3.10–T3.12 |
| **I-5** | lower the gaps: QUERY path (G2), SANCTION path (G3), deceptive-assertion end-to-end (G1) | `crates/sim/src/lib.rs` (dispatch), `crates/behavior/src/mechanics.rs` (wiring), `crates/agents` (speech-act candidates), `crates/domain/src/action.rs` (new action variants if needed) | I-2 | T1.9 (deceptive assertion audit), T4.9 (query), T4.10 (sanction) |
| **I-6** | autonomous world: deliberation tiers, scarcity, factions, catch-up + classification, conflict resolution, budget caps, **background typed mutation audit (G4)** | `crates/behavior/src/planner.rs`, `crates/agents/src/lib.rs`, `crates/sim/src/lib.rs`, `crates/content/src/lib.rs` | I-3, I-4a/b, I-5 | T4.1–T4.8, M1 audit, M6 |
| **I-7** | quest recognizer + constrained director | `crates/narrative/src/lib.rs` | I-3 | T5.1–T5.4 |
| **I-8** | proof kernels + CI: production-kernel harnesses, numeric-arbitration proof, limits.toml + ongoing CI check, AUDIT.md + audit script, replay/safety property tests | `crates/proof-kernels/*`, `crates/eval/properties/*`, `.github/workflows/ci.yml`, `scripts/check-proof-audit.sh`, `Makefile` | I-2, I-3, I-5 | T6.1–T6.8 |
| **I-9** | micro-world generated evaluation (design §13): generator, ≥300 seeded variants, M3/M7 automated gates | `crates/eval/src/microworld.rs` (new), `crates/eval/tests/microworld.rs` (new) | I-2, I-5 (full act coverage) | TM.1–TM.5, M3, M7 |
| **I-10** | fair-scheduler liveness harness (bounded monitored evidence, labeled) | `crates/eval/src/liveness.rs` (new), `crates/eval/tests/liveness.rs`, `crates/sim` (activation scheduler interface) | I-3, I-6 | T-L1–T-L5 |
| **I-11** | legacy migration completion: grocer scenario flips to new pathway; literal-id branches leave `heuristic_decision` too | `crates/agents/src/lib.rs`, `crates/content` fixtures stay as content | I-3b, I-5, I-9 parity | T2.6/T2.7 strengthened (no pathway has id branches), T0.1 |

Parallel-safe pairs: I-4 ∥ I-4a ∥ I-5 (all after I-2); I-3b after I-4a; I-6 ∥ I-7 (after I-3); I-8 harness scaffolding after I-2 (deal kernel after I-3); I-9 after I-5; I-10 after I-6.

## 7. Requirement → owner → test mapping (complete; no unowned full-design requirements)

Every design-section requirement has exactly one owning increment and at least one named test. (Test IDs → test spec.)

| Design requirement (section) | Owner | Tests |
|---|---|---|
| §1 interaction definition: feasibility vs permission; not-possible/not-permitted/not-believed/attempt-failed distinctions | I-5 (validation vocabulary), I-3b | T4.9–T4.11, T2.8 |
| §1.1 16-primitive effect algebra, world-authoritative vs mind-local | I-1 ✅ types, I-5/I-6 lowering | T6.6, T-sim `every_accepted_action_produces_typed_effects`, per-primitive rows §3 |
| §1.2 typed speech acts + parser candidates + Unknown fallback; never raw-text world writes | I-5 (agents parsing), I-1 ✅ types | T4.12, T4.8 |
| §2 interaction matrix (behavioral ontology; typed extension mechanism, unknown-intent behavior) | I-5 + §12.1 guardrail | T4.8, T1.6 |
| §3 state model separations 1–8 (true/believed, claim/observation, knowledge/disclosure, agreement/performance, authority/capability, individual/institution, effect/appraisal, memory/narrative) | I-2 ✅ (1,2,3), I-3 (4), I-5 (5), I-6 (6), I-4b (7), I-7 (8) | T1.3–T1.6, T2.3, T4.10, T4.3, T3.10, T5.3 |
| §3.1 evidence ledger, source chains, correlation protection, approximate inference, secret protection | I-2 ✅ | T1.1–T1.4, T1.8b |
| §3.2 directed contextual relationships, saturating updates, decay | I-4 | T3.1–T3.4 |
| §3 `N` needs (decay, thresholds) | I-4a | T3.7–T3.8 |
| §3 `E` emotions (decay, thresholds, sensitivities) | I-4b | T3.10–T3.12 |
| §4.1 six-stage decision algorithm | I-3b/I-6 (stages 3–6), I-2 ✅ (1–2) | T2.8, T2.1, T4.4 |
| §4.2 utility with beliefs-only expectations, lexicographic constraints, switch penalty, seeded softmax sampling with recorded samples | I-3b | T2.8, T2.9b (recorded sampling), T3.9 |
| §4.3 deliberation tiers, heavy path gated on expected value; proposals never instantiate new physics/rights | I-6 | T4.4, T4.13 |
| §5.1 dyadic relationship history | I-4 | T3.1, T3.5 |
| §5.2 contract FSM, condition-met ≠ benefit-delivered, partial delivery/interruption semantics | §4 (this doc) + I-3 | T2.2–T2.5x, T6.3 |
| §5.3 knowledge network (edges, source chains, no universal broadcasts) | I-2 ✅ chains, I-6 (channel routing) | T1.2, T1.7b, T4.6 |
| §5.4 resources and work: production consumes inputs, conservation modulo sources/sinks | I-6 (scarcity), I-8 (kernel) | T4.2, T6.1 |
| §5.5 group coordination: factions as institutions, member-report aggregates | I-6 | T4.3 |
| §6.1 macro emergent structures (mechanisms, not guarantees) | I-6 + I-9 (measurement) | TM.1–TM.5 |
| §6.2 quests as goal/obstacle graphs, substitute satisfaction | I-7 | T5.1–T5.2 |
| §6.3 director constrained power | I-7 | T5.3 |
| §6.4 stability/feedback control (bounded attention, cooldowns, satiation, hysteresis, finite resources) | I-4a/I-4b/I-6 | T3.8, T3.11, T4.2, T-L1 |
| §7 worked causal chain (grocer→spy→faction, mechanically explainable steps) | covered by scenario `grocer_conditional_bargain` + `private_conversation_and_spy` + traces | T0.1, T-sim `causal_traces_explain_goal_belief_event_cost` |
| §8 macro execution algorithm: event-driven multi-rate, deterministic ordering, budgeted processing, catch-up | I-6 | T4.5, T4.6, T4.7, T4.4 |
| §9.1 safety 1 conservation | I-6/I-8 | T4.2, T6.1, T6.5 |
| §9.1 safety 2 disclosure integrity | I-2 ✅ / I-5 | T1.5, T1.9 |
| §9.1 safety 3 claim separation | I-2 ✅ | T1.4, T-sim `report_crime_records_evidence_without_belief` |
| §9.1 safety 4 perception boundaries | perception ✅ + I-2 | T1.5, T-sim knowledge-boundaries scenario |
| §9.1 safety 5 authoritative action | I-1/I-2 ✅, I-6 audit | T-sim `every_accepted_action_produces_typed_effects`, M1 |
| §9.1 safety 6 commitment integrity | I-3 | T2.3, T6.3 |
| §9.1 safety 7 no retroactive narrative contradiction | I-7 | T5.3 |
| §9.1 safety 8 rights/roles scoping | I-5 (AUTHORIZE ✅) + I-3 | T-sim `knowledge_is_not_permission`, T2.2 |
| §9.1 safety 9 temporal consistency | I-6 (ordering) | T4.5 |
| §9.1 safety 10 no dangling references | I-6/persistence | T6.5 |
| §9.1 safety 11 deterministic replay | ✅ delivered + I-8 | T0.5, T6.4, T6.5 |
| §9.1 safety 12 information noninterference | I-2 ✅ smoke, I-8 | T1.5b, T6.5 |
| §9.2 liveness 1 no permanent task deadlock | I-10 | T-L1 (bounded evidence) |
| §9.2 liveness 2 eventual resolution under fair scheduler | I-3/I-10 | T-L2 (bounded evidence), T6.3 (proof of settlement exactly-once) |
| §9.2 liveness 3 offscreen autonomy | I-6/I-10 | T4.1, T-L3 |
| §9.2 liveness 4 budget boundedness | I-6/I-10 | T4.4, T-L4 |
| §9.2 liveness 5 narrative access (multiple independent routes) | I-7/I-10 | T-L5, scenario `three_independent_routes` |
| §9.3 verification layers | I-8 | T6.1–T6.8 |
| §9.4 inductive strategy + single-mutation-interface caveat + CI gate | I-5/I-8 | T6.6, T6.7, M1 |
| §11 Phase 0–6 deliverables | I-1..I-8 | per-phase tests + T0.1 |
| §11 metrics M1–M7 | I-5 (M2), I-8 (M1/M4/M5/M6), I-9 (M3/M7) | §9 metrics section |
| §12 guardrails 1–12 | mapped: 1→I-5 (T4.8), 2→I-2 (T1.5), 3→I-4 (T3.2), 4→I-4/I-6+I-9 (TM.2), 5→I-2 (T1.6b), 6→I-6 (T4.4), 7→I-3 (T2.1b), 8→I-6 (T4.12), 9→I-7 (T5.3), 10→I-8 scope (T6.6), 11→I-6 (T-L1/T4.2), 12→I-6 (T0.5, fixed-point record) |
| §13 micro-world experiment (hundreds of seeded variants; no `if actor_id == "grocer"`, no utterance-keyword hacks; replay; invariants) | I-9 | TM.1–TM.5, T2.6 |

## 8. Acceptance criteria (summary; full detail in test-spec artifact)

1. All 10 existing scenarios PASS at seed 2026 at every phase boundary (verified again this revision).
2. Phase gates (deliverables per §5) each have named tests in the test spec; every §7 mapping row has ≥1 test.
3. Metrics (design §11): 100% typed event-logged mutations (M1, incl. background mutation classification per G4); 100% claims with provenance including intentional lies (M2, unblocked by I-5/G1); **0 omniscient disclosures across ≥300 seeded micro-world variants (M3, automated per-variant check)**; 0 stale contract completions in bounded harnesses (M4); 100% deterministic replay for identical recorded inputs+versions (M5, full event-log property test); configurable planning/LLM budgets enforced (M6); **M7 measurable gates: ≥3 distinct terminal outcome classes across the variant battery; 100% of decisions carry causal traces (goal+belief/observation+cost); 100% of variants replay exactly; 0 literal actor-id branches in the production new pathway (T2.6)**.
4. `make verify` (fmt + clippy + test + scenarios) green at all phases; `make verify` extended by I-8 with property tests + proof job (bounds/audit checks included).
5. **WASM, directly built**: `cargo build -p rpg_behavior --target wasm32-unknown-unknown` (the behavior crate itself — pure std) **and** `cargo build -p web-game --target wasm32-unknown-unknown` (which consumes behavior transitively via rpg_eval→rpg_sim) both succeed at every phase boundary.
6. **Benchmark acceptance (recorded)**: baseline artifact `crates/eval/tests/behavior_baseline.rs` measured **93 us/step** (300 ticks, seed 2026, 3 decisions, 83 events) with full replay stability (event checksum + world JSON + epistemics JSON). Acceptance: the test's replay-stability assertions must hold at every boundary (hard gate); per-step time regression budget ≤ 2× baseline (≤ 186 us/step) as a calibrated soft gate until more load shapes exist (re-measure at each phase boundary; record in the benchmark doc `docs/benchmarks/` when the battery grows).

## 9. Risks and ambiguities

- **Formal tools unavailable locally**: kani/verus/lean not installed; local Phase 6 is limited to property tests. Kani runs as a new CI Linux job (I-8); risk: CI runtime cost — mitigated by documented bounds in `limits.toml` (the ongoing CI check keeps them explicit).
- **Correlated-evidence weighting is explicitly approximate** (design §3.1): calibration values are design choice; acceptance is behavioral (liar-fools-one-skeptic-asks), with the numeric arbitration kernel additionally proven bounded by I-8 (P1–P5).
- **Utility weights w_ik, α, β, γ, ρ, η** are modeling assumptions; risk of uniform behavior; mitigated by diversity gates (M7, TM.2) and needs-differential ranking tests (T3.9).
- **Contract FSM edge cases**: partial delivery, interruption semantics are pinned in §4 *before* implementation (removes the prior circularity); T2.5x tests exercise the pinned text. Residual risk: §4 installments add schema; persistence bump handled in I-3 with serde(default) + version ladder.
- **Grocer migration**: the two `f_grocer_*` content fixtures are scenario content (allowed) — only the algorithmic id-checks must leave the production new pathway; the scoped check + rename-equivalence test (T2.6) prevents both false positives (content matches) and false negatives (legacy-pathway ids during migration). Risk of breaking `grocer_conditional_bargain` if the legacy pathway is removed too early → I-11 sequences it.
- **Catch-up fidelity**: needs explicit classification list in I-6 (debts/deaths/clues/resources exact; animation approximate); the M1 audit enforces that classification is declared, not implicit.
- **Budget**: agents crate is a 2634-line single file; I-3b/I-6 refactors must keep `heuristic_decision` fallback intact for the acceptance gate until I-11.
- **Proof-kernel harnessability**: `aggregate_about` is already a pure `pub fn` (harness-ready); sim transfer paths may need a pure helper *extracted into* production (single source of truth) — the AUDIT.md + CI script (I-8) makes any accidental duplication fail CI.
- **Liveness evidence vs proofs**: T-L1–T-L5 are bounded monitored evidence under a stated fair scheduler — they are **not** proofs and must never be reported as such; the only proof artifacts are the Kani kernels (T6.1–T6.4). Reviewers must keep the distinction.
