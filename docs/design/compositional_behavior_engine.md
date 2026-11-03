# IRL — A Compositional Mathematical Model for Emergent Behavior

**Status:** Proposed architecture, not implemented. **Date:** 2026-11-01.  
**Repository assessed:** [JPrier/IRL](https://github.com/JPrier/IRL), `main` branch as read 2026-11-01.

## Executive design decision

Do not try to enumerate all human interactions as unique rules. That list is infinite and fragile. Instead, define a small **closed vocabulary of typed interaction effects** and compose them into complex behavior. An interaction must be able to modify physical state, information, relationships, intentions, rights, obligations, resources, time, or institutional standing. A new kind of encounter is usually a new *composition* of existing primitives, not a new hardcoded NPC-response tree.

Build a **partially observable, event-sourced, bounded-rational, multi-agent simulation**:

- **Ontology:** actions, actors, properties, claims, evidence, contracts, locations, institutions, and events have precise types.
- **Physics:** deterministic authoritative transition functions guarantee mechanically valid world state.
- **Minds:** individual beliefs and motivations produce imperfect, situated decisions.
- **Society:** interactions propagate over an evolving relationship, information, and resource network.
- **Narrative:** quests and plot possibilities are recognized from causal patterns; they do not dictate reality.
- **Formal methods:** prove *specified invariants*, not universal human realism.

The existing IRL repo already provides `crates/domain`, `perception`, `social`, `sim`, `agents`, `narrative`, `eval`, and event persistence. The proposal is an extension of those boundaries.

---

## 1. Definition of an interaction

An interaction is a typed, time-dependent causal operation:

\[
I=(actor,targets,intent,context,guards,cost,duration,effects,observability)
\]

- `actor`: entity initiating the act; may be a person, group, environment, or institution.
- `targets`: objects/people/locations affected, potentially empty or many.
- `intent`: the actor's desired effect; not identical to what happens.
- `context`: location, time, privacy, prior events, permissions, relationship, capabilities.
- `guards`: *physical* feasibility; distinguish from normative permission.
- `cost`: time, labor, items, money, risk, energy, attention, reputation.
- `duration`: atomic or extended action, with cancellation/interruption behavior.
- `effects`: objective world effects plus subjective appraisal where events are perceived.
- `observability`: who can see/hear/infer the event, with uncertainty.

For an action `a` in world state `W`, first evaluate `physically_feasible(a,W)`, then calculate authoritative `apply(a,W) -> (W',events)`. A crime can be physically feasible even if socially prohibited; laws generate possible detection/sanctions, not magical barriers. Distinguish **not possible**, **not permitted**, **not desired**, **not believed possible**, and **attempt failed**.

### 1.1 The primitive effect algebra

Every high-level action lowers into zero or more of these composable effect families (some are descriptions of intended or observable effects, not unconditional updates):

| Primitive | Meaning | Examples | Mechanical guarantee |
|---|---|---|---|
| `MOVE` | Change position or containment | walk, flee, escort, sneak | reachable path/speed/resource guard |
| `TRANSFER` | Change custody or ownership of resources | give, sell, steal, confiscate | quantity/ownership ledger integrity |
| `TRANSFORM` | Produce/consume/alter an object | repair, craft, burn, sabotage | input/output accounting |
| `OBSERVE` | Deliver a sensory observation to a perceiver | witness, overhear, inspect | no observation without defined channel |
| `ASSERT` | Transmit a proposition as speech or writing | tell truth, lie, rumor, testimony | creates claim record, not automatic belief |
| `QUERY` | Request information | ask, interrogate, investigate | no omniscient answer |
| `REQUEST` | Ask for behavior | plead, order, invite, solicit | request != compliance |
| `OFFER` | Propose conditional exchange | trade, barter, pay for clue | acceptance != fulfillment |
| `COMMIT` | Adopt an obligation/plan | promise, contract, oath | identities, conditions, lifecycle |
| `AUTHORIZE` | Grant/revoke permission or role | consent, recruit, delegate, dismiss | bounded capability and scope |
| `APPRAISE` | Subjectively evaluate an experience | fear, gratitude, jealousy, anger | derived from NPC values and interpretation |
| `SANCTION` | Impose institutional/social consequences | fine, arrest, boycott, praise | authority, process, cost and evidence |
| `SCHEDULE` | Reserve future time/actions | meet, work, deliver | clock and conflict handling |
| `CHANGE_GOAL` | Create, reprioritize, complete or abandon goal | vow revenge, decide to help | no illicit world-truth mutation |
| `CHANGE_TIE` | Update directed social relation | gain trust, resent, befriend | no unmediated global reputation |
| `CREATE_EVENT` | Record occurrence, stimulus, or environmental incident | storm, theft alarm, death | immutable event identifier and ordering |

Some effects are **world-authoritative** (`TRANSFER`, `TRANSFORM`); some are **mind-local** (`APPRAISE`, `CHANGE_GOAL`). Observing an objective effect is *not* guaranteed. A coerced promise is represented as a proposal/adoption with credible motives and later sanctions; it is not mind control.

### 1.2 Speech acts are not one primitive

A natural sentence may contain multiple semantic acts: `ASSERT`, `QUERY`, `REQUEST`, `OFFER`, `THREAT`, `COMMIT`, `REFUSE`, `ACKNOWLEDGE`, `EMOTE`. `THREAT` is a higher-level act combining `ASSERT(conditioned_harm)`, `REQUEST`, and an evaluation of actor credibility/capability. `PERSUADE` is not a state mutation: it is a conversational strategy whose effects depend on evidence, recipient beliefs, values, and motivation.

Free-form voice/text enters through a parser returning a **typed list of candidate speech acts with confidence**, plus noncanonical wording. Unclear requests require clarification or a harmless fallback. Never let parser text directly set a belief, resource, quest, or world fact. The NPC may choose no action, partial compliance, negotiation, deception, or escalation.

---

## 2. Micro-interaction matrix

Each row defines what the affected person *experiences*, what can change, and what might happen next. Complex interactions are combinations of rows. This is deliberately broader than the current `Action` enum; it is a behavioral ontology, not a request to implement every row immediately.

| Interaction | Atomic interpretation and local effects | Possible generated responses |
|---|---|---|
| Greeting / acknowledgment | attention + recognition; familiarity may change | greet, ignore, avoid, initiate topic |
| Ask question | query + attention cost; assess privacy and motive | answer, evade, ask why, price information |
| Tell claim | receive evidence item with speaker and origin | believe, doubt, request proof, repeat as rumor |
| Disagree / correct | conflicting evidence and appraised challenge | revise belief, defend belief, investigate, resent |
| Lie / mislead | assertion independent of actor's own belief | recipient may be fooled; liar risks exposure |
| Conceal / omit | withholding without a false assertion | suspicion if omission inferred; no magical detection |
| Overhear / spy | perception delivery with contextual reliability | remember, warn, exploit, misinterpret |
| Ask for favor | requested future action with effort/cost | agree, refuse, counteroffer, delegate |
| Promise | obligation with trigger, deadline, witness | expect delivery, plan, monitor, complain |
| Break promise | transition to violation after criteria met | reduced trust, sanctions, revenge, forgiveness |
| Offer / barter | proposed mutually conditional resource exchange | accept, reject, counter, renegotiate |
| Gift | transfer with social meaning determined by values/context | gratitude, suspicion, obligation, offense |
| Bribe | offer + requested improper action | accept, expose, report, demand more |
| Threaten / blackmail | requested act + credible adverse consequence | comply, bluff, flee, retaliate, seek protection |
| Comfort / apologize | communicate empathy/remorse; appraised sincerity | reconcile, reject, require reparations |
| Insult / praise | reputational/identity-relevant speech event | anger, pride, shame, indifference |
| Invite / recruit | offer membership/coordination role | join, refuse, negotiate terms, tell others |
| Order / command | request plus claimed authority | obey, question authority, resist, report |
| Consent / refuse | explicit permission or withholding it | respect boundary, pressure, leave, renegotiate |
| Work / deliver | time + skills + physical transfer/change | wage, satisfaction, delay, dispute |
| Buy / sell | conditional transfer of value and goods | price movement, new inventory, rumors |
| Steal / trespass | feasible prohibited physical act | conceal, notice, investigate, sanction |
| Attack / defend | physical force, damage, detection, escape | retaliation, fear, medical aid, police response |
| Aid / rescue | improve another's physical condition | gratitude, obligation, witnesses, prestige |
| Move / follow | location change subject to travel constraints | encounters, observations, pursuit, fatigue |
| Meet / avoid | synchronization/avoidance of schedules and space | exchange, missed opportunity, suspicion |
| Investigate | evidence acquisition with time and search constraints | deduce, fail, find alternative lead |
| Accuse / report | claim to third party/institution, often requesting action | inquiry, disbelief, hearing, retaliation |
| Teach / learn | transfer skill/explanation under comprehension model | capability updates, misunderstandings |
| Vote / organize | coordinate actions and aggregate preferences | coalition, faction action, defection |
| Publicize / protest | broadcast claims, express norms, organize crowd | public attention, repression, support |
| Repair / sabotage | object transformation; optional concealment | damaged services, jobs, suspicion |
| Death / accident / disaster | environmental event and physical consequence | grief, inheritance, repair needs, instability |
| Time passes | needs, availability, deadlines, debts evolve | urgent goals, missed meetings, routines |

**Compositional examples**:
- `Extortion = REQUEST + credible THREAT + conditional COMMIT + monitored compliance`.
- `Courtship = repeated OBSERVE + COMMUNICATE + reciprocal APPRAISE + COMMIT or DECLINE` (not an affection slider).
- `Gossip = ASSERT + OBSERVE + belief update + conditional retransmission`.
- `Business failure = repeated deficient TRANSFER + costs + inventory depletion + unmet obligations + layoffs`.
- `Emergent quest = some agent has a goal + obstacle + available help + plausible contract/path`.
- `Revolt = perceived repeated grievance + group communication + collective goal + recruitment + coordination + collective action`.

A taxonomy of primitives is an *extensible grammar*, not proof that all meaningful human behavior has been captured.

---

## 3. Multi-layer state model

Global canonical state:

\[
W_t=(P_t,R_t,F_t,C_t,K_t,H_t,Q_t)
\]

- `P`: locations, physical objects, health, inventories, world mechanics.
- `R`: resource flows, prices, supply, jobs, institutional assets.
- `F`: world facts, immutable anchors, committed factual constraints.
- `C`: contracts, claims on resources, commitments, permissions, deadlines.
- `K`: institutions, factions, norms, laws, roles and membership.
- `H`: append-only causal event history and references to source events.
- `Q`: flexible narrative possibilities and authored hard bounds, not omniscient actor knowledge.

For each actor `i`:

\[
X_i(t)=(\hat W_i,B_i,V_i,N_i,E_i,G_i,J_i,R_i,Cap_i,Mem_i,Sched_i)
\]

- `W-hat`: subjective, sparse estimate of relevant world state.
- `B`: evidence-based beliefs with confidence and provenance.
- `V`: stable values, preferences, personality, ethical boundaries.
- `N`: physical/social needs (safety, belonging, wealth, health, status).
- `E`: temporary emotion/appraisal variables with decay, thresholds, sensitivities.
- `G`: goals; each objective has predicate, urgency, beneficiary, and abandonment criteria.
- `J`: intentions/plans committed to, including step and continuation state.
- `R`: directed relationship model per known other actor; different people have different views.
- `Cap`: physical competence, access, knowledge, authority and resources.
- `Mem`: selected autobiographical experiences and event-linked cues.
- `Sched`: obligations, routines, appointments and temporal constraints.

Crucial separations:

1. **True vs believed**: `world.fact` may be true, false or undecided; an actor may have contradictory evidence and reach a wrong conclusion. Prefer three-/four-valued representations for unresolved and inconsistent cases rather than simple booleans for everything.
2. **Claim vs observation**: testimony about a robbery is not witnessing a robbery.
3. **Knowledge vs willingness to disclose**: knowing a secret does not imply telling it.
4. **Agreement vs performance**: an accepted deal does not prove payment/delivery.
5. **Authority vs physical capability**: an actor can perform a rule-violating act, with consequences.
6. **Individual vs institution**: faction policy cannot directly read members' private minds or move their possessions without authorized mechanisms.
7. **Objective effect vs subjective appraisal**: receiving a gift need not improve trust; it may be perceived as manipulation.
8. **Memory vs narrative summary**: compact summaries are nonauthoritative; immutable source events remain available for audit.

### 3.1 Belief updates: evidence ledger

Current IRL `Belief` stores `fact`, `believes_true`, `confidence`, one `Provenance`, and acquisition tick. That is useful for a prototype but can overwrite independent evidence with the latest report.

Propose a per-actor, per-proposition **evidence ledger**:

```
Evidence {
    proposition_id,
    observer_id,
    asserted_polarity,
    origin_event_id,
    source_actor_id?,
    acquisition_channel,
    perceived_reliability,
    received_tick,
    source_chain_id,
    independent_corrob_group?,
}
```

Claims preserve their origin chain, so five repeated reports from one fabricated source do **not** count as five independent confirmations. Correlated evidence must be grouped or down-weighted. Evidence can support positive or negative polarity, and an NPC can hold contradictory reports. Repeated assertion should not, by itself, yield certainty.

A simplified Bayesian update for genuinely conditionally independent evidence `e` about proposition `h` is:

\[
P_i(h\mid e) = \frac{P(e\mid h)P_i(h)}{P(e\mid h)P_i(h)+P(e\mid\neg h)(1-P_i(h))}
\]

The likelihoods depend on the recipient's subjective assessment of source accuracy, incentives, directness, and context. Real testimony is typically correlated; don't naively apply this formula once per retelling. Store the raw evidence and implement an explicitly approximate, calibratable inference rule for game performance.

Lying is a choice of a proposition to assert; it is **not** a mutation of truth. The chance of being believed comes from recipient appraisal and evidence, not a boolean `assertive` that can force acceptance. Protect secrets by restricting disclosure choices and recipient observations; do not automatically remove a secret from memory when disclosure is forbidden.

### 3.2 Relationships as directed, contextual edges

`Rel(i,j)` is not `Rel(j,i)`. Use separate trust-in-accuracy, trust-in-benevolence, affection, respect, fear, anger, obligation, familiarity, and perceived power. Existing IRL already has many of these dimensions. Apply bounded, saturating updates and decay at variable rates. Allow context: trusting someone as a mechanic does not imply trusting them with a family secret.

For event `e`, the emotional appraisal must depend on actor `i`:

\[
\Delta R_{ij} = f(e, V_i, B_i, goals_i, R_{ij}, context)
\]

**Do not define an event as universally `+0.2 trust`.** The recipient may infer affection, bribery, humiliation, coercion, or a hidden motive.

---

## 4. The micro decision algorithm

A response is chosen from candidate **plans/actions** using a subjective world model. This resembles BDI (belief–desire–intention) with approximate utility-based planning. Fully general multi-agent POMDP solutions are computationally intractable at game scale; use scoped, shallow planning.

### 4.1 Six stages

1. **Perceive:** `o_i = Observe(i, visible_events, W_t)` using sight, hearing, access and attention constraints.
2. **Interpret:** translate ambiguous speech/sensory data into candidate speech acts and evidence; preserve uncertainty.
3. **Update mind:** revise beliefs, appraisal, goal urgency, relations, commitments and memory salience.
4. **Generate options:** routine continuation, reactions, obligations due, plans supporting goals, trade, request, refusal, counteroffer, avoidance.
5. **Filter/rank:** screen physically impossible options; score remaining options under subjective information, values, norms, risks and bounded planning budget.
6. **Propose:** send *typed* action to authoritative sim, then record actual outcome, rejection and new observations. Never assume a proposal succeeded.

### 4.2 Utility and bounded rationality

Let `A_i(W)` be physically feasible candidates known or imagined by agent `i` (the latter may be wrong). An illustrative utility is:

\[
U_i(a)=\sum_k w_{ik}\Delta goal_{ik}(a) + \alpha_i\Delta needs_i + \beta_i social_i + \gamma_i duty_i - c_i(a)-\rho_i risk_i(a) - \eta_i switch_i(a)
\]

The expectation is calculated with the **actor's beliefs**, not omniscient world truth. Terms should be constrained or contextual; blindly adding normalized scores will not produce convincing psychology. `switch_i` discourages oscillation. Some values may be lexicographic hard personal constraints rather than tradable weights (e.g., an NPC refuses betrayal except under severe coercion). The system distinguishes physically impossible actions from personally unacceptable actions.

For occasional imperfect choices, seeded sampling can use:

\[
P_i(a) = \frac{\exp(U_i(a)/\tau_i)}{\sum_{b\in A_i}\exp(U_i(b)/\tau_i)}
\]

Low `tau` gives more predictable behavior; high `tau` more variation. This is a **modeling assumption**, not a validated theory of human decision-making. Never allow random choice of an invalid action; preserve deterministic replay by including the RNG seed and sample in the replay record.

### 4.3 Deliberation is optional and event-driven

Fast path: continue routine or active intention without new inference.  
Medium path: react to new claim, obligation, offer, threat, or violated plan using deterministic heuristics.  
Heavy path: invoke an LLM or bounded search for a novel social strategy, plan decomposition or expressive dialogue **only when expected value warrants cost**.

Inferences can propose *new strategies composed of allowed primitives*; they cannot instantiate new physics, secrets, resources, or rights. The model may output `ask for evidence` or `seek witness`, while the behavior engine executes the validated plan.

---

## 5. Micro to meso: stable interaction structures

### 5.1 Dyadic relationships

Repeated events between A and B produce a directed relationship history. Repeated help may yield friendship; help followed by exposure of a deception may instead yield resentment. Trust is context-specific, and a past promise can affect later cooperation even if it is not mentioned in dialogue.

### 5.2 Negotiation and contracts

A contract should be a finite-state process, not a free-text quest marker:

```
Proposed -> Accepted -> Active -> ConditionMet -> PerformanceDue
            |            |                    |
            v            v                    v
          Rejected      Canceled           Performed / Violated / Excused
```

Represent conditions as typed predicates with owner, trigger, deadline, evidence of fulfillment, allowed cancellation, and exactly-once resolution. **Condition met** is not **benefit delivered**. If the grocer is owed medicine, receiving the medicine may activate a promise to share a ledger; separate those two state transitions. A false claim of fulfillment has no effect until evidence or world predicates validate it. Partial delivery or interruption needs defined semantics.

### 5.3 Knowledge network

Let `G_K` have directed edges reflecting the opportunity and likelihood to communicate. Each message traverses a perceptual/social edge, carrying a source chain. Rumor reach emerges from opportunities, ties, urgency, incentives and attention. Prevent universal knowledge broadcasts except for actual broadcast channels (announcement, newspaper, audible speech in range).

### 5.4 Resources and work

Resource transitions generate needs and feasible plans. Production consumes ingredients, labor and time. Costs and scarcities cause shortages; shortages motivate offers and requests; those cause debts or competing faction decisions. Money and physical inventory must be conserved modulo recorded sources/sinks.

### 5.5 Group coordination

A faction is an institution with assets, officers, norms, policies, membership, internal disagreements, and power to propose coordinated plans. Its actions are carried out by real members, consuming real resources and exposing them to observation. Group beliefs/decisions may be aggregates of communicated member reports, not arbitrary access to the world's truth.

---

## 6. Macro model: an emergent society

The macro-world is not a separate AI that decides what happens. It is the observed aggregate of many local decisions plus explicit institutional/environment rules:

\[
W_{t+1}=T(W_t,\{a_i\}_{i\in Active_t},\xi_t)
\]

Each actor's behavior depends on partial observations:

\[
a_i \sim \pi_i(\hat W_i,G_i,J_i,R_i,Mem_i)
\]

Events and resources make agents causally coupled; interactions change which actions become possible for others. Use a time-indexed **causal multigraph** with actors, places, items, facts, contracts, institutions, and events as nodes. Edges represent sight, custody, membership, source attribution, obligations, enmity, employment, transport, and influence.

### 6.1 Macro emergent structures

| Meso phenomena | Emerges from micro operations | Macro manifestation |
|---|---|---|
| Friendship / rivalry | recurring aid, insult, promises, appraisal | alliances, feuds |
| Gossip / public opinion | attributed assertions, repeated retelling | reputation cascades, panic |
| Trade / shortage | supply transfers, work, needs, prices | market prices, bankruptcies |
| Crime / policing | theft, evidence, reporting, authority acts | criminal networks, enforcement changes |
| Politics | interests, coalition offers, memberships, votes | leadership changes, faction splits |
| Social norms | observed behavior, judgments, sanction/reward | compliance patterns, stigma |
| Institutional collapse | unmet duties, missing resources, defections | strikes, breakdown, reform |
| Conflict | grievance, capabilities, alliances, strategic plans | raids, wars, retaliations |
| Migration | perceived safety/opportunity, schedules, resources | neighborhood changes |
| Quests | goal + obstacle + potential agreement/action | emergent mission chains |

Avoid assuming macro changes are guaranteed. This table lists *mechanisms that may produce them*, depending on scenario parameters and agent behavior.

### 6.2 Quests as goal/obstacle graphs, not scripts

A quest is a machine-readable causal explanation:

\[
Q=(beneficiary,goal,blocking\ predicates,achievable\ pathways,stake,contract?)
\]

Example: grocer wants medicine, cannot access supply, player can obtain it. The director **recognizes** `deliver medicine` as a quest opportunity and may surface it through dialogue. No generated task is real until its underlying goal, guards, and accepted commitments exist. When the player finds an unexpected substitute, evaluate whether that substitute satisfies the intended state predicate. Do not require one authored solution.

### 6.3 Narrative director has constrained power

The director can select among **uncommitted, plausible** opportunities and adapt presentation to player interest. It may not change committed truth, invent witnessed events, teleport resources, or turn repeatedly investigated innocents into culprits. It can propose an encounter, clue carrier, new role, or faction initiative; those are checked against causal prerequisites and authored anchors.

### 6.4 Stability, feedback and complexity control

Unconstrained feedback causes absurd outcomes: everyone gossips forever, revenge escalates indefinitely, a rich actor monopolizes everything, or NPCs ping-pong between goals. The model needs bounded attention, cooldowns, forgetting of *salience* (not erasure of authoritative events), satiation, diminishing utility, friction, recovery, institutional constraints, hysteresis, limited travel, finite resources, and liveness/progress policies. Avoid global narrative resets as a primary stabilizer.

---

## 7. Worked causal chain: from one question to a town conflict

1. **Question**: player asks grocer who altered a shipment ledger (`QUERY`). Grocer is aware but has secrecy incentive. No knowledge is transferred simply because a question is asked.
2. **Counteroffer**: grocer requests medicine for family (`OFFER` + potential `COMMIT`), because shortage drives an existing goal and the player could help.
3. **Player agrees**: a conditional agreement exists (`ACCEPT`). Still no medicine and no disclosure. Events register intention and terms.
4. **Player acquires medicine**: inventory transfer/sale may affect supplier stocks and prices, possibly observed by a courier.
5. **Delivery**: grocer physically receives item (`TRANSFER`); condition becomes satisfied. Obligation to disclose becomes due. A promise can be violated; outcomes remain recorded.
6. **Disclosure**: grocer asserts a ledger claim (`ASSERT`). Player obtains attributed evidence; player may believe or doubt. The grocer has not necessarily told objective truth.
7. **Spy overhears**: spy's perception depends on exact location, occlusion, loudness and timing. If heard, the spy receives a partial utterance/evidence event, not the global transcript.
8. **Spy warns faction**: faction receives a claim with source chain. A guard may distrust it. If the faction judges risk high, it can allocate resources to hide records or intimidate the grocer.
9. **Threat**: grocer fears reprisals and may request protection. New objective `protect grocer` is mechanically derivable, not manually inserted.
10. **Response**: player recruits a guard, who weighs duty, evidence, corruption, and risk. Failed recruitment could prompt a complaint; successful recruitment costs patrol labor.
11. **Macro outcome**: labor shifts from the market to security; rumors alter trust; a faction diverts funds; an innocent NPC may become a key witness. Several plausible quests arise because the world has been causally affected.

The story is compelling **because every step is mechanically explainable**. If the spy was out of range, step 7 never occurs, and a different story follows.

---

## 8. The macro execution algorithm

Use event-driven, multi-rate planning, not an LLM query per actor per frame:

```
step_world(tick, external_proposals):
    advance_physics_and_clock()
    apply_due_environmental_events()
    collect_player_and_external_proposals()

    # deterministic ordering and budgeted processing
    for proposal in ordered_proposals:
        result = authoritative_sim.validate_then_apply(proposal)
        append_events(result)

    observations = perception.route_new_events()
    for (npc, observed_events) in observations:
        update_evidence_and_memory(npc, observed_events)
        appraise_events(npc, observed_events)
        update_relationships_and_goal_urgencies(npc)
        schedule_reconsideration(npc, priority_for(observed_events))

    process_commitment_deadlines_and_faction_accounts()

    for npc in pick_active_agents_with_budget():
        if current_plan_still_valid_and_appropriate(npc):
            continue_plan(npc)
        else:
            candidates = generate_feasible_local_plans(npc)
            plan = choose_subjectively_best_bounded_plan(npc, candidates)
            submit_next_typed_action(plan)

    if director_has_relevant_opportunity():
        submit_director_proposal_to_narrative_validator()

    commit_event_log_and_snapshot_if_due()
```

**Deterministic conflict resolution:** order competing transfers, object edits and shared-resource claims by recorded tick, priority class and stable sequence. Consider optimistic validation against current state with explicit rejection and replan, rather than assuming simultaneous actions are independent. If you later add true concurrency, use atomic transactions and rigorous conflict properties.

**Multi-rate scheduling:**

- Frame/physics: player movement, collision, audiovisual client (no LLM dependency).
- Simulation fixed step: authoritative actions, action progress, events.
- Event-triggered micro-reactions: new claim, threat, deadline, witnessed damage.
- NPC minutes: routines, needs, memory consolidation, plan revision.
- Faction hours/days: staffing, finances, supplies, policy deliberation.
- Director on consequential changes: narrative opportunity checks.

For unloaded areas, derive bounded catch-up events from schedules and ledgers, then expand around the player. Preserve causal equivalence where it matters (debts, deaths, clues, resources); low-salience incidental animation need not be simulated exactly.

---

## 9. Formal verification program

Theorem provers cannot establish that agents behave 'like people' without a formal specification of that phrase. They *can* establish invariants of the executable world and proofs about a mathematical policy core.

### 9.1 Safety properties

1. **Conservation:** item and money quantities cannot become negative; transfers do not create resources unintentionally.
2. **Disclosure integrity:** standard factual disclosure requires a valid knowledge basis; intentional unauthorized leaks or lies must be explicit actions with their own possible consequences, not accidental prompt leakage.
3. **Claim separation:** receiving a claim does not imply believing it. No event silently writes world truth from testimony.
4. **Perception boundaries:** absent a valid sensory/communication route, no observed-event record is delivered.
5. **Authoritative action:** all state changes route through typed validated transitions.
6. **Commitment integrity:** an accepted obligation cannot become `Performed` without the contracted performance predicate; satisfaction and performance are distinct.
7. **No retroactive narrative contradiction:** committed facts or witnessed evidence are never silently mutated by the director.
8. **Rights/roles:** delegated action authority is scoped and cannot grant arbitrary capabilities.
9. **Temporal consistency:** a future event cannot cause an earlier event unless explicitly modeled as retrospective evidence.
10. **No dangling references:** events and contracts reference actors/objects/facts that exist or have immutable historical tombstones.
11. **Deterministic replay:** same complete initial state, seed, ordered input, code version and model proposal sequence produce identical events.
12. **Information noninterference:** changing an unreachable secret does not change a given actor's observable actions under a fixed random and policy state, *provided all other inputs are identical*.

### 9.2 Liveness and gameplay properties (assumptions required)

- **No permanent task deadlock:** every critical narrative goal retains some achievable lead under stated resource/actor assumptions or declares a clear failure state.
- **Eventual resolution:** accepted, due promises are eventually performed, failed, renegotiated or explicitly abandoned under a fair scheduler. This does **not** guarantee NPCs honor promises.
- **Offscreen autonomy:** world continues to advance under no player input.
- **Budget boundedness:** a bounded number of inference calls and planner nodes per world step.
- **Narrative access:** critical revelations retain multiple independent routes while scenario assumptions remain satisfied.

These require conditions about scheduling fairness, world reachability and time; do not claim absolute liveness without them.

### 9.3 Suggested verification layers

- **Ordinary Rust unit/integration tests:** every primitive effect, semantic boundary, failure mode and regression.
- **Property-based testing:** generate arbitrary bounded actors, actions, contracts, inventories and conflicting evidence; check postconditions.
- **Kani:** exhaustively model-check finite harnesses around transfer/accounting, claim acceptance boundaries, contract lifecycle and action validation.
- **Verus:** verify compact pure Rust transition functions and specification contracts with explicit pre/postconditions.
- **Lean 4:** prove abstract algebraic statements about composition, invariant preservation, epistemic separation or contract progression. Extract executable test vectors/spec traces or maintain a carefully reviewed correspondence to Rust semantics; a Lean proof of a separate model alone does not verify Rust.
- **Model checking / TLA+:** explore asynchronous scheduling, competing transfers, simultaneous deadlines, race conditions if concurrent workers are introduced.
- **Statistical playtests:** validate believability, diversity, causal intelligibility, and enjoyment. These are empirical properties, not pure mathematical theorems.

### 9.4 Inductive proof strategy

Given invariant `Inv(W)`:

\[
Inv(W_0) \wedge \forall W,a:\ Inv(W)\land Valid(W,a)\Rightarrow Inv(Apply(W,a))
\]

prove that all event-reachable states preserve `Inv`. Each new action type must either satisfy its own preservation lemma or only use already-verified primitive effects. This is the core benefit of compositional construction.

A critical caveat: if the implementation bypasses verified primitives, proofs no longer cover that path. Require a single authoritative mutation interface, auditing and a CI gate.

---

## 10. Mapping the proposal onto the current repository

Observed current design from repository `README.md` and selected Rust files (not a complete security/code audit):

| Current component | Already present | Proposed evolution |
|---|---|---|
| `domain` | `Action`, `QuestPredicate`, `World`, `Belief`, `Goal`, `Commitment`, typed events | richer speech acts, evidence ledger, interpreted intent, context-typed predicates, action effect contracts |
| `perception` | sight/hearing, addressees, walls, overhearers | attention/uncertainty, observation channels, source correlation and delayed evidence |
| `social` | directed relationship dimensions, `share_claim`, commitments, rumors | appraisal model, multievidence inference, contextual trust, norm/obligation outcomes |
| `agents` | prompt scoping, heuristic fallback, policies, priority scheduler | bounded candidate plan generation, ongoing intention state, utility ranking, escalation on novel situations |
| `sim` | authoritative validated transitions, seeded replay, event logging | verified atomic effects, action conflicts, variable-duration effects, evidence-to-mind dispatch |
| `narrative` | anchors, validated clue placement and role-promotion proposals | quest recognizer (goals + obstacles + valid pathways), opportunity ranker with causal guard checks |
| `persistence` | save/load + event snapshots | schema migration for evidence and intention ledgers; event/version invariants |
| `eval` | deterministic 10-scenario gate | generated microproperty tests, causal traces, world-scale emergence measures |

Specific current risk/extension: `crates/social/src/lib.rs` has a `share_claim` heuristic that samples belief acceptance using trust, suspicion, speaker confidence and assertiveness; `record_belief` writes one `Belief` per `FactId`. This works as a small prototype but cannot fully represent multiple contradictory reports or their correlated origins. Extend this first while preserving existing tests and content compatibility.

**Migration principle:** do not replace `Action` all at once. Add a new pure `behavior`/`cognition` module or crate with an adapter to existing actions. Migrate one interaction family at a time. Keep original scenario runner and deterministic fallback as acceptance gates.

---

## 11. Build sequence with measurable acceptance

### Phase 0 — Instrumentation and shared schema

Define typed `Intent`, `SpeechAct`, `Evidence`, `Effect`, `ActionOutcome`, `GoalPredicate`, and `CausalTrace`. Add event provenance and serializer versioning. Benchmark baseline time per step, decisions per simulated minute, replay stability and current 10 scenario cases. **Deliverable:** inspectable causal traces: *NPC did X because of goal Y, belief Z, observed event E, and cost C.*

### Phase 1 — Epistemic microcore

Implement multi-evidence belief storage with source chains and conflicting evidence. Make asking, asserting, reporting, deception, overhearing, questioning, and refusal use shared mechanics. Test no omniscience, no automatic acceptance, correlation protection and replay. **Deliverable:** a liar can fool one NPC while a skeptic requests additional evidence, without special-case scripts.

### Phase 2 — Goals, intentions and commitments

Introduce persistent intentions, utility-based option comparison and typed conditional contracts with deadlines. Generate a favor request from a goal/resource obstacle rather than checking a specific NPC name. **Deliverable:** remove hardcoded grocer-specific bargain decision *from the new pathway*, while still passing grocer scenario and preserving existing gameplay.

### Phase 3 — Social graph

Expand appraisals, directed contextual trust, needs, memory salience, and friend/foe response; avoid universal relationship deltas. **Deliverable:** reputation changes only through plausible chains; NPCs can forgive, reject or retaliate according to state.

### Phase 4 — Autonomous world

Introduce bounded event-driven plans, scarcity accounting, factions as institutions and offscreen catch-up. **Deliverable:** factions and NPCs respond to events without player interaction or continual LLM inference.

### Phase 5 — Emergent quest and story recognition

Recognize solvable obstacles and meaningful contracts; reuse existing narrative anchor validation. **Deliverable:** a player-created problem or NPC's independent goal can produce a coherent optional quest without a scripted quest giver.

### Phase 6 — Formal proof coverage

Verify physically small kernels first: asset transfer, belief-observation separation, commitment lifecycle, authoritative replay. Each added primitive must have at least one invariant preserving proof or model-checking harness. **Deliverable:** CI rejects changes that violate specified invariants, with bounded model-check limits transparently documented.

### Suggested metrics (targets to calibrate, not current measurements)

- 100% of authoritative world mutations have typed, event-logged transitions.
- 100% of NPC claims have provenance and supported access paths (including explicitly intentional lies).
- 0 detected omniscient disclosures in generated adversarial scenarios.
- 0 stale/invalid contract completions in bounded model-check harnesses.
- 100% deterministic replay for identical recorded inputs and versions.
- Per-event NPC planning budget and LLM request caps are configurable and enforced.
- Qualitative evaluation: diverse but comprehensible responses, consequences tied to observable causes, no railroading, no uncontrolled rumor/retaliation loops.

---

## 12. Failure modes and design guardrails

1. **Fake universality:** no finite ontology guarantees capture of every real-life interaction. Provide a typed extension mechanism and unknown-intent behavior.
2. **Omniscient planners:** keep subjective world state separate from authoritative truth.
3. **A single social score:** keep dimensions, context, provenance and history.
4. **Utility maximizers all act alike:** vary goals, values, information, risk, tolerance, attention and imperfection, and test diversity.
5. **Beliefs updated from exact truth:** compute belief from accessible evidence, not author knowledge.
6. **Unbounded planner explosion:** limit local neighborhood, horizon, action templates, event budgets and inference. Prefer routine continuation.
7. **NPC commitments overwritten by dialogue:** agreements survive rendering; speech is not authoritative fulfillment.
8. **Fluent but consequentially false language:** output must be grounded in decided action and current view; never let speech fabricate committed world events.
9. **Director as puppeteer:** restrict its writes to validated uncommitted opportunities.
10. **Oververification:** prove invariants in the small deterministic kernels and behavior data transformations; use statistical playtesting to validate human believability.
11. **Unstable macro feedback:** cap influence, add credible friction, heterogeneity, decay, finite resources and institutional mediation.
12. **Loss of simulation determinism via floats/races:** stable update order and recorded RNG samples; fixed-point for money and consider for probability calibration if exact cross-platform reproducibility is required.

---

## 13. The first research experiment

Create a sealed three-person micro-world: **player, grocer, guard**, with optional spy. Give each actor different belief access, values, economic needs and trust parameters. Generate hundreds of seeded variations of:

- asking for a ledger;
- offering medicine, money, protection or misinformation;
- withholding or delivering a promise;
- asserting conflicting evidence;
- broadcasting versus private speech;
- observing or missing an event;
- threatening or helping an NPC;
- guard duty and possible corruption.

Success means the engine produces *different but causally explainable* behaviors from initial conditions; all hard invariants hold; a run can replay exactly; and the outcome doesn't rely on `if actor_id == "grocer"` or hand-written utterance keywords. Preserve the original game content as one acceptance scenario rather than making it the algorithm.

**Central hypothesis:** a sufficiently expressive, formally constrained interaction vocabulary plus situated, bounded decision-making can generate credible open-ended RPG stories without scripting every quest or requiring an LLM to determine world truth.

This hypothesis is testable, not a theorem. Test it in the micro-world, then expand toward social networks and institutions only after the microcausality is solid.

---

## References and grounding

- IRL source and design documents: https://github.com/JPrier/IRL ; https://github.com/JPrier/IRL/blob/main/docs/design/agentic_rpg_complete_design.md
- Rao & Georgeff, *The Semantics of Intention Maintenance for Rational Agents* (IJCAI 1995): https://ijcai.org/proceedings/1995/92
- BDI Agent Architectures survey (IJCAI 2020): https://ijcai.org/proceedings/2020/684
- BDI agents and discrete-event simulation (2026): https://link.springer.com/article/10.1007/S10458-026-09744-W
- Decentralized POMDPs and macro-actions: https://pmc.ncbi.nlm.nih.gov/articles/PMC6814204/
- Kani Rust verifier: https://model-checking.github.io/kani/
- Verus reference: https://verus-lang.github.io/verus/guide/overview.html
