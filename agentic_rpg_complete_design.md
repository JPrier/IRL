# Agentic RPG — Complete Game & Technical Design

**Status:** Consolidated concept / pre-production design, 20 October 2026  
**Working name:** *Agentic RPG* (not a final title)  
**Scope:** Game experience, narrative design, autonomous NPCs, information systems, social simulation, faction economics, local-LLM architecture, and prototype evaluation.

> **Central premise:** Build a role-playing game in which players can speak naturally to nearly anyone, pursue goals that were not manually scripted, and meaningfully influence the story. The world is finite and mechanically coherent, but the *paths* through it—and which people matter—can be enormously varied. NPCs pursue their own interests; a separate director improvises credible narrative connections within immutable constraints.

## 1. Origin, intent, and the player fantasy

The originating idea is the feeling of playing a tabletop RPG with a responsive game master: the player says something unexpected, and the world has a meaningful answer instead of a canned line or invisible wall. The design is not “an LLM chatbot placed inside an RPG.” It is a **simulated world with dialogue as a first-class action interface**.

The goal is a world in which the player can:

- Talk to NPCs in open-ended language and receive responses that reflect personality, knowledge, fear, desire, and circumstances.
- Make promises, exchange favors, negotiate, persuade, threaten, bribe, deceive, comfort, or betray; those conversations change world state.
- Discover or create **quests organically**. An informant may demand an object before revealing a secret. A frightened witness might require protection. A player may find evidence to blackmail an influential person. These are quests even if no formal quest-giver or quest marker exists.
- Form relationships, recruit allies, build crews, provoke enemies, or manipulate social networks.
- Investigate mysteries through multiple routes. Clues and key people can become relevant through natural interaction, without forcing everyone down one authored corridor.
- Observe a living city with routines, crimes, damage, rumors, changing businesses, deaths, repairs, and economic pressure—without requiring the player to be the universe's center.
- Experience a main story with enduring stakes and meaningful revelations, but not a rigid chain of prescribed interactions.

**The sensation to pursue is agency, not literal infinity.** A relatively compact world can feel expansive when its characters, relationships, information, locations, and player strategies combine in unforeseen ways.

## 2. Design pillars and non-goals

1. **People first, plot roles second.** NPCs begin with ordinary needs, cultures, loyalties, knowledge, and lives. Sustained player interaction can make an initially minor character important.
2. **Few strong anchors; many loose checkpoints.** The mystery's premise, central conflict, and high-level antagonistic goals remain stable, while discoverable details, delivery mechanisms, and secondary roles can be improvised.
3. **Conversations cause actions.** Speech can establish commitments and obligations, reveal information, alter emotion, change plans, or trigger simulated events.
4. **Consequences emerge from systems.** Costs, limited resources, perception, social ties, schedules, and information flows create credible reactions; no single global “good/evil” or honor meter determines them.
5. **World autonomy without stealing the story.** Everyday simulation proceeds independently. The core plot should generally not conclude offscreen while the player is doing something else; major narrative advances have explicit triggers or configurable time-pressure rules.
6. **Player creativity is respected but world truth is protected.** The director may exploit uncommitted possibilities; it cannot erase discovered facts, grant arbitrary abilities, or always validate the player's suspicions.
7. **Local-first, bounded inference.** Hundreds of NPCs are not hundreds of continuously running LLMs. Most behavior is deterministic; small shared models reason only when necessary.
8. **Observability and testability.** Perception, memories, decisions, director changes, costs, and generated outcomes should be inspectable and replayable.

**Non-goals:** infinite hand-authored content; unrestricted model control over the simulation; every NPC participating in every plot; fully modeled national economies; perfectly human psychology; mandatory real-time inference for all residents; or a game that cannot proceed unless the player follows a glowing quest marker.

## 3. The moment-to-moment game loop

The player explores a 2D or eventually 3D space, watches NPCs carry out routines, talks through voice or text, examines evidence, and takes physical and social actions. A typical sequence:

1. **Observe:** notice the grocer meeting a courier, a guard listening nearby, or a damaged public building.
2. **Approach:** choose a person or location; communicate by voice, typed dialogue, or conventional interaction.
3. **Interpret:** the input system determines who was addressed, who overheard, what the player seems to intend, and whether any actionable commitments or requests were made.
4. **Decide:** the addressed NPC evaluates its goals, beliefs, emotional state, and available actions; the world engine checks consequences and feasibility.
5. **Respond:** NPC dialogue, animation, movement, trade, rumor transmission, or a concrete task emerges.
6. **Propagate:** relevant NPCs, groups, and the story director learn only what they could plausibly observe or discover.
7. **Persist:** commitments, facts, beliefs, events, and world changes survive across sessions and influence later encounters.

Exploration, investigation, trading, stealth, combat, travel, and conventional RPG progression can coexist with this loop. Their exact feature sets are undecided and should not become prerequisites for the first prototype.

## 4. Narrative architecture: anchors, checkpoints, and improvisation

### 4.1 Three levels of story certainty

| Layer | Meaning | May director change it? |
| --- | --- | --- |
| **Immutable anchors** | Fundamental premise, established events, setting rules, the existence of a central faction, its broad goals, player-observed facts | **No**, absent an explicitly designed in-world mechanism |
| **Commit-on-discovery checkpoints** | Undiscovered identity, relationship, evidence placement, secondary culprit, route to a truth, optional betrayal | **Sometimes**, until evidence or dependent consequences commit it |
| **Emergent narrative** | Promises, revenge, temporary alliances, accidental disputes, side stories, incidental deaths, local disasters, NPC responses | Evolves through validated game events |

The author should define the few truths that make the story *about something*. The improvisational systems construct plausible routes between them. A checkpoint becomes a hard fact once it is meaningfully established through player observation, testimony with provenance, or an irreversible event.

### 4.2 Hidden Mafia — the notebook's main example

**Immutable anchors:** an underground mafia exists; it has broad objectives and a hierarchy; the player can discover the organization and attempt to uncover its leadership.

**Flexible checkpoints:** the public identity of the leader might not be fixed at start, if the game has deliberately left it undecided. Which intermediaries expose the network, which records provide evidence, and which character becomes a pivotal informant can vary. A satisfying reveal or twist is desirable but never at the expense of consistency.

**Hierarchical disclosure:** people closer to the mafia's leadership generally have stronger incentives, training, risks, and loyalties that make information harder to extract. This is not merely a numeric “higher level = harder skill check”; it means the player may need better evidence, increased trust, leverage, protection, or proof that disclosure serves the NPC's goals.

**Two sound implementations:** (a) set the leader in the world seed to support true deduction; or (b) maintain a constrained set of candidate leaders and commit one before evidence pointing to a unique person is generated. Never switch the culprit after a player has established incompatible facts.

### 4.3 Story checkpoints as information gates

The notes propose that a **threshold of information**—gained through conversation, reading, visiting locations, witnessing events, or other exploration—unlocks additional information that can be placed in **not-yet-investigated** spaces.

Implementation:

- Represent important discoveries as semantic facts and prerequisites, not page-specific cutscenes.
- Track which facts the player possesses, their confidence/source, and which areas/objects/characters have been inspected.
- When conditions are met, activate an eligible clue package: a document, object, overheard exchange, rumor, scene, courier route, or other exposure.
- Prefer placements in unrevealed and credible places; preserve already examined spaces unless there is an explicit new delivery or world-change event.
- Record the chosen placement and evidence so it cannot migrate again once established.
- Ensure multiple independent ways into a critical revelation (a useful tabletop guideline is **at least three viable leads**).

This system avoids both rigid bottlenecks and blatantly teleporting evidence to follow the player.

### 4.4 Information carriers (“information gatherers” in the notes)

A story fact should describe **what it means** separately from **how it may be delivered**.

Each information entry should have: canonical statement, certainty state, reveal conditions, eligible carrier formats, relevant social groups, plausible places, accessibility/security, witnesses who know or could know it, consequences of disclosure, and fallback exposure paths.

Carrier options include speech, letters, ledgers, signs, overheard conversations, books, inventories, physical traces, incidents, conversations between NPCs, and observed actions. People directly associated with the knowledge should be prioritized as sources; unrelated NPCs cannot disclose omniscient facts.

### 4.5 NPC promotion and floating roles

Mark's tabletop point becomes a formal system: an NPC repeatedly sought out by the player can be considered for *story roles* such as informant, organizer, rival, patron, intermediary, witness, victim, or potential betrayer. A candidate must fit existing history, group memberships, access, and motivations. Role promotion increases involvement; it does not arbitrarily rewrite personality or secret knowledge.

The director may attach the next valid opportunity to the character the player is already invested in—rather than demanding the player meet an entirely different predetermined quest-giver. Conversely, an innocent character can remain innocent; engaging with them can still have consequences.

## 5. NPC identity, culture, and personal agency

### 5.1 Persistent NPC model

Each NPC should have:

- **Identity and backstory:** occupation, home, family, personality traits, personal history, boundaries.
- **Overlapping cultural/social tags:** religion, guild, clan, class, subculture, fandom, politics, philosophy, family, profession, neighborhood. Tags can conflict; membership does not determine all behavior.
- **Goals and needs:** survival, affection, safety, wealth, status, duty, ideology, ambition, protection of specific people.
- **Resources and capabilities:** possessions, time, money, skill, access, legal authority, influence, allies.
- **Emotions and relationships:** trust, attachment, fear, respect, resentment, dependence, obligation, suspicion.
- **Beliefs and knowledge:** what they saw, heard, inferred, incorrectly believe, conceal, or deliberately lie about.
- **Plans and routines:** current objective, preferred action, fallback plan, schedule, location.
- **Potential narrative roles:** eligible opportunities, not permanent plot destiny.

### 5.2 Decisions, not scripted performances

An NPC's decision should consider its own objective, available actions, expected benefit, cost, risk, moral constraints, and relationships. Agent-selected tools might allow `move`, `speak`, `offer_trade`, `request_help`, `share_claim`, `hide_evidence`, `warn_contact`, `report_crime`, `schedule_meeting`, `recruit`, `repair`, and `commit_resources`.

A model **proposes** an action. Deterministic game systems determine whether the NPC is able to perform it, what it costs, what others notice, and how the world changes. NPCs may develop hidden goals, independently plan betrayals, lure players into traps, or deliberately speak within hearing range of spies.

### 5.3 Overlapping group loyalties

An NPC can simultaneously love family, need a merchant guild, distrust police, and practice the religion associated with the cult. Conflicting membership creates nuanced dialogue, motives, alliances, and compromises. It also makes moving narrative roles between NPCs more plausible.

### 5.4 Emotion and social influence

The player's actions can elicit temptation, desire, anger, annoyance, fear, sadness, comfort, happiness, guilt, grief, loyalty, and concern. These states change choices, disclosure, cooperation, and willingness to take risks. They should not be interchangeable persuasion buttons.

Particular ideas from the notes:

- **Temptation:** offer something an NPC actually wants in exchange for a concession; a requested item or service can become a quest.
- **Anger/annoyance:** provoke impatience, hostility, impulsivity, or refusal—depending on the person.
- **Fear:** intimidation may yield short-term compliance while building resentment, escape plans, or reports to authority.
- **Sadness/comfort:** making an NPC unhappy and consoling them may influence their response. If they do **not know** the player caused their hardship, their reaction should differ from a case where they know or suspect it. This supports manipulation as an in-world tactic, with eventual discovery and consequences.
- **Empathy and loyalty:** sustained assistance and common cause can build crews and allies willing to take extraordinary risks or defend the player. Loyalty should arise naturally, with plausible limits rather than universal mind control.

The goal is closer to believable individual cause-and-effect than a single global **Red Dead Redemption 2-style honor indicator**. Social standing may still emerge as derived public reputation, but it should be distinct from each NPC's private relationship and knowledge.

## 6. Organic quests, bargaining, coercion, and player-created objectives

**This is a quest-driven RPG, but quests need not be handed out as scripted cards.** People want things. They may refuse to cooperate until their needs are met. A player who needs testimony might be asked to retrieve medicine, stop a threat, pay a debt, escort a relative, or expose a rival. The player can accept, bargain, lie, find another route, or force the issue.

Objective structures should be generated from validated world entities:

`NPC desire → obstacle → proposed bargain → player commitment → trackable world condition → NPC response`.

Example: A witness withholds the mafia ledger because their sibling is threatened. The player can protect the sibling, bribe a guard, uncover compromising evidence to blackmail the witness, steal the ledger, secure alternative evidence, or persuade an ally to intervene. These are materially different strategies with different relationship effects.

Quest completion should come from authoritative state predicates—*the sibling is safe*, *the record changed hands*, *the threat was neutralized*—not the LLM declaring that a vague conversation “completed the quest.” Formal quest logs may be optional; obligations must nonetheless be persistently trackable.

## 7. Surveillance, spies, rumors, and privacy

The notebook specifies **spies for different groups**, ranging from citizens reporting crime to informants actively seeking useful intelligence.

Design requirements:

- Residents can observe crime and choose whether/how to report it.
- Spies can listen to player and NPC conversations, track people, gather useful facts, and relay partial intelligence to employers.
- Players can notice eavesdroppers and adapt: watch for people sneaking around, change plans, spread misinformation, or ask someone to talk privately.
- When asked to move somewhere quiet, NPCs choose plausible nearby areas, **preferably a place they control or trust**, while considering danger and privacy.
- Conversation privacy depends on location, hearing distance, walls/occlusion, ambient sound, attention, and who was physically present—not whether a scene is marked “secret.”
- NPCs can deliberately speak near spies, bait the player into a trap, leak rumors, or deceive an ally.
- Information transmission is an event: a recipient must acquire it by witnessing, hearing, receiving a message, or being told by a plausible source.

**Truth must be separate from claims.** A rumor can be false; a sincere witness can be mistaken; an informant can distort a report. Track source, timestamp, confidence, and dissemination history to prevent universal NPC knowledge.

## 8. Perception and conversation routing

The player may speak naturally instead of using an explicit dialogue menu.

The system first detects whether speech is directed at an NPC, at several people, or at nobody in particular. Inputs include player gaze, facing, proximity, current conversation participant, names or pronouns used, and optional explicit targeting. When ambiguous, the game should ask or provide a lightweight target-selection cue rather than silently routing speech incorrectly.

If a specific NPC is addressed, its agent receives the utterance and nearby context. Other NPCs in hearing range receive **perception events**, not the whole global transcript. Nearby characters should react to player behavior **and other NPC behavior** when they have a reason to notice it.

A spatial perception system should model line of sight, audible distance, occlusion, attention, noise, facing, and event salience. In the prototype, visualize what each character could see and hear so perception errors are diagnosable.

**Audio path:** microphone → optional voice activity detection → local speech-to-text → addressee/intention inference → NPC decision and validated action → text response → local text-to-speech → synchronized playback/animation. Typed dialogue should also work. TTS may stream the spoken response, but gameplay state must follow the authoritative decision rather than a generated utterance.

## 9. A living world without an autoplayed main quest

Two ideas must coexist:

1. The world should be **dynamic without the player**. People work, spend money, pursue goals, commit crimes, fall ill, die, move, start and close businesses, damage buildings, and repair infrastructure.
2. The **core narrative should not meaningfully progress or resolve without player interaction** by default. The player should not discover that the central mystery was solved entirely offscreen because they spent an hour conversing with a shopkeeper.

Use two advancement lanes:

- **Autonomous simulation lane:** bounded everyday change, NPC routines, economic events, rumors, localized crises, repairs, faction preparation, ambient consequences.
- **Narrative advancement lane:** major reveals, irreversible antagonist escalations, endgame transitions, and mystery resolution gated on player action, significant discoveries, or explicit chapter/act progression.

A separate optional *time-pressure campaign mode* may allow main-plot clocks to advance independently, provided the player is warned and the story remains meaningfully playable. The earlier example of a seven-day crisis is one **optional scenario**, not an assumed rule for this project.

NPCs should not behave as if the player is uniquely important unless the game's fiction establishes that—for instance, a story about a chosen hero (the notebook references Doomguy / a Halo Spartan / the Dragonborn). Most people have other priorities.

## 10. Autonomous incidents and reactive repairs

The world needs a library of **loose event types**, not only plot-specific scripts: crime, theft, rumor, illness, death, bankruptcy, fire, property damage, business openings/closures, changes in neighborhood activity, staffing shortages, migration, construction, and repairs.

If a player destroys a structure, the owner or responsible authority should evaluate repairing it. Outcomes depend on **money, time, urgency, available workers/materials, and competing priorities**. Sometimes it gets fixed quickly; sometimes it stays damaged; sometimes the owner goes bankrupt. No universal “damaged object respawns after X minutes” rule is necessary.

Background simulation should use event-driven updates and coarse aggregates for distant areas, reserving detailed NPC-level logic for nearby or narratively significant interactions.

## 11. Factions and full, bounded economies

Groups—governments, guilds, businesses, gangs, cults, clans—need resource constraints and priorities.

A government should not repair a single telephone pole while its most important infrastructure remains destroyed unless an explicit reason justifies that decision. Each group chooses actions against its capacity:

- Money, supplies, staff, expertise, time, authority, public legitimacy, and access.
- Competing goals and priorities.
- Direct action cost, operating/maintenance cost, opportunity cost, and **cost of doing nothing**.
- Effects of delay, escalating threats, and commitments already made.

This enables strategic consequences. Players can impose recurring expenses on an organization, disrupt supplies, and potentially bankrupt it; the organization can retaliate, reallocate funds, ignore low-priority failures, recruit support, or collapse.

**“Full economy” should mean economically accountable behavior, not simulating every individual commodity transaction at national scale.** For performance, use budget envelopes, inventories at useful granularity, periodic accounting, priority queues, and causal events. Economic pressure must affect visible game behavior.

Story-critical factions can have special design constraints preventing nonsensical early collapse, but those constraints should be expressed as credible resources, fallback plans, or explicit scenario rules—not invisible infinite money.

## 12. The AI director (DM agent)

The notes sketch **DM Agent → Story Controller**, alongside **NPC Agent → NPC Controller** and a shared **LLM**, with a TTS model for speaking. This is a useful responsibility split.

The **Story Controller** is the deterministic owner of checkpoints, anchors, eligibility, commitments, available narrative actions, and progression rules. The **DM agent** is a proposal generator and exception handler: it notices missing connections, player interest, emergent tensions, and possible contradictions, then suggests coherent changes. The controller validates or refuses those changes.

Key director operations:

- Evaluate player-investment signals and open narrative opportunities.
- Attach a flexible clue or story role to an eligible character/location.
- Propose a new plausible intermediary or side consequence.
- Find ways to route around inaccessible or deadlocked quest paths.
- Detect contradictions before state mutation; repair **uncommitted** missing details, not established facts.
- Decide when to present a new narrative beat under the game's pacing rules.
- Avoid forcing the player back onto a specific plot route when a valid alternate path exists.

The director should **not** micromanage routine NPCs, directly modify canonical world records, impersonate omniscient NPC knowledge, or secretly alter fixed clues to make a player's theory correct.

### 12.1 Grocery store improvisation example

In Mark's example, a cult drives the central plot but the player becomes obsessed with an apparently unrelated grocer.

The director identifies player interest and checks whether a valid cult-related opportunity exists. It might propose that a wholesaler supplying the store has also served the cult, that a courier uses the storeroom, or that the grocer knows a suspicious accountant. The story controller verifies geography, timeline, access, discovery history, and whether any fact would be contradicted. If accepted, the connection is committed and future evidence can manifest through realistic events.

Alternatively, the grocer is truly innocent: the player may offend them, generate gossip, or attract the Watch. Player attention is a reason to consider a story connection, **not a guarantee that the suspicion is true**.

## 13. Technical system architecture

```text
Player (voice / text / controls)
  |
  v
Input & perception router --- spatial audibility / line of sight
  |
  v
Intent + addressee interpreter
  |
  +----> NPC agent task ------> shared local LLM ------> proposed dialogue/action
  |          |                                         |
  |          +--> context & memories                  v
  |                                             NPC controller / rules
  |                                                    |
  +----> observation & event bus <---- validated event--+
  |                     |
  |                     v
  |                 world state / event log / beliefs / economies
  |                     |
  |                     v
  +----> story controller <---- proposal ---- DM/director agent
                         |                    (same/shared or larger LLM)
                         +--> validated story commits

Dialogue text --> local TTS --> spatialized voice / animation
```

**Core principle:** one authoritative game runtime and event log; many *logical* agent contexts; a small number of physical models. The engine can be Godot for the client, with deterministic gameplay systems in GDScript or Rust and a local inference service. SQLite or another embedded database can hold authoritative events and snapshots. Model and runtime choices should remain replaceable.

### 13.1 Tools and state-changing actions

Expose **capabilities**, not raw database write access. Tools should specify required actor, target, location, preconditions, possible outcomes, cost, and visibility. Example tools:

`move_to`, `talk_to`, `request_private_conversation`, `share_claim`, `report_crime`, `offer_trade`, `request_favor`, `make_promise`, `give_item`, `hire`, `investigate`, `hide_object`, `repair`, `allocate_budget`, `send_messenger`, `schedule_meeting`.

The world engine enumerates currently valid affordances when possible. Small models choose from that shortlist; more flexible planning translates an unusual request into valid actions or a longer-term plan.

NPC speech **never itself commits** a state change. A proposed promise, threat, bargain, item transfer, or action becomes a typed event only after validation and, when required, the player's acceptance.

### 13.2 Example event pipeline

```json
{
  "event": "player_utterance",
  "speaker": "player",
  "addressee": "grocer_01",
  "text": "If you show me the ledger, I'll protect your family.",
  "overhearers": ["spy_02"]
}
```

The NPC agent may emit:

```json
{
  "speech": "Get them out of the city first. Then we talk.",
  "proposed_action": {
    "type": "offer_conditional_disclosure",
    "item": "ledger_04",
    "condition": "family_safe"
  }
}
```

The controller verifies the NPC can access the ledger and truly has that motive, then creates a pending offer. The spy only learns the portion of speech it could hear. The player can agree, refuse, or propose something else. No hidden quest completion happens from mere words.

### 13.3 State ownership and information integrity

Maintain separate stores for:

1. **World truth:** what happened and what objectively exists.
2. **Agent beliefs:** what an individual thinks or claims, with provenance/confidence.
3. **Player knowledge:** what the game has actually revealed (and what the player thinks they know, if represented).
4. **Narrative plan:** anchors, flexible checkpoints, role assignments, committed clue placements.
5. **Resources:** items, money, time, group budgets, scheduled obligations.
6. **Event log:** ordered, replayable actions and outcomes.

Every state change should be attributable to a validated action, passive simulation rule, or director proposal accepted by the story controller. Use versioned schemas, deterministic seeds, snapshots, and rollback/replay for development.

## 14. Local models, contexts, scheduling, and voice

### 14.1 Logical agents are not model instances

An NPC agent is a compact identity and memory record plus a policy and action interface, **not** a separately loaded LLM. The same model handles different NPCs by receiving context assembled on demand. The director may use that model or a stronger optional model.

A practical experimental hierarchy:

- **No model:** routines, physics, perception checks, basic economy accounting, fixed rules, ordinary travel.
- **Tiny decision model:** select from valid actions or classify intent; possibly distilled later from a stronger model.
- **Small conversational model:** character dialogue and slightly more open action proposals.
- **Larger/less frequent director model:** multi-step narrative reasoning and conflict repair.

Specific model families and sizes are benchmark candidates, **not locked design requirements**. Start with one local model and swap components only after measuring failures.

### 14.2 Minimal, sufficient NPC context

The notes explicitly require each NPC's context to be **as small as possible while retaining story information, backstory, and contextually retrievable interactions**.

Construct context from stable identity, present observations, active plans, relationship/belief facts, a short recent exchange, and retrieved salient memories. Avoid placing the entire campaign history or all tool definitions into every prompt. Allow the model to query selected memory or capabilities when needed; keep tool descriptions narrow and schema-driven.

Consolidate memories outside the critical conversation path, but keep the full authoritative history for replay and correction. Retrieval should account for recency, relevance, salience, source credibility, and whether the NPC actually experienced the event.

### 14.3 Inference and audio latency

Prioritize requests:

| Priority | Activity | Runtime approach |
| --- | --- | --- |
| P0 | Direct player conversation, immediate response | Short context/output, streaming where safe, deadline + fallback |
| P1 | Nearby NPC response, important local action | Event driven, batch if it reduces latency |
| P2 | Remote NPC decisions, director planning, memory consolidation | Queue, coalesce, defer or use deterministic fallbacks |

Never block game simulation on every background citizen's inference. Cache reusable identity prefixes where supported; cap generation length; use structured outputs and constrained decoding; reuse a single loaded model; minimize active KV cache contexts to fit target hardware. Distinguish *time to first audible response* from *time to completed validated action*.

Local ASR/TTS are important for the intended immersion; they introduce their own CPU/GPU costs, memory pressure, voice-selection concerns, and synchronization requirements. Prototype with full LLM + TTS as requested, but retain text-only mode and instrument individual pipeline stages.

### 14.4 Hardware and portability

The notes explicitly identify **VRAM as the likely largest constraint**, and propose using the **Steam Hardware Survey** to guide target hardware—not merely developing for an enthusiast PC. Define realistic VRAM/CPU/RAM tiers from current survey data at implementation time; evaluate quantized model sizes, KV cache cost, TTS/ASR coexistence, FPS, and performance on representative machines.

A bounded 2D prototype should answer whether acceptable responsiveness is possible locally, which workloads need different models, and which simulation layers require abstraction. No unsupported tokens-per-second or hardware compatibility claims should be treated as requirements.

## 15. Prototype: observable 2D village simulation

**Purpose:** establish whether believable NPC agency, player-directed dialogue, local speech, flexible story changes, and reliable rule enforcement can coexist under realistic hardware limits. This is also an **agentic development test harness**.

### Scope

- **2D simulation** with a town or neighborhood, interactable buildings, audibility/vision overlays, NPC schedules, and inspectable state.
- ~10–15 named NPCs (tunable), with some background crowd/routine actors; groups with different goals and resource limits.
- One central conspiracy or **Hidden Mafia** story containing immutable anchors and multiple flexible checkpoints.
- One intentionally unimportant grocery store NPC that can become meaningful *or remain irrelevant*.
- Spies, reporters, rumors, private conversations, social emotions, promises, bargaining, a simple recruitable crew, and a possible betrayal.
- Basic inventory, money, repair priorities, and faction cost/inaction modeling.
- **Full local LLM plus TTS path**, with text mode and developer controls.
- Director agent + story controller + NPC controller separation.

### Prototype instrumentation

Visualize and log: perception cones/ranges, sound propagation and occlusion, addressed vs overhearing actors, retrieved memories, prompt tokens, generated tokens, response latency, selected actions, rejected actions and reasons, cost/resource changes, world-truth versus NPC-belief divergence, information provenance, director proposed/accepted/rejected changes, active clues and commitments, and each story anchor/checkpoint's status.

### Automated player scenarios

Use a **large predefined suite of simulated-player scenarios** to validate both fidelity and reliability. Include:

1. Friendly investigation through several NPCs.
2. Ignore all obvious plot hooks; investigate the grocery store instead.
3. Threaten, bribe, comfort, or deceive an informant.
4. Secretly hurt an NPC, then try to console them without being identified.
5. Ask for a private conversation while spies are listening.
6. Spread a false rumor and measure propagation/contradictions.
7. Recruit allies, then require a risky favor; test their limits.
8. Break infrastructure repeatedly and test repair priorities, budgets, and bankruptcy.
9. Kill, alienate, or isolate a planned informant; verify alternate evidence paths.
10. Ask about knowledge an NPC cannot possess.
11. Accuse innocent people; verify that the director sometimes leaves them innocent.
12. Attempt open-ended solutions not present in a fixed quest menu.
13. Wait for long periods: verify routine world changes but no unintended offscreen main-story resolution.
14. Stress nearby listeners, simultaneous events, long-running memories, ASR ambiguity, and low VRAM.
15. Replay the same seed with different player styles; require different causal outcomes, not cosmetic dialogue changes.

### Acceptance criteria and metrics

Measure **P50/P95 first-audio latency**, complete response latency, tokens and peak VRAM per agent task, concurrent throughput, frame-rate impact, intent/addressing accuracy, correct perception/overhearing, schema-valid actions, controller rejection rates, memory retrieval fidelity, hallucinated-knowledge rate, consistency violations, number of independent discovery paths, narrative dead ends, faction budget integrity, and diversity of *material* story outcomes.

Set numeric performance targets only after baseline profiling on representative Steam-survey hardware. Behavior should remain deterministic and understandable when inference fails or times out.

## 16. Development sequence

**Phase 0 — Specification and harness.** Formal event schemas, NPC/relationship/world state, story anchor model, tool contracts, scenario runner, deterministic replay, metrics, and seedable tests. Include the 2D perception visualizer immediately.

**Phase 1 — Local conversation and NPC actions.** Addressed/overheard speech, single model runtime, minimal context assembly, tool proposals, authoritative validation, basic TTS, and persistence. NPCs can trade, move, share claims, decline, and make conditional promises.

**Phase 2 — Social and intelligence.** Emotions, trust, fear, obligations, manipulation, group memberships, espionage, privacy, rumors, belief provenance, and crews.

**Phase 3 — Improvisation and investigations.** Immutable anchors, flexible checkpoints, clue-carrier system, three independent paths, optional role promotion, and validated director proposals; implement Hidden Mafia + grocery store scenario.

**Phase 4 — Economy and ambient life.** Faction budgets, costs of action/inaction, prioritization, everyday incidents, repairs, business change, background simulation, and core-plot advancement gating.

**Phase 5 — Optimization and expansion.** Quantization experiments, bounded decision models, dynamic capability catalogs, batching and cache tests, streaming speech, multi-tier hardware matrix, repeated simulated-player runs, and larger worlds only after performance and coherence are demonstrated.

This is a **dependency sequence**, not a suggestion to discard the integrated prototype goal. Stub or minimally implement every cross-system contract early, then harden each system while testing the whole vertical slice continuously.

## 17. Major design tensions and explicit resolutions

| Tension | Resolution |
| --- | --- |
| Living world vs player-paced story | Simulate ordinary life continuously; gate the main story's irreversible beats unless an explicit timer is part of the scenario. |
| Improvised mystery vs deducible truth | Fixed anchors and evidence commitments; choose between seeded-answer and constrained-undecided mysteries *before* gameplay. |
| Player interest vs pandering | Promote eligible characters; allow innocent dead ends, meaningful failure, and unfulfilled expectations. |
| NPC agency vs narrative coherence | NPCs act for themselves; director shapes opportunity, story controller validates and paces commitments. |
| Fully modeled economies vs consumer hardware | Economically consequential budgets and costs, but aggregate distant and low-impact activity. |
| Free language vs finite game actions | Interpret intent, compose available capabilities, and provide planner fallbacks without inventing impossible effects. |
| Individual memory vs context limits | Persist all events; retrieve compact NPC-specific slices and summarize in background. |
| Natural emotion vs exploitable meters | Multidimensional relationships, beliefs, and event causality; no universal honor score. |
| Low latency vs agent autonomy | Event-driven requests, priorities, batching, precomputation, deadlines, and deterministic fallbacks. |

## 18. Decisions deliberately left open

These should be answered through prototype evidence rather than disguised as settled design:

- **Setting and title:** fantasy, historical, contemporary, or science fiction? *Hidden Mafia* and *The Last Bell* are example scenarios, not a mandatory final world.
- **Narrative mode:** mystery with seeded fixed answer; constrained collaborative reveal; or a per-scenario mixture?
- **Main-story timing:** generally player-gated (current default), configurable crisis clocks, or occasional authored timed chapters?
- **Presentation:** typed dialogue, voice by default, or both? How important are animation and fully voiced background barks?
- **Consequences:** how harsh may betrayal, permanent death, failed quests, and group collapse be?
- **Agent autonomy:** how much long-horizon planning is permitted without explicit player influence?
- **Economy scale:** exact goods, local budget abstraction, or simulation detail based on proximity and narrative importance?
- **Target hardware and offline mode:** what minimum Steam-survey hardware segment must be supported, and what fallbacks are acceptable?
- **Evaluation:** what player-study results are needed before declaring emergent storytelling successful?

## 19. Source reconciliation — all eight photographed pages

This appendix is a **normalized coverage inventory**, not a claim of forensic verbatim transcription. It preserves substantive ideas and examples from every page; obvious strike-throughs are treated as drafting edits, not separate features. The page order below is the uploaded image order.

| Page / image | Notebook ideas captured | Where developed |
| --- | --- | --- |
| **1 — 1000203500** | “Ideas”: spies for different groups; citizens report crime; spies seek useful information; spies listen to player/NPCs and spread intel; players detect snooping or listening NPCs; request quiet conversations; NPCs choose nearby/private or owned areas; hidden goals, betrayal, traps, deliberately speaking near spies. | §§ 5, 7–8, 15 |
| **2 — 1000203501** | Players influence NPC emotions: temptation/desires, anger/annoyance, fear, sadness/comfort, happiness; causing sadness covertly then using comfort as leverage; empathy to build crews or defenders; more natural than RDR2's honor effects. | §§ 5–6, 15 |
| **3 — 1000203502** | Few strong story anchors; many weaker improvisable checkpoints; *Hidden Mafia* to uncover underground group and leader; twists can be rewarding; hierarchy impacts difficulty of revealing secrets; leader identity could be improv checkpoint while mafia existence/high-level goals are unchanging anchors. | §§ 4, 15 |
| **4 — 1000203503** | “Improvements”: strong improvisation system; (1) story checkpoints unlocked by thresholds of knowledge from speech, reading, location, events, etc.; put newly available objects/speech/info in areas not explored; (2) information “gatherers/catchers,” detailed allowable presentation forms, many options, predefined appropriate areas and NPC types, associated people should disclose the info. | § 4 |
| **5 — 1000203504** | NPCs need real goals/intentions and must act to fulfill them; dynamic world, but story should not substantially progress without player interaction; NPCs believable and not player-centered unless player is canonically special (Doomguy/Halo Spartan/Dragonborn examples); loose ambient stories—crime, death, bankruptcy, changes to businesses/buildings; ordinary city operations; broken things repaired when budget/priority/time allow. | §§ 5, 9–10 |
| **6 — 1000203505** | Groups have meaningful economies and constraints; government should not prioritize a telephone pole above destroyed major infrastructure; weigh actions and desires against capacity; every action has cost and **inaction has cost**; groups can be bankrupted; story needs may set boundaries; orchestrating DM agent resolves contradictions/missing pieces alongside automated story system. | §§ 11–12, 17 |
| **7 — 1000203506** | “Tech” drawing: DM Agent → Story Controller; NPC Agent → NPC Controller; LLM and TTS paths; route spoken player input by whether it is aimed at someone, activate appropriate NPC; per-NPC context should be very small but preserve current plot, history, backstory, and relevant past interactions; nearby NPCs react to player and NPC behavior. | §§ 8, 12–14 |
| **8 — 1000203508** | “Prototype”: 2D simulation; measure/visualize what NPCs can see and hear; NPC actions using full LLM and TTS; large prepared set of simulated-player scenarios; discover tech bottlenecks and optimizations; create robust tests for agentic development; broad systems integration; Steam Hardware Survey guides machine limits, VRAM likely most concerning. | §§ 14–16 |

### Earlier conversation ideas explicitly retained

- The original dream of conversationally open NPCs whose responses meaningfully change the story, coupled with the challenge of making a finite world feel infinite.
- Mark's three major tabletop insights: promote beloved NPCs into greater story relevance, provide at least three paths to key discoveries, and use flexible hooks/clues instead of rigidly scripting every event.
- Mark's group/culture/clan and overlapping membership tags, plus the *Brindlewood Bay*-like possibility of leaving certain mystery details undecided until a coherent reveal is formed.
- The grocery store / suspicious shopkeeper / cult example as a concrete test of the director's ability to redirect a plot without simply validating every suspicion.
- NPC requests and hidden quests: trading a favor or item for information, discovering secrets and using blackmail, withholding access until conditions are met.
- Local LLMs with many logical agent contexts, explicit tool-based NPC decisions, a supervising DM/director, authoritative controller validation, selective memory, and shared asynchronous inference instead of one model running constantly per NPC.
- The earlier optional *Last Bell* fantasy-city crisis example remains a valid setting candidate, **not** an adopted premise; its seven-day automatic countdown would conflict with the notebook's default player-paced central story unless deliberately enabled.

---

## 20. Final design statement

This game should operate as **three coupled simulations**:

1. **A social world:** individuals and groups perceive, remember, desire, act, trade, hide information, build loyalties, and absorb costs.
2. **A bounded narrative world:** strong authored anchors and flexible checkpoints establish meaningful stakes while allowing several credible discovery and resolution paths.
3. **A local agentic runtime:** voice/text becomes intent; NPC agents choose bounded actions; the DM agent proposes improvisation; deterministic controllers enforce state, cost, causality, and continuity.

Success means the player can investigate the “wrong” store, ask for a private conversation, manipulate an informant, bankrupt a faction, uncover a mafia through an unexpected witness, or be betrayed by a trusted ally—and the game handles those possibilities as **connected outcomes in the same persistent world**, not as disconnected LLM improvisations.

## Appendix A — Close normalized reading of the original notes

The following preserves the notebook's **wording and order as closely as legible**, with punctuation and obvious spelling normalized for readability. This is included in addition to the conceptual mapping above, so small asides do not disappear during synthesis. Square brackets indicate an uncertain reading or editorial clarification; crossed-out drafting fragments are not treated as requirements.

### Photograph 1 — `1000203500.jpg` — “Ideas”

- “Spies for different groups.”
  - “Citizens reporting crime.”
  - “Spies looking for useful info.”
  - “Spies will listen to player and NPCs [and] spread [information/stories]. Then players can look out for NPCs sneaking around & NPCs listening.”
- “Players can ask NPCs to go somewhere quiet to avoid spies.”
  - “NPCs will go to nearby quiet areas, preferring owned areas.”
- “NPCs will have hidden goals and may betray players.”
  - “NPCs can lead players to traps, talk near spies or more.”

### Photograph 2 — `1000203501.jpg` — emotions and empathy

- “Players can mess with NPCs' emotions.”
  - “Temptation — tempt NPCs with things they desire to get the given result.”
  - “Anger/annoyance.”
  - “Fear.”
  - “Sadness/comfort — players who make an NPC sad will get different results than making them happy.”
  - “If a player makes an NPC sad but the NPC doesn't know the player was involved, the player can use that to their advantage (like IRL manipulation).”
- “Players can use this empathy system to build crews or create NPCs that will defend them without question.”
- “This system should feel more natural than the typical core effects like how RDR2 works with honor.”

### Photograph 3 — `1000203502.jpg` — story anchors and Hidden Mafia

- “The story should have a few strong anchor points with systems that heavily work off of weaker checkpoints that can be improvised off of.”
- “Example story — Hidden Mafia.”
  - “We want the player to uncover the underground mafia and find who's been leading it.”
  - “Bonus points for making it a twist.”
  - “For this we can have NPCs higher up on the hierarchy of the gang have higher and higher thresholds for giving away info.”
  - “Deciding on who the mafia leader is could be an improv checkpoint but the existence of the mafia and their high-level goals would be unchanging anchor points.”

### Photograph 4 — `1000203503.jpg` — improvisation subsystems

- “Improvements.”
- “The game needs to have a strong improv system. To do this the game needs two subsystems.”
  1. “Story checkpoints — when a player is given a threshold of info either through speech, reading, locations, events, and more the system enables new information to be placed in areas the player hasn't looked at yet. This can be objects, speech, and more.”
  2. “Information [catchers/gatherers] — information needs to have strong definitions on what it can be given as. Info should have as many options as possible to make improv easier. Info also needs to have predetermined areas and types of people. People associated with the info should be the ones to give it.”

### Photograph 5 — `1000203504.jpg` — autonomy and background world

- “NPCs will need to act with real goals and intention. This should lead to them running actions to fulfill these goals. The world should be dynamic but the story shouldn't meaningfully progress without player interaction. But NPCs should still be believable.”
- “Things shouldn't revolve around the player unless the story is specifically about the player being a special person in the world (think Doomguy/Halo Spartan/Dragonborn).”
- “To fulfill the goal of believable and dynamic worlds there should be a set of loose story types that happen without player interaction. Think like crimes happening, death, bankruptcy, changes in businesses, buildings and more. Cities should act as normal. Breaking something should result in the concerned party fixing it if it's possible given their capacity (money, priority, time).”

### Photograph 6 — `1000203505.jpg` — group economics and orchestration

- “All groups should have full economies to abide by. For example a government shouldn't fix a telephone pole if a major infrastructure was destroyed as that priority wouldn't make sense.”
- “This would require actions and desires to also be weighed against capacity to do that action. Each action should have a cost but things should have a cost for not getting done. This means it's possible to bankrupt groups by doing things that cost too much for them to continue.”
- “This system would need to be limited based on how the story needs it to end. But this is where an orchestration (DM) agent could handle how this works. The automated system should decide the story and the agent should decide on how that story changes when there are contradictions or missing pieces.”

### Photograph 7 — `1000203506.jpg` — “Tech”

- Drawing: **DM Agent → Story Controller**; **NPC Agent → NPC Controller**; **LLM** linked to agents; **TTS model** connected to the NPC's speech path.
- “Player speaks.”
  - “Process what was said, [and] was it directed towards someone.”
  - “If it was, who is the player looking at? Activate the NPC agent for the relevant NPC.”
- “Each NPC will have their own context.”
  - “Needs to be as small as possible.”
  - “But needs to contain everything required to get the current story info, backstory, and contextually retrievable interactions.”
- “Nearby NPCs should also react to player and other NPC behavior.”

### Photograph 8 — `1000203508.jpg` — “Prototype”

- “2D simulation.”
  - “Measure and visualize what each NPC can see and hear.”
- “Have NPCs make the actions and use full LLM and TTS systems.”
- “Use a large predetermined set of ‘player’ scenarios to validate the system.”
- “Goal is to determine bottlenecks on the tech, where we need to optimize, and to build a good test framework for agentic development.”
- “The system needs to work on a large portion of systems. Use Steam hardware survey as the guidance on limitations. VRAM is likely the biggest concern.”

**Reading caveat:** The handwritten term in photograph 4 looks like “Information Catchers” but could mean “Information Gatherers”; both express the idea of explicitly defined carriers and locations for clues. Photograph 1 includes a few overwritten words around what spies do with conversations. Neither uncertainty affects a substantive feature's inclusion.
