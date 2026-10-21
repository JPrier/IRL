# Emergent Quest-Driven RPG — Consolidated Game & Technical Design

**Status:** Working design, not a finalized implementation specification  
**Date:** October 20, 2026  
**Scope:** Consolidates the original RPG idea, Mark's tabletop-design feedback, the local-LLM/agent architecture discussion, the quest-system clarification, and all eight pages of handwritten notes.  
**Working setting:** *The Last Bell* (illustrative, replaceable). A hidden-mafia investigation is a second example and an excellent prototype scenario.

---

## 1. Vision

Create a **quest-driven, open-ended RPG** in which players can speak naturally to characters, form genuine relationships, pursue unusual solutions, and meaningfully change the world. Quests should feel like things people actually want or need, not arbitrary assignments issued by quest dispensers.

The fundamental design distinction is **finite systems, open-ended interactions**. The game need not contain an infinite map or generate limitless bespoke content. It needs enough consistent rules, characters, motivations, resources, and relationships that a large range of believable stories can emerge.

The player may be investigating a conspiracy, helping a shopkeeper obtain medicine, organizing a community, blackmailing a merchant, or entirely ignoring the main story. NPCs and factions continue pursuing goals in every case. An AI story director recognizes interesting player choices and develops narrative opportunities without destroying established facts or removing the possibility of failure.

**Central promise:** *Any character can matter, but not every character exists for the player. Any meaningful obstacle can become a quest, but not every quest is assigned.*

### 1.1 Design pillars

1. **Emergent quests are central.** Player and NPC goals create requests, obstacles, bargains, opportunities, investigations, and hidden quests. Strong authored quests remain welcome.
2. **NPCs have autonomy.** Characters have goals, intentions, limited knowledge, schedules, identities, loyalties, fears, and tools to act on the world, including when the player is absent.
3. **A living world with consequences.** Crime, death, businesses, construction, finances, rumors, faction initiatives, and changing relationships exist independently of plot triggers.
4. **Flexible narrative, committed reality.** Strong story anchors coexist with weaker movable checkpoints. The director may improvise unobserved details, but may not casually retcon what has been established.
5. **Situated communication.** People see and hear things within their perceptual reach. Conversations can be overheard, secrets can leak, and privacy can be negotiated.
6. **Emotional and social causality.** Fear, temptation, anger, sadness, friendship, reputation, obligation, and manipulation influence choices. Being liked is not the same as being obeyed.
7. **Local-first, hardware-conscious technology.** Lightweight simulation runs continuously; shared local LLM inference is reserved for decisions and conversations where it adds value.
8. **Testable emergence.** Reproducible scenarios and automated players probe coherence, freedom, latency, cost, and narrative quality.

### 1.2 Tabletop design foundations from Mark

Mark's feedback supplies concrete techniques that the computerized game should preserve:

- **Follow player attention.** When a player repeatedly returns to an unexpected NPC, give that person a chance to become important rather than insisting on a preselected quest giver. Narrative importance can be assigned through an eligible role or connection.
- **Move flexible hooks, not established truths.** When a player goes in a different direction from the prepared clue, adapt where an undiscovered clue or relevant situation can plausibly appear; sometimes reskin a prepared hook to fit the place and people involved.
- **Provide at least three useful routes to major revelations.** Avoid single-NPC or single-item bottlenecks so the sandbox can survive missed clues, hostile characters, or unexpected decisions.
- **Build people from overlapping affiliations.** A character may simultaneously identify with a neighborhood, profession, guild, religion, fandom, philosophy, and family. These memberships help determine dialogue, trust, disagreement, and generated requests.
- **Prepare a premise and situations, not every event.** Encourage improvisation when the player pushes at the world's boundaries; use believable connections to draw the player back toward major conflicts when appropriate, but do not force it.
- **Borrow carefully from *Brindlewood Bay*.** Its collaboratively constructed investigation suggests a way to present flexible clues and delay some answers. That approach should coexist with fixed-answer mysteries when genuine deduction is important.

These are design influences rather than a requirement to make every storyline follow the same improvisational rules.

### 1.3 What the game is *not*

It is not merely a chatbot standing in front of an RPG map. It is not a world where every suspicion becomes true, every NPC instantly knows everything, or the director secretly forces all choices into one ending. Nor is it anti-quest: delivering something someone needs, negotiating an exchange, pursuing a secret, or sabotaging a rival are all meaningful quest gameplay.

---

## 2. Example Worlds and Core Player Experience

### 2.1 Illustrative larger RPG: *The Last Bell*

The player arrives in a fantasy city protected by a magical bell that is deteriorating, with an initial seven-day crisis timeline. A council wants stability, a religious order has its own interpretation, merchants protect their influence, and neighborhoods consider evacuation. The initial crisis, faction motivations, and world rules are authored. How the player learns about them, whom they befriend, how they change the balance of power, and whether the city survives can vary.

This premise is **an example, not a locked project requirement**. The architecture should be setting-independent.

### 2.2 Prototype mystery: the hidden mafia

A town contains an underground criminal organization. The player may try to discover its existence, identify its leader, and deal with it. The **organization's existence and broad goals** are story anchors. The exact clues, intermediary conversations, and plausible routes to identification can be flexible. A surprising but coherent revelation is desirable; it must not be fabricated at the cost of prior evidence.

Higher-ranking members have stronger incentives, protective measures, and thresholds against revealing secrets. Reaching the leader should be possible through several independent chains: records, informants, surveillance, leaked gossip, faction conflict, or a player's own coercive strategies.

### 2.3 The grocery-store example

The player is investigating a cult but becomes obsessed with an apparently ordinary grocery store. The director detects sustained interest. It may, **if the facts permit**, attach a relevant unresolved story connection to the store, such as a wholesaler tied to the cult, a courier, a victim, or an informant. Or the grocer may remain innocent, and breaking into the shop creates an entirely different problem.

The store does not become cult-affiliated automatically because the player guessed it. Once a detail has been revealed, tested, or otherwise committed, the director cannot silently overturn it.

---

## 3. Quests: Goals, Obstacles, Dependencies, and Commitments

**Quests are not restricted to a quest-board object or dialogue option.** They are meaningful efforts to achieve goals under constraints.

The engine should distinguish:

- **Goal:** A desired world state held by a player, NPC, group, or faction.
- **Obstacle:** Something preventing progress: unwillingness, lack of resources, distance, risk, missing knowledge, law, or opposition.
- **Opportunity:** A currently feasible method to make progress.
- **Commitment:** A promise, deal, contract, threat, debt, or responsibility recognized by one or more parties.
- **Quest:** A player-relevant pursuit composed of one or more of the above; it may be explicit, implicit, discovered retrospectively, or entirely self-directed.
- **Outcome:** A change in world state, relationships, knowledge, or resources, not merely a completed checkbox.

### 3.1 Example: requested favor

The player needs shipment information from a merchant, but the merchant is unwilling to give it. The merchant's daughter needs medicine, and the merchant proposes an exchange: obtain the medicine in return for the records. This is a naturally emerging fetch quest because the daughter, merchant's concern, medicine, and supplier all have reality in the world.

The apothecary may refuse because the guard confiscated key ingredients, creating another obstacle and potentially a larger storyline. The player could negotiate, purchase, steal, substitute treatment, expose the guard, or abandon the favor.

### 3.2 Example: hidden blackmail quest

The player could instead discover the merchant is concealing an illegal trade. Nobody offers the objective “find dirt on the merchant.” The player investigates, obtains credible evidence, and chooses to pressure him. The merchant weighs the credibility of exposure, fear, ethics, relationship with the player, alternatives, and danger. He might comply, lie, flee, retaliate, call guards, or offer another bargain.

Importantly, **a believable outcome is not guaranteed compliance**. The player may obtain the information and earn a dangerous enemy at the same time.

### 3.3 Quest types

| Type | How it emerges | Example |
|---|---|---|
| Explicit request | NPC expresses a need | “Bring medicine and we can talk.” |
| Negotiated agreement | Player and NPC bargain | Protect a delivery for information. |
| Implicit/hidden | Player notices exploitable state | Find evidence for blackmail. |
| Investigation | Information gap blocks a goal | Trace shipments to a cult. |
| Relationship | Trust or obligation opens activity | Help an ally organize neighbors. |
| Faction | Group goals generate assignments | Move supplies before a rival faction does. |
| Authored | Handcrafted dramatic situation using shared systems | A major character's personal arc. |
| Self-directed | Player invents a goal | Establish a communal kitchen or become a landlord. |

NPCs can create dependencies and quests **for one another**, without involving the player. The player may stumble onto these chains or become an intermediary.

### 3.4 Design rules for organic quests

- NPC demands must be motivated by what they want, need, fear, or believe—not automatically generated as arbitrary chores.
- Promised rewards must be tied to resources, information, or influence an NPC plausibly controls.
- Commitments persist when the player walks away; breaking a promise should have consequences.
- Completing an objective by another route must remain valid; the original NPC may react differently.
- Multiple methods should usually exist, including unexpected but mechanically valid solutions.
- A journal may track known commitments and inferred leads without revealing hidden quests as if the player were omniscient.
- Quests can become impossible, change ownership, or fail if people die, opportunities close, or the world changes.

---

## 4. NPC Identity, Autonomy, and Knowledge

Every significant NPC should have three identity layers:

1. **Personal:** Personality, habits, history, values, fears, ambitions, capabilities, occupation, and moral boundaries.
2. **Social:** Membership in several groups, such as family, religion, guild, class, political movement, philosophy, or subculture. Memberships create compatible and conflicting loyalties.
3. **Narrative:** Relationships to current story conflicts, secrets, player attention, existing obligations, and possible emergent narrative roles.

An NPC is not just a prompt. It is **persistent simulation state + behavior policies + a scoped set of executable actions + optional LLM reasoning**.

### 4.1 Goals and independent behavior

NPCs should actively pursue goals: obtain food, protect family, keep a shop solvent, avoid exposure, build an alliance, spread a rumor, investigate a crime, or take revenge. They should move, trade, negotiate, and cooperate using the same fundamental world rules as the player, subject to simulation budgets.

Daily schedules and routine navigation do **not** need LLM calls. Unexpected conflicts, significant negotiations, novel goals, or important social choices are candidates for LLM reasoning.

An NPC may hide a goal, deceive another person, bait the player into a trap, lure the player near spies, or betray a promise. Such behavior must follow their actual motives and plausible knowledge.

### 4.2 Separate truth, observation, belief, and disclosure

Maintain four distinct concepts:

- **World truth:** What has actually happened and the facts the engine has committed.
- **Observed information:** What an NPC personally saw, heard, read, or reliably received.
- **Beliefs:** What the NPC thinks is true, including rumors, misconceptions, and lies.
- **Disclosure policy:** What the NPC is willing to share, and with whom, under current conditions.

A baker may sincerely but incorrectly believe the council kidnapped her child. Another character may know the truth but lie about it. The director may know both perspectives. No NPC should get omniscient information from the model's general story context.

### 4.3 Emotional and social model

At minimum, model **trust, affection, respect, fear, anger, sadness, temptation/desire, suspicion, obligation, and perceived power** as distinct factors, with personality-based differences in response.

Examples from the notes:

- A desirable gift or opportunity can tempt an NPC into a bargain.
- Angering a character may increase resistance or provoke retaliation.
- Making someone sad may produce withdrawal or a request for help rather than hostility.
- If the player covertly caused the sadness and the NPC does not know it, the NPC may still trust the player and become vulnerable to manipulation. The causal truth and the NPC's belief must remain separate.
- Long-term relationships may create allies, bodyguards, crews, or people who defend the player—without making them mindlessly loyal.

The social system should feel **more natural and multidimensional than a single honor meter**, though games such as *Red Dead Redemption 2* are useful points of reference for reputational consequences. A community may see the player differently from the authorities, merchants, or a specific family.

### 4.4 Narrative role promotion

Important story roles—informant, organizer, betrayer, protector, witness, rival—can be assigned to plausible NPCs as a campaign evolves. Candidate eligibility depends on established skills, location, relationships, motive, knowledge, and player investment.

Player attention can increase an NPC's *story relevance*, but should not rewrite their character to fit a predetermined plot. A grocer with contacts could become an informant; they should not become a wizard out of nowhere. Roles may remain unfilled.

---

## 5. Perception, Conversation, Spying, and Rumors

Conversation is spatial and social, not a private direct message to whichever NPC is currently selected.

### 5.1 Hearing and seeing

The engine determines who can perceive an event before any NPC reasons about it. Candidate signals include:

- Distance, line of sight, facing direction, lighting, obstructions, and movement.
- Speech loudness, whispering, room acoustics, ambient noise, doors/walls, and attention.
- Whether speech is explicitly addressed to an NPC; if not, infer the likely recipient from facing, gaze, proximity, conversational context, and fallback UI selection.
- Nearby NPCs may react to the player's words **and to the behavior of other NPCs**.

Each observer gets a **subjective perception event**, not the entire action payload or privileged world truth. Overheard speech may be incomplete or uncertain.

### 5.2 Privacy as gameplay

Players can ask a character to talk somewhere quiet or avoid eavesdroppers. The NPC can accept, reject, suggest another location, or lead the player to a nearby secluded area using navigation and preference rules. Spies may follow, loiter nearby, or exploit predictable meeting places.

The player can observe suspicious loitering, search for watchers, intercept couriers, or listen in on NPC discussions. Privately exchanging information has real strategic value.

### 5.3 Rumor propagation and spies

Factions can employ informants or spies to gather useful information. Civilians may report crimes to authorities. Rumors travel through interactions and social networks rather than instantly appearing in everybody's memory.

A rumor record needs: the claim, source, witnesses, chain of transmission, confidence, perceived credibility, known distortions, secrecy classification, and time. Recipients interpret it according to their beliefs and relationships; it may mutate or be suppressed.

Spies have goals (e.g., identify an organizer, locate hidden inventory, intercept a deal), schedules, and visibility risks. Discovering and avoiding intelligence networks becomes an emergent subgame.

---

## 6. Story Architecture: Anchors, Checkpoints, and Information Containers

The story should have **a few strong anchors** and many **weaker, improvable checkpoints**. This is one of the core design ideas in the handwritten notes.

### 6.1 Strong anchors

Anchors are fixed or deliberately author-controlled constraints, such as:

- The hidden mafia exists.
- It has particular long-term goals and high-level organizational behavior.
- The bell is failing, and established physical/magical rules constrain repairs.
- A major event already witnessed by the player happened as recorded.

An anchor need **not** predetermine an ending. It can establish the premise or facts of a conflict while leaving outcomes open.

### 6.2 Flexible checkpoints

Checkpoints are story developments the player can reach via various means, such as discovering a criminal organization's existence, locating a courier, or identifying a plausible connection between two factions.

The director can vary **where, through whom, and in what form** information appears. For a valid clue about shipments, it might select a manifest, merchant conversation, intercepted letter, observed delivery, or testimony from a reliable source.

A player reaching a knowledge threshold through reading, speech, exploration, or world events can unlock additional *eligible* clues in areas not yet investigated. This is not license to create evidence retroactively in places the player already thoroughly searched or to force people to know things they could not know.

### 6.3 Information containers

Treat a discoverable fact independently of its delivery method. The story engine describes a fact, its sensitivity, prerequisites, eligible holders, eligible areas, and presentation forms. A **container** is a permitted realization of that fact: person, document, environmental object, overheard remark, transaction, or event.

Placement constraints include:

- Geographical and organizational plausibility.
- People likely to possess that information.
- Story progression and access prerequisites.
- Existing observation and search history.
- Independence and redundancy of clues.
- Whether the container itself has been observed and committed.

Information may be discoverable through several channels. Follow the **Three Clue Rule** for critical revelations where practical: provide several independent leads, not simply three differently worded versions of one bottleneck.

### 6.4 Mystery modes

Support two intentionally distinct modes:

- **Fixed-answer mysteries:** The underlying answer is generated or authored in advance. Evidence is discoverable through flexible routes, but the answer remains stable.
- **Collaboratively resolved mysteries:** Some answer details remain uncommitted while players develop theories. The director can choose a coherent resolution among still-possible alternatives, then permanently commit it.

Do not mix the modes invisibly in a way that invalidates deduction. The player must be able to investigate and be wrong. If the grocer has been demonstrably cleared, the director cannot subsequently turn that same evidence into a secret confession without an internally valid new explanation.

### 6.5 Director's responsibilities and limits

The AI story director should:

- Detect player interest, neglected threads, pacing problems, and emerging conflicts.
- Propose hooks, assign plausible narrative roles, and select information containers.
- Coordinate story-facing consequences of independent faction/NPC activity.
- Weigh twists and revelations against setup and consistency.
- Keep narrative opportunities available without guaranteed success or teleporting the world around the player.

It **must not** bypass the world engine, overwrite committed facts, force all choices into a single outcome, or make everybody secretly connected to the same conspiracy. It can propose changes; the engine validates and commits them.

---

## 7. World and Faction Simulation

The world should progress without player interaction, while remaining observable and causally believable.

### 7.1 Normal events

Simulate ordinary systemic consequences where relevant: theft, investigation, criminal activity, death, lost jobs, bankruptcies, business openings/closures, construction, building damage, supply shortages, migrations, leadership changes, and faction disputes. These should not all require a narrative director event.

NPC and faction actions should have feasible preconditions. Actions have costs, but **inaction also has costs**: ignored damage worsens, businesses lose customers, allies grow dissatisfied, and opportunities expire.

### 7.2 Factions have resources and priorities

A faction needs goals, governance, membership, territory, information, finances, manpower, assets, standing commitments, threats, and action capacity. An important group cannot do everything it wishes.

The handwritten government example is instructive: a government should not prioritize fixing a telephone pole while major infrastructure is collapsing, absent an exceptional reason. The same constraint applies to criminal organizations, merchants' guilds, churches, and player-founded groups.

Decisions should weigh expected benefit, urgency, cost, uncertainty, legitimacy, obligations, and opportunity cost. A faction may go bankrupt by pursuing overly expensive plans, or suffer from failing to act.

### 7.3 Simulation time and bounded scope

The game's timeline advances consistently. Not all actors need detailed simulation at all times: use high-resolution behavior near the player and coarser state transitions for distant actors, with conservation of inventory, money, time, and other resources. World-wide consequences must be reconstructible even when offscreen events were aggregated.

It is acceptable for a story to focus on the player—e.g., if the player is uniquely powerful or fated—**but that should be an explicit setting rule**, not a side effect of every NPC waiting for them.

---

## 8. Multi-Agent Architecture

The proposed logical design follows the sketch in the handwritten notes:

```text
                        +--------------------------+
                        |  Story Director / DM     |
                        |  Narrative Controller    |
                        +------------+-------------+
                                     | proposals
                                     v
+------------------+      +----------+-------------+      +------------------+
| Player I/O       | ---> | World Engine           | <--> | Event Store       |
| text / speech    |      | Rules, time, perception|      | Facts, beliefs,  |
| movement/actions | <--- | action validation      |      | commitments      |
+------------------+      +----------+-------------+      +------------------+
                                     ^
                                     | scoped actions/events
                        +------------+-------------+
                        | NPC / Faction Controllers|
                        | goals, plans, tools      |
                        +------------+-------------+
                                     ^
                                     | queued inference
                        +------------+-------------+
                        | Shared Local LLM Runtime |
                        | optional STT / TTS       |
                        +--------------------------+
```

These are **logical agents**, not necessarily separately loaded models or perpetual threads. A shared model can impersonate many NPCs using their individual context. The director can use the same model initially, then a larger model if tests justify it.

### 8.1 Responsibility boundaries

| Subsystem | Owns | Does not own |
|---|---|---|
| World engine | Canonical state, schedules, movement, physics/perception, legal actions, validation | Freely rewriting story facts |
| NPC controller | NPC goals, available actions, situational reasoning and dialogue | Omniscient knowledge or unchecked state mutation |
| Faction controller | Collective goals, resources, priorities, delegated actions | Unlimited funding or perfect information |
| Story director | Narrative proposals, attention tracking, clue placement, pacing | Executing unvalidated changes |
| Conversation adapter | Player speech transcription/routing, reply rendering, spatial audio | Inferring world truth from dialogue alone |
| Memory system | Event history, fact/belief retrieval, summaries, provenance | Treating summarized guesses as facts |

### 8.2 Tools and action execution

Instead of generating prose that claims an action happened, agents produce structured **action proposals**. Initially offer small capability-scoped sets, such as:

```text
speak(person, utterance)
move_to(location)
offer_trade(person, terms)
share_information(person, information_id)
request_help(person, goal_id)
make_promise(person, terms)
report_crime(authority, evidence_id)
investigate(target)
transfer_item(person, item_id, quantity)
propose_group_action(faction, plan_id)
```

The engine validates range, inventory, authority, time, permissions, knowledge, capacity, and world invariants. It emits an accepted event or a typed rejection. Rejection is useful feedback for a planner but should not trigger unbounded retry loops.

**Novel player actions:** Start with generated legal candidates for inexpensive routine decisions; allow a broader planner to decompose unanticipated requests into valid primitives. A request to establish a community kitchen could become a plan involving rent, supplies, staffing, permits, and ongoing operation. Novelty does not authorize unconstrained tool creation or engine mutation.

### 8.3 Transactional and replayable events

Record tool proposals and responses separately from committed game events. A possible envelope:

```json
{
  "event_id": "evt_0214",
  "sim_time": 18620,
  "actor_id": "merchant_01",
  "type": "InformationOffered",
  "payload": {
    "recipient": "player",
    "fact_id": "shipment_17",
    "condition": "medicine_delivered"
  },
  "observed_by": ["player", "witness_03"],
  "caused_by": ["interaction_013"],
  "commitment_level": "canonical"
}
```

An offer is not automatically a completed exchange. The player must accept or satisfy its conditions. Replay logs should include model version, inference input reference, chosen action, validator outcome, RNG seed, and resulting world events.

### 8.4 Compact NPC context, full persistent history

Each NPC has small, on-demand context composed of:

- Stable identity and currently relevant goals.
- Immediate observations and conversational state.
- Nearby actors, relationships, and accessible tools.
- Retrieved relevant memories, commitments, rumors, and secrets.
- Knowledge-provenance boundaries that prevent omniscience.

Keep the complete history in persistent storage, but **never stuff the entire history into every prompt**. Separate stable facts from lossy summaries. Memory consolidation should not turn uncertainty into certainty or overwrite canonical events.

### 8.5 Agent invocation scheduler

Use event-driven inference, not one LLM ticking continuously per NPC:

| Priority | Typical tasks | Latency expectation |
|---|---|---|
| P0 | Player-facing dialogue and urgent decisions | As responsive as hardware permits |
| P1 | Nearby consequential NPC actions and reactions | Short deferral acceptable |
| P2 | Director proposals, distant faction plans, reflection, memory consolidation | Opportunistic/background |

Coalesce repeated events, cancel obsolete work, put deadlines on tasks, and cap background work. If inference is overloaded, routine NPCs follow deterministic policies while low-priority narrative changes wait. Avoid making the player wait behind dozens of background agent calls.

### 8.6 Example invocation

When the player says “I have your smuggling records” to a merchant:

1. Speech or text input produces an utterance and a likely recipient.
2. Perception determines who heard it, including nearby listeners.
3. Relevant player and merchant state is assembled; no hidden director-only facts are exposed.
4. The agent chooses from scoped actions (deny, bargain, warn, flee, threaten, seek help) and may generate dialogue.
5. The engine validates the proposal, modifies observable state, and records the event.
6. Other listeners receive perception events and may act later.
7. The director is notified only if this event is narratively significant.

---

## 9. Local Inference, Speech, and Hardware Strategy

**Feasibility:** A shared local LLM handling bounded social decisions and short dialogue is technically plausible. High-quality long-horizon storytelling, simultaneous voice conversations, and low-end hardware coverage require empirical testing.

### 9.1 Model strategy

Begin with a **single quantized small instruction model** handling multiple roles through different prompts and JSON schemas. Compare a roughly 2B-class model for action selection, 4B-class for short dialogue, and 8–9B-class for complex director proposals if needed; these are **benchmark categories, not validated recommendations**.

Do not equate “one NPC agent” with “one model loaded into VRAM.” Many NPCs can share the same inference server and remain dormant while outside consequential interactions.

Use constrained, terse outputs for common decisions. A 20-token action choice and a 60-token reply may be much cheaper than a long chain-of-thought plus multi-round tool calls. For unusual planning, allow a slower fallback. Compare separate action/dialogue calls against a combined call using measurements rather than assumptions.

### 9.2 Memory and VRAM

VRAM is a likely major constraint, as identified in the notes. Account for model weights **plus** KV caches, batch sizes, context length, inference buffers, STT/TTS models, and any GPU usage from the renderer. Simultaneously keeping several specialized models resident may be worse than sharing one model.

Investigate actual consumer hardware distribution using **Steam Hardware Survey data**, but do not set requirements based on VRAM alone: measure memory pressure under real renderer + dialogue + background workload combinations. Maintain a fallback or lower-fidelity mode for less capable systems.

### 9.3 Voice pipeline

```text
Player microphone -> voice activity detection -> speech recognition
                  -> recipient inference / acoustic perception
                  -> NPC decision + dialogue generation
                  -> speech synthesis -> positional audio + lip/face cues
```

Speech recognition and text-to-speech (STT/TTS) are distinct from the LLM. The prototype should exercise both text and voice eventually because speaking fluidly is central to the fantasy; initial text-only debugging is still useful.

Test turn-taking, interruption, overhearing, microphone input leakage, speaker identity, whispering, and NPC-to-NPC speech. TTS should be interruptible, and a model response must not block movement or essential controls. Only committed dialogue should modify NPC hearing/memory.

### 9.4 Performance budgets and metrics

Measure on representative PCs:

- Time to first NPC response audio/text and total completed response latency.
- Input transcription latency and endpointing behavior.
- Tool-selection accuracy, invalid-call rate, and retries.
- Tokens and inference time per interaction; cache-hit rates.
- GPU memory peak, average CPU/GPU use, and rendering frame-time impact.
- Scheduler queue depth under many simultaneous NPC events.
- Quality under prompt compression and small-model substitution.

These are **targets to discover**, not measured results. Set minimum hardware specifications after benchmarking, not before.

---

## 10. Prototype: 2D Social-Mystery Simulation

The immediate goal is **not** to ship a full RPG. It is to build a controlled testbed that reveals technological bottlenecks, validates agent behavior, and creates a reusable automated evaluation framework.

### 10.1 Proposed scope

- Small 2D town/neighborhood with 6–8 traversable locations.
- About 12–15 individually modeled NPCs and 2–3 factions.
- One central mafia/cult mystery with strong anchors and flexible clues.
- Explicit and hidden quests: bargaining, investigation, eavesdropping, blackmail, protection, and trust-building.
- A limited but meaningful resource economy: money, inventory, favors, faction capacity, time.
- Sight/hearing simulation showing exactly what each NPC can see and hear.
- Full LLM actions and dialogue; text mode first for inspection, then voice input/output with STT/TTS.
- One shared local inference runtime and a low-frequency director.
- Headless, reproducible sessions plus an interactive player client.

A Godot client with a deterministic game-simulation core and a structured agent/inference boundary is one practical option. Language and runtime details should be selected after testing integration, portability, and performance. The prototype can begin with simple rendering rather than building a large content pipeline.

### 10.2 Automated player testing

Create a **large predetermined suite of simulated-player scenarios**, varying curiosity, aggressiveness, secrecy, persuasion, cooperation, exploration, deception, and quest refusal. Include adversarial players who repeatedly test boundaries.

Scenarios should include:

1. Befriend an irrelevant NPC until they become narratively important, without changing their established identity.
2. Investigate an innocent grocer despite a central cult plot.
3. Obtain the same information by favor, stealth, payment, persuasion, or blackmail.
4. Whisper near a spy; move to a quieter location; determine who really heard.
5. Cause a character sadness covertly and then exploit their trust; ensure mistaken beliefs remain separate from truth.
6. Kill, alienate, or lose an important clue holder; demonstrate alternate routes where appropriate.
7. Refuse the main story while NPCs and factions continue acting.
8. Push a faction toward bankruptcy; verify resource conservation and consequences.
9. Force conflicting world claims and verify the director cannot commit contradictions.
10. Generate several playthroughs from the same initial seed and compare meaningful divergence.

### 10.3 Acceptance criteria

| Area | What counts as success |
|---|---|
| NPC autonomy | NPC pursues an independently assigned goal and executes validated actions without player prompting |
| Organic quests | An NPC generates a motivated request, with multiple mechanically valid resolutions |
| Hidden quests | Player can exploit undisclosed facts to create an objective no NPC explicitly offered |
| Knowledge boundaries | No NPC reveals facts they could not know; rumors retain source/uncertainty |
| Perception | Observer set matches geometry/acoustic rules; unaddressed listeners react correctly |
| Story adaptation | A previously incidental NPC/location can become relevant without violating prior observations |
| Narrative anchors | Fixed premises remain stable while flexible checkpoints and outcomes vary |
| Factions | Offscreen decisions consume resources and reflect priorities and opportunity costs |
| Reproducibility | Recorded outputs + seed replay the same canonical events deterministically |
| Local performance | Meets measured platform-specific interaction and memory budgets |
| Emergence | Distinct strategies yield materially different consequences, not reworded identical quests |

### 10.4 Suggested development sequence

**Phase 1 — Canonical simulation and event model.** Implement locations, NPC identities, goals, resources, actions, promises, witnesses, and event replay. No LLM required yet.

**Phase 2 — NPC tools and social decisions.** Add one local model, restricted tool choices, knowledge retrieval, dialogue, and validity checks. Measure throughput/latency.

**Phase 3 — Perception and rumor network.** Add visual/acoustic events, directed conversation, private meetings, spying, secrecy, and belief propagation.

**Phase 4 — Emergent quest system.** Add bargains, dependencies, favors, investigations, blackmail, and NPC-to-NPC commitments using shared goals and resources.

**Phase 5 — Narrative director.** Add anchor/checkpoint graphs, information containers, role assignment, opportunity proposals, and consistency validation.

**Phase 6 — World autonomy and voice.** Add faction resources, offscreen events, STT/TTS, and large-scale scenario testing. Use independent automation to identify model-quality and hardware bottlenecks.

This is an implementation sequence, **not a proposal to drop systems from the final prototype**. The intended prototype tests the integrated experience.

---

## 11. Architecture Risks and Open Design Decisions

These should be measured or explicitly resolved; they are not settled requirements:

| Question | Why it matters | Initial approach |
|---|---|---|
| How much can the director improvise? | Too little feels scripted; too much destroys mystery | Immutable/committed facts + explicit uncommitted story slots |
| How many NPCs need full planning? | Inference cost scales with consequential decisions | Event-driven reasoning; deterministic routine behavior |
| How are unexpected actions supported? | Finite tools can undermine freedom | Hierarchical plans decomposed into validated primitives |
| How do we measure story quality? | Structural validity alone is not satisfying gameplay | Replayable scenarios + diverse-player and human evaluations |
| Can small models reliably maintain motives? | Dialogue may be plausible but inconsistent | Structured context, constrained actions, adversarial tests |
| How should narrative pacing work? | Autonomous worlds can progress beyond the player | Progress clocks, weighted events, explicit deadlines, bounded director intervention |
| What is the lower-end PC target? | Local voice/LLM + renderer compete for VRAM | Benchmark hardware tiers and Steam hardware distribution |
| How should mysteries declare their rules? | Fixed vs constructed answers support different play styles | Mark each mystery mode explicitly in story metadata |
| Should the player see quest markers? | UI can spoil emergent objectives | Optional journal of observations, goals, and accepted commitments |
| How should simulation pause or accelerate? | Offscreen plans and deadlines need coherent time | One authoritative simulated clock, event scheduler, pausable agent queue |

### Failure modes to deliberately prevent

- **Rubber-band storytelling:** Everything the player investigates becomes part of the main plot.
- **Quest vending:** NPCs repeatedly invent arbitrary trades disconnected from actual needs.
- **Omniscient gossip:** Everyone immediately knows secrets irrespective of hearing or transmission.
- **Tool hallucination:** Agents describe successful actions that the engine never performed.
- **Consequence-free influence:** Player can maintain absolute loyalty with one relationship score.
- **Resource-free factions:** Governments and guilds act without money, staff, time, or priorities.
- **Invisible retroactive clues:** Director plants contradictory evidence in previously inspected spaces.
- **Background inference storms:** Dozens of nonessential agents block live dialogue.
- **False replay determinism:** Re-running fresh model inference is mistaken for reproducing the prior outputs; capture actual model decisions for exact replay.

---

## 12. Worked End-to-End Story Example

A player hears that an underground mafia controls town shipments. They approach a grocer for records. The grocer refuses because exposing the smuggling network threatens his family. He offers information if the player retrieves medicine for his daughter.

The player instead watches the store. A nearby spy observes the player speaking to a courier and reports the meeting to the mafia. The player asks the grocer to move somewhere private. The grocer suggests a backroom, but that room is within earshot of an employee with divided loyalties. The player's decision to speak quietly becomes mechanically meaningful.

The player discovers the grocer has concealed illegal invoices and threatens blackmail. The grocer is fearful but also angry; depending on his perceived options and personal traits, he may reveal a partial clue, retaliate, lie, bargain for family safety, or inform the mafia. The engine validates evidence possession and all resulting actions.

The story director observes that the player has invested in the grocer and proposes a connection between the business's wholesaler and a mafia courier. If this slot is still uncommitted and consistent, it creates a real clue container: a manifest in an as-yet-unsearched warehouse, witnessed by a plausible employee. The grocer's guilt remains whatever the world has already established; the director does not simply endorse the player's theory.

Meanwhile, the mafia spends money relocating inventory, the town guard investigates unrelated thefts, and the merchant guild debates whether to cooperate. These processes occur without the player. As events progress, some opportunities close while others appear. The same scenario can produce a rescue quest, an investigative arc, a faction confrontation, an exposed betrayal, or an entirely different personal story.

The resulting *quest structure* is real, but it is discovered through human-seeming conversations, bargains, pressure, and consequences rather than a fixed series of dialogue-tree nodes.

---

## 13. Coverage of the Handwritten Notes

This section indexes the eight photographed pages so their original ideas remain traceable rather than disappearing inside the polished specification. Wording below is paraphrased for clarity.

| Page | Notes integrated into this document |
|---|---|
| **1 — Ideas: spies and secrets** | Faction spies, civilian crime reports, listening for useful information, NPCs sharing rumors; player spotting listeners; asking people to move to quieter nearby places; hidden NPC goals, betrayal, traps, and deliberate information exposure. |
| **2 — Social emotions and crews** | Temptation/bribes, anger/annoyance, fear, sadness and manipulation; player covertly causing distress without attribution; empathy and the ability to create loyal groups/defenders; natural social consequences beyond *RDR2*-style honor. |
| **3 — Story anchors / hidden mafia** | Few strong anchors, weaker improvised checkpoints, uncover an underground mafia, high-ranking members with higher thresholds for disclosing information, dramatic leader reveal, fixed organization existence/goals but variable revelation path. |
| **4 — Story checkpoints / information containers** | New relevant information after knowledge thresholds are reached; clues in unvisited spaces, speech, objects, and events; multiple delivery formats; appropriate predetermined areas and kinds of people who can possess clues. |
| **5 — NPC agency / living world** | NPC actions guided by goals and intentions; believable story independent of player intervention; crimes, deaths, bankruptcy, businesses, buildings and other world changes; possibility of explicitly player-centered settings. |
| **6 — Group constraints / orchestration** | Groups have real priorities, budgets, capacities, costs of action and inaction; failure and bankruptcy; director/orchestrator reconciles evolving story with autonomous outcomes and missing/contradictory narrative pieces. |
| **7 — Technology sketch** | DM agent + story controller, NPC agent + NPC controller, shared LLM and TTS; detect whom speech is directed to, infer from orientation if necessary; minimal NPC context containing relevant story/background/interactions; nearby NPCs respond to player and other NPCs. |
| **8 — Prototype and hardware** | 2D simulation; visualize NPC sight and hearing; NPCs acting through LLM and TTS; large predefined suite of simulated player scenarios; optimize bottlenecks and build an agentic testing framework; use Steam Hardware Survey to guide minimum requirements; VRAM concern. |

The notes establish intent, not measured system behavior. Where implementation choices are still open, this document proposes mechanisms without claiming they have already been validated.

---

## 14. Core Design Decisions to Preserve

1. **This is a quest-driven RPG**, with authored and emergent quests arising from believable needs, bargains, obstacles, and player-created goals.
2. **The world engine is authoritative.** Agents select actions through controlled tools; claims in dialogue are not automatically world facts.
3. **NPCs and factions lead independent lives.** The game should not require player proximity to make everything happen.
4. **Strong story anchors coexist with flexible checkpoints.** The director makes constrained proposals and cannot erase committed history.
5. **Information has location and provenance.** Sight, hearing, gossip, lies, secrecy, trust, and blackmail must be mechanically meaningful.
6. **Autonomy is resource-constrained.** Time, money, staffing, risk, and inaction constrain individuals and groups.
7. **Many logical agents share a small number of local models.** Inference is scheduled, prioritized, and measured.
8. **The prototype is an experiment in emergent play.** Success means coherent, genuinely divergent outcomes on commodity hardware—not just convincing conversation demos.

**Next deliverable:** a machine-readable world/event/goal schema plus a vertical-slice test plan covering the mafia/grocer scenario, free-form interaction, sight/hearing, emergent favors, blackmail, director proposals, and reproducible local-inference benchmarks.
