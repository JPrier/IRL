# Architecture

## The one rule

> **Models propose; Rust validates and commits.**

Nothing outside `rpg_sim` may mutate canonical state. The renderer, the director,
NPC prompt builders and the audio listeners are all *clients* of the world.
Every state change is a typed event produced by an accepted action.

## Data flow

```
player text / mic
      │
      ▼
rpg_agents::NpcPolicy ──► InferenceEngine (local LLM or deterministic fallback)
      │                         │
      │                         ▼
      │                   ActionProposal (untrusted)
      ▼                         │
rpg_sim::Simulation::apply ◄────┘
      │
      ├─ validate (range, inventory, knowledge, capability, offer state, …)
      ├─ perception routing (addressee + overhearers)
      ├─ social mutation (relationships, beliefs, commitments, rumours)
      └─ append typed Event(s) to EventLog
                │
                ▼
        World (canonical) + EventLog (replay)
                │
                ├─► rpg_persistence (JSON / SQLite)
                └─► clients (text game, future Bevy renderer)
```

## Crates and responsibilities

| Crate | Owns | Must not own |
|---|---|---|
| `rpg_domain` | IDs, geometry, schemas, actions, ports, deterministic RNG | simulation logic |
| `rpg_perception` | LoS, FOV, hearing, speech routing | beliefs or truth |
| `rpg_social` | relationships, beliefs/provenance, commitments, rumours | action validation |
| `rpg_narrative` | anchors, checkpoints, clue validation/commit | direct world mutation outside `commit_change` |
| `rpg_sim` | clock, action validation, event log, economy | rendering, model I/O |
| `rpg_agents` | context build, policy, scheduler, director proposals | committing state |
| `rpg_speech` | WAV I/O, diagnostic synth, metrics, neural port | gameplay decisions |
| `rpg_persistence` | snapshots, SQLite event store | gameplay rules |
| `rpg_content` | the town and story data | rules |
| `rpg_eval` | simulated players, scenarios, metrics | production rules |

## Determinism

* `DeterministicRng` is a SplitMix64 implemented in `rpg_domain`; no external RNG
  is used by simulation code.
* Proposals are sorted by `(actor, action_kind)` before application.
* The event log has a stable FNV-style checksum used by tests.
* `Simulation::restore` rebuilds from world + log; replay does not re-run
  inference.
* Float positions are `f32`, but every decision uses the same helpers, so
  identical inputs give identical outputs on the same platform.

## Action validation

`Simulation::validate` is a single match over `Action`. It enforces:

* actor exists and is alive;
* targets exist, are alive and are in conversational range (same room + within
  `talk_radius`);
* items/facts/factions/offers exist;
* the actor holds enough items/money;
* the actor believes a fact before disclosing it;
* inner-circle facts require the `inner_circle` flag;
* movement destinations are reachable via A*;
* offers are pending and only the recipient may accept/reject;
* a faction can afford a budget allocation; a responsible faction can afford a
  repair.

Rejections are logged as `ActionRejected` with a typed `RejectReason`, and are
never silently swallowed.

## Offers, commitments and "organic quests"

An `Offer` is a proposal, not a transfer. `AcceptOffer`:

1. verifies both sides actually hold what they promised;
2. transfers items/money;
3. optionally creates a `Commitment` from the offer's condition;
4. optionally discloses a fact.

A `Commitment` stores a `QuestPredicate`. Every `step` evaluates predicates and
marks satisfied commitments. This is how a quest completes from **world state**
rather than from an LLM declaring success.

Supported predicates include item possession, actor alive/dead/at, belief,
fact committed, location repaired, faction payment and arbitrary scenario flags.

## Perception

`PerceptionEngine` uses a Bresenham line for occlusion:

* sight = range ∧ FOV cone ∧ no blocking cells;
* hearing = distance ≤ `loudness.radius × location.transmission`, attenuated per
  wall, with a minimum audible radius;
* `route_speech` resolves an explicit addressee or infers the nearest actor in
  the speaker's FOV, then computes overhearers from the remaining audible actors.

Observers receive `observed_by` on the event; they never receive the global
transcript.

## Narrative controller

`StoryState` holds anchors, checkpoints and the fixed answer. A director
proposal is a `NarrativeChange`; `validate_change` rejects:

* unknown facts/actors/locations;
* re-placing an already-committed fact;
* placing a clue in a location the player already inspected;
* placing a clue with unmet prerequisites;
* using an ineligible container kind or holder;
* promoting an NPC with no established affiliation;
* activating a checkpoint whose prerequisites the player does not know.

Only `commit_change` (or `place_next_clue`) writes the commit.

## Inference port

```rust
pub trait InferenceEngine: Send {
    fn engine_name(&self) -> String;
    fn model_id(&self) -> String;
    fn warm(&mut self) -> Result<(), InferenceError>;
    fn complete(&mut self, req: &InferenceRequest) -> Result<InferenceResponse, InferenceError>;
    fn cancel(&mut self, _id: u64) -> bool { false }
}
```

`ScriptedEngine` and `HeuristicEngine` are deterministic test/baseline
implementations. A real backend must be resident and warmed; spawning a process
per utterance is explicitly forbidden by the design.

## Speech ports

```rust
pub trait SpeechSynthesizer: Send { /* voices, warm, synthesize, cancel */ }
pub trait SpeechRecognizer: Send { /* warm, transcribe */ }
```

`DiagnosticToneSynth` is a non-neural pipeline test device. `MissingNeuralBackend`
returns `MissingAsset` with the expected path. No fake audio is ever presented as
neural TTS.

## Save format

`SaveFile { version, seed, world, events }`, serialized as JSON. The SQLite
store keeps the same event log and periodic world snapshots. `SAVE_VERSION` is
checked on load; schema changes must add an explicit migration.
