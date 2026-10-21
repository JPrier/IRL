# Playtest guide — Hidden Mafia (text vertical slice)

## Run

```bash
cargo run -p game
# optional deterministic seed:
cargo run -p game -- 2026
```

You start in the Market Plaza at 08:00 with 250 coin.

## Commands

| Command | Effect |
|---|---|
| `look` | Describe the current location and who is present. |
| `go <location>` | Walk there (advances simulation time). |
| `talk <npc> <text>` | Speak; the addressed NPC decides and acts. |
| `private <npc>` | Ask to move somewhere quiet (real acoustic privacy). |
| `investigate <id>` | Search a location or person; may reveal a placed clue. |
| `give <npc> <item>` | Hand over an item. |
| `threaten <npc> <demand>` | Threaten; raises fear **and** anger, lowers trust. |
| `status` | Money, inventory, known facts, open obligations. |
| `wait <n>` | Advance simulation ticks. |
| `save [path]` / `load [path]` | Persist / restore the world (default `agentic_rpg_save.json`). |
| `quit` | Exit. |

Location ids: `plaza`, `grocer`, `grocer_backroom`, `warehouse`,
`guard_office`, `alley`, `residence`, `mafia_hall`.

## A suggested first playthrough

```text
go grocer
talk grocer I need to see your ledger.
status
```

The grocer makes a **conditional offer**: bring medicine for her daughter and
she will show you the ledger. This is a pending offer, not a completed trade.

1. Accept the offer (`talk grocer Yes, we have a deal.` — the fallback policy
   will respond, but acceptance is a player-side action; the text client exposes
   it through the offer flow).
2. Find medicine (currently the scenario harness grants it; in the text client
   you can inspect the warehouse and the alley).
3. `give grocer medicine`, then `wait 1` — the commitment is satisfied by the
   world predicate.
4. `talk grocer The medicine is yours. The ledger?` — she now discloses.

Other beats to try:

* `private grocer` with `spy` nearby — check `status`/`look`; the spy stays in
  the shop and does not observe a whisper in the backroom.
* `threaten grocer Show me the ledger.` — watch `status` relationships; fear and
  anger both rise.
* `go guard_office` and talk to `guard_captain`.
* `go warehouse` and `investigate warehouse` to look for a placed clue.
* `save` then `load` — obligations and known facts persist.

## What to look for (and report)

* Does an NPC ever disclose a fact they should not know?
* Does a private conversation still get overheard by someone who should not hear
  it?
* Does the grocer's bargain feel like a real obligation (not just flavor)?
* Does ignoring the plot still let the town change (budgets, damage) without the
  mystery solving itself?
* Does the same sequence produce the same result after `save`/`load`?

## Graphical and voice clients

* `cargo run -p bevy-game -- --model .cache/models/<model>.gguf` — the Bevy
  client. Walk with WASD, press `V` to hold a voice conversation, `Y`/`N` to
  accept/reject offers, `E` to investigate, `F3` for diagnostics, `F5`/`F9` to
  save/load. Requires the pinned speech assets (`voice-demo install`) for voice.
* `cargo run -p voice-play -- once <file.wav> --model .cache/models/<model>.gguf`
  — headless full voice loop over a recorded fixture (WAV → ASR → local LLM →
  authoritative validation → grounded NPC action).

## Known limitations

* Raw small-model decision quality is limited; the authoritative simulation
  enforces knowledge boundaries, satisfied obligations and the designed bargain
  regardless of what the model proposes, and every fallback is reported.
* Neural TTS is whole-utterance (Kokoro), so there is no streaming/barge-in yet.
* Actors are flat coloured rectangles; there is no sprite art or walk animation.
* Combat and a full economy remain out of scope.
* Steam Deck and live-microphone results are NOT TESTED on this host.
