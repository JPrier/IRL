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

## Known limitations

* The text client uses the deterministic fallback policy, not a real LLM, so
  dialogue is short and templated.
* There is no neural voice playback in the shipped binary yet; the speech
  pipeline and metrics are exercised by `model-bench --speech`.
* Combat, animation, a full economy and a Bevy renderer are out of scope for
  this increment.
