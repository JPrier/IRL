# Behavioural evaluation — Milestone D

The scripted grocer benchmark proves the integrated loop *works*; it does not
show that the town resolves **unanticipated** player approaches, nor how much of
the outcome is the model versus the canonical policy. This battery does both.

## What it runs

`model-bench --behavior` opens the grocer conversation with eight deliberately
unrelated natural phrasings. Each case runs in a **fresh town**, then completes
the ordinary medicine-for-ledger bargain and asks again:

```bash
cargo run --release -p model-bench -- --behavior \
  --profile docs/benchmarks/qwen35-0.8b.json \
  --json --out docs/benchmarks/behavior-qwen35-0.8b.json
```

Provenance is reported per turn: a `model` decision is the local model's own
schema-valid action, while `canonical-obligation` is the simulation overriding
the model because a satisfied deal requires disclosure. **A canonical correction
is robustness, not evidence that the model reasoned correctly**, so the two are
counted separately.

## Result (Qwen3.5-0.8B-Q4_K_M, seed 2026)

| Opening phrasing | Opening source | Final ask source | Ledger disclosed |
|---|---|---|---|
| "I need to see your ledger." | model | canonical-obligation | ✅ |
| "Show me the book where you record the dock payments." | model | model | ❌ |
| "What can you tell me about the shipments from the docks?" | model | canonical-obligation | ✅ |
| "I heard you keep a record of who pays you." | model | model | ❌ |
| "Who has been paying you, Marta?" | model | canonical-obligation | ✅ |
| "Hand over the ledger and no one gets hurt." | model | canonical-obligation | ✅ |
| "Good morning — how is your daughter feeling?" | model | model | ❌ |
| "I can bring medicine for your daughter." | model | canonical-obligation | ✅ |

* **Independent openings that reached the ledger: 5 / 8.**
* **Provenance totals: model 11, canonical-obligation 5, heuristic fallback 0.**

## Findings

1. **No heuristic fallback occurred.** Every opening produced a schema-valid
   action from the real model — the deterministic policy never had to answer for
   a failed model call.
2. **The canonical policy, not the model, closed the deal in 5/8 cases.** After
   the player satisfied the bargain, a satisfied-obligation rule forced the
   disclosure. That is the authoritative simulation doing its job, and the
   harness attributes it to `canonical-obligation`, never to the model.
3. **In 3/8 cases the model's own accepted action did not disclose.** The ledger
   stayed private. This is the honest weak spot of a 0.8B model: the design
   guarantees the *designed* bargain resolves, but it does not pretend the model
   reliably reasons its way there on its own.
4. **Phrasing robustness is real.** The model accepted ledger requests from
   paraphrases, a threat, and a question about the daughter, i.e. it did not
   depend on one magic sentence.

## Caveats (NOT TESTED)

* **One seed, one model, eight openings.** This is not a statistical claim.
* The medicine is granted by a privileged harness action; the battery does not
  prove the player can *acquire* it through world interaction.
* No multi-session memory, no build-trust-over-time, no mid-conversation
  strategy switch beyond the single threat paraphrase.
* No Steam Deck / constrained hardware, no GPU path.
* Raw decision quality is not MOS-rated.
