# IRL in the browser (WebAssembly + OpenRouter)

The browser build is the same game as the native client — the same
simulation, NPC policy, content, validation and Bevy presentation — compiled
to `wasm32-unknown-unknown` and rendered with WebGL2. NPC decisions come from
[OpenRouter](https://openrouter.ai/) with the player's own API key, or from the
deterministic policy when no key is given. Nothing has to be installed and no
model weights are downloaded.

The native Windows/Linux game is unchanged: local llama.cpp (GGUF) stays the
default, offline neural speech stays, and no key or network is needed for
local play. OpenRouter is an *additional*, explicitly selected provider on
native too.

## Runtime configurations

| Configuration | Inference | Speech | Platform |
|---|---|---|---|
| Native local (default) | llama.cpp / GGUF (`--model`), or the deterministic policy without one | offline sherpa-onnx (Whisper, Kokoro, Silero) | Windows / Linux |
| Native remote | OpenRouter (`--provider openrouter`) | the same offline sherpa-onnx stack | Windows / Linux |
| Browser | OpenRouter (bring your own key), or deterministic | typed (always); optional browser speech | WebAssembly / WebGL2 |
| Tests | `--provider heuristic`, scripted engines, mocked HTTP | fakes | all |

## Architecture

```text
                     Shared Rust gameplay (one implementation)
        rpg_domain (incl. the action contract) · rpg_sim · rpg_agents
        rpg_content · rpg_social · rpg_perception · rpg_narrative · rpg_eval::Session
        rpg_runtime::SimCore — conversation state machine, correlation ids,
                               staleness, validation + commit, save/load
        irl_client — the Bevy presentation (world, HUD, dialogue, offers,
                     inventory, diagnostics)
                                     │
                ┌────────────────────┴─────────────────────┐
        Native (apps/bevy-game)                     Browser (apps/web-game)
        rpg_runtime::spawn: irl-sim thread          rpg_web_runtime::WebRuntime:
          + mic / ASR / inference / TTS workers       SimCore pumped every frame,
        llama.cpp or OpenRouter (ureq) on the         fixed 120 ms ticks from the
          inference worker                            page clock; OpenRouter as
        cpal + sherpa-onnx speech                     fetch futures; Web Speech /
        JSON files + SQLite                           speechSynthesis; IndexedDB
                │                                               │
          Windows / Linux                                 WASM + WebGL2
```

What is shared and what is not:

* **Shared, unchanged:** the authoritative simulation, the NPC context
  builder (each NPC sees only its own knowledge), the canonical policy
  (agenda, obligations), action validation, commitments, story conditions,
  world mutation, save format, and `SimCore` — the very type the native
  `irl-sim` thread runs. The browser does not duplicate the simulation; it
  calls `SimCore::handle`/`SimCore::tick` from the event loop instead of from
  a thread.
* **Shared presentation:** `crates/client` (`irl_client`) is the former body
  of `apps/bevy-game/src/main.rs`. Both apps use it; only key bindings differ
  (the browser has no quit key or native microphone diagnostics, and adds
  `K`/`L` as save/load because `F5` is the browser's reload key).
* **Platform-specific:** threads vs. futures, native audio vs. browser
  speech, SQLite/JSON files vs. IndexedDB, and the HTTP transport (blocking
  `ureq` on a worker vs. `fetch`).

The browser dependency graph contains no llama.cpp, ONNX runtime, cpal,
SQLite, `ureq`/`rustls` or speech crates. CI checks this with `cargo tree`
(`ci.yml` → `web` → "Browser dependency isolation").

### One action contract, every provider

`rpg_domain::contract` holds the NPC action schema. llama.cpp turns it into a
grammar; OpenRouter receives its *strict* rendering (every property required,
optional ones nullable, `additionalProperties: false`) as

```json
{
  "response_format": {"type": "json_schema",
    "json_schema": {"name": "irl_npc_action", "strict": true, "schema": { … }}},
  "provider": {"require_parameters": true}
}
```

`require_parameters` makes OpenRouter route only to endpoints that support
every parameter sent, so an endpoint without structured outputs is never
silently used. Before connecting, both clients also check the live model
catalog for `structured_outputs` + `response_format` support (and drop `seed`
when the model does not list it).

`strict: true` is treated as a request, not a guarantee. Every answer, from
any backend, goes through the same steps:

1. extract and parse the JSON object;
2. validate it against the contract (`validate_action_json`): known action,
   correct field types, no unknown fields — a violation falls back to the
   deterministic policy and is recorded as such;
3. convert it into the typed `Action`;
4. canonical grounding and policy resolution (the NPC's designed agenda and
   satisfied obligations win over the model);
5. validation against the *current* world (the world kept moving while the
   request ran);
6. commit only through `Session::commit_npc_decision` → `Simulation::apply`.

Contract validation applies to the local model too. Its grammar already
enforces the schema, so this changes nothing in normal operation; but if the
grammar cannot be built (or a profile overrides it), local output with
extra keys is now a recorded fallback rather than silently accepted.

Before connecting, optional parameters the model does not list (`seed`,
`temperature`, `reasoning`) are dropped, and `max_completion_tokens` is used
for models that only accept that name, so `require_parameters` never leaves
no endpoint to route to. An answer cut off by the output-token limit is
reported as truncated.

A remote result can never mutate the world directly, and provenance (model /
canonical agenda / canonical obligation / fallback with reason) is recorded
identically to local inference. A canonical correction is never presented as
model reasoning.

A **compatibility mode** (`json_object`, or prompt-only) exists for models
without structured outputs. It must be selected explicitly, sends no schema
guarantee, and reports responses as *unstructured*; validation is unchanged.

### Asynchrony, staleness and cancellation

Each decision request carries the conversation id, request id and world
epoch. In the browser it is a future with its own `AbortController`;
natively it runs on the inference worker (the HTTP request on a helper
thread, so an interrupt returns immediately). Either way:

* the simulation keeps ticking while the request runs (tested with a
  deliberately delayed 10 s response: ticks and frames continue);
* cancel (`C` / the Cancel button), disconnect, a decision timeout and loading
  a save all abort the request; a late result is rejected as stale and never
  committed or spoken;
* duplicate and out-of-order results are rejected by the same checks;
* streamed output is reported to the core only as a *count* ("answer
  arriving…"); partial, unvalidated JSON is never shown or acted on. A line
  is shown/spoken only after its action is committed.

Streaming (SSE) handles chunk boundaries anywhere (including inside UTF-8
sequences), `\n`/`\r\n`/`\r`, multi-line `data:`, `: OPENROUTER PROCESSING`
keep-alives, the usage chunk, `[DONE]`, errors before the stream (HTTP
status + JSON body), errors mid-stream (`error` object / `finish_reason:
"error"` on a 200), and interrupted streams. Transient failures (429 with a
short `Retry-After`, 502, 503) are retried at most once; long `Retry-After`
values are not waited out; the model is never switched.

## Build and run the browser client

Requirements: the Rust toolchain from `rust-toolchain.toml`, the wasm target,
and `wasm-bindgen-cli` at the exact version in `Cargo.lock`
(currently 0.2.129):

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129 --locked
```

Optional: `wasm-opt` (binaryen) for a smaller release module.

```bash
./scripts/web-dev.sh                  # dev build + local server → http://127.0.0.1:8080/
./scripts/web-dev.sh --release --base /IRL/   # release build served under a subpath
./scripts/web-build.sh                # release build → dist/web (static site)
```

PowerShell: `./scripts/web-dev.ps1 [-Release] [-Port 8080] [-Base /IRL/]` and
`./scripts/web-build.ps1 [-Dev]`.

`dist/web` is a plain static site (`index.html`, `app.js`, `host.js`,
`style.css`, `pkg/web_game.js`, `pkg/web_game_bg.wasm`). All URLs are
relative, so it works at `/` or under any subpath. Serve `.wasm` as
`application/wasm`. No backend is needed: OpenRouter is called directly from
the page.

### Deploy

* **GitHub Pages:** `.github/workflows/pages.yml` builds and deploys on every
  push to `main` (and on demand). Enable Pages with "GitHub Actions" as the
  source. The site works under the project subpath
  (`https://<user>.github.io/IRL/`).
* **Any static HTTPS host:** upload `dist/web`. Pre-compress
  `web_game_bg.wasm` (gzip/brotli) if the host does not compress on the fly.
  The page needs no special headers (no `SharedArrayBuffer`/COOP/COEP).
* **Preview:** every CI run uploads the built site as the `irl-web` artifact.

### Play

1. Open the page and wait for the town to appear.
2. Optional: paste a disposable OpenRouter key, pick a model (the list shows
   models that advertise structured outputs), adjust the limits, and press
   **Connect & test**. This validates the key with `GET /api/v1/key` (free)
   and checks the model in the catalog; it spends nothing.
3. Click the town and walk with `WASD`. Stand next to someone (they turn
   yellow) and press `Enter` to type (or use the box under the town).
4. Offers appear at the bottom right: `Y` accepts, `N` rejects, `Tab` cycles.
   `E` investigates. `K`/`F5` save to browser storage, `L`/`F9` load.
   `F3` shows the full diagnostics overlay.

A complete browser demonstration: walk to the grocer, ask about her ledger;
the remote model answers, the canonical agenda turns it into a conditional
offer (medicine for the ledger), accepting it creates a commitment, and the
offer, commitment and event log change in the HUD and diagnostics.

Without a key the game runs with the deterministic policy — the same
explicit heuristic mode the tests use.

## Credentials and cost

This is a bring-your-own-key build for testing.

* The key is typed (or created by OAuth) in the page, passed once to the
  game, and the field is cleared. It lives only in the WASM module's memory
  for the current tab.
* It is never embedded in the build, never committed, never put in a URL or
  fragment, never written to `localStorage`/`sessionStorage`/IndexedDB, never
  part of a save or snapshot, never logged, never sent anywhere but
  `https://openrouter.ai/api/v1` as an `Authorization: Bearer` header
  (`credentials: "omit"`, `mode: "cors"`; never `no-cors`).
* **Disconnect** cancels outstanding requests and drops the key. The key's
  own buffer is overwritten on drop, but this is best-effort: the page's
  input, the JS string passed to WASM and transient header copies are
  ordinary garbage-collected/heap memory.
* The key field is not inside a form and is marked `autocomplete=
  "new-password"`, so browsers should not offer to save it in a password
  manager.
* The smoke suite checks the page URL, every request URL and body, console
  output, game state, provider status, DOM inputs, all browser storage and an
  exported save for the key.

**A web page cannot keep a secret the way a native process or a server can.**
Browser extensions, injected scripts and developer tools can read page memory
and network requests. Use a disposable key with a low spending limit, revoke
it after testing, and do not use the browser build with a key you cannot
afford to lose.

**OAuth PKCE (optional).** "Sign in with OpenRouter" runs OpenRouter's PKCE
flow (S256, a random 48-byte verifier and random `state`): the page redirects
to `openrouter.ai/auth`, the authorization code is removed from the address
bar as soon as the page loads, `state` and the 10-minute code lifetime are
checked, and the code is exchanged at `POST /api/v1/auth/keys`. The verifier
is kept in `sessionStorage` only across the redirect and deleted afterwards.
The resulting key is handled exactly like a pasted one (memory only). Manual
key entry always remains available.

**Cost protection.** Per session (tab): a request budget (default 40), a cost
cap in USD (default $0.10, applied to the provider-*reported* cost, or the
local estimate when none is reported), and a per-request output-token cap
(default 400). When a budget is exhausted, NPCs fall back to the
deterministic policy and the reason is shown. The panel shows requests,
tokens, **reported** cost and **estimated** cost (from catalog prices)
separately; they are never added together. There is no automatic model
switching and no unbounded retrying.

## Voice in the browser

Typed dialogue is the primary input and always works. Voice is optional and
off until enabled in the panel:

* **Spoken NPC lines** use `speechSynthesis` (local or OS voices). Only a
  committed line is spoken; cancel/barge-in/load stop it immediately.
* **Push-to-talk (`V`)** uses the Web Speech recognition API where available
  (Chrome, Edge, Safari; not Firefox). It may send audio to an online
  recognition service (Chrome's does), so it is **not offline**. Permission
  errors, no-speech and network failures are reported and the turn returns to
  idle.

The browser does not run the native sherpa-onnx/cpal pipeline. The native
game's offline neural voice is unchanged.

## Saves in the browser

`K`/`F5` (or **Save**) writes the canonical versioned `SaveFile` JSON — the
same format as native — to IndexedDB; `L`/`F9` loads it. **Export save**
downloads it as a file and **Import save** loads one (from the browser or the
desktop game), with the same version check as native. Saves contain world
state only, never credentials. If storage is unavailable (private mode,
blocked site data) the game says so and Export/Import still works. Loading
during a request cancels it; its late answer is discarded.

## Diagnostics

The panel and the in-game `F3` overlay show: platform, provider and model,
connection state (never the key), simulation tps and render fps, the pending
request, request latency and time to first streamed token, token usage,
reported vs estimated cost, model / canonical / fallback decision counts,
the last decision's provenance, rejected actions, stale/cancelled/timed-out
responses and the last error.

## Testing

| Layer | Command | Notes |
|---|---|---|
| Native regression | `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`, `cargo run -p scenario-runner -- --seed 2026` | unchanged gate |
| OpenRouter, mocked HTTP | `cargo test -p rpg_openrouter` | local mock server: JSON + SSE, keep-alives, split chunks, usage, 401/402/429/502, `Retry-After`, mid-stream errors, dropped connections, cancel, deadlines, budgets, catalog checks, canonical commit path |
| Browser runtime | `cargo test -p rpg_web_runtime -p rpg_runtime` | scheduler, 10 s outstanding request, cancel/stale, errors, save export/import (no browser needed) |
| WASM compile | `./scripts/web-build.sh` | the real game, not an example |
| Browser smoke | `cd apps/web-game/tests && npm ci && npx playwright test smoke.spec.js` | built site under `/IRL/`, OpenRouter mocked at the network layer |
| Performance | `npx playwright test perf.spec.js` | writes `perf-<browser>.json` |
| Real OpenRouter (billable, opt-in) | `OPENROUTER_API_KEY=… cargo test -p rpg_openrouter --test real_openrouter -- --ignored --nocapture` and `IRL_REAL_OPENROUTER=1 OPENROUTER_API_KEY=… npx playwright test real-openrouter.spec.js` | manual workflow `openrouter-real.yml`, gated by the `openrouter-test` environment; never on PRs |

## Known limitations

* The page must be visible to run at full speed: browsers throttle
  `requestAnimationFrame` in background tabs. On return, at most four
  catch-up ticks run and the rest are counted as *skipped* (the native
  runtime does the same after a stall).
* Browser speech recognition is browser-dependent and may be online; Firefox
  has none. Speech synthesis voices vary by OS.
* WebGPU is not enabled; the build targets WebGL2 for compatibility.
* The deterministic policy and the simulation are identical across
  platforms, but remote models are not deterministic even with `seed`.
* Cancelling a non-streaming request, or a streaming one on a provider
  without stream cancellation, still bills the full generation (OpenRouter
  documents which providers stop on disconnect).
