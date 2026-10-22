// IRL browser page: loads the WebAssembly game and wires the page controls
// to its exported functions. The controls only send commands (the same ones
// the keyboard sends) or manage the OpenRouter connection; all game logic
// runs in the Rust simulation.

import init, * as irl from "./pkg/web_game.js";

const $ = (id) => document.getElementById(id);
const DEFAULT_MODEL = "openai/gpt-6-luna";

function setStatus(text, state) {
  const el = $("conn-status");
  el.textContent = text;
  el.dataset.state = state || "";
}

function setPill(state, text) {
  const pill = $("provider-pill");
  pill.dataset.state = state;
  pill.textContent = text;
}

// --- loading with progress ----------------------------------------------

async function fetchWithProgress(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`could not download ${url} (HTTP ${res.status})`);
  const total = Number(res.headers.get("Content-Length")) || 0;
  if (!res.body || !total) return res;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    $("load-bar").style.width = `${Math.min(100, (got / total) * 100).toFixed(1)}%`;
    $("load-text").textContent = `downloading ${(got / 1048576).toFixed(1)} / ${(total / 1048576).toFixed(1)} MB`;
  }
  return new Response(new Blob(chunks), { headers: { "Content-Type": "application/wasm" } });
}

function webgl2Available() {
  try {
    return !!document.createElement("canvas").getContext("webgl2");
  } catch (_) {
    return false;
  }
}

async function start() {
  window.irlTiming = { navigationStart: performance.timeOrigin, scriptStart: performance.now() };
  if (!webgl2Available()) {
    window.irlHost.fatal("WebGL2 is not available in this browser or is disabled.");
    return;
  }
  try {
    const wasmUrl = new URL("./pkg/web_game_bg.wasm", import.meta.url);
    const response = await fetchWithProgress(wasmUrl);
    window.irlTiming.downloaded = performance.now();
    $("load-text").textContent = "starting…";
    window.irlHost.attach(irl);
    try {
      await init({ module_or_path: response });
    } catch (e) {
      // winit escapes its event-loop setup with a control-flow exception on
      // some versions; the game keeps running.
      if (!String(e).includes("Using exceptions for control flow")) throw e;
    }
    window.irlTiming.initialized = performance.now();
  } catch (e) {
    window.irlHost.fatal(String(e && e.stack ? e.stack : e));
    return;
  }
  // Read-only views for diagnostics, smoke tests and performance probes.
  // Neither can return the credential.
  window.irlGame = {
    debugState: () => JSON.parse(irl.irl_debug_state()),
    status: () => JSON.parse(irl.irl_status()),
  };
  waitForFirstFrame();
  wireControls();
  loadModels();
  finishOAuthIfReturning();
  setInterval(refresh, 500);
}

function waitForFirstFrame() {
  const check = () => {
    const s = JSON.parse(irl.irl_debug_state());
    if (s.ready) {
      window.irlTiming.firstSnapshot = performance.now();
      $("loading").hidden = true;
      $("irl-canvas").focus();
    } else {
      requestAnimationFrame(check);
    }
  };
  check();
}

// --- controls -------------------------------------------------------------

function settings() {
  const num = (id) => {
    const v = $(id).value.trim();
    return v === "" ? null : Number(v);
  };
  return JSON.stringify({
    model: $("model").value.trim() || DEFAULT_MODEL,
    max_output_tokens: num("max-tokens"),
    max_requests: num("max-requests"),
    max_cost_usd: num("max-cost"),
    structured: $("structured").value,
    stream: $("stream").checked,
  });
}

async function connectWith(key) {
  setStatus("Checking the key and model…", "busy");
  setPill("connecting", "connecting…");
  try {
    const summary = JSON.parse(await irl.irl_connect(key, settings()));
    const limit = summary.limit_remaining == null ? "no limit set on this key" : `$${summary.limit_remaining.toFixed(4)} left on this key`;
    setStatus(`Connected: ${summary.model} · ${summary.structured} · ${limit}`, "ok");
  } catch (e) {
    setStatus(`Not connected: ${e}`, "error");
  }
  refresh();
}

function wireControls() {
  $("model").value = DEFAULT_MODEL;
  // Not a <form> submit, so browsers do not offer to store the key in a
  // password manager.
  const onConnect = () => {
    const input = $("api-key");
    const key = input.value;
    // The page keeps no copy: the field is cleared before anything else.
    input.value = "";
    if (!key.trim()) {
      setStatus("Paste an OpenRouter API key first.", "error");
      return;
    }
    connectWith(key);
  };
  $("connect").addEventListener("click", onConnect);
  $("api-key").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      ev.preventDefault();
      onConnect();
    }
  });
  $("disconnect").addEventListener("click", () => {
    $("api-key").value = "";
    irl.irl_disconnect();
    setStatus("Disconnected — key cleared from memory; deterministic policy.", "");
    refresh();
  });
  $("oauth").addEventListener("click", () => {
    window.irlHost.oauthStart().catch((e) => setStatus(String(e), "error"));
  });
  $("say-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    irl.irl_say($("say").value);
    $("say").value = "";
  });
  document.querySelectorAll("[data-action]").forEach((b) =>
    b.addEventListener("click", () => {
      irl.irl_action(b.dataset.action);
      if (b.closest(".grid")) $("irl-canvas").focus();
    }),
  );
  $("import").addEventListener("change", async (ev) => {
    const file = ev.target.files && ev.target.files[0];
    if (!file) return;
    irl.irl_import_save(await file.text());
    ev.target.value = "";
  });
  const voice = () => irl.irl_set_voice($("tts").checked, $("asr").checked);
  $("tts").addEventListener("change", voice);
  $("asr").addEventListener("change", voice);
  const support = window.irlHost.speechSupport();
  $("tts").disabled = !support.synthesis;
  $("asr").disabled = !support.recognition;
  if (!support.recognition) {
    $("voice-note").textContent = "This browser has no speech recognition; typing works everywhere. " + $("voice-note").textContent;
  }
}

async function loadModels() {
  try {
    const models = JSON.parse(await irl.irl_models());
    const list = $("models");
    list.replaceChildren(
      ...models.map((m) => {
        const o = document.createElement("option");
        o.value = m.id;
        const price = m.completion_price != null ? ` · $${(m.completion_price * 1e6).toFixed(2)}/M out` : "";
        o.label = `${m.name}${price}`;
        return o;
      }),
    );
    const known = models.some((m) => m.id === DEFAULT_MODEL);
    $("model-hint").textContent = `${models.length} models list structured-output support.` +
      (known ? "" : ` The default ${DEFAULT_MODEL} is not among them — pick another.`);
  } catch (e) {
    $("model-hint").textContent = `Could not load the model list (${e}); you can still type a model slug.`;
  }
}

async function finishOAuthIfReturning() {
  try {
    const key = await window.irlHost.oauthFinish();
    if (key) await connectWith(key);
  } catch (e) {
    setStatus(String(e), "error");
  }
}

// --- live status ------------------------------------------------------------

function money(x) {
  return x == null ? "—" : `$${x.toFixed(5)}`;
}

function refresh() {
  let st;
  let s;
  try {
    st = JSON.parse(irl.irl_status());
    s = JSON.parse(irl.irl_debug_state());
  } catch (_) {
    return;
  }
  if (st.provider === "openrouter") setPill("openrouter", `OpenRouter · ${st.model}`);
  else if (st.provider === "connecting") setPill("connecting", "connecting…");
  else setPill("heuristic", "deterministic policy");
  const u = st.usage || {};
  $("usage").textContent = st.provider === "openrouter" || u.requests
    ? `requests ${u.requests ?? 0} (${u.failed ?? 0} failed)` +
      (st.budget && st.budget.max_requests ? ` of ${st.budget.max_requests}` : "") +
      `\ntokens ${u.prompt_tokens ?? 0} in / ${u.completion_tokens ?? 0} out` +
      `\ncost reported ${money(u.reported_cost_usd)} · estimated ${money(u.estimated_cost_usd)}` +
      (st.budget && st.budget.max_cost_usd ? ` (cap $${st.budget.max_cost_usd})` : "") +
      `\nlast latency ${u.last_latency_ms != null ? Math.round(u.last_latency_ms) + " ms" : "—"}` +
      (u.last_error ? `\nlast error: ${u.last_error}` : "")
    : "";
  if (!s.ready) return;
  const d = s.diagnostics;
  $("diag").textContent =
    `platform ${d.platform}  fps ${s.fps.toFixed(0)}  tps ${d.tps.toFixed(1)}/${d.target_tps.toFixed(1)}\n` +
    `tick ${s.tick}  day ${s.day} ${s.clock}  epoch ${s.world_epoch}\n` +
    `provider ${s.engine} (${s.model})\n` +
    `stage ${s.stage}  pending ${d.pending_inference}  last ${d.llm_infer_ms != null ? Math.round(d.llm_infer_ms) + " ms" : "—"}\n` +
    `decisions ${s.decisions.model} model / ${s.decisions.canonical} canonical / ${s.decisions.fallback} fallback\n` +
    `rejected ${d.rejected_actions}  stale ${d.stale}  cancelled ${d.cancelled}  timeouts ${d.timeouts}\n` +
    `last decision: ${d.last_decision ?? "—"}`;
}

start();
