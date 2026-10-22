// Shared helpers for the browser smoke tests.

const { expect } = require("@playwright/test");

/** A fake key with an unmistakable marker, used to prove it never leaks. */
const TEST_KEY = "sk-or-v1-IRLTESTKEY-0123456789abcdef0123456789abcdef";
const KEY_MARKER = "IRLTESTKEY";

const MODELS = {
  data: [
    {
      id: "openai/gpt-6-luna",
      name: "Mock Luna (structured)",
      context_length: 128000,
      pricing: { prompt: "0.00000005", completion: "0.00000025" },
      supported_parameters: ["response_format", "structured_outputs", "seed", "max_tokens", "temperature"],
    },
    {
      id: "mock/no-structured",
      name: "Mock plain",
      context_length: 8000,
      pricing: { prompt: "0", completion: "0" },
      supported_parameters: ["max_tokens", "temperature"],
    },
  ],
};

const CORS = {
  "access-control-allow-origin": "*",
  "access-control-allow-headers": "authorization, content-type, x-openrouter-title",
  "access-control-allow-methods": "GET, POST, OPTIONS",
};

/** An SSE body for a streamed completion of `content`, split into chunks. */
function sseBody(content, { usage = { prompt_tokens: 420, completion_tokens: 31, cost: 0.0000123 } } = {}) {
  const parts = [];
  for (let i = 0; i < content.length; i += 12) parts.push(content.slice(i, i + 12));
  const lines = [": OPENROUTER PROCESSING", ""];
  for (const p of parts) {
    lines.push(
      "data: " + JSON.stringify({ id: "gen-mock", model: "openai/gpt-6-luna", provider: "MockAI",
        choices: [{ index: 0, delta: { content: p }, finish_reason: null }] }),
      "",
    );
  }
  lines.push(
    "data: " + JSON.stringify({ id: "gen-mock", choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }),
    "",
    "data: " + JSON.stringify({ id: "gen-mock", choices: [{ index: 0, delta: { content: "" }, finish_reason: "stop" }], usage }),
    "",
    "data: [DONE]",
    "",
  );
  return lines.join("\n");
}

function action(obj) {
  // The strict rendering: every field present, unused ones null.
  const fields = ["target", "text", "fact", "assertive", "give", "want", "note", "terms", "condition", "demand", "loudness"];
  const full = { action: obj.action };
  for (const f of fields) full[f] = obj[f] ?? null;
  return JSON.stringify(full);
}

/**
 * Mock OpenRouter at the network layer (the real https://openrouter.ai URL,
 * so the page needs no override). `chat(route, request, calls)` answers
 * chat completions; returns the list of recorded calls.
 */
async function mockOpenRouter(page, { chat } = {}) {
  const calls = [];
  await page.route("https://openrouter.ai/api/v1/**", async (route) => {
    const req = route.request();
    if (req.method() === "OPTIONS") {
      return route.fulfill({ status: 204, headers: CORS });
    }
    const url = new URL(req.url());
    const call = { method: req.method(), path: url.pathname, search: url.search, headers: req.headers(), body: req.postData() };
    calls.push(call);
    const json = (status, body, extra = {}) =>
      route.fulfill({ status, headers: { ...CORS, "content-type": "application/json", ...extra }, body: JSON.stringify(body) });
    if (url.pathname.endsWith("/models")) return json(200, MODELS);
    if (url.pathname.endsWith("/key")) {
      if (call.headers.authorization !== `Bearer ${TEST_KEY}`) {
        return json(401, { error: { code: 401, message: "No auth credentials found" } });
      }
      return json(200, { data: { label: "irl-smoke", limit: 1, limit_remaining: 0.99, usage: 0.01, is_free_tier: false } });
    }
    if (url.pathname.endsWith("/chat/completions")) {
      if (call.headers.authorization !== `Bearer ${TEST_KEY}`) {
        return json(401, { error: { code: 401, message: "No auth credentials found" } });
      }
      if (chat) return chat(route, req, call);
      return route.fulfill({
        status: 200,
        headers: { ...CORS, "content-type": "text/event-stream" },
        body: sseBody(action({ action: "speak", target: "player", text: "Mock model says good evening." })),
      });
    }
    return json(404, { error: { code: 404, message: "not mocked" } });
  });
  return calls;
}

/** Load the game and collect console output and page errors. */
async function boot(page) {
  const logs = [];
  const errors = [];
  const urls = [];
  page.on("console", (m) => logs.push(`${m.type()}: ${m.text()}`));
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("request", (r) => urls.push(r.url()));
  await page.goto("./");
  // Ready, or the page's fatal overlay (no WebGL2, failed init): fail fast
  // with the page's own message instead of waiting out the timeout.
  await page.waitForFunction(
    () => (window.irlGame && window.irlGame.debugState().ready) || !document.getElementById("fatal").hidden,
    null,
    { timeout: 90_000 },
  );
  const fatal = page.locator("#fatal");
  if (await fatal.isVisible()) {
    throw new Error(`the game reported a fatal error: ${await fatal.locator(".msg").textContent()}`);
  }
  await expect(page.locator("#loading")).toBeHidden();
  return { logs, errors, urls };
}

const state = (page) => page.evaluate(() => window.irlGame.debugState());
const status = (page) => page.evaluate(() => window.irlGame.status());

/** Wait until `pred(state)` holds, polling the game state. */
async function until(page, pred, { timeout = 30_000, message = "condition" } = {}) {
  const start = Date.now();
  let s;
  while (Date.now() - start < timeout) {
    s = await state(page);
    if (pred(s)) return s;
    await page.waitForTimeout(100);
  }
  throw new Error(`timed out waiting for ${message}; last state: ${JSON.stringify(s).slice(0, 2000)}`);
}

async function say(page, text) {
  await page.fill("#say", text);
  await page.press("#say", "Enter");
}

async function connect(page) {
  await page.fill("#api-key", TEST_KEY);
  await page.click("#connect");
  await expect(page.locator("#conn-status")).toHaveAttribute("data-state", "ok", { timeout: 30_000 });
  await expect(page.locator("#api-key")).toHaveValue("");
  await until(page, (s) => s.engine === "openrouter", { message: "engine switched to openrouter" });
}

function npcLines(s) {
  return s.dialogue.filter((l) => l.kind === "npc").map((l) => l.text);
}

/** Console errors that are not expected browser noise. */
function realErrors(logs) {
  return logs.filter(
    (l) =>
      l.startsWith("error:") &&
      // The mocked 401 test provokes this on purpose.
      !l.includes("401") &&
      !l.includes("Failed to load resource"),
  );
}

module.exports = {
  TEST_KEY, KEY_MARKER, MODELS, CORS, sseBody, action, mockOpenRouter, boot, state, status,
  until, say, connect, npcLines, realErrors,
};
