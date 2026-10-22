// Opt-in, billable: the real browser build against the real OpenRouter API.
//
// Skipped unless IRL_REAL_OPENROUTER=1 and OPENROUTER_API_KEY are set. Use a
// disposable key with a low spending limit. Proves, from a real browser
// origin: CORS preflight + authenticated POST, key validation, model
// selection, a valid structured action through the same canonical
// validation as native, a committed gameplay event, latency and token
// usage — and writes evidence without the key.

const { test, expect } = require("@playwright/test");
const fs = require("fs");
const h = require("./helpers");

const KEY = process.env.OPENROUTER_API_KEY || "";
const MODEL = process.env.IRL_OPENROUTER_MODEL || "openai/gpt-6-luna";

test.skip(process.env.IRL_REAL_OPENROUTER !== "1" || !KEY, "real OpenRouter test is opt-in");

// Traces, screenshots and videos would capture the typed key and the
// Authorization header; this spec records only its own key-free evidence.
// No retries: every attempt is billable.
test.use({ trace: "off", screenshot: "off", video: "off" });
test.describe.configure({ retries: 0 });

test("a real OpenRouter decision changes the shared simulation", async ({ page }, info) => {
  const network = [];
  page.on("requestfinished", async (r) => {
    if (!r.url().startsWith("https://openrouter.ai/")) return;
    const resp = await r.response();
    network.push({ method: r.method(), url: r.url(), status: resp && resp.status(), timing: r.timing() });
  });
  page.on("requestfailed", (r) => {
    if (r.url().startsWith("https://openrouter.ai/")) network.push({ method: r.method(), url: r.url(), failure: r.failure() });
  });
  const { logs } = await h.boot(page);
  await expect(page.locator("#model-hint")).toContainText("structured-output support", { timeout: 30_000 });

  await page.locator("summary", { hasText: "Limits" }).click();
  await page.fill("#max-requests", "4");
  await page.fill("#max-tokens", "300");
  await page.fill("#max-cost", "0.05");
  await page.fill("#model", MODEL);
  await page.fill("#api-key", KEY);
  await page.click("#connect");
  await expect(page.locator("#conn-status")).toHaveAttribute("data-state", "ok", { timeout: 60_000 });

  // Walk away and back so the player is next to the grocer by movement.
  await page.locator("#irl-canvas").click();
  const start = await h.state(page);
  await page.keyboard.press("a");
  await h.until(page, (s) => s.player.x < start.player.x - 1, { message: "walked" });
  await page.keyboard.press("d");
  await h.until(page, (s) => s.nearby === "grocer", { message: "next to the grocer", timeout: 20_000 });

  const before = await h.state(page);
  const t0 = Date.now();
  await h.say(page, "Evening, Marta. How has business been at the shop?");
  const after = await h.until(page, (s) => s.stage === "idle" && s.diagnostics.inference.completed + s.diagnostics.inference.errors >= 1, {
    message: "a real answer", timeout: 90_000,
  });
  const turnMs = Date.now() - t0;
  const inf = after.diagnostics.inference;
  expect(inf.errors, `remote error: ${inf.last_error}`).toBe(0);
  expect(after.decisions.model + after.decisions.canonical).toBeGreaterThanOrEqual(1);
  expect(after.recent_events.length).toBeGreaterThan(0);

  const evidence = {
    model: MODEL,
    origin: new URL(page.url()).origin,
    turn_ms: turnMs,
    decision: after.diagnostics.last_decision,
    npc_lines: h.npcLines(after).slice(-2),
    events_before: before.recent_events.slice(-1),
    events_after: after.recent_events.slice(-3),
    inference: inf,
    first_token_ms: after.diagnostics.llm_first_token_ms,
    fps: after.fps,
    tps: after.diagnostics.tps,
    network: network.map((n) => ({ ...n, url: n.url.split("?")[0] })),
  };
  const text = JSON.stringify(evidence, null, 2);
  expect(text.includes(KEY)).toBe(false);
  for (const l of logs) expect(l.includes(KEY)).toBe(false);
  fs.writeFileSync(info.outputPath("real-openrouter-evidence.json"), text);
  await info.attach("evidence", { body: text, contentType: "application/json" });
  console.log(text);

  await page.click("#disconnect");
  await h.until(page, (s) => s.engine.includes("heuristic"), { message: "disconnected" });
});
