// Browser smoke tests: the real built site, a real browser, the real WASM
// game. OpenRouter is mocked at the network layer (no paid calls).

const { test, expect } = require("@playwright/test");
const fs = require("fs");
const h = require("./helpers");

test("loads, initialises WebAssembly and renders the town in WebGL2", async ({ page }, info) => {
  const { logs, errors } = await h.boot(page);
  const s = await h.state(page);
  expect(s.diagnostics.platform).toBe("browser");
  expect(s.locations).toBeGreaterThanOrEqual(6);
  const ids = s.actors.map((a) => a.id);
  for (const id of ["player", "grocer", "priest", "guard_captain"]) expect(ids).toContain(id);
  expect(s.engine).toContain("heuristic");

  // The canvas is a real WebGL2 surface with drawn content (not one colour).
  const canvas = page.locator("#irl-canvas");
  const box = await canvas.boundingBox();
  expect(box.width).toBeGreaterThan(400);
  const shot = await canvas.screenshot({ path: info.outputPath("town.png") });
  const distinct = await page.evaluate(async (b64) => {
    const img = new Image();
    img.src = "data:image/png;base64," + b64;
    await img.decode();
    const c = document.createElement("canvas");
    c.width = img.width;
    c.height = img.height;
    const g = c.getContext("2d");
    g.drawImage(img, 0, 0);
    const d = g.getImageData(0, 0, c.width, c.height).data;
    const seen = new Set();
    for (let i = 0; i < d.length; i += 4 * 97) seen.add((d[i] >> 4) << 8 | (d[i + 1] >> 4) << 4 | (d[i + 2] >> 4));
    return seen.size;
  }, shot.toString("base64"));
  expect(distinct, "rendered canvas has varied pixels").toBeGreaterThan(5);
  await info.attach("town", { body: shot, contentType: "image/png" });

  expect(errors, "no uncaught page errors").toEqual([]);
  expect(h.realErrors(logs), "no console errors").toEqual([]);
});

test("the simulation ticks at the fixed rate and the player can walk", async ({ page }) => {
  await h.boot(page);
  const a = await h.state(page);
  await page.waitForTimeout(3000);
  const b = await h.state(page);
  const tps = (b.tick - a.tick) / 3;
  expect(tps).toBeGreaterThan(6.5);
  expect(tps).toBeLessThan(10);

  await page.locator("#irl-canvas").click();
  const x0 = b.player.x;
  await page.keyboard.press("d");
  const moved = await h.until(page, (s) => s.player.x > x0 + 1, { message: "player walked east" });
  expect(moved.player.x).toBeGreaterThan(x0);
});

test("typed dialogue works without credentials (deterministic policy)", async ({ page }) => {
  await h.boot(page);
  // Typed in the game canvas itself, exactly like the desktop client.
  await page.locator("#irl-canvas").click();
  await page.keyboard.press("Enter");
  await page.keyboard.type("Good morning. Business been steady?");
  await page.keyboard.press("Enter");
  const s = await h.until(page, (s) => h.npcLines(s).length > 0, { message: "an NPC reply" });
  expect(s.dialogue.some((l) => l.kind === "player" && l.text.includes("Business been steady"))).toBe(true);
  expect(s.decisions.fallback + s.decisions.model + s.decisions.canonical).toBe(0);
  expect(s.diagnostics.last_decision).toContain("heuristic");
});

test("OpenRouter configuration: connect, model list, wrong key, disconnect", async ({ page }) => {
  const calls = await h.mockOpenRouter(page);
  await h.boot(page);
  await expect(page.locator("#model-hint")).toContainText("structured-output support");
  expect(await page.locator("#models option").count()).toBe(1);

  // A wrong key is rejected with a visible error and nothing changes.
  await page.fill("#api-key", "sk-or-v1-wrong");
  await page.click("#connect");
  await expect(page.locator("#conn-status")).toHaveAttribute("data-state", "error");
  await expect(page.locator("#conn-status")).toContainText("authentication");
  expect((await h.status(page)).provider).toBe("heuristic");

  // A model without structured outputs is refused unless compatibility is chosen.
  await page.fill("#model", "mock/no-structured");
  await page.fill("#api-key", h.TEST_KEY);
  await page.click("#connect");
  await expect(page.locator("#conn-status")).toContainText("structured output unsupported");

  await page.fill("#model", "openai/gpt-6-luna");
  await h.connect(page);
  const st = await h.status(page);
  expect(st.provider).toBe("openrouter");
  expect(st.model).toBe("openai/gpt-6-luna");
  await expect(page.locator("#provider-pill")).toContainText("OpenRouter");
  expect(calls.some((c) => c.path.endsWith("/key"))).toBe(true);

  await page.click("#disconnect");
  await h.until(page, (s) => s.engine.includes("heuristic"), { message: "back to heuristic" });
  expect((await h.status(page)).provider).toBe("heuristic");
});

test("a mocked remote decision is validated and committed by the simulation", async ({ page }) => {
  const calls = await h.mockOpenRouter(page);
  await h.boot(page);
  await h.connect(page);
  await h.say(page, "Lovely evening, isn't it?");
  const s = await h.until(page, (s) => h.npcLines(s).includes("Mock model says good evening."), {
    message: "the model's line",
  });
  expect(s.decisions.model).toBe(1);
  expect(s.diagnostics.inference.completed).toBe(1);
  expect(s.diagnostics.inference.completion_tokens).toBe(31);
  expect(s.diagnostics.inference.reported_cost_usd).toBeCloseTo(0.0000123, 9);
  expect(s.diagnostics.inference.last_remote).toContain("MockAI");
  expect(s.recent_events.some((e) => e.includes("Mock model says good evening."))).toBe(true);

  // The request carried the shared action contract, strictly.
  const post = calls.find((c) => c.path.endsWith("/chat/completions"));
  const body = JSON.parse(post.body);
  expect(body.model).toBe("openai/gpt-6-luna");
  expect(body.response_format.type).toBe("json_schema");
  expect(body.response_format.json_schema.strict).toBe(true);
  expect(body.response_format.json_schema.schema.required).toContain("action");
  expect(body.provider.require_parameters).toBe(true);
  expect(body.stream).toBe(true);
  expect(body.messages[1].content).toContain("says:");
});

test("canonical rules shape remote decisions: the grocer's bargain becomes a real offer", async ({ page }) => {
  await h.mockOpenRouter(page);
  await h.boot(page);
  await h.connect(page);
  await h.say(page, "I need to see your ledger. The one with the dock payments.");
  const s = await h.until(page, (s) => s.offers.some((o) => o.state === "pending" && o.to_player), {
    message: "a pending offer to the player",
  });
  expect(s.decisions.canonical).toBe(1);
  expect(s.offers[0].want.length).toBeGreaterThan(0);
  await page.locator("#irl-canvas").click();
  await page.keyboard.press("y");
  const after = await h.until(page, (s) => s.commitments.length > 0, { message: "a commitment" });
  expect(after.offers.find((o) => o.to_player).state).toBe("accepted");
});

test("an invalid remote proposal never changes the world", async ({ page }) => {
  await h.mockOpenRouter(page, {
    chat: (route) =>
      route.fulfill({
        status: 200,
        headers: { ...h.CORS, "content-type": "text/event-stream" },
        body: h.sseBody('{"action":"speak","text":"hi","teleport_player":"vault"}'),
      }),
  });
  await h.boot(page);
  await h.connect(page);
  await h.say(page, "Hello there.");
  const s = await h.until(page, (s) => s.decisions.fallback === 1, { message: "contract fallback" });
  expect(s.diagnostics.last_decision).toContain("violated the action contract");
  expect(h.npcLines(s)).not.toContain("hi");
});

test("save, load, export and import use the versioned save format", async ({ page }, info) => {
  await h.boot(page);
  await page.locator("#irl-canvas").click();
  await page.keyboard.press("k");
  await h.until(page, (s) => s.dialogue.some((l) => l.text.includes("saved to browser storage")), { message: "saved" });
  const saved = await h.state(page);
  await page.waitForTimeout(2500);
  expect((await h.state(page)).tick).toBeGreaterThan(saved.tick + 10);
  await page.keyboard.press("l");
  const loaded = await h.until(page, (s) => s.world_epoch === saved.world_epoch + 1, { message: "loaded" });
  expect(loaded.tick).toBeLessThan(saved.tick + 6);
  expect(loaded.dialogue.some((l) => l.text.includes("loaded from browser storage"))).toBe(true);

  const [download] = await Promise.all([page.waitForEvent("download"), page.click('[data-action="export"]')]);
  const file = info.outputPath("export.json");
  await download.saveAs(file);
  const text = fs.readFileSync(file, "utf8");
  const save = JSON.parse(text);
  expect(save.version).toBe(1);
  expect(save.world).toBeTruthy();

  await page.setInputFiles("#import", file);
  await h.until(page, (s) => s.dialogue.some((l) => l.text.includes("loaded from imported file")), { message: "imported" });

  // A save from an incompatible build is refused with a clear message.
  save.version = 999;
  const bad = info.outputPath("bad.json");
  fs.writeFileSync(bad, JSON.stringify(save));
  await page.setInputFiles("#import", bad);
  await h.until(page, (s) => s.dialogue.some((l) => l.text.includes("unsupported save version 999")), { message: "rejected" });
});

test("the game keeps rendering and ticking during a 10-second remote response", async ({ page }) => {
  let release;
  const gate = new Promise((r) => (release = r));
  await h.mockOpenRouter(page, {
    chat: async (route) => {
      await gate;
      await route.fulfill({
        status: 200,
        headers: { ...h.CORS, "content-type": "text/event-stream" },
        body: h.sseBody(h.action({ action: "speak", target: "player", text: "Sorry, I was thinking." })),
      });
    },
  });
  await h.boot(page);
  await h.connect(page);
  await h.say(page, "Take your time.");
  const a = await h.until(page, (s) => s.stage === "thinking", { message: "thinking" });
  const frames0 = await page.evaluate(() => new Promise((r) => {
    let n = 0;
    const t0 = performance.now();
    const f = () => (performance.now() - t0 < 10_000 ? (n++, requestAnimationFrame(f)) : r(n));
    requestAnimationFrame(f);
  }));
  const b = await h.state(page);
  expect(b.stage).toBe("thinking");
  expect(b.tick - a.tick, "simulation ticks while waiting").toBeGreaterThan(70);
  // Rendering continues (software WebGL in CI is slow, so the bar is low;
  // perf.spec.js records the actual rate).
  expect(frames0, "frames rendered while waiting").toBeGreaterThan(5);
  release();
  const c = await h.until(page, (s) => h.npcLines(s).includes("Sorry, I was thinking."), { message: "late answer" });
  expect(c.diagnostics.inference.last_latency_ms).toBeGreaterThan(9000);
});

test("cancelling aborts the request and a late answer never commits", async ({ page }) => {
  let aborted = false;
  await h.mockOpenRouter(page, {
    chat: async (route) => {
      await new Promise((r) => setTimeout(r, 4000));
      try {
        await route.fulfill({
          status: 200,
          headers: { ...h.CORS, "content-type": "text/event-stream" },
          body: h.sseBody(h.action({ action: "speak", target: "player", text: "This must never be said." })),
        });
      } catch (_) {
        aborted = true;
      }
    },
  });
  await h.boot(page);
  await h.connect(page);
  await h.say(page, "Tell me something.");
  await h.until(page, (s) => s.stage === "thinking", { message: "thinking" });
  await page.click('.say [data-action="cancel"]');
  await h.until(page, (s) => s.stage === "idle", { message: "idle after cancel" });
  await page.waitForTimeout(5000);
  const s = await h.state(page);
  expect(h.npcLines(s)).not.toContain("This must never be said.");
  expect(s.diagnostics.cancelled).toBe(1);
  expect(s.diagnostics.inference.errors + s.diagnostics.stale).toBeGreaterThanOrEqual(1);
  expect(aborted || s.diagnostics.stale >= 1).toBe(true);

  // The next turn works normally.
  await h.say(page, "Never mind.");
  await h.until(page, (s) => s.stage === "thinking", { message: "next turn" });
});

for (const [name, status, body] of [
  ["rate limit", 429, { error: { code: 429, message: "Rate limit exceeded" } }],
  ["provider outage", 502, { error: { code: 502, message: "Provider returned error" } }],
  ["out of credits", 402, { error: { code: 402, message: "Insufficient credits" } }],
]) {
  test(`an API error (${name}) returns the game to a usable state`, async ({ page }) => {
    await h.mockOpenRouter(page, {
      chat: (route) =>
        route.fulfill({
          status,
          headers: { ...h.CORS, "content-type": "application/json", "retry-after": "0" },
          body: JSON.stringify(body),
        }),
    });
    await h.boot(page);
    await h.connect(page);
    await h.say(page, "Hello?");
    const s = await h.until(page, (s) => s.decisions.fallback === 1 && s.stage === "idle", { message: "fallback" });
    expect(s.diagnostics.inference.last_error).toBeTruthy();
    expect(h.npcLines(s).length).toBe(1); // the deterministic policy answered
    await h.say(page, "Anyone?");
    await h.until(page, (s) => s.decisions.fallback === 2, { message: "second turn" });
  });
}

test("the request budget stops remote calls without breaking the game", async ({ page }) => {
  const calls = await h.mockOpenRouter(page);
  await h.boot(page);
  await page.locator("summary", { hasText: "Limits" }).click();
  await page.fill("#max-requests", "1");
  await h.connect(page);
  await h.say(page, "First.");
  await h.until(page, (s) => s.decisions.model === 1, { message: "first remote turn" });
  await h.say(page, "Second.");
  const s = await h.until(page, (s) => s.decisions.fallback === 1, { message: "budget fallback" });
  expect(s.diagnostics.last_decision).toContain("budget");
  expect(calls.filter((c) => c.path.endsWith("/chat/completions")).length).toBe(1);
});

test("the API key never appears in URLs, logs, state, storage or saves", async ({ page }, info) => {
  const calls = await h.mockOpenRouter(page);
  const { logs, urls } = await h.boot(page);
  await h.connect(page);
  await h.say(page, "Hello.");
  await h.until(page, (s) => h.npcLines(s).length > 0, { message: "reply" });
  await page.locator("#irl-canvas").click();
  await page.keyboard.press("k");
  await h.until(page, (s) => s.dialogue.some((l) => l.text.includes("saved to browser storage")), { message: "saved" });
  await page.keyboard.press("F3");
  const [download] = await Promise.all([page.waitForEvent("download"), page.click('[data-action="export"]')]);
  const exported = fs.readFileSync(await download.path(), "utf8");

  const storage = await page.evaluate(async () => {
    const dump = { local: JSON.stringify(localStorage), session: JSON.stringify(sessionStorage), idb: [] };
    const db = await new Promise((res, rej) => {
      const r = indexedDB.open("irl");
      r.onsuccess = () => res(r.result);
      r.onerror = () => rej(r.error);
    });
    const all = await new Promise((res) => {
      const req = db.transaction("saves").objectStore("saves").getAll();
      req.onsuccess = () => res(req.result);
    });
    dump.idb = all;
    return JSON.stringify(dump);
  });
  const evidence = {
    page_url: page.url(),
    urls,
    logs,
    state: JSON.stringify(await h.state(page)),
    status: JSON.stringify(await h.status(page)),
    dom_inputs: await page.evaluate(() => [...document.querySelectorAll("input")].map((i) => i.value).join("|")),
    storage,
    exported,
    page_text: await page.evaluate(() => document.body.innerText),
  };
  for (const [where, text] of Object.entries(evidence)) {
    expect(String(JSON.stringify(text)).includes(h.KEY_MARKER), `key found in ${where}`).toBe(false);
  }
  // It was only ever sent in the Authorization header, to OpenRouter.
  const withKey = calls.filter((c) => JSON.stringify(c.headers).includes(h.KEY_MARKER));
  expect(withKey.length).toBeGreaterThan(0);
  for (const c of withKey) expect(c.headers.authorization).toBe(`Bearer ${h.TEST_KEY}`);
  expect(calls.some((c) => (c.search + (c.body || "")).includes(h.KEY_MARKER))).toBe(false);
  await info.attach("leak-check", { body: JSON.stringify(Object.keys(evidence)), contentType: "application/json" });
});
