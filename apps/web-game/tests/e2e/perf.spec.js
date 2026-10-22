// Performance probe for the browser build. Records numbers rather than
// asserting tight thresholds (CI machines vary); writes perf.json.
//
//   npx playwright test perf.spec.js --project=chromium

const { test, expect } = require("@playwright/test");
const fs = require("fs");
const h = require("./helpers");

test("browser performance profile", async ({ page }, info) => {
  const sizes = {};
  page.on("response", async (r) => {
    const u = new URL(r.url());
    if (u.hostname === "127.0.0.1") {
      const len = Number(r.headers()["content-length"] || 0);
      sizes[u.pathname.split("/").pop() || "index.html"] = len;
    }
  });
  await h.mockOpenRouter(page, {
    chat: async (route) => {
      await new Promise((r) => setTimeout(r, 250)); // a fixed mock "network + model" time
      await route.fulfill({
        status: 200,
        headers: { ...h.CORS, "content-type": "text/event-stream" },
        body: h.sseBody(h.action({ action: "speak", target: "player", text: "Measured." })),
      });
    },
  });
  const t0 = Date.now();
  await h.boot(page);
  const bootMs = Date.now() - t0;
  const timing = await page.evaluate(() => ({ ...window.irlTiming, nav: performance.getEntriesByType("navigation")[0]?.toJSON() }));

  // Frame times over 5 s of idle play.
  const frames = await page.evaluate(() => new Promise((resolve) => {
    const times = [];
    let last = performance.now();
    const t0 = last;
    const f = (now) => {
      times.push(now - last);
      last = now;
      if (now - t0 < 5000) requestAnimationFrame(f);
      else resolve(times);
    };
    requestAnimationFrame(f);
  }));
  frames.sort((a, b) => a - b);
  const pct = (p) => frames[Math.min(frames.length - 1, Math.floor(frames.length * p))];
  const s1 = await h.state(page);

  // Input → world event: a key press until the authoritative position moves.
  await page.locator("#irl-canvas").click();
  const x0 = (await h.state(page)).player.x;
  const tKey = Date.now();
  await page.keyboard.press("d");
  await h.until(page, (s) => s.player.x > x0 + 0.05, { message: "moved" });
  const inputToEventMs = Date.now() - tKey;

  // Save / load round trip through IndexedDB.
  const tSave = Date.now();
  await page.keyboard.press("k");
  await h.until(page, (s) => s.dialogue.some((l) => l.text.includes("saved to browser storage")), { message: "saved" });
  const saveMs = Date.now() - tSave;
  const epoch = (await h.state(page)).world_epoch;
  const tLoad = Date.now();
  await page.keyboard.press("l");
  await h.until(page, (s) => s.world_epoch === epoch + 1, { message: "loaded" });
  const loadMs = Date.now() - tLoad;

  // End-to-end NPC turn with a 250 ms mocked provider.
  await h.connect(page);
  await page.locator("#irl-canvas").click();
  await page.keyboard.press("a"); // walk back towards the grocer
  await h.until(page, (s) => s.nearby, { message: "someone nearby", timeout: 20_000 });
  const tTurn = Date.now();
  await h.say(page, "Quick question.");
  const after = await h.until(page, (s) => h.npcLines(s).includes("Measured."), { message: "answer" });
  const turnMs = Date.now() - tTurn;

  const memory = await page.evaluate(() => (performance.memory ? {
    usedJSHeapMB: performance.memory.usedJSHeapSize / 1048576,
    totalJSHeapMB: performance.memory.totalJSHeapSize / 1048576,
  } : null));
  const wasmMemoryMB = await page.evaluate(() => {
    const e = performance.getEntriesByType("resource").find((r) => r.name.endsWith(".wasm"));
    return e ? { transferMB: e.transferSize / 1048576, decodedMB: e.decodedBodySize / 1048576, fetchMs: e.duration } : null;
  });

  const report = {
    browser: info.project.name,
    download_bytes: sizes,
    wasm: wasmMemoryMB,
    boot_to_first_snapshot_ms: bootMs,
    page_timing_ms: timing,
    fps_mean: 1000 / (frames.reduce((a, b) => a + b, 0) / frames.length),
    frame_ms: { p50: pct(0.5), p95: pct(0.95), p99: pct(0.99), max: frames[frames.length - 1] },
    tps: s1.diagnostics.tps,
    target_tps: s1.diagnostics.target_tps,
    max_tick_lag_ms: s1.diagnostics.max_tick_lag_ms,
    input_to_world_event_ms: inputToEventMs,
    save_ms: saveMs,
    load_ms: loadMs,
    npc_turn_end_to_end_ms: turnMs,
    npc_turn_inference_ms: after.diagnostics.llm_infer_ms,
    npc_turn_first_token_ms: after.diagnostics.llm_first_token_ms,
    mocked_provider_delay_ms: 250,
    memory,
  };
  const text = JSON.stringify(report, null, 2);
  fs.writeFileSync(info.outputPath(`perf-${info.project.name}.json`), text);
  await info.attach("perf", { body: text, contentType: "application/json" });
  console.log(text);
  expect(report.tps).toBeGreaterThan(6);
});
