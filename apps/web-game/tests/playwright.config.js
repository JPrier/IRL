// Playwright configuration for IRL's browser build.
//
// The built site (./scripts/web-build.sh → dist/web) is served by
// scripts/web-serve.py under a subpath (/IRL/), the way GitHub Pages serves
// a project site. OpenRouter is mocked at the network layer in the smoke
// suite; the real-provider spec only runs when explicitly enabled.
const { defineConfig, devices } = require("@playwright/test");
const path = require("path");

const root = path.resolve(__dirname, "../../..");
const port = Number(process.env.IRL_WEB_PORT || 8099);
const site = process.env.IRL_WEB_DIR || path.join(root, "dist/web");

module.exports = defineConfig({
  testDir: "./e2e",
  timeout: 120_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  // In CI, failures also become GitHub annotations (readable without
  // downloading artifacts).
  reporter: process.env.CI
    ? [["list"], ["github"], ["html", { open: "never", outputFolder: "playwright-report" }]]
    : [["list"], ["html", { open: "never", outputFolder: "playwright-report" }]],
  outputDir: "test-results",
  use: {
    baseURL: `http://127.0.0.1:${port}/IRL/`,
    viewport: { width: 1280, height: 800 },
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "off",
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 1280, height: 800 },
        launchOptions: {
          // Software WebGL2 in headless/CI environments without a GPU.
          args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
        },
      },
    },
    {
      name: "firefox",
      use: {
        ...devices["Desktop Firefox"],
        viewport: { width: 1280, height: 800 },
        launchOptions: {
          // Headless Firefox on GPU-less CI disables WebGL unless forced.
          firefoxUserPrefs: { "webgl.force-enabled": true, "webgl.disabled": false },
        },
      },
    },
  ],
  webServer: {
    command: `python3 ${path.join(root, "scripts/web-serve.py")} --dir ${site} --port ${port} --base /IRL/`,
    url: `http://127.0.0.1:${port}/IRL/`,
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
});
