"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const viewerPath = path.join(__dirname, "..", "viewer", "index.legacy.html");
const html = fs.readFileSync(viewerPath, "utf8");
const inline = html.match(/<script>([\s\S]*?)<\/script>/)[1];
assert.match(inline, /\ninit\(\);\s*$/);
// Run the original viewer itself, without starting its asynchronous data loader.
const script = new vm.Script(inline.replace(/\ninit\(\);\s*$/, "\n") + `
  globalThis.api = { S, summarize, summarizeRows, bindEvents, renderModelBars,
    renderLeaderboard, renderAllModelsScatter, renderLaunchAllOrgs, renderReasoningScatter,
    renderSizeScatterChart };
  renderAll = refreshTopVariantToggle = renderModelToggles = () => {};
`, { filename: viewerPath });

function harness() {
  const nodes = new Map(), listeners = {};
  const node = id => {
    if (!nodes.has(id)) {
      const classes = new Set();
      nodes.set(id, {
        innerHTML: "", textContent: "", value: "", checked: true, disabled: false,
        attributes: {}, listeners: {},
        classList: { toggle(name, enabled) { enabled ? classes.add(name) : classes.delete(name); }, contains(name) { return classes.has(name); } },
        getAttribute(name) { return this.attributes[name]; },
        setAttribute(name, value) { this.attributes[name] = value; },
        querySelectorAll() { return []; },
        addEventListener(name, fn) { (this.listeners[name] ||= []).push(fn); },
      });
    }
    return nodes.get(id);
  };
  const modes = ["include", "exclude"].map(mode => {
    const button = node(mode);
    const tag = html.match(new RegExp(`<button[^>]*data-refusal-mode="${mode}"[^>]*>`))[0];
    button.setAttribute("data-refusal-mode", mode);
    button.setAttribute("aria-pressed", tag.match(/aria-pressed="([^"]+)"/)[1]);
    button.classList.toggle("is-active", /class="[^"]*is-active/.test(tag));
    return button;
  });
  const context = vm.createContext({
    URL, URLSearchParams,
    Element: class Element {},
    window: { location: new URL("https://benchmark.test/viewer/index.legacy.html"), addEventListener() {} },
    document: {
      getElementById: node,
      querySelectorAll(selector) { return selector === "[data-refusal-mode]" ? modes : []; },
      addEventListener(name, fn) { (listeners[name] ||= []).push(fn); },
    },
  });
  script.runInContext(context);
  return { ...context.api, node, modes, listeners, context };
}

const exampleRows = () => [
  { model: "openai/example", consensus_score: 2 },
  { model: "openai/example", consensus_score: 2 },
  { model: "openai/example", response_refusal: true, consensus_score: null },
  { model: "openai/example", status: "error", consensus_score: null },
];

test("original viewer defaults to excluding candidate refusals while retaining errors", () => {
  const h = harness(), rows = exampleRows(), before = JSON.stringify(rows);
  assert.equal(h.S.refusalStatsMode, "exclude");
  assert.equal(h.node("exclude").getAttribute("aria-pressed"), "true");
  assert.equal(h.node("include").getAttribute("aria-pressed"), "false");
  const excluded = h.summarize(rows);
  assert.equal(excluded.rows, 4);
  assert.equal(excluded.rateRows, 3);
  assert.ok(Math.abs(excluded.greenRate - 200 / 3) < 1e-10);
  assert.ok(Math.abs(excluded.errorRate - 100 / 3) < 1e-10);
  assert.equal(excluded.refusalRate, 25);
  assert.equal(excluded.refusalDisplayRate, 0);
  assert.equal(JSON.stringify(rows), before);
});

test("explicit All attempts selection restores the full denominator; reset restores exclusion", () => {
  const h = harness();
  h.bindEvents();
  for (const handler of h.listeners.click) handler({ target: {
    closest(selector) { return selector === "[data-refusal-mode]" ? h.node("include") : null; },
  } });
  assert.equal(h.S.refusalStatsMode, "include");
  assert.equal(h.node("include").getAttribute("aria-pressed"), "true");
  assert.equal(h.node("exclude").classList.contains("is-active"), false);
  const included = h.summarize(exampleRows());
  assert.equal(included.rateRows, 4);
  assert.equal(included.greenRate, 50);
  assert.equal(included.errorRate, 25);
  assert.equal(included.refusalRate, 25);
  assert.equal(included.refusalDisplayRate, 25);
  for (const handler of h.node("clearFiltersBtn").listeners.click) handler();
  assert.equal(h.S.refusalStatsMode, "exclude");
  assert.equal(h.node("exclude").getAttribute("aria-pressed"), "true");
  assert.equal(h.node("include").classList.contains("is-active"), false);
  assert.equal(h.summarize(exampleRows()).rateRows, 3);
});

test("all-refusal groups have no displayed primary rate or plotted point when excluded", () => {
  const h = harness(), rows = [{ model: "openai/example", response_refusal: true }];
  assert.equal(h.summarizeRows(rows).rateRows, 0);
  h.renderModelBars(rows);
  assert.match(h.node("modelBars").innerHTML, /title="No non-refusal attempts">—<\/span>/);
  h.renderLeaderboard(rows);
  assert.match(h.node("lbBody").innerHTML, /<td>—<\/td>\s*<td>—<\/td>\s*<td>—<\/td>/);
  // Supply metadata so missing metadata cannot accidentally explain point exclusion.
  vm.runInContext(`
    launchMetaForRow = () => ({ launch_date: "2026-09-01" });
    paramsMetaForRow = () => ({ total_params_b: 7, active_params_b: 7 });
  `, h.context);
  h.renderAllModelsScatter(rows);
  h.renderLaunchAllOrgs(rows);
  h.renderReasoningScatter(rows);
  h.renderSizeScatterChart(rows, { svgId: "sizeFixture", metaId: "sizeMeta", legendId: "sizeLegend", metricKey: "totalParamsB", metaLabel: "total", emptyText: "No measured rates" });
  for (const id of ["allModelsScatter", "launchAllOrgs", "reasoningScatter", "sizeFixture"]) {
    assert.ok(h.node(id).innerHTML.length > 0);
    assert.doesNotMatch(h.node(id).innerHTML, /<circle\b/);
  }
  h.S.refusalStatsMode = "include";
  assert.equal(h.summarizeRows(rows).rateRows, 1);
  h.renderLeaderboard(rows);
  assert.match(h.node("lbBody").innerHTML, /<td>0\.0%<\/td>/);
});
