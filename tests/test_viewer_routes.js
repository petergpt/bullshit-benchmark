"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const root = path.join(__dirname, "..");

for (const entry of ["index.html", "viewer/index.html", "viewer/index.v2.html"]) {
  test(`${entry} opens the dashboard under a Pages project subpath`, () => {
    const html = fs.readFileSync(path.join(root, entry), "utf8");
    const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
    const fallback = html.match(/http-equiv="refresh" content="0; url=([^"]+)"/)[1];
    const address = new URL(entry, "https://example.github.io/bullshit-benchmark/");
    const expected = "https://example.github.io/bullshit-benchmark/viewer/index.next.html";
    assert.equal(new URL(fallback, address).href, expected);
    for (const state of [
      { search: "", hash: "" },
      { search: "?benchmark=v1&view=explorer&chart=labs&model=example%2Fmodel", hash: "#model" },
      { search: "?version=%20V1%20", hash: "#results" },
    ]) {
      let destination;
      const link = {};
      vm.runInNewContext(script, {
        window: { location: { ...state, replace: target => { destination = target; } } },
        document: { getElementById: () => link },
      });
      assert.equal(new URL(destination, address).href, expected + state.search + state.hash);
    }
  });
}

test("the legacy viewer remains directly reachable from the dashboard", () => {
  const html = fs.readFileSync(path.join(root, "viewer/index.next.html"), "utf8");
  assert.match(html, /href="\.\/index\.legacy\.html"[^>]*>Legacy viewer/);
  assert.ok(fs.existsSync(path.join(root, "viewer/index.legacy.html")));
});

test("dashboard startup retains legacy suite URLs and benchmark precedence", () => {
  const source = fs.readFileSync(path.join(root, "viewer/next/app.mjs"), "utf8");
  const startup = source.slice(source.indexOf("const url = new URL(location.href);"), source.indexOf("let dataset,"));
  for (const [query, expected] of [
    ["", "v2"],
    ["?version=v1", "v1"],
    ["?version=%20V1%20", "v1"],
    ["?benchmark=%20V1%20", "v1"],
    ["?benchmark=v2&version=v1", "v2"],
    ["?benchmark=&version=v1", "v1"],
    ["?version=unknown", "v2"],
  ]) {
    const context = {
      URL,
      location: new URL("https://example.github.io/bullshit-benchmark/viewer/index.next.html" + query),
      sortDefinitions: [{ key: "greenRate", direction: -1 }],
      columnDefinitions: [],
    };
    vm.runInNewContext(startup + "\nglobalThis.version = state.version;", context);
    assert.equal(context.version, expected, query);
  }
});
