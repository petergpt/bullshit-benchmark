"use strict";

const assert = require("node:assert/strict");
const { createHash, webcrypto } = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const { gzipSync } = require("node:zlib");

const viewerPath = path.join(__dirname, "..", "viewer", "index.legacy.html");
const html = fs.readFileSync(viewerPath, "utf8");
const inline = html.match(/<script>([\s\S]*?)<\/script>/)[1];
assert.match(inline, /\ninit\(\);\s*$/);
// Evaluate the actual complete viewer script; suppress only automatic DOM bootstrap.
const viewerScript = new vm.Script(inline.replace(/\ninit\(\);\s*$/, "\n") + [
  "globalThis.viewerTest = { S, currentBenchmarkConfig, validatePublishedStorage, fetchPublishedAsset,",
  "loadDatasetCandidate, loadResponseDetails, scheduleResponseDetailsLoad, loadData, prettyModel, rowIsRefusal, responseDetailHtml, requestResponseDetails, requestElementResponseDetails, updateResponseDetailElements, renderDrilldown, toggleTechniqueExamples, realRenderCompare: renderCompare, modelKey };",
  "renderDomainInsights = renderDifficultyBreakdown = renderCompare = () => {};",
  "refreshJudgeLabels = refreshFilters = renderAll = randomGreenRed = () => {};",
].join("\n"), { filename: viewerPath });

const sha = bytes => createHash("sha256").update(bytes).digest("hex");
const plain = value => JSON.parse(JSON.stringify(value));
const source = base => ({ name: "Fixture dataset", basePath: ".." + base });
const defaultRows = () => Array.from({ length: 9 }, (_, index) => ({
  sample_id: "sample-" + index,
  model: index === 8 ? "openai/chatgpt-4o-latest" : "example/model@reasoning=" + (index % 2 ? "max" : "low"),
  model_reasoning_level: index % 2 ? "max" : "low",
  question_id: "question-" + Math.floor(index / 3),
  response_text: index === 2 ? "" : "Answer café ✓ 😀\nline two: " + index,
  response_refusal: index === 2,
  consensus_score: index === 2 ? null : index % 3,
  numeric_example: 1,
}));

function fixture({ base = "/data/latest", chunkSize = 1, legacy = false, storageVersion = 2, overrides = {} } = {}) {
  const records = defaultRows();
  const responses = overrides.responses || records.map(row => ({ ...row }));
  const aggregate = overrides.aggregate || records.map(({ response_text, ...row }) => row);
  const summary = overrides.summary || aggregate.map(row => ({ ...row }));
  const details = overrides.details || records.map(({ sample_id, response_text }) => ({ sample_id, response_text }));
  const files = new Map();
  const partPaths = new Set();
  const assets = {};
  // Deliberately retain number/Unicode spelling that JSON.parse + stringify changes.
  const serialize = row => JSON.stringify(row).replace(/"numeric_example":1(?=[,}])/g, '"numeric_example":1.0').replace(/café/g, "caf\\u00e9");
  for (const [name, rows, format] of [
    ["responses.jsonl", responses, "jsonl"],
    ["aggregate.jsonl", aggregate, "jsonl"],
    ["viewer_rows.json.gz", summary, "json-gzip"],
    ["viewer_details.json.gz", details, "json-gzip"],
  ]) {
    const groups = [];
    for (let index = 0; index < rows.length; index += chunkSize) groups.push(rows.slice(index, index + chunkSize));
    if (!groups.length) groups.push([]);
    if (name === "viewer_details.json.gz" && !legacy && storageVersion >= 2) {
      groups.length = 0;
      const byId = new Map(summary.map(row => [row?.sample_id, row?.question_id]));
      const byQuestion = new Map();
      for (const row of rows) {
        const qid = byId.get(row?.sample_id) || "question-0";
        if (!byQuestion.has(qid)) byQuestion.set(qid, []);
        byQuestion.get(qid).push(row);
      }
      for (const questionRows of byQuestion.values()) {
        for (let i = 0; i < questionRows.length; i += chunkSize) groups.push(questionRows.slice(i, i + chunkSize));
      }
      if (!groups.length) groups.push([]);
    }
    const parts = groups.map((group, index) => {
      const raw = format === "jsonl"
        ? group.map(serialize).join("\n") + (group.length ? "\n" : "")
        : "[" + group.map(serialize).join(",") + "]";
      const bytes = format === "jsonl" ? Buffer.from(raw) : gzipSync(raw);
      const stem = name.replace(/\.jsonl$|\.json\.gz$/, "");
      const suffix = format === "jsonl" ? ".jsonl" : ".json.gz";
      const partPath = legacy ? name : stem + "/sha256-" + sha(bytes) + suffix;
      files.set(base + "/" + partPath, bytes);
      partPaths.add(base + "/" + partPath);
      return { path: partPath, rows: group.length, bytes: bytes.length, sha256: sha(bytes), uncompressed_bytes: Buffer.byteLength(raw),
        ...(name === "viewer_details.json.gz" && !legacy && storageVersion >= 2
          ? { question_ids: group.length ? [summary.find(row => row?.sample_id === group[0].sample_id)?.question_id || "question-0"] : [] } : {}) };
    });
    const raw = format === "jsonl"
      ? rows.map(serialize).join("\n") + (rows.length ? "\n" : "")
      : "[" + rows.map(serialize).join(",") + "]";
    assets[name] = { format, rows: rows.length, uncompressed_sha256: sha(raw), parts };
  }
  files.set(base + "/collection_stats.json", Buffer.from("{}"));
  files.set(base + "/panel_summary.json", Buffer.from('{"judge_models":["example/judge"]}'));
  files.set(base + "/aggregate_summary.json", Buffer.from("{}"));
  files.set(base + "/recent_additions.json", Buffer.from("{}"));
  files.set(base + "/model_launch_dates.csv", Buffer.from("model_id,launch_date\n"));
  files.set(base + "/model_params.csv", Buffer.from("model_id,open_model_status\n"));
  const questions = { techniques: [{ technique: "fixture", questions: records.map(row => ({ id: row.question_id, question: "Fixture question" })) }] };
  files.set("/questions.json", Buffer.from(JSON.stringify(questions)));
  files.set("/questions.v2.json", Buffer.from(JSON.stringify(questions)));
  return {
    base, files, partPaths, records, questions,
    manifest: {
      generated_at_utc: "2026-01-01T00:00:00Z",
      ...(legacy ? {} : { storage: { version: storageVersion, max_file_bytes: 4096, assets } }),
    },
  };
}

function harness(data, { decompression = true, crypto = true, delay = true } = {}) {
  const calls = [];
  const completedParts = [];
  let activeParts = 0;
  let maxActiveParts = 0;
  const nodes = new Map();
  const documentListeners = {};
  const classList = () => {
    const values = new Set();
    return {
      add(...classes) { classes.forEach(value => values.add(value)); },
      remove(...classes) { classes.forEach(value => values.delete(value)); },
      contains(value) { return values.has(value); },
      toggle(value, force = !values.has(value)) { if (force) values.add(value); else values.delete(value); return force; },
    };
  };
  const node = id => {
    if (!nodes.has(id)) nodes.set(id, {
      textContent: "", innerHTML: "", value: "", listeners: {}, attributes: {}, classList: classList(),
      setAttribute(name, value) { this.attributes[name] = value; },
      getAttribute(name) { return this.attributes[name]; },
      scrollIntoView() {},
      querySelector() { return null; },
      closest() { return null; },
      querySelectorAll() { return []; },
      addEventListener(event, fn) { (this.listeners[event] ||= []).push(fn); },
    });
    return nodes.get(id);
  };
  const context = vm.createContext({
    URL, URLSearchParams, Response, TextEncoder, TextDecoder, Uint8Array, ArrayBuffer,
    DecompressionStream: decompression ? DecompressionStream : undefined,
    crypto: crypto ? webcrypto : undefined,
    window: { location: new URL("https://benchmark.test/viewer/index.legacy.html?benchmark=v1") },
    document: { addEventListener(event, fn) { (documentListeners[event] ||= []).push(fn); }, getElementById: node, querySelectorAll() { return []; } },
    console: { error() {}, warn() {}, log() {} },
    fetch: async (url, options) => {
      const pathname = new URL(url, "https://benchmark.test/viewer/index.legacy.html").pathname;
      calls.push({ path: pathname, url, options });
      const isPart = data.partPaths.has(pathname);
      if (isPart) {
        activeParts += 1;
        maxActiveParts = Math.max(maxActiveParts, activeParts);
      }
      try {
        if (isPart && delay) {
          const index = [...data.partPaths].indexOf(pathname);
          await new Promise(resolve => setTimeout(resolve, index === 0 ? 12 : 1));
        }
        if (pathname === data.base + "/manifest.json") {
          return data.manifest === null ? new Response("", { status: 404 })
            : new Response(typeof data.manifest === "string" ? data.manifest : JSON.stringify(data.manifest));
        }
        const bytes = data.files.get(pathname);
        return bytes === undefined ? new Response("", { status: 404 }) : new Response(bytes);
      } finally {
        if (isPart) {
          activeParts -= 1;
          completedParts.push(pathname);
        }
      }
    },
  });
  viewerScript.runInContext(context);
  return {
    api: context.viewerTest, context, calls, completedParts, node, documentListeners,
    get maxActiveParts() { return maxActiveParts; },
  };
}

test("v1 and v2 summaries load without details; question requests deduplicate and preserve filtered pairing", async t => {
  for (const [version, base] of [["v1", "/data/latest"], ["v2", "/data/v2/latest"]]) await t.test(version, async () => {
    const data = fixture({ base });
    const h = harness(data);
    h.api.S.benchmarkVersion = version;
    await h.api.loadData();
    assert.equal(h.node("dataState").textContent, "8 answers");
    assert.equal(h.api.S.responseDetailsAsset.expectedIds.size, 9);
    assert.equal(h.calls.some(call => call.path.includes("/viewer_details/")), false);
    assert.equal(h.calls.some(call => call.path.includes("/responses/")), false);
    assert.match(h.api.responseDetailHtml(h.api.S.rows[0]), /Loading response/);
    const before = h.calls.length;
    await Promise.all([h.api.loadResponseDetails(["question-0"]), h.api.loadResponseDetails(["question-0"])]);
    const detailCalls = h.calls.slice(before).filter(call => call.path.includes("/viewer_details/"));
    assert.equal(detailCalls.length, 3);
    assert.equal(new Set(detailCalls.map(call => call.path)).size, 3);
    assert.equal(h.api.S.responseDetailQuestions.size, 1);
    assert.equal(h.api.S.responseDetailsLoaded, false);
    assert.equal(h.api.S.rows[0].response_text, data.records[0].response_text);
    assert.equal(h.api.S.rows[3].response_text, "");
    assert.doesNotMatch(h.api.responseDetailHtml(h.api.S.rows[0]), /Loading|unavailable/);
    assert.match(h.api.responseDetailHtml(h.api.S.rows[2]), /No response/);
    await h.api.loadResponseDetails();
    assert.equal(h.api.S.responseDetailsLoaded, true);
    assert.deepEqual(Array.from(h.api.S.rows, row => row.response_text), data.records.slice(0, 8).map(row => row.response_text));
    assert.equal(h.maxActiveParts, 4);
    assert.equal(h.calls.find(call => call.path.endsWith("manifest.json")).options.cache, "no-cache");
    assert.ok(h.calls.filter(call => data.partPaths.has(call.path)).every(call => call.options.cache === "default" && !call.url.includes("?")));
  });
});

test("without DecompressionStream, all plaintext shards are verified and paired", async () => {
  const data = fixture();
  const h = harness(data, { decompression: false });
  const loaded = await h.api.loadDatasetCandidate(source(data.base));
  assert.equal(loaded.mode, "aggregate");
  assert.equal(loaded.responseDetailsAsset, null);
  assert.deepEqual(plain(loaded.responses), data.records);
  assert.equal(loaded.aggregateRows.length, 9);
  assert.equal(h.calls.some(call => call.path.endsWith(".gz")), false);
  assert.equal(h.maxActiveParts, 4);
});

test("canonical answered outcomes survive refusal finish metadata and lazy response text", async t => {
  for (const decompression of [true, false]) await t.test(String(decompression), async () => {
    const responses = defaultRows();
    responses[0] = {
      ...responses[0], response_outcome: "response", response_native_finish_reason: "refusal",
      consensus_score: 2,
    };
    const aggregate = responses.map(({ response_text, ...row }) => row);
    const data = fixture({ overrides: { responses, aggregate, summary: aggregate } });
    const h = harness(data, { decompression, delay: false });
    await h.api.loadData();
    const answered = h.api.S.rows.find(row => row.sample_id === responses[0].sample_id);
    assert.equal(answered.response_refusal, false);
    assert.equal(answered.response_outcome, "response");
    assert.equal(answered.consensus_score, 2);
    assert.equal(h.api.rowIsRefusal(answered), false);
    if (decompression) await h.api.loadResponseDetails();
    assert.equal(answered.response_text, responses[0].response_text);
    assert.equal(h.api.rowIsRefusal(answered), false);
  });
});

test("two-judge coverage survives lazy answers and labels only affected responses", async () => {
  const responses = defaultRows();
  responses[0] = { ...responses[0], status: "ok", consensus_score: 1.5,
    judge_valid_count: 2, judge_expected_count: 3, judge_coverage: "2/3 judges",
    judge_failure_policy: "retry_then_two_valid", judge_3_score: null, judge_3_status: "error" };
  const aggregate = responses.map(({ response_text, ...row }) => row);
  const data = fixture({ overrides: { responses, aggregate, summary: aggregate } });
  const h = harness(data, { delay: false });
  await h.api.loadData();
  const affected = h.api.S.rows.find(row => row.sample_id === responses[0].sample_id);
  const unaffected = h.api.S.rows.find(row => row.sample_id === responses[1].sample_id);
  assert.equal(affected.judge_3_score, null);
  assert.equal(affected.judge_3_status, "error");
  assert.match(h.api.responseDetailHtml(affected), /2\/3 judges/);
  assert.match(h.api.responseDetailHtml(affected), /2\/3 judges · 1 answer/);
  assert.doesNotMatch(h.api.responseDetailHtml(unaffected), /2\/3 judges/);
  await h.api.loadResponseDetails([affected.question_id]);
  assert.match(h.api.responseDetailHtml(affected), /2\/3 judges/);
  assert.match(h.api.responseDetailHtml(affected), /Answer café/);
  assert.equal(affected.judge_3_score, null);
});

test("legacy refusal inference distinguishes absent text from known empty text", () => {
  const { api } = harness(fixture(), { delay: false });
  const nativeRefusal = { response_native_finish_reason: "refusal" };
  assert.equal(api.rowIsRefusal(nativeRefusal), false);
  assert.equal(api.rowIsRefusal({ ...nativeRefusal, response_text: "An actual answer" }), false);
  assert.equal(api.rowIsRefusal({ ...nativeRefusal, response_text: "" }), true);
  assert.equal(api.rowIsRefusal({ ...nativeRefusal, response_text: "[Model returned an empty response.]" }), true);
  assert.equal(api.rowIsRefusal({ response_refusal: true }), true);
  assert.equal(api.rowIsRefusal({ response_outcome: "refusal" }), true);
  assert.equal(api.rowIsRefusal({ ...nativeRefusal, response_refusal: "false" }), false);
  assert.equal(api.rowIsRefusal({ response_raw: { choices: [{ message: { refusal: "Declined" } }] } }), true);
});

test("manifest storage also handles single-file assets and empty datasets", async t => {
  for (const empty of [false, true]) await t.test(String(empty), async () => {
    const overrides = empty ? { responses: [], aggregate: [], summary: [], details: [] } : {};
    const data = fixture({ chunkSize: 100, overrides });
    for (const decompression of [true, false]) {
      const h = harness(data, { decompression, delay: false });
      const loaded = await h.api.loadDatasetCandidate(source(data.base));
      assert.equal(loaded.responses.length, empty ? 0 : 9);
      assert.equal(loaded.aggregateRows.length, empty ? 0 : 9);
      assert.equal(h.calls.some(call => call.path.includes("/part-")), false);
    }
  });
});

test("legacy v2 monoliths load with and without gzip support, without WebCrypto", async t => {
  for (const decompression of [true, false]) await t.test(String(decompression), async () => {
    const data = fixture({ base: "/data/v2/latest", chunkSize: 100, legacy: true });
    const h = harness(data, { decompression, crypto: false, delay: false });
    const loaded = await h.api.loadDatasetCandidate(source(data.base));
    assert.equal(loaded.responses.length, 9);
    assert.equal(loaded.aggregateRows.length, 9);
    assert.equal(loaded.mode, decompression ? "compact_viewer_rows" : "aggregate");
    assert.equal(loaded.responseDetailsUrl.endsWith(".gz?v=2026-01-01T00%3A00%3A00Z"), decompression);
  });
});

test("unsafe paths and unsupported declarations fail before requesting parts", async t => {
  const paths = ["../outside.jsonl", "/outside", "https://elsewhere.test/a", "//elsewhere/a",
    "a\\b", "a/../b", "a/./b", "a//b", "a/%2e%2e/b", "a?x=1", "a#fragment"];
  for (const partPath of paths) await t.test(partPath, async () => {
    const data = fixture();
    data.manifest.storage.assets["responses.jsonl"].parts[0].path = partPath;
    const h = harness(data, { delay: false });
    await assert.rejects(h.api.loadDatasetCandidate(source(data.base)), /Unsafe or duplicate/);
    assert.equal(h.calls.length, 1);
  });
  for (const mutate of [
    data => { data.manifest.storage.version = 99; },
    data => { data.manifest.storage = null; },
    data => { delete data.manifest.storage.assets["aggregate.jsonl"]; },
    data => { data.manifest.storage.assets["responses.jsonl"].format = "json-gzip"; },
    data => { data.manifest.storage.assets["responses.jsonl"].parts[1].path = data.manifest.storage.assets["responses.jsonl"].parts[0].path; },
    data => { data.manifest.storage.max_file_bytes = 1; },
  ]) {
    const data = fixture();
    mutate(data);
    const h = harness(data, { delay: false });
    await assert.rejects(h.api.loadDatasetCandidate(source(data.base)));
    assert.equal(h.calls.length, 1);
  }
});

test("new storage requires WebCrypto rather than silently trusting declared data", async () => {
  const data = fixture();
  const h = harness(data, { crypto: false });
  await assert.rejects(h.api.loadDatasetCandidate(source(data.base)), /WebCrypto/);
  assert.equal(h.calls.length, 1);
});

test("missing, truncated, checksum and row-count errors never become partial raw datasets", async t => {
  const mutations = {
    missing(data, asset) { data.files.delete(data.base + "/" + asset.parts[3].path); },
    truncated(data, asset) {
      const name = data.base + "/" + asset.parts[3].path;
      data.files.set(name, data.files.get(name).subarray(0, -1));
    },
    oversized(data, asset) {
      const name = data.base + "/" + asset.parts[3].path;
      data.files.set(name, Buffer.concat([data.files.get(name), Buffer.from(" ")]));
    },
    checksum(_data, asset) { asset.parts[3].sha256 = "0".repeat(64); },
    logicalChecksum(_data, asset) { asset.uncompressed_sha256 = "0".repeat(64); },
    partRows(_data, asset) { asset.parts[0].rows += 1; asset.parts[1].rows -= 1; },
    totalRows(_data, asset) { asset.rows += 1; },
  };
  for (const [name, mutate] of Object.entries(mutations)) await t.test(name, async () => {
    const data = fixture();
    mutate(data, data.manifest.storage.assets["aggregate.jsonl"]);
    const h = harness(data, { decompression: false, delay: false });
    await assert.rejects(h.api.loadDatasetCandidate(source(data.base)));
  });
});

test("invalid IDs and mismatched canonical pairings are rejected", async t => {
  for (const [name, mutate] of [
    ["scalar row", rows => { rows[0] = 17; }],
    ["null row", rows => { rows[0] = null; }],
    ["empty ID", rows => { rows[0].sample_id = ""; }],
    ["whitespace ID", rows => { rows[0].sample_id = " sample-0 "; }],
    ["duplicate ID", rows => { rows[1].sample_id = rows[0].sample_id; }],
    ["unpaired ID", rows => { rows[0].sample_id = "unexpected-sample"; }],
  ]) await t.test(name, async () => {
    const aggregate = defaultRows().map(({ response_text, ...row }) => row);
    mutate(aggregate);
    const data = fixture({ overrides: { aggregate } });
    const h = harness(data, { decompression: false, delay: false });
    await assert.rejects(h.api.loadDatasetCandidate(source(data.base)), /sample (ID|pairing)/);
  });
});

test("corrupt compact rows fall back to complete verified canonical rows", async () => {
  const data = fixture();
  const part = data.manifest.storage.assets["viewer_rows.json.gz"].parts[0];
  data.files.set(data.base + "/" + part.path, Buffer.alloc(part.bytes));
  const h = harness(data, { delay: false });
  const loaded = await h.api.loadDatasetCandidate(source(data.base));
  assert.equal(loaded.mode, "aggregate");
  assert.deepEqual(plain(loaded.responses), data.records);
  assert.equal(loaded.aggregateRows.length, 9);
});

test("detail corruption is atomic per question, visible, and retryable without raw bulk downloads", async () => {
  const data = fixture();
  const part = data.manifest.storage.assets["viewer_details.json.gz"].parts[1];
  const filename = data.base + "/" + part.path;
  const original = data.files.get(filename);
  const h = harness(data);
  await h.api.loadData();
  const before = plain(h.api.S.rows);
  data.files.delete(filename);
  await assert.rejects(h.api.loadResponseDetails(["question-0"]), /Failed published part/);
  assert.equal(h.api.S.responseDetailsLoaded, false);
  assert.equal(h.api.S.responseDetailRequests.size, 0);
  assert.deepEqual(plain(h.api.S.rows), before);
  assert.match(h.api.responseDetailHtml(h.api.S.rows[0]), /Response unavailable.*Retry/);
  assert.equal(h.calls.some(call => call.path.includes("/responses/")), false);
  data.files.set(filename, original);
  await h.api.loadResponseDetails(["question-0"], { retry: true });
  assert.equal(h.api.S.rows[0].response_text, data.records[0].response_text);
  assert.equal(h.api.S.responseDetailErrors.size, 0);
  assert.doesNotMatch(h.api.responseDetailHtml(h.api.S.rows[0]), /unavailable|Retry/);
});

test("incorrect detail IDs cannot be attached to another response", async () => {
  const details = defaultRows().map(({ sample_id, response_text }) => ({ sample_id, response_text }));
  details[0].sample_id = "wrong-detail-id";
  const data = fixture({ overrides: { details } });
  const h = harness(data, { delay: false });
  await h.api.loadData();
  await assert.rejects(h.api.loadResponseDetails(["question-0"]), /sample pairing/);
  assert.equal(h.api.S.rows[0].response_text, "");
  assert.equal(h.api.S.responseDetailTexts.size, 0);
});

test("embedded dataset reset prevents stale lazy details from modifying new state", async () => {
  const data = fixture();
  const h = harness(data);
  await h.api.loadData();
  const pending = h.api.loadResponseDetails();
  const row = { ...data.records[0], response_text: "Embedded response" };
  h.context.BSB_V2_EMBED = { responses: [row], aggregateRows: [row], questions: data.questions };
  h.api.S.benchmarkVersion = "v2";
  await h.api.loadData();
  await pending;
  assert.equal(h.api.S.responseDetailsAsset, null);
  assert.equal(h.api.S.responseDetailsUrl, "");
  assert.equal(h.api.S.responseDetailsLoaded, true);
  assert.equal(h.api.S.rows[0].response_text, "Embedded response");
});

test("Fable display labels remain unchanged", () => {
  const h = harness(fixture());
  for (const effort of ["low", "max"]) {
    assert.equal(h.api.prettyModel({ model: "anthropic/claude-fable-5.1", model_reasoning_level: effort }),
      "Claude Fable 5.1 (" + effort[0].toUpperCase() + effort.slice(1) + ")");
  }
});

test("storage v1 remains readable as a verified full-detail dataset", async () => {
  const data = fixture({ storageVersion: 1 });
  const h = harness(data, { delay: false });
  await h.api.loadData();
  await h.api.loadResponseDetails(["question-0"]);
  assert.equal(h.api.S.responseDetailsLoaded, true);
  assert.equal(h.api.S.rows[7].response_text, data.records[7].response_text);
});

test("question descriptor counts and identifiers must agree with compact summary", async t => {
  for (const change of [part => { part.question_ids = ["missing-question"]; }, part => { part.question_ids = []; }, part => { part.question_ids = [" question-0 "]; }]) {
    const data = fixture();
    change(data.manifest.storage.assets["viewer_details.json.gz"].parts[0]);
    const h = harness(data, { delay: false });
    await assert.rejects(h.api.loadDatasetCandidate(source(data.base)), /question/);
  }
});

test("immutable metadata is verified and selected from the manifest snapshot", async () => {
  const data = fixture();
  data.manifest.files = {};
  for (const name of ["collection_stats.json", "panel_summary.json", "aggregate_summary.json", "recent_additions.json", "model_launch_dates.csv", "model_params.csv"]) {
    const bytes = data.files.get(data.base + "/" + name);
    const filename = "metadata/" + sha(bytes) + "-" + name;
    data.files.set(data.base + "/" + filename, bytes);
    data.manifest.files[name] = { path: filename, sha256: sha(bytes), bytes: bytes.length };
    data.files.set(data.base + "/" + name, Buffer.from("wrong current snapshot"));
  }
  const h = harness(data, { delay: false });
  const loaded = await h.api.loadDatasetCandidate(source(data.base));
  assert.deepEqual(plain(loaded.stats), {});
  assert.equal(h.calls.some(call => /latest\/(collection_stats|panel_summary|aggregate_summary|recent_additions)\.json$/.test(call.path)), false);
  const descriptor = data.manifest.files["collection_stats.json"];
  data.files.set(data.base + "/" + descriptor.path, Buffer.alloc(descriptor.bytes));
  await assert.rejects(h.api.loadDatasetCandidate(source(data.base)), /metadata checksum/);
});

function freezeFixtureQuestions(data, flat = false) {
  const questions = [...new Set(data.records.map(row => row.question_id))]
    .map(id => ({ id, question: "Original wording " + id, domain_group: "Original domain", technique: "fixture" }));
  data.manifest.files = {};
  for (const name of ["collection_stats.json", "panel_summary.json", "aggregate_summary.json", "recent_additions.json", "model_launch_dates.csv", "model_params.csv", "questions.json"]) {
    const bytes = name === "questions.json" ? Buffer.from(JSON.stringify(flat ? questions : { techniques: [{ questions }] }))
      : data.files.get(data.base + "/" + name);
    const filename = "metadata/" + sha(bytes) + "-" + name;
    data.files.set(data.base + "/" + filename, bytes);
    data.manifest.files[name] = { path: filename, sha256: sha(bytes), bytes: bytes.length };
  }
  data.files.set("/questions.json", Buffer.from("root questions have changed"));
  return questions;
}

test("question wording and domains use the immutable publication, including flat collection snapshots", async t => {
  for (const flat of [false, true]) await t.test(flat ? "flat" : "grouped", async () => {
    const data = fixture();
    const questions = freezeFixtureQuestions(data, flat);
    const h = harness(data, { delay: false });
    await h.api.loadData();
    assert.equal(h.node("dataState").textContent, "8 answers");
    assert.equal(h.api.S.questionMap.get("question-0").question, questions[0].question);
    assert.equal(h.api.S.questionMap.get("question-0").domain_group, "Original domain");
    assert.equal(h.calls.some(call => call.path === "/questions.json"), false);
  });
});

test("corrupt frozen questions cannot fall back to mutable wording", async () => {
  const data = fixture();
  freezeFixtureQuestions(data);
  const descriptor = data.manifest.files["questions.json"];
  data.files.set(data.base + "/" + descriptor.path, Buffer.alloc(descriptor.bytes));
  const h = harness(data, { delay: false });
  await assert.rejects(h.api.loadDatasetCandidate(source(data.base)), /metadata checksum/);
  assert.equal(h.calls.some(call => call.path === "/questions.json"), false);
});

test("response surfaces request their displayed questions and hydration preserves open panels", async t => {
  for (const surface of ["model", "technique", "domain", "compare"]) await t.test(surface, async () => {
    const data = fixture();
    const h = harness(data, { delay: false });
    await h.api.loadData();
    const row = h.api.S.rows[0];
    const span = h.node("response-span");
    span.attributes = { "data-response-id": row.sample_id, "data-response-limit": "0" };
    h.context.document.querySelectorAll = selector => selector === "[data-response-id]" ? [span] : [];
    let panel;
    if (surface === "model") {
      h.api.S.drilldown = { mk: h.api.modelKey(row), cat: "red" };
      panel = h.node("drilldownPanel");
      panel.classList.add("hidden-block");
      h.api.renderDrilldown([row]);
      assert.equal(panel.classList.contains("hidden-block"), false);
    } else if (surface === "technique") {
      const bar = h.node("technique-bar");
      bar.attributes["data-tech"] = "fixture";
      panel = h.node("techEx_fixture");
      panel.querySelectorAll = () => [span];
      h.api.toggleTechniqueExamples(bar);
      assert.equal(panel.classList.contains("open"), true);
    } else if (surface === "domain") {
      const cell = h.node("heat-cell");
      cell.attributes = { "data-mk": h.api.modelKey(row), "data-domain": "all" };
      h.api.S.filteredRows = [row];
      const handler = h.documentListeners.click.find(fn => String(fn).includes('.heat-cell-click[data-mk][data-domain]'));
      handler({ target: { closest() { return cell; } } });
      panel = h.node("domainDrilldown");
      assert.match(panel.innerHTML, /domain-drilldown/);
    } else {
      h.node("compareQuestion").value = row.question_id;
      h.node("compareViewMode").value = "pair";
      h.api.S.responseCompareActive = true;
      h.api.realRenderCompare(h.api.S.rows);
      panel = h.node("cardA");
      assert.match(panel.innerHTML, /data-response-id/);
    }
    assert.equal(h.api.S.responseDetailRequests.has(row.question_id), true);
    const before = panel.innerHTML;
    await h.api.loadResponseDetails([row.question_id]);
    assert.equal(panel.innerHTML, before);
    assert.match(span.outerHTML, /Answer café/);
    assert.equal(h.api.S.responseDetailQuestions.size, 1);
    assert.equal(h.calls.filter(call => call.path.includes("/viewer_details/")).length, 3);
    if (surface === "technique") assert.equal(panel.classList.contains("open"), true);
  });
});
