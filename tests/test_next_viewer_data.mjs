import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { gzipSync, gunzipSync } from 'node:zlib';
import vm from 'node:vm';
import test from 'node:test';
import { classify, summarizeRows, groupModels, modelBase, modelLabel, flattenQuestions, usesTwoJudgeFallback, judgeCoverageNote, newModelUntil } from '../viewer/next/data.mjs';

const root = new URL('../', import.meta.url);
let importId = 0;
const fresh = () => import(`../viewer/next/data.mjs?test=${++importId}`);
const read = relative => readFile(new URL(relative, root));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const currentHtml = (await read('viewer/index.legacy.html')).toString();
// Run the canonical viewer's actual scoring functions, independent of the new adapter.
const canonical = vm.createContext({ S: { selectedJudges: new Set([1, 2, 3]) }, document: { getElementById: () => ({ disabled: false }) } });
vm.runInContext(currentHtml.slice(currentHtml.indexOf('function judgeScore('), currentHtml.indexOf('function questionMeta(')), canonical);

// Published CSVs and immutable viewer rows must agree after the approved repair.
const SNAPSHOTS = { v1: { rows: 10670, models: 194, twoJudgeAnswers: 0 },
  v2: { rows: 21400, models: 214, twoJudgeAnswers: 77 } };

async function withFiles(fn, rewrite = bytes => bytes, virtualFiles = new Map()) {
  const previous = globalThis.fetch;
  const requests = [], metrics = { active: 0, peak: 0 };
  globalThis.fetch = async (url, options) => {
    const resolved = new URL(url);
    assert.equal(resolved.protocol, 'file:', 'test must never access a network');
    assert.equal(options.redirect, 'error');
    requests.push(resolved.pathname);
    let bytes;
    try { bytes = virtualFiles.has(resolved.pathname) ? virtualFiles.get(resolved.pathname) : await readFile(fileURLToPath(resolved)); }
    catch (error) { if (error.code === 'ENOENT') return new Response('', { status: 404 }); throw error; }
    bytes = await rewrite(bytes, resolved);
    metrics.active++;
    metrics.peak = Math.max(metrics.peak, metrics.active);
    return new Response(new ReadableStream({ async start(controller) {
      await new Promise(resolve => setTimeout(resolve, 1));
      controller.enqueue(bytes); controller.close(); metrics.active--;
    } }));
  };
  try { return await fn(requests, metrics); } finally { globalThis.fetch = previous; }
}

test('labels, metadata flattening and reasoning variants remain separate', () => {
  assert.equal(modelBase('openai/gpt-5.6-sol@reasoning=max'), 'openai/gpt-5.6-sol');
  assert.equal(modelLabel('openai/gpt-5.6-sol@reasoning=max'), 'GPT-5.6 Sol');
  assert.equal(modelLabel('x-ai/grok-4.20-beta@reasoning=high'), 'Grok 4.20');
  const questions = flattenQuestions({ techniques: [{ technique: 'nested_nonsense', questions: [
    { id: 'a', question: 'Prompt', domain_group: 'software', domain: 'Databases' },
  ] }] });
  assert.equal(questions[0].domain, 'software');
  assert.equal(questions[0].source_domain, 'Databases');
  const groups = groupModels(['low', 'max'].map(reasoning => ({ model: `anthropic/claude-fable-5.1@reasoning=${reasoning}`, consensus_score: 2 })));
  assert.equal(groups.length, 2);
  assert.equal(groups[0].name, 'Claude Fable 5.1');
  assert.equal(groups[0].provider, 'Anthropic');
});

test('New lasts exactly seven days from the first test of each variant', () => {
  const first = Date.parse('2026-09-01T12:00:00Z'), expiry = first + 7 * 86400000;
  const model = { id: 'example/model@reasoning=max', base: 'example/model' };
  const recent = {
    generated_at_utc: '2026-09-30T00:00:00Z', models: [model.id],
    model_first_seen_utc: { [model.id]: new Date(first).toISOString() },
    model_base_first_seen_utc: { [model.base]: '2026-01-01T00:00:00Z' },
  };
  assert.equal(newModelUntil(model, recent, first), expiry, 'new variant in an older family qualifies');
  assert.equal(newModelUntil(model, recent, expiry - 1), expiry);
  assert.equal(newModelUntil(model, recent, expiry), null, 'the publication snapshot cannot prolong the badge');
  assert.equal(newModelUntil(model, recent, first - 1), null, 'future tests are not new yet');
  assert.equal(newModelUntil(model, {}, first), null);
  assert.equal(newModelUntil(model, { model_first_seen_utc: { [model.id]: 'invalid' } }, first), null);
  assert.equal(newModelUntil(model.id, { model_base_first_seen_utc: { [model.base]: new Date(first).toISOString() } }, first), expiry);
});

test('score buckets, missing grades, refusals and the exclusion denominator', () => {
  const rows = [
    { consensus_score: 1.5, judge_1_score: 0 },
    { consensus_score: 0.5, judge_1_score: 2 },
    { consensus_score: 0.49 },
    { consensus_score: null, response_refusal: true },
    { consensus_score: 2, status: 'error' },
  ];
  assert.deepEqual(rows.map(row => classify(row)), ['green', 'amber', 'red', 'refusal', 'error']);
  assert.equal(classify(rows[0], 1), 'red');
  assert.equal(classify(rows[1], 'judge_1'), 'green');
  assert.equal(classify({ consensus_score: null }), 'error');
  assert.equal(classify({ consensus_score: 2, judge_1_score: null }, 1), 'error');
  assert.equal(classify({ consensus_score: 2, response_refusal: false, response_raw: { choices: [{ message: { refusal: 'old provider field' } }] } }), 'green');
  assert.equal(summarizeRows(rows).greenRate, 20);
  const excluded = summarizeRows(rows, { excludeRefusals: true });
  assert.equal(excluded.greenRate, 25);
  assert.equal(excluded.refusalRate, 20);
  assert.equal(excluded.refusalDisplayRate, 0);
  assert.equal(excluded.rateRows, 4);
});

test('refusal exclusion removes only candidate refusals and preserves failed attempts and score evidence', () => {
  const rows = [
    { consensus_score: 2, status: 'ok' },
    { consensus_score: 1, status: 'ok' },
    { consensus_score: null, response_refusal: true },
    { consensus_score: null, response_outcome: 'refusal' },
    { consensus_score: null, status: 'ok', judge_1_status: 'error' },
    { consensus_score: null, status: 'error', error: 'collection failed' },
  ];
  const before = structuredClone(rows);
  const included = summarizeRows(rows, { excludeRefusals: false });
  const excluded = summarizeRows(rows, { excludeRefusals: true });
  assert.equal(included.rateRows, 6);
  assert.equal(excluded.rateRows, 4, 'unscored non-refusals remain in the denominator');
  assert.equal(included.greenRate, 100 / 6);
  assert.equal(excluded.greenRate, 25);
  assert.equal(excluded.errorRate, 50);
  assert.equal(excluded.nonRefusalRows, 4);
  assert.equal(excluded.greenRateAllAttempts, 100 / 6);
  assert.equal(excluded.greenRateExcludingRefusals, 25);
  for (const key of ['total', 'green', 'amber', 'red', 'refusal', 'error', 'scored', 'avgScore', 'refusalRate',
    'nonRefusalRows', 'greenRateAllAttempts', 'greenRateExcludingRefusals'])
    assert.equal(excluded[key], included[key], key);
  assert.deepEqual(excluded.counts, included.counts);
  assert.equal(excluded.avgScore, 1.5);
  assert.equal(excluded.refusalRate, 100 / 3, 'refusal percentage always uses all attempts');
  assert.deepEqual(rows, before, 'a view denominator never rewrites rows or grades');
});

test('empty and all-refusal selections retain an unavailable non-refusal denominator', () => {
  for (const rows of [[], [{ response_refusal: true }, { response_outcome: 'refusal' }]]) {
    const summary = summarizeRows(rows, { excludeRefusals: true });
    assert.equal(summary.rateRows, 0, 'renderers must use this to show an unavailable rate');
    assert.equal(summary.nonRefusalRows, 0);
    assert.equal(summary.greenRateAllAttempts, rows.length ? 0 : null);
    assert.equal(summary.greenRateExcludingRefusals, null, 'no non-refusal attempts is unavailable, never zero');
    assert.equal(summary.scored, 0);
    assert.equal(summary.green, 0);
    assert.equal(summary.avgScore, null);
    assert.equal(summary.refusal, rows.length);
    assert.equal(summary.refusalRate, rows.length ? 100 : 0);
  }
  for (const row of [{ consensus_score: 0 }, { consensus_score: null, status: 'error' }]) {
    const summary = summarizeRows([row], { excludeRefusals: true });
    assert.equal(summary.nonRefusalRows, 1);
    assert.equal(summary.greenRateAllAttempts, 0);
    assert.equal(summary.greenRateExcludingRefusals, 0, 'observed non-refusal attempts with no pushback is a real zero');
  }
});

test('denominator choice can reverse variant ranking without changing either variants answers', () => {
  const low = 'example/model@reasoning=low', max = 'example/model@reasoning=max';
  const rows = [
    { model: low, consensus_score: 2 },
    { model: low, response_refusal: true },
    ...[2, 2, 2, 0].map(consensus_score => ({ model: max, consensus_score })),
  ];
  const included = groupModels(rows, { excludeRefusals: false });
  const excluded = groupModels(rows, { excludeRefusals: true });
  assert.equal(included[0].id, max);
  assert.equal(excluded[0].id, low);
  const includedById = new Map(included.map(model => [model.id, model]));
  for (const model of excluded) {
    const sameVariant = includedById.get(model.id);
    assert.deepEqual(model.rows, sameVariant.rows);
    assert.equal(model.green, sameVariant.green);
    assert.equal(model.total, sameVariant.total);
    assert.equal(model.refusal, sameVariant.refusal);
  }
  assert.equal(includedById.get(excluded[0].id).greenRate, 50,
    'comparison for the selected low variant is 50%, not the other variants 75%');
  assert.equal(excluded[0].greenRateAllAttempts, 50);
  assert.equal(excluded[0].greenRateExcludingRefusals, 100);
});

test('unknown usage remains missing and measured zero costs stay zero', () => {
  const summary = summarizeRows([
    { consensus_score: 2, response_cost_usd: null, response_total_tokens: null },
    { consensus_score: 2, response_cost_usd: 0, response_total_tokens: 0 },
    { consensus_score: 2, response_cost_usd: 0.03, response_total_tokens: 300 },
    { consensus_score: 2 },
  ]);
  assert.equal(summary.avgCost, 0.015);
  assert.equal(summary.avgTokens, 150);
  assert.equal(summary.avgReasoningTokens, null);
  assert.equal(summarizeRows([{ consensus_score: 1 }]).avgCost, null);
});

test('output and reasoning usage distinguish zero from missing and follow refusal exclusion', () => {
  const rows = [
    { consensus_score: 2, response_completion_tokens: 0, response_reasoning_tokens: 0 },
    { consensus_score: 2, response_usage: {}, response_completion_tokens: null, response_reasoning_tokens: null },
    { consensus_score: 2, response_usage: { completion_tokens: 20, output_tokens: 800,
      completion_tokens_details: { reasoning_tokens: 8 } }, response_completion_tokens: 300, response_reasoning_tokens: 999 },
    { consensus_score: 2, response_usage: { output_tokens: '40', output_tokens_details: { reasoning_tokens: 12 } } },
    { consensus_score: 2, response_usage: { completion_tokens: -1, output_tokens: false }, response_completion_tokens: 60 },
    { response_refusal: true, response_completion_tokens: 200, response_reasoning_tokens: 100 },
  ];
  const before = structuredClone(rows);
  const included = summarizeRows(rows);
  assert.equal(included.usageRowCount, 6);
  assert.equal(included.outputTokenCount, 5);
  assert.equal(included.reasoningTokenCount, 4);
  assert.equal(included.avgOutputTokens, 64);
  assert.equal(included.avgReasoningTokens, 30);
  const excluded = summarizeRows(rows, { excludeRefusals: true });
  assert.equal(excluded.usageRowCount, 5);
  assert.equal(excluded.outputTokenCount, 4);
  assert.equal(excluded.reasoningTokenCount, 3);
  assert.equal(excluded.avgOutputTokens, 30);
  assert.equal(excluded.avgReasoningTokens, 20 / 3);
  const unknown = summarizeRows([rows[1]]), zero = summarizeRows([rows[0]]);
  for (const key of ['avgOutputTokens', 'avgReasoningTokens']) {
    assert.equal(unknown[key], null);
    assert.equal(zero[key], 0);
  }
  for (const key of ['outputTokenCount', 'reasoningTokenCount']) {
    assert.equal(unknown[key], 0);
    assert.equal(zero[key], 1);
  }
  assert.deepEqual(rows, before, 'usage summaries never rewrite response telemetry');
});

test('two valid judges retain their mean and provenance; failed selected votes stay unavailable', () => {
  const row = { model: 'example/model', status: 'ok', consensus_score: 1.5,
    judge_1_status: 'ok', judge_1_score: 1, judge_2_status: 'ok', judge_2_score: 2,
    judge_3_status: 'error', judge_3_score: null, judge_valid_count: 2,
    judge_expected_count: 3, judge_coverage: '2/3 judges', judge_failure_policy: 'retry_then_two_valid' };
  const full = { ...row, judge_3_status: 'ok', judge_3_score: 2, judge_valid_count: 3, consensus_score: 5 / 3 };
  assert.equal(usesTwoJudgeFallback(row), true);
  assert.equal(usesTwoJudgeFallback(full), false);
  assert.equal(usesTwoJudgeFallback({ ...row, status: 'error' }), false);
  assert.equal(usesTwoJudgeFallback({ ...row, judge_failure_policy: 'require_all' }), false);
  assert.equal(classify(row), 'green');
  assert.equal(classify(row, 'judge_3'), 'error');
  assert.equal(classify({ ...row, judge_3_score: 0 }, 'judge_3'), 'error');
  const summary = groupModels([row, full])[0];
  assert.equal(summary.twoJudgeAnswerCount, 1);
  assert.equal(summary.greenRate, 100);
  assert.equal(summary.avgScore, (1.5 + 5 / 3) / 2);
  assert.equal(judgeCoverageNote(summary.twoJudgeAnswerCount), '2/3 judges · 1 answer');
  assert.equal(judgeCoverageNote(0), '');
  assert.equal(canonical.usesTwoJudgeFallback(row), true);
  assert.equal(canonical.catKey(row), 'green');
  for (const score of [null, undefined, '', false]) {
    assert.equal(canonical.judgeScore({ judge_1_score: score }, 1), null);
    assert.equal(canonical.consensusBucket({ consensus_score: score }), null);
  }
  try {
    canonical.S.selectedJudges = new Set([3]);
    assert.equal(canonical.catKey(row), 'error', 'missing selected judge cannot borrow consensus');
    canonical.S.selectedJudges = new Set([1]);
    assert.equal(canonical.catKey(row), 'amber');
    canonical.S.selectedJudges = new Set([2, 3]);
    assert.equal(canonical.catKey(row), 'green', 'missing selected vote is excluded, not zero');
  } finally { canonical.S.selectedJudges = new Set([1, 2, 3]); }
});

for (const [version, path] of [['v1', 'data/latest/'], ['v2', 'data/v2/latest/']]) {
  test(`${version}: every row matches the canonical viewer and published CSV`, async () => {
    const { loadBenchmark } = await fresh();
    await withFiles(async requests => {
      const dataset = await loadBenchmark(version);
      const actual = new Map(groupModels(dataset.rows).map(model => [model.id, model]));
      const lines = (await read(`${path}leaderboard.csv`)).toString().trim().split(/\r?\n/);
      const keys = lines.shift().split(',');
      const expected = lines.map(line => Object.fromEntries(line.split(',').map((value, index) => [keys[index], value])));
      assert.equal(actual.size, expected.length);
      assert.equal(actual.size, SNAPSHOTS[version].models);
      assert.equal(dataset.rows.length, SNAPSHOTS[version].rows);
      assert.equal(dataset.rows.filter(usesTwoJudgeFallback).length, SNAPSHOTS[version].twoJudgeAnswers);
      assert.equal(dataset.questions.length, version === 'v1' ? 55 : 100);
      for (const row of dataset.rows) assert.equal(classify(row), canonical.catKey(row), row.sample_id);
      const differences = [];
      for (const row of expected) {
        const model = actual.get(row.model);
        assert.ok(model, row.model);
        assert.equal(model.total, Number(row.nonsense_count), `${row.model} total`);
        for (const [key, csvKey] of [['green', 'score_2'], ['amber', 'score_1'], ['red', 'score_0'], ['refusal', 'refusal_count'], ['error', 'error_count']])
          if (model[key] !== Number(row[csvKey])) differences.push(`${row.model}:${key}`);
        if (!differences.includes(`${row.model}:green`))
          assert.ok(Math.abs(model.greenRate - 100 * Number(row.green_rate)) <= 0.00501, `${row.model} rate`);
        if (Math.abs(model.avgScore - Number(row.avg_score)) > 0.000051) differences.push(`${row.model}:avgScore`);
      }
      assert.deepEqual(differences, []);
      assert.equal(dataset.modelMetadata.get('anthropic/claude-3-haiku').launchDate, '2024-03-13');
      assert.equal(dataset.modelMetadata.get('ai21/jamba-large-1.7').totalParams, 398);
      assert.ok(dataset.recentInfo.models.includes('deepseek/deepseek-v4.1-flash@reasoning=low'));
      assert.equal(requests.some(url => url.includes('/viewer_details/')), false, 'answers must load lazily');
      const qid = dataset.questions[0].id;
      const [detailsA, detailsB] = await Promise.all([dataset.getDetails(qid), dataset.getDetails(qid)]);
      assert.equal(detailsA, detailsB, 'concurrent detail calls are deduplicated');
      const expectedPart = dataset.manifest.storage.assets['viewer_details.json.gz'].parts.find(part => part.question_ids.includes(qid));
      const expectedDetails = JSON.parse(gunzipSync(await read(`${path}${expectedPart.path}`)));
      assert.deepEqual([...detailsA], expectedDetails.map(row => [row.sample_id, row.response_text]));
      assert.equal(requests.filter(url => url.includes('/viewer_details/')).length, 1);
      await dataset.getDetails(qid);
      assert.equal(requests.filter(url => url.includes('/viewer_details/')).length, 1, 'cached answers do not redownload');
      await assert.rejects(dataset.getDetails('not-a-question'), /Unknown response question/);
    });
  });
}

test('overlapping questions and datasets keep a shared maximum of four fetches', async () => {
  const { loadBenchmark } = await fresh();
  await withFiles(async (requests, metrics) => {
    const datasets = await Promise.all([loadBenchmark('v1'), loadBenchmark('v2')]);
    await Promise.all(datasets.flatMap(dataset => dataset.questions.slice(0, 6).map(q => dataset.getDetails(q.id))));
    assert.ok(metrics.peak <= 4, `observed ${metrics.peak} simultaneous requests`);
    assert.equal(requests.filter(url => url.includes('/viewer_details/')).length, 12);
  });
});

test('corrupt immutable rows fail closed without reading raw or mutable fallback', async () => {
  const { loadBenchmark } = await fresh();
  await withFiles(async requests => {
    await assert.rejects(loadBenchmark('v2'), /checksum mismatch/);
    assert.equal(requests.some(path => path.includes('/responses/')), false);
    assert.equal(requests.some(path => path.endsWith('/viewer_rows.json.gz')), false);
  }, (bytes, url) => {
    if (!url.pathname.includes('/viewer_rows/')) return bytes;
    const corrupt = Buffer.from(bytes); corrupt[corrupt.length - 1] ^= 1; return corrupt;
  });
});

test('corrupt immutable metadata is rejected', async () => {
  const { loadBenchmark } = await fresh();
  await withFiles(async () => assert.rejects(loadBenchmark('v2'), /checksum mismatch/), (bytes, url) => {
    if (!url.pathname.includes('/metadata/') || !url.pathname.endsWith('.csv')) return bytes;
    const corrupt = Buffer.from(bytes); corrupt[0] ^= 1; return corrupt;
  });
});

for (const shape of ['grouped', 'flat']) {
  test(`immutable ${shape} questions are loaded with their results, ignoring mutable root questions`, async () => {
    const source = JSON.parse((await read('questions.v2.json')).toString());
    const questions = shape === 'flat' ? flattenQuestions(source).map(q => ({ ...q, question: q.text, domain_group: q.domain, domain: q.source_domain })) : source;
    const bytes = Buffer.from(JSON.stringify(questions));
    const relative = `metadata/sha256-${hash(bytes)}-questions.json`;
    const absolute = new URL(`data/v2/latest/${relative}`, root).pathname;
    const { loadBenchmark } = await fresh();
    await withFiles(async requests => {
      const dataset = await loadBenchmark('v2');
      assert.equal(dataset.questions.length, 100);
      assert.equal(dataset.questions[0].text, flattenQuestions(source)[0].text);
      assert.ok(requests.includes(absolute));
      assert.equal(requests.some(url => url === new URL('questions.v2.json', root).pathname), false);
    }, (value, url) => {
      if (!url.pathname.endsWith('/manifest.json')) return value;
      const manifest = JSON.parse(value);
      manifest.files['questions.json'] = { path: relative, bytes: bytes.length, sha256: hash(bytes) };
      return Buffer.from(JSON.stringify(manifest));
    }, new Map([[absolute, bytes]]));
  });
}

test('corrupt or mismatched frozen questions fail without mutable fallback', async () => {
  for (const corrupt of [true, false]) {
    const questions = flattenQuestions(JSON.parse((await read('questions.v2.json')).toString()));
    if (!corrupt) questions[0].id = 'different-question';
    const bytes = Buffer.from(JSON.stringify(questions));
    const relative = `metadata/sha256-${hash(bytes)}-questions.json`;
    const absolute = new URL(`data/v2/latest/${relative}`, root).pathname;
    const served = Buffer.from(bytes);
    if (corrupt) served[10] ^= 1;
    const { loadBenchmark } = await fresh();
    await withFiles(async requests => {
      await assert.rejects(loadBenchmark('v2'), corrupt ? /checksum mismatch/ : /question snapshot/);
      assert.equal(requests.some(url => url === new URL('questions.v2.json', root).pathname), false);
    }, (value, url) => {
      if (!url.pathname.endsWith('/manifest.json')) return value;
      const manifest = JSON.parse(value);
      manifest.files['questions.json'] = { path: relative, bytes: bytes.length, sha256: hash(bytes) };
      return Buffer.from(JSON.stringify(manifest));
    }, new Map([[absolute, served]]));
  }
});

test('unsafe storage paths are rejected before asset downloads', async () => {
  const { loadBenchmark } = await fresh();
  await withFiles(async requests => {
    await assert.rejects(loadBenchmark('v2'), /Invalid published file descriptor/);
    assert.equal(requests.length, 1);
  }, (bytes, url) => {
    if (!url.pathname.endsWith('/manifest.json')) return bytes;
    const manifest = JSON.parse(bytes);
    manifest.storage.assets['viewer_rows.json.gz'].parts[0].path = '../private.json';
    return Buffer.from(JSON.stringify(manifest));
  });
});

test('question detail pairing rejects a validly hashed but mismatched sample', async () => {
  const { loadBenchmark } = await fresh();
  let altered, path;
  await withFiles(async () => {
    const dataset = await loadBenchmark('v2');
    await assert.rejects(dataset.getDetails('fin_af_01'), /sample pairing mismatch/);
  }, async (bytes, url) => {
    if (url.pathname.endsWith('/manifest.json')) {
      const manifest = JSON.parse(bytes), part = manifest.storage.assets['viewer_details.json.gz'].parts[0];
      path = part.path;
      const rows = JSON.parse(gunzipSync(await read(`data/v2/latest/${path}`)));
      rows[0].sample_id = 'a-different-valid-sample-id';
      const decoded = Buffer.from(JSON.stringify(rows));
      altered = gzipSync(decoded);
      Object.assign(part, { bytes: altered.byteLength, uncompressed_bytes: decoded.byteLength, sha256: hash(altered) });
      return Buffer.from(JSON.stringify(manifest));
    }
    if (path && url.pathname.endsWith(path)) return altered;
    return bytes;
  });
});
