// Static-only adapter for index.next.html. No API, build step, or dataset writes.
// Rates are percentages (0–100); parameter counts are billions; absent usage is null.
const ROOT = new URL('../../', import.meta.url);
const DATASETS = new Map();
const FORMATS = { 'responses.jsonl': 'jsonl', 'aggregate.jsonl': 'jsonl',
  'viewer_rows.json.gz': 'json-gzip', 'viewer_details.json.gz': 'json-gzip' };
const encoder = new TextEncoder();
const decoder = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true });
const isHash = value => /^[a-f0-9]{64}$/.test(value || '');
const isPath = value => typeof value === 'string' && /^[A-Za-z0-9][A-Za-z0-9._/-]*$/.test(value)
  && !value.split('/').some(part => !part || part === '.' || part === '..');
const numeric = value => value === null || value === undefined || value === '' || typeof value === 'boolean'
  ? null : Number.isFinite(Number(value)) ? Number(value) : null;
const nonnegative = value => { const n = numeric(value); return n !== null && n >= 0 ? n : null; };
const mean = values => { const known = values.filter(value => value !== null); return known.length ? known.reduce((a, b) => a + b, 0) / known.length : null; };
const title = value => String(value || '').replace(/\b\w/g, letter => letter.toUpperCase());

const PROVIDERS = { anthropic: 'Anthropic', openai: 'OpenAI', google: 'Google', meta: 'Meta',
  'x-ai': 'xAI', mistralai: 'Mistral', deepseek: 'DeepSeek', 'bytedance-seed': 'ByteDance',
  xiaomi: 'Xiaomi', qwen: 'Qwen', moonshotai: 'Moonshot AI', minimax: 'MiniMax',
  baidu: 'Baidu', 'prime-intellect': 'Prime Intellect', 'z-ai': 'Z.AI', stealth: 'Stealth',
  nvidia: 'NVIDIA', ai21: 'AI21', poolside: 'Poolside', cohere: 'Cohere' };
const ALIASES = { 'ox-alpha': 'OX Alpha', 'mistral-large-2512': 'Mistral Large 3',
  'mistral-small-2603': 'Mistral Small 4', 'grok-4.20-beta': 'Grok 4.20',
  'grok-4.20-multi-agent-beta': 'Grok 4.20 Multi-Agent', 'o3': 'o3',
  'o4-mini': 'o4-mini', 'o4mini': 'o4-mini' };
const WORDS = { gpt: 'GPT', oss: 'OSS', ai: 'AI', it: 'IT', glm: 'GLM',
  minimax: 'MiniMax', deepseek: 'DeepSeek', chatgpt: 'ChatGPT', qwen: 'Qwen' };
const orgKey = org => ({ 'meta-llama': 'meta', steath: 'stealth' })[String(org || '').toLowerCase()]
  || String(org || 'unknown').toLowerCase();

export function modelBase(model) {
  return String(typeof model === 'object' ? model?.model_id || model?.model || '' : model || '')
    .replace(/@reasoning=[^@]+$/i, '');
}
// Recency follows the first benchmark test, never the model's release or the
// latest publication date. An exact reasoning variant takes precedence.
export function newModelUntil(model, recentInfo = {}, now = Date.now()) {
  const id = typeof model === 'string' ? model : model?.id;
  const firstSeen = Date.parse(recentInfo.model_first_seen_utc?.[id]
    ?? recentInfo.model_base_first_seen_utc?.[model?.base || modelBase(id)]);
  const until = firstSeen + 7 * 86400000;
  return Number.isFinite(firstSeen) && firstSeen <= now && now < until ? until : null;
}
export function providerLabel(org) { const key = orgKey(org); return PROVIDERS[key] || title(key); }
export function modelLabel(model) {
  const raw = String(typeof model === 'object' ? model?.model_name || modelBase(model).split('/').at(-1) : modelBase(model).split('/').at(-1))
    .replace(/:free$/i, '').replace(/@reasoning=[^@]+$/i, '').replace(/_/g, '-');
  if (ALIASES[raw.toLowerCase()]) return ALIASES[raw.toLowerCase()];
  const words = raw.split('-').filter(Boolean).map(word => WORDS[word.toLowerCase()] || title(word));
  if (words[0] === 'GPT' && /^\d/.test(words[1] || '')) words.splice(0, 2, `GPT-${words[1]}`);
  return words.join(' ') || 'Unknown model';
}
function reasoning(row) {
  const level = String(row.model_reasoning_level || row.model?.match(/@reasoning=([^@]+)$/i)?.[1] || 'none').toLowerCase();
  return level === 'default' ? 'none' : level;
}
function isRefusal(row) {
  const explicit = String(row.response_refusal ?? '').trim().toLowerCase();
  if (explicit === 'false') return false;
  if (explicit === 'true') return true;
  const outcome = String(row.response_outcome || '').toLowerCase();
  if (outcome === 'refusal') return true;
  if (outcome === 'response') return false;
  const noAnswer = typeof row.response_text === 'string'
    && ['', '[Model returned an empty response.]'].includes(row.response_text.trim());
  if (String(row.response_native_finish_reason || '').toLowerCase() === 'refusal' && noAnswer) return true;
  const raw = row.response_raw;
  const choice = raw?.choices?.[0];
  if (choice?.message?.refusal) return true;
  if (raw?.output?.some(item => item.content?.some(content => content.type === 'refusal'))) return true;
  return !!(noAnswer && String(choice?.native_finish_reason || '').toLowerCase() === 'refusal');
}
function scoreFor(row, judge) {
  if (judge === 'consensus') return numeric(row.consensus_score);
  const index = String(judge).replace(/^judge_/, '');
  if (!['1', '2', '3'].includes(index)) return null;
  const status = row[`judge_${index}_status`];
  if (status && status !== 'ok') return null;
  const score = numeric(row[`judge_${index}_score`]);
  return [0, 1, 2, 3].includes(score) ? score : null;
}
export function classify(row, judge = 'consensus') {
  if (isRefusal(row)) return 'refusal';
  if (row.status ? row.status !== 'ok' : !!row.error) return 'error';
  const score = scoreFor(row, judge);
  if (score === null) return 'error';
  const bucket = Math.floor(Math.max(0, Math.min(3, score)) + 0.5);
  return bucket === 2 ? 'green' : bucket === 1 ? 'amber' : 'red';
}
export function usesTwoJudgeFallback(row) {
  return row?.status === 'ok' && row.judge_failure_policy === 'retry_then_two_valid'
    && numeric(row.judge_valid_count) === 2 && numeric(row.judge_expected_count) === 3
    && numeric(row.consensus_score) !== null && !isRefusal(row);
}
export function judgeCoverageNote(count) {
  return count > 0 ? `2/3 judges · ${count} answer${count === 1 ? '' : 's'}` : '';
}
export function summarizeRows(rows, { excludeRefusals = false, judge = 'consensus' } = {}) {
  const counts = { green: 0, amber: 0, red: 0, refusal: 0, error: 0 };
  const scores = [], tokens = [], reasoningTokens = [], outputTokens = [], costs = [];
  for (const row of rows) {
    const category = classify(row, judge);
    counts[category]++;
    if (!['refusal', 'error'].includes(category)) scores.push(scoreFor(row, judge));
    if (excludeRefusals && category === 'refusal') continue;
    const usage = row.response_usage;
    tokens.push(nonnegative(usage?.total_tokens ?? row.response_total_tokens));
    reasoningTokens.push(nonnegative(usage?.completion_tokens_details?.reasoning_tokens
      ?? usage?.output_tokens_details?.reasoning_tokens ?? row.response_reasoning_tokens));
    outputTokens.push(nonnegative(usage?.completion_tokens) ?? nonnegative(usage?.output_tokens)
      ?? nonnegative(row.response_completion_tokens));
    costs.push(nonnegative(usage?.cost ?? row.response_cost_usd));
  }
  const total = rows.length, nonRefusalRows = total - counts.refusal;
  const rateRows = excludeRefusals ? nonRefusalRows : total;
  const pct = (n, d = rateRows) => d ? 100 * n / d : 0;
  return { total, ...counts, counts, rateRows, nonRefusalRows,
    greenRateAllAttempts: total ? 100 * counts.green / total : null,
    greenRateExcludingRefusals: nonRefusalRows ? 100 * counts.green / nonRefusalRows : null,
    twoJudgeAnswerCount: rows.filter(usesTwoJudgeFallback).length,
    scored: counts.green + counts.amber + counts.red,
    greenRate: pct(counts.green), amberRate: pct(counts.amber), redRate: pct(counts.red),
    refusalRate: pct(counts.refusal, total), refusalDisplayRate: excludeRefusals ? 0 : pct(counts.refusal),
    errorRate: pct(counts.error), avgScore: mean(scores), avgTokens: mean(tokens),
    avgReasoningTokens: mean(reasoningTokens), avgOutputTokens: mean(outputTokens),
    reasoningTokenCount: reasoningTokens.filter(value => value !== null).length,
    outputTokenCount: outputTokens.filter(value => value !== null).length,
    usageRowCount: tokens.length, avgCost: mean(costs) };
}
export function groupModels(rows, options = {}) {
  const groups = new Map();
  for (const row of rows) {
    if (!groups.has(row.model)) groups.set(row.model, []);
    groups.get(row.model).push(row);
  }
  return [...groups].map(([id, items]) => {
    const first = items[0], org = orgKey(first.model_org || modelBase(first).split('/')[0]);
    return { id, base: modelBase(first), name: modelLabel(first), org, provider: providerLabel(org),
      reasoning: reasoning(first), rows: items, ...summarizeRows(items, options) };
  }).sort((a, b) => b.greenRate - a.greenRate || a.redRate - b.redRate || a.name.localeCompare(b.name));
}
export function flattenQuestions(payload) {
  const groups = Array.isArray(payload) ? [{ questions: payload }] : payload?.techniques || [{ questions: payload?.questions || [] }];
  return groups.flatMap(group => (group.questions || []).map(q => ({ ...q,
    id: q.id, text: q.question || q.text || '', domain: q.domain_group || q.domain || 'Other',
    source_domain: q.domain || '', technique: q.technique || group.technique || '' })));
}

// One pool bounds network use across the two benchmarks and overlapping answer selections.
let active = 0;
const queue = [];
function limited(fn) {
  return new Promise((resolve, reject) => {
    queue.push({ fn, resolve, reject });
    drain();
  });
}
function drain() {
  while (active < 4 && queue.length) {
    const job = queue.shift();
    active++;
    Promise.resolve().then(job.fn).then(job.resolve, job.reject).finally(() => { active--; drain(); });
  }
}
async function sha(bytes) {
  if (!globalThis.crypto?.subtle) throw new Error('Dataset integrity checks require HTTPS or localhost.');
  return [...new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256', bytes))]
    .map(byte => byte.toString(16).padStart(2, '0')).join('');
}
function checkDescriptor(part, maxBytes = Number.MAX_SAFE_INTEGER) {
  if (!part || !isPath(part.path) || !Number.isSafeInteger(part.bytes) || part.bytes < 0
      || part.bytes > maxBytes || !isHash(part.sha256)) throw new Error('Invalid published file descriptor.');
}
function validateStorage(manifest) {
  if (!manifest || !Object.hasOwn(manifest, 'storage')) return null;
  const storage = manifest.storage;
  if (!storage || ![1, 2].includes(storage.version) || !Number.isSafeInteger(storage.max_file_bytes)
      || storage.max_file_bytes <= 0 || !storage.assets) throw new Error('Unsupported published storage.');
  const paths = new Set();
  let count = null;
  for (const [name, format] of Object.entries(FORMATS)) {
    const asset = storage.assets[name];
    if (!asset || asset.format !== format || !Number.isSafeInteger(asset.rows) || asset.rows < 0
        || !isHash(asset.uncompressed_sha256) || !Array.isArray(asset.parts) || !asset.parts.length)
      throw new Error(`Invalid published asset: ${name}`);
    if (count !== null && count !== asset.rows) throw new Error('Published asset row counts disagree.');
    count = asset.rows;
    let rows = 0;
    for (const part of asset.parts) {
      checkDescriptor(part, storage.max_file_bytes);
      if (paths.has(part.path) || !Number.isSafeInteger(part.rows) || part.rows < 0
          || (part.uncompressed_bytes !== undefined && (!Number.isSafeInteger(part.uncompressed_bytes) || part.uncompressed_bytes < 0)))
        throw new Error('Invalid or duplicate published part.');
      paths.add(part.path);
      rows += part.rows;
      if (storage.version === 2 && name === 'viewer_details.json.gz'
          && (!Array.isArray(part.question_ids) || part.question_ids.length !== (part.rows ? 1 : 0)
            || part.question_ids.some(id => typeof id !== 'string' || !id.trim() || id !== id.trim())))
        throw new Error('Invalid published question index.');
    }
    if (rows !== asset.rows) throw new Error(`Published part counts disagree: ${name}`);
  }
  return storage;
}
async function fetchText(url, optional = false) {
  return limited(async () => {
    const response = await fetch(url, { cache: 'no-cache', redirect: 'error' });
    if (optional && response.status === 404) return '';
    if (!response.ok) throw new Error(`HTTP ${response.status}: ${url}`);
    return response.text();
  });
}
async function verifiedBytes(base, descriptor, suffix = '') {
  checkDescriptor(descriptor);
  const immutable = /(?:^|\/)sha256-[a-f0-9]{64}\./.test(descriptor.path);
  const url = new URL(descriptor.path + (immutable ? '' : suffix), base);
  const bytes = await limited(async () => {
    const response = await fetch(url, { cache: 'default', redirect: 'error' });
    if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}: ${url}`);
    const reader = response.body.getReader(), chunks = [];
    let size = 0;
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        size += value.byteLength;
        if (size > descriptor.bytes) { await reader.cancel(); throw new Error('Published byte size mismatch.'); }
        chunks.push(value);
      }
    } finally { reader.releaseLock(); }
    if (size !== descriptor.bytes) throw new Error('Published byte size mismatch.');
    const result = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) { result.set(chunk, offset); offset += chunk.byteLength; }
    return result;
  });
  if (await sha(bytes) !== descriptor.sha256) throw new Error(`Published checksum mismatch: ${descriptor.path}`);
  return bytes;
}
async function unzip(bytes) {
  if (typeof DecompressionStream !== 'function') throw new Error('This browser needs gzip support to read the dataset.');
  return new Uint8Array(await new Response(new Response(bytes).body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
}
function rowIds(rows, label) {
  const ids = new Set();
  for (const row of rows) {
    if (!row || typeof row.sample_id !== 'string' || !row.sample_id.trim() || row.sample_id !== row.sample_id.trim() || ids.has(row.sample_id))
      throw new Error(`Invalid or duplicate sample ID: ${label}`);
    ids.add(row.sample_id);
  }
  return ids;
}
function pair(rows, expected, label) {
  const ids = rowIds(rows, label);
  if (ids.size !== expected.size || [...ids].some(id => !expected.has(id))) throw new Error(`Published sample pairing mismatch: ${label}`);
}
async function readPart(base, asset, part, suffix) {
  const bytes = await verifiedBytes(base, part, suffix);
  const decoded = asset.format === 'json-gzip' ? await unzip(bytes) : bytes;
  if (part.uncompressed_bytes !== undefined && decoded.byteLength !== part.uncompressed_bytes) throw new Error('Published decoded size mismatch.');
  const text = decoder.decode(decoded);
  if (asset.format === 'json-gzip' && (text[0] !== '[' || text.at(-1) !== ']')) throw new Error('Invalid published array framing.');
  if (asset.format === 'jsonl' && text && !text.endsWith('\n')) throw new Error('Missing published JSONL boundary.');
  const rows = asset.format === 'json-gzip' ? JSON.parse(text) : text.split('\n').filter(Boolean).map(JSON.parse);
  if (!Array.isArray(rows) || rows.length !== part.rows) throw new Error('Published part row count mismatch.');
  rowIds(rows, part.path);
  return { rows, text };
}
async function readAsset(base, asset, suffix) {
  const parts = await Promise.all(asset.parts.map(part => readPart(base, asset, part, suffix)));
  const rows = parts.flatMap(part => part.rows);
  const text = asset.format === 'jsonl' ? parts.map(part => part.text).join('')
    : '[' + parts.filter(part => part.rows.length).map(part => part.text.slice(1, -1)).join(',') + ']';
  if (rows.length !== asset.rows || await sha(encoder.encode(text)) !== asset.uncompressed_sha256)
    throw new Error('Published logical checksum or row count mismatch.');
  rowIds(rows, 'asset');
  return rows;
}
async function metadata(base, manifest, name, json = true) {
  const descriptor = manifest?.files?.[name];
  // A manifest is a snapshot: never substitute a mutable sidecar for a declared file.
  if (manifest?.files && !descriptor) return json ? null : '';
  const text = descriptor ? decoder.decode(await verifiedBytes(base, descriptor)) : await fetchText(new URL(name, base), true);
  return json ? text ? JSON.parse(text) : null : text;
}
function parseCsv(text) {
  const records = [], row = [];
  let field = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === '"') {
      if (quoted && text[i + 1] === '"') { field += '"'; i++; } else quoted = !quoted;
    } else if (!quoted && (ch === ',' || ch === '\n' || ch === '\r')) {
      row.push(field.trim()); field = '';
      if (ch !== ',') { if (row.some(Boolean)) records.push(row.splice(0)); else row.length = 0; if (ch === '\r' && text[i + 1] === '\n') i++; }
    } else field += ch;
  }
  if (field || row.length) { row.push(field.trim()); records.push(row); }
  const headers = records.shift() || [];
  return records.map(cells => Object.fromEntries(headers.map((header, i) => [header, cells[i] || ''])));
}
function buildMetadata(launchText, paramsText) {
  const map = new Map();
  for (const row of parseCsv(launchText)) if (row.model_id) map.set(modelBase(row.model_id), { launchDate: row.launch_date || null,
    launchSource: row.evidence_url || '', totalParams: null, activeParams: null, license: '', openStatus: 'unknown' });
  for (const row of parseCsv(paramsText)) if (row.model_id) {
    const base = modelBase(row.model_id);
    map.set(base, { launchDate: null, ...map.get(base), totalParams: nonnegative(row.total_params_b),
      activeParams: nonnegative(row.active_params_b), license: row.license || '', openStatus: row.open_model_status || 'unknown' });
  }
  return map;
}
async function legacyGzip(url) {
  const bytes = await limited(async () => {
    const response = await fetch(url, { cache: 'no-cache', redirect: 'error' });
    if (!response.ok) throw new Error(`HTTP ${response.status}: ${url}`);
    return new Uint8Array(await response.arrayBuffer());
  });
  return JSON.parse(decoder.decode(await unzip(bytes)));
}

export function loadBenchmark(version = 'v2') {
  if (!['v1', 'v2'].includes(version)) return Promise.reject(new Error('Unknown benchmark version.'));
  if (!DATASETS.has(version)) {
    const request = load(version).catch(error => { DATASETS.delete(version); throw error; });
    DATASETS.set(version, request);
  }
  return DATASETS.get(version);
}
async function load(version) {
  const base = new URL(version === 'v1' ? 'data/latest/' : 'data/v2/latest/', ROOT);
  const manifestText = await fetchText(new URL('manifest.json', base), true);
  const manifest = manifestText ? JSON.parse(manifestText) : null;
  const storage = validateStorage(manifest);
  const suffix = manifest?.generated_at_utc ? `?v=${encodeURIComponent(manifest.generated_at_utc)}` : '';
  const [rawRows, questionText, panel, recentInfo, launchText, paramsText] = await Promise.all([
    storage ? readAsset(base, storage.assets['viewer_rows.json.gz'], suffix) : legacyGzip(new URL('viewer_rows.json.gz' + suffix, base)),
    manifest?.files?.['questions.json'] ? metadata(base, manifest, 'questions.json', false)
      : fetchText(new URL(version === 'v1' ? 'questions.json' : 'questions.v2.json', ROOT)),
    metadata(base, manifest, 'panel_summary.json'), metadata(base, manifest, 'recent_additions.json'),
    metadata(base, manifest, 'model_launch_dates.csv', false), metadata(base, manifest, 'model_params.csv', false),
  ]);
  if (!Array.isArray(rawRows)) throw new Error('Invalid viewer rows.');
  const allIds = rowIds(rawRows, 'viewer rows');
  const questions = flattenQuestions(JSON.parse(questionText)), questionMap = new Map(questions.map(q => [q.id, q]));
  if (manifest?.files?.['questions.json'] && (questionMap.size !== questions.length
      || questions.some(q => typeof q.id !== 'string' || !q.id.trim())
      || rawRows.some(row => !questionMap.has(row.question_id))))
    throw new Error('Published question snapshot does not match the results.');
  const rows = rawRows.map(row => { const q = questionMap.get(row.question_id); return { ...row,
    question: q?.text || row.question || '', domain: q?.domain || row.domain_group || row.domain || 'Other',
    source_domain: q?.source_domain || row.domain || '', technique: row.technique || q?.technique || '' }; });
  const expected = new Map(), partsByQuestion = new Map();
  for (const row of rows) {
    if (typeof row.question_id !== 'string' || !row.question_id.trim() || row.question_id !== row.question_id.trim()) throw new Error('Invalid published question ID.');
    if (!expected.has(row.question_id)) expected.set(row.question_id, new Set());
    expected.get(row.question_id).add(row.sample_id);
  }
  if (storage?.version === 2) {
    for (const part of storage.assets['viewer_details.json.gz'].parts) for (const qid of part.question_ids) {
      if (!partsByQuestion.has(qid)) partsByQuestion.set(qid, []);
      partsByQuestion.get(qid).push(part);
    }
    if (partsByQuestion.size !== expected.size || [...expected].some(([qid, ids]) =>
      (partsByQuestion.get(qid) || []).reduce((sum, part) => sum + part.rows, 0) !== ids.size))
      throw new Error('Published detail question counts disagree with summary.');
  }
  const cache = new Map(), pending = new Map();
  let legacyDetails;
  async function getDetails(questionId) {
    if (!expected.has(questionId)) throw new Error(`Unknown response question: ${questionId}`);
    if (cache.has(questionId)) { const value = cache.get(questionId); cache.delete(questionId); cache.set(questionId, value); return value; }
    if (pending.has(questionId)) return pending.get(questionId);
    const request = (async () => {
      let details;
      if (storage?.version === 2) {
        const parts = await Promise.all(partsByQuestion.get(questionId).map(part => readPart(base, storage.assets['viewer_details.json.gz'], part, suffix)));
        details = parts.flatMap(part => part.rows);
      } else {
        if (!legacyDetails) legacyDetails = (async () => {
          const records = storage ? await readAsset(base, storage.assets['viewer_details.json.gz'], suffix)
            : await legacyGzip(new URL('viewer_details.json.gz' + suffix, base));
          pair(records, allIds, 'response details');
          return records;
        })().catch(error => { legacyDetails = null; throw error; });
        details = (await legacyDetails).filter(row => expected.get(questionId).has(row.sample_id));
      }
      pair(details, expected.get(questionId), questionId);
      if (details.some(row => typeof row.response_text !== 'string')) throw new Error('Invalid published response text.');
      const result = new Map(details.map(row => [row.sample_id, row.response_text]));
      cache.set(questionId, result);
      // Responses are the heavy payload: retain only the most recent twelve questions.
      while (cache.size > 12) cache.delete(cache.keys().next().value);
      return result;
    })().finally(() => pending.delete(questionId));
    pending.set(questionId, request);
    return request;
  }
  return { version, rows, questions, questionMap, modelMetadata: buildMetadata(launchText, paramsText),
    recentInfo: recentInfo || {}, panel: panel || {}, manifest, getDetails };
}
