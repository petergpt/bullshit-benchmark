import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { exportExplorerPng } from '../viewer/next/capture.mjs';

const source = readFileSync(new URL('../viewer/next/app.mjs', import.meta.url), 'utf8');
const chartOptions = source.slice(source.indexOf('function chartOptions('), source.indexOf('function inspectorOverlay('));
const exportController = source.slice(source.indexOf('function updatePngButton('), source.indexOf('function questionsInScope('));
const captureImport = /import\('\.\/capture\.mjs\?[^']+'\)/g;
assert.equal([...exportController.matchAll(captureImport)].length, 1, 'test substitutes only the capture module boundary');

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function harness({ view = 'explorer', mode = 'timeline', moduleGate, capture } = {}) {
  const calls = [], downloads = [], messages = [], errors = [], revoked = [];
  const label = { textContent: '' }, attributes = new Map();
  const button = { setAttribute: (name, value) => attributes.set(name, value), querySelector: () => label };
  const state = {
    view, chartMode: mode, version: 'v2', domain: 'all', judge: 'consensus',
    provider: 'all', reasoning: 'all', access: 'all', recent: 'all', query: '',
    bestVariants: false, highlightOnly: false, excludeRefusals: false,
    highlighted: new Set(['model-b']), colorMode: 'detection',
    showUnchanged: true, showRefusalComparison: false,
  };
  const nodes = {
    marker: {},
    rows: [
      ['model-a', 60, 100], ['model-b', 100, 140],
      ['model-c', 140, 180], ['model-d', 480, 520],
    ].map(([id, top, bottom]) => ({ dataset: { model: id }, getBoundingClientRect: () => ({ top, bottom }) })),
  };
  const context = {
    state, Set, innerWidth: 1440, innerHeight: 600,
    visibleModels: ['a', 'b', 'c', 'd'].map((suffix, index) => ({ id: `model-${suffix}`, rank: index + 9 })),
    filteredRows: [{ model: 'model-b', question_id: 'question-1' }],
    dataset: { questions: [{ id: 'question-1' }], modelMetadata: new Map() }, scopeModelCount: 18,
    brandLogo: () => '/logo.svg', brandColor: () => '#23804d',
    companyName: value => `Company ${value}`, reasoningLabel: value => value.toUpperCase(),
    focusModel() {}, selectQuestion() {}, persist() {},
    options: () => ({ judge: state.judge, excludeRefusals: true }),
    $: selector => {
      if (selector === '#pngButton') return button;
      if (selector === '#judgeFilter') return { selectedOptions: [{ textContent: 'Consensus' }] };
      if (selector === '.table-scroll') return { getBoundingClientRect: () => ({ top: 60, bottom: 500 }) };
      if (selector === '.rank-table thead') return { getBoundingClientRect: () => ({ bottom: 100 }) };
      if (selector.startsWith('#chartCanvas ')) return nodes.marker;
      throw new Error(`Unexpected selector: ${selector}`);
    },
    document: {
      querySelectorAll(selector) {
        assert.equal(selector, '.rank-table tbody tr');
        return nodes.rows;
      },
      createElement(tag) {
        assert.equal(tag, 'a');
        return { click() { downloads.push({ href: this.href, filename: this.download }); } };
      },
    },
    URL: { createObjectURL: () => 'blob:chart', revokeObjectURL: value => revoked.push(value) },
    setTimeout: callback => callback(),
    toast: message => messages.push(message), console: { error: error => errors.push(error) },
  };
  const exporters = Object.fromEntries(['exportRankingsPng', 'exportExplorerPng'].map(name => [name, async settings => {
    calls.push({ name, settings });
    return capture ? capture(settings) : { blob: {}, filename: `${settings.mode || 'dashboard'}.png` };
  }]));
  let imports = 0;
  context.loadCapture = async () => {
    imports++;
    if (moduleGate) await moduleGate.promise;
    return exporters;
  };
  vm.runInNewContext(`let chartExportPending = false;\n${chartOptions}\n${exportController.replace(captureImport, 'loadCapture()')}`, context);
  return { context, state, nodes, button, label, attributes, calls, downloads, messages, errors, revoked, get imports() { return imports; } };
}

for (const mode of ['timeline', 'labs', 'reasoning', 'size']) {
  test(`${mode} PNG dispatches to the explorer exporter with the plotted denominator`, async () => {
    const h = harness({ mode });
    h.context.updatePngButton();
    assert.equal(h.button.hidden, false);
    assert.equal(h.button.disabled, false);
    assert.equal(h.button.title, 'Save chart as PNG');
    await h.context.downloadChart();
    assert.equal(h.calls.length, 1);
    const { name, settings } = h.calls[0];
    assert.equal(name, 'exportExplorerPng');
    assert.equal(settings.mode, mode);
    assert.equal(settings.models, h.context.visibleModels, 'explorer exports all filtered candidates');
    assert.equal(settings.excludeRefusals, true, 'the unchecked Dashboard bars control cannot change chart scoring');
    assert.equal(settings.colorMode, 'detection');
    assert.equal(settings.showUnchanged, true);
    assert.equal(settings.showRefusalComparison, false);
    assert.deepEqual([...settings.selected], ['model-b']);
    assert.equal(h.downloads[0].filename, `${mode}.png`);
    assert.equal(h.attributes.get('aria-busy'), 'false');
    assert.equal(h.label.textContent, 'PNG');
    assert.deepEqual(h.revoked, ['blob:chart']);
  });
}

test('Dashboard retains fully visible rows, their ranks and its bar denominator', async () => {
  const h = harness({ view: 'dashboard' });
  h.state.excludeRefusals = false;
  await h.context.downloadChart();
  const { name, settings } = h.calls[0];
  assert.equal(name, 'exportRankingsPng');
  assert.deepEqual(Array.from(settings.models, model => [model.id, model.rank]), [['model-b', 10], ['model-c', 11]]);
  assert.equal(settings.totalModels, 18);
  assert.equal(settings.excludeRefusals, false);
  assert.equal(settings.mode, undefined);
  assert.equal(h.button.title, 'Save visible rows as PNG');
  assert.deepEqual(h.messages, ['PNG saved · 2 rows']);
});

test('export settings are captured before awaiting the module while controls follow the current view', async () => {
  const gate = deferred(), h = harness({ mode: 'labs', moduleGate: gate });
  Object.assign(h.state, { version: 'v1', domain: 'Physics', provider: 'openai', reasoning: 'high', query: 'old query' });
  const originalModels = h.context.visibleModels, originalRows = h.context.filteredRows, originalMetadata = h.context.dataset.modelMetadata;
  const completion = h.context.downloadChart();
  assert.equal(h.button.disabled, true);
  assert.equal(h.label.textContent, 'Saving…');
  assert.equal(h.attributes.get('aria-busy'), 'true');
  await h.context.downloadChart();
  assert.equal(h.imports, 1, 'repeated clicks cannot launch duplicate exports');

  Object.assign(h.state, { view: 'responses', chartMode: 'size', version: 'v2', domain: 'all', judge: 'judge_2',
    provider: 'google', reasoning: 'none', query: 'new query', colorMode: 'provider', showUnchanged: false, showRefusalComparison: true });
  h.state.highlighted.clear();
  h.context.visibleModels = [];
  h.context.filteredRows = [];
  h.context.dataset = { questions: [], modelMetadata: new Map() };
  h.nodes.marker = null;
  gate.resolve();
  await completion;
  const { name, settings } = h.calls[0];
  assert.equal(name, 'exportExplorerPng');
  assert.equal(settings.mode, 'labs');
  assert.equal(settings.version, 'v1');
  assert.equal(settings.domain, 'Physics');
  assert.equal(settings.judge, 'consensus');
  assert.equal(settings.filterLabel, 'Company openai · HIGH reasoning · Search: old query');
  assert.equal(settings.models, originalModels);
  assert.equal(settings.rows, originalRows);
  assert.equal(settings.metadata, originalMetadata);
  assert.equal(settings.colorMode, 'detection');
  assert.equal(settings.showUnchanged, true);
  assert.equal(settings.showRefusalComparison, false);
  assert.deepEqual([...settings.selected], ['model-b']);
  assert.deepEqual([...settings.highlighted], ['model-b']);
  assert.equal(h.button.hidden, true, 'completion must not show PNG controls on Responses');
  assert.equal(h.button.disabled, true);
  assert.equal(h.attributes.get('aria-busy'), 'false');
  assert.equal(h.label.textContent, 'PNG');
});

for (const failImport of [true, false]) {
  test(`PNG controls recover after ${failImport ? 'module loading' : 'chart encoding'} fails`, async () => {
    const gate = deferred();
    let fail = true;
    const h = harness({ moduleGate: failImport ? gate : undefined,
      capture: async () => { if (fail) throw new Error('Image unavailable'); return { blob: {}, filename: 'recovered.png' }; } });
    const completion = h.context.downloadChart();
    if (failImport) gate.reject(new Error('Module unavailable'));
    await completion;
    assert.equal(h.downloads.length, 0);
    assert.equal(h.errors.length, 1);
    assert.match(h.messages[0], /^PNG export failed: (Module|Image) unavailable$/);
    assert.equal(h.button.disabled, false);
    assert.equal(h.attributes.get('aria-busy'), 'false');
    assert.equal(h.label.textContent, 'PNG');
    fail = false;
    gate.promise = Promise.resolve();
    await h.context.downloadChart();
    assert.equal(h.downloads[0].filename, 'recovered.png', 'a failed attempt must not leave export locked');
  });
}

test('suite loading disables PNG and never attempts to read the missing dataset', async () => {
  for (const view of ['dashboard', 'explorer']) {
    const h = harness({ view });
    h.context.dataset = null;
    h.context.updatePngButton();
    assert.equal(h.button.disabled, true, view);
    await h.context.downloadChart();
    assert.equal(h.imports, 0, view);
    assert.equal(h.errors.length, 0, view);
    assert.equal(h.downloads.length, 0, view);
  }
});

test('unplottable selections disable PNG, while unchanged reasoning pairs remain exportable', async () => {
  const h = harness();
  h.nodes.marker = null;
  h.context.updatePngButton();
  assert.equal(h.button.disabled, true, 'having model rows alone does not mean a chart can be plotted');
  h.nodes.marker = { className: 'next-chart-unchanged-row' };
  h.context.updatePngButton();
  assert.equal(h.button.disabled, false, 'all-unchanged reasoning families still have an export');
  h.context.visibleModels = [];
  h.nodes.marker = null;
  h.context.updatePngButton();
  await h.context.downloadChart();
  assert.equal(h.imports, 0);
  h.state.view = 'responses';
  h.context.visibleModels = [{ id: 'example' }];
  await h.context.downloadChart();
  assert.equal(h.imports, 0);
});

test('Dashboard does not create a PNG when no full row is visible', async () => {
  const h = harness({ view: 'dashboard' });
  h.nodes.rows = h.nodes.rows.filter(row => ['model-a', 'model-d'].includes(row.dataset.model));
  await h.context.downloadChart();
  assert.equal(h.imports, 0);
  assert.deepEqual(h.messages, ['Scroll to a full row']);
});

test('explorer export rejects unsupported modes and empty input before creating a rendering surface', async () => {
  await assert.rejects(exportExplorerPng({ mode: 'responses', models: [{ id: 'example' }] }), /does not have a chart/);
  for (const mode of ['timeline', 'labs', 'reasoning', 'size']) {
    await assert.rejects(exportExplorerPng({ mode, models: [] }), /Choose at least one model/);
  }
});

test('an unplottable chart rejects with its empty-state message and removes the offscreen surface', async () => {
  const captureSource = readFileSync(new URL('../viewer/next/capture.mjs', import.meta.url), 'utf8');
  const snapshot = captureSource.slice(captureSource.indexOf('async function explorerSnapshot('), captureSource.indexOf('function svgLayer('));
  let appended = 0, removed = 0;
  const host = {
    style: {}, setAttribute() {}, remove: () => removed++,
    querySelector: selector => selector === '.next-chart-empty strong' ? { textContent: 'No release dates for these models' } : null,
    querySelectorAll: () => [],
  };
  const context = {
    FONT: 'sans-serif',
    loadChartRenderer: async () => ({ renderExplorer() {} }),
    document: { fonts: { ready: Promise.resolve() }, createElement: () => host, body: { append: () => appended++ } },
  };
  vm.runInNewContext(snapshot.replace(/import\('\.\/charts\.mjs\?[^']+'\)/, 'loadChartRenderer()'), context);
  await assert.rejects(context.explorerSnapshot({ mode: 'timeline', models: [{ id: 'undated' }] }, 1200, 600), /No release dates for these models/);
  assert.equal(appended, 1);
  assert.equal(removed, 1, 'failed exports must not retain their hidden DOM');
});
