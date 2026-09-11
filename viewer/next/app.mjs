import { loadBenchmark, modelLabel, groupModels, summarizeRows, classify, usesTwoJudgeFallback, judgeCoverageNote, newModelUntil } from './data.mjs?v=20260907-dashboard';
import { renderExplorer } from './charts.mjs?v=20260907-dashboard';
import { brandLogo, brandColor, brandName } from './brands.mjs';

const $ = selector => document.querySelector(selector);
const popoverIds = ['labPicker', 'moreFilters', 'columnPicker'];
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const fmt = (n, digits = 0) => n === null || n === undefined || !Number.isFinite(n) ? '—' : n.toLocaleString('en-US', { maximumFractionDigits: digits });
const pct = n => n === null || n === undefined || !Number.isFinite(n) ? '—' : `${fmt(n, 1)}%`;
const money = n => n === null || n === undefined ? '—' : `$${n.toFixed(n < .01 ? 4 : 3)}`;
const columnDefinitions = [
  { id: 'accepted', key: 'redRate', label: 'Accepted', title: 'Accepted premise; excludes refusals', width: 86, format: model => pct(model.redRate) },
  { id: 'score', key: 'avgScore', label: 'Grade / 2', title: 'Average grade (0–2)', width: 86, format: model => fmt(model.avgScore, 2) },
  { id: 'tokens', key: 'avgTokens', label: 'Tokens', title: 'Tokens / answer', width: 78, format: model => fmt(model.avgTokens) },
  { id: 'reasoning', key: 'avgReasoningTokens', label: 'Reasoning tok.', title: 'Reported reasoning tokens / answer', width: 112, format: model => fmt(model.avgReasoningTokens), coverage: 'reasoningTokenCount' },
  { id: 'output', key: 'avgOutputTokens', label: 'Output tok.', title: 'Provider-reported output tokens / answer', width: 98, format: model => fmt(model.avgOutputTokens), coverage: 'outputTokenCount' },
  { id: 'cost', key: 'avgCost', label: 'Cost', title: 'USD / answer', width: 88, format: model => money(model.avgCost) },
];
const effortOrder = ['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra'];
const sortDefinitions = [
  { key: 'greenRate', label: 'Clear pushback', direction: -1 },
  { key: 'redRate', label: 'Least accepted', direction: 1 },
  { key: 'avgScore', label: 'Average grade', direction: -1 },
  { key: 'name', label: 'Model A–Z', direction: 1 },
  { key: 'reasoning', label: 'Reasoning level', direction: 1 },
  { key: 'avgTokens', label: 'Total tokens', direction: 1 },
  { key: 'avgReasoningTokens', label: 'Reasoning tokens', direction: 1 },
  { key: 'avgOutputTokens', label: 'Output tokens', direction: 1 },
  { key: 'avgCost', label: 'Cost', direction: 1 },
];
const outcomeNames = { green: 'Clear pushback', amber: 'Partial challenge', red: 'Accepted premise', refusal: 'Refusal', error: 'Grade unavailable' };
const shortNames = { green: 'Clear', amber: 'Partial', red: 'Accepted', refusal: 'Refusal', error: 'Error' };
const color = model => brandColor(model.org);
const reasoningLabel = value => ({none:'None',minimal:'Minimal',low:'Low',medium:'Medium',high:'High',xhigh:'xHigh',max:'Max',ultra:'Ultra'})[value] || value;
const companyName = org => org === 'all' ? 'All labs' : brandName(org);
function companyIcon(org) {
  if (org === 'all') return '<svg viewBox="0 0 20 20"><rect x="3" y="3" width="5" height="5" rx="1"/><rect x="12" y="3" width="5" height="5" rx="1"/><rect x="3" y="12" width="5" height="5" rx="1"/><rect x="12" y="12" width="5" height="5" rx="1"/></svg>';
  const logo = brandLogo(org);
  return logo ? `<img src="${esc(logo)}" alt="">` : `<span class="lab-initial">${esc(companyName(org)[0])}</span>`;
}
function companyMark(model) { const logo = brandLogo(model.org); return logo ? `<img class="company-logo" src="${esc(logo)}" alt="${esc(model.provider)}" title="${esc(model.provider)}">` : `<span class="company-initial" title="${esc(model.provider)}">${esc(model.provider[0])}</span>`; }
const fullName = model => model ? `${model.name} · ${reasoningLabel(model.reasoning)}` : 'Choose a model';
function newBadge(model) {
  const until = newModelUntil(model, dataset.recentInfo);
  return until ? `<span class="new-badge" data-new-until="${until}" title="First tested in the past 7 days">New</span>` : '';
}
let newBadgeTimer;
function expireNewBadges() {
  clearTimeout(newBadgeTimer);
  const now = Date.now(), remaining = [];
  document.querySelectorAll('[data-new-until]').forEach(badge => {
    const until = Number(badge.dataset.newUntil);
    if (until <= now) badge.remove();
    else remaining.push(until);
  });
  if (remaining.length) newBadgeTimer = setTimeout(expireNewBadges, Math.min(...remaining) - now + 1);
}
document.addEventListener('visibilitychange', () => { if (!document.hidden) expireNewBadges(); });
function coverageBadge(count, answer = false) {
  if (!count) return '';
  return `<span class="judge-coverage" title="${esc(judgeCoverageNote(count))}" aria-label="${esc(judgeCoverageNote(count))}">2/3 judges${answer ? '' : ` · ${count} answer${count === 1 ? '' : 's'}`}</span>`;
}
const url = new URL(location.href);
const requestedVersion = String(url.searchParams.get('benchmark') || url.searchParams.get('version') || '').trim().toLowerCase();
const legacyRankings = url.searchParams.get('view') === 'rankings';
const requestedColumns = new Set(url.searchParams.getAll('columns').flatMap(value => value.split(',')));
const initialSort = sortDefinitions.find(item => item.key === url.searchParams.get('sort')) || sortDefinitions[0];
const state = {
  view: ['dashboard', 'explorer', 'responses'].includes(url.searchParams.get('view')) ? url.searchParams.get('view') : 'dashboard',
  columns: new Set(columnDefinitions.filter(column => (legacyRankings && ['score','tokens','cost'].includes(column.id)) || requestedColumns.has(column.id)).map(column => column.id)),
  version: requestedVersion === 'v1' ? 'v1' : 'v2',
  domain: url.searchParams.get('domain') || 'all', query: url.searchParams.get('q') || '', provider: url.searchParams.get('provider') || 'all', reasoning: url.searchParams.get('reasoning') || 'all', access: url.searchParams.get('access') || 'all', recent: ['7','30','90'].includes(url.searchParams.get('recent')) ? url.searchParams.get('recent') : 'all',
  judge: ['judge_1','judge_2','judge_3'].includes(url.searchParams.get('judge')) ? url.searchParams.get('judge') : 'consensus', excludeRefusals: url.searchParams.get('excludeRefusals') === '1', sort: initialSort.key, direction: ['1','-1'].includes(url.searchParams.get('direction')) ? Number(url.searchParams.get('direction')) : initialSort.direction,
  focus: url.searchParams.get('model'), compare: [],
  question: null, questionQuery: '', answerTab: 0,
  chartMode: ['timeline', 'labs', 'reasoning', 'size'].includes(url.searchParams.get('chart')) ? url.searchParams.get('chart') : 'timeline',
  colorMode: url.searchParams.get('colour')==='detection' ? 'detection' : 'provider',
  highlighted: new Set(url.searchParams.getAll('highlight')), highlightOnly: url.searchParams.get('highlightOnly')==='1', pinned: url.searchParams.get('pin')==='1',
  inspector: false, questionsOpen: false, showUnchanged: false, showRefusalComparison: url.searchParams.get('compareRefusals') !== '0', bestVariants: url.searchParams.get('best')==='1', outcome: null, outcomeModel: null,
};
let dataset, allModels = [], modelMap = new Map(), domains = [], visibleModels = [], filteredRows = [], scopeModelCount = 0;
let loadGeneration = 0, answerGeneration = 0, searchTimer, toastTimer;
// Scores, rankings and averages always exclude refusals. The checkbox controls bars only.
const options = () => ({ judge: state.judge, excludeRefusals: true });
const metaFor = model => dataset.modelMetadata.get(model.base) || {};
const isOpen = model => /^(open|open[-_ ]weights|open[-_ ]source)$/i.test(metaFor(model).openStatus || '');

function persist() {
  const next = new URL(location.href);
  next.search = '';
  if (state.version !== 'v2') next.searchParams.set('benchmark', state.version);
  if (state.view !== 'dashboard') next.searchParams.set('view', state.view);
  if (state.columns.size) next.searchParams.set('columns', columnDefinitions.filter(column => state.columns.has(column.id)).map(column => column.id).join(','));
  if (state.domain !== 'all') next.searchParams.set('domain', state.domain);
  // Focus, answer selections and the current question stay within this page.
  if (state.view === 'explorer') { next.searchParams.set('chart', state.chartMode); if(state.colorMode!=='provider') next.searchParams.set('colour',state.colorMode); }
  for(const id of state.highlighted) next.searchParams.append('highlight',id);
  if(state.highlightOnly) next.searchParams.set('highlightOnly','1');
  if(state.pinned) next.searchParams.set('pin','1');
  for(const [key,param] of [['query','q'],['provider','provider'],['reasoning','reasoning'],['access','access'],['recent','recent']]) if(state[key] && state[key]!=='all') next.searchParams.set(param,state[key]);
  if(state.judge!=='consensus') next.searchParams.set('judge',state.judge);
  if(state.excludeRefusals) next.searchParams.set('excludeRefusals','1');
  if(state.sort !== 'greenRate' || state.direction !== -1) { next.searchParams.set('sort',state.sort); next.searchParams.set('direction',String(state.direction)); }
  if(!state.showRefusalComparison) next.searchParams.set('compareRefusals','0');
  if(state.bestVariants) next.searchParams.set('best','1');
  history.replaceState(null, '', next);
}
function toast(message) {
  $('#toast').textContent = message; $('#toast').hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { $('#toast').hidden = true; }, 2500);
}
async function copyText(text) {
  try { await navigator.clipboard.writeText(text); return; } catch {}
  const focus = document.activeElement, input = document.createElement('textarea');
  input.value = text;
  input.style.cssText = 'position:fixed;top:0;left:-9999px';
  document.body.append(input); input.focus(); input.select();
  try { if (!document.execCommand('copy')) throw new Error('Clipboard unavailable'); }
  finally { input.remove(); focus?.focus({ preventScroll: true }); }
}
function resetFilters() {
  Object.assign(state, { query: '', provider: 'all', reasoning: 'all', access: 'all', recent: 'all', domain: 'all', judge: 'consensus', excludeRefusals: false, sort: 'greenRate', direction: -1, bestVariants: false, outcome: null, outcomeModel: null });
  syncControls(); refresh();
}
function syncControls() {
  for (const [id, key] of [['domainFilter','domain'], ['modelSearch', 'query'], ['reasoningFilter', 'reasoning'], ['accessFilter', 'access'], ['recentFilter', 'recent'], ['judgeFilter', 'judge'], ['suiteSelect', 'version']]) $(`#${id}`).value = state[key];
  $('#excludeRefusals').checked = state.excludeRefusals;
  $('#bestVariants').checked = state.bestVariants;
  $('.lab-select').classList.toggle('is-filtered', state.provider !== 'all');
  $('#labMark').innerHTML = companyIcon(state.provider);
  $('#labName').textContent = companyName(state.provider);
  $('#labTrigger').setAttribute('aria-label', `Company: ${companyName(state.provider)}`);
  $('#labTrigger').title = companyName(state.provider);
  document.querySelectorAll('[data-company]').forEach(button => {
    const selected = button.dataset.company === state.provider;
    button.setAttribute('aria-checked', String(selected));
    button.tabIndex = selected ? 0 : -1;
  });
  $('#highlightOnly').checked=state.highlightOnly;
  document.querySelectorAll('[data-column]').forEach(input => { input.checked = state.columns.has(input.dataset.column); });
}
function updateChrome() {
  document.querySelectorAll('[data-view]').forEach(button => {
    button.classList.toggle('active', button.dataset.view === state.view);
    button.setAttribute('aria-current', button.dataset.view === state.view ? 'page' : 'false');
  });
  document.querySelectorAll('.navigation [data-chart]').forEach(button => { const active = state.view === 'explorer' && state.chartMode === button.dataset.chart; button.classList.toggle('active', active); button.setAttribute('aria-current', active ? 'page' : 'false'); });
  $('#chartColour').hidden=state.view!=='explorer' || ['reasoning','labs'].includes(state.chartMode);
  $('#chartColour').value=state.colorMode;
  $('#columnPicker').hidden=state.view!=='dashboard';
  $('.denominator-toggle').hidden=state.view!=='dashboard';
  $('#sortControl').hidden=state.view!=='dashboard';
  $('#sortFilter').innerHTML=sortDefinitions.map(item=>`<option value="${item.key}" ${state.sort===item.key?'selected':''}>${item.label}${state.sort===item.key?(state.direction<0?' ↓':' ↑'):''}</option>`).join('');
  if (state.view !== 'dashboard' || state.pinned) $('#columnPicker').open=false;
  document.querySelectorAll('[data-column]').forEach(input => { input.checked = state.columns.has(input.dataset.column); });
  $('.workspace').classList.toggle('is-pinned',state.pinned);
  $('#pinButton').classList.toggle('active',state.pinned);
  $('#pinButton').setAttribute('aria-pressed',String(state.pinned));
  $('#pinButton').setAttribute('aria-label',state.pinned ? 'Unpin screenshot view' : 'Pin benchmark for screenshots');
  $('#clearHighlights').hidden=state.highlighted.size===0;
  $('#clearHighlights').textContent=`${state.highlighted.size} highlighted ×`;
  $('.highlight-toggle').hidden=state.highlighted.size===0;
  $('#highlightOnly').disabled=state.highlighted.size===0;
  $('#highlightOnly').checked=state.highlightOnly;
  $('#highlightAnswers').hidden=state.highlighted.size===0 || state.view==='responses';
  $('#highlightAnswers').textContent=state.highlighted.size>1 ? 'Compare answers' : 'View answers';
  $('#pngButton').hidden=state.view!=='dashboard';
  $('#pngButton').disabled=!visibleModels.length;
  const advanced = state.domain !== 'all' || state.reasoning !== 'all' || state.access !== 'all' || state.recent !== 'all' || state.judge !== 'consensus' || state.bestVariants || state.sort !== 'greenRate' || state.direction !== -1;
  $('#filterDot').hidden = !advanced;
  $('#clearSearch').hidden = !(advanced || state.excludeRefusals || state.query || state.provider !== 'all' || state.reasoning !== 'all' || state.domain !== 'all');
  queueMixLabels();
  requestAnimationFrame(positionPopovers);
  persist();
}
function matchingModels() {
  const query = state.query.trim().toLowerCase();
  const now = Date.now();
  return allModels.filter(model => {
    if (state.view === 'explorer' && state.chartMode === 'labs' && !['openai', 'anthropic', 'google'].includes(model.org)) return false;
    if (query && !`${model.id} ${model.name} ${model.provider} ${model.reasoning}`.toLowerCase().includes(query)) return false;
    if (state.provider !== 'all' && model.org !== state.provider) return false;
    if (state.reasoning !== 'all' && model.reasoning !== state.reasoning) return false;
    if (state.access === 'open' && !isOpen(model)) return false;
    if (state.access === 'closed' && !/^closed(?:[-_ ](?:weights|source))?$/i.test(metaFor(model).openStatus || '')) return false;
    if (state.recent !== 'all') {
      const firstSeen = Date.parse(dataset.recentInfo.model_first_seen_utc?.[model.id]);
      if (!Number.isFinite(firstSeen) || firstSeen < now - Number(state.recent) * 86400000) return false;
    }
    return true;
  });
}
function refresh() {
  if (!dataset) return;
  const matching = new Set(matchingModels().map(model => model.id));
  filteredRows = dataset.rows.filter(row => matching.has(row.model) && (state.domain === 'all' || row.domain === state.domain));
  let grouped = groupModels(filteredRows, options());
  if (state.bestVariants) {
    const bases = new Set();
    grouped = grouped.filter(model => { if (bases.has(model.base)) return false; bases.add(model.base); return true; });
    const kept = new Set(grouped.map(model => model.id));
    filteredRows = filteredRows.filter(row => kept.has(row.model));
  }
  scopeModelCount=grouped.length;
  let ranked=grouped.map((model,index)=>({...model,rank:index+1,redRate:model.rateRows ? 100*model.red/model.rateRows : null,newUntil:newModelUntil(model,dataset.recentInfo)}));
  if(state.highlightOnly){ranked=ranked.filter(model=>state.highlighted.has(model.id));const kept=new Set(ranked.map(model=>model.id));filteredRows=filteredRows.filter(row=>kept.has(row.model));}
  visibleModels = ranked.sort((a, b) => {
    if(state.sort==='reasoning') return (effortOrder.indexOf(a.reasoning)-effortOrder.indexOf(b.reasoning))*state.direction || a.rank-b.rank;
    const av = a[state.sort], bv = b[state.sort];
    if (av == null || bv == null) return av == null ? bv == null ? 0 : 1 : -1;
    return (typeof av === 'string' ? av.localeCompare(bv) : av - bv) * state.direction || b.greenRate - a.greenRate || a.rank - b.rank;
  });
  if (!visibleModels.some(model => model.id === state.focus)) state.focus = visibleModels[0]?.id || null;
  updateChrome();
  renderView();
}
function mix(model, label = true) {
  const barRows = state.excludeRefusals ? model.nonRefusalRows : model.total;
  const categories = ['green', 'amber', 'red', 'refusal', 'error'];
  const title = categories.filter(key => model[key]).map(key => `${outcomeNames[key]}: ${fmt(model[key])}`).join(' · ');
  return `<div class="mix" ${label ? `role="img" aria-label="${esc(title)}" title="${esc(title)}"` : ''}>${categories.map(key => {
    const n = state.excludeRefusals && key === 'refusal' ? 0 : model[key];
    const rate = barRows ? 100 * n / barRows : 0;
    return `<span class="${key}" style="width:${rate}%" ${n && model.id ? `role="button" tabindex="0" data-outcome="${key}" data-segment-model="${esc(model.id)}" aria-label="${esc(model.name)}: ${fmt(n)} ${outcomeNames[key]} answers"` : ''}>${rate > 0 ? `<b class="mix-label" aria-hidden="true">${fmt(rate)}%</b>` : ''}</span>`;
  }).join('')}</div>`;
}
let mixLabelFrame = 0;
function fitMixLabels() {
  const labels = [...document.querySelectorAll('.mix-label')];
  // Reset together, measure together, then apply sizes to avoid repeated layouts.
  labels.forEach(label => label.style.removeProperty('font-size'));
  const measured = labels.map(label => ({
    label, width: label.getBoundingClientRect().width,
    available: label.parentElement.getBoundingClientRect().width - 2,
    preferred: parseFloat(getComputedStyle(label).fontSize),
  }));
  measured.forEach(({label, width, available, preferred}) => {
    const size = width > 0 ? Math.floor(Math.min(preferred, preferred * available / width) * 2) / 2 : 0;
    const fits = size >= 9;
    label.classList.toggle('fits', fits);
    if (fits) label.style.fontSize = `${size}px`;
  });
}
function queueMixLabels() {
  if (!mixLabelFrame) mixLabelFrame = requestAnimationFrame(() => { mixLabelFrame = 0; fitMixLabels(); });
}
const mixSizeObserver = typeof ResizeObserver === 'function' ? new ResizeObserver(queueMixLabels) : null;
function observeMixLabels() {
  mixSizeObserver?.disconnect();
  document.querySelectorAll('.mix').forEach(bar => mixSizeObserver?.observe(bar));
  cancelAnimationFrame(mixLabelFrame); mixLabelFrame = 0;
  fitMixLabels();
}
window.addEventListener('resize', queueMixLabels);
document.fonts?.ready.then(queueMixLabels);
function usageCoverage(model, key) {
  return model[key] ? `Reported on ${model[key]}/${model.usageRowCount} non-refusal attempts` : 'Not reported';
}
function modelRate(model) { return model.rateRows ? pct(model.greenRate) : '—'; }
function rateComparison(model) { return model.refusal ? `${pct(model.greenRateExcludingRefusals)} excl. · ${pct(model.greenRateAllAttempts)} incl. refusals` : `${fmt(model.green)}/${fmt(model.total)} clear`; }
function legend() { return `<div class="legend">${['green', 'amber', 'red', 'refusal'].filter(key => key !== 'refusal' || !state.excludeRefusals).map(key => `<span title="${outcomeNames[key]}"><i class="dot ${key}"></i>${shortNames[key]}</span>`).join('')}</div>`; }
function selectedModel() { return visibleModels.find(model => model.id === state.focus); }
function inspectorHTML() {
  const model = selectedModel();
  if (!model) return '';
  const meta = metaFor(model), source = modelMap.get(model.id), grouped = new Map();
  for (const row of source.rows) { if (!grouped.has(row.domain)) grouped.set(row.domain, []); grouped.get(row.domain).push(row); }
  return `<aside id="modelInspector" class="inspector ${state.inspector ? 'is-open' : ''}" aria-label="Model details">
    <div class="inspector-top"><div class="inspector-provider">${companyMark(model)}${esc(model.provider)}</div><div class="inspector-title-line"><h2>${esc(model.name)}</h2><button class="close-inspector icon-button" data-action="close-inspector" aria-label="Close model details">×</button></div><span class="effort">Requested effort: ${esc(reasoningLabel(model.reasoning))}</span></div>
    <div class="inspector-main"><div class="score-line"><div><div class="score-value">${model.rateRows ? fmt(model.greenRate, 1) : '—'}<small>${model.rateRows ? '%' : ''}</small></div><div class="score-caption">Clear pushback${state.domain === 'all' ? '' : ` · ${esc(state.domain)}`}</div></div><span class="score-samples">${fmt(model.rateRows)} non-refusal attempts</span></div>${model.refusal ? `<div class="inspector-rate-comparison">${esc(rateComparison(model))}</div>` : ''}${mix(model)}<div class="mix-labels">${['green', 'amber', 'red', 'refusal'].map(key => `<div><span><i class="dot ${key}"></i>${shortNames[key]}</span><strong>${fmt(model[key])}</strong></div>`).join('')}</div>${model.error ? `<div class="score-caption">${fmt(model.error)} unavailable grades</div>` : ''}</div>
    <div class="inspector-metrics"><div title="Average grade (0–2)"><span>Avg. grade</span><strong>${fmt(model.avgScore, 2)}<small> / 2</small></strong></div><div><span>Total tokens</span><strong>${fmt(model.avgTokens)}</strong></div><div><span>Cost / answer</span><strong>${money(model.avgCost)}</strong></div></div>
    <div class="inspector-domains"><div class="section-label">Across domains</div>${domains.map(domain => { const stats = summarizeRows(grouped.get(domain) || [], options()); return `<button class="domain-mini ${state.domain === domain ? 'active' : ''}" data-domain="${esc(domain)}" title="Filter to ${esc(domain)}"><span>${esc(domain)}</span><i><b style="width:${stats.greenRate}%"></b></i><strong>${stats.rateRows ? pct(stats.greenRate) : '—'}</strong></button>`; }).join('')}</div>
    <div class="inspector-actions"><button class="secondary-button" data-action="toggle-highlight">${state.highlighted.has(model.id) ? 'Unhighlight' : 'Highlight'}</button><button class="primary-button" data-action="answers">View answers</button></div>
    <div class="inspector-foot"><span>${meta.launchDate ? `Released ${esc(meta.launchDate.slice(0, 10))}` : 'Release unknown'}</span><span>${isOpen(model) ? `${meta.totalParams ? `${fmt(meta.totalParams, 1)}B · ` : ''}Open weights` : meta.openStatus === 'closed' ? 'Closed weights' : 'Weights unknown'}</span></div>
  </aside>`;
}
function emptyHTML() { return `<div class="empty-state"><strong>No matching models</strong><button class="secondary-button" data-action="reset">Clear filters</button></div>`; }
function tableHTML() {
  const sorted=(key,label)=>`<button data-sort="${key}" class="${state.sort===key?'sorted':''}">${label}${state.sort===key ? state.direction<0?' ↓':' ↑' : ''}</button>`;
  const columns=columnDefinitions.filter(column=>state.columns.has(column.id));
  const columnHeads=columns.map(column=>`<th class="metric-col metric-${column.id}" title="${column.title}">${sorted(column.key,column.label)}</th>`).join('');
  const hasHighlights=visibleModels.some(model=>state.highlighted.has(model.id));
  const rows=visibleModels.map(model=>{
    const cells=columns.map(column=>`<td class="metric-col metric-${column.id}"${column.coverage ? ` title="${esc(usageCoverage(model, column.coverage))}"` : ''}>${column.format(model)}</td>`).join('');
    return `<tr data-model="${esc(model.id)}" class="${state.highlighted.has(model.id)?'highlighted':''}" tabindex="0" aria-selected="${state.highlighted.has(model.id)}" aria-label="Highlight ${esc(fullName(model))}, ${modelRate(model)} clear pushback"><td class="check-col"><input class="table-check" type="checkbox" data-highlight="${esc(model.id)}" aria-label="Highlight ${esc(fullName(model))}" ${state.highlighted.has(model.id)?'checked':''}></td><td class="rank-col">${model.rank}</td><td class="model-col"><div class="model-cell">${companyMark(model)}<span class="model-identity"><span class="model-name" title="${esc(model.name)}">${esc(model.name)}</span>${newBadge(model)}</span></div></td><td class="reasoning-col"><span class="reasoning-value ${model.reasoning==='none'?'none':''}">${esc(reasoningLabel(model.reasoning))}</span></td><td class="rate-col"><span class="rate" title="${esc(rateComparison(model))}">${modelRate(model)}</span></td><td class="mix-col">${mix(model)}</td>${cells}<td class="row-actions"><button class="row-details" data-inspect-model="${esc(model.id)}" aria-label="Details for ${esc(fullName(model))}" title="Model details">⋯</button></td></tr>`;
  }).join('');
  return `<section class="table-panel main-bar-chart ${columns.length?'has-extra-columns':''}" style="--optional-columns-width:${columns.reduce((sum,column)=>sum+column.width,0)}px">${visibleModels.length?`<div class="table-scroll"><table class="rank-table ${hasHighlights?'has-highlights':''}"><thead><tr><th class="check-col" title="Highlight models"></th><th class="rank-col" title="Rank by clear pushback, excluding refusals">#</th><th class="model-col">${sorted('name','Model')}</th><th class="reasoning-col" title="Requested reasoning effort">${sorted('reasoning','Reasoning')}</th><th class="rate-col" title="Clear pushback; excludes refusals">${sorted('greenRate','Clear')}</th><th class="mix-col"><div class="mix-heading"><span class="mobile-clear-sort">${sorted('greenRate','Clear')}</span>${legend()}</div></th>${columnHeads}<th class="row-actions"></th></tr></thead><tbody>${rows}</tbody></table></div>`:emptyHTML()}</section>`;
}
function chartOptions(mode) {
  return {mode,models:visibleModels,rows:filteredRows,questions:dataset.questions,metadata:dataset.modelMetadata,selected:new Set(state.highlighted),onSelect:focusModel,onQuestion:selectQuestion,colorMode:state.colorMode,brandColor,brandLogo,showUnchanged:state.showUnchanged,onUnchangedChange:value=>{state.showUnchanged=value;},showRefusalComparison:state.showRefusalComparison,onRefusalComparisonChange:value=>{state.showRefusalComparison=value;persist();},...options()};
}
function inspectorOverlay() { return `${state.inspector?inspectorHTML():''}<div class="inspector-backdrop ${state.inspector?'show':''}" data-action="close-inspector"></div>`; }
function renderDashboard() { $('#content').innerHTML=`<div class="rank-layout">${tableHTML()}</div>${inspectorOverlay()}`; observeMixLabels(); expireNewBadges(); }
function renderCharts() {
  $('#content').innerHTML=`<div class="explorer-layout"><section class="explorer-panel"><div id="chartCanvas" class="explorer-canvas"></div></section></div>${inspectorOverlay()}`;
  renderExplorer($('#chartCanvas'),chartOptions(state.chartMode));
  observeMixLabels();
}
function rerenderPreservingScroll(render = renderView) {
  const selectors=['.table-scroll','.next-chart-family','.next-chart-scroll'];
  const positions=selectors.map(selector=>[...document.querySelectorAll(selector)].map(node=>({top:node.scrollTop,left:node.scrollLeft})));
  render();
  selectors.forEach((selector,i)=>document.querySelectorAll(selector).forEach((node,j)=>{if(positions[i][j]){node.scrollTop=positions[i][j].top;node.scrollLeft=positions[i][j].left;}}));
}
async function downloadChart() {
  const viewport=$('.table-scroll'); if(!viewport || !visibleModels.length) return;
  const bounds=viewport.getBoundingClientRect(),visibleBottom=Math.min(bounds.bottom,innerHeight),headerBottom=$('.rank-table thead').getBoundingClientRect().bottom;
  const ids=[...document.querySelectorAll('.rank-table tbody tr')].filter(row=>{const r=row.getBoundingClientRect();return r.top>=Math.max(bounds.top,headerBottom)-1 && r.bottom<=visibleBottom+1;}).map(row=>row.dataset.model);
  const models=ids.map(id=>visibleModels.find(model=>model.id===id)).filter(Boolean);
  if(!models.length){toast('Scroll to a full row');return;}
  const button=$('#pngButton');button.disabled=true;
  try {
    const {exportRankingsPng}=await import('./capture.mjs?v=20260907-dashboard');
    const filterLabel=[state.provider!=='all'?companyName(state.provider):null,state.reasoning!=='all'?`${reasoningLabel(state.reasoning)} reasoning`:null,state.access!=='all'?`${state.access==='open'?'Open':'Closed'} weights`:null,state.recent!=='all'?`Added in last ${state.recent} days`:null,state.bestVariants?'Best per model':null,state.highlightOnly?'Highlighted only':null,state.query?`Search: ${state.query}`:null].filter(Boolean).join(' · ');
    const result=await exportRankingsPng({models,totalModels:scopeModelCount,filterLabel,version:state.version,domain:state.domain,judge:state.judge,judgeLabel:$('#judgeFilter').selectedOptions[0].textContent,excludeRefusals:state.excludeRefusals,highlighted:new Set(state.highlighted),brandLogo,brandColor,width:Math.max(1200,Math.min(1800,innerWidth))});
    const href=URL.createObjectURL(result.blob),a=document.createElement('a');a.href=href;a.download=result.filename;a.click();setTimeout(()=>URL.revokeObjectURL(href),30000);toast(`PNG saved · ${models.length} rows`);
  } catch(error){console.error(error);toast(`PNG export failed: ${error.message}`);} finally{button.disabled=false;}
}
function questionsInScope() {
  const query = state.questionQuery.toLowerCase().trim();
  return dataset.questions.filter(q => (state.domain === 'all' || q.domain === state.domain) && (!query || `${q.id} ${q.text} ${q.domain}`.toLowerCase().includes(query)) && (!state.outcome || modelMap.get(state.outcomeModel)?.rows.some(row => row.question_id === q.id && classify(row,state.judge) === state.outcome)));
}
function questionListHTML(questions) {
  const stats = new Map();
  for (const row of filteredRows) { if (!stats.has(row.question_id)) stats.set(row.question_id, []); stats.get(row.question_id).push(row); }
  return questions.length ? questions.map(q => { const summary = summarizeRows(stats.get(q.id) || [], options()); return `<button class="question-item ${q.id === state.question ? 'active' : ''}" data-question="${esc(q.id)}"><div class="question-item-top"><span>${esc(q.id)} · ${esc(q.domain)}</span><strong title="Clear pushback">${summary.rateRows ? pct(summary.greenRate) : '—'}</strong></div><p>${esc(q.text)}</p></button>`; }).join('') : '<div class="empty-state">No matching questions</div>';
}
function questionHTML(questions) {
  const q = dataset.questionMap.get(state.question);
  if (!q) return '<div class="empty-state">Choose a question</div>';
  const index = questions.findIndex(item => item.id === q.id);
  return `<section class="question-header"><div class="question-meta"><button class="response-questions-toggle" data-action="toggle-questions">Questions ▾</button><span>${esc(q.id)}</span><span class="pill">${esc(q.domain)}</span><div class="next-question"><button data-question-step="-1" aria-label="Previous question" ${index <= 0 ? 'disabled' : ''}>←</button><button data-question-step="1" aria-label="Next question" ${index < 0 || index >= questions.length - 1 ? 'disabled' : ''}>→</button></div></div><h2>${esc(q.text)}</h2></section>`;
}
function answerRow(id) { return modelMap.get(id)?.rows.find(row => row.question_id === state.question); }
function answerCardsHTML() {
  const available = [...new Map([...matchingModels(), ...state.compare.map(id => modelMap.get(id)).filter(Boolean)].map(model => [model.id, model])).values()].sort((a, b) => a.provider.localeCompare(b.provider) || a.name.localeCompare(b.name) || effortOrder.indexOf(a.reasoning) - effortOrder.indexOf(b.reasoning));
  return `<div class="answer-switch" role="group" aria-label="Compare answers">${[0, 1].map(index => `<button data-answer-tab="${index}" class="${state.answerTab === index ? 'active' : ''}" aria-pressed="${state.answerTab === index}">${index ? 'B' : 'A'} · ${esc(modelMap.get(state.compare[index])?.name || 'Choose model')}</button>`).join('')}</div><div class="answer-grid">${[0, 1].map(index => {
    const id = state.compare[index], row = answerRow(id), category = row ? classify(row, state.judge) : null;
    const stats = row ? summarizeRows([row], { ...options(), excludeRefusals: false }) : null;
    return `<article class="answer-panel ${state.answerTab === index ? 'active' : ''}" data-answer-pane="${index}"><div class="answer-toolbar"><span class="answer-letter">${index ? 'B' : 'A'}</span><select data-answer-model="${index}" aria-label="Response ${index ? 'B' : 'A'} model"><option value="" ${id ? '' : 'selected'} disabled>Choose a model</option>${available.map(model => `<option value="${esc(model.id)}" ${model.id === id ? 'selected' : ''}>${esc(fullName(model))}</option>`).join('')}</select><button class="icon-button copy-answer" data-copy-answer="${index}" aria-label="Copy response ${index ? 'B' : 'A'}" title="Copy response" disabled><svg viewBox="0 0 20 20" aria-hidden="true"><rect x="7" y="6" width="9" height="11" rx="1.5"/><path d="M12 6V3H3v11h4"/></svg></button></div><div class="answer-metrics">${category ? `<span class="outcome-label ${category}">${outcomeNames[category]}</span>${coverageBadge(usesTwoJudgeFallback(row) ? 1 : 0, true)}<span>${fmt(stats.avgTokens)} tokens</span><span class="metric-end">${money(stats.avgCost)}</span>` : '<span>No response</span>'}</div><div id="answerText${index}" class="answer-text">${row ? '<div class="load-answer" role="status"><span class="spinner"></span>Loading answer…</div>' : '<div class="empty-answer">Choose another model.</div>'}</div></article>`;
  }).join('')}</div>`;
}
function renderResponses() {
  const questions = questionsInScope();
  if (!questions.some(q => q.id === state.question)) state.question = questions[0]?.id || null;
  $('#content').innerHTML = `<div class="responses-layout"><aside class="question-panel ${state.questionsOpen ? 'mobile-open' : ''}"><div class="panel-top"><h2>Questions</h2>${state.outcome ? `<button class="outcome-label ${state.outcome}" data-action="clear-outcome">${shortNames[state.outcome]} ×</button>` : ''}<button class="icon-button response-questions-toggle" data-action="toggle-questions" aria-label="Close questions">×</button></div><div class="question-search"><input type="search" id="questionSearch" placeholder="Search questions…" aria-label="Search questions" value="${esc(state.questionQuery)}"></div><div class="question-list">${questionListHTML(questions)}</div></aside><div class="response-main">${questionHTML(questions)}${answerCardsHTML()}</div></div>`;
  persist(); hydrateAnswers();
}
// Render only a deliberately small Markdown subset. Model output can never inject HTML.
function answerMarkup(text) {
  return text.split(/```[^\n]*\n([\s\S]*?)```/g).map((part, index) => index % 2 ? `<pre><code>${esc(part)}</code></pre>` : esc(part).replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>').replace(/`([^`\n]+)`/g, '<code>$1</code>')).join('');
}
let currentAnswerTexts = [];
async function hydrateAnswers() {
  const generation = ++answerGeneration;
  currentAnswerTexts = [];
  const qid = state.question;
  if (!qid || !state.compare.some(id => answerRow(id))) return;
  try {
    const details = await dataset.getDetails(qid);
    if (generation !== answerGeneration || state.view !== 'responses') return;
    state.compare.forEach((id, index) => {
      const row = answerRow(id), target = $(`#answerText${index}`);
      if (!row || !target) return;
      const text = details.get(row.sample_id);
      if (text === undefined) throw new Error('Response missing.');
      currentAnswerTexts[index] = text;
      target.innerHTML = text.trim() ? answerMarkup(text) : '<div class="empty-answer">Empty response.</div>';
      $(`[data-copy-answer="${index}"]`).disabled = !text;
    });
  } catch (error) {
    if (generation !== answerGeneration || state.view !== 'responses') return;
    document.querySelectorAll('.answer-text').forEach(target => { target.innerHTML = `<div class="empty-answer">Answer unavailable. <button class="text-button" data-action="retry-answers">Retry</button></div>`; });
    console.error(error);
  }
}
function renderView() {
  ++answerGeneration;
  mixSizeObserver?.disconnect();
  if (state.view === 'dashboard') renderDashboard();
  else if (state.view === 'explorer') renderCharts();
  else renderResponses();
}
function focusModel(id) {
  if(!visibleModels.some(model=>model.id===id))return;
  state.focus=id;
  if(state.highlighted.has(id))state.highlighted.delete(id);else state.highlighted.add(id);
  if(state.highlightOnly && !state.highlighted.size)state.highlightOnly=false;
  if(state.highlightOnly){refresh();return;}
  if(state.view==='explorer')rerenderPreservingScroll();
  else{
    const has=visibleModels.some(model=>state.highlighted.has(model.id));
    $('.rank-table')?.classList.toggle('has-highlights',has);
    document.querySelectorAll('.rank-table tbody tr').forEach(row=>{const active=state.highlighted.has(row.dataset.model);row.classList.toggle('highlighted',active);row.setAttribute('aria-selected',String(active));row.querySelector('[data-highlight]').checked=active;});
    if($('#modelInspector')) { $('#modelInspector').outerHTML=inspectorHTML(); observeMixLabels(); }
  }
  updateChrome();
}
function openHighlightedAnswers(){
  const ids=[...state.highlighted].filter(id=>modelMap.has(id));
  state.compare=[...new Set([...ids,...state.compare,...visibleModels.map(model=>model.id)])].slice(0,2);
  changeView('responses');
}
function changeView(view, preserveOutcome = false) {
  $('#columnPicker').open=false;
  if (view === 'responses' && !preserveOutcome) { state.outcome=null; state.outcomeModel=null; }
  if (innerWidth <= 1100) state.inspector = false;
  state.view = view; state.questionsOpen = false;
  if (view === 'responses' && state.compare.length < 2) {
    const candidates = [state.focus, ...visibleModels.map(model => model.id), ...allModels.map(model => model.id)].filter(Boolean);
    state.compare = [...new Set([...state.compare, ...candidates])].slice(0, 2);
  }
  refresh();
}
function selectQuestion(id) {
  if (!dataset.questionMap.has(id)) return;
  state.question = id; state.questionsOpen = false;
  if (state.view !== 'responses') { changeView('responses'); return; }
  const questions = questionsInScope();
  document.querySelectorAll('[data-question]').forEach(button => button.classList.toggle('active', button.dataset.question === id));
  $('.question-panel').classList.remove('mobile-open');
  $('.response-main').innerHTML = questionHTML(questions) + answerCardsHTML();
  persist(); hydrateAnswers();
}
function exportResults() {
  if (!dataset) return;
  const fields = ['model', 'reasoning', 'domain', 'attempts', 'rate_denominator', 'clear_pushback_pct', 'clear_pushback_pct_excluding_refusals', 'clear_pushback_pct_all_attempts', 'clear', 'partial', 'accepted', 'refusal', 'error', 'avg_score', 'avg_total_tokens', 'avg_cost_usd', 'judge', 'exclude_refusals', 'two_judge_answer_count', 'judge_coverage_note', 'bars_exclude_refusals', 'bar_denominator', 'avg_reasoning_tokens', 'reasoning_token_count', 'avg_output_tokens', 'output_token_count', 'usage_row_count'];
  const quote = value => `"${String(value ?? '').replace(/"/g, '""')}"`;
  const rows = visibleModels.map(m => [m.id, m.reasoning, state.domain, m.total, m.rateRows, m.rateRows ? m.greenRate : '', m.greenRateExcludingRefusals, m.greenRateAllAttempts, m.green, m.amber, m.red, m.refusal, m.error, m.avgScore, m.avgTokens, m.avgCost, state.judge, true, m.twoJudgeAnswerCount, judgeCoverageNote(m.twoJudgeAnswerCount), state.excludeRefusals, state.excludeRefusals ? m.nonRefusalRows : m.total, m.avgReasoningTokens, m.reasoningTokenCount, m.avgOutputTokens, m.outputTokenCount, m.usageRowCount]);
  const blob = new Blob(['\uFEFF' + [fields, ...rows].map(row => row.map(quote).join(',')).join('\r\n')], { type: 'text/csv;charset=utf-8' });
  const href = URL.createObjectURL(blob), a = document.createElement('a');
  a.href = href; a.download = `bullshitbench-${state.version}-${state.domain.toLowerCase().replace(/[^a-z0-9]+/g, '-')}.csv`; a.click(); setTimeout(() => URL.revokeObjectURL(href), 1000);
  toast(`CSV saved · ${visibleModels.length} variants`);
}
async function openDataset() {
  state.outcome=null; state.outcomeModel=null; state.questionQuery='';
  const generation = ++loadGeneration; ++answerGeneration;
  $('#content').innerHTML = '<div class="loading-screen" role="status"><span class="spinner"></span>Loading benchmark…</div>';
  $('#suiteSelect').value = state.version;
  dataset = null;
  try {
    const result = await loadBenchmark(state.version);
    if (generation !== loadGeneration) return;
    dataset = result; allModels = groupModels(dataset.rows, options()); modelMap = new Map(allModels.map(model => [model.id, model]));
    const domainOrder = ['software','finance','legal','medical','physics'];
    const domainRank = domain => domainOrder.includes(domain) ? domainOrder.indexOf(domain) : 99;
    domains = [...new Set(dataset.questions.map(q => q.domain))].sort((a,b) => domainRank(a)-domainRank(b) || a.localeCompare(b));
    if (!domains.includes(state.domain)) state.domain = 'all';
    state.compare = state.compare.filter(id => modelMap.has(id));
    state.highlighted=new Set([...state.highlighted].filter(id=>modelMap.has(id)));
    if(!state.highlighted.size)state.highlightOnly=false;
    $('#domainFilter').innerHTML='<option value="all">All domains</option>'+domains.map(domain=>`<option value="${esc(domain)}">${esc(domain[0].toUpperCase()+domain.slice(1))}</option>`).join('');
    const defaults = [allModels[0]?.id, allModels.find(m => /fable-5\.1/.test(m.id) && m.reasoning === 'low')?.id, allModels[1]?.id].filter(Boolean);
    state.compare = [...new Set([...state.compare, ...defaults])].slice(0, 2);
    const providers = [...new Map(allModels.map(m => [m.org, companyName(m.org)]))].sort((a, b) => a[1].localeCompare(b[1]));
    $('#labOptions').innerHTML = [['all', 'All labs'], ...providers].map(([value, label]) => `<button type="button" class="lab-option" role="menuitemradio" aria-label="${esc(label)}" aria-checked="false" tabindex="-1" data-company="${esc(value)}"><span class="lab-option-mark" aria-hidden="true">${companyIcon(value)}</span><span class="lab-option-name">${esc(label)}</span><svg class="lab-check" viewBox="0 0 16 16" aria-hidden="true"><path d="m3 8 3 3 7-7"/></svg></button>`).join('');
    const efforts = [...new Set(allModels.map(m => m.reasoning))].sort((a, b) => effortOrder.indexOf(a) - effortOrder.indexOf(b));
    $('#reasoningFilter').innerHTML = '<option value="all">Any reasoning</option>' + efforts.map(value => `<option value="${esc(value)}">${esc(reasoningLabel(value))}</option>`).join('');
    $('#judgeFilter').innerHTML = '<option value="consensus">Judge average</option>' + [1, 2, 3].map((n, index) => `<option value="judge_${n}">${esc(modelLabel(dataset.panel.judge_models?.[index] || `Judge ${n}`))}</option>`).join('');
    if (!providers.some(([value]) => value === state.provider)) state.provider = 'all';
    if (!efforts.includes(state.reasoning)) state.reasoning = 'all';
    syncControls(); refresh();
  } catch (error) {
    if (generation !== loadGeneration) return;
    $('#content').innerHTML = `<div class="empty-state"><strong>Couldn’t load the benchmark</strong><p>${esc(error.message)}</p><button class="primary-button" data-action="retry-dataset">Try again</button></div>`;
    console.error(error);
  }
}

document.addEventListener('click', async event => {
  const target = event.target.closest('button, [data-model], [data-action], [data-outcome]');
  if (!target) return;
  if (target.dataset.view) { if (dataset) changeView(target.dataset.view); return; }
  if (target.dataset.domain) { state.domain = target.dataset.domain; refresh(); return; }
  if (target.dataset.chart) { state.chartMode = target.dataset.chart; changeView('explorer'); return; }
  if (target.dataset.outcome) { state.questionQuery=''; state.outcome=target.dataset.outcome; state.outcomeModel=target.dataset.segmentModel; state.focus=state.outcomeModel; state.compare=[...new Set([state.focus,...state.compare])].slice(0,2); changeView('responses',true); return; }
  if (target.dataset.sort) {
    state.direction = state.sort === target.dataset.sort ? -state.direction : sortDefinitions.find(item=>item.key===target.dataset.sort)?.direction || -1;
    state.sort = target.dataset.sort; refresh(); return;
  }
  if(target.dataset.inspectModel){state.focus=target.dataset.inspectModel;state.inspector=true;rerenderPreservingScroll();return;}
  if (target.dataset.answerTab !== undefined) {
    state.answerTab = Number(target.dataset.answerTab);
    document.querySelectorAll('[data-answer-tab]').forEach(button => { button.classList.toggle('active', Number(button.dataset.answerTab) === state.answerTab); button.setAttribute('aria-pressed', String(Number(button.dataset.answerTab) === state.answerTab)); });
    document.querySelectorAll('[data-answer-pane]').forEach(panel => panel.classList.toggle('active', Number(panel.dataset.answerPane) === state.answerTab)); return;
  }
  if (target.dataset.question) { selectQuestion(target.dataset.question); return; }
  if (target.dataset.questionStep) {
    const questions = questionsInScope(), index = questions.findIndex(q => q.id === state.question);
    const next = questions[index + Number(target.dataset.questionStep)]; if (next) selectQuestion(next.id); return;
  }
  if (target.dataset.copyAnswer !== undefined) {
    try { await copyText(currentAnswerTexts[Number(target.dataset.copyAnswer)] || ''); toast('Response copied'); }
    catch { toast('Copy unavailable'); } return;
  }
  if (target.dataset.model && !event.target.closest('input')) { focusModel(target.dataset.model); return; }
  switch (target.dataset.action) {
    case 'close-inspector': state.inspector = false; updateChrome(); renderView(); break;
    case 'toggle-highlight': if(state.focus)focusModel(state.focus); break;
    case 'answers': state.compare = [...new Set([state.focus, ...state.compare, ...visibleModels.map(m => m.id)])].filter(Boolean).slice(0, 2); changeView('responses'); break;
    case 'compare': openHighlightedAnswers(); break;
    case 'reset': resetFilters(); break;
    case 'clear-outcome': state.outcome=null; state.outcomeModel=null; renderResponses(); break;
    case 'toggle-questions': state.questionsOpen = !state.questionsOpen; $('.question-panel').classList.toggle('mobile-open', state.questionsOpen); break;
    case 'retry-answers': hydrateAnswers(); break;
    case 'retry-dataset': openDataset(); break;
  }
});
document.addEventListener('change', event => {
  const target = event.target;
  if (target.dataset.column) {
    const column=columnDefinitions.find(column=>column.id===target.dataset.column);
    if (!column) return;
    if (target.checked) state.columns.add(column.id); else state.columns.delete(column.id);
    if (!target.checked && state.sort===column.key) {
      state.sort='greenRate'; state.direction=-1;
      rerenderPreservingScroll(refresh);
    } else {
      updateChrome(); rerenderPreservingScroll();
    }
    return;
  }
  if(target.dataset.highlight){focusModel(target.dataset.highlight);return;}
  if(target.id==='chartColour'){state.colorMode=target.value;renderCharts();persist();return;}
  if (target.dataset.answerModel !== undefined) {
    if (state.outcome) { state.outcome=null; state.outcomeModel=null; $('.question-panel .outcome-label')?.remove(); $('.question-list').innerHTML=questionListHTML(questionsInScope()); }
    state.compare[Number(target.dataset.answerModel)] = target.value;
    $('.answer-switch').remove(); $('.answer-grid').outerHTML = answerCardsHTML(); updateChrome(); hydrateAnswers(); return;
  }
});
$('#modelSearch').addEventListener('input', event => { state.query = event.target.value; clearTimeout(searchTimer); searchTimer = setTimeout(refresh, 90); });
$('#content').addEventListener('input', event => {
  if (event.target.id !== 'questionSearch') return;
  state.questionQuery = event.target.value;
  const questions = questionsInScope();
  $('.question-list').innerHTML = questionListHTML(questions);
});
for (const [id, key] of [['domainFilter','domain'], ['reasoningFilter', 'reasoning'], ['accessFilter', 'access'], ['recentFilter', 'recent'], ['judgeFilter', 'judge']]) $(`#${id}`).addEventListener('change', event => { state[key] = event.target.value; syncControls(); refresh(); });
$('#labOptions').addEventListener('click', event => {
  const button = event.target.closest('[data-company]');
  if (!button) return;
  state.provider = button.dataset.company;
  $('#labPicker').open = false;
  syncControls(); refresh();
  $('#labTrigger').focus({preventScroll:true});
});
function focusCompany(button) {
  if (!button) return;
  document.querySelectorAll('[data-company]').forEach(option => { option.tabIndex = option === button ? 0 : -1; });
  button.focus({preventScroll:true});
  button.scrollIntoView({block:'nearest'});
}
$('#labTrigger').addEventListener('keydown', event => {
  if (!['ArrowDown', 'ArrowUp'].includes(event.key)) return;
  event.preventDefault();
  $('#labPicker').open = true;
  positionPopovers();
  focusCompany($('#labOptions [aria-checked="true"]'));
});
let companyTypeahead = '', companyTypeaheadTimer;
$('#labOptions').addEventListener('keydown', event => {
  const buttons = [...document.querySelectorAll('[data-company]')];
  const index = buttons.indexOf(event.target.closest('[data-company]'));
  if (index < 0) return;
  let next;
  if (event.key === 'ArrowRight') next = index + 1;
  if (event.key === 'ArrowLeft') next = index - 1;
  if (event.key === 'ArrowDown') next = index === 0 ? 1 : index + 2;
  if (event.key === 'ArrowUp') next = index <= 2 ? 0 : index - 2;
  if (event.key === 'Home') next = 0;
  if (event.key === 'End') next = buttons.length - 1;
  if (event.key.length === 1 && event.key !== ' ' && !event.ctrlKey && !event.metaKey && !event.altKey) {
    clearTimeout(companyTypeaheadTimer);
    companyTypeahead += event.key.toLowerCase();
    companyTypeaheadTimer = setTimeout(() => { companyTypeahead = ''; }, 600);
    next = buttons.findIndex(button => companyName(button.dataset.company).toLowerCase().startsWith(companyTypeahead));
    if (next < 0) return;
  }
  if (next !== undefined) {
    event.preventDefault();
    focusCompany(buttons[(next + buttons.length) % buttons.length]);
  }
  if (event.key === 'Tab') {
    $('#labPicker').open = false;
    $('#labTrigger').focus({preventScroll:true});
  }
});
$('#sortFilter').addEventListener('change',event => {
  const selected=sortDefinitions.find(item=>item.key===event.target.value); if(!selected)return;
  state.sort=selected.key;state.direction=selected.direction;
  const column=columnDefinitions.find(item=>item.key===selected.key);if(column)state.columns.add(column.id);
  refresh();
});
$('#bestVariants').addEventListener('change',event => { state.bestVariants=event.target.checked; syncControls(); refresh(); });
$('#excludeRefusals').addEventListener('change', event => { state.excludeRefusals = event.target.checked; refresh(); });
$('#suiteSelect').addEventListener('change', event => { state.version = event.target.value; openDataset(); });
$('#resetFilters').addEventListener('click', resetFilters); $('#clearSearch').addEventListener('click', resetFilters);
$('#pinButton').addEventListener('click',()=>{state.pinned=!state.pinned;updateChrome();});
$('#pngButton').addEventListener('click',downloadChart);
$('#highlightOnly').addEventListener('change',event=>{state.highlightOnly=event.target.checked;refresh();});
$('#clearHighlights').addEventListener('click',()=>{state.highlighted.clear();state.highlightOnly=false;refresh();});
$('#highlightAnswers').addEventListener('click',openHighlightedAnswers);
$('#exportButton').addEventListener('click', exportResults);
$('#scoringButton').addEventListener('click', () => $('#scoringDialog').showModal());
$('[data-close-dialog]').addEventListener('click', () => $('#scoringDialog').close());
$('#scoringDialog').addEventListener('click', event => { if (event.target === $('#scoringDialog')) $('#scoringDialog').close(); });
function positionPopovers() {
  const viewport = window.visualViewport;
  const left = viewport?.offsetLeft || 0, top = viewport?.offsetTop || 0;
  const width = viewport?.width || innerWidth, height = viewport?.height || innerHeight;
  for (const id of popoverIds) {
    const menu = $(`#${id}`);
    if (!menu.open) continue;
    const panel = menu.querySelector('.popover-content');
    const anchor = menu.querySelector('summary').getBoundingClientRect();
    panel.style.maxWidth = `${width - 16}px`;
    panel.style.maxHeight = `${height - 16}px`;
    const panelWidth = panel.getBoundingClientRect().width;
    const panelHeight = Math.min(panel.scrollHeight + 2, height - 16);
    const below = top + height - anchor.bottom - 14;
    const above = anchor.top - top - 14;
    const openAbove = below < Math.min(panelHeight, 180) && above > below;
    const available = Math.max(48, openAbove ? above : below);
    panel.style.maxHeight = `${available}px`;
    panel.style.left = `${Math.max(left + 8, Math.min(anchor.left, left + width - panelWidth - 8))}px`;
    panel.style.top = `${Math.max(top + 8, openAbove ? anchor.top - 6 - Math.min(panelHeight, available) : anchor.bottom + 6)}px`;
  }
}
for (const id of popoverIds) {
  $(`#${id}`).addEventListener('toggle', () => {
    if (!$(`#${id}`).open) return;
    for (const other of popoverIds) if (other !== id) $(`#${other}`).open = false;
    positionPopovers();
    if (id === 'labPicker') {
      companyTypeahead = '';
      focusCompany($('#labOptions [aria-checked="true"]'));
    }
  });
}
window.addEventListener('resize', positionPopovers);
window.visualViewport?.addEventListener('resize', positionPopovers);
window.visualViewport?.addEventListener('scroll', positionPopovers);
document.addEventListener('keydown', event => {
  const typing = /INPUT|SELECT|TEXTAREA/.test(event.target.tagName);
  if (event.key === '/' && !typing) { event.preventDefault(); $('#modelSearch').focus(); }
  if ((event.key === 'Enter' || event.key === ' ') && event.target.matches('[data-outcome]')) { event.preventDefault(); event.target.click(); }
  if ((event.key === 'Enter' || event.key === ' ') && event.target.matches('tr[data-model]')) { event.preventDefault(); focusModel(event.target.dataset.model); }
  if (event.key === 'Escape') {
    const openMenu = popoverIds.find(id => $(`#${id}`).open);
    if (openMenu) {
      $(`#${openMenu}`).open=false;
      $(`#${openMenu} summary`).focus({preventScroll:true});
      event.preventDefault(); return;
    }
    $('#moreFilters').open = false;
    if(state.pinned){state.pinned=false;updateChrome();}
    if (state.questionsOpen) { state.questionsOpen = false; $('.question-panel')?.classList.remove('mobile-open'); }
    if (innerWidth <= 1100 && state.inspector) { state.inspector = false; updateChrome(); renderView(); }
  }
});
document.addEventListener('pointerdown',event=>{for (const id of popoverIds) if(!event.target.closest(`#${id}`))$(`#${id}`).open=false;});
document.addEventListener('focusin', event => { if (!$('#labPicker').contains(event.target)) $('#labPicker').open = false; });
openDataset();

const compactScreen = matchMedia('(max-width:1100px)');
compactScreen.addEventListener('change', event => { if(event.matches) state.inspector=false; if(dataset) { updateChrome(); renderView(); } });
