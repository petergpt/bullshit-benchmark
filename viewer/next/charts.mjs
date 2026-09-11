import { summarizeRows, judgeCoverageNote } from './data.mjs?v=20260907-dashboard';
import { placeLabels } from './labels.mjs?v=20260906-latest-labels';

const NS = 'http://www.w3.org/2000/svg';
const INK = '#1d2521';
const TEAL = '#1d5b50';
const PANEL = '#fcfbf7';
const MUTED = '#5c6861';
const LINE = '#cfd6ce';
const HIGHER = '#23804d';
const LOWER = '#b6493f';
const EFFORT_ORDER = ['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra'];
const chartObservers = new WeakMap();
const number = new Intl.NumberFormat('en', { maximumFractionDigits: 1 });
const dateLabel = new Intl.DateTimeFormat('en', { month: 'short', year: '2-digit', timeZone: 'UTC' });
const fullDate = new Intl.DateTimeFormat('en', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' });

function el(tag, attrs = {}, text = '') {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== null && value !== undefined) node.setAttribute(key, value);
  }
  if (text !== '') node.textContent = text;
  return node;
}

function svg(tag, attrs = {}, text = '') {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== null && value !== undefined) node.setAttribute(key, value);
  }
  if (text !== '') node.textContent = text;
  return node;
}

function finite(value) {
  return value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value));
}

function pct(value) { return finite(value) ? `${number.format(Number(value))}%` : '—'; }
function effort(model) { return String(model.reasoning || 'none').toLowerCase(); }
function effortLabel(model) { const value=effort(model); return value==='xhigh' ? 'xHigh' : value; }
function tokenDetails(model) {
  return [['avgReasoningTokens','reasoningTokenCount','reasoning'],['avgOutputTokens','outputTokenCount','output']].map(([key,count,label])=>finite(model[key]) ? `${number.format(model[key])} ${label} tokens / answer${model[count]<model.usageRowCount ? ` · ${model[count]}/${model.usageRowCount} reported` : ''}` : `${label[0].toUpperCase()+label.slice(1)} tokens not reported`);
}
function effortIndex(model) {
  const index = EFFORT_ORDER.indexOf(effort(model));
  return index === -1 ? 4 : index;
}
function selectedModel(selected, id) {
  return selected instanceof Set ? selected.has(id) : Array.isArray(selected) ? selected.includes(id) : selected === id || selected?.id === id;
}
function baseOf(model) { return model.base || String(model.id || '').replace(/@reasoning=[^@]+$/i, ''); }
function metaFor(metadata, model) {
  return metadata instanceof Map ? metadata.get(baseOf(model)) || {} : metadata?.[baseOf(model)] || {};
}
function nameOf(model) { return String(model.name || model.id || 'Unknown model'); }
function providerOf(model) { return model.org || baseOf(model).split('/')[0] || 'unknown'; }
function detectionColor(rate) {
  const t = Math.max(0, Math.min(100, Number(rate) || 0)) / 50;
  const from = t <= 1 ? [182, 73, 63] : [185, 120, 22];
  const to = t <= 1 ? [185, 120, 22] : [35, 128, 77];
  return `rgb(${from.map((channel, index) => Math.round(channel + (to[index] - channel) * (t <= 1 ? t : t - 1))).join(' ')})`;
}
function colorOf(model, options) {
  return options.colorMode === 'detection' ? detectionColor(model.greenRate) : options.brandColor?.(providerOf(model)) || TEAL;
}

const STYLES = `
.next-chart{position:relative;display:flex;flex-direction:column;min-width:0;min-height:0;height:100%;color:${INK};font:inherit;font-weight:400;--chart-ink:${INK};--chart-line:${LINE}}
.next-chart *{box-sizing:border-box}.next-chart button{font-family:inherit}
.next-chart svg{stroke:none}
.next-chart-meta{display:flex;flex:none;align-items:center;justify-content:space-between;gap:8px;min-height:28px;padding:0 0 5px;color:${MUTED};font-size:12px;line-height:1.3}
.next-chart-meta strong{font-size:12px;font-weight:500;color:${INK}}.next-chart-key{display:flex;align-items:center;justify-content:flex-end;gap:8px;flex-wrap:wrap}
.next-chart-plot{display:block;width:100%;height:auto;overflow:visible;font-family:inherit}
.next-chart-plot-scroll{flex:1;min-height:220px;max-width:100%;overflow:auto;scrollbar-width:thin}.next-chart-plot-scroll .next-chart-plot{min-width:320px}
.next-chart-point-label{pointer-events:none}.next-chart-label-text{paint-order:stroke;stroke-linejoin:round}
.next-chart-mark{cursor:pointer;outline:none}.next-chart-mark:hover circle,.next-chart-mark:focus circle{stroke:${INK};stroke-width:2.5}.next-chart-mark:focus text{fill:${INK};font-weight:600}
.next-chart-mark.is-selected circle{stroke:${INK};stroke-width:2.5}.next-chart-mark.is-selected{opacity:1!important}
.next-chart-tooltip{position:absolute;z-index:8;max-width:260px;padding:7px 9px;border:1px solid ${LINE};border-radius:3px;background:${PANEL};color:${INK};box-shadow:0 2px 6px #1d252118;font-size:12px;line-height:1.4;pointer-events:none;transform:translateY(-100%)}
.next-chart-tooltip[hidden]{display:none}.next-chart-tooltip strong{display:block;font-size:12px;font-weight:500}.next-chart-tooltip span{display:block;color:${MUTED}}
.next-chart-empty{flex:1;min-height:220px;display:grid;place-content:center;gap:4px;text-align:center;color:${MUTED};font-size:12px}.next-chart-empty strong{color:${INK};font-size:14px;font-weight:500}
.next-chart-scroll{flex:1;min-height:0;overflow:auto;scrollbar-width:thin;border:1px solid ${LINE};border-radius:3px;background:${PANEL}}
.next-chart-table{width:100%;border-collapse:separate;border-spacing:0;text-align:left;font-size:12px;line-height:1.2;font-variant-numeric:tabular-nums}
.next-chart-table th{position:sticky;top:0;z-index:2;background:#f1ede4;padding:6px;border-bottom:1px solid ${LINE};font-weight:500;color:${INK};white-space:nowrap;text-align:center}
.next-chart-table th:first-child{text-align:left;left:0;z-index:3;min-width:170px;max-width:250px}.next-chart-table td{height:29px;padding:2px;border-bottom:1px solid ${LINE};text-align:center}.next-chart-table td:first-child{position:sticky;left:0;z-index:1;background:${PANEL};text-align:left;padding:0 7px;max-width:250px}
.next-chart-table tr:last-child td{border-bottom:0}.next-chart-table tr.is-selected td:first-child{background:#e6efe9;box-shadow:inset 3px 0 ${TEAL}}.next-chart-table tr:hover td:first-child{background:#f1ede4}
.next-chart-model{display:flex;align-items:center;gap:5px;border:0;background:none;color:${INK};padding:3px 0;width:100%;min-width:155px;text-align:left;cursor:pointer;font-size:12px;font-weight:500;line-height:1.2}
.next-chart-model-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:220px}.next-chart-effort{font-size:10px;color:${MUTED};font-weight:500;white-space:nowrap}
.next-chart-cell{display:block;min-width:52px;width:100%;padding:3px 4px;border:1px solid transparent;border-radius:2px;font-size:12px;font-weight:500;line-height:1.2;font-variant-numeric:tabular-nums;cursor:pointer}.next-chart-cell:hover,.next-chart-cell:focus-visible{border-color:${TEAL};outline:none}.next-chart-cell[disabled]{cursor:default;color:#89928f;background:#f1ede4}
.next-chart-heat-key{height:6px;width:54px;border-radius:1px;background:linear-gradient(90deg,${PANEL},#c6d6cc,#7b9d8c,${TEAL})}
.next-chart-family{flex:1;min-height:0;overflow:auto;scrollbar-width:thin;border:1px solid ${LINE};border-radius:3px;background:${PANEL}}.next-chart-family-axis{display:grid;grid-template-columns:minmax(140px,29%) 1fr 44px;gap:8px;position:sticky;top:0;z-index:2;background:#f1ede4;border-bottom:1px solid ${LINE};padding:1px 9px;color:${INK};font-size:11px;align-items:center}.next-chart-family-axis svg{width:100%;height:25px;display:block;overflow:visible}
.next-chart-family-row{display:grid;grid-template-columns:minmax(140px,29%) 1fr 44px;gap:8px;align-items:center;padding:0 9px;border-bottom:1px solid ${LINE};height:30px}.next-chart-family-row:last-child{border-bottom:0}.next-chart-family-row:hover{background:#f1ede4}.next-chart-family-row.is-selected{background:#e6efe9;box-shadow:inset 3px 0 ${TEAL}}
.next-chart-family-title{font-size:12px;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.next-chart-family-row svg{width:100%;height:28px;overflow:visible;display:block}.next-chart-delta{font-size:12px;text-align:right;font-variant-numeric:tabular-nums;color:${MUTED}}.next-chart-delta.up{color:#23804d}.next-chart-delta.down{color:#b6493f}
.next-chart-foot{display:flex;flex:none;justify-content:space-between;gap:8px;padding-top:4px;font-size:11px;color:${MUTED};line-height:1.3}.next-chart-foot button{color:${TEAL};background:none;border:0;padding:0;cursor:pointer;font-size:inherit;text-align:left}
.next-chart.is-compact .next-chart-meta{height:26px;min-height:26px;padding-bottom:4px;font-size:11px}.next-chart.is-compact .next-chart-meta strong{font-size:12px}.next-chart.is-compact .next-chart-key{gap:5px;font-size:11px}.next-chart.is-compact .next-chart-table th{padding:4px;font-size:11px}.next-chart.is-compact .next-chart-table th:first-child{min-width:138px;max-width:205px}.next-chart.is-compact .next-chart-table td{height:24px;padding:1px;border-bottom-color:#cfd6ce80}.next-chart.is-compact .next-chart-table td:first-child{padding-left:6px;padding-right:6px;max-width:205px}.next-chart.is-compact .next-chart-model{min-width:130px;padding:2px 0;font-size:12px}.next-chart.is-compact .next-chart-model-name{max-width:170px}.next-chart.is-compact .next-chart-cell{min-width:40px;padding:2px 3px;font-size:11px}.next-chart.is-compact .next-chart-family-row{height:26px}.next-chart.is-compact .next-chart-family-row svg{height:24px}.next-chart.is-compact .next-chart-foot{display:none}
.next-chart-color-key{display:flex;flex:none;align-items:center;justify-content:flex-end;gap:6px;height:22px;min-height:22px;margin-bottom:2px;white-space:nowrap;font-size:11px;color:${MUTED}}.next-chart-color-ramp{height:6px;width:64px;flex:none;border-radius:1px;background:linear-gradient(90deg,#b6493f,#b97816,#23804d)}
.next-chart-reasoning-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;flex:1;min-height:0}.next-chart-reasoning-panel{display:flex;flex-direction:column;min-width:0;min-height:0;border:1px solid ${LINE};border-radius:3px;background:${PANEL};overflow:hidden}.next-chart-reasoning-title{display:flex;flex:none;align-items:center;justify-content:space-between;gap:8px;min-height:29px;padding:4px 8px;border-bottom:1px solid ${LINE};background:#f1ede4}.next-chart-reasoning-title strong{font-size:12px;font-weight:500}.next-chart-reasoning-title span{font-size:11px;color:${MUTED}}.next-chart-reasoning-panel .next-chart-family{border:0;border-radius:0}.next-chart-reasoning-panel .next-chart-family-axis,.next-chart-reasoning-panel .next-chart-family-row{grid-template-columns:minmax(115px,38%) minmax(80px,1fr) 37px;gap:6px;padding-left:7px;padding-right:7px}.next-chart-reasoning-panel .next-chart-family-row{height:28px}.next-chart-reasoning-panel .next-chart-family-title{display:flex;flex-direction:column;justify-content:center;line-height:1.1;font-size:11px;min-width:0}.next-chart-family-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.next-chart-pair{font-size:10px;font-weight:400;color:${MUTED}}.next-chart-reasoning-panel .next-chart-family-row svg{height:26px}.next-chart-reasoning-panel .next-chart-empty{min-height:80px}.next-chart-reasoning-panel .next-chart-delta{font-size:11px}
.next-chart-reasoning-panel.is-higher{--change:${HIGHER};--change-bg:#e5f0e3}.next-chart-reasoning-panel.is-lower{--change:${LOWER};--change-bg:#f7e7e1}
.next-chart-reasoning-panel .next-chart-reasoning-title{background:var(--change-bg);color:var(--change);justify-content:flex-start;gap:5px}.next-chart-reasoning-title .direction-arrow{font-size:15px;font-weight:600;color:inherit}.next-chart-reasoning-title strong{font-weight:500}
.next-chart-reasoning-panel .next-chart-delta{font-weight:500;color:var(--change)}.next-chart-reasoning-panel .next-chart-family-title{flex-direction:row;align-items:center;justify-content:flex-start;gap:6px}.next-chart-family-title img{width:15px;height:15px;object-fit:contain;flex:none}.next-chart-family-text{display:flex;flex-direction:column;min-width:0}
.next-chart-unchanged{flex:none;max-height:30%;overflow:auto;border:1px solid ${LINE};border-radius:3px;margin-top:6px;background:#f1ede4;font-size:11px;color:${MUTED}}.next-chart-unchanged summary{cursor:pointer;padding:5px 8px;font-weight:500}.next-chart-unchanged-items{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:1px;background:${LINE};border-top:1px solid ${LINE}}.next-chart-unchanged-row{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:5px 8px;background:${PANEL};min-width:0}.next-chart-unchanged-name{overflow:hidden;white-space:nowrap;text-overflow:ellipsis;font-weight:500}.next-chart-unchanged-scores{display:flex;gap:4px;align-items:center;flex:none}.next-chart-unchanged-score{background:transparent;border:1px solid ${LINE};border-radius:3px;padding:3px 5px;color:${MUTED};font:inherit;cursor:pointer}.next-chart-unchanged-score:hover,.next-chart-unchanged-score.is-selected{border-color:${TEAL};background:#e6efe9;color:${INK}}
.next-chart-trend-heading{flex:none;margin:0;padding:2px 0 4px;font-size:18px;font-weight:600;line-height:1.25}
.next-chart.is-lab-trends .next-chart-plot{min-width:0}
.next-chart-trend-note{flex:none;padding:0 0 10px;font-size:11px;line-height:1.4;color:${MUTED}}
.next-chart-lab-summary{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;flex:none;padding:0 0 14px}
.next-chart-lab{min-width:0;border-top:3px solid var(--lab-color);padding-top:6px}
.next-chart-lab-name{display:flex;align-items:center;gap:6px;font-size:12px;line-height:1.3;font-weight:500}.next-chart-lab-name img{width:17px;height:17px;object-fit:contain;flex:none}.next-chart-lab-name b{margin-left:auto;font-weight:600;font-variant-numeric:tabular-nums}
.next-chart-lab-model{display:block;overflow-wrap:anywhere;padding-top:4px;color:${MUTED};font-size:11px;line-height:1.4}
.next-chart-lab-all-rate{display:block;padding-top:3px;color:${MUTED};font-size:10px;line-height:1.4}
.next-chart-rate-key{display:flex;align-items:center;flex-wrap:wrap;gap:8px 20px;flex:none;margin-bottom:7px;min-height:22px;font-size:11px;color:${MUTED}}
.next-chart-line-key{display:inline-flex;align-items:center;gap:6px;white-space:nowrap}.next-chart-line-key i{width:24px;border-top:2px solid ${MUTED}}.next-chart-line-key.is-dotted i{border-top-style:dashed;opacity:.65}.next-chart-line-key input{width:12px;height:12px;accent-color:${TEAL};margin:0}.next-chart-line-key:has(input){cursor:pointer}
@media(max-width:640px){.next-chart-trend-heading{font-size:15px}.next-chart-trend-note{font-size:10px;padding-bottom:8px}.next-chart-lab-summary{gap:10px;padding-bottom:10px}.next-chart-lab-name{gap:4px;flex-wrap:wrap;font-size:11px}.next-chart-lab-name img{width:15px;height:15px}.next-chart-lab-name b{width:100%;margin:0;font-size:15px}.next-chart-lab-model{font-size:10px}}
@media(max-width:760px){.next-chart-reasoning-grid{grid-template-columns:1fr;grid-template-rows:repeat(2,minmax(0,1fr))}.next-chart-reasoning-panel .next-chart-family-axis,.next-chart-reasoning-panel .next-chart-family-row{grid-template-columns:minmax(120px,35%) minmax(80px,1fr) 37px}}
@media(max-width:640px){.next-chart-meta{align-items:center;gap:6px}.next-chart-key{gap:5px;font-size:11px}.next-chart-table th:first-child{min-width:150px}.next-chart-model-name{max-width:170px}.next-chart-table td:first-child{max-width:220px}.next-chart-family-axis,.next-chart-family-row{grid-template-columns:minmax(100px,34%) 1fr 36px;gap:6px;padding-left:6px;padding-right:6px}.next-chart-family-title{font-size:11px}.next-chart-foot{font-size:10px}}
`;

function detectionLegend(container, options) {
  if (options.colorMode !== 'detection') return;
  const key = el('div', { class: 'next-chart-color-key', 'aria-label': 'Color shows clear pushback rate' });
  key.append(el('span', {}, '0%'), el('span', { class: 'next-chart-color-ramp' }), el('span', {}, '100%'));
  container.append(key);
}

function header(container, strong, right) {
  const node = el('div', { class: 'next-chart-meta' });
  node.append(el('strong', {}, strong));
  if (typeof right === 'string') node.append(el('span', {}, right));
  else if (right) node.append(right);
  container.append(node);
}

function empty(container, title, subtitle) {
  const node = el('div', { class: 'next-chart-empty', role: 'status' });
  node.append(el('strong', {}, title));
  if (subtitle) node.append(el('span', {}, subtitle));
  container.append(node);
}

function tooltipFor(container) {
  const tip = el('div', { class: 'next-chart-tooltip', hidden: '' });
  container.append(tip);
  function show(target, title, detail, event) {
    tip.replaceChildren(el('strong', {}, title), ...detail.map(line => el('span', {}, line)));
    tip.hidden = false;
    const box = container.getBoundingClientRect();
    const mark = target.getBoundingClientRect();
    const x = event?.clientX ?? mark.x + mark.width / 2;
    const y = event?.clientY ?? mark.y;
    const left = Math.min(Math.max(8, x - box.x + 12), Math.max(8, box.width - tip.offsetWidth - 8));
    tip.style.left = `${left}px`;
    tip.style.top = `${Math.max(tip.offsetHeight + 5, y - box.y - 12)}px`;
  }
  return { show, hide: () => { tip.hidden = true; } };
}

function updateSelection(container, options, id) {
  const selected = new Set(options.selected instanceof Set || Array.isArray(options.selected) ? options.selected : options.selected ? [options.selected.id || options.selected] : []);
  if (selected.has(id)) selected.delete(id); else selected.add(id);
  options.selected = selected;
  container.querySelectorAll('[data-model]').forEach(node => {
    const chosen = selected.has(node.dataset.model);
    node.classList.toggle('is-selected', chosen);
    if (node.hasAttribute('aria-pressed')) node.setAttribute('aria-pressed', String(chosen));
    if (node.classList.contains('next-chart-mark')) {
      const circle = node.querySelector('circle');
      if (circle) {
        circle.setAttribute('r', Number(circle.dataset.radius) + (chosen ? 2 : 0));
        circle.setAttribute('fill', circle.dataset.color);
        circle.setAttribute('fill-opacity', chosen ? '1' : circle.dataset.opacity);
      }
      if (chosen) node.parentElement.append(node);
    }
  });
  container.querySelectorAll('.next-chart-family-row').forEach(row => {
    row.classList.toggle('is-selected', !!row.querySelector('.next-chart-mark.is-selected'));
  });
  options.onSelect?.(id);
}

function selectFrom(event, container, options, id) {
  // The host also delegates clicks by data-model; dispatch exactly one selection.
  event.stopPropagation();
  updateSelection(container, options, id);
}

function rateDetails(model, options) {
  const primary = `${pct(model.greenRate)} clear`;
  if (!model.refusal) return [`${primary} · ${number.format(model.total)} attempts`];
  const other = options.excludeRefusals ? model.greenRateAllAttempts : model.greenRateExcludingRefusals;
  const comparison = finite(other) && pct(other) !== pct(model.greenRate)
    ? ` · ${pct(other)} ${options.excludeRefusals ? 'incl.' : 'excl.'} refusals` : '';
  return [`${primary}${comparison}`, `${number.format(model.total)} attempts · ${model.refusal} refusals`];
}
function interactiveMark(group, model, options, tooltip, detail) {
  detail = [...rateDetails(model, options), ...detail.filter(Boolean)];
  if (model.twoJudgeAnswerCount) detail = [...detail, judgeCoverageNote(model.twoJudgeAnswerCount)];
  const label = `${nameOf(model)}${effort(model) !== 'none' ? `, ${effortLabel(model)}` : ''}`;
  group.setAttribute('tabindex', '0');
  group.setAttribute('role', 'button');
  group.setAttribute('aria-label', `${label}: ${detail.join(', ')}. Toggle highlight.`);
  group.setAttribute('aria-pressed', String(selectedModel(options.selected, model.id)));
  group.dataset.model = model.id;
  group.classList.add('next-chart-mark');
  const circle = group.querySelector('circle');
  if (circle) {
    circle.dataset.radius = circle.getAttribute('r');
    circle.dataset.color = circle.getAttribute('fill');
    circle.dataset.opacity = circle.getAttribute('fill-opacity') || '1';
  }
  if (selectedModel(options.selected, model.id)) {
    group.classList.add('is-selected');
    if (circle) {
      circle.setAttribute('r', Number(circle.dataset.radius) + 2);
      circle.setAttribute('fill', circle.dataset.color);
      circle.setAttribute('fill-opacity', '1');
    }
  }
  group.append(svg('title', {}, `${label}\n${detail.join('\n')}`));
  group.addEventListener('click', event => selectFrom(event, group.closest('.next-chart'), options, model.id));
  group.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      selectFrom(event, group.closest('.next-chart'), options, model.id);
    }
    if (['ArrowLeft', 'ArrowUp', 'ArrowRight', 'ArrowDown'].includes(event.key)) {
      event.preventDefault();
      const marks = [...group.closest('.next-chart').querySelectorAll('.next-chart-mark')];
      const delta = event.key === 'ArrowLeft' || event.key === 'ArrowUp' ? -1 : 1;
      marks[(marks.indexOf(group) + delta + marks.length) % marks.length]?.focus();
    }
  });
  group.addEventListener('pointerenter', event => tooltip.show(group, label, detail, event));
  group.addEventListener('pointermove', event => tooltip.show(group, label, detail, event));
  group.addEventListener('pointerleave', tooltip.hide);
  group.addEventListener('focus', () => tooltip.show(group, label, detail));
  group.addEventListener('blur', tooltip.hide);
}

function renderDomains(container, options) {
  const questions = new Map((options.questions || []).map(question => [String(question.id), question]));
  const domainOf = row => String(row.domain || questions.get(String(row.question_id))?.domain || 'Other');
  const domains = [...new Set(options.models.flatMap(model => (model.rows || []).map(domainOf)))].sort((a, b) => a.localeCompare(b));
  if (!domains.length) return empty(container, 'No domain results', 'Try another filter.');
  const key = el('div', { class: 'next-chart-key', 'aria-label': 'Detection rate color scale, 0 to 100 percent' });
  key.append(el('span', {}, '0'), el('span', { class: 'next-chart-heat-key' }), el('span', {}, '100% detected'));
  header(container, `${options.models.length} models · ${domains.length} domains`, key);
  const scroll = el('div', { class: 'next-chart-scroll', tabindex: '0', 'aria-label': 'Domain comparison, scroll for more models' });
  const table = el('table', { class: 'next-chart-table', 'aria-label': 'Detection rate by model and domain' });
  const head = el('thead');
  const top = el('tr');
  top.append(el('th', { scope: 'col' }, 'Model'));
  domains.forEach(domain => top.append(el('th', { scope: 'col' }, domain.replace(/\s*&\s*/g, ' & '))));
  top.append(el('th', { scope: 'col' }, 'Overall'));
  head.append(top);
  const body = el('tbody');
  const sorted = [...options.models].sort((a, b) => b.greenRate - a.greenRate || nameOf(a).localeCompare(nameOf(b)));
  sorted.forEach(model => {
    const row = el('tr', { class: selectedModel(options.selected, model.id) ? 'is-selected' : '', 'data-model': model.id });
    const labelCell = el('td');
    const button = el('button', { type: 'button', class: 'next-chart-model', 'data-model': model.id, title: `${nameOf(model)} · ${effortLabel(model)}`, 'aria-pressed': String(selectedModel(options.selected, model.id)) });
    button.append(el('span', { class: 'next-chart-model-name' }, nameOf(model)));
    if (effort(model) !== 'none') button.append(el('span', { class: 'next-chart-effort' }, effortLabel(model)));
    button.addEventListener('click', event => selectFrom(event, container, options, model.id));
    row.addEventListener('click', event => selectFrom(event, container, options, model.id));
    labelCell.append(button);
    row.append(labelCell);
    const byDomain = new Map(domains.map(domain => [domain, []]));
    (model.rows || []).forEach(entry => byDomain.get(domainOf(entry))?.push(entry));
    [...domains, 'Overall'].forEach(domain => {
      const summary = domain === 'Overall' ? model : summarizeRows(byDomain.get(domain), { judge: options.judge, excludeRefusals: options.excludeRefusals });
      const rate = summary.greenRate;
      const cell = el('td');
      const known = (summary.rateRows ?? summary.total) > 0 && finite(rate);
      const value = known ? Number(rate) : 0;
      const color = value >= 76 ? PANEL : INK;
      const heat = `rgb(${[252, 251, 247].map((channel, index) => Math.round(channel + ([29, 91, 80][index] - channel) * value / 100)).join(' ')})`;
      const title = known ? `${nameOf(model)} · ${domain}: ${pct(rate)} detected · ${summary.green}/${summary.rateRows ?? Math.max(0, summary.total - (options.excludeRefusals ? summary.refusal : 0))} responses` : `${nameOf(model)} · ${domain}: no results`;
      const valueButton = el('button', { type: 'button', class: 'next-chart-cell', title, 'aria-label': title, style: known ? `background:${heat};color:${color}${domain === 'Overall' ? ';box-shadow:inset 0 0 0 1px #1d5b5028' : ''}` : '', disabled: !known ? '' : null }, known ? number.format(value) : '—');
      valueButton.addEventListener('click', event => selectFrom(event, container, options, model.id));
      cell.append(valueButton);
      row.append(cell);
    });
    body.append(row);
  });
  table.append(head, body);
  scroll.append(table);
  container.append(scroll);
}

function dateTicks(start, end) {
  const span = end - start;
  const month = 30.4375 * 86400000;
  const step = span > month * 32 ? 12 : span > month * 16 ? 6 : span > month * 8 ? 3 : span > month * 3 ? 1 : 0;
  if (!step) return Array.from({ length: 5 }, (_, index) => start + span * index / 4);
  const first = new Date(start);
  let year = first.getUTCFullYear();
  let m = Math.ceil(first.getUTCMonth() / step) * step;
  let next = Date.UTC(year, m, 1);
  if (next < start) next = Date.UTC(year, m += step, 1);
  const ticks = [];
  while (next <= end && ticks.length < 12) {
    ticks.push(next);
    next = Date.UTC(year, m += step, 1);
  }
  return ticks.length > 1 ? ticks : [start, end];
}

function logTicks(start, end) {
  const ticks = [];
  for (let power = Math.floor(Math.log10(start)); power <= Math.ceil(Math.log10(end)); power++) {
    for (const factor of [1, 2, 5]) {
      const value = factor * 10 ** power;
      if (value >= start && value <= end) ticks.push(value);
    }
  }
  return ticks.length ? ticks : [start, end];
}

function chartGeometry(container, compact) {
  const computed = getComputedStyle(container);
  const innerWidth = container.clientWidth - (parseFloat(computed.paddingLeft) || 0) - (parseFloat(computed.paddingRight) || 0);
  const innerHeight = container.clientHeight - (parseFloat(computed.paddingTop) || 0) - (parseFloat(computed.paddingBottom) || 0);
  const width = Math.max(container.classList.contains('is-lab-trends') ? 260 : 320, Math.round(innerWidth || 1000));
  const keyHeight = [...container.querySelectorAll(':scope > .next-chart-color-key, :scope > .next-chart-trend-heading, :scope > .next-chart-trend-note, :scope > .next-chart-lab-summary, :scope > .next-chart-rate-key')].reduce((sum, node) => sum + node.getBoundingClientRect().height + (parseFloat(getComputedStyle(node).marginBottom) || 0), 0);
  const fallbackHeight = compact ? 250 : Math.max(300, window.innerHeight - container.getBoundingClientRect().top - keyHeight - 24);
  const height = Math.max(220, Math.round(innerHeight > 70 ? innerHeight - keyHeight : fallbackHeight));
  return { width, height };
}

function labelPoints(plot, marked, bounds) {
  const layer = svg('g', { class: 'next-chart-labels', 'aria-hidden': 'true' });
  plot.append(layer);
  const selected = marked.filter(point => point.chosen);
  const families = new Map(), leaders = new Map(), variantCounts = new Map();
  for (const point of [...marked].sort((a, b) => b.model.greenRate - a.model.greenRate || a.model.id.localeCompare(b.model.id))) {
    const family = baseOf(point.model), provider = providerOf(point.model);
    variantCounts.set(family, (variantCounts.get(family) || 0) + 1);
    if (!families.has(family)) families.set(family, point);
    if (!leaders.has(provider)) leaders.set(provider, point.model.id);
  }
  const selectedFamilies = new Set(selected.map(point => baseOf(point.model)));
  const candidates = [...selected, ...[...families.values()].filter(point => !selectedFamilies.has(baseOf(point.model)))];
  const nodes = new Map();
  const fontSize = bounds.width < 500 ? 10 : 11;
  const measured = candidates.map(point => {
    const { model, chosen, latest, x, y } = point;
    const suffix = variantCounts.get(baseOf(model)) > 1 ? ` · ${effortLabel(model)}` : '';
    const label = `${nameOf(model)}${suffix}`;
    const group = svg('g', { class: `next-chart-point-label${chosen ? ' is-selected' : ''}${latest ? ' is-latest' : ''}`, 'data-model': model.id, visibility: 'hidden' });
    const node = svg('text', { class: 'next-chart-label-text', x: 0, y: 0, fill: chosen ? TEAL : INK, 'font-size': fontSize, 'font-weight': chosen || latest ? 600 : 500, stroke: PANEL, 'stroke-width': 4 }, label);
    group.append(node); layer.append(group);
    let box = node.getBBox();
    const maxWidth = Math.min(260, bounds.width - 12);
    if (box.width > maxWidth) {
      let name = nameOf(model);
      while (name.length > 4 && box.width > maxWidth) {
        name = name.slice(0, -1);
        node.textContent = `${name.trimEnd()}…${suffix}`;
        box = node.getBBox();
      }
    }
    nodes.set(model.id, { group, node, box });
    return { id: model.id, x, y, width: box.width + 8, height: box.height + 6, selected: chosen, preferred: latest, priority: (leaders.get(providerOf(model)) === model.id ? 20 : 0) + model.greenRate / 10 };
  });
  const limit = Math.max(candidates.filter(point => point.chosen || point.latest).length, Math.min(28, Math.max(6, Math.floor(bounds.width * bounds.height / 32000))));
  const placements = placeLabels(measured, { bounds, points: marked.map(point => ({ x: point.x, y: point.y, radius: point.chosen ? 6.3 : 4.3 })), limit });
  const kept = new Set();
  for (const { id, rect, line, distance, selected: chosen } of placements) {
    kept.add(id);
    const { group, node, box } = nodes.get(id);
    node.setAttribute('x', rect.x + 4 - box.x);
    node.setAttribute('y', rect.y + 3 - box.y);
    group.removeAttribute('visibility');
    if (distance > 14) group.insertBefore(svg('line', { class: 'next-chart-label-leader', x1: line.from.x, y1: line.from.y, x2: line.to.x, y2: line.to.y, stroke: chosen ? TEAL : '#88958b', 'stroke-width': chosen ? 1 : .8, 'stroke-opacity': chosen ? .8 : .6 }), node);
    if (chosen) group.insertBefore(svg('rect', { x: rect.x, y: rect.y, width: rect.width, height: rect.height, rx: 3, fill: PANEL, stroke: '#86a292', 'stroke-width': .7 }), node);
  }
  for (const [id, { group }] of nodes) if (!kept.has(id)) group.remove();
}

function scatter(container, options, config) {
  const points = config.points.filter(point => finite(point.x) && finite(point.model.greenRate) && (point.model.rateRows ?? point.model.total) > 0);
  if (!points.length) return empty(container, config.emptyTitle, config.emptyDetail);
  const omissions = config.missingText || `${options.models.length - points.length} without ${config.missingLabel}`;
  container.setAttribute('aria-description', omissions);
  detectionLegend(container, options);
  const { width, height } = chartGeometry(container, options.compact);
  const margin = { left: width < 500 ? 42 : 48, right: width < 500 ? 12 : 20, top: options.compact ? 22 : 26, bottom: options.compact ? 42 : 50 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const values = points.map(point => point.x);
  let start = Math.min(...values);
  let end = Math.max(...values);
  if (config.domain) {
    [start, end] = config.domain;
  } else if (config.log) {
    start = Math.max(Number.MIN_VALUE, start / 1.2);
    end *= 1.2;
  } else {
    const padding = Math.max((end - start) * 0.05, 86400000 * 10);
    start -= padding;
    end += padding;
  }
  const scaleX = config.log
    ? value => margin.left + (Math.log10(value) - Math.log10(start)) / (Math.log10(end) - Math.log10(start)) * plotWidth
    : value => margin.left + (value - start) / (end - start) * plotWidth;
  const scaleY = value => margin.top + (100 - Math.min(100, Math.max(0, value))) / 100 * plotHeight;
  const plot = svg('svg', { class: 'next-chart-plot', viewBox: `0 0 ${width} ${height}`, 'aria-label': config.label, role: 'group' });
  plot.append(svg('title', {}, config.label));
  for (const value of [0, 25, 50, 75, 100]) {
    const y = scaleY(value);
    plot.append(svg('line', { x1: margin.left, x2: width - margin.right, y1: y, y2: y, stroke: LINE, 'stroke-dasharray': value ? '3 5' : null }));
    plot.append(svg('text', { x: margin.left - 10, y: y + 4, fill: MUTED, 'font-size': 12, 'text-anchor': 'end' }, `${value}%`));
  }
  plot.append(svg('text', { x: margin.left, y: 13, fill: MUTED, 'font-size': 12 }, 'Clear pushback'));
  const allTicks = config.ticks || (config.log ? logTicks(start, end) : dateTicks(start, end));
  const maxTicks = Math.max(2, Math.floor(plotWidth / 85));
  const ticks = allTicks.length <= maxTicks ? allTicks : Array.from({ length: maxTicks }, (_, index) => allTicks[Math.round(index * (allTicks.length - 1) / (maxTicks - 1))]);
  ticks.forEach(value => {
    const x = scaleX(value);
    plot.append(svg('line', { x1: x, x2: x, y1: height - margin.bottom, y2: height - margin.bottom + 5, stroke: '#aab5ad' }));
    plot.append(svg('text', { x, y: height - margin.bottom + 20, fill: MUTED, 'font-size': 12, 'text-anchor': 'middle' }, config.tick(value)));
  });
  plot.append(svg('text', { x: margin.left + plotWidth / 2, y: height - 5, fill: MUTED, 'font-size': 12, 'text-anchor': 'middle' }, config.xLabel));
  const plotScroll = el('div', { class: 'next-chart-plot-scroll', tabindex: '0', 'aria-label': config.label });
  plotScroll.append(plot);
  container.append(plotScroll);
  const tooltip = tooltipFor(container);
  if (config.connectLabs) {
    const series = new Map();
    for (const point of points) {
      const org = providerOf(point.model);
      if (!series.has(org)) series.set(org, []);
      series.get(org).push(point);
    }
    for (const [org, releases] of series) {
      releases.sort((a, b) => a.x - b.x);
      // Keep the exact selected model at every date; only the denominator changes.
      if (options.excludeRefusals && options.showRefusalComparison && releases.some(point => Math.abs(point.model.greenRate - point.model.greenRateAllAttempts) > .01)) {
        const comparison = releases.map((point, index) => `${index ? 'L' : 'M'}${scaleX(point.x)} ${scaleY(point.model.greenRateAllAttempts)}`).join(' ');
        plot.append(svg('path', { class: 'next-chart-lab-comparison', 'data-lab': org, d: comparison, fill: 'none', stroke: options.brandColor?.(org) || TEAL, 'stroke-width': 1.5, 'stroke-opacity': .48, 'stroke-dasharray': '3 5', 'stroke-linejoin': 'round', 'stroke-linecap': 'round', 'aria-label': `${org}: same selected models, including refusals` }));
      }
      const path = releases.map((point, index) => `${index ? 'L' : 'M'}${scaleX(point.x)} ${scaleY(point.model.greenRate)}`).join(' ');
      plot.append(svg('path', { class: 'next-chart-lab-line', 'data-lab': org, d: path, fill: 'none', stroke: options.brandColor?.(org) || TEAL, 'stroke-width': 2, 'stroke-opacity': .7, 'stroke-linejoin': 'round', 'stroke-linecap': 'round', 'aria-hidden': 'true' }));
    }
  }
  // Draw selected models last so their focus ring remains visible in dense clusters.
  const ordered = [...points].sort((a, b) => Number(selectedModel(options.selected, a.model.id)) - Number(selectedModel(options.selected, b.model.id)));
  const marked = [];
  ordered.forEach(point => {
    const chosen = selectedModel(options.selected, point.model.id);
    const x = scaleX(point.x);
    const y = scaleY(point.model.greenRate);
    const group = svg('g');
    const color = colorOf(point.model, options);
    group.append(svg('circle', { cx: x, cy: y, r: options.compact ? 3.3 : 4.3, fill: color, 'fill-opacity': 0.68, stroke: PANEL, 'stroke-width': 1.2 }));
    interactiveMark(group, point.model, options, tooltip, config.detail(point));
    plot.append(group);
    marked.push({ x, y, model: point.model, chosen, latest: config.latestModels?.has(point.model.id) });
  });
  labelPoints(plot, marked, { x: margin.left + 3, y: margin.top + 3, width: plotWidth - 6, height: plotHeight - 6 });
  if (typeof ResizeObserver !== 'undefined') {
    const observer = new ResizeObserver(() => {
      if (!container.isConnected) { observer.disconnect(); return; }
      const next = chartGeometry(container, options.compact);
      if (next.width === width && next.height === height) return;
      const scrollLeft = plotScroll.scrollLeft;
      renderExplorer(container, options);
      container.querySelector('.next-chart-plot-scroll')?.scrollTo({ left: scrollLeft });
    });
    observer.observe(container);
    chartObservers.set(container, observer);
  }
}

function renderTimeline(container, options) {
  const points = options.models.map(model => {
    const metadata = metaFor(options.metadata, model);
    const x = metadata.launchDate ? Date.parse(metadata.launchDate) : NaN;
    return { model, metadata, x };
  }).filter(point => Number.isFinite(point.x));
  scatter(container, options, {
    points,
    label: 'Model detection rate by release date',
    xLabel: 'Release date',
    tick: value => dateLabel.format(value),
    missingLabel: 'release dates',
    emptyTitle: options.models.length ? 'No release dates for these models' : 'No matching models',
    emptyDetail: 'Try another lab or benchmark.',
    detail: point => [fullDate.format(point.x)],
  });
}

function renderLabTrends(container, options) {
  const labs = new Map([['openai', 'OpenAI'], ['anthropic', 'Anthropic'], ['google', 'Google']]);
  const candidates = options.models.filter(model => labs.has(providerOf(model)));
  const byRelease = new Map();
  let missingDates = 0;
  for (const model of candidates) {
    const metadata = metaFor(options.metadata, model);
    const x = metadata.launchDate ? Date.parse(metadata.launchDate) : NaN;
    if (!Number.isFinite(x)) { missingDates++; continue; }
    if (!finite(model.greenRate) || !(model.rateRows > 0)) continue;
    const key = `${providerOf(model)}:${x}`;
    const existing = byRelease.get(key);
    // Same exact-day rule as the original chart; ID breaks score ties stably.
    if (!existing || model.greenRate > existing.model.greenRate || (model.greenRate === existing.model.greenRate && model.id.localeCompare(existing.model.id) < 0)) byRelease.set(key, { model, metadata, x });
  }
  const points = [...byRelease.values()].sort((a, b) => a.x - b.x || a.model.id.localeCompare(b.model.id));
  container.append(el('h2', { class: 'next-chart-trend-heading' }, 'OpenAI, Anthropic & Google'));
  container.append(el('div', { class: 'next-chart-trend-note' }, `Best per lab and release${missingDates ? ` · ${missingDates} undated` : ''}`));
  if (!points.length) return empty(container, 'No matching releases', 'Try All labs.');
  const summary = el('div', { class: 'next-chart-lab-summary', 'aria-label': 'Latest plotted release per lab' });
  const latestModels = new Set();
  for (const [org, name] of labs) {
    const latest = points.filter(point => providerOf(point.model) === org).at(-1);
    if (latest) latestModels.add(latest.model.id);
    const item = el('div', { class: 'next-chart-lab', style: `--lab-color:${options.brandColor?.(org) || TEAL}` });
    const heading = el('div', { class: 'next-chart-lab-name' });
    const logo = options.brandLogo?.(org);
    if (logo) heading.append(el('img', { src: logo, alt: '' }));
    heading.append(el('span', {}, name), el('b', {}, latest ? pct(latest.model.greenRate) : '—'));
    const modelText = latest ? `${nameOf(latest.model)} · ${effortLabel(latest.model)}` : 'No matching releases';
    item.append(heading, el('span', { class: 'next-chart-lab-model', title: latest ? `${modelText} · ${fullDate.format(latest.x)}` : modelText }, modelText));
    if (latest?.model.refusal) {
      const otherRate = options.excludeRefusals ? latest.model.greenRateAllAttempts : latest.model.greenRateExcludingRefusals;
      item.append(el('span', { class: 'next-chart-lab-all-rate' }, `${pct(otherRate)} ${options.excludeRefusals ? 'all attempts' : 'excl. refusals'}`));
    }
    summary.append(item);
  }
  container.append(summary);
  const key = el('div', { class: 'next-chart-rate-key' });
  const primary = el('span', { class: 'next-chart-line-key' });
  primary.append(el('i'), el('span', {}, options.excludeRefusals ? 'Excluding refusals' : 'All attempts'));
  key.append(primary);
  if (options.excludeRefusals && points.some(point => point.model.refusal > 0)) {
    const comparison = el('label', { class: 'next-chart-line-key is-dotted', title: 'Same models, including refusals' });
    const toggle = el('input', { type: 'checkbox', 'aria-label': 'Show all-attempts comparison' });
    toggle.checked = !!options.showRefusalComparison;
    toggle.addEventListener('change', () => {
      options.onRefusalComparisonChange?.(toggle.checked);
      renderExplorer(container, { ...options, showRefusalComparison: toggle.checked });
      container.querySelector('[aria-label="Show all-attempts comparison"]')?.focus({ preventScroll: true });
    });
    comparison.append(toggle, el('i'), el('span', {}, 'All attempts'));
    key.append(comparison);
  }
  container.append(key);
  const first = new Date(points[0].x), last = points.at(-1).x;
  const ticks = [];
  for (let date = Date.UTC(first.getUTCFullYear(), Math.floor(first.getUTCMonth() / 3) * 3, 1); date <= last; ) {
    if (date >= points[0].x) ticks.push(date);
    const current = new Date(date);
    date = Date.UTC(current.getUTCFullYear(), current.getUTCMonth() + 3, 1);
  }
  scatter(container, { ...options, colorMode: 'provider' }, {
    points, connectLabs: true, latestModels, ticks: ticks.length ? ticks : [points[0].x],
    label: 'Clear pushback over time for OpenAI, Anthropic and Google',
    xLabel: 'Release date', tick: value => dateLabel.format(value),
    missingText: `${points.length} best-per-release points from ${candidates.length} variants; ${missingDates} without release dates.`,
    detail: point => [fullDate.format(point.x)],
  });
}

function renderSize(container, options) {
  const points = options.models.map(model => {
    const metadata = metaFor(options.metadata, model);
    return { model, metadata, x: Number(metadata.totalParams) };
  }).filter(point => finite(point.metadata.totalParams) && point.x > 0);
  scatter(container, options, {
    points,
    log: true,
    label: 'Detection rate by total parameter count, logarithmic scale',
    xLabel: 'Parameters (B, log scale)',
    tick: value => number.format(value),
    missingLabel: 'public parameter counts',
    emptyTitle: 'Model sizes unavailable',
    emptyDetail: '',
    detail: point => [
      `${number.format(point.x)}B parameters${finite(point.metadata.activeParams) && Number(point.metadata.activeParams) !== point.x ? ` · ${number.format(Number(point.metadata.activeParams))}B active` : ''}`,
      point.metadata.license || point.metadata.openStatus,
    ],
  });
}

function renderReasoning(container, options) {
  const families = new Map();
  options.models.forEach(model => {
    if ((model.rateRows ?? model.total) <= 0) return;
    const base = baseOf(model);
    if (!families.has(base)) families.set(base, []);
    families.get(base).push(model);
  });
  const matched = [...families.values()].filter(models => new Set(models.map(effort)).size > 1).map(models => {
    models.sort((a, b) => effortIndex(a) - effortIndex(b) || a.greenRate - b.greenRate);
    return { models, delta: models.at(-1).greenRate - models[0].greenRate };
  });
  if (!matched.length) return empty(container, 'No reasoning pairs', 'Try fewer filters.');
  const higher = matched.filter(family => family.delta > 0).sort((a, b) => b.delta - a.delta || nameOf(a.models[0]).localeCompare(nameOf(b.models[0])));
  const lower = matched.filter(family => family.delta < 0).sort((a, b) => a.delta - b.delta || nameOf(a.models[0]).localeCompare(nameOf(b.models[0])));
  const unchanged = matched.filter(family => family.delta === 0).sort((a, b) => nameOf(a.models[0]).localeCompare(nameOf(b.models[0])));
  const grid = el('div', { class: 'next-chart-reasoning-grid' });
  container.append(grid);
  const tooltip = tooltipFor(container);
  const panels = [
    { title: 'Higher', direction: 'higher', arrow: '↑', items: higher },
    { title: 'Lower', direction: 'lower', arrow: '↓', items: lower },
  ];
  const measured = [];
  panels.forEach(({ title, direction, arrow, items }) => {
    const panel = el('section', { class: `next-chart-reasoning-panel is-${direction}`, 'aria-label': `${title} clear pushback with higher requested effort` });
    const heading = el('div', { class: 'next-chart-reasoning-title' });
    heading.append(el('span', { class: 'direction-arrow', 'aria-hidden': 'true' }, arrow), el('strong', {}, title));
    panel.append(heading);
    grid.append(panel);
    if (!items.length) { empty(panel, 'None in this selection'); return; }
    const wrapper = el('div', { class: 'next-chart-family', tabindex: '0', 'aria-label': `${title}: detection by reasoning effort, scroll for more models` });
    const top = el('div', { class: 'next-chart-family-axis' });
    top.append(el('span', { title: 'Requested effort; hover for reported token usage' }, 'Model · requested effort'));
    const axis = svg('svg', { 'aria-hidden': 'true' });
    top.append(axis, el('span', { title: 'Highest minus lowest reasoning effort, in percentage points' }, 'Δ pp'));
    wrapper.append(top);
    panel.append(wrapper);
    const graphics = items.map(({ models, delta }) => {
      const row = el('div', { class: `next-chart-family-row${models.some(model => selectedModel(options.selected, model.id)) ? ' is-selected' : ''}` });
      const name = nameOf(models[0]).replace(/\s*\((?:none|minimal|low|medium|high|xhigh|max|ultra)\)$/i, '');
      const pair = `${effortLabel(models[0])} → ${effortLabel(models.at(-1))}`;
      const modelTitle = el('div', { class: 'next-chart-family-title', title: `${name} · ${pair}` });
      const logo = options.brandLogo?.(providerOf(models[0]));
      if (logo) modelTitle.append(el('img', { src: logo, alt: '', 'aria-hidden': 'true' }));
      const modelText = el('span', { class: 'next-chart-family-text' });
      modelText.append(el('span', { class: 'next-chart-family-name' }, name), el('span', { class: 'next-chart-pair' }, pair));
      modelTitle.append(modelText);
      row.append(modelTitle);
      const graphic = svg('svg', { role: 'group', 'aria-label': `${name}, clear pushback by reasoning level` });
      row.append(graphic);
      row.append(el('span', { class: `next-chart-delta ${delta > 0 ? 'up' : delta < 0 ? 'down' : ''}`, title: `${pair}: ${delta > 0 ? '+' : ''}${number.format(delta)} percentage points` }, `${delta > 0 ? '+' : ''}${number.format(delta)}`));
      wrapper.append(row);
      return { graphic, models, delta };
    });
    measured.push({ axis, wrapper, graphics });
  });
  measured.forEach(item => {
    const axisBox = item.axis.getBoundingClientRect();
    item.width = axisBox.width || 250;
    item.height = axisBox.height || 25;
    item.axis.setAttribute('viewBox', `0 0 ${item.width} ${item.height}`);
    [0, 50, 100].forEach(value => item.axis.append(svg('text', { x: 8 + value / 100 * (item.width - 16), y: item.height / 2 + 4, fill: MUTED, 'font-size': 12, 'text-anchor': value === 0 ? 'start' : value === 100 ? 'end' : 'middle' }, `${value}%`)));
    item.graphics.forEach(({ graphic, models, delta }) => {
      const box = graphic.getBoundingClientRect();
      const width = box.width || item.width;
      const height = box.height || 26;
      graphic.setAttribute('viewBox', `0 0 ${width} ${height}`);
      const x = rate => 8 + Math.max(0, Math.min(100, rate)) / 100 * (width - 16);
      const y = height / 2;
      const changeColor = delta > 0 ? HIGHER : LOWER;
      [0, 50, 100].forEach(value => graphic.append(svg('line', { x1: x(value), x2: x(value), y1: 0, y2: height, stroke: LINE, 'stroke-dasharray': '2 4', 'vector-effect': 'non-scaling-stroke' })));
      graphic.append(svg('line', { x1: x(Math.min(...models.map(model => model.greenRate))), x2: x(Math.max(...models.map(model => model.greenRate))), y1: y, y2: y, stroke: '#c4ccc5', 'stroke-width': 1.5, 'vector-effect': 'non-scaling-stroke' }));
      graphic.append(svg('line', { class: 'next-chart-change-line', x1: x(models[0].greenRate), x2: x(models.at(-1).greenRate), y1: y, y2: y, stroke: changeColor, 'stroke-opacity': .65, 'stroke-width': 3, 'vector-effect': 'non-scaling-stroke' }));
      models.forEach((model, index) => {
        const group = svg('g');
        const coincident = models.filter(other => other.greenRate === model.greenRate);
        const markerY = y + (coincident.indexOf(model) - (coincident.length - 1) / 2) * 3;
        const last = index === models.length - 1;
        group.append(svg('circle', { cx: x(model.greenRate), cy: markerY, r: last ? 4.5 : index === 0 ? 4 : 3, fill: index === 0 ? PANEL : changeColor, 'fill-opacity': index === 0 || last ? 1 : .55, stroke: index === 0 ? MUTED : changeColor, 'stroke-width': 1.7, 'vector-effect': 'non-scaling-stroke' }));
        interactiveMark(group, model, options, tooltip, tokenDetails(model));
        graphic.append(group);
      });
    });
  });
  if (unchanged.length) {
    const disclosure = el('details', { class: 'next-chart-unchanged' });
    disclosure.open = !!options.showUnchanged;
    disclosure.append(el('summary', {}, 'Unchanged'));
    const list = el('div', { class: 'next-chart-unchanged-items' });
    unchanged.forEach(({ models }) => {
      const row = el('div', { class: 'next-chart-unchanged-row' });
      row.append(el('span', { class: 'next-chart-unchanged-name' }, nameOf(models[0])));
      const scores = el('span', { class: 'next-chart-unchanged-scores' });
      models.forEach((model, index) => {
        if (index) scores.append(el('span', { 'aria-hidden': 'true' }, '→'));
        const chosen = selectedModel(options.selected, model.id);
        const button = el('button', { type: 'button', class: `next-chart-unchanged-score${chosen ? ' is-selected' : ''}`, 'data-model': model.id, 'aria-pressed': String(chosen), 'aria-label': `Highlight ${nameOf(model)}, ${effortLabel(model)}, ${pct(model.greenRate)} clear pushback` }, `${effortLabel(model)} ${pct(model.greenRate)}`);
        button.addEventListener('click', event => selectFrom(event, container, options, model.id));
        scores.append(button);
      });
      row.append(scores); list.append(row);
    });
    disclosure.append(list);
    disclosure.addEventListener('toggle', () => { options.showUnchanged = disclosure.open; options.onUnchangedChange?.(disclosure.open); });
    container.append(disclosure);
  }
  if (typeof ResizeObserver !== 'undefined' && measured.length) {
    const observer = new ResizeObserver(() => {
      if (!container.isConnected) { observer.disconnect(); return; }
      if (measured.every(item => {
        const next = item.axis.getBoundingClientRect();
        return Math.abs(next.width - item.width) < 0.5 && Math.abs(next.height - item.height) < 0.5;
      })) return;
      const scrolls = measured.map(item => item.wrapper.scrollTop);
      renderExplorer(container, options);
      container.querySelectorAll('.next-chart-family').forEach((wrapper, index) => wrapper.scrollTo({ top: scrolls[index] || 0 }));
    });
    observer.observe(container);
    chartObservers.set(container, observer);
  }
}

/** Render an independent, read-only chart view over already-loaded published rows. */
export function renderExplorer(container, options = {}) {
  chartObservers.get(container)?.disconnect();
  chartObservers.delete(container);
  const settings = { mode: 'domains', models: [], rows: [], questions: [], metadata: new Map(), selected: new Set(), judge: 'consensus', excludeRefusals: true, compact: false, colorMode: 'provider', ...options };
  container.replaceChildren();
  container.removeAttribute('aria-description');
  container.classList.add('next-chart');
  container.classList.toggle('is-compact', !!settings.compact);
  container.classList.toggle('is-lab-trends', settings.mode === 'labs');
  container.append(el('style', {}, STYLES));
  if (settings.mode === 'timeline') return renderTimeline(container, settings);
  if (settings.mode === 'labs') return renderLabTrends(container, settings);
  if (!settings.models.length) return empty(container, 'No matches', 'Try fewer filters.');
  if (settings.mode === 'reasoning') return renderReasoning(container, settings);
  if (settings.mode === 'size') return renderSize(container, settings);
  return renderDomains(container, settings);
}
