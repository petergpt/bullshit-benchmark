/** Native, self-contained chart export for the alternate viewer. */
import { brandName } from './brands.mjs';
const PALETTE = {
  background: '#f2eee6', panel: '#fcfbf7', ink: '#1d2521', muted: '#637068',
  forest: '#18493f', teal: '#1d5b50', line: '#d6dbd0', stripe: '#f5f4ec',
  highlight: '#e2eddf', green: '#23804d', amber: '#b97816', red: '#b6493f',
  refusal: '#89928f', error: '#505856',
};
const OUTCOMES = [
  ['green', 'Clear'], ['amber', 'Partial'],
  ['red', 'Accepted'], ['refusal', 'Refusal'], ['error', 'Unscored'],
];
const FONT = '"Avenir Next", Avenir, "Helvetica Neue", Arial, sans-serif';
const SCALE = 2;
const ROW_HEIGHT = 40;
const MAX_EDGE = 32760;
const MAX_PIXELS = 64 * 1024 * 1024;
const mascotURL = new URL('../../docs/images/bsbench.png', import.meta.url).href;
const images = new Map();
const CHART_TITLES = { timeline: 'Timeline', labs: 'Lab trends', reasoning: 'Reasoning', size: 'Model size' };

function count(value) {
  const n = Number(value);
  return Number.isFinite(n) && n >= 0 ? n : 0;
}
function number(value, digits = 0) {
  return Number(value).toLocaleString('en-US', { maximumFractionDigits: digits });
}
function font(ctx, size = 14, weight = 500, family = FONT) {
  ctx.font = `${weight} ${size}px ${family}`;
  if ('letterSpacing' in ctx) ctx.letterSpacing = '0px';
}
function text(ctx, value, x, y, { size = 14, weight = 500, color = PALETTE.ink, align = 'left', family = FONT, spacing = '0px' } = {}) {
  font(ctx, size, weight, family);
  if ('letterSpacing' in ctx) ctx.letterSpacing = spacing;
  ctx.fillStyle = color;
  ctx.textAlign = align;
  ctx.textBaseline = 'middle';
  ctx.fillText(String(value), x, y);
}
function fittedText(ctx, value, x, y, width, options = {}) {
  const full = String(value ?? '');
  let size = options.size || 14;
  const minimum = options.minimum || size;
  font(ctx, size, options.weight || 500);
  while (size > minimum && ctx.measureText(full).width > width) {
    size = Math.max(minimum, size - .5);
    font(ctx, size, options.weight || 500);
  }
  if (ctx.measureText(full).width <= width) {
    text(ctx, full, x, y, { ...options, size });
    return ctx.measureText(full).width;
  }
  const characters = [...full];
  let low = 0, high = characters.length;
  while (low < high) {
    const midpoint = Math.ceil((low + high) / 2);
    if (ctx.measureText(`${characters.slice(0, midpoint).join('')}…`).width <= width) low = midpoint;
    else high = midpoint - 1;
  }
  const fitted = `${characters.slice(0, low).join('').trimEnd()}…`;
  text(ctx, fitted, x, y, { ...options, size });
  return ctx.measureText(fitted).width;
}
function roundPath(ctx, x, y, width, height, radius = 4) {
  const r = Math.max(0, Math.min(radius, width / 2, height / 2));
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + width - r, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + r);
  ctx.lineTo(x + width, y + height - r);
  ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  ctx.lineTo(x + r, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}
function rectangle(ctx, x, y, width, height, color, radius = 0) {
  ctx.fillStyle = color;
  if (radius) { roundPath(ctx, x, y, width, height, radius); ctx.fill(); }
  else ctx.fillRect(x, y, width, height);
}
function line(ctx, x1, y1, x2, y2, color = PALETTE.line) {
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
  ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.stroke();
}
function contrastText(color) {
  const hex = color.replace('#', '');
  if (!/^[0-9a-f]{6}$/i.test(hex)) return PALETTE.ink;
  const channels = [0, 2, 4].map(index => parseInt(hex.slice(index, index + 2), 16) / 255)
    .map(channel => channel <= .04045 ? channel / 12.92 : ((channel + .055) / 1.055) ** 2.4);
  const luminance = .2126 * channels[0] + .7152 * channels[1] + .0722 * channels[2];
  return (1.05 / (luminance + .05)) >= 4.5 ? '#ffffff' : PALETTE.ink;
}
function validColor(value) {
  return typeof value === 'string' && /^#[0-9a-f]{6}$/i.test(value) ? value : PALETTE.teal;
}
function assetURL(value) {
  if (!value) return null;
  try {
    const url = new URL(value, document.baseURI);
    return ['http:', 'https:'].includes(url.protocol) && url.origin === location.origin ? url.href : null;
  } catch { return null; }
}
function loadImage(value) {
  const url = assetURL(value);
  if (!url) return Promise.resolve(null);
  if (!images.has(url)) images.set(url, new Promise(resolve => {
    const image = new Image();
    let settled = false;
    const finish = result => {
      if (settled) return;
      settled = true; clearTimeout(timer);
      image.onload = null; image.onerror = null;
      resolve(result);
    };
    const timer = setTimeout(() => finish(null), 3000);
    image.onload = () => finish(image.naturalWidth && image.naturalHeight ? image : null);
    image.onerror = () => finish(null);
    image.crossOrigin = 'anonymous';
    image.src = url;
  }));
  return images.get(url);
}
function drawImageContained(ctx, image, x, y, width, height) {
  const ratio = Math.min(width / image.naturalWidth, height / image.naturalHeight);
  const w = image.naturalWidth * ratio, h = image.naturalHeight * ratio;
  ctx.drawImage(image, x + (width - w) / 2, y + (height - h) / 2, w, h);
}
function drawPin(ctx, x, y) {
  ctx.save();
  ctx.translate(x, y); ctx.rotate(-Math.PI / 5);
  ctx.strokeStyle = PALETTE.forest; ctx.fillStyle = PALETTE.forest;
  ctx.lineWidth = 1.4; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(-3.5, -5); ctx.lineTo(3.5, -5); ctx.lineTo(2.5, 0);
  ctx.lineTo(4.5, 2); ctx.lineTo(-4.5, 2); ctx.lineTo(-2.5, 0); ctx.closePath();
  ctx.fill();
  ctx.beginPath(); ctx.moveTo(0, 2); ctx.lineTo(0, 7); ctx.stroke();
  ctx.restore();
}
function denominator(model, excludeRefusals) {
  return Math.max(0, count(model.total) - (excludeRefusals ? count(model.refusal) : 0));
}
function reasoningLabel(value) {
  const effort = String(value || 'none').toLowerCase();
  if (effort === 'none' || effort === 'default') return 'None';
  if (effort === 'xhigh') return 'xHigh';
  return effort[0].toUpperCase() + effort.slice(1);
}
function drawMix(ctx, model, x, y, width, height, excludeRefusals) {
  const attempts = denominator(model, excludeRefusals);
  rectangle(ctx, x, y, width, height, '#e7e8df', 3);
  if (!attempts) {
    text(ctx, 'No eligible attempts', x + width / 2, y + height / 2, { size: 12, color: PALETTE.muted, align: 'center' });
    return;
  }
  ctx.save(); roundPath(ctx, x, y, width, height, 3); ctx.clip();
  let left = x;
  for (const [key] of OUTCOMES) {
    const amount = excludeRefusals && key === 'refusal' ? 0 : count(model[key]);
    const percent = 100 * amount / attempts;
    const segmentWidth = width * amount / attempts;
    if (segmentWidth <= 0) continue;
    rectangle(ctx, left, y, segmentWidth, height, PALETTE[key]);
    const label = `${number(percent, percent < 10 ? 1 : 0)}%`;
    const size = [12, 11, 10, 9].find(size => {
      font(ctx, size, 500);
      return ctx.measureText(label).width + 2 <= segmentWidth;
    });
    if (size) {
      text(ctx, label, left + segmentWidth / 2, y + height / 2, { size, weight: 500, color: key === 'amber' ? '#ffffff' : contrastText(PALETTE[key]), align: 'center' });
    }
    left += segmentWidth;
  }
  ctx.restore();
}
function safelyCall(fn, key) {
  try { return typeof fn === 'function' ? fn(key) : null; } catch { return null; }
}
function filenamePart(value) {
  return String(value).normalize('NFKD').replace(/[\u0300-\u036f]/g, '').toLowerCase()
    .replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'all-domains';
}

function wrappedLines(ctx, value, width, size = 12) {
  font(ctx, size);
  const lines = [];
  let current = '';
  for (const word of String(value).split(/\s+/)) {
    const next = current ? `${current} ${word}` : word;
    if (current && ctx.measureText(next).width > width) { lines.push(current); current = word; }
    else current = next;
  }
  if (current) lines.push(current);
  return lines;
}

/** Render the same chart offscreen, without observers or changes to the live view. */
async function explorerSnapshot(options, width, plotHeight) {
  const { renderExplorer } = await import('./charts.mjs?v=20260927-compact-labs');
  await document.fonts?.ready;
  const host = document.createElement('div');
  host.inert = true;
  host.setAttribute('aria-hidden', 'true');
  host.style.cssText = `position:fixed;left:-20000px;top:0;width:${width}px;height:${plotHeight}px;font:14px ${FONT};pointer-events:none;`;
  document.body.append(host);
  try {
    const panelWidth = (width - 20) / 2;
    const nameWidth = Math.round(panelWidth * .46);
    const reasoningWidth = panelWidth - nameWidth - 70;
    renderExplorer(host, { ...options, compact: false, onSelect: undefined, onUnchangedChange: undefined,
      onRefusalComparisonChange: undefined, snapshotSize: { width, height: plotHeight, reasoningWidth } });
    const plot = host.querySelector('.next-chart-plot');
    const panels = [...host.querySelectorAll('.next-chart-reasoning-panel')].map(panel => ({
      title: panel.querySelector('strong').textContent,
      higher: panel.classList.contains('is-higher'),
      axis: panel.querySelector('.next-chart-family-axis svg')?.cloneNode(true),
      rows: [...panel.querySelectorAll('.next-chart-family-row')].map(row => ({
        name: row.querySelector('.next-chart-family-name').textContent,
        pair: row.querySelector('.next-chart-pair').textContent,
        logo: row.querySelector('img')?.src,
        delta: row.querySelector('.next-chart-delta').textContent,
        selected: row.classList.contains('is-selected'),
        graphic: row.querySelector('svg').cloneNode(true),
      })),
    }));
    const unchanged = [...host.querySelectorAll('.next-chart-unchanged-row')].map(row => ({
      name: row.querySelector('.next-chart-unchanged-name').textContent,
      scores: [...row.querySelectorAll('button')].map(button => button.textContent).join(' → '),
      selected: !!row.querySelector('.is-selected'),
    }));
    if (!plot && !panels.some(panel => panel.rows.length) && !unchanged.length) {
      throw new Error(host.querySelector('.next-chart-empty strong')?.textContent || 'No chart data to export.');
    }
    const marks = host.querySelectorAll(`.next-chart-mark${options.showUnchanged ? ', .next-chart-unchanged-score' : ''}`);
    const modelIds = new Set([...marks].map(mark => mark.dataset.model));
    return { plot: plot?.cloneNode(true), panels, unchanged, panelWidth, nameWidth, reasoningWidth,
      models: options.models.filter(model => modelIds.has(model.id)),
      description: host.getAttribute('aria-description') || '',
      comparison: !!host.querySelector('.next-chart-lab-comparison'),
      labs: [...host.querySelectorAll('.next-chart-lab')].map(lab => ({
        name: lab.querySelector('.next-chart-lab-name span').textContent,
        rate: lab.querySelector('b').textContent,
        model: lab.querySelector('.next-chart-lab-model').textContent,
        otherRate: lab.querySelector('.next-chart-lab-all-rate')?.textContent,
        logo: lab.querySelector('img')?.src,
        color: lab.style.getPropertyValue('--lab-color'),
      })),
    };
  } finally { host.remove(); }
}

function svgLayer(width, height) {
  const root = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  root.setAttribute('width', width * SCALE);
  root.setAttribute('height', height * SCALE);
  root.setAttribute('viewBox', `0 0 ${width} ${height}`);
  root.setAttribute('font-family', FONT);
  root.setAttribute('stroke', 'none');
  const style = document.createElementNS(root.namespaceURI, 'style');
  style.textContent = '.next-chart-label-text{paint-order:stroke;stroke-linejoin:round}.next-chart-mark.is-selected circle{stroke:#1d2521;stroke-width:2.5}';
  root.append(style);
  return {
    root,
    add(node, x, y, w, h) {
      if (!node) return;
      for (const [key, value] of Object.entries({ x, y, width: w, height: h })) node.setAttribute(key, value);
      // Rasterize strokes at the same 2× resolution as text and markers.
      node.querySelectorAll('[vector-effect]').forEach(mark => mark.removeAttribute('vector-effect'));
      root.append(node);
    },
  };
}

async function paintSvg(ctx, root) {
  const href = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(root)], { type: 'image/svg+xml;charset=utf-8' }));
  try {
    const image = await new Promise((resolve, reject) => {
      const img = new Image();
      const timer = setTimeout(() => { img.onload = img.onerror = null; reject(new Error('Chart image took too long to render.')); }, 10000);
      img.onload = () => { clearTimeout(timer); resolve(img); };
      img.onerror = () => { clearTimeout(timer); reject(new Error('The browser could not render the chart image.')); };
      img.src = href;
    });
    ctx.drawImage(image, 0, 0, Number(root.getAttribute('width')) / SCALE, Number(root.getAttribute('height')) / SCALE);
  } finally { URL.revokeObjectURL(href); }
}

/** All explorer charts, composed with the Dashboard PNG's branding at 2× resolution. */
export async function exportExplorerPng({ version = 'v2', domain = 'all', judgeLabel = '', filterLabel = '', width = 1280, ...options } = {}) {
  if (!CHART_TITLES[options.mode]) throw new Error('This view does not have a chart to export.');
  if (!Array.isArray(options.models) || !options.models.length) throw new Error('Choose at least one model to export.');
  options = { judge: 'consensus', excludeRefusals: true, ...options };
  const sheetWidth = Math.round(Math.max(1200, Math.min(1800, Number(width) || 1280)));
  const pad = 28, innerWidth = sheetWidth - pad * 2, plotHeight = Math.round(innerWidth * .46);
  const snapshot = await explorerSnapshot(options, innerWidth - 32, plotHeight);
  const imageSources = [...new Set([...snapshot.labs.map(lab => lab.logo), ...snapshot.panels.flatMap(panel => panel.rows.map(row => row.logo))].filter(Boolean))];
  const [mascot, ...loadedImages] = await Promise.all([loadImage(mascotURL), ...imageSources.map(loadImage)]);
  const logos = new Map(imageSources.map((src, index) => [src, loadedImages[index]]));
  // Measure wrapping before sizing the sheet, so filters and provenance never get clipped.
  const canvas = document.createElement('canvas'), ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('This browser could not create the image.');
  const filters = wrappedLines(ctx, String(filterLabel).replace(/\s+/g, ' ').trim(), innerWidth, 12);
  const titleY = 124, subtitleY = 148;
  const subtitle = options.mode === 'reasoning' ? 'Clear pushback by requested reasoning effort · lowest → highest'
    : options.mode === 'labs' ? ''
    : options.mode === 'size' ? 'Clear pushback by total parameter count · logarithmic scale' : 'Clear pushback by release date';
  const chartTop = subtitle ? 170 : 148;
  const chartY = chartTop + filters.length * 18;
  const labWidth = (innerWidth - 32 - 36) / 3;
  const labDetails = snapshot.labs.map(lab => {
    const lines = wrappedLines(ctx, lab.model, labWidth - 16);
    font(ctx, 12);
    const lastLineWidth = ctx.measureText(lines.at(-1) || '').width;
    font(ctx, 11);
    const inlineRate = !!lab.otherRate && lastLineWidth + 14 + ctx.measureText(lab.otherRate).width <= labWidth;
    return { lines, inlineRate, lineCount: lines.length + (lab.otherRate && !inlineRate ? 1 : 0) };
  });
  const labsHeight = snapshot.labs.length ? 48 + (Math.max(...labDetails.map(detail => detail.lineCount)) - 1) * 17 : 0;
  const companyKeys = [];
  let keyX = 0, keyRow = 0;
  if (['timeline', 'size'].includes(options.mode) && options.colorMode !== 'detection') {
    font(ctx, 12);
    for (const org of [...new Set(snapshot.models.map(model => model.org))].sort((a, b) => brandName(a).localeCompare(brandName(b)))) {
      const label = brandName(org), keyWidth = ctx.measureText(label).width + 34;
      if (keyX && keyX + keyWidth > innerWidth - 32) { keyX = 0; keyRow++; }
      companyKeys.push({ label, color: validColor(safelyCall(options.brandColor, org)), x: keyX, y: keyRow * 24 });
      keyX += keyWidth;
    }
  }
  const keyHeight = companyKeys.length ? (keyRow + 1) * 24 + 12 : options.mode === 'labs' || options.colorMode === 'detection' ? 32 : 0;
  const maxRows = Math.max(1, ...snapshot.panels.map(panel => panel.rows.length));
  const reasoningHeight = 68 + maxRows * 44;
  const unchangedHeight = snapshot.unchanged.length ? 40 + (options.showUnchanged ? Math.ceil(snapshot.unchanged.length / 2) * 48 : 0) : 0;
  const chartHeight = options.mode === 'reasoning' ? reasoningHeight + unchangedHeight + 32 : labsHeight + plotHeight + keyHeight + 32;
  const twoJudgeAnswers = options.judge === 'consensus' ? snapshot.models.reduce((sum, model) => sum + count(model.twoJudgeAnswerCount), 0) : 0;
  const footer = [
    `${snapshot.models.length} plotted variant${snapshot.models.length === 1 ? '' : 's'}`,
    options.judge === 'consensus' ? '' : judgeLabel || String(options.judge).replace(/^judge_/, 'Judge '),
    options.excludeRefusals ? 'Clear: excl. refusals' : 'Clear: all attempts',
    twoJudgeAnswers ? `2/3 judges on ${number(twoJudgeAnswers)} answer${twoJudgeAnswers === 1 ? '' : 's'}` : '',
    snapshot.description,
  ].filter(Boolean).join(' · ');
  const footerLines = wrappedLines(ctx, footer, innerWidth, 11);
  const sheetHeight = chartY + chartHeight + 14 + footerLines.length * 16 + 16;
  const pixelWidth = sheetWidth * SCALE, pixelHeight = sheetHeight * SCALE;
  if (pixelHeight > MAX_EDGE || pixelWidth * pixelHeight > MAX_PIXELS) throw new Error('This snapshot is too large. Filter the chart to fewer models.');
  canvas.width = pixelWidth; canvas.height = pixelHeight;
  ctx.scale(SCALE, SCALE); ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
  try {
    const { suite, scope } = drawHeader(ctx, { width: sheetWidth, height: sheetHeight, version, domain, mascot });
    text(ctx, CHART_TITLES[options.mode], pad, titleY, { size: 23, weight: 600, color: PALETTE.forest });
    if (subtitle) text(ctx, subtitle, pad, subtitleY, { size: 13, color: PALETTE.muted });
    filters.forEach((label, index) => text(ctx, label, pad, chartTop + 1 + index * 18, { size: 12, color: PALETTE.teal }));
    rectangle(ctx, pad, chartY, innerWidth, chartHeight, PALETTE.panel, 6);
    const layer = svgLayer(sheetWidth, sheetHeight);
    const x = pad + 16, y = chartY + 16;
    if (options.mode === 'reasoning') {
      snapshot.panels.forEach((panel, index) => {
        const left = x + index * (snapshot.panelWidth + 20), w = snapshot.panelWidth;
        const color = panel.higher ? PALETTE.green : PALETTE.red;
        rectangle(ctx, left, y, w, 34, panel.higher ? '#e5f0e3' : '#f7e7e1', 4);
        text(ctx, `${panel.higher ? '↑' : '↓'} ${panel.title}`, left + 12, y + 17, { size: 15, weight: 600, color });
        text(ctx, `${panel.rows.length} model${panel.rows.length === 1 ? '' : 's'}`, left + w - 12, y + 17, { size: 12, color, align: 'right' });
        text(ctx, 'Model · requested effort', left + 10, y + 51, { size: 11, color: PALETTE.muted });
        text(ctx, 'Δ pp', left + w - 10, y + 51, { size: 11, color: PALETTE.muted, align: 'right' });
        layer.add(panel.axis, left + snapshot.nameWidth, y + 38, snapshot.reasoningWidth, 25);
        if (!panel.rows.length) text(ctx, 'None in this selection', left + 12, y + 90, { size: 12, color: PALETTE.muted });
        panel.rows.forEach((row, rowIndex) => {
          const top = y + 68 + rowIndex * 44;
          rectangle(ctx, left, top, w, 44, row.selected ? PALETTE.highlight : rowIndex % 2 ? PALETTE.stripe : PALETTE.panel);
          if (row.selected) rectangle(ctx, left, top, 3, 44, PALETTE.teal);
          line(ctx, left, top + 44, left + w, top + 44);
          const logo = logos.get(row.logo);
          if (logo) drawImageContained(ctx, logo, left + 10, top + 12, 20, 20);
          const textX = left + (logo ? 38 : 10);
          fittedText(ctx, row.name, textX, top + 15, left + snapshot.nameWidth - textX - 8, { size: 14, minimum: 13, weight: row.selected ? 600 : 400 });
          text(ctx, row.pair, textX, top + 32, { size: 12, color: PALETTE.muted });
          text(ctx, row.delta, left + w - 10, top + 22, { size: 13, weight: 600, color, align: 'right' });
          layer.add(row.graphic, left + snapshot.nameWidth, top + 8, snapshot.reasoningWidth, 28);
        });
      });
      if (snapshot.unchanged.length) {
        const top = y + reasoningHeight + 18;
        text(ctx, `Unchanged · ${snapshot.unchanged.length} model${snapshot.unchanged.length === 1 ? '' : 's'}${options.showUnchanged ? '' : ' (collapsed)'}`, x, top, { size: 13, color: PALETTE.muted });
        if (options.showUnchanged) snapshot.unchanged.forEach((row, index) => {
          const left = x + (index % 2) * (snapshot.panelWidth + 20), rowY = top + 18 + Math.floor(index / 2) * 48;
          rectangle(ctx, left, rowY, snapshot.panelWidth, 46, row.selected ? PALETTE.highlight : PALETTE.stripe, 3);
          fittedText(ctx, row.name, left + 10, rowY + 14, snapshot.panelWidth - 20, { size: 12, weight: row.selected ? 600 : 400 });
          fittedText(ctx, row.scores, left + 10, rowY + 32, snapshot.panelWidth - 20, { size: 11, color: PALETTE.muted });
        });
      }
    } else {
      snapshot.labs.forEach((lab, index) => {
        const left = x + index * (labWidth + 18);
        const detail = labDetails[index];
        rectangle(ctx, left, y, labWidth, 3, lab.color);
        const logo = logos.get(lab.logo);
        if (logo) drawImageContained(ctx, logo, left, y + 7, 22, 22);
        text(ctx, lab.name, left + (logo ? 30 : 0), y + 18, { size: 14, weight: 600 });
        text(ctx, lab.rate, left + labWidth, y + 18, { size: 18, weight: 600, color: PALETTE.forest, align: 'right' });
        detail.lines.forEach((label, lineIndex) => text(ctx, label, left, y + 40 + lineIndex * 17, { size: 12, color: PALETTE.muted }));
        if (lab.otherRate) text(ctx, lab.otherRate, left + (detail.inlineRate ? labWidth : 0), y + 40 + (detail.lineCount - 1) * 17, { size: 11, color: PALETTE.muted, align: detail.inlineRate ? 'right' : 'left' });
      });
      const keyY = y + labsHeight + 14;
      if (options.mode === 'labs') {
        line(ctx, x, keyY, x + 25, keyY, PALETTE.muted);
        text(ctx, options.excludeRefusals ? 'Excluding refusals' : 'All attempts', x + 33, keyY, { size: 12, color: PALETTE.muted });
        if (snapshot.comparison) {
          ctx.setLineDash([3, 5]); line(ctx, x + 185, keyY, x + 210, keyY, PALETTE.muted); ctx.setLineDash([]);
          text(ctx, 'All attempts', x + 218, keyY, { size: 12, color: PALETTE.muted });
        }
      } else if (options.colorMode === 'detection') {
        const gradient = ctx.createLinearGradient(x + 167, 0, x + 247, 0);
        gradient.addColorStop(0, PALETTE.red); gradient.addColorStop(.5, PALETTE.amber); gradient.addColorStop(1, PALETTE.green);
        text(ctx, 'Colour: clear pushback', x, keyY, { size: 12, color: PALETTE.muted });
        text(ctx, '0%', x + 140, keyY, { size: 11, color: PALETTE.muted });
        rectangle(ctx, x + 167, keyY - 4, 80, 8, gradient, 2);
        text(ctx, '100%', x + 256, keyY, { size: 11, color: PALETTE.muted });
      } else {
        for (const key of companyKeys) {
          rectangle(ctx, x + key.x, keyY + key.y - 4, 8, 8, key.color, 2);
          text(ctx, key.label, x + key.x + 15, keyY + key.y, { size: 12, color: PALETTE.muted });
        }
      }
      layer.add(snapshot.plot, x, y + labsHeight + keyHeight, innerWidth - 32, plotHeight);
    }
    await paintSvg(ctx, layer.root);
    footerLines.forEach((label, index) => text(ctx, label, pad, chartY + chartHeight + 22 + index * 16, { size: 11, color: PALETTE.muted }));
    const blob = await new Promise((resolve, reject) => canvas.toBlob(value => value ? resolve(value) : reject(new Error('The browser could not encode the snapshot.')), 'image/png'));
    return { blob, filename: `bullshitbench-${suite.toLowerCase()}-${filenamePart(CHART_TITLES[options.mode])}-${filenamePart(scope)}.png`, width: pixelWidth, height: pixelHeight };
  } finally { canvas.width = 1; canvas.height = 1; }
}

function drawHeader(ctx, { width, height, version, domain, mascot }) {
  const pad = 28;
  rectangle(ctx, 0, 0, width, height, PALETTE.background);
  rectangle(ctx, 0, 0, width, 96, PALETTE.forest);
  if (mascot) drawImageContained(ctx, mascot, pad, 20, 83, 66);
  else text(ctx, 'B!', pad + 41, 52, { size: 42, weight: 850, color: '#f7fbf9', align: 'center' });
  text(ctx, 'BullshitBench', pad + 100, 53, { size: 34, weight: 500, family: '"Futura", "Avenir Next", sans-serif', spacing: '-0.51px', color: '#f7fbf9' });
  const suite = version === 'v1' ? 'V1' : 'V2';
  const scope = domain && domain !== 'all' ? String(domain) : 'All domains';
  const scopeMax = width - (pad + 480) - pad - 61;
  font(ctx, 14, 650);
  const scopeWidth = Math.min(scopeMax, ctx.measureText(scope).width + 24);
  rectangle(ctx, width - pad - 52, 38, 52, 30, '#2a5a50', 4);
  text(ctx, suite, width - pad - 26, 53, { size: 16, weight: 750, color: '#ffffff', align: 'center' });
  if (domain && domain !== 'all') {
    rectangle(ctx, width - pad - 61 - scopeWidth, 38, scopeWidth, 30, '#e2e7dc', 4);
    fittedText(ctx, scope, width - pad - 61 - scopeWidth / 2, 53, scopeWidth - 20, { size: 14, weight: 650, color: PALETTE.teal, align: 'center' });
  }
  return { suite, scope };
}

/**
 * Export the supplied ranking rows, in their supplied order. No data is fetched
 * apart from the same-origin mascot and provider images. Neither DOM nor UI
 * state is changed. `width` is in CSS pixels; returned dimensions are PNG pixels.
 * Highlighted IDs add an outline, pin and bold name without dimming other rows.
 * The checkbox controls bar composition; the Clear score always excludes refusals.
 * `filterLabel` records any additional provider, reasoning or selection filters.
 * @returns {Promise<{blob: Blob, filename: string, width: number, height: number}>}
 */
export async function exportRankingsPng({
  models, version = 'v2', domain = 'all', judge = 'consensus', judgeLabel = '',
  excludeRefusals = false, highlighted = new Set(), brandLogo, brandColor, width = 1280, totalModels, filterLabel = '',
} = {}) {
  if (!Array.isArray(models) || !models.length) throw new Error('Choose at least one model to export.');
  if (models.some(model => !model || typeof model !== 'object')) throw new Error('A model in the snapshot is unavailable.');
  const sheetWidth = Math.round(Math.max(960, Math.min(1800, Number(width) || 1280)));
  const filters = String(filterLabel || '').replace(/\s+/g, ' ').trim();
  const twoJudgeAnswers = judge === 'consensus' ? models.reduce((sum, model) => sum + count(model.twoJudgeAnswerCount), 0) : 0;
  const headerHeight = 96;
  const tableY = headerHeight + (filters ? 34 : 14), tableHeadHeight = 32, footerHeight = 34;
  const sheetHeight = tableY + tableHeadHeight + models.length * ROW_HEIGHT + footerHeight;
  const pixelWidth = sheetWidth * SCALE, pixelHeight = sheetHeight * SCALE;
  if (pixelHeight > MAX_EDGE || pixelWidth * pixelHeight > MAX_PIXELS) {
    throw new Error('This snapshot is too tall. Filter the ranking or export only highlighted models.');
  }
  const selected = highlighted instanceof Set ? highlighted : new Set(highlighted || []);
  const highlightedCount = models.filter(model => selected.has(model.id)).length;
  const logos = new Map();
  const organizations = [...new Set(models.map(model => model.org))];
  const [mascot] = await Promise.all([
    loadImage(mascotURL),
    ...organizations.map(async org => logos.set(org, await loadImage(safelyCall(brandLogo, org)))),
    document.fonts?.ready,
  ]);
  const canvas = document.createElement('canvas');
  canvas.width = pixelWidth; canvas.height = pixelHeight;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('This browser could not create the image. Try a smaller selection.');
  ctx.scale(SCALE, SCALE);
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
  const pad = 28, innerWidth = sheetWidth - pad * 2;
  const rankWidth = 48, modelWidth = Math.round(innerWidth * .30), effortWidth = 104, rateWidth = 82;
  const modelX = pad + rankWidth, effortX = modelX + modelWidth, rateX = effortX + effortWidth;
  const barX = rateX + rateWidth, barWidth = sheetWidth - pad - barX - 14;

  const { suite, scope } = drawHeader(ctx, { width: sheetWidth, height: sheetHeight, version, domain, mascot });
  if (filters) fittedText(ctx, filters, pad, headerHeight + 17, innerWidth, { size: 12, minimum: 10, weight: 600, color: PALETTE.teal });

  rectangle(ctx, pad, tableY, innerWidth, tableHeadHeight, '#e7ebdf', 4);
  text(ctx, '#', pad + 29, tableY + 17, { size: 12, weight: 500, color: PALETTE.muted, align: 'center' });
  text(ctx, 'Model', modelX + 3, tableY + 17, { size: 12, weight: 500, color: PALETTE.muted });
  text(ctx, 'Reasoning', effortX + 8, tableY + 17, { size: 12, weight: 500, color: PALETTE.muted });
  text(ctx, 'Clear', rateX + rateWidth - 18, tableY + 17, { size: 12, weight: 500, color: PALETTE.muted, align: 'right' });
  let legendX = barX + 1;
  const legend = OUTCOMES.filter(([key]) => !(excludeRefusals && key === 'refusal') && (key !== 'error' || models.some(model => count(model.error))));
  for (const [key, label] of legend) {
    rectangle(ctx, legendX, tableY + 13, 8, 8, PALETTE[key], 2);
    text(ctx, label, legendX + 13, tableY + 17, { size: 11, color: PALETTE.muted });
    font(ctx, 11, 500); legendX += ctx.measureText(label).width + 26;
  }
  models.forEach((model, index) => {
    const rowY = tableY + tableHeadHeight + index * ROW_HEIGHT, centerY = rowY + ROW_HEIGHT / 2;
    const pinned = selected.has(model.id);
    rectangle(ctx, pad, rowY, innerWidth, ROW_HEIGHT, pinned ? PALETTE.highlight : index % 2 ? PALETTE.stripe : PALETTE.panel);
    line(ctx, pad, rowY + ROW_HEIGHT, sheetWidth - pad, rowY + ROW_HEIGHT, '#e2e5dc');
    if (pinned) {
      roundPath(ctx, pad + 1, rowY + 1, innerWidth - 2, ROW_HEIGHT - 2, 4);
      ctx.strokeStyle = PALETTE.teal; ctx.lineWidth = 1.5; ctx.stroke();
      drawPin(ctx, pad + 10, centerY - 1);
    }
    text(ctx, model.rank ?? index + 1, pad + 30, centerY, { size: 13, color: pinned ? PALETTE.forest : PALETTE.muted, weight: pinned ? 600 : 400, align: 'center' });
    const companyImage = logos.get(model.org);
    if (companyImage) drawImageContained(ctx, companyImage, modelX + 3, centerY - 11, 22, 22);
    else {
      const companyColor = validColor(safelyCall(brandColor, model.org));
      rectangle(ctx, modelX + 3, centerY - 11, 22, 22, companyColor, 4);
      text(ctx, String(model.provider || model.org || '?')[0].toUpperCase(), modelX + 14, centerY + .5, { size: 12, weight: 750, color: contrastText(companyColor), align: 'center' });
    }
    const isNew = Number.isFinite(model.newUntil) && model.newUntil > Date.now();
    const nameWidth = fittedText(ctx, model.name || model.base || model.id, modelX + 36, centerY, modelWidth - 45 - (isNew ? 42 : 0), { size: 14, minimum: 12, weight: pinned ? 600 : 400 });
    if (isNew) {
      const badgeX = modelX + 36 + nameWidth + 5;
      font(ctx, 10, 700);
      const badgeWidth = ctx.measureText('NEW').width + 10;
      rectangle(ctx, badgeX, centerY - 8.5, badgeWidth, 17, PALETTE.green, 3);
      text(ctx, 'NEW', badgeX + badgeWidth / 2, centerY, { size: 10, weight: 700, color: '#ffffff', align: 'center' });
    }
    fittedText(ctx, reasoningLabel(model.reasoning), effortX + 8, centerY, effortWidth - 16, { size: 14, color: PALETTE.muted, weight: pinned ? 600 : 400 });
    const attempts = denominator(model, true);
    const clear = attempts ? Math.min(100, 100 * count(model.green) / attempts) : null;
    text(ctx, clear === null ? '—' : `${number(clear, 1)}%`, rateX + rateWidth - 18, centerY, { size: 14, weight: pinned ? 600 : 400, color: PALETTE.forest, align: 'right' });
    drawMix(ctx, model, barX, centerY - 13, barWidth, 26, excludeRefusals);
  });

  const footerY = tableY + tableHeadHeight + models.length * ROW_HEIGHT;
  const judgeName = judgeLabel || (judge === 'consensus' ? 'Consensus' : String(judge).replace(/^judge_/, 'Judge '));
  const total = Math.max(models.length, Math.floor(count(totalModels)));
  const ranks = models.map(model => Number(model.rank));
  const contiguous = ranks.every((rank, index) => Number.isInteger(rank) && rank > 0 && (!index || rank === ranks[index - 1] + 1));
  const rangeLabel = total > models.length && contiguous
    ? `${models.length === 1 ? `Rank ${number(ranks[0])}` : `Ranks ${number(ranks[0])}–${number(ranks.at(-1))}`} of ${number(total)}`
    : `${number(models.length)}${total > models.length ? ` of ${number(total)}` : ''} variants`;
  const coverage = twoJudgeAnswers ? `2/3 judges on ${number(twoJudgeAnswers)} answer${twoJudgeAnswers === 1 ? '' : 's'}` : '';
  const scopeLabel = [rangeLabel, judge === 'consensus' ? '' : judgeName, excludeRefusals ? 'Refusals excluded' : 'Bars: all attempts · Clear: excl. refusals', coverage].filter(Boolean).join(' · ');
  fittedText(ctx, scopeLabel, pad, footerY + 17, innerWidth, { size: 11, minimum: 10, color: PALETTE.muted });

  let blob;
  try {
    blob = await new Promise((resolve, reject) => canvas.toBlob(value => value ? resolve(value) : reject(new Error('The browser could not encode the snapshot. Try fewer rows.')), 'image/png'));
  } finally {
    canvas.width = 1; canvas.height = 1;
  }
  return {
    blob,
    filename: `bullshitbench-${suite.toLowerCase()}-${filenamePart(scope)}-${models.length}-variants${highlightedCount ? '-highlighted' : ''}.png`,
    width: pixelWidth, height: pixelHeight,
  };
}
