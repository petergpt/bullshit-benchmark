/** Native, self-contained chart export for the alternate viewer. */
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

  rectangle(ctx, 0, 0, sheetWidth, sheetHeight, PALETTE.background);
  rectangle(ctx, 0, 0, sheetWidth, headerHeight, PALETTE.forest);
  if (mascot) drawImageContained(ctx, mascot, pad, 20, 83, 66);
  else text(ctx, 'B!', pad + 41, 52, { size: 42, weight: 850, color: '#f7fbf9', align: 'center' });
  text(ctx, 'BullshitBench', pad + 100, 53, { size: 34, weight: 500, family: '"Futura", "Avenir Next", sans-serif', spacing: '-0.51px', color: '#f7fbf9' });
  const suite = version === 'v1' ? 'V1' : 'V2';
  const scope = domain && domain !== 'all' ? String(domain) : 'All domains';
  const scopeMax = sheetWidth - (pad + 480) - pad - 61;
  font(ctx, 14, 650);
  const scopeWidth = Math.min(scopeMax, ctx.measureText(scope).width + 24);
  rectangle(ctx, sheetWidth - pad - 52, 38, 52, 30, '#2a5a50', 4);
  text(ctx, suite, sheetWidth - pad - 26, 53, { size: 16, weight: 750, color: '#ffffff', align: 'center' });
  if (domain && domain !== 'all') {
    rectangle(ctx, sheetWidth - pad - 61 - scopeWidth, 38, scopeWidth, 30, '#e2e7dc', 4);
    fittedText(ctx, scope, sheetWidth - pad - 61 - scopeWidth / 2, 53, scopeWidth - 20, { size: 14, weight: 650, color: PALETTE.teal, align: 'center' });
  }
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
