/** Place measured chart labels without covering labels, points or leader lines. */
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const overlaps = (a, b, gap = 0) => a.x < b.x + b.width + gap && a.x + a.width + gap > b.x && a.y < b.y + b.height + gap && a.y + a.height + gap > b.y;

function crossesRect(line, rect, padding = 0) {
  const low = [rect.x - padding, rect.y - padding];
  const high = [rect.x + rect.width + padding, rect.y + rect.height + padding];
  let enter = 0, exit = 1;
  for (const axis of ['x', 'y']) {
    const index = axis === 'x' ? 0 : 1;
    const origin = line.from[axis], delta = line.to[axis] - origin;
    if (Math.abs(delta) < 1e-9) {
      if (origin < low[index] || origin > high[index]) return false;
    } else {
      const a = (low[index] - origin) / delta, b = (high[index] - origin) / delta;
      enter = Math.max(enter, Math.min(a, b));
      exit = Math.min(exit, Math.max(a, b));
      if (enter > exit) return false;
    }
  }
  return true;
}

function crossesLine(a, b) {
  const cross = (p, q, r) => (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x);
  return cross(a.from, a.to, b.from) * cross(a.from, a.to, b.to) < 0 && cross(b.from, b.to, a.from) * cross(b.from, b.to, a.to) < 0;
}

function crossesPoint(line, point) {
  const dx = line.to.x - line.from.x, dy = line.to.y - line.from.y;
  const t = clamp(((point.x - line.from.x) * dx + (point.y - line.from.y) * dy) / (dx * dx + dy * dy || 1), 0, 1);
  return Math.hypot(point.x - line.from.x - t * dx, point.y - line.from.y - t * dy) < (point.radius || 5) + 1;
}

function candidates(item, bounds) {
  const result = [], seen = new Set();
  const add = (x, y, preference = 0) => {
    x = clamp(x, bounds.x, bounds.x + bounds.width - item.width);
    y = clamp(y, bounds.y, bounds.y + bounds.height - item.height);
    const key = `${Math.round(x * 10)},${Math.round(y * 10)}`;
    if (seen.has(key)) return;
    seen.add(key);
    const rect = { x, y, width: item.width, height: item.height };
    const to = { x: clamp(item.x, x, x + item.width), y: clamp(item.y, y, y + item.height) };
    const distance = Math.hypot(to.x - item.x, to.y - item.y);
    if (distance < 8) return;
    const from = { x: item.x + (to.x - item.x) * 6 / distance, y: item.y + (to.y - item.y) * 6 / distance };
    result.push({ rect, line: { from, to }, distance, cost: distance + preference });
  };
  for (const gap of [10, 22, 38, 58, 84, ...(item.selected || item.preferred ? [116, 152] : [])]) {
    add(item.x + gap, item.y - item.height / 2);
    add(item.x - gap - item.width, item.y - item.height / 2, .5);
    add(item.x - item.width / 2, item.y - gap - item.height, 2);
    add(item.x - item.width / 2, item.y + gap, 3);
    for (const side of [-1, 1]) {
      add(side < 0 ? item.x - item.width - 10 : item.x + 10, item.y - gap - item.height, 4);
      add(side < 0 ? item.x - item.width - 10 : item.x + 10, item.y + gap, 5);
    }
  }
  // Crowded series endpoints often need a shallow diagonal into open chart space.
  if (item.selected || item.preferred) {
    for (const gap of [22, 38, 58, 84, 116, 152]) {
      for (const offset of [-58, -38, -22, 22, 38, 58]) {
        add(item.x - gap - item.width, item.y + offset - item.height / 2, 5);
        add(item.x + gap, item.y + offset - item.height / 2, 5);
      }
    }
  }
  return result.sort((a, b) => a.cost - b.cost);
}

export function placeLabels(items, { bounds, points = [], limit = 24 } = {}) {
  if (!bounds || bounds.width <= 0 || bounds.height <= 0) return [];
  const pending = items.filter(item => [item.x, item.y, item.width, item.height].every(Number.isFinite) && item.width > 0 && item.height > 0 && item.width <= bounds.width && item.height <= bounds.height);
  const obstacles = points.map(point => {
    const radius = (point.radius || 5) + 2;
    return { x: point.x - radius, y: point.y - radius, width: radius * 2, height: radius * 2 };
  });
  const placed = [];
  const score = item => {
    const separation = placed.length ? Math.min(...placed.map(other => Math.hypot((item.x - other.anchor.x) / bounds.width, (item.y - other.anchor.y) / bounds.height))) : 0;
    return (item.selected ? 10000 : 0) + (item.preferred ? 1000 : 0) + (item.priority || 0) + separation * 80;
  };
  while (pending.length && placed.length < limit) {
    let index = 0;
    for (let i = 1; i < pending.length; i++) if (score(pending[i]) > score(pending[index])) index = i;
    const [item] = pending.splice(index, 1);
    const position = candidates(item, bounds).find(candidate => {
      if (obstacles.some(rect => overlaps(candidate.rect, rect))) return false;
      if (points.some(point => Math.hypot(point.x - item.x, point.y - item.y) > .1 && crossesPoint(candidate.line, point))) return false;
      if (placed.some(other => overlaps(candidate.rect, other.rect, 4))) return false;
      if (placed.some(other => crossesRect(candidate.line, other.rect, 2) || crossesRect(other.line, candidate.rect, 2))) return false;
      return !placed.some(other => crossesLine(candidate.line, other.line));
    });
    if (position) placed.push({ id: item.id, anchor: { x: item.x, y: item.y }, selected: !!item.selected, ...position });
  }
  return placed;
}
