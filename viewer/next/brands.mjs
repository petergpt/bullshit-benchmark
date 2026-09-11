// Local company marks. Provenance and licenses: assets/brands/SOURCES.md.
const COLORS = Object.freeze({
  anthropic: '#f97316', openai: '#1f9d55', google: '#1a73e8', meta: '#0866ff',
  'x-ai': '#111111', mistralai: '#ec4899', deepseek: '#06b6d4',
  'bytedance-seed': '#0ea5e9', xiaomi: '#ff6900', qwen: '#7c3aed',
  moonshotai: '#0284c7', minimax: '#14b8a6', baidu: '#1d4ed8',
  'prime-intellect': '#64748b', 'z-ai': '#0f766e', stealth: '#6b7280',
});
const NAMES = Object.freeze({
  ai21: 'AI21', anthropic: 'Anthropic', 'arcee-ai': 'Arcee AI', baidu: 'Baidu',
  'bytedance-seed': 'ByteDance', deepseek: 'DeepSeek', google: 'Google', meta: 'Meta',
  minimax: 'MiniMax', mistralai: 'Mistral', moonshotai: 'Moonshot AI', nvidia: 'NVIDIA',
  openai: 'OpenAI', poolside: 'Poolside', 'prime-intellect': 'Prime Intellect',
  qwen: 'Qwen', stealth: 'Stealth', stepfun: 'StepFun', tencent: 'Tencent',
  'x-ai': 'xAI', xiaomi: 'Xiaomi', 'z-ai': 'Z.AI',
});
const SVG_BRANDS = new Set(['ai21', 'anthropic', 'arcee-ai', 'baidu', 'bytedance-seed',
  'deepseek', 'google', 'meta', 'minimax', 'mistralai', 'moonshotai', 'nvidia', 'openai',
  'poolside', 'qwen', 'stepfun', 'tencent', 'x-ai', 'xiaomi', 'z-ai']);
const normalize = org => {
  const key = String(org || 'unknown').trim().toLowerCase();
  return ({ 'meta-llama': 'meta', steath: 'stealth' })[key] || key;
};

/** A same-origin URL. Unknown/anonymous organizations return null for an initial fallback. */
export function brandLogo(org) {
  const key = normalize(org);
  const filename = SVG_BRANDS.has(key) ? `${key}.svg` : key === 'prime-intellect' ? 'prime-intellect.png' : null;
  return filename ? new URL(`./assets/brands/${filename}`, import.meta.url).href : null;
}
export function brandName(org) {
  const key = normalize(org);
  return NAMES[key] || key.charAt(0).toUpperCase() + key.slice(1);
}
/** Original viewer ORG_COLORS, including its hashed HSL fallback converted to hex. */
export function brandColor(org) {
  const key = normalize(org);
  if (Object.hasOwn(COLORS, key)) return COLORS[key];
  let hash = 0;
  for (let i = 0; i < key.length; i++) hash = ((hash << 5) - hash + key.charCodeAt(i)) | 0;
  const hue = Math.abs(hash) % 360;
  // hsl(hue 60% 42%), exactly the canonical viewer's fallback color formula.
  const channel = offset => {
    const k = (offset + hue / 30) % 12;
    return Math.round(255 * (.42 - .252 * Math.max(-1, Math.min(k - 3, 9 - k, 1))));
  };
  return '#' + [channel(0), channel(8), channel(4)].map(n => n.toString(16).padStart(2, '0')).join('');
}
