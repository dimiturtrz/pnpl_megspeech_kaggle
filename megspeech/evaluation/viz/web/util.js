export const $ = (id) => document.getElementById(id);

export function h(tag, props = {}, ...kids) {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...kids);
  return e;
}

export const esc = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
export const dataUrl = (file) => `data/${file}`;
export const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

export function labelColor(label, alpha = 1) {
  let x = 2166136261;
  for (const c of String(label)) x = Math.imul(x ^ c.charCodeAt(0), 16777619);
  return `hsla(${(x >>> 0) % 360},70%,62%,${alpha})`;
}

const VIRIDIS = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
export function viridis(t, alpha = 1) {
  const x = clamp(t, 0, 1) * (VIRIDIS.length - 1), i = Math.min(Math.floor(x), VIRIDIS.length - 2), f = x - i;
  const [a, b] = [VIRIDIS[i], VIRIDIS[i + 1]];
  return `rgba(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(",")},${alpha})`;
}

export function niceStep(span, maxTicks) {
  const raw = span / maxTicks, p = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 5, 10].map((m) => m * p).find((s) => s >= raw);
}

export const store = {
  get(k, dflt) { try { const v = localStorage.getItem(k); return v === null ? dflt : JSON.parse(v); } catch { return dflt; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage unavailable */ } },
};

export class Layer {
  constructor(cv) {
    this.cv = cv;
    this.base = document.createElement("canvas");
    this.overlay = null;
    this.W = 1; this.H = 1;
  }

  draw(fn) {
    const r = this.cv.getBoundingClientRect(), d = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(r.width * d)), hh = Math.max(1, Math.round(r.height * d));
    for (const c of [this.cv, this.base]) if (c.width !== w || c.height !== hh) { c.width = w; c.height = hh; }
    this.W = r.width; this.H = r.height;
    const ctx = this.base.getContext("2d");
    ctx.setTransform(d, 0, 0, d, 0, 0);
    ctx.clearRect(0, 0, this.W, this.H);
    fn(ctx, this.W, this.H);
    this.paint();
  }

  paint() {
    const ctx = this.cv.getContext("2d"), d = window.devicePixelRatio || 1;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.cv.width, this.cv.height);
    ctx.drawImage(this.base, 0, 0);
    if (!this.overlay) return;
    ctx.setTransform(d, 0, 0, d, 0, 0);
    this.overlay(ctx, this.W, this.H);
  }
}
