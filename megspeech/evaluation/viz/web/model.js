import { state, out, errorOf } from "./state.js";
import { $, h, esc, Layer, labelColor, errorColor, store, wordCount } from "./util.js";
import { Legend, sourceColor } from "./legend.js";

const PAD = 16;
const R = 3.2;
const HIT = 9;
const DIM = 0.1;
const SHOWN = 0.9;
const RING = 8;
const COLOR_KEY = "megspeech-viz:color-by";
const PREFERRED = "umap";
const AXES = { pca: "PC" };

const grey = (a) => `rgba(110,118,130,${a * 0.35})`;
const positionColor = (t, a = 1) => `hsla(${200 - 170 * Math.min(1, Math.max(0, t))},80%,58%,${a})`;
const label = (p) => p.label;

const COLORINGS = {
  label: { key: label, color: labelColor, hint: "colour = true label · hover a legend word to highlight, click to pin" },
  word: { key: label, color: labelColor, follows: true, hint: "colour = the selected sample's word (or a pinned legend word)" },
  source: { key: (p) => p.s.source, color: sourceColor, hint: "colour = source" },
  position: { value: (p) => p.rel, color: positionColor, scale: ["start", "end", "word position in its sentence"],
    hint: "colour = where the word sits in its sentence" },
  error: { value: (p) => errorOf(p.o), color: errorColor, scale: ["0", "1", "error = 1 − p(true)"],
    hint: "colour = error (1 − p true), green → red" },
};

const relative = (s) => {
  const n = wordCount(s.text);
  return n > 1 ? wordCount(s.text.slice(0, s.target?.[0] ?? 0)) / (n - 1) : 0;
};

export class ModelPanel {
  constructor(onPick) {
    this.layer = new Layer($("scatter"));
    this.layer.overlay = (ctx) => this.overlay(ctx);
    this.pts = [];
    this.hover = null;
    this.legend = new Legend(() => this.draw({ legend: false }));
    const cv = this.layer.cv;
    cv.addEventListener("mousemove", (e) => this.onMove(e));
    cv.addEventListener("mouseleave", () => { this.hover = null; $("tip").hidden = true; this.layer.paint(); this.mark(); });
    cv.addEventListener("click", (e) => { const p = this.nearest(e); if (p) onPick(p.id); });
    for (const b of $("colorby").children) b.onclick = () => this.setColor(b.dataset.c);
    $("projsel").onchange = () => { state.projection = $("projsel").value; this.draw(); };
    this.setColor(store.get(COLOR_KEY, "label"), { redraw: false });
  }

  setColor(c, { redraw = true } = {}) {
    state.colorBy = c in COLORINGS ? c : "label";
    state.focus = null;
    store.set(COLOR_KEY, state.colorBy);
    for (const b of $("colorby").children) b.classList.toggle("on", b.dataset.c === state.colorBy);
    $("schint").textContent = `${COLORINGS[state.colorBy].hint} · filled = correct · hollow = wrong · faint = filtered out · ↑/↓ step · space play`;
    if (redraw) this.draw();
  }

  showSample() {
    const s = state.byId.get(state.sel);
    $("activename").textContent = state.active ?? "";
    if (!s) return;
    this.topk(s);
    if (COLORINGS[state.colorBy].follows) this.draw();
    else this.layer.paint();
  }

  topk(s) {
    const o = out(state.active, s.id), box = $("topk");
    if (!o) { box.textContent = "no output for this sample"; return; }
    const max = o.topk[0]?.[1] || 1;
    box.replaceChildren(...o.topk.map(([lab, p]) => {
      const cls = `kbar${lab === s.label ? " truth" : ""}${lab === o.pred ? " pred" : ""}`;
      const bar = h("i");
      bar.style.width = `${(100 * p) / max}%`;
      return h("div", { className: cls, title: `${lab}: ${p}` },
        h("span", { className: "kname", textContent: lab }),
        h("span", { className: "ktrack" }, bar),
        h("span", { className: "kval", textContent: p.toFixed(3) }));
    }));
    if (!o.topk.some(([lab]) => lab === s.label)) {
      box.append(h("div", { className: "hint", textContent: `true label "${s.label}" not in top-${o.topk.length} · p(true) ${(o.p_true ?? 0).toFixed(3)}` }));
    }
  }

  projection(m) {
    const names = Object.keys(m?.projections ?? {}), pick = $("projsel");
    if ([...pick.options].map((o) => o.value).join() !== names.join()) {
      pick.replaceChildren(...names.map((n) => h("option", { value: n, textContent: n })));
    }
    if (!names.includes(state.projection)) state.projection = names.includes(PREFERRED) ? PREFERRED : names[0] ?? null;
    pick.value = state.projection ?? "";
    pick.hidden = !names.length;
    return state.projection;
  }

  spot() {
    const c = COLORINGS[state.colorBy];
    return c.key ? this.legend.spot(c.follows ? state.byId.get(state.sel)?.label : null) : null;
  }

  fill(p, a, spot) {
    const c = COLORINGS[state.colorBy];
    if (!c.key) return c.color(c.value(p), a);
    const k = c.key(p);
    return spot != null && k !== spot ? grey(a) : c.color(k, a);
  }

  mark() {
    const c = COLORINGS[state.colorBy];
    this.legend.mark(this.spot(), this.hover && c.key ? c.key(this.hover) : null);
  }

  draw({ legend = true } = {}) {
    const m = state.models.get(state.active), proj = this.projection(m);
    const axis = AXES[proj] ?? `${(proj ?? "").toUpperCase()} `;
    $("embhead").textContent = proj === "pca"
      ? (m.projections.pca.explained ?? []).map((e, i) => `PC${i + 1} ${(100 * e).toFixed(1)}%`).join(" · ")
      : "";
    this.layer.draw((ctx, W, H) => {
      this.pts = [];
      if (!m) return;
      const raw = state.samples.map((s) => ({ s, o: m.samples[s.id] })).map((p) => ({ ...p, e: p.o?.emb?.[proj] })).filter((p) => p.e);
      if (!raw.length) return;
      const xs = raw.map((p) => p.e[0]), ys = raw.map((p) => p.e[1]);
      const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
      const sx = (W - 2 * PAD) / (x1 - x0 || 1), sy = (H - 2 * PAD) / (y1 - y0 || 1);
      this.pts = raw.map(({ s, o, e }) => ({ id: s.id, label: s.label, s, o, rel: relative(s),
        x: PAD + (e[0] - x0) * sx, y: H - PAD - (e[1] - y0) * sy, on: state.visSet.has(s.id) }));
      const c = COLORINGS[state.colorBy], spot = this.spot();
      const rank = (p) => 2 * p.on + (spot != null && c.key?.(p) === spot);
      for (const p of [...this.pts].sort((a, b) => rank(a) - rank(b))) {
        const col = this.fill(p, p.on ? SHOWN : DIM, spot);
        ctx.beginPath(); ctx.arc(p.x, p.y, R, 0, 2 * Math.PI);
        if (p.o.correct) { ctx.fillStyle = col; ctx.fill(); } else { ctx.strokeStyle = col; ctx.lineWidth = 1.3; ctx.stroke(); }
      }
      ctx.save(); ctx.fillStyle = "#8b94a3"; ctx.font = "10px system-ui"; ctx.textAlign = "right";
      ctx.fillText(`${axis}1 →`, W - PAD, H - 3);
      ctx.translate(10, PAD); ctx.rotate(-Math.PI / 2); ctx.fillText(`${axis}2 →`, 0, 0); ctx.restore();
      if (legend) this.legend.render(c, this.pts, spot);
      else this.mark();
    });
  }

  overlay(ctx) {
    const ring = (p, col, r) => { ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, 2 * Math.PI); ctx.stroke(); };
    const sel = this.pts.find((p) => p.id === state.sel);
    if (this.hover) ring(this.hover, "#8b94a3", RING - 2);
    if (sel) { ring(sel, "#0e1116", RING + 1.5); ring(sel, "#ffffff", RING); }
  }

  nearest(e) {
    const r = this.layer.cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    let best = null, bd = HIT * HIT;
    for (const p of this.pts) {
      const d = (p.x - mx) ** 2 + (p.y - my) ** 2;
      if (d < bd || (d === bd && p.on)) { bd = d; best = p; }
    }
    return best;
  }

  onMove(e) {
    const p = this.nearest(e), tip = $("tip");
    if (p !== this.hover) { this.hover = p; this.layer.paint(); this.mark(); }
    if (!p) { tip.hidden = true; return; }
    tip.innerHTML = `<b>${esc(p.id)}</b><br>label ${esc(p.label)} · ${esc(p.s.source)} · pred <span class="${p.o.correct ? "ok" : "no"}">${esc(p.o.pred)}</span>`
      + `<br>p(true) ${(p.o.p_true ?? 0).toFixed(3)}`;
    tip.hidden = false;
    const W = this.layer.W, left = p.x + 12 + tip.offsetWidth > W ? p.x - 12 - tip.offsetWidth : p.x + 12;
    tip.style.left = `${Math.max(0, left)}px`;
    tip.style.top = `${Math.max(0, p.y - tip.offsetHeight - 6)}px`;
  }
}
