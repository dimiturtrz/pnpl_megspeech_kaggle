import { state, out, checkedModels } from "./state.js";
import { $, h, esc, Layer, labelColor, viridis, store } from "./util.js";

const PAD = 16;
const R = 3.2;
const HIT = 9;
const DIM = 0.1;
const SHOWN = 0.9;
const RING = 8;
const COLOR_KEY = "megspeech-viz:color-by";

const COLORINGS = {
  label: { fill: (p, a) => labelColor(p.label, a), hint: "colour = true label" },
  error: { fill: (p, a) => viridis(p.o.error ?? 1 - (p.o.p_true ?? 0), a), hint: "colour = error (1 − p true)" },
};

export class ModelPanel {
  constructor(onPick) {
    this.layer = new Layer($("scatter"));
    this.layer.overlay = (ctx) => this.overlay(ctx);
    this.pts = [];
    this.hover = null;
    const cv = this.layer.cv;
    cv.addEventListener("mousemove", (e) => this.onMove(e));
    cv.addEventListener("mouseleave", () => { this.hover = null; $("tip").hidden = true; this.layer.paint(); });
    cv.addEventListener("click", (e) => { const p = this.nearest(e); if (p) onPick(p.id); });
    for (const b of $("colorby").children) b.onclick = () => this.setColor(b.dataset.c);
    this.setColor(store.get(COLOR_KEY, "label"), { redraw: false });
  }

  setColor(c, { redraw = true } = {}) {
    state.colorBy = c in COLORINGS ? c : "label";
    store.set(COLOR_KEY, state.colorBy);
    for (const b of $("colorby").children) b.classList.toggle("on", b.dataset.c === state.colorBy);
    $("cbar").hidden = state.colorBy !== "error";
    $("schint").textContent = `${COLORINGS[state.colorBy].hint} · filled = correct · hollow = wrong · faint = filtered out · ↑/↓ step · space play`;
    if (redraw) this.draw();
  }

  showSample() {
    const s = state.byId.get(state.sel);
    $("activename").textContent = state.active ?? "";
    if (!s) return;
    this.topk(s);
    this.agree(s);
    this.layer.paint();
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

  agree(s) {
    const names = checkedModels(), box = $("agree");
    if (names.length < 2) { box.replaceChildren(); return; }
    const outs = names.map((n) => [n, out(n, s.id)]), right = outs.filter(([, o]) => o?.correct).length;
    box.innerHTML = `<span class="ahead">${right}/${names.length} models correct</span>`
      + outs.map(([n, o]) => `<span><span class="${o?.correct ? "ok" : "no"}">${o ? (o.correct ? "✓" : "✗") : "·"}</span> ${esc(n)}: ${esc(o?.pred ?? "—")}</span>`).join("");
  }

  draw() {
    const m = state.models.get(state.active);
    $("embhead").textContent = m ? `${m.projection ?? ""} · ${(m.explained ?? []).map((e, i) => `PC${i + 1} ${(100 * e).toFixed(1)}%`).join(" · ")}` : "";
    this.layer.draw((ctx, W, H) => {
      this.pts = [];
      if (!m) return;
      const raw = state.samples.map((s) => ({ s, o: m.samples[s.id] })).filter((p) => p.o?.emb);
      if (!raw.length) return;
      const xs = raw.map((p) => p.o.emb[0]), ys = raw.map((p) => p.o.emb[1]);
      const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
      const sx = (W - 2 * PAD) / (x1 - x0 || 1), sy = (H - 2 * PAD) / (y1 - y0 || 1);
      this.pts = raw.map(({ s, o }) => ({ id: s.id, label: s.label, o,
        x: PAD + (o.emb[0] - x0) * sx, y: H - PAD - (o.emb[1] - y0) * sy, on: state.visSet.has(s.id) }));
      const fill = COLORINGS[state.colorBy].fill;
      for (const pass of [false, true]) {
        for (const p of this.pts) {
          if (p.on !== pass) continue;
          const col = fill(p, pass ? SHOWN : DIM);
          ctx.beginPath(); ctx.arc(p.x, p.y, R, 0, 2 * Math.PI);
          if (p.o.correct) { ctx.fillStyle = col; ctx.fill(); } else { ctx.strokeStyle = col; ctx.lineWidth = 1.3; ctx.stroke(); }
        }
      }
      ctx.fillStyle = "#8b94a3"; ctx.font = "10px system-ui";
      ctx.fillText("PC1 →", W - PAD - 30, H - 3);
      ctx.save(); ctx.translate(10, PAD + 30); ctx.rotate(-Math.PI / 2); ctx.fillText("PC2 →", 0, 0); ctx.restore();
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
    if (p !== this.hover) { this.hover = p; this.layer.paint(); }
    if (!p) { tip.hidden = true; return; }
    tip.innerHTML = `<b>${esc(p.id)}</b><br>label ${esc(p.label)} · pred <span class="${p.o.correct ? "ok" : "no"}">${esc(p.o.pred)}</span>`
      + `<br>p(true) ${(p.o.p_true ?? 0).toFixed(3)}`;
    tip.hidden = false;
    const W = this.layer.W, left = p.x + 12 + tip.offsetWidth > W ? p.x - 12 - tip.offsetWidth : p.x + 12;
    tip.style.left = `${Math.max(0, left)}px`;
    tip.style.top = `${Math.max(0, p.y - tip.offsetHeight - 6)}px`;
  }
}
