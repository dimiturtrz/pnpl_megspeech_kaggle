import { heat, percentiles } from "./heat.js";
import { store } from "./util.js";

const COL = { head: "rgba(160,170,190,.55)", idle: "rgba(120,130,150,.55)", ring: "#fff" };
const SIZE_KEY = "megspeech-viz:helmet", MIN_PX = 64, MAX_SHARE = 0.8;
const PAD = 0.08, FILL = 0.9, DOT = 0.024, HIT = 0.05;

export class HelmetMap {
  constructor(canvas, pick, onPick) {
    this.canvas = canvas;
    this.pick = pick;
    this.sites = null;
    this.box = canvas.parentElement;
    this.resize(store.get(SIZE_KEY, 0));
    this.box.querySelector(".grip").onpointerdown = (e) => this.drag(e, onPick);
    canvas.onclick = (e) => {
      const site = this.hit(e);
      if (site) { pick.near(site.id); onPick(); }
    };
  }

  resize(px) {
    if (!px) return;
    const most = (this.box.parentElement?.clientWidth || Infinity) * MAX_SHARE;
    const size = `${Math.round(Math.min(most, Math.max(MIN_PX, px)))}px`;
    Object.assign(this.box.style, { width: size, height: size });
  }

  drag(e, onResize) {
    e.preventDefault();
    const s0 = this.box.offsetWidth, x0 = e.clientX, y0 = e.clientY;
    const move = (ev) => { this.resize(s0 + Math.max(ev.clientY - y0, x0 - ev.clientX)); onResize(); };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      store.set(SIZE_KEY, this.box.offsetWidth);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }

  setSites(sites) {
    this.sites = sites?.length ? sites : null;
    this.box.hidden = !this.sites;
    if (!this.sites) return;
    const xs = this.sites.map((s) => s.xy[0]), ys = this.sites.map((s) => s.xy[1]);
    this.centre = [(Math.min(...xs) + Math.max(...xs)) / 2, (Math.min(...ys) + Math.max(...ys)) / 2];
    this.reach = Math.max(...this.sites.map(({ xy }) => Math.hypot(xy[0] - this.centre[0], xy[1] - this.centre[1])));
  }

  frame() {
    const r = this.canvas.getBoundingClientRect(), s = Math.min(r.width, r.height);
    return { s, cx: r.width / 2, cy: r.height / 2, R: s * (0.5 - PAD), w: r.width, h: r.height };
  }

  at({ xy }, { cx, cy, R }) {
    const k = (FILL * R) / this.reach;
    return [cx + (xy[0] - this.centre[0]) * k, cy - (xy[1] - this.centre[1]) * k];
  }

  hit(e) {
    if (!this.sites) return null;
    const f = this.frame(), r = this.canvas.getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    const dist = (site) => { const [x, y] = this.at(site, f); return Math.hypot(x - px, y - py); };
    const best = this.sites.reduce((a, b) => (dist(b) < dist(a) ? b : a));
    return dist(best) < HIT * f.s ? best : null;
  }

  draw(weights, rows) {
    if (!this.sites) return;
    const f = this.frame(), dpr = window.devicePixelRatio || 1;
    Object.assign(this.canvas, { width: Math.round(f.w * dpr), height: Math.round(f.h * dpr) });
    const ctx = this.canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const { cx, cy, R } = f;
    ctx.strokeStyle = COL.head; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI);
    ctx.moveTo(cx - R * 0.12, cy - R * 0.99); ctx.lineTo(cx, cy - R * 1.1); ctx.lineTo(cx + R * 0.12, cy - R * 0.99);
    ctx.stroke();
    const sums = weights ? this.sites.map((site) => site.channels.reduce((a, c) => a + weights[c], 0)) : null;
    const q = sums ? percentiles(sums) : null, shown = new Set(rows.map((c) => this.pick.siteOf(c)));
    this.sites.forEach((site, i) => {
      const [x, y] = this.at(site, f);
      ctx.fillStyle = q ? heat(q[i], 1) : COL.idle;
      ctx.beginPath(); ctx.arc(x, y, DOT * f.s, 0, 2 * Math.PI); ctx.fill();
      if (shown.has(site.id)) { ctx.strokeStyle = COL.ring; ctx.lineWidth = 1.5; ctx.stroke(); }
    });
  }
}
