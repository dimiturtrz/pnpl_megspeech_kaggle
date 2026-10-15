import { $, Layer, dataUrl, niceStep, clamp, store } from "./util.js";
import { ChannelPick } from "./channels.js";
import { state, weightOf } from "./state.js";

const PAD = { l: 46, r: 10, t: 6, b: 18 };
const AMP = 0.3;
const RMS_ROWS = 2;
const CACHE_MAX = 24;
const MAX_TICKS = 8;
const LABEL_MIN_ROW = 9;
const MEG_KEY = "megspeech-viz:meg-open";
const COL = { grid: "#222937", axis: "#8b94a3", chan: "rgba(159,179,209,.75)", rms: "#5b8def", wave: "#7fdca4", head: "#ff6a5a", target: "rgba(91,141,239,.16)" };

const COLD = [159, 179, 209], HOT = [255, 140, 40], FAINT = 0.3, FULL = 255;
const weighted = (w) => `rgba(${COLD.map((c, i) => Math.round(c + (HOT[i] - c) * w)).join(",")},${FAINT + (1 - FAINT) * w})`;

const cache = new Map();
function fetchBuf(url) {
  if (!cache.has(url)) {
    cache.set(url, fetch(url).then((r) => {
      if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
      return r.arrayBuffer();
    }).catch((e) => { cache.delete(url); throw e; }));
    if (cache.size > CACHE_MAX) cache.delete(cache.keys().next().value);
  }
  return cache.get(url);
}

function parseWav(buf) {
  const v = new DataView(buf);
  const tag = (o) => String.fromCharCode(...new Uint8Array(buf, o, 4));
  let off = 12, fmt = null;
  while (off + 8 <= v.byteLength) {
    const id = tag(off), len = v.getUint32(off + 4, true), body = off + 8;
    if (id === "fmt ") fmt = { ch: v.getUint16(body + 2, true), sr: v.getUint32(body + 4, true) };
    if (id === "data" && fmt) {
      const stride = 2 * fmt.ch, n = Math.floor(Math.min(len, v.byteLength - body) / stride);
      const pcm = new Float32Array(n);
      for (let i = 0; i < n; i++) pcm[i] = v.getInt16(body + i * stride, true) / 32768;
      return { pcm, sfreq: fmt.sr };
    }
    off = body + len + (len & 1);
  }
  throw new Error("wav: no fmt/data chunk");
}

function rmsOf(data, C, T) {
  const r = new Float32Array(T);
  for (let c = 0; c < C; c++) for (let i = 0, o = c * T; i < T; i++) r[i] += data[o + i] ** 2;
  let mx = 1e-9;
  for (let i = 0; i < T; i++) { r[i] = Math.sqrt(r[i] / C); mx = Math.max(mx, r[i]); }
  return { rms: r, rmsMax: mx };
}

export class Signals {
  constructor() {
    this.megL = new Layer($("meg"));
    this.audL = new Layer($("wave"));
    this.player = $("player");
    this.meg = null; this.aud = null; this.target = null; this.win = [0, 1]; this.cursor = 0; this.token = 0; this.blobUrl = null;
    for (const L of [this.megL, this.audL]) {
      L.overlay = (ctx, W, H) => this.playhead(ctx, W, H);
      L.cv.addEventListener("click", (e) => this.seek(e, L));
    }
    $("nch").oninput = () => this.drawMeg();
    this.pick = new ChannelPick(() => this.drawMeg());
    $("megtoggle").onclick = () => this.setMegOpen(!this.megOpen);
    this.setMegOpen(store.get(MEG_KEY, true));
    const loop = () => { this.paint(); if (!this.player.paused) requestAnimationFrame(loop); };
    this.player.addEventListener("play", () => requestAnimationFrame(loop));
    for (const ev of ["seeked", "pause", "timeupdate"]) this.player.addEventListener(ev, () => this.paint());
  }

  setMegOpen(open) {
    this.megOpen = open;
    store.set(MEG_KEY, open);
    $("mid").classList.toggle("megoff", !open);
    $("megtoggle").setAttribute("aria-expanded", String(open));
    this.draw();
  }

  togglePlay() {
    if (!this.aud) return;
    if (this.player.paused) this.player.play(); else this.player.pause();
  }

  setSensors(sensors) { this.pick.setSensors(sensors, state.saliency.size > 0); }

  async show(s) {
    const tok = ++this.token;
    this.s = s;
    this.player.pause();
    const [megBuf, wavBuf] = await Promise.all([
      fetchBuf(dataUrl(s.meg.file)),
      s.audio ? fetchBuf(dataUrl(s.audio.file)) : null,
    ]);
    if (tok !== this.token) return;
    const { channels: C, times: T, sfreq, t0 } = s.meg, data = new Float32Array(megBuf);
    if (data.length !== C * T) throw new Error(`${s.meg.file}: ${data.length} floats, expected ${C}×${T}`);
    this.meg = { data, C, T, sfreq, t0, ...rmsOf(data, C, T) };
    this.aud = wavBuf ? { ...parseWav(wavBuf), t0: s.audio.t0 } : null;
    const megEnd = t0 + T / sfreq;
    this.win = this.aud
      ? [Math.min(t0, this.aud.t0), Math.max(megEnd, this.aud.t0 + this.aud.pcm.length / this.aud.sfreq)]
      : [t0, megEnd];
    this.cursor = this.win[0];
    this.target = s.target_t ?? null;
    $("nch").max = C;
    $("meghead").textContent = `${C} ch · ${sfreq} Hz · ${(T / sfreq).toFixed(2)} s`;
    $("audhead").textContent = this.aud ? `${this.aud.sfreq} Hz · ${(this.aud.pcm.length / this.aud.sfreq).toFixed(2)} s`
      : `none — pnpl ships stimulus audio for Sherlock only`;
    $("mid").classList.toggle("noaud", !this.aud);
    if (this.blobUrl) URL.revokeObjectURL(this.blobUrl);
    this.blobUrl = wavBuf ? URL.createObjectURL(new Blob([wavBuf], { type: "audio/wav" })) : null;
    this.player.hidden = !wavBuf;
    if (this.blobUrl) this.player.src = this.blobUrl; else this.player.removeAttribute("src");
    this.draw();
  }

  draw() {
    this.drawMeg();
    this.drawAudio();
  }

  x(t, W) { return PAD.l + ((t - this.win[0]) / (this.win[1] - this.win[0])) * (W - PAD.l - PAD.r); }
  t(x, W) { return this.win[0] + ((x - PAD.l) / (W - PAD.l - PAD.r)) * (this.win[1] - this.win[0]); }

  tNow() { return this.aud ? this.aud.t0 + this.player.currentTime : this.cursor; }

  axis(ctx, W, H) {
    const [a, b] = this.win, step = niceStep(b - a, MAX_TICKS);
    ctx.font = "10px system-ui"; ctx.lineWidth = 1; ctx.textBaseline = "alphabetic"; ctx.textAlign = "center";
    for (let t = Math.ceil(a / step) * step; t <= b + 1e-9; t += step) {
      const x = this.x(t, W);
      ctx.strokeStyle = COL.grid; ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, H - PAD.b); ctx.stroke();
      ctx.fillStyle = COL.axis; ctx.fillText(`${+t.toFixed(3)} s`, x, H - 4);
    }
    ctx.textAlign = "left";
    if (!this.target) return;
    const [x0, x1] = this.target.map((t) => this.x(t, W));
    ctx.fillStyle = COL.target;
    ctx.fillRect(x0, PAD.t, x1 - x0, H - PAD.t - PAD.b);
  }

  trace(ctx, arr, off, n, t0, sfreq, y0, k, W) {
    const x0 = this.x(t0, W), x1 = this.x(t0 + (n - 1) / sfreq, W), cols = Math.max(1, Math.round(x1 - x0));
    ctx.beginPath();
    if (n <= 2 * cols) {
      for (let i = 0; i < n; i++) {
        const x = x0 + ((x1 - x0) * i) / Math.max(1, n - 1), y = y0 - arr[off + i] * k;
        if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      }
    } else {
      for (let c = 0; c < cols; c++) {
        let lo = Infinity, hi = -Infinity;
        for (let i = Math.floor((c * n) / cols), e = Math.floor(((c + 1) * n) / cols); i < e; i++) {
          const v = arr[off + i]; if (v < lo) lo = v; if (v > hi) hi = v;
        }
        if (c) ctx.lineTo(x0 + c, y0 - hi * k); else ctx.moveTo(x0 + c, y0 - hi * k);
        ctx.lineTo(x0 + c, y0 - lo * k);
      }
    }
    ctx.stroke();
  }

  drawMeg() {
    if (!this.megOpen) return;
    const m = this.meg, weights = this.s ? weightOf(state.active, this.s.id) : null;
    const { rows, total } = m ? this.pick.rows(m, this.target, +$("nch").value, weights) : { rows: [], total: 0 };
    const n = rows.length;
    $("nchlabel").textContent = m ? `${n}/${total}` : "";
    this.megL.draw((ctx, W, H) => {
      this.axis(ctx, W, H);
      if (!m) return;
      const rowH = (H - PAD.t - PAD.b) / (n + RMS_ROWS), rmsBase = PAD.t + rowH * RMS_ROWS;
      ctx.strokeStyle = COL.rms; ctx.lineWidth = 1.4;
      this.trace(ctx, m.rms, 0, m.T, m.t0, m.sfreq, rmsBase, (rowH * RMS_ROWS - 2) / m.rmsMax, W);
      ctx.fillStyle = COL.rms; ctx.textBaseline = "middle"; ctx.fillText("RMS", 4, rmsBase - rowH);
      ctx.strokeStyle = COL.chan; ctx.lineWidth = 0.8;
      for (let r = 0; r < n; r++) {
        const c = rows[r], y0 = rmsBase + (r + 0.5) * rowH;
        ctx.strokeStyle = weights ? weighted(weights[c] / FULL) : COL.chan;
        this.trace(ctx, m.data, c * m.T, m.T, m.t0, m.sfreq, y0, rowH * AMP, W);
        if (rowH >= LABEL_MIN_ROW) { ctx.fillStyle = COL.axis; ctx.fillText(this.pick.label(c), 4, y0); }
      }
    });
  }

  drawAudio() {
    this.audL.draw((ctx, W, H) => {
      this.axis(ctx, W, H);
      const a = this.aud;
      if (!a) return;
      ctx.strokeStyle = COL.wave; ctx.lineWidth = 1;
      const y0 = (PAD.t + H - PAD.b) / 2;
      this.trace(ctx, a.pcm, 0, a.pcm.length, a.t0, a.sfreq, y0, (H - PAD.t - PAD.b) / 2, W);
    });
  }

  playhead(ctx, W, H) {
    if (!this.meg) return;
    const t = this.tNow(), x = this.x(t, W);
    ctx.strokeStyle = COL.head; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, H - PAD.b); ctx.stroke();
  }

  paint() {
    if (this.megOpen) this.megL.paint();
    this.audL.paint();
    $("tlabel").textContent = this.meg ? `${this.tNow().toFixed(3)} s` : "";
  }

  seek(e, L) {
    if (!this.meg) return;
    const r = L.cv.getBoundingClientRect(), t = clamp(this.t(e.clientX - r.left, r.width), ...this.win);
    if (this.aud) this.player.currentTime = clamp(t - this.aud.t0, 0, this.player.duration || 0);
    else this.cursor = t;
    this.paint();
  }
}
