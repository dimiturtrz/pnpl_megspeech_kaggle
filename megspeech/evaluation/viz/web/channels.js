import { $, h, store } from "./util.js";

const PICK_KEY = "megspeech-viz:channels";
const KIND = { mag: "magnetometers", grad: "gradiometers" };
const SENSOR = { 1: "mag", 2: "grad1", 3: "grad2" };
const PER_SITE = Object.keys(SENSOR).length;

const all = (C) => Array.from({ length: C }, (_, c) => c);
const spread = (members, n) => Array.from({ length: Math.min(n, members.length) },
  (_, r) => members[Math.floor((r * members.length) / Math.min(n, members.length))]);

function rmsBetween(meg, c, lo, hi) {
  let s = 0;
  for (let i = lo, o = c * meg.T; i < hi; i++) s += meg.data[o + i] ** 2;
  return Math.sqrt(s / Math.max(1, hi - lo));
}

function evoked(meg, target) {
  if (!target) return all(meg.C);
  const at = (t) => Math.min(meg.T, Math.max(0, Math.round((t - meg.t0) * meg.sfreq)));
  const [lo, hi] = target.map(at);
  if (hi <= lo) return all(meg.C);
  const ratio = all(meg.C).map((c) => rmsBetween(meg, c, lo, hi) / (rmsBetween(meg, c, 0, meg.T) || 1));
  return all(meg.C).sort((a, b) => ratio[b] - ratio[a]);
}

function sitesOf(sensors) {
  const by = new Map();
  sensors.names.forEach((name, c) => {
    const id = name.slice(3, 6);
    if (!by.has(id)) by.set(id, { id, channels: [], pts: [] });
    by.get(id).channels.push(c);
    if (sensors.xy?.[c]) by.get(id).pts.push(sensors.xy[c]);
  });
  return [...by.values()].map(({ id, channels, pts }) =>
    ({ id, channels, xy: pts.length ? [0, 1].map((k) => pts.reduce((a, p) => a + p[k], 0) / pts.length) : null }));
}

export class ChannelPick {
  constructor(onChange) {
    this.sensors = null;
    this.sites = null;
    this.sel = $("chsel");
    this.sel.onchange = () => { store.set(PICK_KEY, this.sel.value); onChange(); };
  }

  setSensors(sensors, weighted) {
    this.sensors = sensors?.names ? sensors : null;
    this.sites = this.sensors ? sitesOf(this.sensors) : null;
    const kinds = this.sensors ? [...new Set(this.sensors.kinds)] : [];
    const regions = Object.keys(this.sensors?.regions ?? {});
    const opt = (value, text, title = "") => h("option", { value, textContent: text, title });
    this.sel.replaceChildren(
      opt("spread", "spread", "every location, evenly spaced"),
      ...(weighted ? [opt("weight", "model weight", "locations ranked by the active model's summed integrated-gradients share for this word")] : []),
      opt("evoked", "word-evoked", "ranked by RMS inside the target word over RMS of the whole window"),
      ...kinds.map((k) => opt(`kind:${k}`, KIND[k] ?? k)),
      ...regions.map((r) => opt(`region:${r}`, r.toLowerCase())));
    const saved = store.get(PICK_KEY, "spread");
    if (saved.startsWith("near:") && this.site(saved.slice(5))?.xy) this.addNear(saved.slice(5));
    this.sel.value = [...this.sel.options].some((o) => o.value === saved) ? saved : "spread";
  }

  located() { return this.sites?.some((s) => s.xy) ? this.sites.filter((s) => s.xy) : null; }

  site(id) { return this.sites?.find((s) => s.id === id); }

  siteOf(c) { return this.sensors?.names[c]?.slice(3, 6) ?? c; }

  addNear(id) {
    const value = `near:${id}`;
    this.sel.querySelector('option[value^="near:"]')?.remove();
    this.sel.append(h("option", { value, textContent: `near #${id}`, title: "this helmet location and its nearest neighbours" }));
    return value;
  }

  near(id) {
    this.sel.value = this.addNear(id);
    store.set(PICK_KEY, this.sel.value);
  }

  nearest(id) {
    const [x, y] = this.site(id).xy, d = (s) => Math.hypot(s.xy[0] - x, s.xy[1] - y);
    return this.located().sort((a, b) => d(a) - d(b)).flatMap((s) => s.channels);
  }

  byWeight(weights) {
    const sum = {};
    weights.forEach((w, c) => { sum[this.siteOf(c)] = (sum[this.siteOf(c)] ?? 0) + w; });
    return all(weights.length).sort((a, b) => sum[this.siteOf(b)] - sum[this.siteOf(a)] || weights[b] - weights[a]);
  }

  members(meg, target, weights) {
    const [mode, arg] = this.sel.value.split(":");
    const named = this.sensors?.names.length === meg.C;
    if (named && mode === "weight" && weights?.length === meg.C) return { list: this.byWeight(weights), ranked: true };
    if (named && mode === "near" && this.site(arg)?.xy) return { list: this.nearest(arg), ranked: true };
    if (mode === "evoked") return { list: evoked(meg, target), ranked: true };
    if (named && mode === "kind") return { list: all(meg.C).filter((c) => this.sensors.kinds[c] === arg), ranked: false };
    if (named && mode === "region") return { list: this.sensors.regions[arg] ?? [], ranked: false };
    return { list: all(meg.C), ranked: false };
  }

  rows(meg, target, n, weights) {
    const { list, ranked } = this.members(meg, target, weights);
    const byChannel = !this.sites || this.sensors.names.length !== meg.C || this.sel.value.startsWith("kind:");
    if (byChannel) return { rows: ranked ? list.slice(0, n) : spread(list, n), total: list.length };
    const order = [...new Set(list.map((c) => this.siteOf(c)))], keep = new Set(list);
    const take = Math.ceil(n / PER_SITE);
    const ids = ranked ? order.slice(0, take) : spread(order, take);
    return { rows: ids.flatMap((id) => this.site(id).channels.filter((c) => keep.has(c))), total: list.length };
  }

  label(c) {
    const s = this.sensors;
    const name = s?.names[c];
    return name ? `#${name.slice(3, 6)} ${SENSOR[name[6]] ?? name}` : `ch ${c}`;
  }
}
