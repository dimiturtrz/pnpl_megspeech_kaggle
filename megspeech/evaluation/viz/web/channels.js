import { $, h, store } from "./util.js";

const PICK_KEY = "megspeech-viz:channels";
const KIND = { mag: "magnetometers", grad: "gradiometers" };
const SHORT = { mag: "m", grad: "g" };

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

function byWeight(weights) {
  return all(weights.length).sort((a, b) => weights[b] - weights[a]);
}

export class ChannelPick {
  constructor(onChange) {
    this.sensors = null;
    this.sel = $("chsel");
    this.sel.onchange = () => { store.set(PICK_KEY, this.sel.value); onChange(); };
  }

  setSensors(sensors, weighted) {
    this.sensors = sensors?.names ? sensors : null;
    const kinds = this.sensors ? [...new Set(this.sensors.kinds)] : [];
    const regions = Object.keys(this.sensors?.regions ?? {});
    const opt = (value, text, title = "") => h("option", { value, textContent: text, title });
    this.sel.replaceChildren(
      opt("spread", "spread", "every channel, evenly spaced by index"),
      ...(weighted ? [opt("weight", "model weight", "ranked by the active model's integrated-gradients share for this word")] : []),
      opt("evoked", "word-evoked", "ranked by RMS inside the target word over RMS of the whole window"),
      ...kinds.map((k) => opt(`kind:${k}`, KIND[k] ?? k)),
      ...regions.map((r) => opt(`region:${r}`, r.toLowerCase())));
    const saved = store.get(PICK_KEY, "spread");
    this.sel.value = [...this.sel.options].some((o) => o.value === saved) ? saved : "spread";
  }

  members(meg, target, weights) {
    const [mode, arg] = this.sel.value.split(":");
    if (mode === "weight" && weights?.length === meg.C) return { list: byWeight(weights), ranked: true };
    const named = this.sensors?.names.length === meg.C;
    if (mode === "evoked") return { list: evoked(meg, target), ranked: true };
    if (named && mode === "kind") return { list: all(meg.C).filter((c) => this.sensors.kinds[c] === arg), ranked: false };
    if (named && mode === "region") return { list: this.sensors.regions[arg] ?? [], ranked: false };
    return { list: all(meg.C), ranked: false };
  }

  rows(meg, target, n, weights) {
    const { list, ranked } = this.members(meg, target, weights);
    return { rows: ranked ? list.slice(0, n) : spread(list, n), total: list.length };
  }

  label(c) {
    const s = this.sensors;
    return s?.names[c] ? `${s.names[c].replace("MEG", "")} ${SHORT[s.kinds[c]] ?? ""}` : `ch ${c}`;
  }
}
