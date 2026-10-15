import { setLabels, wordCount } from "./util.js";
export const state = {
  man: null,
  samples: [],
  byId: new Map(),
  models: new Map(),
  active: null,
  sel: null,
  colorBy: "label",
  focus: null,
  projection: null,
  filter: { strata: {}, ranges: {}, q: "" },
  sort: null,
  visible: [],
  visSet: new Set(),
};

export const out = (model, id) => state.models.get(model)?.samples[id];

export const RANGES = {
  length: { label: "sentence words", of: (s) => wordCount(s.text) },
};

const within = (v, [lo, hi]) => (lo == null || v >= lo) && (hi == null || v <= hi);

export const activeFilters = () => {
  const f = state.filter;
  return Object.values(f.strata).filter(Boolean).length
    + Object.values(f.ranges).filter(([lo, hi]) => lo != null || hi != null).length;
};

export const errorOf = (o) => o.error ?? 1 - (o.p_true ?? 0);

const compare = (a, b) => (a < b ? -1 : a > b ? 1 : 0);

export function applyFilter() {
  const f = state.filter, q = f.q.trim().toLowerCase();
  state.visible = state.samples.filter((s) =>
    Object.entries(f.strata).every(([k, v]) => !v || s.strata?.[k] === v)
    && Object.entries(f.ranges).every(([k, r]) => within(RANGES[k].of(s), r))
    && (!q || `${s.id} ${s.label} ${s.text}`.toLowerCase().includes(q)));
  const { by, dir } = state.sort ?? {};
  if (by) {
    const keyed = state.visible.map((s) => [by(s), s]);
    keyed.sort(([a], [b]) => dir * compare(a, b));
    state.visible = keyed.map(([, s]) => s);
  }
  state.visSet = new Set(state.visible.map((s) => s.id));
}

async function getJson(path) {
  const r = await fetch(`data/${path}`);
  if (!r.ok) throw new Error(`data/${path}: HTTP ${r.status}`);
  return r.json();
}

export async function load() {
  state.man = await getJson("manifest.json");
  const [samples, ...models] = await Promise.all([
    getJson(state.man.samples),
    ...state.man.models.map((m) => getJson(`models/${encodeURIComponent(m.name)}.json`)),
  ]);
  state.samples = samples;
  for (const s of samples) state.byId.set(s.id, s);
  state.man.models.forEach((m, i) => state.models.set(m.name, models[i]));
  state.active = state.man.models[0]?.name ?? null;
  setLabels(state.man.models[0]?.labels ?? []);
}
