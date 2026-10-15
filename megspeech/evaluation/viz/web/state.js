import { setLabels } from "./util.js";
export const state = {
  man: null,
  samples: [],
  byId: new Map(),
  models: new Map(),
  checked: new Set(),
  active: null,
  sel: null,
  colorBy: "label",
  filter: { strata: {}, correct: "all", q: "" },
  visible: [],
  visSet: new Set(),
};

export const out = (model, id) => state.models.get(model)?.samples[id];
export const checkedModels = () => state.man.models.map((m) => m.name).filter((n) => state.checked.has(n));

const CORRECT = {
  all: () => true,
  correct: (o) => !!o?.correct,
  wrong: (o) => !!o && !o.correct,
};

export function applyFilter() {
  const f = state.filter, q = f.q.trim().toLowerCase(), keep = CORRECT[f.correct];
  state.visible = state.samples.filter((s) =>
    Object.entries(f.strata).every(([k, v]) => !v || s.strata?.[k] === v)
    && keep(out(state.active, s.id))
    && (!q || `${s.id} ${s.label} ${s.text}`.toLowerCase().includes(q)));
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
  for (const m of state.man.models) state.checked.add(m.name);
  setLabels(state.man.models[0]?.labels ?? []);
}
