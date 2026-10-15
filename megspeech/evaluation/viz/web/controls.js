import { state, out, RANGES, activeFilters, errorOf } from "./state.js";
import { $, h, store, errorColor } from "./util.js";

const fmtScore = (v) => (typeof v === "number" ? v.toFixed(3) : "");

export function buildModels(onChange) {
  const box = $("models");
  const pick = h("select", { className: "mselect", title: "active model" });
  for (const m of state.man.models) {
    pick.append(h("option", { value: m.name, textContent: `${fmtScore(m.score)}  ${m.name}`.trim() }));
  }
  pick.value = state.active ?? "";
  pick.onchange = () => { state.active = pick.value; onChange(); };
  const metric = state.man.models[0]?.metric ?? "score";
  box.replaceChildren(pick, h("div", { className: "mut", textContent: `${metric} on this sample` }));
}

function select(box, label, options, current, set, onChange) {
  const s = h("select");
  for (const [v, t] of options) s.append(h("option", { value: v, textContent: t }));
  s.value = current;
  s.onchange = () => { set(s.value); onChange(); };
  box.append(h("label", { className: "mapgroup" }, h("span", { className: "glabel", textContent: label }), s));
}

function range(box, k, { label, of }, onChange) {
  const vals = state.samples.map(of), lo = Math.min(...vals), hi = Math.max(...vals);
  const cur = state.filter.ranges[k] ?? [null, null];
  const bound = (i, word) => {
    const e = h("input", { type: "number", min: lo, max: hi, step: 1, placeholder: [lo, hi][i], value: cur[i] ?? "" });
    e.oninput = () => {
      const r = [...(state.filter.ranges[k] ?? [null, null])];
      r[i] = e.value === "" ? null : Number(e.value);
      state.filter.ranges[k] = r;
      onChange();
    };
    return [h("span", { className: "mut", textContent: word }), e];
  };
  box.append(h("div", { className: "mapgroup wide" }, h("span", { className: "glabel", textContent: label }),
    h("div", { className: "range" }, ...bound(0, "from"), ...bound(1, "to"))));
}

export function buildFilters(onChange) {
  const box = $("filters"), dlg = $("filterbox"), btn = $("filterbtn");
  const changed = () => {
    const n = activeFilters();
    btn.textContent = n ? `filters · ${n}` : "filters";
    btn.classList.toggle("on", n > 0);
    onChange();
  };
  box.replaceChildren();
  for (const k of state.man.strata) {
    if (RANGES[k]) { range(box, k, RANGES[k], changed); continue; }
    const vals = [...new Set(state.samples.map((s) => s.strata?.[k]).filter((v) => v != null))].sort();
    select(box, k, [["", "all"], ...vals.map((v) => [v, v])], state.filter.strata[k] ?? "",
      (v) => { state.filter.strata[k] = v; }, changed);
  }
  btn.onclick = () => dlg.showModal();
  dlg.onclick = (e) => { if (e.target === dlg) dlg.close(); };
  $("filterdone").onclick = () => dlg.close();
  $("filterreset").onclick = () => {
    Object.assign(state.filter, { strata: {}, ranges: {} });
    buildFilters(onChange);
    changed();
  };
  $("q").oninput = () => { state.filter.q = $("q").value; onChange(); };
}

const SORT_KEY = "megspeech-viz:sort";
const rankOf = (s, o) => {
  const i = o?.topk.findIndex(([lab]) => lab === s.label) ?? -1;
  return i < 0 ? Infinity : i + 1;
};
const COLUMNS = [
  { key: "id", title: "#", by: (s) => s.id, text: (s) => s.id, cls: () => "mut" },
  { key: "word", title: "word", by: (s) => s.label, text: (s) => s.label },
  { key: "pred", title: "pred", by: (s, o) => o?.pred ?? "", text: (s, o) => o?.pred ?? "—",
    cls: (s, o) => (o ? (o.correct ? "ok" : "no") : "") },
  { key: "error", title: "error", by: (s, o) => (o ? errorOf(o) : -1), text: (s, o) => (o ? errorOf(o).toFixed(2) : ""), num: true,
    tint: (s, o) => (o ? errorColor(errorOf(o)) : null) },
  { key: "rank", title: "rank", by: rankOf, text: (s, o) => (o ? (rankOf(s, o) === Infinity ? `>${o.topk.length}` : rankOf(s, o)) : ""),
    num: true, title_: "rank of the true word in the model's top-k" },
  { key: "src", title: "src", by: (s) => s.source, text: (s) => s.source.slice(0, 3), cls: () => "mut" },
];
const DESCENDING = new Set(["error", "rank"]);

export class SampleTable {
  constructor(onPick) {
    this.onPick = onPick;
    this.rows = new Map();
    this.on = null;
    const saved = store.get(SORT_KEY, { key: "id", dir: 1 });
    this.sortBy(COLUMNS.some((c) => c.key === saved.key) ? saved : { key: "id", dir: 1 });
    $("list").addEventListener("click", (e) => {
      const head = e.target.closest("[data-col]");
      if (head) { this.resort(head.dataset.col); return; }
      const hit = e.target.closest("[data-id]");
      if (hit) onPick(hit.dataset.id);
    });
  }

  sortBy({ key, dir }) {
    const col = COLUMNS.find((c) => c.key === key);
    state.sort = { key, dir, by: (s) => col.by(s, out(state.active, s.id)) };
    store.set(SORT_KEY, { key, dir });
  }

  resort(key) {
    const dir = state.sort.key === key ? -state.sort.dir : (DESCENDING.has(key) ? -1 : 1);
    this.sortBy({ key, dir });
    this.onResort();
  }

  header() {
    return h("div", { className: "trow thead" }, ...COLUMNS.map((c) => {
      const arrow = state.sort.key === c.key ? (state.sort.dir > 0 ? " ▲" : " ▼") : "";
      const cell = h("span", { className: c.num ? "num" : "", textContent: `${c.title}${arrow}`, title: c.title_ ?? `sort by ${c.title}` });
      cell.dataset.col = c.key;
      return cell;
    }));
  }

  row(s) {
    const o = out(state.active, s.id);
    const r = h("div", { className: "trow", title: s.text }, ...COLUMNS.map((c) => {
      const cell = h("span", { className: `${c.num ? "num " : ""}${c.cls?.(s, o) ?? ""}`, textContent: c.text(s, o) });
      const tint = c.tint?.(s, o);
      if (tint) cell.style.color = tint;
      return cell;
    }));
    r.dataset.id = s.id;
    this.rows.set(s.id, r);
    return r;
  }

  render() {
    this.rows.clear();
    $("list").replaceChildren(this.header(), ...state.visible.map((s) => this.row(s)));
    const sentences = new Set(state.visible.map((s) => s.sentence ?? s.id)).size;
    $("count").textContent = `${sentences} sentences · ${state.visible.length} / ${state.samples.length} words`;
    this.on = null;
    this.mark();
  }

  mark() {
    this.on?.classList.remove("on");
    this.on = this.rows.get(state.sel) ?? null;
    this.on?.classList.add("on");
    this.on?.scrollIntoView({ block: "nearest" });
  }

  step(d) {
    const v = state.visible;
    if (!v.length) return;
    const i = v.findIndex((s) => s.id === state.sel);
    this.onPick(v[i < 0 ? 0 : Math.max(0, Math.min(v.length - 1, i + d))].id);
  }
}
