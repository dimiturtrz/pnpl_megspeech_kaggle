import { state, out } from "./state.js";
import { $, h } from "./util.js";

const fmtScore = (v) => (typeof v === "number" ? v.toFixed(3) : "");

export function buildModels(onChange) {
  const box = $("models");
  box.replaceChildren();
  for (const m of state.man.models) {
    const cb = h("input", { type: "checkbox", checked: state.checked.has(m.name), title: "show in comparison" });
    const name = h("button", { className: `mname${m.name === state.active ? " on" : ""}`, textContent: m.name,
      title: `${m.name} · ${m.track ?? ""} — make active` });
    cb.onchange = () => {
      if (cb.checked) state.checked.add(m.name);
      else if (state.checked.size > 1) state.checked.delete(m.name);
      if (!state.checked.has(state.active)) state.active = [...state.checked][0];
      onChange();
    };
    name.onclick = () => { state.checked.add(m.name); state.active = m.name; onChange(); };
    box.append(h("div", { className: "mrow" }, cb, name,
      h("span", { className: "mscore", textContent: `${m.metric ?? ""} ${fmtScore(m.score)}`.trim(), title: `${m.metric ?? "score"} on this sample · ${m.track ?? ""}` })));
  }
}

function select(box, label, options, current, set, onChange) {
  const s = h("select");
  for (const [v, t] of options) s.append(h("option", { value: v, textContent: t }));
  s.value = current;
  s.onchange = () => { set(s.value); onChange(); };
  box.append(h("label", { className: "mapgroup" }, h("span", { className: "glabel", textContent: label }), s));
}

export function buildFilters(onChange) {
  const box = $("filters");
  box.replaceChildren();
  for (const k of state.man.strata) {
    const vals = [...new Set(state.samples.map((s) => s.strata?.[k]).filter((v) => v != null))].sort();
    select(box, k, [["", "all"], ...vals.map((v) => [v, v])], state.filter.strata[k] ?? "",
      (v) => { state.filter.strata[k] = v; }, onChange);
  }
  select(box, "active model", [["all", "all"], ["correct", "✓ correct"], ["wrong", "✗ wrong"]], state.filter.correct,
    (v) => { state.filter.correct = v; }, onChange);
  $("q").oninput = () => { state.filter.q = $("q").value; onChange(); };
}

export class SampleList {
  constructor(onPick) {
    this.onPick = onPick;
    this.rows = new Map();
    this.on = null;
    $("list").addEventListener("click", (e) => {
      const r = e.target.closest(".row");
      if (r) onPick(r.dataset.id);
    });
  }

  render() {
    const frag = document.createDocumentFragment();
    this.rows.clear();
    for (const s of state.visible) {
      const o = out(state.active, s.id);
      const r = h("div", { className: "row", title: s.text },
        h("span", { className: "rid", textContent: s.id }),
        h("span", { className: "rlab", textContent: s.label }),
        h("span", { className: `badge ${o ? (o.correct ? "ok" : "no") : ""}`, textContent: o ? (o.correct ? "✓" : "✗") : "·" }));
      r.dataset.id = s.id;
      this.rows.set(s.id, r);
      frag.append(r);
    }
    $("list").replaceChildren(frag);
    $("count").textContent = `${state.visible.length} / ${state.samples.length}`;
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
