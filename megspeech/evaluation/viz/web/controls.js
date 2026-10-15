import { state, out } from "./state.js";
import { $, h } from "./util.js";

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
    this.open = new Set();
    this.on = null;
    $("list").addEventListener("click", (e) => {
      const hit = e.target.closest("[data-id]");
      if (hit) { onPick(hit.dataset.id); return; }
      const head = e.target.closest(".sent");
      if (head) this.toggle(head.dataset.key);
    });
  }

  static key(s) { return s.sentence ?? s.id; }

  groups() {
    const groups = new Map();
    for (const s of state.visible) {
      const k = SampleList.key(s);
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push(s);
    }
    return groups;
  }

  toggle(k) {
    if (this.open.has(k)) this.open.delete(k); else this.open.add(k);
    this.render();
  }

  word(s) {
    const o = out(state.active, s.id), [a, b] = s.target ?? [0, 0];
    const mark = h("span", { className: `sw ${o ? (o.correct ? "ok" : "no") : ""}`, textContent: s.text.slice(a, b) || s.label });
    mark.dataset.id = s.id;
    this.rows.set(s.id, mark);
    return [mark, b];
  }

  head(k, words) {
    words = [...words].sort((x, y) => (x.target?.[0] ?? 0) - (y.target?.[0] ?? 0));
    const text = words[0].text, parts = [];
    let at = 0, right = 0;
    for (const s of words) {
      const [a] = s.target ?? [0, 0], [mark, b] = this.word(s);
      parts.push(text.slice(at, a), mark);
      at = b;
      right += out(state.active, s.id)?.correct ? 1 : 0;
    }
    parts.push(text.slice(at));
    const open = this.open.has(k);
    const row = h("div", { className: `sent${open ? " open" : ""}`, title: `${words[0].source} · ${k}` },
      h("span", { className: "rsrc", textContent: `${open ? "▾" : "▸"} ${words[0].source.slice(0, 3)}` }),
      h("span", { className: "stext" }, ...parts),
      h("span", { className: "badge", textContent: `${right}/${words.length}` }));
    row.dataset.key = k;
    return row;
  }

  render() {
    const frag = document.createDocumentFragment(), groups = this.groups();
    this.rows.clear();
    for (const [k, words] of groups) {
      frag.append(this.head(k, words));
      if (!this.open.has(k)) continue;
      for (const s of words) {
        const o = out(state.active, s.id);
        const r = h("div", { className: "row", title: `${s.id} · ${s.label}` },
          h("span", { className: "rid", textContent: s.id }),
          h("span", { className: "rlab", textContent: `${s.label}${o ? ` → ${o.pred}` : ""}` }),
          h("span", { className: `badge ${o ? (o.correct ? "ok" : "no") : ""}`, textContent: o ? (o.correct ? "✓" : "✗") : "·" }));
        r.dataset.id = s.id;
        this.rows.set(s.id, r);
        frag.append(r);
      }
    }
    $("list").replaceChildren(frag);
    $("count").textContent = `${groups.size} sentences · ${state.visible.length} / ${state.samples.length} words`;
    this.on = null;
    this.mark();
  }

  mark() {
    const k = SampleList.key(state.byId.get(state.sel) ?? {});
    if (state.sel && !this.open.has(k) && state.visSet.has(state.sel)) { this.open.add(k); this.render(); return; }
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
