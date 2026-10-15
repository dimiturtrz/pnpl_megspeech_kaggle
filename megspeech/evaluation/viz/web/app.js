import { state, load, applyFilter, out, errorOf, errorShare } from "./state.js";
import { $, esc, wordCount, errorColor } from "./util.js";
import { buildModels, buildFilters, SampleTable } from "./controls.js";
import { Signals } from "./signals.js";
import { ModelPanel } from "./model.js";
import { Split } from "./split.js";

const TYPING = new Set(["INPUT", "SELECT", "TEXTAREA"]);
const KEYS = { ArrowDown: () => list.step(1), ArrowUp: () => list.step(-1), " ": () => signals.togglePlay() };

const list = new SampleTable(select);
list.onResort = () => refilter();
const signals = new Signals();
const panel = new ModelPanel(select);

function status(msg) { $("status").textContent = msg; }

const errorTint = (o) => (o ? errorColor(errorShare(state.active, o)) : "var(--mut)");
const errorTip = (o) => `error ${errorOf(o).toFixed(2)}, worse than ${Math.round(100 * errorShare(state.active, o))}% of this model's words`;

function sentenceHtml(s) {
  let html = "", at = 0;
  for (const w of state.bySentence.get(s.sentence ?? s.id) ?? [s]) {
    const [a, b] = w.target ?? [0, 0];
    if (a < at) continue;
    const o = out(state.active, w.id), tip = o ? `${w.label} · ${errorTip(o)}` : w.label;
    html += `${esc(s.text.slice(at, a))}<span class="elig${w.id === s.id ? " sel" : ""}" data-id="${esc(w.id)}" `
      + `style="--err:${errorTint(o)}" title="${esc(tip)}">${esc(s.text.slice(a, b))}</span>`;
    at = b;
  }
  return html + esc(s.text.slice(at));
}

function renderText() {
  const s = state.byId.get(state.sel);
  if (!s) return;
  $("sentence").innerHTML = sentenceHtml(s);
  const o = out(state.active, s.id);
  const exact = o?.pred === s.label;
  $("preds").innerHTML = `<span>label <b class="ok">${esc(s.label)}</b></span>`
    + `<span>pred <b class="${exact ? "ok" : "no"}">${esc(o?.pred ?? "—")}</b></span>`
    + (o?.rank ? `<span style="color:${errorTint(o)}" title="${errorTip(o)}">true word at rank ${o.rank}</span>` : "");
  const strata = Object.entries(s.strata ?? {}).filter(([k]) => k !== "source").map(([k, v]) => `${k}=${v}`).join(" · ");
  $("meta").textContent = `subject ${s.subject} · session ${s.session} · ${wordCount(s.text)} words${strata ? ` · ${strata}` : ""}`;
}

function writeHash() {
  history.replaceState(null, "", `#${new URLSearchParams({ s: state.sel ?? "", m: state.active ?? "" })}`);
}

function readHash() {
  const p = new URLSearchParams(location.hash.slice(1)), m = p.get("m");
  if (m && state.models.has(m)) state.active = m;
  return p.get("s");
}

function select(id) {
  if (!state.byId.has(id)) return;
  state.sel = id;
  writeHash();
  list.mark();
  renderText();
  panel.showSample();
  status("");
  signals.show(state.byId.get(id)).catch((e) => status(`load error: ${e.message}`));
}

function refilter() {
  applyFilter();
  list.render();
  panel.draw();
}

function modelsChanged() {
  buildModels(modelsChanged);
  refilter();
  renderText();
  panel.showSample();
  writeHash();
}

function restore() {
  const want = readHash();
  modelsChanged();
  select(state.byId.has(want) ? want : (state.visible[0] ?? state.samples[0])?.id);
}

async function main() {
  await load();
  signals.setSensors(state.man.sensors);
  buildFilters(refilter);
  restore();
  window.addEventListener("hashchange", restore);
  $("sentence").onclick = (e) => { const id = e.target.closest(".elig")?.dataset.id; if (id) select(id); };
  document.addEventListener("keydown", (e) => {
    const act = KEYS[e.key];
    if (!act || TYPING.has(e.target.tagName) || e.target.tagName === "AUDIO") return;
    e.preventDefault();
    act();
  });
  new Split($("mid"), ["ptext", "pmeg", "paud"].map($), [1, 2, 1], "megspeech-viz:split");
  let pending = 0;
  const resized = new ResizeObserver(() => {
    cancelAnimationFrame(pending);
    pending = requestAnimationFrame(() => { signals.draw(); panel.draw(); });
  });
  for (const el of [document.querySelector("main"), $("megbox"), $("wave")]) resized.observe(el);
}

main().catch((e) => status(`load error: ${e.message} — serve with \`python -m megspeech.evaluation.viz serve\``));
