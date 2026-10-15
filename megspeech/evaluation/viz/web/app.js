import { state, load, applyFilter, out } from "./state.js";
import { $, esc, wordCount } from "./util.js";
import { buildModels, buildFilters, SampleTable } from "./controls.js";
import { Signals } from "./signals.js";
import { ModelPanel } from "./model.js";

const TYPING = new Set(["INPUT", "SELECT", "TEXTAREA"]);
const KEYS = { ArrowDown: () => list.step(1), ArrowUp: () => list.step(-1), " ": () => signals.togglePlay() };

const list = new SampleTable(select);
list.onResort = () => refilter();
const signals = new Signals();
const panel = new ModelPanel(select);

function status(msg) { $("status").textContent = msg; }

function renderText() {
  const s = state.byId.get(state.sel);
  if (!s) return;
  const [a, b] = s.target ?? [0, 0];
  $("sentence").innerHTML = `${esc(s.text.slice(0, a))}<mark>${esc(s.text.slice(a, b))}</mark>${esc(s.text.slice(b))}`;
  const o = out(state.active, s.id);
  $("preds").innerHTML = `<span>label <b class="ok">${esc(s.label)}</b></span>`
    + `<span>pred <b class="${o?.correct ? "ok" : "no"}">${esc(o?.pred ?? "—")}</b> ${o ? (o.correct ? "✓" : "✗") : ""}</span>`;
  const strata = Object.entries(s.strata ?? {}).map(([k, v]) => `${k}=${v}`).join(" · ");
  $("meta").textContent = `${s.id} · ${s.source} · subject ${s.subject} · session ${s.session} · ${wordCount(s.text)} words${strata ? ` · ${strata}` : ""}`;
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
  const man = state.man;
  $("sub").textContent = `${man.name} · ${state.samples.length} samples · seed ${man.seed} · ${man.models.length} models`;
  signals.setSensors(man.sensors);
  buildFilters(refilter);
  restore();
  window.addEventListener("hashchange", restore);
  document.addEventListener("keydown", (e) => {
    const act = KEYS[e.key];
    if (!act || TYPING.has(e.target.tagName) || e.target.tagName === "AUDIO") return;
    e.preventDefault();
    act();
  });
  let pending = 0;
  new ResizeObserver(() => {
    cancelAnimationFrame(pending);
    pending = requestAnimationFrame(() => { signals.draw(); panel.draw(); });
  }).observe(document.querySelector("main"));
}

main().catch((e) => status(`load error: ${e.message} — serve with \`python -m megspeech.evaluation.viz serve\``));
