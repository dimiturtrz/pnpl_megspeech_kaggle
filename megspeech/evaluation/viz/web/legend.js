import { state } from "./state.js";
import { $, h } from "./util.js";

const SOURCES = { Sherlock: [212, 85, 62], TheMoth: [28, 85, 58], MOCHATIMIT: [275, 70, 70] };
export const sourceColor = (src, a = 1) => {
  const [hue, s, l] = SOURCES[src] ?? [0, 0, 60];
  return `hsla(${hue},${s}%,${l}%,${a})`;
};

export class Legend {
  constructor(onChange) {
    this.box = $("legend");
    this.hover = null;
    this.onChange = onChange;
  }

  spot(fallback = null) { return this.hover ?? state.focus ?? fallback; }

  categories(coloring, pts, spot) {
    const counts = new Map();
    for (const p of pts) if (p.on) counts.set(coloring.key(p), (counts.get(coloring.key(p)) ?? 0) + 1);
    const chips = [...counts].sort((a, b) => b[1] - a[1]).map(([key, n]) => {
      const sw = h("i");
      sw.style.background = coloring.color(key);
      const chip = h("button", { className: `chip${key === spot ? " on" : ""}`, title: `${key}: ${n} shown · click to pin` },
        sw, h("span", { textContent: key }), h("span", { className: "n", textContent: n }));
      chip.dataset.key = key;
      chip.onmouseenter = () => { this.hover = key; this.onChange(); };
      chip.onmouseleave = () => { this.hover = null; this.onChange(); };
      chip.onclick = () => { state.focus = state.focus === key ? null : key; this.onChange(); };
      return chip;
    });
    this.box.replaceChildren(...chips);
  }

  gradient(coloring) {
    const [lo, hi, note] = coloring.scale, bar = h("div", { className: "cgrad" });
    bar.style.background = `linear-gradient(90deg,${[0, 0.5, 1].map((t) => coloring.color(t)).join(",")})`;
    this.box.replaceChildren(h("div", { className: "cbar" },
      h("span", { textContent: lo }), bar, h("span", { textContent: hi }), h("span", { className: "mut", textContent: note })));
  }

  render(coloring, pts, spot) {
    this.box.classList.toggle("cats", !!coloring.key);
    if (coloring.key) this.categories(coloring, pts, spot);
    else this.gradient(coloring);
  }

  mark(spot, hot) {
    for (const c of this.box.querySelectorAll(".chip")) {
      c.classList.toggle("on", c.dataset.key === spot);
      c.classList.toggle("hot", c.dataset.key === hot);
    }
  }
}
