import { h, store } from "./util.js";

const MIN_PX = 32;

export class Split {
  constructor(container, panes, shares, key) {
    this.panes = panes;
    this.key = key;
    const saved = store.get(key, shares);
    this.shares = saved.length === panes.length ? saved : shares;
    panes.slice(1).forEach((pane, i) => {
      const bar = h("div", { className: "gutter", title: "drag to resize" });
      bar.onpointerdown = (e) => this.drag(e, i);
      container.insertBefore(bar, pane);
    });
    this.apply();
  }

  apply() {
    this.panes.forEach((p, i) => { p.style.flexGrow = this.shares[i]; });
  }

  drag(e, i) {
    e.preventDefault();
    const [a, b] = [this.panes[i], this.panes[i + 1]];
    const [ha, hb] = [a.getBoundingClientRect().height, b.getBoundingClientRect().height];
    const total = this.shares[i] + this.shares[i + 1], y0 = e.clientY;
    const move = (ev) => {
      const ya = Math.min(ha + hb - MIN_PX, Math.max(MIN_PX, ha + ev.clientY - y0));
      this.shares[i] = (total * ya) / (ha + hb);
      this.shares[i + 1] = total - this.shares[i];
      this.apply();
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      store.set(this.key, this.shares);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }
}
