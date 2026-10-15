const HEAT = [[70, 95, 150], [55, 165, 170], [235, 200, 60], [255, 75, 40]], FAINT = 0.35;

export function heat(q, alpha = FAINT + (1 - FAINT) * q) {
  const x = q * (HEAT.length - 1), i = Math.min(HEAT.length - 2, Math.floor(x)), f = x - i;
  return `rgba(${HEAT[i].map((c, k) => Math.round(c + (HEAT[i + 1][k] - c) * f)).join(",")},${alpha})`;
}

export function percentiles(weights) {
  const q = new Float32Array(weights.length);
  [...weights.keys()].sort((a, b) => weights[a] - weights[b]).forEach((c, r) => { q[c] = r / Math.max(1, weights.length - 1); });
  return q;
}
