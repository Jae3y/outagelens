// SVG renderers for the OutageLens control room. Every function returns markup as a string.

export const C = {
  ink: '#121619', body2: '#3E464C', muted: '#59626A', edge: '#A3ABA7', silent: '#8C9590', ctrl: '#C9CFCB',
  soft: '#ECEEEC', fault: '#C83A1C', fault2: '#E2917B', blue: '#1E4FD0', green: '#12704A', amber: '#D98A00',
  amberText: '#8F5600', redText: '#A2321A', cut: '#F6DCD4', baseline: '#B4BBB7',
};

export const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const r1 = (v) => Math.round(v * 10) / 10;

function mix(a, b, t) {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16));
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16));
  return '#' + pa.map((v, i) => Math.round(v + (pb[i] - v) * t).toString(16).padStart(2, '0')).join('');
}

/** Grey → salmon → vermilion as fault probability rises (matches the map legend). */
export function heat(p) {
  const t = Math.sqrt(Math.max(0, Math.min(1, p)));
  return t < 0.5 ? mix(C.edge, C.fault2, t * 2) : mix(C.fault2, C.fault, (t - 0.5) * 2);
}

// Hand-placed label offsets for the dense IEEE 34-bus layout (from the design), [dx, dy] from the node.
const LABEL_OFFSETS = {
  ieee34: {
    800: [8, -14], 802: [8, -14], 810: [14, -5], 814: [-4, 27], 850: [-4, 27], 816: [-4, 23], 818: [8, 4], 820: [8, 4],
    822: [8, 4], 824: [-4, 23], 826: [14, 4], 828: [-4, -10], 830: [-4, -10], 854: [10, -3], 856: [14, 4], 852: [10, 4],
    832: [10, -10], 888: [10, 4], 890: [10, 4], 858: [-4, 21], 864: [10, 4], 834: [-4, -10], 842: [-4, -10],
    844: [-4, -12], 846: [-4, -12], 848: [-4, -10], 860: [10, 4], 836: [-4, 21], 840: [10, 4], 862: [-24, -4], 838: [16, 4],
  },
};

function overlaps(a, b) {
  return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
}

/** Feeder map with the posterior heat, call-outs for the leading sections and (optionally) the truth. */
export function mapSvg(data, probs, { showTruth, feederKey }) {
  const W = 870, pad = 40;
  const xs = data.nodes.map((n) => n.x), ys = data.nodes.map((n) => n.y);
  const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
  const scale = Math.min((W - 2 * pad) / Math.max(xmax - xmin, 1e-6), 370 / Math.max(ymax - ymin, 1e-6));
  const ox = (W - (xmax - xmin) * scale) / 2;
  const H = Math.round((ymax - ymin) * scale + 2 * pad);
  const pos = {};
  data.nodes.forEach((n) => { pos[n.bus] = [r1(ox + (n.x - xmin) * scale), r1(pad + (ymax - n.y) * scale)]; });
  const nodeBy = Object.fromEntries(data.nodes.map((n) => [n.bus, n]));
  const offsets = LABEL_OFFSETS[feederKey] || {};

  const ranked = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
  const out = [];

  // Base network
  data.edges.forEach((e) => {
    if (probs[e.idx] >= 0.01) return;
    const [x1, y1] = pos[e.u], [x2, y2] = pos[e.v];
    out.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${C.edge}" stroke-width="${e.trunk ? 3 : 2}" stroke-linecap="round"><title>${esc(e.display)} · ${(probs[e.idx] * 100).toFixed(1)}%</title></line>`);
  });
  // De-energized zone (truth only): white dashes over the grey lines
  if (showTruth) {
    data.edges.forEach((e) => {
      if (nodeBy[e.u].energized || probs[e.idx] >= 0.01) return;
      const [x1, y1] = pos[e.u], [x2, y2] = pos[e.v];
      out.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#FFFFFF" stroke-width="${e.trunk ? 4 : 3}" stroke-dasharray="5 5"></line>`);
    });
  }
  // Probable sections, weakest first so the leader sits on top
  const hot = data.edges.filter((e) => probs[e.idx] >= 0.01).sort((a, b) => probs[a.idx] - probs[b.idx]);
  hot.forEach((e) => {
    const p = probs[e.idx];
    const [x1, y1] = pos[e.u], [x2, y2] = pos[e.v];
    if (e.idx === ranked[0][1] && p >= 0.3) {
      out.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${C.fault}" stroke-opacity="0.16" stroke-width="24" stroke-linecap="round"></line>`);
    }
    out.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${heat(p)}" stroke-width="${r1(3 + 6 * Math.sqrt(p))}" stroke-linecap="round"><title>${esc(e.display)} · ${(p * 100).toFixed(1)}%</title></line>`);
  });

  // Obstacles for call-out placement: nodes and their labels
  const obstacles = [];
  const labels = [];
  data.nodes.forEach((n) => {
    const [x, y] = pos[n.bus];
    obstacles.push({ x: x - 12, y: y - 12, w: 24, h: 24 });
    const [dx, dy] = offsets[n.bus] || (n.sensor ? [15, -13] : [10, -12]);
    labels.push([n.bus, x + dx, y + dy]);
    obstacles.push({ x: x + dx - 2, y: y + dy - 10, w: n.bus.length * 6.6 + 4, h: 13 });
  });

  // Call-outs for the top three (always #1, others when ≥ 1 %)
  const callouts = [];
  ranked.slice(0, 3).forEach(([p, idx], rank) => {
    if (rank > 0 && p < 0.01) return;
    const e = data.edges[idx];
    const [x1, y1] = pos[e.u], [x2, y2] = pos[e.v];
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
    const kind = e.kind === 'line' ? '' : ({ regulator: 'reg ', transformer: 'xfmr ', switch: 'sw ' }[e.kind]);
    const text = `#${rank + 1}  ${kind}${e.u}→${e.v}  ${(p * 100).toFixed(1)}%`;
    const w = Math.round(text.length * 7.55 + 22), h = 26;
    const tries = [
      [mx - 14 - w, my - h / 2], [mx + 14, my - h / 2], [mx - 14 - w, my - h / 2 - 34], [mx + 14, my - h / 2 - 34],
      [mx - 14 - w, my - h / 2 + 34], [mx + 14, my - h / 2 + 34], [mx - w / 2, my + 16], [mx - w / 2, my - 16 - h],
      [mx - 14 - w, my - h / 2 - 68], [mx + 14, my - h / 2 + 68],
    ];
    let best = null;
    for (const [bx, by] of tries) {
      const box = { x: bx, y: by, w, h };
      if (bx < 2 || by < 2 || bx + w > W - 2 || by + h > H - 2) continue;
      if ([...obstacles, ...callouts].some((o) => overlaps(box, o))) continue;
      best = box; break;
    }
    if (!best) best = { x: Math.max(2, Math.min(W - w - 2, mx + 14)), y: Math.max(2, my - h / 2), w, h };
    callouts.push(best);
    const lx = best.x + best.w < mx ? best.x + best.w : best.x > mx ? best.x : mx;
    const ly = best.y + h / 2;
    out.push(`<line x1="${r1(lx)}" y1="${r1(ly)}" x2="${r1(mx + (lx < mx ? -4 : lx > mx ? 4 : 0))}" y2="${r1(my)}" stroke="${C.ink}" stroke-width="1.2"></line>`);
    const lead = rank === 0;
    out.push(`<rect x="${r1(best.x)}" y="${r1(best.y)}" width="${w}" height="${h}" rx="4" fill="${lead ? C.ink : '#FFFFFF'}" stroke="${C.ink}" stroke-width="1.2"></rect>`);
    out.push(`<text x="${r1(best.x + 10)}" y="${r1(best.y + 17.3)}" font-size="12.5" font-weight="500" fill="${lead ? '#FFFFFF' : C.ink}" xml:space="preserve">${esc(text)}</text>`);
  });

  // Nodes, sensor rings and labels
  data.nodes.forEach((n) => {
    const [x, y] = pos[n.bus];
    const state = n.source ? 'substation' : n.meter === 1 ? 'energized, reporting' : n.meter === 0 ? 'last-gasp outage ping' : 'meter silent';
    let tip = `Bus ${n.bus}: ${state}`;
    if (n.sensor && n.v_pre_obs !== null) tip += ` · sensor ${n.v_pre_obs.toFixed(4)} → ${n.v_post_obs.toFixed(4)} p.u.`;
    out.push(`<g><title>${esc(tip)}</title>`);
    if (n.sensor) out.push(`<circle cx="${x}" cy="${y}" r="11" fill="none" stroke="${C.blue}" stroke-width="2"></circle>`);
    if (n.source) out.push(`<rect x="${x - 7}" y="${y - 7}" width="14" height="14" fill="${C.ink}"></rect>`);
    else if (n.meter === 1) out.push(`<circle cx="${x}" cy="${y}" r="5.5" fill="${C.green}"></circle>`);
    else if (n.meter === 0) out.push(`<polygon points="${x},${r1(y - 7)} ${r1(x + 6.5)},${y + 5} ${r1(x - 6.5)},${y + 5}" fill="${C.amber}"></polygon>`);
    else out.push(`<circle cx="${x}" cy="${y}" r="5" fill="#FFFFFF" stroke="${C.silent}" stroke-width="1.6"></circle>`);
    out.push('</g>');
  });
  labels.forEach(([bus, lx, ly]) => out.push(`<text x="${r1(lx)}" y="${r1(ly)}" font-size="10.5" fill="${C.muted}">${esc(bus)}</text>`));

  if (showTruth) {
    const e = data.edges[data.truth.edge];
    const [x1, y1] = pos[e.u], [x2, y2] = pos[e.v];
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
    const vertical = Math.abs(x2 - x1) < Math.abs(y2 - y1);
    const cx = vertical ? mx + 22 : mx, cy = vertical ? my : my + 22;
    out.push(`<g><line x1="${cx - 6}" y1="${cy - 6}" x2="${cx + 6}" y2="${cy + 6}" stroke="${C.ink}" stroke-width="2.6"></line><line x1="${cx + 6}" y1="${cy - 6}" x2="${cx - 6}" y2="${cy + 6}" stroke="${C.ink}" stroke-width="2.6"></line><text x="${cx + 12}" y="${cy + 4}" font-size="11.5" font-weight="500" fill="${C.ink}">actual fault</text></g>`);
  }

  const lead = ranked[0];
  const label = `Feeder map. ${data.edges[lead[1]].display} carries ${(lead[0] * 100).toFixed(1)} percent fault probability.`;
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(label)}">${out.join('')}</svg>`;
}

// ---------------------------------------------------------------- line charts
function axes({ x0, x1, y0, y1, yTicks, yFmt, xTicks, xFmt, xTitle, yAt, xAt }) {
  const o = [`<line x1="${x0}" y1="${y0}" x2="${x1}" y2="${y0}" stroke="${C.ctrl}"></line>`];
  yTicks.forEach((t, i) => {
    const y = yAt(t);
    if (i > 0) o.push(`<line x1="${x0}" y1="${y}" x2="${x1}" y2="${y}" stroke="${C.soft}"></line>`);
    o.push(`<text x="${x0 - 10}" y="${y + 4}" font-size="11" fill="${C.muted}" text-anchor="end">${yFmt(t, i)}</text>`);
  });
  xTicks.forEach((t) => o.push(`<text x="${xAt(t)}" y="${y0 + 20}" font-size="11" fill="${C.muted}" text-anchor="middle">${xFmt(t)}</text>`));
  if (xTitle) o.push(`<text x="${(x0 + x1) / 2}" y="${y0 + 32}" font-size="10.5" fill="${C.muted}" text-anchor="middle">${esc(xTitle)}</text>`);
  return o.join('');
}

const poly = (pts, attrs) => `<polyline points="${pts.map(([x, y]) => `${r1(x)},${r1(y)}`).join(' ')}" fill="none" stroke-linejoin="round" ${attrs}></polyline>`;

export const METHOD_STYLE = {
  'OutageLens Ensemble (Bayes + GNN)': { label: 'OutageLens ensemble', stroke: C.ink, width: 3.5, dash: '' },
  'OutageLens Bayesian + Physics': { label: 'Bayes + physics', stroke: C.blue, width: 2.5, dash: '' },
  'GNN only': { label: 'GNN only', stroke: C.muted, width: 2, dash: '4 3' },
  'Naive LCA Baseline': { label: 'Naive baseline', stroke: C.baseline, width: 3, dash: '' },
};

export function legendLine({ stroke, width, dash }) {
  return `<svg width="22" height="8" aria-hidden="true"><line x1="0" y1="4" x2="22" y2="4" stroke="${stroke}" stroke-width="${width}" ${dash ? `stroke-dasharray="${dash}"` : ''}></line></svg>`;
}

export function accuracyChart(bench) {
  const x0 = 64, x1 = 694, y0 = 286, top = 46;
  const xAt = (r) => x0 + (x1 - x0) * r;
  const yAt = (a) => y0 - (y0 - top) * a / 100;
  const o = [axes({ x0, x1, y0, y1: top, yTicks: [0, 25, 50, 75, 100], yFmt: (t) => `${t}%`, xTicks: [...new Set(bench.summary.map((s) => s.rate))], xFmt: (t) => `${Math.round(t * 100)}%`, xTitle: 'smart-meter reports missing', yAt, xAt })];
  const order = [...bench.methods].reverse();
  const ens = bench.methods[0];
  const pts = (m, key) => bench.summary.filter((s) => s.method === m).map((s) => [xAt(s.rate), yAt(s[key])]);
  o.push(poly(pts(ens, 'top3'), `stroke="${C.ink}" stroke-width="1.2" stroke-dasharray="2 3"`));
  order.forEach((m) => {
    const st = METHOD_STYLE[m];
    o.push(poly(pts(m, 'top1'), `stroke="${st.stroke}" stroke-width="${st.width}" ${st.dash ? `stroke-dasharray="${st.dash}"` : ''}`));
  });
  [[ens, C.ink], [bench.baseline_method, C.baseline]].forEach(([m, col]) => {
    const last = bench.summary.filter((s) => s.method === m).at(-1);
    if (!last) return;
    o.push(`<circle cx="${xAt(last.rate)}" cy="${yAt(last.top1)}" r="4.5" fill="${col}"></circle>`);
    o.push(`<text x="${xAt(last.rate) + 10}" y="${yAt(last.top1) + 4}" font-size="12" font-weight="500" fill="${m === ens ? C.ink : C.muted}">${last.top1.toFixed(1).replace('.0', '')}%</text>`);
  });
  const e95 = bench.summary.filter((s) => s.method === ens).at(-1);
  return `<svg viewBox="0 0 720 320" role="img" aria-label="Top-1 accuracy against missing meter rate. OutageLens reaches ${e95 ? e95.top1.toFixed(1) : ''} percent at the highest missing rate.">${o.join('')}</svg>`;
}

export function patrolChart(bench) {
  const x0 = 64, x1 = 500, y0 = 286, top = 46;  // narrower canvas: this card is ~2/5 of the row
  const ens = bench.methods[0], base = bench.baseline_method;
  const conv = bench.headline.find((h) => h.method === ens)?.conv_km ?? 0;
  const vmax = Math.max(conv, ...bench.summary.filter((s) => s.method === base).map((s) => s.km));
  const step = vmax > 40 ? 10 : vmax > 8 ? 2 : vmax > 3 ? 1 : 0.5;
  const ymaxv = Math.ceil(vmax * 1.12 / step) * step;
  const ticks = []; for (let t = 0; t <= ymaxv + 1e-9; t += step) ticks.push(Math.round(t * 100) / 100);
  const xAt = (r) => x0 + (x1 - x0) * r;
  const yAt = (k) => y0 - (y0 - top) * k / ymaxv;
  const o = [axes({ x0, x1, y0, y1: top, yTicks: ticks, yFmt: (t, i) => (i === ticks.length - 1 ? `${t} km` : `${t}`), xTicks: [0.1, 0.4, 0.7, 0.95], xFmt: (t) => `${Math.round(t * 100)}%`, xTitle: 'smart-meter reports missing', yAt, xAt })];
  o.push(`<line x1="${x0}" y1="${r1(yAt(conv))}" x2="${x1}" y2="${r1(yAt(conv))}" stroke="${C.amberText}" stroke-width="1.5" stroke-dasharray="6 4"></line>`);
  o.push(`<text x="${x0 + 8}" y="${r1(yAt(conv) - 8)}" font-size="11.5" fill="${C.amberText}">conventional ${conv.toFixed(1)} km</text>`);
  const pts = (m) => bench.summary.filter((s) => s.method === m).map((s) => [xAt(s.rate), yAt(s.km)]);
  o.push(poly(pts(base), `stroke="${C.baseline}" stroke-width="3"`));
  o.push(poly(pts(ens), `stroke="${C.ink}" stroke-width="3.5"`));
  [[ens, C.ink], [base, C.baseline]].forEach(([m, col]) => {
    const last = bench.summary.filter((s) => s.method === m).at(-1);
    if (!last) return;
    const ly = yAt(last.km);
    o.push(`<circle cx="${xAt(last.rate)}" cy="${r1(ly)}" r="4.5" fill="${col}"></circle>`);
    o.push(`<text x="${xAt(last.rate) + 10}" y="${r1(Math.max(ly + 4, top + 14))}" font-size="12" font-weight="500" fill="${m === ens ? C.ink : C.muted}">${last.km.toFixed(1)}</text>`);
  });
  return `<svg viewBox="0 0 540 320" role="img" aria-label="Patrol distance against missing meter rate, with the conventional patrol at ${conv.toFixed(1)} km.">${o.join('')}</svg>`;
}

export function calibrationChart(cal) {
  const x0 = 48, x1 = 308, y0 = 282, top = 22;
  const X = (c) => x0 + (x1 - x0) * c, Y = (a) => y0 - (y0 - top) * a;
  const o = [
    `<line x1="${x0}" y1="${y0}" x2="${x1}" y2="${y0}" stroke="${C.ctrl}"></line>`,
    `<line x1="${x0}" y1="${top}" x2="${x0}" y2="${y0}" stroke="${C.ctrl}"></line>`,
    `<line x1="${x0}" y1="${y0}" x2="${x1}" y2="${top}" stroke="${C.baseline}" stroke-dasharray="4 4"></line>`,
    `<text x="40" y="${y0 + 4}" font-size="10.5" fill="${C.muted}" text-anchor="end">0</text>`,
    `<text x="40" y="${top + 4}" font-size="10.5" fill="${C.muted}" text-anchor="end">100</text>`,
    `<text x="${x1}" y="300" font-size="10.5" fill="${C.muted}" text-anchor="end">confidence 100%</text>`,
    `<text x="${x0 + 6}" y="${top + 14}" font-size="10.5" fill="${C.muted}">hit rate</text>`,
  ];
  o.push(poly(cal.points.map((p) => [X(p.conf), Y(p.acc)]), `stroke="${C.ink}" stroke-width="2"`));
  cal.points.forEach((p) => o.push(`<circle cx="${r1(X(p.conf))}" cy="${r1(Y(p.acc))}" r="${r1(3 + Math.sqrt(p.n) * 0.9)}" fill="${C.ink}" fill-opacity="0.88"><title>said ${(p.conf * 100).toFixed(0)}%, right ${(p.acc * 100).toFixed(0)}% (n=${p.n})</title></circle>`));
  return `<svg viewBox="0 0 330 310" role="img" aria-label="Calibration plot, expected calibration error ${(cal.ece * 100).toFixed(1)} points.">${o.join('')}</svg>`;
}

// ------------------------------------------------------------ voltage profile
export function profileChart(data, strongest, showTruth = true) {
  const x0 = 64, x1 = 694, y0 = 286, top = 40;
  const nodes = data.nodes;
  const byBus = Object.fromEntries(nodes.map((n) => [n.bus, n]));
  const kmMax = Math.max(...nodes.map((n) => n.km));
  const step = kmMax > 30 ? 10 : kmMax > 6 ? 2 : 0.5;
  const xmax = Math.ceil(kmMax / step) * step || 1;
  const live = nodes.filter((n) => n.energized);
  const vals = [...nodes.map((n) => n.v_pre), ...live.map((n) => n.v_post)];
  const vmin = Math.min(0.94, Math.floor((Math.min(...vals) - 0.01) * 50) / 50);
  const vmax = Math.max(1.10, Math.ceil((Math.max(...vals) + 0.01) * 50) / 50);
  const X = (k) => x0 + (x1 - x0) * k / xmax, Y = (v) => y0 - (y0 - top) * (v - vmin) / (vmax - vmin);
  const vt = []; for (let v = Math.ceil(vmin * 20) / 20; v <= vmax + 1e-9; v += 0.05) vt.push(Math.round(v * 100) / 100);
  const xt = []; for (let k = 0; k <= xmax + 1e-9; k += step) xt.push(Math.round(k * 10) / 10);
  const o = [];

  const dead = nodes.filter((n) => !n.energized);
  if (showTruth && dead.length) {
    const a = Math.min(...dead.map((n) => n.km)), b = Math.max(...dead.map((n) => n.km));
    const clean = !live.some((n) => n.km > a + 1e-6 && n.km < b - 1e-6);
    if (clean) {
      const bx = X(a) - 5, bw = Math.max(X(b) - X(a) + 10, 14);
      o.push(`<rect x="${r1(bx)}" y="${top}" width="${r1(bw)}" height="${y0 - top}" fill="${C.cut}"></rect>`);
      o.push(`<text x="${r1(bx + bw / 2)}" y="${top + 18}" font-size="11" fill="${C.redText}" text-anchor="middle">${dead.length} bus${dead.length > 1 ? 'es' : ''}</text>`);
      o.push(`<text x="${r1(bx + bw / 2)}" y="${top + 32}" font-size="11" fill="${C.redText}" text-anchor="middle">cut off</text>`);
    } else {
      dead.forEach((n) => o.push(`<text x="${r1(X(n.km))}" y="${y0 - 6}" font-size="12" fill="${C.redText}" text-anchor="middle">×<title>Bus ${esc(n.bus)} cut off</title></text>`));
    }
  }
  o.push(axes({ x0, x1, y0, y1: top, yTicks: vt, yFmt: (t) => t.toFixed(2), xTicks: xt, xFmt: (t, i) => `${t}`, xTitle: 'electrical distance from substation (km)', yAt: Y, xAt: X }));

  data.edges.forEach((e) => {
    const a = byBus[e.u], b = byBus[e.v];
    o.push(`<line x1="${r1(X(a.km))}" y1="${r1(Y(a.v_pre))}" x2="${r1(X(b.km))}" y2="${r1(Y(b.v_pre))}" stroke="${C.edge}" stroke-width="2"></line>`);
  });
  data.edges.forEach((e) => {
    const a = byBus[e.u], b = byBus[e.v];
    if (showTruth && a.energized && b.energized) o.push(`<line x1="${r1(X(a.km))}" y1="${r1(Y(a.v_post))}" x2="${r1(X(b.km))}" y2="${r1(Y(b.v_post))}" stroke="${C.blue}" stroke-width="2.6"></line>`);
    if (e.kind === 'regulator' && Math.abs(b.v_pre - a.v_pre) > 0.02) {
      o.push(`<text x="${r1(X(a.km) + 6)}" y="${r1(Math.max(Y(a.v_pre), Y(a.v_post)) + 16)}" font-size="10.5" fill="${C.muted}">regulator ${esc(e.u)}→${esc(e.v)} boost</text>`);
    }
  });
  nodes.filter((n) => n.sensor && !n.source && n.v_pre_obs !== null).forEach((n) => {
    o.push(`<circle cx="${r1(X(n.km))}" cy="${r1(Y(n.v_pre_obs))}" r="5" fill="#FFFFFF" stroke="${C.blue}" stroke-width="2"><title>Bus ${esc(n.bus)} before: ${n.v_pre_obs.toFixed(4)} p.u.</title></circle>`);
    if (n.v_post_obs > 0.3) o.push(`<circle cx="${r1(X(n.km))}" cy="${r1(Y(n.v_post_obs))}" r="5" fill="${C.blue}"><title>Bus ${esc(n.bus)} after: ${n.v_post_obs.toFixed(4)} p.u.</title></circle>`);
  });
  if (strongest) {
    const n = byBus[strongest.bus];
    const x = X(n.km), ya = Y(n.v_pre_obs), yb = Y(n.v_post_obs);
    o.push(`<line x1="${r1(x)}" y1="${r1(ya - 6)}" x2="${r1(x)}" y2="${r1(yb + 8)}" stroke="${C.blue}" stroke-width="1.2" stroke-dasharray="3 3"></line>`);
    // Anchor the label to the left of the point so it never runs into the cut-off band on the right.
    o.push(`<text x="${r1(Math.max(x0 + 150, x - 10))}" y="${r1(Math.max(top + 12, yb - 10))}" font-size="11.5" font-weight="500" fill="${C.blue}" text-anchor="end">${esc(n.bus)}: ${strongest.dv >= 0 ? '+' : ''}${strongest.dv.toFixed(3)} p.u.</text>`);
  }
  return `<svg viewBox="0 0 720 330" role="img" aria-label="Voltage profile against distance before and after the fault.">${o.join('')}</svg>`;
}
