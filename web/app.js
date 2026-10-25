// OutageLens control room: state, routing, API calls and view rendering.
import { C, METHOD_STYLE, accuracyChart, calibrationChart, esc, legendLine, mapSvg, patrolChart, profileChart } from './charts.js';

const state = {
  view: 'control',
  feeders: [],
  feeder: 'ieee34',
  perFeeder: {},           // remembers fault + seed per feeder
  dropout: 0.65, coverage: 0.25, glitch: 0.02,
  model: 'ensemble', placement: 'optimised', blind: false,
  data: null, bench: null, benchKey: null, benchRun: null, benchBusy: false,
  dispatch: null, dispatchKey: null, dispatchBusy: false, crewSeed: 1, study: {},
  audit: null, verify: null,
};
const MODEL_LABEL = { ensemble: 'Ensemble', bayes: 'Bayes', gnn: 'GNN' };
const TAG_COLOR = { PHYSICS: C.blue, SCADA: C.ink, METERS: C.green, TOPOLOGY: C.green, OUTLIER: C.amberText, CONFLICT: C.amberText };
const $ = (sel) => document.querySelector(sel);
const pct = (p, d = 1) => (p < 0.001 && p > 0 ? '<0.1%' : `${(p * 100).toFixed(d)}%`);
const num = (v, d = 0) => v.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });

// ------------------------------------------------------------------ API
let reqId = 0;
function params() {
  const f = state.perFeeder[state.feeder] || {};
  const p = { feeder: state.feeder, dropout: state.dropout, coverage: state.coverage, glitch: state.glitch, placement: state.placement };
  if (f.fault !== undefined) p.fault = f.fault;
  if (f.seed !== undefined) p.seed = f.seed;
  return new URLSearchParams(p);
}

async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 200)}`);
  return r.json();
}

function toast(msg) {
  const t = $('#toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toast.h);
  toast.h = setTimeout(() => t.classList.remove('show'), 4200);
}

async function loadScenario() {
  const id = ++reqId;
  document.querySelector('#results')?.classList.add('busy');
  try {
    const data = await getJSON(`/api/scenario?${params()}`);
    if (id !== reqId) return;
    state.data = data;
    state.perFeeder[state.feeder] = { fault: data.scenario.fault, seed: data.scenario.seed };
    renderEngine();
    syncControls();
    render();
  } catch (err) {
    if (id === reqId) toast(`Could not load the scenario: ${err.message}`);
  } finally {
    if (id === reqId) document.querySelector('#results')?.classList.remove('busy');
  }
}

let debounce;
const reload = (ms = 140) => { clearTimeout(debounce); debounce = setTimeout(loadScenario, ms); };

async function loadBench(run = false) {
  const key = `${state.feeder}|${state.coverage}|${state.glitch}`;
  if (!run && state.benchKey === key && state.bench) return;
  state.benchBusy = true;
  if (state.view === 'robustness') render();
  try {
    const q = new URLSearchParams({ feeder: state.feeder, coverage: state.coverage, glitch: state.glitch });
    const bench = await getJSON(run ? `/api/benchmark/run?${q}` : `/api/benchmark?${q}`, run ? { method: 'POST' } : undefined);
    state.bench = bench;
    state.benchKey = key;
    state.benchRun = run ? bench.seed : null;
  } catch (err) {
    toast(`Benchmark failed: ${err.message}`);
  } finally {
    state.benchBusy = false;
    if (state.view === 'robustness') render();
  }
}

// --------------------------------------------------------------- header
function renderEngine() {
  const e = state.data?.engine;
  if (!e) return;
  const dev = e.max_dev_pu;
  const label = e.opendss ? `OpenDSS · cross-checked ${dev.toExponential(1)} p.u.` : 'NumPy DistFlow engine';
  $('#engine').innerHTML = `<span class="dot ${e.opendss ? '' : 'dot-warn'}"></span>${esc(label)}`;
  $('#engine').title = e.detail;
}

function renderFeederPicker() {
  $('#feeder').innerHTML = state.feeders.map((f) =>
    `<option value="${f.key}" ${f.key === state.feeder ? 'selected' : ''}>${esc(f.short)} · ${esc(f.kv)} · ${f.km} km</option>`).join('');
}

// ------------------------------------------------------------- controls
function buildControls() {
  const box = $('#controls');
  box.hidden = state.view !== 'control';
  if (box.hidden) return;
  const feeder = state.feeders.find((f) => f.key === state.feeder);
  const fault = state.perFeeder[state.feeder]?.fault ?? feeder?.default_fault;
  box.innerHTML = `
  <div class="wrap controls-row">
    <label class="field w-fault">Faulted section
      <select id="c-fault">${(feeder?.sections || []).map((s) => `<option value="${s.idx}" ${s.idx === fault ? 'selected' : ''}>${esc(s.label)}</option>`).join('')}</select>
    </label>
    ${slider('dropout', 'Meters missing', 0, 95, 5, state.dropout, 'w-slider')}
    ${slider('coverage', 'Voltage sensors', 10, 100, 5, state.coverage, 'w-slider')}
    ${slider('glitch', 'Comms glitches', 0, 10, 1, state.glitch, 'w-slider-sm')}
    ${seg('model', 'Model', [['ensemble', 'Ensemble'], ['bayes', 'Bayes'], ['gnn', 'GNN']], state.model)}
    ${seg('placement', 'Sensor placement', [['optimised', 'Optimised'], ['random', 'Random']], state.placement)}
    <div class="actions">
      <button type="button" class="btn" id="c-blind" aria-pressed="${state.blind}">${state.blind ? 'Reveal ground truth' : 'Blind judge mode'}</button>
      <button type="button" class="btn" id="c-resample">Re-sample meters</button>
      <button type="button" class="btn btn-primary" id="c-mystery">Trigger mystery outage</button>
    </div>
  </div>`;
  syncControls();

  $('#c-fault').addEventListener('change', (ev) => {
    state.perFeeder[state.feeder] = { ...state.perFeeder[state.feeder], fault: Number(ev.target.value) };
    loadScenario();
  });
  ['dropout', 'coverage', 'glitch'].forEach((k) => {
    const el = $(`#c-${k}`);
    el.addEventListener('input', () => {
      state[k] = Number(el.value) / 100;
      $(`#v-${k}`).textContent = `${el.value}%`;
      reload();
    });
  });
  box.querySelectorAll('.seg button').forEach((b) => b.addEventListener('click', () => {
    const { group, value } = b.dataset;
    state[group] = value;
    box.querySelectorAll(`.seg button[data-group="${group}"]`).forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    if (group === 'placement') loadScenario(); else render();
  }));
  $('#c-blind').addEventListener('click', () => { state.blind = !state.blind; syncControls(); render(); });
  $('#c-resample').addEventListener('click', () => {
    state.perFeeder[state.feeder] = { ...state.perFeeder[state.feeder], seed: Math.floor(Math.random() * 1e6) + 1 };
    loadScenario();
  });
  $('#c-mystery').addEventListener('click', () => {
    const n = feeder.sections.length;
    state.perFeeder[state.feeder] = { fault: Math.floor(Math.random() * n), seed: Math.floor(Math.random() * 1e6) + 1 };
    state.blind = true;
    loadScenario();
    toast('Mystery outage injected. Ground truth is locked until you reveal it.');
  });
}

function slider(key, label, min, max, step, value, cls) {
  const v = Math.round(value * 100);
  return `<label class="field ${cls}"><span class="field-head">${label} <b id="v-${key}">${v}%</b></span>
    <input id="c-${key}" type="range" min="${min}" max="${max}" step="${step}" value="${v}"></label>`;
}

function seg(group, legend, options, current) {
  return `<fieldset class="seg-field"><legend>${legend}</legend><div class="seg">${options.map(([v, l]) =>
    `<button type="button" data-group="${group}" data-value="${v}" aria-pressed="${v === current}">${l}</button>`).join('')}</div></fieldset>`;
}

function syncControls() {
  const fault = state.perFeeder[state.feeder]?.fault;
  const sel = $('#c-fault');
  if (sel) {
    // Blind mode must not leak the answer through the fault picker.
    sel.querySelector('option[data-hidden]')?.remove();
    if (state.blind) {
      sel.insertAdjacentHTML('afterbegin', '<option data-hidden value="">Hidden until reveal</option>');
      sel.value = '';
    } else if (fault !== undefined) {
      sel.value = String(fault);
    }
    sel.disabled = state.blind;
  }
  const b = $('#c-blind');
  if (b) { b.textContent = state.blind ? 'Reveal ground truth' : 'Blind judge mode'; b.setAttribute('aria-pressed', String(state.blind)); }
}

// ----------------------------------------------------------------- views
function loadingView(msg) {
  return `<div class="loading" role="status"><div class="spin"></div>${esc(msg)}</div>`;
}

function controlView(d) {
  const m = d.models[state.model];
  const show = !state.blind;
  const t = d.truth;
  let tag = `<span class="tag tag-locked">GROUND TRUTH LOCKED</span>`;
  if (show) {
    tag = { exact: '<span class="tag tag-exact">EXACT MATCH</span>',
      top3: `<span class="tag tag-near">IN TOP-3 · ${m.hops} HOP${m.hops === 1 ? '' : 'S'} OFF</span>`,
      near: '<span class="tag tag-near">WITHIN 1 HOP</span>',
      miss: `<span class="tag tag-miss">MISS · ${m.hops} HOPS OFF</span>` }[m.status];
  }
  const b = d.baseline;
  let baseWhy = `Built from ${b.pings} outage ping${b.pings === 1 ? '' : 's'}, nothing else.`;
  let baseCls = '';
  if (show && b.correct) baseWhy = `Correct this time, from ${b.pings} outage ping${b.pings === 1 ? '' : 's'}.`;
  if (show && !b.correct) {
    baseCls = 'bad';
    baseWhy = `Wrong by ${b.hops} section${b.hops === 1 ? '' : 's'}.` +
      (b.upstream_glitches.length ? ` A glitched ping at ${b.upstream_glitches.slice(0, 2).join(', ')} dragged it upstream.` : '');
  }

  const conv = d.conventional;
  const km = show ? m.patrol_km : m.expected_km;
  const convKm = show ? conv.km : conv.expected_km;
  const mins = show ? m.patrol_min : m.expected_min;
  const convMin = show ? conv.min : conv.expected_min;
  const kmCut = convKm > 0 ? Math.round(100 * (1 - km / convKm)) : 0;
  const shares = m.shares;
  const w = d.fusion_weight;
  const note = state.model === 'ensemble'
    ? `Bayes ${pct(m.candidates[0].bayes)} · GNN ${pct(m.candidates[0].gnn)} on #1 · ${w >= 0.999 ? 'Bayes decisive, GNN not consulted' : `near tie, GNN weight ${(1 - w).toFixed(2)}`}`
    : state.model === 'bayes' ? 'Bayesian topology + physics model alone' : 'Graph neural network alone (data-driven)';

  return `
  <section class="verdict" aria-label="Verdict">
    <div class="verdict-main">
      <div class="kicker">Most likely faulted section</div>
      <div class="display">${esc(m.display)}</div>
      <div class="verdict-meta">
        <span><span class="conf">${pct(m.confidence)}</span> confidence</span>
        <span class="sub" style="font-size:15px">Top-3 holds ${pct(m.top3_mass, 0)} · entropy ${m.entropy_bits.toFixed(2)} bits</span>
        ${d.decision ? `<a class="chip" href="#/audit" title="Logged to the tamper-evident decision chain">Decision #${d.decision.seq} · sha256 ${d.decision.hash.slice(0, 10)}…</a>` : ''}
        ${tag}
      </div>
    </div>
    <div class="baseline-card">
      <div class="kicker">Naive baseline (outage pings only)</div>
      <div class="name">${esc(b.display)}</div>
      <div class="why ${baseCls}">${esc(baseWhy)}</div>
    </div>
  </section>

  <section class="kpis" aria-label="Operational impact">
    <div class="kpi"><div class="kicker">${show ? 'Patrol distance' : 'Expected patrol'}</div><div class="val">${km.toFixed(km < 10 ? 2 : 1)} km</div>
      <div class="sub">vs ${convKm.toFixed(1)} km ${show ? 'conventional' : 'average conventional'} · −${kmCut}%</div></div>
    <div class="kpi"><div class="kicker">Crew time</div><div class="val">${num(mins)} min</div>
      <div class="sub">vs ${num(convMin)} min · ${num(Math.max(convMin - mins, 0))} min saved</div></div>
    <div class="kpi"><div class="kicker">Interrupted</div><div class="val">${show ? `${num(t.interrupted_kw)} kW` : 'Locked'}</div>
      <div class="sub">${show ? `${t.dark_buses} buses dark · load at ${Math.round(d.scenario.load_level * 100)}% of plan` : 'revealed after the call'}</div></div>
    <div class="kpi"><div class="kicker">Evidence received</div><div class="val">${d.counts.reporting} / ${d.counts.meters}</div>
      <div class="sub">meters reporting · ${d.counts.pings} pings · ${d.counts.sensors} sensors</div></div>
  </section>

  <div class="row">
    <section class="card grow-main" aria-label="Feeder map">
      <div class="card-head"><h2 class="h2">${esc(d.feeder.short)} feeder · fault probability</h2>
        <span class="sub">Line weight and colour = ${MODEL_LABEL[state.model].toLowerCase()} P(fault)</span></div>
      <div class="map-box">${mapSvg(d, d.probs[state.model], { showTruth: show, feederKey: d.feeder.key })}</div>
      <div class="legend">
        <span><svg width="14" height="14" aria-hidden="true"><rect x="1" y="1" width="12" height="12" fill="${C.ink}"></rect></svg>Substation</span>
        <span><svg width="14" height="14" aria-hidden="true"><circle cx="7" cy="7" r="5.5" fill="${C.green}"></circle></svg>Energized, reporting</span>
        <span><svg width="14" height="14" aria-hidden="true"><polygon points="7,1 13,12 1,12" fill="${C.amber}"></polygon></svg>Last-gasp ping</span>
        <span><svg width="14" height="14" aria-hidden="true"><circle cx="7" cy="7" r="5" fill="#FFFFFF" stroke="${C.silent}" stroke-width="1.6"></circle></svg>Silent meter</span>
        <span><svg width="18" height="18" aria-hidden="true"><circle cx="9" cy="9" r="7.5" fill="none" stroke="${C.blue}" stroke-width="2"></circle></svg>Voltage sensor</span>
        ${show ? `<span><svg width="22" height="8" aria-hidden="true"><line x1="0" y1="4" x2="22" y2="4" stroke="${C.edge}" stroke-width="3" stroke-dasharray="5 5"></line></svg>De-energized</span>` : ''}
        <span class="scale">0%<i class="scale-bar"></i>100% P(fault)</span>
      </div>
    </section>

    <aside class="grow-side">
      <section class="card" aria-label="Ranked candidates">
        <h2 class="h2" style="margin-bottom:12px">Ranked sections</h2>
        ${m.candidates.map((c, i) => {
          const col = i === 0 ? C.fault : c.p >= 0.05 ? '#D0705A' : C.silent;
          return `<div class="rank-row"><span class="n">${c.rank}</span>
            <div style="min-width:0"><div class="sec">${esc(c.display)}${show && c.idx === t.edge ? ' <span style="font-size:12px;font-weight:500;color:var(--muted)">· actual fault</span>' : ''}</div><div class="meta">${esc(c.meta)}</div></div>
            <span class="pct" style="color:${col}">${pct(c.p)}</span>
            <span></span><div class="bar"><i style="width:${Math.max(c.p * 100, 0.2)}%;background:${col}"></i></div></div>`;
        }).join('')}
        <div class="note">${esc(note)}</div>
      </section>

      <section class="card" aria-label="Evidence">
        <h2 class="h2">Why ${esc(m.section)}</h2>
        <div class="attr" role="img" aria-label="Evidence share: voltage physics ${Math.round(shares.physics)} percent, meters ${Math.round(shares.meters)} percent, SCADA ${Math.round(shares.scada)} percent, prior ${Math.round(shares.prior)} percent">
          <i style="width:${shares.physics}%;background:${C.blue}"></i><i style="width:${shares.meters}%;background:${C.green}"></i><i style="width:${shares.scada}%;background:${C.ink}"></i><i style="width:${shares.prior}%;background:${C.edge}"></i>
        </div>
        <div class="attr-key"><span><b style="color:${C.blue}">${Math.round(shares.physics)}%</b> voltage physics</span><span><b style="color:${C.green}">${Math.round(shares.meters)}%</b> meters</span><span><b>${Math.round(shares.scada)}%</b> SCADA</span><span><b style="color:${C.muted}">${Math.round(shares.prior)}%</b> prior</span></div>
        ${m.evidence.map((ev) => `<div class="ev-row"><span class="t" style="color:${TAG_COLOR[ev.tag] || C.ink}">${ev.tag}</span><span class="x">${esc(ev.text)}</span></div>`).join('')}
      </section>
    </aside>
  </div>`;
}

function robustnessView() {
  const b = state.bench;
  const header = `
  <section class="row" style="align-items:flex-end;justify-content:space-between;gap:16px 40px">
    <div style="flex:1 1 560px;min-width:0">
      <div class="kicker">Robustness benchmark</div>
      <h1 class="h1">Does it hold up when meters go silent?</h1>
      <p class="lede">${b ? `${b.n_scenarios}` : '280'} simulated outages on the ${esc(state.data?.feeder.short || '')} feeder. Every one has a random faulted section, an unknown load level, a random sensor layout and ${Math.round(state.glitch * 100)}% comms glitches. A crew patrols sections in the order each method ranks them.</p>
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:10px">
      <button type="button" class="btn btn-primary" id="b-run" ${state.benchBusy ? 'disabled' : ''}>${state.benchBusy ? 'Running simulations…' : state.benchRun ? 'Run again with a new seed' : 'Run 154-scenario benchmark'}</button>
      <a class="btn" href="#table" id="b-table">Full table</a>
    </div>
  </section>`;
  if (!b) return header + loadingView('Running the Monte Carlo benchmark (first time only)…');

  const ens = b.methods[0];
  const H = Object.fromEntries(b.headline.map((h) => [h.method, h]));
  const e = H[ens], base = H[b.baseline_method];
  const lastE = b.summary.filter((s) => s.method === ens).at(-1);
  const lastB = b.summary.filter((s) => s.method === b.baseline_method).at(-1);
  const groups = (rows, keyField, variants, fmtLabel) => {
    const keys = [...new Set(rows.map((r) => r[keyField]))].sort((x, y) => x - y);
    return { keys, html: keys.map((k) => `<div class="bar-group">${variants.map(([name, color]) => {
      const r = rows.find((x) => x[keyField] === k && (x.variant || x.layout) === name);
      const v = r ? Math.round(r.top1) : 0;
      return `<div class="bar-col"><span>${v}%</span><i style="height:${v * 1.62}px;background:${color}" title="${esc(name)}: ${v}%"></i></div>`;
    }).join('')}</div>`).join(''), labels: keys.map(fmtLabel) };
  };
  const abl = groups(b.ablation, 'rate', [['Naive LCA Baseline', '#D5DAD6'], ['Smart meters only', C.silent], ['Meters + voltage sensors', C.blue], ['Meters + voltage + feeder-head SCADA', C.ink]], (k) => `${Math.round(k * 100)}% missing`);
  const pl = groups(b.placement, 'coverage', [['Random placement', C.baseline], ['Optimised placement', C.ink]], (k, i) => `${Math.round(k * 100)}%${i === 0 ? ' of buses' : ''}`);
  const top95 = b.calibration.points.filter((p) => p.conf >= 0.95);
  const n95 = top95.reduce((a, p) => a + p.n, 0);
  const acc95 = n95 ? top95.reduce((a, p) => a + p.acc * p.n, 0) / n95 : 0;

  return header + `
  ${state.benchRun ? `<div class="sub">Showing your fresh run · seed ${state.benchRun}</div>` : ''}
  <section class="kpis ${state.benchBusy ? 'busy' : ''}" aria-label="Headline results">
    <div class="kpi"><div class="kicker">Top-1 accuracy</div><div class="val">${e.top1.toFixed(1)}%</div><div class="sub">baseline ${base.top1.toFixed(1)}%</div></div>
    <div class="kpi"><div class="kicker">Top-3 accuracy</div><div class="val">${e.top3.toFixed(1)}%</div><div class="sub">baseline ${base.top3.toFixed(1)}%</div></div>
    <div class="kpi"><div class="kicker">${Math.round(lastE.rate * 100)}% meters silent</div><div class="val">${lastE.top1.toFixed(1)}%</div><div class="sub">Top-1 · baseline ${lastB.top1.toFixed(1)}%</div></div>
    <div class="kpi"><div class="kicker">Patrol per outage</div><div class="val">${e.km.toFixed(1)} km</div><div class="sub">baseline ${base.km.toFixed(1)} · conventional ${e.conv_km.toFixed(1)}</div></div>
    <div class="kpi"><div class="kicker">Crew time per outage</div><div class="val">${num(e.min)} min</div><div class="sub">baseline ${num(base.min)} · conventional ${num(e.conv_min)}</div></div>
  </section>

  <div class="row ${state.benchBusy ? 'busy' : ''}">
    <section class="card chart" style="flex:3 1 560px" aria-label="Accuracy versus missing meters">
      <h2 class="h2">Top-1 accuracy as meter data disappears</h2>
      <div class="chart-legend">${b.methods.map((mm) => `<span>${legendLine(METHOD_STYLE[mm])}${METHOD_STYLE[mm].label}</span>`).join('')}<span>${legendLine({ stroke: C.ink, width: 1.2, dash: '2 3' })}Ensemble Top-3</span></div>
      ${accuracyChart(b)}
    </section>
    <section class="card chart" style="flex:2 1 420px" aria-label="Patrol distance versus missing meters">
      <h2 class="h2">Line patrolled before the fault is found</h2>
      <div class="chart-legend"><span>${legendLine(METHOD_STYLE[ens])}OutageLens</span><span>${legendLine(METHOD_STYLE[b.baseline_method])}Naive baseline</span><span>${legendLine({ stroke: C.amberText, width: 1.5, dash: '6 4' })}Conventional patrol</span></div>
      ${patrolChart(b)}
    </section>
  </div>

  <div class="row ${state.benchBusy ? 'busy' : ''}">
    <section class="card" style="flex:1 1 420px" aria-label="Evidence ablation">
      <h2 class="h2">Which data stream carries the answer</h2>
      <p class="sub" style="margin:6px 0 14px">Top-1 accuracy as each evidence stream is added.</p>
      <div class="bars">${abl.html}</div>
      <div class="bar-labels">${abl.labels.map((l) => `<span>${l}</span>`).join('')}</div>
      <div class="chart-legend" style="margin-top:14px;font-size:12px"><span><i class="swatch" style="background:#D5DAD6"></i>Baseline</span><span><i class="swatch" style="background:${C.silent}"></i>Meters only</span><span><i class="swatch" style="background:${C.blue}"></i>+ voltage sensors</span><span><i class="swatch" style="background:${C.ink}"></i>+ feeder-head SCADA</span></div>
    </section>
    <section class="card" style="flex:1 1 340px" aria-label="Sensor placement">
      <h2 class="h2">Same budget, smarter placement</h2>
      <p class="sub" style="margin:6px 0 14px">Top-1 with 50–95% of meters missing.</p>
      <div class="bars">${pl.html}</div>
      <div class="bar-labels">${pl.labels.map((l) => `<span>${l}</span>`).join('')}</div>
      <div class="chart-legend" style="margin-top:14px;font-size:12px"><span><i class="swatch" style="background:${C.baseline}"></i>Random installs</span><span><i class="swatch" style="background:${C.ink}"></i>OutageLens placement</span></div>
    </section>
    <section class="card chart" style="flex:1 1 340px" aria-label="Calibration">
      <h2 class="h2">Can the confidence be trusted?</h2>
      <p class="sub" style="margin:6px 0 8px">Stated confidence vs hit rate · ECE ${(b.calibration.ece * 100).toFixed(1)} pts</p>
      ${calibrationChart(b.calibration)}
      <p class="sub" style="margin:6px 0 0;font-size:12px">Dot size = number of outages. ${n95} of ${b.n_scenarios} calls land at ≥95% confidence; ${Math.round(acc95 * 100)}% of them are right.</p>
    </section>
  </div>

  <section class="card" id="table" aria-label="Results by method">
    <h2 class="h2" style="margin-bottom:12px">Results by method · ${b.n_scenarios} outages</h2>
    <div class="table-box"><table class="data" style="min-width:720px">
      <thead><tr><th>Method</th><th class="num">Top-1</th><th class="num">Top-3</th><th class="num">Hop error</th><th class="num">Patrol km</th><th class="num">Crew min</th></tr></thead>
      <tbody>
        ${b.headline.map((h) => `<tr class="${h.method === ens ? 'strong' : ''}"><td>${esc(METHOD_STYLE[h.method]?.label || h.method)}</td><td class="num">${h.top1.toFixed(1)}%</td><td class="num">${h.top3.toFixed(1)}%</td><td class="num">${h.hop.toFixed(1)}</td><td class="num">${h.km.toFixed(1)}</td><td class="num">${num(h.min)}</td></tr>`).join('')}
        <tr style="color:var(--muted)"><td>Conventional patrol</td><td class="num">—</td><td class="num">—</td><td class="num">—</td><td class="num">${e.conv_km.toFixed(1)}</td><td class="num">${num(e.conv_min)}</td></tr>
      </tbody></table></div>
  </section>`;
}

function physicsView(d) {
  const m = d.models[state.model];
  const show = !state.blind;
  const sensors = d.nodes.filter((n) => n.sensor && !n.source && n.v_pre_obs !== null).sort((a, b) => a.km - b.km);
  const live = sensors.filter((n) => n.v_post_obs > 0.3);
  const strongestNode = live.length ? live.reduce((a, b) => ((b.v_post_obs - b.v_pre_obs) > (a.v_post_obs - a.v_pre_obs) ? b : a)) : null;
  const strongest = strongestNode ? { bus: strongestNode.bus, dv: strongestNode.v_post_obs - strongestNode.v_pre_obs } : null;
  const maxDv = Math.max(1e-4, ...live.map((n) => Math.abs(n.v_post_obs - n.v_pre_obs)), ...live.map((n) => Math.abs(m.sensor_pred[n.bus] || 0)));
  const sc = 200 / maxDv;
  const cols = sensors.map((n) => {
    if (n.v_post_obs <= 0.3) {
      return { bus: n.bus, note: 'cut off', label: '0 p.u.', color: C.redText, h: 6, fill: C.cut, pred: null };
    }
    const dv = n.v_post_obs - n.v_pre_obs, pr = m.sensor_pred[n.bus] ?? 0;
    return { bus: n.bus, note: `${n.km.toFixed(1)} km`, label: `${dv >= 0 ? '+' : '−'}${Math.abs(dv).toFixed(3)}`, color: C.ink,
      h: Math.max(4, Math.abs(dv) * sc), fill: dv >= 0 ? C.blue : '#7C93D6', pred: Math.abs(pr) * sc };
  });
  const meterTxt = (n) => (!n.has_meter ? ['SCADA', C.muted] : n.meter === 1 ? ['Energized', C.green] : n.meter === 0 ? [n.energized ? 'Outage ping (glitch)' : 'Outage ping', C.amberText] : ['Silent', C.muted]);
  const f4 = (v) => (v === null ? '—' : v.toFixed(4));

  return `
  <section style="max-width:900px">
    <div class="kicker">Voltage physics</div>
    <h1 class="h1">The voltage rise is the fingerprint</h1>
    <p class="lede">When a section opens, the load beyond it disappears. Current on every upstream segment falls, the I·Z drop shrinks, and upstream voltages rise. How much they rise depends on how much load was lost and where, so a handful of sensors can tell a trunk fault from a lateral even when nearly every meter is silent.</p>
  </section>
  <div class="sub mono">${show ? `Scenario: ${esc(d.truth.section)} open` : 'Scenario: blind mode (truth hidden)'} · seed ${d.scenario.seed} · ${Math.round(d.scenario.dropout * 100)}% meters missing</div>

  <div class="row" id="results">
    <section class="card chart grow-main" aria-label="Voltage profile">
      <h2 class="h2">Feeder voltage profile, before and after</h2>
      <div class="chart-legend">
        <span>${legendLine({ stroke: C.edge, width: 2.5 })}Before the fault</span>
        ${show ? `<span>${legendLine({ stroke: C.blue, width: 3 })}After, still energized</span>` : ''}
        <span><svg width="14" height="14" aria-hidden="true"><circle cx="7" cy="7" r="5.5" fill="#FFFFFF" stroke="${C.blue}" stroke-width="2"></circle></svg>Sensor reading</span>
        ${show ? `<span><i class="swatch" style="width:14px;background:${C.cut}"></i>Cut off (0 p.u.)</span>` : ''}
      </div>
      ${profileChart(d, strongest, show)}
    </section>
    <aside class="card grow-side" style="gap:0">
      <h2 class="h2" style="margin-bottom:12px">How OutageLens reads it</h2>
      <ol class="steps">
        <li><b>1</b><span>For every candidate section, one power flow predicts what each sensor should read if that section opened.</span></li>
        <li><b>2</b><span>Predictions are rescaled by today's load level, estimated from substation SCADA (${Math.round(d.scenario.load_level_hat * 100)}% of plan here).</span></li>
        <li><b>3</b><span>Observed shifts are scored against each prediction with a robust likelihood, so one bad sensor cannot veto the answer.</span></li>
      </ol>
      <div class="callout-blue">Feeder head: <b>${num(d.scada.pre_kw)} → ${num(d.scada.post_kw)} kW</b>. The ${num(d.scada.shed_kw)} kW drop ${Math.abs(d.scada.shed_kw - m.shed_kw_exp) < 0.15 * Math.max(m.shed_kw_exp, 1) ? 'matches' : 'is compared with'} the ~${num(m.shed_kw_exp)} kW downstream of ${esc(m.section)}.</div>
    </aside>
  </div>

  <section class="card" aria-label="Voltage shift at sensors">
    <div class="card-head"><h2 class="h2">Voltage shift at each sensor: observed vs predicted for ${esc(m.section)}</h2>
      <div class="chart-legend" style="margin:0"><span><i class="swatch" style="width:12px;height:12px;background:${C.blue}"></i>Observed ΔV</span><span><i class="swatch" style="width:16px;height:3px;background:${C.ink}"></i>Physics prediction</span></div></div>
    ${cols.length ? `<div class="table-box" style="margin-top:16px">
      <div class="dv-bars">${cols.map((c) => `<div class="dv-col"><span class="v" style="color:${c.color};margin-bottom:${c.pred !== null ? Math.max(0, c.pred - c.h + 4).toFixed(1) : 0}px">${c.label}</span><div class="b" style="height:${c.h.toFixed(1)}px;background:${c.fill}">${c.pred !== null ? `<div class="p" style="bottom:${(c.pred - 1.5).toFixed(1)}px"></div>` : ''}</div></div>`).join('')}</div>
      <div class="dv-names">${cols.map((c) => `<div><div class="mono" style="font-size:13px;font-weight:500">${esc(c.bus)}</div><div class="sub" style="font-size:12px">${esc(c.note)}</div></div>`).join('')}</div>
    </div>` : '<p class="sub">No voltage sensors besides the substation at this coverage.</p>'}
  </section>

  <section class="card" aria-label="Telemetry">
    <div class="card-head" style="align-items:center;margin-bottom:12px"><h2 class="h2">Telemetry snapshot</h2>
      <a class="btn" href="/api/telemetry.csv?${params()}" download>Download CSV</a></div>
    <div class="table-box" style="max-height:440px;overflow-y:auto"><table class="data" style="min-width:760px">
      <thead><tr><th>Bus</th><th class="num">km</th><th>Meter</th><th>Sensor</th><th class="num">V before</th><th class="num">V after</th><th class="num">ΔV</th></tr></thead>
      <tbody>${d.nodes.slice().sort((a, b) => a.km - b.km).map((n) => {
        const [mt, mc] = meterTxt(n);
        const dv = n.v_pre_obs === null ? null : n.v_post_obs - n.v_pre_obs;
        const dvTxt = dv === null ? '—' : n.v_post_obs <= 0.3 ? 'cut off' : `${dv >= 0 ? '+' : '−'}${Math.abs(dv).toFixed(4)}`;
        const dvCol = dv === null ? C.muted : n.v_post_obs <= 0.3 ? C.redText : C.blue;
        return `<tr><td class="mono" style="font-weight:500">${esc(n.bus)}</td><td class="num">${n.km.toFixed(1)}</td><td style="color:${mc}">${show || n.meter !== 0 ? mt : 'Outage ping'}</td><td>${n.sensor ? 'Yes' : '—'}</td><td class="num">${f4(n.v_pre_obs)}</td><td class="num">${f4(n.v_post_obs)}</td><td class="num" style="color:${dvCol}">${dvTxt}</td></tr>`;
      }).join('')}</tbody></table></div>
  </section>`;
}

function methodView(d) {
  const steps = [
    ['01', 'Feeder model', 'IEEE 13 and 34-bus data: line lengths, impedances, loads, capacitors, regulators.'],
    ['02', 'Power flow', 'OpenDSS, cross-checked every launch against an exact NumPy sweep.'],
    ['03', 'Partial observability', 'Meter dropout, comms glitches, sparse noisy sensors, unknown load level.'],
    ['04', 'Three models', 'Naive LCA baseline, Bayesian topology + physics, tree-structured GNN.'],
    ['05', 'Dispatch + audit', 'An agent plans the crew search; every decision is hash-chained for audit.'],
  ];
  const ideas = [
    ['Silence is neutral, not evidence', 'A missing meter adds the same likelihood to every candidate, so it cancels. Naive rules read “no ping” as “no outage”.'],
    ['The voltage rise is a fingerprint', 'Upstream sensors rise in proportion to the lost load, separating trunk faults from laterals with almost no meters reporting.'],
    ['SCADA says how much, topology says where', 'The substation power drop must equal the load downstream of the true section, and every utility already measures it.'],
  ];
  const deploy = [
    ['IEEE feeder models', 'GIS / CIM network model export (IEC 61968/61970)'],
    ['Simulated last-gasp pings', 'AMI head-end outage events and on-demand meter pings'],
    ['Simulated voltage sensors', 'Line sensors, smart-inverter telemetry, AMI voltage reads'],
    ['Simulated feeder-head power', 'Substation SCADA / RTU analogs (DNP3, IEC 61850)'],
    ['Dashboard ranking', 'Ranked candidates pushed into the OMS crew-dispatch work order'],
  ];
  const limits = [
    'Balanced positive-sequence model; single-phase laterals need a full three-phase model.',
    'One open section per event; multi-fault and protection-device logic come next.',
    'Dropout modelled as random; geographic correlation can be absorbed per area.',
    'GNN trained per feeder; training across many feeders would let it transfer.',
  ];
  const eng = d?.engine;
  return `
  <section style="max-width:900px">
    <div class="kicker">Method</div>
    <h1 class="h1">Topology says where. Physics says how much.</h1>
    <p class="lede">OutageLens fuses the data a utility already collects (network model, smart-meter events, a few voltage sensors, substation SCADA) into a calibrated probability for every section of the feeder.</p>
  </section>
  <section class="pipeline" aria-label="Pipeline">${steps.map(([n, t, b]) => `<div class="card"><div class="n">${n}</div><div class="t">${t}</div><div class="b">${b}</div></div>`).join('')}</section>
  <div class="row">
    <section class="card" style="flex:3 1 560px;padding:20px 22px" aria-label="Equations">
      <h2 class="h2" style="margin-bottom:14px">The model</h2>
      <div style="display:flex;flex-direction:column;gap:16px">
        <div><div class="kicker">Posterior over sections</div><div class="eq eq-big">P(e | D) ∝ P(S | e) · P(V | e) · P(ΔP_head | e) · P(e)</div></div>
        <div><div class="kicker">Meters: silence cancels</div><div class="eq">P(S=0 | i ∈ T(e)) = P(S=1 | i ∉ T(e)) = (1−α)(1−ε)
P(S=∅) = α   for every e</div></div>
        <div><div class="kicker">Physics: exact radial sweep</div><div class="eq">J = BIBC · conj(S / V)      V = T·V₀ − BCBV · J
V̂ₘ(e) = 0 if m ∈ T(e), else Vₘ,pre + λ_load · ΔVₘ(e)</div></div>
        <div><div class="kicker">Tree GNN and margin-gated ensemble</div><div class="eq">H′ = ReLU(H·W₀ + S·H·W₁ + P·H·W₂ + Â·H·W₃ + b)
P_ens ∝ P_Bayes^w · P_GNN^(1−w),  w = 0.65 + 0.35·min(Δ/0.5, 1)</div></div>
      </div>
    </section>
    <section class="ideas" style="flex:2 1 380px" aria-label="Key ideas">
      <h2 class="h2" style="margin-bottom:6px">Three ideas carry it</h2>
      ${ideas.map(([t, b]) => `<div class="i"><b>${t}</b><span>${b}</span></div>`).join('')}
    </section>
  </div>
  <div class="row">
    <section class="card" style="flex:3 1 560px" aria-label="Deployment path">
      <h2 class="h2" style="margin-bottom:12px">From prototype to utility OMS</h2>
      <div class="table-box"><table class="data" style="min-width:560px"><thead><tr><th>In the prototype</th><th>In a utility deployment</th></tr></thead>
      <tbody>${deploy.map(([a, b]) => `<tr><td>${a}</td><td style="color:var(--body2)">${b}</td></tr>`).join('')}</tbody></table></div>
    </section>
    <section class="card" style="flex:2 1 380px" aria-label="Limitations">
      <h2 class="h2" style="margin-bottom:6px">Limits, stated up front</h2>
      ${limits.map((l) => `<div class="limit"><svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true" style="margin-top:4px"><line x1="2" y1="7" x2="12" y2="7" stroke="${C.muted}" stroke-width="2"></line></svg><span>${l}</span></div>`).join('')}
      <div class="mono sub" style="margin-top:8px;font-size:12px">19 tests · gradient-checked GNN · hash-chained decision log · ${eng ? esc(eng.opendss ? `OpenDSS ⇄ NumPy within ${eng.max_dev_pu.toExponential(1)} p.u.` : 'NumPy DistFlow engine active') : 'engine starting'}</div>
    </section>
  </div>`;
}

// ------------------------------------------------------- dispatch + audit
const scenarioKey = () => `${params()}|crew=${state.crewSeed}`;

async function loadDispatch() {
  state.dispatchBusy = true;
  render();
  try {
    const q = params();
    q.set('crew_seed', state.crewSeed);
    state.dispatch = await getJSON(`/api/dispatch?${q}`, { method: 'POST' });
    state.dispatchKey = scenarioKey();
    state.audit = null;
  } catch (err) {
    toast(`Dispatch agent failed: ${err.message}`);
  } finally {
    state.dispatchBusy = false;
    if (state.view === 'dispatch') render();
  }
}

async function loadStudy() {
  if (state.study[state.feeder]) return;
  try {
    state.study[state.feeder] = await getJSON(`/api/dispatch/study?feeder=${state.feeder}`);
    if (state.view === 'dispatch') render();
  } catch (err) {
    toast(`Could not load the dispatch study: ${err.message}`);
  }
}

async function loadAudit(verify) {
  try {
    state.audit = await getJSON('/api/audit?limit=40');
    if (verify) state.verify = await getJSON('/api/audit/verify');
  } catch (err) {
    toast(`Audit log unavailable: ${err.message}`);
  }
  if (state.view === 'audit') render();
}

const ACTION_COLOR = { INSPECT: C.ink, PING: C.blue, RESTORE: C.green };

function dispatchView(d) {
  const r = state.dispatch && state.dispatchKey === scenarioKey() ? state.dispatch : null;
  const st = state.study[state.feeder];
  const show = !state.blind;
  const head = `
  <section class="row" style="align-items:flex-end;justify-content:space-between;gap:16px 40px">
    <div style="flex:1 1 560px;min-width:0">
      <div class="kicker">Dispatch agent</div>
      <h1 class="h1">From a ranked list to a crew at the fault</h1>
      <p class="lede">The agent plans, acts, observes and updates. It sends the crew to the section with the highest fault probability per crew-minute, pings a silent smart meter when the answer is worth the minutes, and after every “nothing found” updates its beliefs with Bayes’ rule before planning again. Crews miss a fault 5% of the time, so it has to cope with that.</p>
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:10px">
      <button type="button" class="btn btn-primary" id="d-run" ${state.dispatchBusy ? 'disabled' : ''}>${state.dispatchBusy ? 'Dispatching…' : r ? 'Run again' : 'Run dispatch agent'}</button>
      <button type="button" class="btn" id="d-luck" ${state.dispatchBusy ? 'disabled' : ''}>New crew luck</button>
    </div>
  </section>
  <div class="sub mono">${show ? `Scenario: ${esc(d.truth.section)} open` : 'Scenario: blind mode (the agent does not know the truth either)'} · seed ${d.scenario.seed} · ${Math.round(d.scenario.dropout * 100)}% meters missing · crew luck ${state.crewSeed}</div>`;

  const studyHtml = st ? `
  <section class="card" aria-label="Agent study">
    <h2 class="h2" style="margin-bottom:6px">Across ${st.n} simulated outages on ${esc(d.feeder.short)}</h2>
    <p class="sub" style="margin:0 0 12px">Average crew time to find the fault, same crew luck for every policy (detection ${Math.round(st.detect * 100)}%).</p>
    <div class="kpis" style="border:0">
      <div class="kpi"><div class="kicker">Dispatch agent</div><div class="val">${num(st.agent)} min</div><div class="sub">median ${num(st.agent_median)} · ${st.pings.toFixed(1)} pings per outage</div></div>
      <div class="kpi"><div class="kicker">Agent without pings</div><div class="val">${num(st.agent_no_pings)} min</div><div class="sub">value of on-demand pings: ${num(st.agent_no_pings - st.agent)} min</div></div>
      <div class="kpi"><div class="kicker">Follow the ranking</div><div class="val">${num(st.ranked)} min</div><div class="sub">median ${num(st.ranked_median)} · no belief updates</div></div>
      <div class="kpi"><div class="kicker">Conventional patrol</div><div class="val">${num(st.conventional)} min</div><div class="sub">no analytics, backbone first</div></div>
    </div>
    <p class="sub" style="margin:10px 0 0;font-size:12px">The agent minimises expected time, so it is not best on every single outage: it ties or beats the static ranking in ${st.agent_not_worse_pct.toFixed(0)}% of them.</p>
  </section>` : loadingView('Loading the agent study…');

  if (!r) return head + `<section class="card"><p class="sub" style="margin:0">Press <b>Run dispatch agent</b> to send a simulated crew after the current Control-room scenario.</p></section>` + studyHtml;

  const a = r.agent;
  const saved = r.ranked.minutes - a.minutes;
  return head + `
  <section class="kpis" aria-label="This outage">
    <div class="kpi"><div class="kicker">Agent found it in</div><div class="val">${num(a.minutes)} min</div><div class="sub">${a.inspections} inspection${a.inspections === 1 ? '' : 's'} · ${a.pings} ping${a.pings === 1 ? '' : 's'}</div></div>
    <div class="kpi"><div class="kicker">Follow the ranking</div><div class="val">${num(r.ranked.minutes)} min</div><div class="sub">${r.ranked.inspections} inspections · ${saved >= 0 ? `${num(saved)} min slower` : `${num(-saved)} min faster this time`}</div></div>
    <div class="kpi"><div class="kicker">Conventional patrol</div><div class="val">${num(r.conventional.minutes)} min</div><div class="sub">${r.conventional.km.toFixed(1)} km of line</div></div>
    <div class="kpi"><div class="kicker">Logged as</div><div class="val" style="font-size:22px"><a href="#/audit">Decision #${r.decision.seq}</a></div><div class="sub mono">sha256 ${r.decision.hash.slice(0, 16)}…</div></div>
  </section>
  <section class="card" aria-label="Agent timeline">
    <h2 class="h2" style="margin-bottom:6px">What the agent did, step by step</h2>
    <p class="sub" style="margin:0 0 6px">Starting belief: ${r.start_top3.map((x) => `${esc(x.section.replace('-', ' → '))} ${pct(x.p)}`).join(' · ')}</p>
    ${a.steps.map((s) => `<div class="ev-row" style="grid-template-columns:44px 84px minmax(0,1fr) 120px">
      <span class="mono sub">${s.n}</span>
      <span class="t" style="color:${ACTION_COLOR[s.action] || C.ink}">${s.action}</span>
      <span class="x"><b>${esc(s.label)}</b> · ${esc(s.observation)}<br><span class="sub">${esc(s.reason)}${s.action !== 'RESTORE' ? ` · belief now: ${esc(s.top_section.replace('-', ' → '))} ${pct(s.top_p)}` : ''}</span></span>
      <span class="mono" style="text-align:right">${s.action === 'RESTORE' ? '' : `+${num(s.minutes)} min · `}${num(s.elapsed)} min</span>
    </div>`).join('')}
  </section>` + studyHtml;
}

function auditView() {
  const a = state.audit;
  const v = state.verify;
  const head = `
  <section style="max-width:900px">
    <div class="kicker">Audit log</div>
    <h1 class="h1">Every decision, tamper-evident</h1>
    <p class="lede">Each localisation and dispatch plan is appended to a SHA-256 hash chain: a digest of what the control room observed, the exact models that ran (weights hash, calibration, engine) and what they concluded. Each record carries the previous record’s hash, so editing, deleting or reordering any entry breaks every hash after it. Simulated ground truth is never recorded.</p>
  </section>`;
  if (!a) return head + loadingView('Loading the decision chain…');
  const status = v ? (v.ok
    ? `<span class="tag tag-exact">CHAIN INTACT</span> <span class="sub">${v.count} records re-hashed in ${v.checked_ms} ms · head matches live log: ${v.matches_live_head ? 'yes' : 'no'}</span>`
    : `<span class="tag tag-miss">BROKEN AT RECORD ${v.first_bad_seq}</span> <span class="sub">${esc(v.problem)}</span>`)
    : '<span class="sub">Not verified yet in this session.</span>';
  const summary = (r) => {
    const p = r.payload;
    if (r.kind === 'localisation') return `${esc(p.feeder)} · top ${esc(p.result.ensemble_top3[0].section)} ${pct(p.result.ensemble_top3[0].p)} · baseline ${esc(p.result.baseline_top)}`;
    if (r.kind === 'dispatch') return `${esc(p.feeder)} · found ${esc(p.outcome.found_section || '—')} in ${num(p.outcome.minutes)} min · ${p.steps.length} steps`;
    return esc(r.kind);
  };
  return head + `
  <section class="card" aria-label="Chain status">
    <div class="card-head" style="align-items:center">
      <div style="display:flex;flex-wrap:wrap;gap:10px;align-items:center">${status}</div>
      <div style="display:flex;flex-wrap:wrap;gap:10px">
        <button type="button" class="btn btn-primary" id="a-verify">Verify the chain</button>
        <button type="button" class="btn" id="a-refresh">Refresh</button>
        <a class="btn" href="/api/audit.jsonl" download>Download log</a>
      </div>
    </div>
    <div class="row" style="margin-top:14px;gap:12px 32px">
      <div><div class="kicker">Records</div><div class="mono" style="font-size:20px">${a.count}</div></div>
      <div style="min-width:0;flex:1 1 420px"><div class="kicker">Current head hash</div><div class="mono" style="font-size:13px;word-break:break-all">${esc(a.head)}</div></div>
    </div>
    <p class="sub" style="margin:12px 0 0">Verify it yourself, with no OutageLens code: <span class="mono">python scripts/verify_audit.py outagelens_decisions.jsonl</span>${a.persisted ? '' : ' · this server keeps the log in memory only'}</p>
  </section>
  <section class="card" aria-label="Recent decisions">
    <h2 class="h2" style="margin-bottom:12px">Recent decisions</h2>
    ${a.records.length ? `<div class="table-box"><table class="data" style="min-width:820px">
      <thead><tr><th>#</th><th>Time (UTC)</th><th>Kind</th><th>Decision</th><th>Hash</th><th>Previous</th></tr></thead>
      <tbody>${a.records.map((r) => `<tr><td class="mono">${r.seq}</td><td class="mono">${esc(r.time.slice(11, 23))}</td><td>${esc(r.kind)}</td><td>${summary(r)}</td><td class="mono">${r.hash.slice(0, 12)}…</td><td class="mono sub">${r.prev_hash.slice(0, 12)}…</td></tr>`).join('')}</tbody>
    </table></div>` : '<p class="sub">No decisions yet. Open the Control room to make one.</p>'}
  </section>`;
}

// ---------------------------------------------------------------- render
function render() {
  const view = $('#view');
  document.querySelectorAll('.nav a').forEach((a) => (a.dataset.view === state.view ? a.setAttribute('aria-current', 'page') : a.removeAttribute('aria-current')));
  const d = state.data;
  if (state.view === 'robustness') {
    view.innerHTML = robustnessView();
    $('#b-run')?.addEventListener('click', () => loadBench(true));
    $('#b-table')?.addEventListener('click', (ev) => { ev.preventDefault(); $('#table')?.scrollIntoView({ behavior: 'smooth' }); });
    return;
  }
  if (state.view === 'method') { view.innerHTML = methodView(d); return; }
  if (state.view === 'audit') {
    view.innerHTML = auditView();
    $('#a-verify')?.addEventListener('click', () => loadAudit(true));
    $('#a-refresh')?.addEventListener('click', () => loadAudit(false));
    return;
  }
  if (state.view === 'dispatch') {
    view.innerHTML = d ? dispatchView(d) : loadingView('Loading the scenario…');
    $('#d-run')?.addEventListener('click', () => loadDispatch());
    $('#d-luck')?.addEventListener('click', () => { state.crewSeed = Math.floor(Math.random() * 1e6) + 1; loadDispatch(); });
    return;
  }
  if (!d) { view.innerHTML = loadingView('Starting the physics engines and loading the GNN. The first launch takes a few seconds…'); return; }
  view.innerHTML = state.view === 'physics' ? physicsView(d) : `<div id="results" style="display:flex;flex-direction:column;gap:24px">${controlView(d)}</div>`;
}

function route() {
  const v = (location.hash.replace('#/', '') || 'control').split('?')[0];
  state.view = ['control', 'dispatch', 'robustness', 'physics', 'audit', 'method'].includes(v) ? v : 'control';
  buildControls();
  render();
  if (state.view === 'robustness') loadBench();
  if (state.view === 'audit') loadAudit(false);
  if (state.view === 'dispatch') loadStudy();
  document.title = `OutageLens · ${({ control: 'Control room', dispatch: 'Dispatch agent', robustness: 'Robustness', physics: 'Voltage physics', audit: 'Audit log', method: 'Method' })[state.view]}`;
}

async function init() {
  render();
  try {
    state.feeders = await getJSON('/api/feeders');
  } catch (err) {
    toast(`Cannot reach the OutageLens server: ${err.message}`);
    return;
  }
  renderFeederPicker();
  $('#feeder').addEventListener('change', (ev) => {
    state.feeder = ev.target.value;
    state.blind = false;
    state.data = null;
    buildControls();
    render();
    loadScenario();
    if (state.view === 'robustness') loadBench();
  });
  window.addEventListener('hashchange', route);
  route();
  await loadScenario();
}

init();
