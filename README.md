# OutageLens ⚡

**Find the faulted section of a distribution feeder when most smart meters stay silent.**

OutageLens combines network topology, sparse voltage telemetry, feeder-head SCADA and incomplete smart-meter
reports into a calibrated probability for every line section. It ranks the likely fault locations, explains
the evidence behind each one, and converts that ranking into patrol kilometres and crew minutes saved.

Built for the IEEE Computer Society PeerLink Online Hackathon 2026 (48-hour prototype) by
**Jackson Abetianbe** and **Peter Eziagor**.

**Code:** https://github.com/Jae3y/outagelens
**Live demo:** see [`docs/DEPLOY.md`](docs/DEPLOY.md) to put it online in about 5 minutes (Hugging Face or Render).
**Submission package:** [`docs/SUBMISSION.md`](docs/SUBMISSION.md).

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run.py                     # prints the URL (http://127.0.0.1:8000, or the next free port)
```

Or `python setup_and_run.py` to install, pre-train and launch in one step. Or use Docker:

```bash
docker build -t outagelens . && docker run -p 8000:8000 outagelens
```

The first launch trains the GNN once per feeder (about 3 s for IEEE 13, about 8 s for IEEE 34) and caches it in
`.cache/`; `python scripts/warm_cache.py` does this ahead of time. The app needs no internet at runtime (fonts
are bundled), so the laptop copy is a safe fallback during a live demo.

`opendssdirect.py` is optional. If it is missing or fails to load, the app switches to its built-in NumPy
power-flow engine automatically. Both engines solve the same feeder model and agree within 10⁻⁴ p.u.

### The control room

| Screen | What it shows |
|---|---|
| **Control room** | Scenario controls (faulted section, missing meters, sensor coverage, glitches, model, sensor placement), the verdict next to the naive baseline, patrol and crew-time impact, the feeder map coloured by fault probability, ranked sections, and tagged evidence. **Blind judge mode** and **Trigger mystery outage** hide the ground truth until you reveal it. |
| **Robustness** | 280-outage Monte Carlo benchmark: accuracy and patrol distance as meters go silent, evidence ablation, sensor-placement study, calibration, and a results table. A button re-runs it with a fresh seed. |
| **Voltage physics** | Before/after voltage profile, observed vs predicted voltage shift at each sensor, the telemetry table and a CSV download. |
| **Method** | Pipeline, equations, the three key ideas, the path into a utility OMS, and the limitations. |

The JSON API behind it is documented at `/docs` (FastAPI's auto-generated OpenAPI page) while the server runs.

## Results (fixed seeds, `python scripts/reproduce_results.py`)

280 Monte Carlo outages per feeder: random faulted section, random load level (80–115 % of planning load,
±10 % per bus), 25 % of buses carrying randomly placed voltage sensors, 2 % comms-glitch rate, and missing
meter rates from 10 % to 95 %.

| IEEE 34-bus (93.9 km of line) | Top-1 | Top-3 | Mean hop error | Patrol km | Crew min |
|---|---|---|---|---|---|
| **OutageLens ensemble (Bayes + GNN)** | **77.5 %** | **97 %** | **0.7** | **4.7** | **51** |
| OutageLens Bayesian + physics | 76 % | 98 % | 0.7 | 4.8 | 52 |
| GNN only | 63 % | 87 % | 1.6 | 5.3 | 63 |
| Naive LCA baseline | 41 % | 56 % | 4.3 | 25.8 | 283 |
| Conventional patrol (no analytics) | – | – | – | 52.3 | 599 |

| IEEE 13-bus | Top-1 | Top-3 | Patrol km vs conventional |
|---|---|---|---|
| **OutageLens ensemble** | **92 %** | **100 %** | **0.22 vs 1.79 km (−88 %)** |
| Naive LCA baseline | 49 % | 66 % | 0.92 vs 1.79 km (−49 %) |

**When 95 % of meters are silent:** OutageLens still gets Top-1 on 80 % (13-bus) and 57 % (34-bus) of outages.
The baseline manages 12 % and 15 %.

**Evidence ablation (13-bus, 95 % missing):** meters only 7 % → plus voltage sensors 50 % → plus feeder-head
SCADA 90 % Top-1.

**Sensor placement:** with the same 40 % sensor budget on IEEE 34, the built-in placement optimiser lifts
Top-1 from 72 % (random placement) to 92 %.

**Calibration:** expected calibration error of the ensemble is 1.7 points on IEEE 13 and 6.0 points on
IEEE 34, where it is mildly overconfident on near-tie cases.

**Ensemble:** the Bayesian model carries most of the accuracy. The GNN is consulted only when Bayes is torn
between its top two sections (see below). On a separate 280-outage evaluation
(`python scripts/compare_ensembles.py`) this gives the best NLL of every strategy tried on IEEE 34 (0.449, vs
0.458 for Bayes alone and 0.454 for a fixed 0.65/0.35 pool) and matches Bayes on IEEE 13 (0.132 vs 0.131).

## How it works

| Layer | What it does | File |
|---|---|---|
| Feeders | IEEE 13 and 34-bus data (published line lengths, configurations, spot and distributed loads, capacitors, regulators) reduced to positive-sequence equivalents, per-unit on a 1 MVA base | `core/feeder_topology.py` |
| Physics | OpenDSS (via `opendssdirect.py`) **and** an exact BIBC/BCBV backward/forward sweep in NumPy. Auto-selected, cross-validated, with OpenDSS isolated on its own thread | `core/opendss_engine.py` |
| Observability | Opens a section, then simulates meter dropout α, false alarms ε, sparse noisy voltage sensors β, feeder-head SCADA and random load levels | `core/sparsity_simulator.py` |
| Model 1 | Naive lowest-common-ancestor / downstream-vote baseline (outage pings only) | `core/baseline_locator.py` |
| Model 2 | Bayesian model: P(e\|D) ∝ P(S\|e)·P(V\|e)·P(ΔP_head\|e)·P(e), with robust outlier mixture, SCADA-estimated load level, temperature-scaled calibration, explanations and a sensor placement optimiser | `core/bayesian_locator.py` |
| Model 3 | Tree-structured GNN (subtree / path-to-substation / neighbour message passing), hand-written NumPy backprop (gradient-checked), Adam, temperature scaling | `core/gnn_locator.py` |
| Ensemble | Margin-gated log-linear pool: Bayes weight w = 0.65 + 0.35·min(Δ/0.5, 1), where Δ is the Bayes top-1/top-2 gap. A decisive Bayes call passes through unchanged; near ties are pooled with the GNN. Tuned on seeds disjoint from every reported benchmark | `core/ensemble.py` |
| Evaluation | Monte Carlo benchmark, Top-k, hop error, patrol km and minutes, ablation, calibration (ECE), placement study | `core/evaluator.py` |
| API | FastAPI service: one call returns a full scenario (verdict, all three models, evidence, map, physics, telemetry); benchmark and CSV endpoints; OpenDSS kept on a dedicated thread | `server/` |
| Control room | Hand-built HTML/CSS/JS ("Patrol Sheet" design), SVG charts drawn from live API data, no build step | `web/` |
| Legacy UI | The original Streamlit dashboard, kept as a fallback (`pip install -r requirements-dev.txt`, `streamlit run app.py`) | `app.py`, `ui/` |

Three ideas carry most of the performance:

1. **Silence is neutral, not evidence.** Under random dropout a missing meter contributes the same likelihood
   to every candidate, so it cancels exactly. The baseline instead treats "no ping" as "no outage".
2. **The voltage rise is a fingerprint.** When a section opens, the load beyond it disappears and every
   upstream sensor's voltage rises in proportion to the lost load. That separates a trunk fault from a
   lateral one even when no downstream meter reports.
3. **Feeder-head SCADA says how much, topology says where.** The substation power drop must equal the
   downstream load of the true section, and every utility already has this measurement.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The 13 tests cover power-flow physics invariants on both feeders, a finite-difference gradient check of the
GNN, the ensemble's guarantees (valid distribution, decisive Bayes passes through, near ties defer to the GNN,
weight continuous in the margin), an end-to-end check that OutageLens beats the baseline on accuracy and patrol
distance, and the HTTP API contract (demo scenario, every section on both feeders, input validation,
benchmark, CSV and the static site). Once the repository is on GitHub, `.github/workflows/ci.yml` runs them on
every push, then builds the Docker image and smoke-tests the demo scenario inside it.

## Limitations (stated up front)

- Balanced positive-sequence model. Phase-specific signatures on single-phase laterals need a full three-phase
  model.
- One open section per event. Multiple simultaneous faults and protection-device logic (recloser/fuse
  coordination) come next.
- Dropout is modelled as random. Geographically correlated last-gasp loss can be absorbed with a per-area α.
- The GNN is trained per feeder. Training across many synthetic feeders would let it transfer without
  retraining.

See [`docs/PITCH.md`](docs/PITCH.md) for the demo script, judging-criteria map and Q&A preparation.
