# OutageLens: submission package

The three deliverables and where each one lives. Items marked **[you]** need your account or your team's
details; everything else is in this repository and working.

## 1 · Working product

| Item | Status | Where |
|---|---|---|
| Web control room (4 screens) + JSON API | Done, 13 tests passing | `web/`, `server/`; run with `python run.py` |
| Physics engines (OpenDSS + NumPy sweep, cross-checked) | Done | `core/opendss_engine.py` |
| Three localisation models + ensemble | Done | `core/baseline_locator.py`, `core/bayesian_locator.py`, `core/gnn_locator.py`, `core/ensemble.py` |
| Reproducible results | Done | `python scripts/reproduce_results.py`, `python scripts/compare_ensembles.py` |
| One-command setup | Done | `python setup_and_run.py`, or Docker |
| Crash sweep | Done: 810 chart renders (every section, both feeders, all models, extreme settings) and a full UI pass, zero errors | — |
| Continuous integration | Ready; runs once the repo is on GitHub | `.github/workflows/ci.yml` |

## 2 · Live demo

| Item | Status | Where |
|---|---|---|
| Container image | Done, builds and serves locally | `Dockerfile` |
| One-click hosting configs | Done | `render.yaml` (Render), Hugging Face steps in `docs/DEPLOY.md` |
| Public URL | **[you]** Deploy from your GitHub account (about 5 minutes), then paste the URL here and in the README | `docs/DEPLOY.md` |
| Offline fallback | Done: the laptop copy needs no internet (fonts bundled) | `python run.py` |
| Demo script and Q&A | Done | `docs/PITCH.md` |

## 3 · Project package

| Item | Status | Where |
|---|---|---|
| README (problem, results, architecture, limits) | Done | `README.md` |
| Pitch / demo playbook + judge Q&A | Done | `docs/PITCH.md` |
| Deployment guide | Done | `docs/DEPLOY.md` |
| UI design (Claude Design canvas) | Done | [OutageLens Redesign](https://claude.ai/artifact/FkzDJYxroAXfNSt797BtjF) (private until you share it) |
| Source repository | Done, public | https://github.com/Jae3y/outagelens |
| Slide deck (8 slides, speaker notes) | Done; fill in team name, members, live URL and GitHub URL | [OutageLens Pitch Deck](https://claude.ai/artifact/Y2qRaHkqfNfApvnGKohijR) (private until you share it; exports to PPTX/PDF) |
| Demo video (if required) | **[you]** Screen-record the 3-minute script in `docs/PITCH.md` | — |
| Team names, emails, track | **[you]** | — |

## Copy-paste blurbs

**One line.** OutageLens finds the faulted section of a power distribution feeder from topology, voltage
physics and incomplete smart-meter data, even when most meters stay silent.

**Short description (≈100 words).** When a distribution outage hits, utilities often know a feeder is down but
not where, because most smart meters never deliver their last-gasp message. OutageLens fuses the network
model, a few voltage sensors, substation SCADA and whatever meter reports arrive into a calibrated probability
for every line section. On the IEEE 34-bus test feeder it names the right section 77.5 % of the time (a naive
baseline manages 41 %), keeps the right answer in its top three 97 % of the time, and cuts the line a crew must
patrol from 52 km to under 5 km on average. It runs as a single container with a web control room.

**Tech stack.** Python, NumPy, pandas, NetworkX, OpenDSS (opendssdirect.py), FastAPI, Uvicorn,
HTML/CSS/JavaScript (no framework), SVG, Docker.

## Slide outline (8 slides)

1. **Title:** OutageLens: find the fault when the meters go quiet.
2. **Problem:** feeders down, location unknown; last-gasp messages lost; crews patrol tens of km.
3. **Insight:** silence is neutral, the voltage rise is a fingerprint, SCADA says how much.
4. **How it works:** pipeline (feeder model → physics → partial observability → three models → ranked work order).
5. **Live demo:** Control room, then a mystery outage.
6. **Results:** 77.5 % vs 41 % Top-1, 97 % Top-3, 4.7 vs 52 km patrol, accuracy-vs-missing-meters chart.
7. **Why it holds up:** ablation, calibration, sensor placement 72 % → 92 %.
8. **Path to deployment and limits:** OMS/AMI/SCADA integration table; the four stated limits; ask.
