# OutageLens: pitch & demo playbook

> I found no public rules or rubric for "PeerLink Hackathon 2026", so this playbook targets the criteria that
> IEEE student hackathons and design competitions publish most often: originality, technical depth,
> practical impact, demo and presentation, and feasibility. **Swap in the official rubric as soon as the
> organisers publish it** and re-order the talk to match its weights.

## Judging-criteria map

| Likely criterion | What to show | Where in the app |
|---|---|---|
| Problem relevance | Utilities without line sensors patrol blind; last-gasp messages get lost | Opening line, impact strip |
| Originality | Fusion of topology + voltage-rise physics + SCADA; silence-cancels insight; optimised sensor placement from the same model | Control room evidence list, Robustness placement study |
| Technical depth | IEEE feeders, OpenDSS cross-validated against a hand-built BIBC/BCBV sweep, Bayesian model, hand-written GNN with gradient check, calibration (ECE), FastAPI service with 13 tests | Method screen, Robustness calibration chart |
| Measured results | 77.5 % vs 41 % Top-1 (IEEE 34), 97 % Top-3, 4.7 km vs 52 km patrol, 80 % vs 12 % Top-1 with 95 % of meters missing (IEEE 13) | Robustness headline strip + charts |
| Practical impact | Crew minutes saved per outage; OMS/ADMS deployment path; uses data utilities already collect | Impact strip, Method deployment table |
| Demo quality | Live mystery outage with the ground truth hidden (even the fault picker locks), then revealed | **Trigger mystery outage** button |
| Agentic AI (the organiser's focus) | Dispatch agent: plans, acts, observes crew reports and meter pings, updates beliefs, restores. 40.9 vs 121.6 crew-minutes on IEEE 34 | **Dispatch agent** screen |
| Audit-verifiable security (the organiser's focus) | Every decision hash-chained with inputs digest and model fingerprint; tamper detection; offline verifier script | **Audit log** screen, decision chip on Control room |
| Honesty / feasibility | Limitations listed up front; reproducible scripts; tests; deployable container | Method screen, README |

## 3-minute demo script (web app, IEEE 34 default scenario)

1. **(0:00) Hook.** "When the lights go out, a utility often knows *that* a feeder is down but not *where*.
   Smart meters should send a last-gasp message, but most never arrive. Crews end up patrolling kilometres
   of line."
2. **(0:20) Control room.** 65 % of meters are silent; 14 of 33 report. The naive baseline says
   **Line 816 → 824** and is wrong by 4 sections: a single glitched ping at bus 826 dragged it upstream.
   OutageLens says **Line 854 → 852 at 79 %**, an exact match, and the regulator next to it holds the other
   21 %. Read the evidence list: bus 856's voltage rose +0.096 p.u. (the model predicted +0.093), the feeder
   head lost 1,632 kW (this section carries ~1,631 kW), six last-gasp pings from downstream, and bus 830 still
   has power, locking the fault below it. Then the impact strip: **11.2 km patrolled instead of 52.8, 96
   crew-minutes instead of 540.**
3. **(1:00) Voltage physics.** "This is the fingerprint. When load disappears, upstream voltage rises. Every
   sensor's measured shift tracks what the physics predicts for this section: 856 rose 0.096 p.u. against a
   predicted 0.093, while the sensor beyond the fault at 838 drops to zero."
4. **(1:20) Mystery outage.** Back on the Control room, click **Trigger mystery outage**. The truth locks;
   even the fault picker says "Hidden until reveal". Push meters missing to 85 %. Ask the judges to guess,
   then press **Reveal ground truth**.
5. **(1:40) Dispatch agent.** "Ranking is not dispatch." Press **Run dispatch agent**: it pings the silent meter
   at 852 (answer: no power), sends the crew to 854 → 852, finds the fault, and plans the restore. Point at the
   study: 40.9 crew-minutes on average versus 121.6 for a crew following the static list. Then click the
   decision number to open the **Audit log**, press **Verify the chain**: "every decision is hash-chained;
   edit one and the chain breaks."
6. **(2:00) Robustness.** The baseline collapses as meters go silent while OutageLens degrades gracefully
   (57.5 % vs 15 % Top-1 at 95 % missing). The ablation shows why: meters alone 7 %, add voltage sensors 53 %,
   add feeder-head SCADA 57 %. On the calibration plot, 125 of 280 calls land at ≥95 % confidence and every
   one of them is right.
7. **(2:25) Placement study.** "The same model tells a utility where its next sensor buys the most
   visibility: 72 % → 92 % Top-1 for the same 40 % budget."
8. **(2:45) Close.** "No new hardware on every line. Uses data utilities already have. A ranked work order
   that cuts patrol from 52 km to under 5 km on average on the IEEE 34-bus feeder."

Tip: switch the feeder to IEEE 13 on the Robustness screen for the most dramatic ablation (7 % → 50 % → 90 %).

## Likely judge questions

- **"Isn't this just matching against your own simulator?"** The scenario loads are randomised (80–115 %
  level, ±10 % per bus) and unknown to the model. Sensors carry noise, meters glitch, and the model only uses
  nominal planning loads rescaled by a SCADA estimate. The benchmark sensor layout is random per scenario.
- **"Why trust OpenDSS *and* your own solver?"** They are cross-checked on every launch (agreement within
  10⁻⁴ p.u.). The badge shows which engine is live, and either one can run the demo.
- **"Why not just use the GNN?"** On its own it is clearly weaker (63 % vs 76 % Top-1 on IEEE 34), and
  physics-informed Bayes is interpretable.
- **"Then why have a GNN at all?"** It is an independent, purely data-driven opinion, consulted only when
  Bayes is torn between its top two sections (margin-gated ensemble). Be honest that the gain is small:
  best NLL on IEEE 34 (0.449 vs 0.458 for Bayes alone), level with Bayes on IEEE 13. Its real value is the
  roadmap: trained on many feeders, it can run where an accurate impedance model does not exist yet.
- **"How did you choose the ensemble weights?"** On two simulation seeds kept separate from every reported
  benchmark, comparing fixed, entropy-weighted, tie-break and margin-gated pools
  (`scripts/compare_ensembles.py` reproduces the comparison).
- **"What about regulators reacting after the fault?"** Taps are frozen at their steady-state positions,
  which reflects the seconds after the event, before tap changers move (typical delay 30–60 s).
- **"Real deployment?"** CIM/GIS model import, AMI head-end events, SCADA analogs, ranked list into the OMS
  work order (Method screen table). The app already runs as a container with a JSON API, so it slots in as a
  service beside an OMS.
- **"Biggest limitation?"** Balanced single-phase-equivalent model and one fault at a time. Both are listed,
  with next steps.

## Before the pitch

- [ ] Run `python scripts/reproduce_results.py` and confirm the numbers match the slides.
- [ ] Open the live URL 5 minutes early (free hosts sleep) and keep `python run.py` running on the laptop as a
      fallback. The laptop copy needs no internet.
- [ ] Visit the Robustness screen once, so the benchmark is cached before judges see it.
- [ ] Rehearse the mystery outage at 85 % missing a few times. A tough draw can land in Top-3 rather than
      Top-1, so be ready to explain why Top-3 still means a short patrol.
- [ ] Confirm the official judging rubric and re-order this script to match it.
