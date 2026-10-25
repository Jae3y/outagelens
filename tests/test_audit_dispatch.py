"""Audit chain guarantees and dispatch-agent behaviour."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from core.audit import GENESIS, AuditLedger, verify_records
from core.dispatch import DispatchAgent
from core.feeder_topology import ieee34
from core.opendss_engine import DistFlowEngine
from core.sparsity_simulator import ObservabilityConfig, simulate

ROOT = Path(__file__).resolve().parents[1]


def _ledger(tmp_path, n=5):
    led = AuditLedger(tmp_path / "log.jsonl")
    for i in range(n):
        led.append("localisation", {"i": i, "top": "854-852"})
    return led


def test_chain_verifies_and_survives_restart(tmp_path):
    led = _ledger(tmp_path)
    assert led.verify()["ok"] and led.count == 5
    first = led.all_records()[0]
    assert first["prev_hash"] == GENESIS
    again = AuditLedger(tmp_path / "log.jsonl")  # reload from disk and keep extending the same chain
    rec = again.append("dispatch", {"x": 1})
    assert rec["seq"] == 6 and rec["prev_hash"] == led.head and again.verify()["ok"]


@pytest.mark.parametrize("attack", ["edit", "delete", "reorder"])
def test_tampering_is_detected(tmp_path, attack):
    records = _ledger(tmp_path).all_records()
    bad = copy.deepcopy(records)
    if attack == "edit":
        bad[2]["payload"]["top"] = "800-802"
    elif attack == "delete":
        del bad[2]
    else:
        bad[1], bad[2] = bad[2], bad[1]
    out = verify_records(bad)
    assert not out["ok"] and out["first_bad_seq"] in (2, 3, 4)


def test_standalone_verifier_script(tmp_path):
    led = _ledger(tmp_path)
    script = ROOT / "scripts" / "verify_audit.py"
    ok = subprocess.run([sys.executable, str(script), str(led.path), led.head], capture_output=True, text=True)
    assert ok.returncode == 0 and "chain intact" in ok.stdout
    lines = led.path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[3])
    rec["payload"]["top"] = "tampered"
    lines[3] = json.dumps(rec)
    led.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    bad = subprocess.run([sys.executable, str(script), str(led.path)], capture_output=True, text=True)
    assert bad.returncode == 1 and "BROKEN at record 4" in bad.stdout


def test_dispatch_agent_always_finds_the_fault_and_beats_static_ranking_on_average():
    f = ieee34()
    eng = DistFlowEngine(f)
    agent = DispatchAgent(f)
    rng = np.random.default_rng(3)
    agent_min, ranked_min = [], []
    for k in range(40):
        fault = k % f.n_edge
        sc = simulate(f, eng, fault, ObservabilityConfig(meter_dropout=0.8), seed=int(rng.integers(1 << 30)))
        p = rng.dirichlet(np.full(f.n_edge, 0.3))  # an arbitrary, often wrong, starting belief
        ep = agent.run(p, sc, seed=k)
        assert ep.found and ep.found_section == f.edges[fault].name
        assert ep.steps[-1].action == "RESTORE"
        assert all(a.elapsed <= b.elapsed for a, b in zip(ep.steps, ep.steps[1:]))
        agent_min.append(ep.minutes)
        ranked_min.append(agent.run_ranked(p, sc, seed=k).minutes)
    assert np.mean(agent_min) < np.mean(ranked_min)
