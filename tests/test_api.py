"""HTTP contract of the web control room's API."""
import pytest
from fastapi.testclient import TestClient

from server.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_demo_scenario_matches_the_pitch(client):
    d = client.get("/api/scenario", params={"feeder": "ieee34"}).json()
    ens = d["models"]["ensemble"]
    assert ens["display"] == "Line 854 → 852" and ens["status"] == "exact"
    assert not d["baseline"]["correct"] and d["baseline"]["hops"] >= 2
    assert ens["patrol_km"] < d["conventional"]["km"]
    assert abs(sum(d["probs"]["ensemble"]) - 1.0) < 1e-6
    assert {e["tag"] for e in ens["evidence"]} >= {"PHYSICS", "SCADA", "METERS"}
    assert len(d["nodes"]) == 34 and len(d["edges"]) == 33


def test_every_section_and_model_is_served(client):
    for feeder, n_edge in (("ieee13", 12), ("ieee34", 33)):
        for fault in (0, n_edge - 1):
            d = client.get("/api/scenario", params={"feeder": feeder, "fault": fault, "seed": 3,
                                                    "dropout": 0.95, "placement": "random"}).json()
            assert d["truth"]["edge"] == fault
            for m in ("ensemble", "bayes", "gnn"):
                assert len(d["models"][m]["candidates"]) == 5


def test_invalid_input_is_rejected(client):
    assert client.get("/api/scenario", params={"feeder": "ieee13", "fault": 99}).status_code == 422
    assert client.get("/api/scenario", params={"dropout": 1.5}).status_code == 422
    assert client.get("/api/scenario", params={"feeder": "ieee999"}).status_code == 422


def test_benchmark_csv_and_static_site(client):
    b = client.get("/api/benchmark", params={"feeder": "ieee13"}).json()
    assert b["n_scenarios"] == 280 and b["methods"][0].startswith("OutageLens")
    head = {h["method"]: h for h in b["headline"]}
    assert head[b["methods"][0]]["top1"] > head[b["baseline_method"]]["top1"]
    csv = client.get("/api/telemetry.csv", params={"feeder": "ieee13"})
    assert csv.status_code == 200 and csv.text.splitlines()[1].startswith("bus,")
    page = client.get("/")
    assert page.status_code == 200 and "OutageLens" in page.text
