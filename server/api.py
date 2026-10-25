"""OutageLens HTTP API and web host.

    python -m uvicorn server.api:app --port 8000      (or: python run.py)

JSON lives under /api; everything else is the static control room in ``web/``.
"""
from __future__ import annotations

import json
import threading
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import service

WEB_DIR = service.ROOT / "web"
FeederKey = Literal["ieee13", "ieee34"]


def _warm() -> None:
    # Build both feeders (engine cross-check, calibration, GNN) before the first click needs them.
    for key in service.FEEDERS:
        service.system(key)
        service.benchmark(key)
        service.dispatch_study(key)


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_warm, name="outagelens-warmup", daemon=True).start()
    yield


app = FastAPI(title="OutageLens API", version="1.0", lifespan=lifespan)


@app.middleware("http")
async def revalidate_static(request: Request, call_next):
    # A redeploy must never leave a judge's browser running yesterday's JavaScript.
    response = await call_next(request)
    if not request.url.path.startswith("/api/") and "/fonts/" not in request.url.path:
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "ready": sorted(service._systems)}


@app.get("/api/feeders")
def feeders() -> list[dict]:
    return service.feeders()


def _scenario(feeder, fault, seed, dropout, coverage, glitch, placement) -> dict:
    try:
        return service.scenario(feeder, fault, seed, dropout, coverage, glitch, placement)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/scenario")
def scenario(feeder: FeederKey = "ieee34", fault: int | None = Query(None, ge=0), seed: int | None = Query(None, ge=0),
             dropout: float = Query(0.65, ge=0.0, le=0.95), coverage: float = Query(0.25, ge=0.05, le=1.0),
             glitch: float = Query(0.02, ge=0.0, le=0.10),
             placement: Literal["optimised", "random"] = "optimised") -> dict:
    return _scenario(feeder, fault, seed, dropout, coverage, glitch, placement)


@app.get("/api/telemetry.csv", response_class=PlainTextResponse)
def telemetry(feeder: FeederKey = "ieee34", fault: int | None = Query(None, ge=0),
              seed: int | None = Query(None, ge=0), dropout: float = Query(0.65, ge=0.0, le=0.95),
              coverage: float = Query(0.25, ge=0.05, le=1.0), glitch: float = Query(0.02, ge=0.0, le=0.10),
              placement: Literal["optimised", "random"] = "optimised") -> PlainTextResponse:
    payload = _scenario(feeder, fault, seed, dropout, coverage, glitch, placement)
    name = f"outagelens_{feeder}_seed{payload['scenario']['seed']}.csv"
    return PlainTextResponse(service.telemetry_csv(payload), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.post("/api/dispatch")
def dispatch(feeder: FeederKey = "ieee34", fault: int | None = Query(None, ge=0), seed: int | None = Query(None, ge=0),
             dropout: float = Query(0.65, ge=0.0, le=0.95), coverage: float = Query(0.25, ge=0.05, le=1.0),
             glitch: float = Query(0.02, ge=0.0, le=0.10), placement: Literal["optimised", "random"] = "optimised",
             crew_seed: int = Query(1, ge=0)) -> dict:
    try:
        return service.dispatch(feeder, fault, seed, dropout, coverage, glitch, placement, crew_seed)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/dispatch/study")
def dispatch_study(feeder: FeederKey = "ieee34") -> dict:
    return service.dispatch_study(feeder)


@app.get("/api/audit")
def audit(limit: int = Query(25, ge=1, le=200)) -> dict:
    led = service.LEDGER
    return {"count": led.count, "head": led.head, "genesis": "0" * 64, "records": led.recent(limit),
            "persisted": led.path is not None, "recovered_from": led.recovered_from}


@app.get("/api/audit/verify")
def audit_verify() -> dict:
    return service.LEDGER.verify()


@app.get("/api/audit.jsonl", response_class=PlainTextResponse)
def audit_download() -> PlainTextResponse:
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in service.LEDGER.all_records())
    return PlainTextResponse(body, media_type="application/x-ndjson",
                             headers={"Content-Disposition": 'attachment; filename="outagelens_decisions.jsonl"'})


@app.get("/api/benchmark")
def benchmark(feeder: FeederKey = "ieee34", coverage: float = Query(0.25, ge=0.05, le=1.0),
              glitch: float = Query(0.02, ge=0.0, le=0.10)) -> dict:
    return service.benchmark(feeder, round(coverage, 2), round(glitch, 2))


@app.post("/api/benchmark/run")
def benchmark_run(feeder: FeederKey = "ieee34", coverage: float = Query(0.25, ge=0.05, le=1.0),
                  glitch: float = Query(0.02, ge=0.0, le=0.10)) -> dict:
    return service.run_full_benchmark(feeder, round(coverage, 2), round(glitch, 2))


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
