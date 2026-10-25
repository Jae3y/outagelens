"""Audit-verifiable decision log: an append-only SHA-256 hash chain.

Every decision OutageLens makes (a fault localisation, a dispatch plan) becomes one record:

    {"seq", "time", "kind", "payload", "prev_hash", "hash"}

``hash`` = SHA-256 over the canonical JSON of every other field, and ``prev_hash`` is the previous
record's ``hash`` (64 zeros for the first). Editing, deleting, inserting or reordering any record breaks
every hash after it, so anyone holding the file can prove it is intact (``verify`` here, or the
standalone ``scripts/verify_audit.py``). Payloads hold digests of the observed telemetry, model
fingerprints and outputs; they never hold the simulated ground truth, which a real utility would not have.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

GENESIS = "0" * 64
FIELDS = ("seq", "time", "kind", "payload", "prev_hash")


def canonical(obj) -> bytes:
    """Deterministic JSON bytes: sorted keys, no whitespace, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record_hash(record: dict) -> str:
    return sha256_hex(canonical({k: record[k] for k in FIELDS}))


def verify_records(records: list[dict]) -> dict:
    """Check a full chain from the start. Returns ok, count, head and the first broken seq if any."""
    prev = GENESIS
    for i, rec in enumerate(records):
        problem = None
        if any(k not in rec for k in (*FIELDS, "hash")):
            problem = "missing field"
        elif rec["seq"] != i + 1:
            problem = f"expected seq {i + 1}"
        elif rec["prev_hash"] != prev:
            problem = "prev_hash does not match the previous record"
        elif record_hash(rec) != rec["hash"]:
            problem = "content does not match its hash"
        if problem:
            return {"ok": False, "count": len(records), "first_bad_seq": rec.get("seq", i + 1), "problem": problem,
                    "head": prev}
        prev = rec["hash"]
    return {"ok": True, "count": len(records), "first_bad_seq": None, "problem": None, "head": prev}


class AuditLedger:
    """Thread-safe append-only chain, persisted as JSON Lines (one record per line)."""

    def __init__(self, path: Path | None = None, keep: int = 500):
        self.path = path
        self._lock = threading.Lock()
        self._recent: deque[dict] = deque(maxlen=keep)
        self._count = 0
        self._head = GENESIS
        self.recovered_from: str | None = None
        if path is not None and path.exists():
            self._load()

    def _load(self) -> None:
        records = []
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
        except (OSError, ValueError):
            records = None
        status = verify_records(records) if records is not None else {"ok": False}
        if not status["ok"]:
            # Never silently extend a broken chain: set it aside and start a fresh one.
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            broken = self.path.with_name(f"{self.path.stem}.broken-{stamp}{self.path.suffix}")
            self.path.rename(broken)
            self.recovered_from = broken.name
            return
        self._count = len(records)
        self._head = status["head"]
        self._recent.extend(records)

    def append(self, kind: str, payload: dict) -> dict:
        with self._lock:
            rec = {"seq": self._count + 1, "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                   "kind": kind, "payload": payload, "prev_hash": self._head}
            rec["hash"] = record_hash(rec)
            if self.path is not None:
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with self.path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except OSError:
                    pass  # read-only disk: the in-memory chain still verifies
            self._count += 1
            self._head = rec["hash"]
            self._recent.append(rec)
            return rec

    @property
    def count(self) -> int:
        return self._count

    @property
    def head(self) -> str:
        return self._head

    def recent(self, n: int = 25) -> list[dict]:
        with self._lock:
            return list(self._recent)[-n:][::-1]

    def all_records(self) -> list[dict]:
        """Full chain from disk when persisted (survives the in-memory window), else what is in memory."""
        with self._lock:
            if self.path is not None and self.path.exists():
                return [json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines() if x.strip()]
            return list(self._recent)

    def verify(self) -> dict:
        t = time.perf_counter()
        records = self.all_records()
        out = verify_records(records)
        if self.path is None and self._count > len(records):
            out["note"] = "only the most recent records are held in memory"
        out["checked_ms"] = round((time.perf_counter() - t) * 1000, 1)
        out["matches_live_head"] = out["ok"] and out["head"] == self._head
        return out
