"""Independently verify an OutageLens decision log (no OutageLens code needed, standard library only).

    python scripts/verify_audit.py decisions.jsonl [expected_head_hash]

Download the log from the app (Audit log screen) or /api/audit.jsonl. Exit code 0 = chain intact.
"""
from __future__ import annotations

import hashlib
import json
import sys

GENESIS = "0" * 64
FIELDS = ("seq", "time", "kind", "payload", "prev_hash")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    with open(sys.argv[1], encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    prev = GENESIS
    for i, rec in enumerate(records):
        body = json.dumps({k: rec[k] for k in FIELDS}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if rec["seq"] != i + 1 or rec["prev_hash"] != prev or digest != rec["hash"]:
            print(f"BROKEN at record {rec.get('seq', i + 1)}: the log was altered from this point on.")
            return 1
        prev = rec["hash"]
    print(f"OK: {len(records)} records, chain intact. Head hash: {prev}")
    if len(sys.argv) > 2 and sys.argv[2] != prev:
        print("WARNING: head hash differs from the one you expected (records may be missing at the end).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
