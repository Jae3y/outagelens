"""Keep test runs out of the real decision log: the service reads this before it builds its ledger."""
import os

os.environ.setdefault("OUTAGELENS_AUDIT", "memory")
