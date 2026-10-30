"""Append-only journal. Every record carries the hash of the previous one, so changing any
record breaks the chain from that point on. The client receives each record's hash at once,
so even the bank cannot quietly rewrite history."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime

GENESIS = "0" * 64


def _digest(prev: str, body: dict) -> str:
    raw = prev + json.dumps(body, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class Ledger:
    records: list[dict] = field(default_factory=list)

    def append(self, at: datetime, event: str, **data) -> dict:
        prev = self.records[-1]["hash"] if self.records else GENESIS
        body = {"n": len(self.records), "at": at.isoformat(), "event": event, **data}
        rec = {**body, "prev": prev, "hash": _digest(prev, body)}
        self.records.append(rec)
        return rec

    def verify(self) -> int | None:
        """Index of the first broken record, or None if the chain is intact."""
        prev = GENESIS
        for i, rec in enumerate(self.records):
            body = {k: v for k, v in rec.items() if k not in ("prev", "hash")}
            if rec["prev"] != prev or rec["hash"] != _digest(prev, body):
                return i
            prev = rec["hash"]
        return None
