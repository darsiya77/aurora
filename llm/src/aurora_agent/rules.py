"""The client's rules ("Мои правила") and the asymmetry that protects them.

Stricter values take effect at once. Looser values wait 24 hours and can be cancelled meanwhile,
which breaks the "do it now, before you change your mind" pattern used by fraudsters.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta
from decimal import Decimal

LOOSEN_DELAY = timedelta(hours=24)


@dataclass
class Rules:
    max_issuer_share_pct: Decimal = Decimal(20)       # no more than this share of the portfolio in one issuer
    big_sale_rub: Decimal = Decimal(100_000)           # sales above this per day need the client's "yes"
    second_key_rub: Decimal = Decimal(50_000)          # withdrawals from this amount need a trusted person
    max_trades_per_day: int = 5
    trust_level: int = 2                               # 1 advice only, 2 with confirmation, 3 autopilot
    panic_pause_min: int = 10                          # "sell everything" waits this long
    # Fixed for every client of the prototype; changing them is never allowed.
    leverage_allowed: bool = False
    third_party_transfers_allowed: bool = False


LOCKED = {"leverage_allowed", "third_party_transfers_allowed"}
# For each rule: True if a larger value is stricter.
LARGER_IS_STRICTER = {
    "max_issuer_share_pct": False, "big_sale_rub": False, "second_key_rub": False,
    "max_trades_per_day": False, "trust_level": False, "panic_pause_min": True,
}


@dataclass(frozen=True)
class Pending:
    rule: str
    value: Decimal | int
    effective_at: datetime


@dataclass
class RuleBook:
    rules: Rules = field(default_factory=Rules)
    pending: list[Pending] = field(default_factory=list)

    def is_stricter(self, rule: str, value) -> bool:
        current = getattr(self.rules, rule)
        return value > current if LARGER_IS_STRICTER[rule] else value < current

    def request_change(self, rule: str, value, now: datetime) -> str:
        """Returns 'applied', 'pending' or 'unchanged'. Raises for unknown or locked rules."""
        if rule not in {f.name for f in fields(Rules)}:
            raise KeyError(f"Unknown rule: {rule}")
        if rule in LOCKED:
            raise PermissionError(f"Rule {rule} cannot be changed")
        value = type(getattr(self.rules, rule))(value)
        if value == getattr(self.rules, rule):
            return "unchanged"
        self.pending = [p for p in self.pending if p.rule != rule]
        if self.is_stricter(rule, value):
            setattr(self.rules, rule, value)
            return "applied"
        self.pending.append(Pending(rule, value, now + LOOSEN_DELAY))
        return "pending"

    def cancel(self, rule: str) -> bool:
        before = len(self.pending)
        self.pending = [p for p in self.pending if p.rule != rule]
        return len(self.pending) < before

    def apply_due(self, now: datetime) -> list[Pending]:
        due = [p for p in self.pending if p.effective_at <= now]
        for p in due:
            setattr(self.rules, p.rule, p.value)
        self.pending = [p for p in self.pending if p.effective_at > now]
        return due
