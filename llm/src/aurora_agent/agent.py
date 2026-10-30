"""Orchestration: planner → gateway → (client confirmation | second key | execution) → ledger."""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .gateway import DayStats, check
from .ledger import Ledger
from .model import Action, Check, Decision, Kind, Origin, Verdict
from .planner import Planner
from .portfolio import Portfolio
from .rules import RuleBook

SECOND_KEY_TTL = timedelta(minutes=15)


@dataclass
class Outcome:
    action: Action
    decision: Decision
    status: str            # executed | waiting_client | waiting_second_key | denied | rule_applied | rule_pending
    ticket: int | None = None


@dataclass
class Aurora:
    portfolio: Portfolio
    planner: Planner
    rulebook: RuleBook = field(default_factory=RuleBook)
    ledger: Ledger = field(default_factory=Ledger)
    clock: callable = datetime.now
    day: DayStats = field(default_factory=DayStats)
    waiting: dict[int, tuple[Action, str, datetime]] = field(default_factory=dict)
    _ids: itertools.count = field(default_factory=lambda: itertools.count(1))

    # Entry points ---------------------------------------------------------------
    def on_message(self, text: str) -> list[Outcome]:
        """The client wrote something."""
        return self._run(self.planner.plan(text, Origin.CLIENT, self.portfolio), text)

    def on_event(self, text: str) -> list[Outcome]:
        """Something happened outside: salary, news, a post. Its text is data, not a command."""
        return self._run(self.planner.plan(text, Origin.AGENT, self.portfolio), text)

    def submit(self, actions: list[Action], source: str) -> list[Outcome]:
        """Actions that a caller (for example the chat assistant) already built and validated."""
        return self._run(actions, source)

    def confirm(self, ticket: int, yes: bool = True) -> Outcome | None:
        """The client answers a proposal that needed their "yes"."""
        return self._resolve(ticket, "waiting_client", yes, "client_confirmed" if yes else "client_declined")

    def second_key(self, ticket: int, approved: bool) -> Outcome | None:
        """The trusted person answers. Silence for 15 minutes counts as "no"."""
        action, kind, created = self.waiting.get(ticket, (None, None, None))
        if kind == "waiting_second_key" and self.clock() - created > SECOND_KEY_TTL:
            self.waiting.pop(ticket)
            self.ledger.append(self.clock(), "second_key_expired", ticket=ticket)
            return None
        return self._resolve(ticket, "waiting_second_key", approved, "second_key_approved" if approved else "second_key_declined")

    def tick(self) -> None:
        for p in self.rulebook.apply_due(self.clock()):
            self.ledger.append(self.clock(), "rule_change_effective", rule=p.rule, value=str(p.value))

    # Internals ------------------------------------------------------------------
    def _run(self, actions: list[Action], source: str) -> list[Outcome]:
        self.tick()
        out = []
        for a in actions:
            d = check(a, self.portfolio, self.rulebook.rules, self.day)
            self.ledger.append(self.clock(), "proposed", source=source[:200], origin=a.origin.value,
                               action=a.describe(), verdict=d.verdict.value, reasons=d.reasons)
            out.append(self._act(a, d))
        return out

    def _act(self, a: Action, d: Decision) -> Outcome:
        if d.verdict is Verdict.DENY:
            return Outcome(a, d, "denied")
        if d.verdict in (Verdict.ASK_CLIENT, Verdict.SECOND_KEY):
            ticket = next(self._ids)
            kind = "waiting_client" if d.verdict is Verdict.ASK_CLIENT else "waiting_second_key"
            self.waiting[ticket] = (a, kind, self.clock())
            return Outcome(a, d, kind, ticket)
        return self._execute(a, d)

    def _resolve(self, ticket: int, kind: str, yes: bool, event: str) -> Outcome | None:
        if ticket not in self.waiting or self.waiting[ticket][1] != kind:
            return None
        a, _, created = self.waiting[ticket]
        pause = timedelta(minutes=self.rulebook.rules.panic_pause_min)
        if yes and a.sell_all and self.clock() - created < pause:
            left = math.ceil((pause - (self.clock() - created)).total_seconds() / 60)
            d = Decision(Verdict.ASK_CLIENT, (Check("panic_pause", False, f"Пауза ещё идёт, осталось {left} мин"),))
            return Outcome(a, d, "waiting_client", ticket)
        self.waiting.pop(ticket)
        self.ledger.append(self.clock(), event, ticket=ticket, action=a.describe())
        if not yes:
            return Outcome(a, Decision(Verdict.DENY), "denied", ticket)
        # Re-check: the portfolio or the rules may have changed while we waited.
        confirmed = Action(**{**a.__dict__, "origin": Origin.CLIENT, "sell_all": False})
        d = check(confirmed, self.portfolio, self.rulebook.rules, self.day)
        answered = {Verdict.ALLOW, Verdict.ASK_CLIENT if kind == "waiting_client" else Verdict.SECOND_KEY}
        if d.verdict in answered:
            return self._execute(confirmed, d)
        if d.verdict is Verdict.SECOND_KEY:  # the client said yes, now the trusted person must too
            return self._act(confirmed, d)
        return Outcome(a, d, "denied", ticket)

    def _execute(self, a: Action, d: Decision) -> Outcome:
        if a.kind is Kind.CHANGE_RULE:
            result = self.rulebook.request_change(a.rule, a.value, self.clock())
            self.ledger.append(self.clock(), f"rule_{result}", rule=a.rule, value=str(a.value))
            return Outcome(a, d, "rule_applied" if result == "applied" else "rule_pending")
        if a.kind in (Kind.BUY, Kind.SELL):
            self.day.trades += 1
            if a.kind is Kind.SELL:
                self.day.sold_rub += self.portfolio.value_of(a.ticker, a.qty)
        self.portfolio.apply(a)  # production: send the order to the broker and record the fill
        self.ledger.append(self.clock(), "executed", action=a.describe(), cash=str(self.portfolio.cash))
        return Outcome(a, d, "executed")
