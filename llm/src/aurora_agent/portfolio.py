"""Client portfolio and market prices. In production this reads the broker's API."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from .model import Action, Kind


@dataclass(frozen=True)
class Instrument:
    ticker: str
    name: str
    kind: str            # stock | bond | fund
    price: Decimal
    issuer_limited: bool = True   # funds and state bonds spread or carry no single-company risk


@dataclass
class Portfolio:
    cash: Decimal
    positions: dict[str, int] = field(default_factory=dict)
    market: dict[str, Instrument] = field(default_factory=dict)

    def value_of(self, ticker: str, qty: int | None = None) -> Decimal:
        q = self.positions.get(ticker, 0) if qty is None else qty
        return self.market[ticker].price * q

    def total(self) -> Decimal:
        return self.cash + sum(self.value_of(t) for t in self.positions)

    def apply(self, a: Action) -> None:
        if a.kind is Kind.BUY:
            self.cash -= self.value_of(a.ticker, a.qty)
            self.positions[a.ticker] = self.positions.get(a.ticker, 0) + a.qty
        elif a.kind is Kind.SELL:
            self.cash += self.value_of(a.ticker, a.qty)
            self.positions[a.ticker] -= a.qty
            if not self.positions[a.ticker]:
                del self.positions[a.ticker]
        elif a.kind is Kind.WITHDRAW:
            self.cash -= a.amount


def demo_portfolio() -> Portfolio:
    market = {i.ticker: i for i in [
        Instrument("SBER", "Сбербанк", "stock", Decimal("312.45")),
        Instrument("YDEX", "Яндекс", "stock", Decimal("4260")),
        Instrument("OFZ26238", "ОФЗ 26238", "bond", Decimal("650"), issuer_limited=False),  # state bonds
        Instrument("TMOS", "Фонд на индекс Мосбиржи", "fund", Decimal("8.75"), issuer_limited=False),
    ]}
    return Portfolio(cash=Decimal("69559"), positions={"SBER": 150, "OFZ26238": 80, "TMOS": 11200}, market=market)
