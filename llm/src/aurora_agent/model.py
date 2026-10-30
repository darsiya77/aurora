"""Domain objects shared by every layer. Money is Decimal rubles, never float."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum


class Kind(str, Enum):
    BUY = "buy"
    SELL = "sell"
    WITHDRAW = "withdraw"                    # to the client's own account
    TRANSFER_THIRD_PARTY = "transfer_third_party"
    CHANGE_RULE = "change_rule"


RULE_LABEL = {
    "max_issuer_share_pct": "Лимит на одну компанию, %", "big_sale_rub": "Крупные продажи без согласия, ₽",
    "second_key_rub": "Порог подтверждения близкого, ₽", "max_trades_per_day": "Сделок в день",
    "trust_level": "Свобода Авроры", "panic_pause_min": "Пауза при панике, мин",
    "leverage_allowed": "Плечо и маржа", "third_party_transfers_allowed": "Переводы другим людям",
}


class Origin(str, Enum):
    CLIENT = "client"   # the client asked for it in this conversation
    AGENT = "agent"     # Aurora proposed it on her own (salary arrived, news, rebalancing)


class Verdict(str, Enum):
    ALLOW = "allow"
    ASK_CLIENT = "ask_client"
    SECOND_KEY = "second_key"
    DENY = "deny"


@dataclass(frozen=True)
class Action:
    kind: Kind
    origin: Origin
    reason: str = ""
    ticker: str | None = None
    qty: int = 0
    amount: Decimal = Decimal(0)
    rule: str | None = None
    value: Decimal | int | None = None
    sell_all: bool = False

    def describe(self) -> str:
        if self.kind in (Kind.BUY, Kind.SELL):
            verb = "Купить" if self.kind is Kind.BUY else "Продать"
            return f"{verb} {self.ticker} × {self.qty}" + (" (всё)" if self.sell_all else "")
        if self.kind is Kind.WITHDRAW:
            return f"Вывести {self.amount:,.0f} ₽ на свою карту".replace(",", " ")
        if self.kind is Kind.TRANSFER_THIRD_PARTY:
            return f"Перевести {self.amount:,.0f} ₽ другому человеку".replace(",", " ")
        return f"{RULE_LABEL.get(self.rule, self.rule)}: {self.value}"


@dataclass(frozen=True)
class Check:
    rule: str
    passed: bool
    message: str


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    checks: tuple[Check, ...] = field(default_factory=tuple)

    @property
    def reasons(self) -> list[str]:
        return [c.message for c in self.checks if not c.passed] or [c.message for c in self.checks]
