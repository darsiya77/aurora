"""The gateway ("Проверка правил"): the only path from a proposal to the client's money.

Pure deterministic code: the same inputs always give the same verdict, so it can be tested,
audited and explained to a regulator. The language model never sits on this path.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .model import RULE_LABEL, Action, Check, Decision, Kind, Origin, Verdict
from .portfolio import Portfolio
from .rules import LOCKED, Rules

# Verdicts ordered from most to least permissive; the strictest triggered check wins.
_SEVERITY = [Verdict.ALLOW, Verdict.ASK_CLIENT, Verdict.SECOND_KEY, Verdict.DENY]


@dataclass
class DayStats:
    trades: int = 0
    sold_rub: Decimal = Decimal(0)


def rub(x: Decimal) -> str:
    return f"{x:,.0f} ₽".replace(",", " ")


def check(a: Action, p: Portfolio, r: Rules, day: DayStats) -> Decision:
    out: list[tuple[Verdict, Check]] = []

    def add(ok: bool, rule: str, msg: str, verdict_if_failed: Verdict, ok_msg: str | None = None):
        out.append((Verdict.ALLOW if ok else verdict_if_failed, Check(rule, ok, ok_msg if ok and ok_msg else msg)))

    if a.kind is Kind.TRANSFER_THIRD_PARTY:
        add(r.third_party_transfers_allowed, "third_party_transfers", "Переводы другим людям запрещены всегда", Verdict.DENY)

    elif a.kind is Kind.CHANGE_RULE:
        add(a.origin is Origin.CLIENT, "rules_by_client_only", "Правила меняет только сам клиент", Verdict.DENY,
            "Изменение просит сам клиент")
        add(a.rule not in LOCKED, "locked_rule", f"«{RULE_LABEL.get(a.rule, a.rule)}» не меняется никогда", Verdict.DENY,
            "Строже — сразу, мягче — через 24 часа")

    elif a.kind is Kind.WITHDRAW:
        add(a.origin is Origin.CLIENT, "withdraw_by_client_only", "Вывод денег начинает только клиент", Verdict.DENY,
            "Вывод начал сам клиент")
        add(a.amount <= p.cash, "enough_cash", f"Свободно {rub(p.cash)}", Verdict.DENY)
        add(a.amount < r.second_key_rub, "second_key",
            f"Вывод от {rub(r.second_key_rub)} подтверждает доверенный человек", Verdict.SECOND_KEY,
            f"Сумма ниже порога подтверждения {rub(r.second_key_rub)}")

    elif a.kind is Kind.BUY:
        inst = p.market.get(a.ticker)
        if inst is None:
            add(False, "known_instrument", f"Бумага {a.ticker} недоступна", Verdict.DENY)
        else:
            cost = inst.price * a.qty
            add(cost <= p.cash, "no_leverage", f"Покупка на {rub(cost)}, свободно {rub(p.cash)}, без плеча", Verdict.DENY)
            if inst.issuer_limited:
                share = (p.value_of(a.ticker) + cost) / p.total() * 100
                add(share <= r.max_issuer_share_pct, "issuer_limit",
                    f"Доля {a.ticker} после сделки {share:.1f}% из лимита {r.max_issuer_share_pct}%", Verdict.DENY)
            add(day.trades < r.max_trades_per_day, "trades_per_day", f"Сделок сегодня {day.trades} из {r.max_trades_per_day}", Verdict.DENY)
            _trust(add, a, r)

    elif a.kind is Kind.SELL:
        held = p.positions.get(a.ticker, 0)
        add(0 < a.qty <= held, "enough_position", f"В портфеле {held} шт. {a.ticker}", Verdict.DENY)
        if held:
            amount = p.value_of(a.ticker, a.qty)
            add(day.sold_rub + amount <= r.big_sale_rub, "big_sale",
                f"Продажи за день {rub(day.sold_rub + amount)}, без согласия до {rub(r.big_sale_rub)}", Verdict.ASK_CLIENT)
        add(not a.sell_all, "panic_pause", f"«Продать всё» — сначала факты и пауза {r.panic_pause_min} минут", Verdict.ASK_CLIENT)
        add(day.trades < r.max_trades_per_day, "trades_per_day", f"Сделок сегодня {day.trades} из {r.max_trades_per_day}", Verdict.DENY)
        _trust(add, a, r)

    verdict = max((v for v, _ in out), key=_SEVERITY.index, default=Verdict.DENY)
    return Decision(verdict, tuple(c for _, c in out))


def _trust(add, a: Action, r: Rules):
    """How much Aurora may do on her own. A client's explicit request counts as consent."""
    if a.origin is Origin.CLIENT:
        add(True, "trust_level", "Сделку попросил сам клиент", Verdict.ALLOW)
    elif r.trust_level >= 3:
        add(True, "trust_level", "Автопилот: в рамках правил Аврора действует сама", Verdict.ALLOW)
    else:
        add(False, "trust_level", "Предложение Авроры ждёт «да» клиента", Verdict.ASK_CLIENT)
