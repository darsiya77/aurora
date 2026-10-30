"""Tools the model can call. Each call becomes an aurora_agent Action and goes through the gateway,
so the model can ask for anything but can only do what the client's rules allow."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from aurora_agent.agent import Aurora
from aurora_agent.model import Action, Kind, Origin
from aurora_agent.rules import Rules

# JSON-schema style descriptions, shown to the model in the system prompt.
TOOLS = [
    {"name": "get_portfolio", "description": "Показать портфель: бумаги, их стоимость и свободные деньги.",
     "parameters": {}},
    {"name": "get_quote", "description": "Узнать текущую цену бумаги.",
     "parameters": {"ticker": "тикер, например SBER"}},
    {"name": "buy", "description": "Купить бумаги.",
     "parameters": {"ticker": "тикер", "qty": "количество, целое число"}},
    {"name": "sell", "description": "Продать бумаги. all=true — продать всю позицию.",
     "parameters": {"ticker": "тикер", "qty": "количество, целое число", "all": "true или false"}},
    {"name": "withdraw", "description": "Вывести деньги на карту самого клиента.",
     "parameters": {"amount": "сумма в рублях"}},
    {"name": "transfer", "description": "Перевести деньги другому человеку.",
     "parameters": {"amount": "сумма в рублях", "recipient": "кому"}},
    {"name": "change_rule", "description": "Изменить правило клиента: " + ", ".join(
        f"{k}" for k in ("max_issuer_share_pct", "big_sale_rub", "second_key_rub", "max_trades_per_day", "trust_level")),
     "parameters": {"rule": "имя правила", "value": "новое значение"}},
]
TOOL_NAMES = {t["name"] for t in TOOLS}
CHANGEABLE_RULES = {"max_issuer_share_pct", "big_sale_rub", "second_key_rub", "max_trades_per_day", "trust_level"}


class ToolError(ValueError):
    """The call is malformed; the message goes back to the model so it can fix the call."""


@dataclass
class ToolCall:
    name: str
    arguments: dict

    def to_dict(self) -> dict:
        return {"name": self.name, "arguments": self.arguments}


def _decimal(x, field: str) -> Decimal:
    try:
        d = Decimal(str(x).replace(" ", "").replace(",", "."))
    except InvalidOperation:
        raise ToolError(f"{field}: ожидается число, получено {x!r}") from None
    if d <= 0:
        raise ToolError(f"{field}: должно быть больше нуля")
    return d


def _qty(x) -> int:
    d = _decimal(x, "qty")
    if d != d.to_integral_value():
        raise ToolError("qty: должно быть целым числом")
    return int(d)


def to_action(call: ToolCall, aurora: Aurora) -> Action:
    """Validate arguments and build the action. The client asked in this chat, so origin is CLIENT."""
    a, market = call.arguments, aurora.portfolio.market
    if call.name in ("buy", "sell"):
        ticker = str(a.get("ticker", "")).upper()
        if ticker not in market:
            raise ToolError(f"Бумага {ticker or '—'} недоступна. Доступны: {', '.join(market)}")
        sell_all = call.name == "sell" and str(a.get("all", "")).lower() in ("true", "1", "да")
        qty = aurora.portfolio.positions.get(ticker, 0) if sell_all else _qty(a.get("qty"))
        kind = Kind.BUY if call.name == "buy" else Kind.SELL
        return Action(kind, Origin.CLIENT, "запрос клиента в чате", ticker=ticker, qty=qty, sell_all=sell_all)
    if call.name == "withdraw":
        return Action(Kind.WITHDRAW, Origin.CLIENT, "запрос клиента в чате", amount=_decimal(a.get("amount"), "amount"))
    if call.name == "transfer":
        return Action(Kind.TRANSFER_THIRD_PARTY, Origin.CLIENT, f"получатель: {a.get('recipient', '—')}",
                      amount=_decimal(a.get("amount"), "amount"))
    if call.name == "change_rule":
        rule = a.get("rule")
        if rule not in CHANGEABLE_RULES:
            raise ToolError(f"Правило {rule!r} нельзя изменить. Можно: {', '.join(sorted(CHANGEABLE_RULES))}")
        value = _decimal(a.get("value"), "value")
        typed = int(value) if isinstance(getattr(Rules(), rule), int) else value
        return Action(Kind.CHANGE_RULE, Origin.CLIENT, "запрос клиента в чате", rule=rule, value=typed)
    raise ToolError(f"Неизвестный инструмент {call.name!r}")


STATUS_TEXT = {"executed": "исполнено", "waiting_client": "ждёт подтверждения клиента",
               "waiting_second_key": "ждёт подтверждения доверенного человека", "denied": "запрещено правилами",
               "rule_applied": "правило вступило сразу", "rule_pending": "правило вступит через 24 часа"}


def execute(call: ToolCall, aurora: Aurora) -> dict:
    """Run one call and return a JSON-friendly result for the model to read."""
    p = aurora.portfolio
    try:
        if call.name == "get_portfolio":
            return {"cash_rub": str(p.cash), "total_rub": str(p.total()),
                    "positions": [{"ticker": t, "name": p.market[t].name, "qty": q, "value_rub": str(p.value_of(t))}
                                  for t, q in p.positions.items()]}
        if call.name == "get_quote":
            t = str(call.arguments.get("ticker", "")).upper()
            if t not in p.market:
                raise ToolError(f"Бумага {t or '—'} недоступна. Доступны: {', '.join(p.market)}")
            return {"ticker": t, "name": p.market[t].name, "price_rub": str(p.market[t].price)}
        outcome = aurora.submit([to_action(call, aurora)], f"чат: {call.name}")[0]
    except ToolError as e:
        return {"error": str(e)}
    act = outcome.action
    details = {"kind": act.kind.value, "ticker": act.ticker, "name": p.market[act.ticker].name if act.ticker else None,
               "qty": act.qty or None, "amount_rub": str(act.amount) if act.amount else None, "rule": act.rule,
               "value": str(act.value) if act.value is not None else None}
    return {"status": outcome.status, "status_text": STATUS_TEXT[outcome.status], "action": act.describe(),
            **{k: v for k, v in details.items() if v is not None},
            "checks": [{"ok": c.passed, "text": c.message} for c in outcome.decision.checks],
            **({"ticket": outcome.ticket} if outcome.ticket else {})}
