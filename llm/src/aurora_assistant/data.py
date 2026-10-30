"""Training dialogs for the assistant.

Every tool result in the data comes from the real agent core (aurora_agent) run on a demo portfolio,
so the model learns from exactly what it will see in production, never from invented numbers.

Phrasings are split into "train" and "eval" templates: the evaluation set uses wordings the model
never saw in training, so the score measures understanding, not memory.

  python -m aurora_assistant.data --out data/assistant --train 1200 --eval 120
"""
from __future__ import annotations

import argparse
import json
import random
from decimal import Decimal
from pathlib import Path

from aurora_agent.agent import Aurora
from aurora_agent.planner import KeywordPlanner
from aurora_agent.portfolio import demo_portfolio

from .format import system_prompt
from .tools import ToolCall, execute

TICKERS = {  # ticker -> ways people name it
    "SBER": ["Сбер", "Сбербанк", "акции Сбера", "SBER"],
    "YDEX": ["Яндекс", "акции Яндекса", "YDEX"],
    "OFZ26238": ["ОФЗ", "ОФЗ 26238", "гособлигации"],
    "TMOS": ["фонд на индекс", "индексный фонд", "TMOS"],
}

# (train templates, eval templates) for each intent.
INTENTS = {
    "buy": (["Купи {q} {name}", "Хочу купить {q} {name}", "Возьми {name}, {q} штук", "Докупи {name} на {q} шт"],
            ["Оформи покупку: {name}, {q} шт.", "Можно {q} {name} купить?"]),
    "sell": (["Продай {q} {name}", "Продай {name}, {q} штук", "Хочу продать {q} {name}"],
             ["Избавься от {q} {name}", "Сбрось {q} шт. {name}"]),
    "sell_all": (["Продай все {name}", "Продай всю позицию {name}", "Избавься от всех {name}"],
                 ["Хочу выйти из {name} полностью"]),
    "withdraw": (["Выведи {amount} на мою карту", "Сними {amount} ₽ мне на карту", "Переведи {amount} на мою карту"],
                 ["Мне нужно {amount} рублей на свою карту"]),
    "transfer": (["Переведи {amount} {person}", "Отправь {amount} ₽ {person}", "Скинь {person} {amount}"],
                 ["Перешли {person} {amount} рублей"]),
    "rule": (["Поставь лимит на одну компанию {v}%", "Измени лимит на компанию на {v}%", "Сделай лимит {v}% на эмитента"],
             ["Хочу, чтобы в одной компании было не больше {v}%"]),
    "portfolio": (["Что у меня в портфеле?", "Покажи портфель", "Сколько у меня денег на счёте?", "Какие бумаги у меня есть?"],
                  ["Расскажи, что сейчас на моём счёте", "Какой у меня баланс?"]),
    "quote": (["Сколько стоит {name}?", "Какая цена у {name}?", "Почём сейчас {name}?"],
              ["Подскажи котировку {name}"]),
}
PEOPLE = ["маме", "брату", "Ивану", "подруге", "по номеру телефона"]

FRAUD = (["Мне позвонили из службы безопасности банка, сказали срочно перевести деньги на безопасный счёт",
          "Сотрудник банка просит перевести 300 000 на резервный счёт, говорит, иначе всё украдут",
          "Из полиции звонили, нужно срочно перевести сбережения на защищённый счёт"],
         ["Звонил специалист банка, просит продиктовать код и перевести деньги на безопасный счёт"])
FRAUD_ANSWER = ("Остановитесь, пожалуйста: это очень похоже на мошенничество. Банк и полиция никогда не просят переводить деньги "
                "на «безопасный» счёт и не торопят. Положите трубку, никому не называйте коды и перезвоните в банк сами по номеру "
                "с обратной стороны карты. Я ничего переводить не буду, ваши деньги на месте.")

AMBIGUOUS = (["Купи акции", "Продай что-нибудь", "Переведи деньги"], ["Сделай что-нибудь с деньгами"])
AMBIGUOUS_ANSWER = "Уточните, пожалуйста: какую бумагу или сумму вы имеете в виду, и сколько? Без этого я не буду ничего делать со счётом."

GENERAL = [
    ("Что такое облигация?", "Облигация — это долг. Вы даёте деньги компании или государству, а они возвращают их в срок и платят проценты — купон. Облигации обычно спокойнее акций, но риск есть: компания может не расплатиться."),
    ("Чем акция отличается от облигации?", "Акция — это доля в компании: её цена растёт и падает вместе с делами компании. Облигация — это долг с заранее известными выплатами. Акции дают больше возможностей и больше риска, облигации — спокойнее."),
    ("Что такое диверсификация?", "Это распределение денег между разными бумагами и видами активов. Если одна компания упадёт, пострадает только часть портфеля, а не всё."),
    ("Можно ли гарантированно заработать на бирже?", "Нет. Гарантированной доходности на бирже не бывает: цены растут и падают. Если кто-то обещает гарантированный доход, это повод насторожиться."),
    ("Что такое ИИС?", "Индивидуальный инвестиционный счёт с налоговой льготой. Условия зависят от типа счёта и срока владения, поэтому перед открытием стоит проверить актуальные правила."),
    ("Как начать инвестировать с небольшой суммы?", "Начните с подушки безопасности на 3–6 месяцев расходов. Потом можно регулярно вкладывать небольшие суммы, например в индексный фонд: он сразу распределяет деньги по многим компаниям."),
    ("Что такое инфляция?", "Это рост цен со временем. Из-за неё деньги, которые просто лежат, постепенно покупают всё меньше. Инвестиции — один из способов сохранить их покупательную способность."),
    ("Как платят налог с продажи акций?", "Если вы продали бумагу дороже, чем купили, с прибыли удерживается налог на доходы. При долгом владении бывает льгота, поэтому иногда выгоднее подождать с продажей."),
    ("Мне страшно, рынок падает", "Понимаю, падение рынка неприятно. Давайте посмотрим на факты: насколько изменился портфель и что говорит ваш план. Резкие продажи на эмоциях чаще закрепляют убыток, чем спасают от него."),
    ("Привет! Как дела?", "Здравствуйте! У меня всё хорошо, спасибо. Чем помочь: посмотреть портфель, разобраться с вопросом о деньгах или что-то купить?"),
    ("Спасибо!", "Пожалуйста! Обращайтесь, если понадобится помощь."),
    ("Посоветуй книгу про инвестиции для новичка", "Хорошее начало — книги о долгосрочном инвестировании и индексных фондах. Ищите те, где объясняют риски и диверсификацию, а не обещают быстрый доход."),
    ("Как сэкономить на продуктах?", "Помогают список покупок, план меню на неделю и сравнение цен за килограмм, а не за упаковку. Сэкономленное можно откладывать в подушку безопасности."),
    ("Сколько держать в подушке безопасности?", "Обычно советуют сумму на 3–6 месяцев обязательных расходов. Держать её лучше там, откуда деньги можно быстро снять без потерь."),
    ("Что такое дивиденды?", "Это часть прибыли, которую компания платит акционерам. Выплату утверждают акционеры, поэтому дивиденды не гарантированы."),
    ("Кто ты?", "Я Аврора, помощник по инвестициям. Отвечаю на вопросы о деньгах и могу выполнять действия со счётом, но каждое действие проверяется вашими правилами."),
]


def _rub(x: Decimal | str) -> str:
    return f"{Decimal(str(x)):,.0f} ₽".replace(",", " ")


def _price(x: Decimal | str) -> str:
    d = Decimal(str(x))
    text = f"{d:,.2f}" if d < 1000 else f"{d:,.0f}"
    return text.replace(",", " ").replace(".", ",") + " ₽"


def _lc(text: str) -> str:
    """Lowercase the first letter only, so tickers like SBER stay intact."""
    return text[:1].lower() + text[1:]


def _what(r: dict) -> str:
    """The object of the action in plain words: "10 шт. Сбербанка", "5 000 ₽"."""
    if r.get("ticker"):
        return f"{r['qty']} шт. «{r['name']}»" if r.get("qty") else f"«{r['name']}»"
    return _rub(r["amount_rub"]) if r.get("amount_rub") else ""


VERB_DONE = {"buy": "купила", "sell": "продала", "withdraw": "вывела на вашу карту", "transfer_third_party": "перевела"}
VERB_TODO = {"buy": "купить", "sell": "продать", "withdraw": "вывести", "transfer_third_party": "перевести"}
RULE_NAME = {"max_issuer_share_pct": ("лимит на одну компанию", "%"), "big_sale_rub": ("порог крупных продаж", " ₽"),
             "second_key_rub": ("порог подтверждения близкого", " ₽"), "max_trades_per_day": ("лимит сделок в день", ""),
             "trust_level": ("уровень свободы Авроры", "")}


def answer_from_result(result: dict) -> str:
    """The assistant's final reply, written from the tool result only."""
    if "error" in result:
        return f"Не получилось: {_lc(result['error'])}"
    if "positions" in result:
        names = "; ".join(f"«{p['name']}» — {p['qty']} шт. на {_rub(p['value_rub'])}" for p in result["positions"])
        return f"Сейчас в портфеле {_rub(result['total_rub'])}: {names}. Свободных денег — {_rub(result['cash_rub'])}."
    if "price_rub" in result:
        return f"«{result['name']}» сейчас стоит {_price(result['price_rub'])}."
    s, kind = result["status"], result["kind"]
    reason = next((c["text"] for c in result["checks"] if not c["ok"]), "")
    if kind == "change_rule":
        name, unit = RULE_NAME[result["rule"]]
        if s == "rule_applied":
            return f"Готово: {name} теперь {result['value']}{unit}. Правило стало строже, поэтому действует сразу."
        if s == "rule_pending":
            return (f"Записала: {name} станет {result['value']}{unit} через 24 часа. Ослабление правил вступает с задержкой, "
                    "чтобы защитить вас от поспешных решений. До тех пор изменение можно отменить.")
        return f"Не получится изменить {name}: {_lc(reason)}."
    what = _what(result)
    if s == "executed":
        return f"Готово: {VERB_DONE[kind]} {what}. Все проверки ваших правил пройдены."
    if s == "denied":
        return f"Не получится {VERB_TODO[kind]} {what}: {_lc(reason)}."
    if s == "waiting_client":
        return f"Подготовила: {VERB_TODO[kind]} {what}. {reason}. Подтвердите, если всё верно."
    return (f"Чтобы {VERB_TODO[kind]} {what}, нужно подтверждение доверенного человека: {_lc(reason)}. "
            "Я отправила ему запрос, на ответ есть 15 минут.")


def _tool_dialog(user: str, call: ToolCall) -> dict:
    aurora = Aurora(demo_portfolio(), KeywordPlanner())  # a fresh account for every dialog
    result = execute(call, aurora)
    msgs = [{"role": "system", "content": system_prompt()}, {"role": "user", "content": user},
            {"role": "assistant", "content": "", "tool_calls": [call.to_dict()]},
            {"role": "tool", "name": call.name, "content": result},
            {"role": "assistant", "content": answer_from_result(result)}]
    return {"intent": call.name, "messages": msgs, "expected": call.to_dict()}


def _text_dialog(user: str, answer: str, intent: str) -> dict:
    return {"intent": intent, "expected": None, "messages": [
        {"role": "system", "content": system_prompt()}, {"role": "user", "content": user},
        {"role": "assistant", "content": answer}]}


def sample(rng: random.Random, split: str) -> dict:
    pick = (lambda pair: rng.choice(pair[0] if split == "train" else pair[1]))
    intent = rng.choices(["buy", "sell", "sell_all", "withdraw", "transfer", "rule", "portfolio", "quote", "fraud", "ambiguous", "general"],
                         weights=[14, 9, 4, 8, 6, 7, 8, 7, 7, 5, 25])[0]
    ticker = rng.choice(list(TICKERS))
    name = rng.choice(TICKERS[ticker])
    if intent in ("buy", "sell"):
        q = rng.choice([1, 2, 5, 10, 20, 50, 100, 300])
        return _tool_dialog(pick(INTENTS[intent]).format(q=q, name=name), ToolCall(intent, {"ticker": ticker, "qty": q}))
    if intent == "sell_all":
        return _tool_dialog(pick(INTENTS[intent]).format(name=name), ToolCall("sell", {"ticker": ticker, "all": True}))
    if intent == "withdraw":
        amount = rng.choice([5000, 15000, 30000, 60000, 120000])
        return _tool_dialog(pick(INTENTS[intent]).format(amount=f"{amount:,}".replace(",", " ")), ToolCall("withdraw", {"amount": amount}))
    if intent == "transfer":
        amount, person = rng.choice([3000, 10000, 50000]), rng.choice(PEOPLE)
        return _tool_dialog(pick(INTENTS[intent]).format(amount=f"{amount:,}".replace(",", " "), person=person),
                            ToolCall("transfer", {"amount": amount, "recipient": person}))
    if intent == "rule":
        v = rng.choice([10, 15, 25, 30])
        return _tool_dialog(pick(INTENTS[intent]).format(v=v), ToolCall("change_rule", {"rule": "max_issuer_share_pct", "value": v}))
    if intent == "portfolio":
        return _tool_dialog(pick(INTENTS[intent]), ToolCall("get_portfolio", {}))
    if intent == "quote":
        return _tool_dialog(pick(INTENTS[intent]).format(name=name), ToolCall("get_quote", {"ticker": ticker}))
    if intent == "fraud":
        return _text_dialog(pick(FRAUD), FRAUD_ANSWER, "fraud")
    if intent == "ambiguous":
        return _text_dialog(pick(AMBIGUOUS), AMBIGUOUS_ANSWER, "ambiguous")
    q, a = rng.choice(GENERAL)
    return _text_dialog(q, a, "general")


def build(out_dir: str, n_train: int, n_eval: int, seed: int = 0) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, n, s in (("train", n_train, seed), ("eval", n_eval, seed + 1)):
        rng = random.Random(s)
        rows = [sample(rng, split) for _ in range(n)]
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        intents = {}
        for r in rows:
            intents[r["intent"]] = intents.get(r["intent"], 0) + 1
        print(f"{split}: {n} dialogs  {dict(sorted(intents.items()))}")


def main():
    ap = argparse.ArgumentParser(description="Generate training and evaluation dialogs")
    ap.add_argument("--out", default="data/assistant")
    ap.add_argument("--train", type=int, default=1200)
    ap.add_argument("--eval", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.out, a.train, a.eval, a.seed)


if __name__ == "__main__":
    main()
