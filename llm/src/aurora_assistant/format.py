"""One text format for tool calls, independent of the base model.

Our conversation (the format used everywhere in this package):
  {"role": "system" | "user", "content": "..."}
  {"role": "assistant", "content": "...", "tool_calls": [{"name": "buy", "arguments": {...}}]}
  {"role": "tool", "name": "buy", "content": {...result...}}

What the model sees, after `to_model_messages`: only system / user / assistant text, which every chat
template supports. Tool calls and results become tagged text:
  assistant:  <tool_call>{"name": "buy", "arguments": {"ticker": "SBER", "qty": 10}}</tool_call>
  user:       <tool_result name="buy">{"status": "executed", ...}</tool_result>
"""
from __future__ import annotations

import json
import re

from .tools import TOOLS, ToolCall

SYSTEM_PROMPT = """Ты Аврора, помощник по инвестициям в приложении банка. Отвечай по-русски, точно, тепло и по делу.
Можешь отвечать на любые вопросы: о финансах, налогах, жизни. Если чего-то не знаешь, так и скажи.
Никогда не обещай доходность и не советуй обходить правила клиента.

Чтобы выполнить действие или узнать данные счёта, вызови инструмент. Формат вызова, ровно одна строка:
<tool_call>{"name": "имя", "arguments": {...}}</tool_call>
После вызова дождись <tool_result> и только потом отвечай клиенту. Не выдумывай цены и суммы: бери их из результатов.
Каждое действие проверяется правилами клиента. Если правила запретили действие, объясни причину простыми словами.
Если человек говорит о «службе безопасности», «безопасном счёте» или его торопят перевести деньги — не вызывай инструменты, предупреди о мошенничестве.

Инструменты:
{tools}"""

_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def system_prompt(tools: list[dict] = TOOLS) -> str:
    lines = []
    for t in tools:
        params = ", ".join(f"{k}: {v}" for k, v in t["parameters"].items()) or "без аргументов"
        lines.append(f"- {t['name']}({params}) — {t['description']}")
    return SYSTEM_PROMPT.replace("{tools}", "\n".join(lines))


def render_call(call: dict) -> str:
    return "<tool_call>" + json.dumps({"name": call["name"], "arguments": call.get("arguments", {})}, ensure_ascii=False) + "</tool_call>"


def render_result(name: str, result) -> str:
    body = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    return f'<tool_result name="{name}">{body}</tool_result>'


def to_model_messages(conversation: list[dict]) -> list[dict]:
    """Our conversation -> plain chat messages for the base model's template."""
    out = []
    for m in conversation:
        role = m["role"]
        if role == "assistant":
            parts = [m.get("content", "").strip()] + [render_call(c) for c in m.get("tool_calls", [])]
            out.append({"role": "assistant", "content": "\n".join(p for p in parts if p)})
        elif role == "tool":
            text = render_result(m["name"], m["content"])
            # Several results in a row are merged into one user turn, as the model would read them.
            if out and out[-1]["role"] == "user" and out[-1]["content"].startswith("<tool_result"):
                out[-1]["content"] += "\n" + text
            else:
                out.append({"role": "user", "content": text})
        elif role in ("system", "user"):
            out.append({"role": role, "content": m["content"]})
        else:
            raise ValueError(f"Unknown role: {role}")
    return out


def parse_assistant(text: str) -> tuple[str, list[ToolCall]]:
    """Model output -> (visible answer, tool calls). Broken JSON inside a tag is ignored, never guessed."""
    calls = []
    for raw in _CALL.findall(text):
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and isinstance(d.get("name"), str) and isinstance(d.get("arguments", {}), dict):
            calls.append(ToolCall(d["name"], d.get("arguments", {})))
    visible = _CALL.sub("", text)
    visible = re.sub(r"<tool_call>.*", "", visible, flags=re.S).strip()  # drop an unfinished tag at the end
    return visible, calls


def tokenize_for_training(tokenizer, conversation: list[dict], max_len: int) -> tuple[list[int], list[int]] | None:
    """Token ids and labels: the model learns only its own (assistant) turns; everything else is -100.

    Each assistant turn is located by rendering the dialog up to that turn with and without it.
    This works for any template whose rendering of a prefix is a prefix of the full rendering;
    we check that and fail loudly otherwise, instead of silently training on the wrong tokens.
    """
    msgs = to_model_messages(conversation)

    def render(upto: int, gen: bool) -> list[int]:
        return tokenizer.apply_chat_template(msgs[:upto], tokenize=True, add_generation_prompt=gen,
                                             return_dict=False)

    ids = render(len(msgs), False)
    labels = [-100] * len(ids)
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        start, end = len(render(i, True)), len(render(i + 1, False))
        if render(i + 1, False) != ids[:end] or start > end:
            raise ValueError("Chat template is not prefix-stable; loss masking would be wrong for this model")
        labels[start:end] = ids[start:end]
    if len(ids) > max_len:
        return None
    return ids, labels
