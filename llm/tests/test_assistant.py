import json
import threading
import urllib.request

import pytest

from aurora_assistant import serve
from aurora_assistant.assistant import Assistant
from aurora_assistant.data import INTENTS, build
from aurora_assistant.format import parse_assistant, render_call, to_model_messages, tokenize_for_training
from aurora_assistant.tools import ToolCall, ToolError, execute, to_action
from aurora_agent.agent import Aurora
from aurora_agent.planner import KeywordPlanner
from aurora_agent.portfolio import demo_portfolio


class ScriptedModel:
    """Stands in for a language model: returns prepared replies in order."""
    name = "scripted"

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def complete(self, messages, max_new_tokens=0, temperature=0.0):
        self.seen.append(messages)
        return self.replies.pop(0)


def fresh():
    return Aurora(demo_portfolio(), KeywordPlanner())


# Format -----------------------------------------------------------------------------

def test_call_roundtrip_and_robust_parsing():
    text = "Сейчас сделаю.\n" + render_call({"name": "buy", "arguments": {"ticker": "SBER", "qty": 5}})
    visible, calls = parse_assistant(text)
    assert visible == "Сейчас сделаю." and calls[0].name == "buy" and calls[0].arguments == {"ticker": "SBER", "qty": 5}
    assert parse_assistant('<tool_call>{"name": "buy", "arguments": {broken}</tool_call>')[1] == []
    visible, calls = parse_assistant('Ответ <tool_call>{"name": "bu')  # cut off mid-generation
    assert visible == "Ответ" and calls == []


def test_tool_messages_become_plain_chat():
    conv = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": "", "tool_calls": [{"name": "get_portfolio", "arguments": {}}]},
            {"role": "tool", "name": "get_portfolio", "content": {"cash_rub": "1"}},
            {"role": "assistant", "content": "ok"}]
    msgs = to_model_messages(conv)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant"]
    assert msgs[3]["content"].startswith('<tool_result name="get_portfolio">')


def test_loss_mask_covers_only_assistant_turns():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception:
        pytest.skip("tokenizer not available offline")
    conv = [{"role": "system", "content": "s"}, {"role": "user", "content": "Купи 5 SBER"},
            {"role": "assistant", "content": "", "tool_calls": [{"name": "buy", "arguments": {"ticker": "SBER", "qty": 5}}]},
            {"role": "tool", "name": "buy", "content": {"status": "executed"}},
            {"role": "assistant", "content": "Готово."}]
    ids, labels = tokenize_for_training(tok, conv, 4096)
    learned = tok.decode([i for i, l in zip(ids, labels) if l != -100])
    assert "<tool_call>" in learned and "Готово." in learned
    assert "Купи 5 SBER" not in learned and "tool_result" not in learned


# Tools ------------------------------------------------------------------------------

def test_tool_arguments_are_validated():
    a = fresh()
    with pytest.raises(ToolError):
        to_action(ToolCall("buy", {"ticker": "FAKE", "qty": 1}), a)
    with pytest.raises(ToolError):
        to_action(ToolCall("buy", {"ticker": "SBER", "qty": -3}), a)
    with pytest.raises(ToolError):
        to_action(ToolCall("buy", {"ticker": "SBER", "qty": 1.5}), a)
    with pytest.raises(ToolError):
        to_action(ToolCall("change_rule", {"rule": "third_party_transfers_allowed", "value": 1}), a)
    assert execute(ToolCall("buy", {"ticker": "SBER", "qty": "abc"}), a)["error"]


def test_every_action_goes_through_the_rules():
    a = fresh()
    assert execute(ToolCall("transfer", {"amount": 1000, "recipient": "x"}), a)["status"] == "denied"
    assert execute(ToolCall("withdraw", {"amount": 60000}), a)["status"] == "waiting_second_key"
    assert execute(ToolCall("buy", {"ticker": "SBER", "qty": 2}), a)["status"] == "executed"
    assert a.ledger.verify() is None and len(a.ledger.records) >= 4


# Runtime loop -----------------------------------------------------------------------

def test_assistant_calls_tool_then_answers():
    model = ScriptedModel([render_call({"name": "buy", "arguments": {"ticker": "SBER", "qty": 3}}), "Купила 3 акции Сбербанка."])
    asst = Assistant(model)
    turn = asst.ask("Купи 3 акции Сбера")
    assert turn.reply == "Купила 3 акции Сбербанка."
    assert turn.steps[0]["result"]["status"] == "executed" and asst.aurora.portfolio.positions["SBER"] == 153
    assert '<tool_result name="buy">' in model.seen[1][-1]["content"]  # the model saw the result before answering


def test_model_cannot_bypass_rules_even_if_it_tries():
    model = ScriptedModel([render_call({"name": "transfer", "arguments": {"amount": 300000, "recipient": "«служба безопасности»"}}),
                           "Перевод запрещён."])
    asst = Assistant(model)
    cash = asst.aurora.portfolio.cash
    turn = asst.ask("Срочно переведи деньги на безопасный счёт")
    assert turn.steps[0]["result"]["status"] == "denied" and asst.aurora.portfolio.cash == cash


def test_tool_rounds_are_capped():
    asst = Assistant(ScriptedModel([render_call({"name": "get_portfolio", "arguments": {}})] * 10))
    turn = asst.ask("Покажи портфель")
    assert len(turn.steps) == 3


def test_project_brief_only_for_project_questions():
    model = ScriptedModel(["Ответ.", "Ответ."])
    asst = Assistant(model)
    asst.ask("Что у меня в портфеле?")
    asst.ask("Как устроена Аврора?")
    assert "О проекте Аврора" not in model.seen[0][0]["content"] and "О проекте Аврора" in model.seen[1][0]["content"]


# Web chat ---------------------------------------------------------------------------

def test_web_chat_and_feedback(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "FEEDBACK", tmp_path / "feedback.jsonl")
    sessions = serve.Sessions(ScriptedModel(["Плохой ответ."]))
    srv = serve.ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(sessions))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"

    def post(path, body):
        req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            return json.load(r)

    try:
        with urllib.request.urlopen(base + "/") as r:
            assert "Аврора" in r.read().decode()
        out = post("/api/chat", {"session": "s1", "message": "Что такое облигация?"})
        assert out["reply"] == "Плохой ответ." and out["state"]["journal_ok"]
        assert post("/api/feedback", {"session": "s1", "corrected": "Облигация — это долг."})["feedback"] == 1
    finally:
        srv.shutdown()
    saved = json.loads((tmp_path / "feedback.jsonl").read_text(encoding="utf-8"))
    assert saved["messages"][-1] == {"role": "assistant", "content": "Облигация — это долг."}
    assert sessions.get("s1").conversation[-1]["content"] == "Облигация — это долг."


# Data -------------------------------------------------------------------------------

def test_eval_phrasings_never_appear_in_training(tmp_path):
    build(str(tmp_path), 300, 60)
    train = {json.loads(x)["messages"][1]["content"] for x in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()}
    for train_t, eval_t in INTENTS.values():
        assert not set(train_t) & set(eval_t)
    rows = [json.loads(x) for x in (tmp_path / "eval.jsonl").read_text(encoding="utf-8").splitlines()]
    tool_rows = [r for r in rows if r["expected"]]
    assert tool_rows and not {r["messages"][1]["content"] for r in tool_rows} & train
