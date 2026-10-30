"""Web chat with our model: talk to it, watch every action pass the rule check, and teach it.

  python -m aurora_assistant.serve --base Qwen/Qwen2.5-0.5B-Instruct --adapter runs/assistant-tiny-qwen/adapter
  python -m aurora_assistant.serve --api http://gpu-server:8000/v1 --model aurora      # our model on a GPU server
Then open http://127.0.0.1:8080

Teaching: any answer can be corrected in the chat. The corrected dialog is saved to
data/assistant/feedback.jsonl, and the next training run picks it up (see `extra_train_files`).
"""
from __future__ import annotations

import argparse
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from aurora_agent.gateway import rub
from aurora_agent.model import RULE_LABEL

from .assistant import Assistant, add_model_args, build_model

PAGE = Path(__file__).with_name("web") / "chat.html"
FEEDBACK = Path("data/assistant/feedback.jsonl")


class Sessions:
    """One assistant (conversation + demo account) per browser tab; one model shared by all."""

    def __init__(self, model):
        self.model, self.items, self.lock = model, {}, threading.Lock()

    def get(self, sid: str) -> Assistant:
        if sid not in self.items:
            self.items[sid] = Assistant(self.model)
        return self.items[sid]


def account_state(a: Assistant) -> dict:
    au, p, r = a.aurora, a.aurora.portfolio, a.aurora.rulebook.rules
    return {
        "total": rub(p.total()), "cash": rub(p.cash),
        "positions": [{"ticker": t, "name": p.market[t].name, "qty": q, "value": rub(p.value_of(t))} for t, q in p.positions.items()],
        "rules": [{"label": RULE_LABEL[k], "value": str(getattr(r, k))} for k in
                  ("max_issuer_share_pct", "second_key_rub", "big_sale_rub", "max_trades_per_day", "trust_level")],
        "pending_rules": [{"label": RULE_LABEL[x.rule], "value": str(x.value), "at": x.effective_at.strftime("%d.%m %H:%M")}
                          for x in au.rulebook.pending],
        "waiting": [{"ticket": t, "action": v[0].describe(), "kind": v[1]} for t, v in au.waiting.items()],
        "journal": [{"n": x["n"], "event": x["event"], "text": x.get("action") or x.get("rule") or "", "verdict": x.get("verdict", ""),
                     "hash": x["hash"][:8]} for x in au.ledger.records[-8:]][::-1],
        "journal_ok": au.ledger.verify() is None,
    }


def save_feedback(a: Assistant, corrected: str) -> int:
    """Store the last exchange with the corrected final answer as a training dialog."""
    conv = a.conversation
    last_user = max(i for i, m in enumerate(conv) if m["role"] == "user")
    dialog = [conv[0], *conv[last_user:]]
    final = next(i for i in range(len(dialog) - 1, -1, -1) if dialog[i]["role"] == "assistant")
    dialog[final] = {"role": "assistant", "content": corrected.strip()}
    FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"intent": "feedback", "expected": None, "messages": dialog}, ensure_ascii=False, default=str) + "\n")
    conv[len(conv) - (len(dialog) - final)] = dialog[final]  # the conversation continues from the corrected answer
    return feedback_count()


def feedback_count() -> int:
    return sum(1 for line in FEEDBACK.read_text(encoding="utf-8").splitlines() if line.strip()) if FEEDBACK.exists() else 0


def make_handler(sessions: Sessions):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, body: dict):
            data = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict:
            try:
                return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except ValueError:
                return {}

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                data = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif url.path == "/api/info":
                self._json(200, {"model": sessions.model.name, "feedback": feedback_count(), "session": uuid.uuid4().hex})
            elif url.path == "/api/state":
                sid = parse_qs(url.query).get("session", [""])[0]
                with sessions.lock:
                    self._json(200, account_state(sessions.get(sid)))
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            path, body = urlparse(self.path).path, self._body()
            sid = str(body.get("session", ""))
            if not sid:
                self._json(400, {"error": "session is required"})
                return
            with sessions.lock:  # one generation at a time: the model is shared
                a = sessions.get(sid)
                if path == "/api/chat":
                    text = str(body.get("message", "")).strip()
                    if not text:
                        self._json(400, {"error": "empty message"})
                        return
                    try:
                        turn = a.ask(text, float(body.get("temperature", 0.3)))
                    except Exception as e:  # the model server is down or returned garbage: tell the user, keep running
                        a.conversation.pop()
                        self._json(502, {"error": f"Модель не ответила: {e}"})
                        return
                    self._json(200, {"reply": turn.reply, "steps": turn.steps, "state": account_state(a)})
                elif path == "/api/confirm":
                    t, yes = int(body.get("ticket", 0)), bool(body.get("yes", True))
                    kind = a.aurora.waiting.get(t, (None, None))[1]
                    out = a.aurora.second_key(t, yes) if kind == "waiting_second_key" else a.aurora.confirm(t, yes)
                    self._json(200, {"status": out.status if out else "expired", "state": account_state(a)})
                elif path == "/api/feedback":
                    corrected = str(body.get("corrected", "")).strip()
                    if not corrected or len(a.conversation) < 3:
                        self._json(400, {"error": "nothing to correct"})
                        return
                    self._json(200, {"feedback": save_feedback(a, corrected)})
                elif path == "/api/reset":
                    a.reset()
                    self._json(200, {"state": account_state(a)})
                else:
                    self._json(404, {"error": "not found"})

        def log_message(self, fmt, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description="Web chat with the assistant")
    add_model_args(ap)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    sessions = Sessions(build_model(a))
    print(f"Чат: http://{a.host}:{a.port}   модель: {sessions.model.name}")
    ThreadingHTTPServer((a.host, a.port), make_handler(sessions)).serve_forever()


if __name__ == "__main__":
    main()
