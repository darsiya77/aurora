"""Minimal HTTP server with an OpenAI-style chat endpoint, so apps can call our model.

  python -m aurora_llm.serve --ckpt runs/tiny-sft/best.pt --port 8000
  curl -s localhost:8000/v1/chat/completions -d '{"messages":[{"role":"user","content":"Привет"}]}'
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .generate import Generator

_lock = threading.Lock()  # one generation at a time; a real deployment would batch requests


def make_handler(gen: Generator, model_name: str):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict):
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok", "model": model_name})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self._send(404, {"error": "not found"})
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                messages = req["messages"]
            except (ValueError, KeyError):
                self._send(400, {"error": "body must be JSON with a 'messages' list"})
                return
            kw = dict(max_new_tokens=int(req.get("max_tokens", 256)), temperature=float(req.get("temperature", 0.7)),
                      top_p=float(req.get("top_p", 0.95)))
            with _lock:
                answer = gen.chat(messages, **kw)
            self._send(200, {
                "id": f"chatcmpl-{int(time.time() * 1000)}", "object": "chat.completion", "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
            })

        def log_message(self, fmt, *args):
            print(f"{self.address_string()} {fmt % args}")

    return Handler


def main():
    ap = argparse.ArgumentParser(description="Serve a checkpoint over HTTP")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--name", default="aurora-llm")
    a = ap.parse_args()
    gen = Generator(a.ckpt)
    print(f"serving {a.ckpt} on http://{a.host}:{a.port}")
    ThreadingHTTPServer((a.host, a.port), make_handler(gen, a.name)).serve_forever()


if __name__ == "__main__":
    main()
