"""The runtime loop: the model answers or calls tools, every call passes the rule check, the model answers again.

Two interchangeable backends:
  LocalModel   a Hugging Face model, optionally with our LoRA adapter (Qwen for tests, Moonlight / Kimi-Linear on a GPU)
  RemoteModel  any OpenAI-compatible API: a vLLM server with our model, or a hosted Kimi model

  python -m aurora_assistant.assistant --base Qwen/Qwen2.5-0.5B-Instruct --adapter runs/assistant-tiny-qwen/adapter
  python -m aurora_assistant.assistant --api https://api.example.com/v1 --model kimi-k2.6 --key $API_KEY
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from aurora_agent.agent import Aurora
from aurora_agent.planner import KeywordPlanner
from aurora_agent.portfolio import demo_portfolio

from .format import parse_assistant, system_prompt, to_model_messages
from .tools import TOOL_NAMES, execute

MAX_TOOL_ROUNDS = 3   # a reply may take a few tool calls, but never loops forever
CONTEXT_FILE = Path(__file__).resolve().parents[2] / "docs" / "AURORA_CONTEXT.md"
# Questions about the project itself get the project brief; account requests keep the exact training prompt.
ABOUT_PROJECT = re.compile(r"аврор|проект|хакатон|ryudo|рюдо|архитектур|как (ты )?устроен|презентаци|жюри|идея", re.I)


class ChatModel(Protocol):
    name: str

    def complete(self, messages: list[dict], max_new_tokens: int, temperature: float) -> str: ...


class LocalModel:
    def __init__(self, base: str, adapter: str | None = None, trust_remote_code: bool = False):
        from .evaluate import load  # torch and transformers are imported only when a local model is used
        self.model, self.tok = load(base, adapter, trust_remote_code)
        self.name = f"{base}" + (f" + {Path(adapter).parent.name}" if adapter else "")

    def complete(self, messages, max_new_tokens=400, temperature=0.3) -> str:
        import torch
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids = {k: v.to(self.model.device) for k, v in ids.items()}
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                                      temperature=temperature if temperature > 0 else None, top_p=0.9 if temperature > 0 else None,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


class RemoteModel:
    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: int = 120):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.key, self.timeout, self.name = model, api_key, timeout, model

    def complete(self, messages, max_new_tokens=800, temperature=0.3) -> str:
        body = json.dumps({"model": self.model, "messages": messages, "max_tokens": max_new_tokens,
                           "temperature": temperature}).encode()
        headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {self.key}"} if self.key else {})}
        with urllib.request.urlopen(urllib.request.Request(self.url, body, headers), timeout=self.timeout) as r:
            return json.load(r)["choices"][0]["message"]["content"]


@dataclass
class Turn:
    """One client message and everything that happened while answering it, for the UI and the logs."""
    reply: str
    steps: list[dict] = field(default_factory=list)   # tool calls with their results


class Assistant:
    def __init__(self, model: ChatModel, aurora: Aurora | None = None, with_project_context: bool = True):
        self.model = model
        self.aurora = aurora or Aurora(demo_portfolio(), KeywordPlanner())
        self.project_brief = CONTEXT_FILE.read_text(encoding="utf-8") if with_project_context and CONTEXT_FILE.exists() else ""
        self.conversation: list[dict] = [{"role": "system", "content": system_prompt()}]

    def _prompt_for(self, text: str) -> list[dict]:
        """The conversation as the model sees it, with the project brief added only when it is relevant."""
        conv = self.conversation
        if self.project_brief and ABOUT_PROJECT.search(text):
            brief = conv[0]["content"] + "\n\nО проекте Аврора (отвечай на вопросы о проекте по этому тексту):\n" + self.project_brief
            conv = [{"role": "system", "content": brief}, *conv[1:]]
        return to_model_messages(conv)

    def ask(self, text: str, temperature: float = 0.3) -> Turn:
        self.conversation.append({"role": "user", "content": text})
        steps: list[dict] = []
        for _ in range(MAX_TOOL_ROUNDS + 1):
            raw = self.model.complete(self._prompt_for(text), max_new_tokens=500, temperature=temperature)
            visible, calls = parse_assistant(raw)
            calls = [c for c in calls if c.name in TOOL_NAMES]
            self.conversation.append({"role": "assistant", "content": visible, "tool_calls": [c.to_dict() for c in calls]})
            if not calls or len(steps) >= MAX_TOOL_ROUNDS:
                return Turn(visible or "Готово.", steps)
            for call in calls:
                result = execute(call, self.aurora)
                steps.append({"tool": call.name, "arguments": call.arguments, "result": result})
                self.conversation.append({"role": "tool", "name": call.name, "content": result})
        return Turn("Не удалось завершить действие за несколько шагов. Попробуйте переформулировать запрос.", steps)

    def reset(self) -> None:
        self.conversation = self.conversation[:1]
        self.aurora = Aurora(demo_portfolio(), KeywordPlanner())


def build_model(a) -> ChatModel:
    if a.api:
        return RemoteModel(a.api, a.model, a.key or os.environ.get("AURORA_API_KEY", ""))
    return LocalModel(a.base, a.adapter, a.trust_remote_code)


def add_model_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct", help="local Hugging Face model")
    ap.add_argument("--adapter", help="our LoRA adapter for the local model")
    ap.add_argument("--trust-remote-code", action="store_true", help="needed for Kimi / Moonlight models")
    ap.add_argument("--api", help="OpenAI-compatible base URL, e.g. http://gpu-server:8000/v1")
    ap.add_argument("--model", default="aurora", help="model name for --api")
    ap.add_argument("--key", help="API key for --api (or AURORA_API_KEY)")


def main():
    ap = argparse.ArgumentParser(description="Chat with the assistant in the terminal")
    add_model_args(ap)
    assistant = Assistant(build_model(ap.parse_args()))
    print(f"Модель: {assistant.model.name}. Пустая строка — выход.")
    while (text := input("вы> ").strip()):
        turn = assistant.ask(text)
        for s in turn.steps:
            print(f"   ⚙ {s['tool']}({json.dumps(s['arguments'], ensure_ascii=False)}) → {s['result'].get('status_text', 'данные получены')}")
        print(f"аврора> {turn.reply}")


if __name__ == "__main__":
    main()
