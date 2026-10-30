"""Talk to a checkpoint from the terminal.

Continue text:  python -m aurora_llm.generate --ckpt runs/tiny/best.pt --prompt "Инвестиции — это"
Chat (after fine-tuning):  python -m aurora_llm.generate --ckpt runs/tiny-sft/best.pt --chat
"""
from __future__ import annotations

import argparse

import torch

from .chat import encode_dialog
from .tokenizer import Tok
from .train_utils import load_checkpoint, pick_device


class Generator:
    def __init__(self, ckpt: str, tokenizer: str | None = None, device: str = "auto"):
        self.device = pick_device(device)
        self.model, self.cfg, _ = load_checkpoint(ckpt, self.device)
        self.tok = Tok(tokenizer or self.cfg.data.tokenizer_path)

    def _run(self, ids: list[int], stop: tuple[int, ...], **kw) -> str:
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(x, stop_ids=stop, **kw)[0, len(ids):].tolist()
        out = [t for t in out if t not in stop]
        return self.tok.decode(out)

    def complete(self, prompt: str, **kw) -> str:
        return self._run([self.tok.bos] + self.tok.encode(prompt), (self.tok.eos,), **kw)

    def chat(self, messages: list[dict], **kw) -> str:
        ids, _ = encode_dialog(self.tok, messages, add_generation_prompt=True)
        return self._run(ids, (self.tok.id("<|end|>"), self.tok.eos), **kw).strip()


def main():
    ap = argparse.ArgumentParser(description="Generate text with a trained checkpoint")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    g = Generator(a.ckpt, a.tokenizer)
    kw = dict(max_new_tokens=a.max_new_tokens, temperature=a.temperature, top_k=a.top_k, top_p=a.top_p)
    if not a.chat:
        print(a.prompt + g.complete(a.prompt, **kw))
        return
    history = [{"role": "system", "content": "Ты Аврора, внимательный помощник по инвестициям."}]
    print("Чат с моделью. Пустая строка — выход.")
    while True:
        q = input("вы> ").strip()
        if not q:
            break
        history.append({"role": "user", "content": q})
        ans = g.chat(history, **kw)
        history.append({"role": "assistant", "content": ans})
        print(f"модель> {ans}")


if __name__ == "__main__":
    main()
