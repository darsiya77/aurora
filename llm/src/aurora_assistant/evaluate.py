"""Does the model call the right tool with the right arguments, and stay silent when it should?

For each evaluation dialog the model sees the system prompt and the client's message, then generates
its first reply. We compare the parsed tool call with the expected one.

  python -m aurora_assistant.evaluate --base Qwen/Qwen2.5-0.5B-Instruct                     # before training
  python -m aurora_assistant.evaluate --base Qwen/Qwen2.5-0.5B-Instruct --adapter runs/.../adapter
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .format import parse_assistant, to_model_messages


def normalize(args: dict) -> dict:
    """Compare meaning, not spelling: "sber" == "SBER", "10" == 10, "true" == True."""
    out = {}
    for k, v in args.items():
        if isinstance(v, str) and v.strip().lower() in ("true", "false"):
            v = v.strip().lower() == "true"
        elif isinstance(v, str):
            try:
                v = float(v.replace(" ", "").replace(",", "."))
            except ValueError:
                v = v.strip().upper() if k == "ticker" else v.strip().lower()
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            v = float(v)
        out[k] = v
    return out


def score(expected: dict | None, calls) -> dict:
    if expected is None:
        return {"correct": not calls, "name_ok": not calls}
    if not calls:
        return {"correct": False, "name_ok": False}
    got = calls[0]
    name_ok = got.name == expected["name"]
    args_ok = name_ok and normalize(got.arguments) == normalize(expected["arguments"])
    return {"correct": args_ok, "name_ok": name_ok}


def load(base: str, adapter: str | None, trust_remote_code: bool = False):
    tok = AutoTokenizer.from_pretrained(adapter or base, trust_remote_code=trust_remote_code)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, trust_remote_code=trust_remote_code)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model.to("cuda" if torch.cuda.is_available() else "cpu").eval(), tok


@torch.no_grad()
def first_reply(model, tok, conversation: list[dict], max_new_tokens: int = 120) -> str:
    """Greedy decoding: evaluation must be repeatable."""
    ids = tok.apply_chat_template(to_model_messages(conversation), add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    ids = {k: v.to(model.device) for k, v in ids.items()}
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


def evaluate(model, tok, eval_file: str, limit: int, show: int = 0) -> dict:
    rows = [json.loads(line) for line in Path(eval_file).read_text(encoding="utf-8").splitlines() if line.strip()][:limit]
    per = defaultdict(lambda: [0, 0, 0])  # intent -> [correct, name_ok, total]
    for i, r in enumerate(rows):
        prompt = r["messages"][:2]  # system + the client's message
        text = first_reply(model, tok, prompt)
        _, calls = parse_assistant(text)
        s = score(r["expected"], calls)
        p = per[r["intent"]]
        p[0] += s["correct"]
        p[1] += s["name_ok"]
        p[2] += 1
        if i < show:
            print(f"--- [{r['intent']}] {prompt[1]['content']}\n{text.strip()[:300]}\n  -> {'OK' if s['correct'] else 'MISS'}")
    total = [sum(v[j] for v in per.values()) for j in range(3)]
    report = {"accuracy": round(total[0] / total[2], 3), "tool_choice_accuracy": round(total[1] / total[2], 3), "n": total[2],
              "by_intent": {k: f"{v[0]}/{v[2]}" for k, v in sorted(per.items())}}
    return report


def main():
    ap = argparse.ArgumentParser(description="Tool-calling accuracy on held-out dialogs")
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--eval-file", default="data/assistant/eval.jsonl")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--show", type=int, default=0, help="print the first N generations")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--out", help="save the report as JSON")
    a = ap.parse_args()
    torch.manual_seed(0)
    model, tok = load(a.base, a.adapter, a.trust_remote_code)
    report = evaluate(model, tok, a.eval_file, a.limit, a.show)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
