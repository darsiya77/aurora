"""Fold the LoRA adapter into the base weights, producing one ordinary model folder.

A merged model is served by any inference server (vLLM, SGLang, TGI) exactly like the original,
with no adapter support required and no speed penalty.

  python -m aurora_assistant.merge --base moonshotai/Moonlight-16B-A3B-Instruct --trust-remote-code \
      --adapter runs/assistant-moonlight/adapter --out runs/assistant-moonlight/merged
"""
from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model")
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trust-remote-code", action="store_true")
    a = ap.parse_args()
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, trust_remote_code=a.trust_remote_code)
    merged = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    merged.save_pretrained(a.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.adapter, trust_remote_code=a.trust_remote_code).save_pretrained(a.out)
    print(f"saved merged model to {a.out}")


if __name__ == "__main__":
    main()
