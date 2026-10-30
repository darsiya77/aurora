"""Perplexity on the held-out split: the main number for comparing pretraining runs.

  python -m aurora_llm.eval.perplexity --ckpt runs/tiny/best.pt
Lower is better. A model that guesses uniformly over the vocabulary scores perplexity = vocab size.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from ..data.dataset import TokenStream
from ..train_utils import autocast, load_checkpoint, pick_device, pick_dtype


@torch.no_grad()
def perplexity(ckpt: str, processed_dir: str | None = None, max_tokens: int = 200_000) -> dict:
    device = pick_device()
    model, cfg, _ = load_checkpoint(ckpt, device)
    model.eval()
    dtype = pick_dtype("auto", device)
    data = TokenStream(processed_dir or cfg.data.processed_dir, "val").data
    T = cfg.model.max_seq_len
    n = min(len(data) - 1, max_tokens)
    nll, count = 0.0, 0
    # Non-overlapping windows over the whole (capped) validation split: deterministic and complete.
    for start in range(0, n - T, T):
        x = torch.from_numpy(data[start:start + T].astype(np.int64))[None].to(device)
        y = torch.from_numpy(data[start + 1:start + 1 + T].astype(np.int64))[None].to(device)
        with autocast(device, dtype):
            _, loss = model(x, y)
        nll += loss.item() * T
        count += T
    if not count:
        raise ValueError("Validation split is shorter than one window")
    mean = nll / count
    return {"tokens": count, "loss": round(mean, 4), "perplexity": round(math.exp(mean), 2),
            "uniform_baseline": cfg.model.vocab_size}


def main():
    ap = argparse.ArgumentParser(description="Perplexity on the validation split")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data")
    ap.add_argument("--max-tokens", type=int, default=200_000)
    a = ap.parse_args()
    print(perplexity(a.ckpt, a.data, a.max_tokens))


if __name__ == "__main__":
    main()
