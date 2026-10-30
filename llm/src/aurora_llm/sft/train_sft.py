"""Supervised fine-tuning: turns a pretrained text model into an assistant.

Input: JSONL, one dialog per line:
  {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}]}
Loss is computed only on the assistant's answers.

  python -m aurora_llm.sft.train_sft --config configs/tiny.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random

import torch

from ..chat import encode_dialog
from ..config import load_config
from ..tokenizer import Tok
from ..train_utils import Logger, autocast, load_checkpoint, lr_at, pick_device, pick_dtype, save_checkpoint


def load_dialogs(path: str, tok: Tok, max_len: int) -> list[tuple[list[int], list[int]]]:
    out, skipped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ids, mask = encode_dialog(tok, json.loads(line)["messages"])
            if len(ids) > max_len or sum(mask) == 0:
                skipped += 1
                continue
            out.append((ids, mask))
    if skipped:
        print(f"skipped {skipped} dialogs that are longer than {max_len} tokens or have no answer")
    if not out:
        raise ValueError(f"No usable dialogs in {path}")
    return out


def collate(batch, pad_id: int, device: str):
    T = max(len(ids) for ids, _ in batch) - 1
    x = torch.full((len(batch), T), pad_id, dtype=torch.long)
    y = torch.full((len(batch), T), pad_id, dtype=torch.long)
    m = torch.zeros((len(batch), T), dtype=torch.long)
    for i, (ids, mask) in enumerate(batch):
        n = len(ids) - 1
        x[i, :n] = torch.tensor(ids[:-1])
        y[i, :n] = torch.tensor(ids[1:])
        m[i, :n] = torch.tensor(mask[1:])
    return x.to(device), y.to(device), m.to(device)


@torch.no_grad()
def eval_loss(model, data, s, pad, device, dtype) -> float:
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(data), s.batch_size):
        x, y, m = collate(data[i:i + s.batch_size], pad, device)
        with autocast(device, dtype):
            _, loss = model(x, y, loss_mask=m)
        tot += loss.item()
        n += 1
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description="Fine-tune a pretrained checkpoint on dialogs")
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    s = cfg.sft
    device = pick_device(cfg.train.device)
    dtype = pick_dtype(cfg.train.dtype, device)
    model, base_cfg, _ = load_checkpoint(s.base_checkpoint, device)
    tok = Tok(cfg.data.tokenizer_path)
    max_len = s.max_seq_len or base_cfg.model.max_seq_len
    train = load_dialogs(s.train_file, tok, max_len)
    val = load_dialogs(s.val_file, tok, max_len) if s.val_file else train[: max(1, len(train) // 10)]
    print(f"dialogs: train={len(train)} val={len(val)}")

    opt = model.optimizer(s.lr, 0.0, (0.9, 0.95), device)
    steps_per_epoch = math.ceil(len(train) / s.batch_size)
    total = steps_per_epoch * s.epochs
    log = Logger(s.out_dir)
    rng = random.Random(cfg.train.seed)
    step, best = 0, math.inf
    model.train()
    for epoch in range(s.epochs):
        rng.shuffle(train)
        for i in range(0, len(train), s.batch_size):
            lr = lr_at(step, s.lr, s.lr * 0.1, s.warmup_steps, total)
            for g in opt.param_groups:
                g["lr"] = lr
            x, y, m = collate(train[i:i + s.batch_size], tok.pad, device)
            with autocast(device, dtype):
                _, loss = model(x, y, loss_mask=m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                log.log(step=step, epoch=epoch, loss=loss.item(), lr=lr)
        v = eval_loss(model, val, s, tok.pad, device, dtype)
        log.log(step=step, epoch=epoch, val_loss=v)
        if v < best:
            best = v
            save_checkpoint(os.path.join(s.out_dir, "best.pt"), model, base_cfg, step, extra={"best_val": best, "sft": True})
    save_checkpoint(os.path.join(s.out_dir, "last.pt"), model, base_cfg, step, extra={"best_val": best, "sft": True})
    print({"steps": step, "best_val": best})


if __name__ == "__main__":
    main()
