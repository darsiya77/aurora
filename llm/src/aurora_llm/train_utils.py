"""Shared pieces of the training loops: device, precision, schedule, checkpoints, logging."""
from __future__ import annotations

import contextlib
import json
import math
import os
import time
from pathlib import Path

import torch

from .config import Config, config_from_dict
from .model import GPT


def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(pref: str, device: str) -> torch.dtype:
    if pref != "auto":
        return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[pref]
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def autocast(device: str, dtype: torch.dtype):
    if dtype == torch.float32 or device == "cpu":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda" if device.startswith("cuda") else device, dtype=dtype)


def lr_at(step: int, base: float, minimum: float, warmup: int, total: int) -> float:
    """Linear warmup, then cosine decay to `minimum`."""
    if step < warmup:
        return base * (step + 1) / warmup
    if step >= total:
        return minimum
    progress = (step - warmup) / max(1, total - warmup)
    return minimum + 0.5 * (base - minimum) * (1 + math.cos(math.pi * progress))


def save_checkpoint(path: str | Path, model: GPT, cfg: Config, step: int, optimizer=None, extra: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict(), "config": cfg.to_dict(), "step": step, "extra": extra or {}}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written checkpoint


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[GPT, Config, dict]:
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_dict(state["config"])
    model = GPT(cfg.model)
    sd = {k.removeprefix("_orig_mod.").removeprefix("module."): v for k, v in state["model"].items()}
    model.load_state_dict(sd)
    return model.to(device), cfg, state


class Logger:
    """Prints a line and appends the same record to a JSONL file for later plots."""

    def __init__(self, out_dir: str | Path, enabled: bool = True):
        self.enabled = enabled
        self.path = Path(out_dir) / "log.jsonl"
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()

    def log(self, **rec):
        if not self.enabled:
            return
        rec["elapsed_s"] = round(time.time() - self.t0, 1)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
