"""Random fixed-length windows from a memory-mapped token file (works for files larger than RAM)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class TokenStream:
    def __init__(self, processed_dir: str | Path, split: str):
        d = Path(processed_dir)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self.data = np.memmap(d / f"{split}.bin", dtype=np.dtype(meta["dtype"]), mode="r")
        self.meta = meta

    def __len__(self) -> int:
        return len(self.data)

    def batch(self, batch_size: int, seq_len: int, device: str, gen: torch.Generator | None = None):
        if len(self.data) <= seq_len + 1:
            raise ValueError(f"Split has {len(self.data)} tokens, need more than seq_len+1={seq_len + 1}")
        ix = torch.randint(len(self.data) - seq_len - 1, (batch_size,), generator=gen)
        x = torch.stack([torch.from_numpy(self.data[i:i + seq_len].astype(np.int64)) for i in ix.tolist()])
        y = torch.stack([torch.from_numpy(self.data[i + 1:i + 1 + seq_len].astype(np.int64)) for i in ix.tolist()])
        if device.startswith("cuda"):
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)
