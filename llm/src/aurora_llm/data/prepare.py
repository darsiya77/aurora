"""Raw text -> cleaned, deduplicated, tokenized binary shards ready for training.

Output in --out:
  train.bin, val.bin   flat arrays of token ids (uint16 if vocab < 65536, else uint32)
  meta.json            counts and dtype, read by the trainer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..tokenizer import Tok
from .clean import doc_hash, normalize, quality_ok, scrub_pii
from .sources import iter_documents


def prepare(raw_dir: str, tokenizer_path: str, out_dir: str, val_fraction: float = 0.005,
            min_chars: int = 200, batch: int = 256) -> dict:
    tok = Tok(tokenizer_path)
    dtype = np.uint16 if tok.vocab_size < 65536 else np.uint32
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {"train": open(out / "train.bin", "wb"), "val": open(out / "val.bin", "wb")}
    stats = {"docs_seen": 0, "dropped_quality": 0, "dropped_duplicate": 0, "docs_kept": 0,
             "tokens": {"train": 0, "val": 0}}
    seen: set[str] = set()
    pending: list[tuple[str, str]] = []
    every = max(1, round(1 / val_fraction)) if val_fraction > 0 else 0

    def flush():
        ids = tok.encode_batch([t for _, t in pending])
        for (split, _), seq in zip(pending, ids):
            arr = np.asarray(seq + [tok.eos], dtype=dtype)
            files[split].write(arr.tobytes())
            stats["tokens"][split] += len(arr)
        pending.clear()

    for doc in iter_documents(raw_dir):
        stats["docs_seen"] += 1
        text = scrub_pii(normalize(doc))
        if not quality_ok(text, min_chars):
            stats["dropped_quality"] += 1
            continue
        h = doc_hash(text)
        if h in seen:
            stats["dropped_duplicate"] += 1
            continue
        seen.add(h)
        stats["docs_kept"] += 1
        # Split by document hash so the same document never lands in both splits.
        split = "val" if every and int(h[:8], 16) % every == 0 else "train"
        pending.append((split, text))
        if len(pending) >= batch:
            flush()
    if pending:
        flush()
    for f in files.values():
        f.close()
    if stats["tokens"]["val"] == 0 and stats["tokens"]["train"] > 0:
        # Tiny corpora can miss the val split entirely; carve the last 2% of train instead.
        train = np.fromfile(out / "train.bin", dtype=dtype)
        cut = int(len(train) * 0.98)
        train[:cut].tofile(out / "train.bin")
        train[cut:].tofile(out / "val.bin")
        stats["tokens"] = {"train": cut, "val": len(train) - cut}
    meta = {"dtype": np.dtype(dtype).name, "vocab_size": tok.vocab_size, "eos": tok.eos, **stats}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main():
    ap = argparse.ArgumentParser(description="Clean, deduplicate and tokenize data/raw into binary shards")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--val-fraction", type=float, default=0.005)
    ap.add_argument("--min-chars", type=int, default=200)
    a = ap.parse_args()
    meta = prepare(a.raw, a.tokenizer, a.out, a.val_fraction, a.min_chars)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
