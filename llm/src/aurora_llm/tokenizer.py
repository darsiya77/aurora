"""Byte-level BPE tokenizer: trained on our own corpus, never fails on unseen characters."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Iterator

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

# Special tokens used by pretraining (<|bos|>, <|eos|>) and by the chat format.
SPECIAL = ["<|pad|>", "<|bos|>", "<|eos|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]


def train_tokenizer(texts: Iterable[str], vocab_size: int, out_path: str | Path, min_frequency: int = 2) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    return tok


class Tok:
    """Thin wrapper with the handful of operations the rest of the code needs."""

    def __init__(self, path: str | Path):
        self.tk = Tokenizer.from_file(str(path))
        self.vocab_size = self.tk.get_vocab_size()
        self.special = {s: self.tk.token_to_id(s) for s in SPECIAL}
        self.bos, self.eos, self.pad = self.special["<|bos|>"], self.special["<|eos|>"], self.special["<|pad|>"]

    def encode(self, text: str) -> list[int]:
        return self.tk.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.tk.encode_batch(texts, add_special_tokens=False)]

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        return self.tk.decode(ids, skip_special_tokens=skip_special)

    def id(self, token: str) -> int:
        return self.special[token]


def iter_corpus(raw_dir: str | Path) -> Iterator[str]:
    from .data.sources import iter_documents
    yield from iter_documents(raw_dir)


def main():
    ap = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer on data/raw")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--out", default="data/tokenizer/tokenizer.json")
    ap.add_argument("--max-docs", type=int, default=0, help="0 = all documents")
    a = ap.parse_args()

    def docs():
        for i, d in enumerate(iter_corpus(a.raw)):
            if a.max_docs and i >= a.max_docs:
                break
            yield d

    tok = train_tokenizer(docs(), a.vocab, a.out)
    sample = "Аврора проверяет каждую сделку по правилам клиента."
    ids = tok.encode(sample).ids
    print(f"vocab={tok.get_vocab_size()}  sample: {len(sample)} chars -> {len(ids)} tokens")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
