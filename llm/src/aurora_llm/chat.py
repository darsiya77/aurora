"""Chat format shared by fine-tuning, generation and the server.

<|bos|><|system|>...<|end|><|user|>...<|end|><|assistant|>...<|end|>
The model learns to answer only inside <|assistant|> ... <|end|>.
"""
from __future__ import annotations

from .tokenizer import Tok

ROLES = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}


def encode_dialog(tok: Tok, messages: list[dict], add_generation_prompt: bool = False) -> tuple[list[int], list[int]]:
    """Returns token ids and a mask that is 1 only on tokens the model must learn to produce."""
    ids, mask = [tok.bos], [0]
    end = tok.id("<|end|>")
    for m in messages:
        role = m["role"]
        if role not in ROLES:
            raise ValueError(f"Unknown role: {role}")
        body = tok.encode(m["content"].strip())
        ids.append(tok.id(ROLES[role]))
        mask.append(0)
        ids.extend(body + [end])
        learn = 1 if role == "assistant" else 0
        mask.extend([learn] * (len(body) + 1))
    if add_generation_prompt:
        ids.append(tok.id("<|assistant|>"))
        mask.append(0)
    return ids, mask
