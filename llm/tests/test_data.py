import json

import numpy as np

from aurora_llm.config import load_config
from aurora_llm.data.clean import doc_hash, normalize, quality_ok, scrub_pii
from aurora_llm.data.dataset import TokenStream
from aurora_llm.data.prepare import prepare
from aurora_llm.tokenizer import Tok, train_tokenizer
from aurora_llm.chat import encode_dialog
from aurora_llm.train_utils import lr_at

TEXT = ("Инвестиции — это вложение денег с целью получить доход в будущем. "
        "Облигации обычно спокойнее акций, а фонды распределяют риск между многими компаниями. ") * 6


def test_normalize_and_pii():
    t = normalize("Привет,  мир!\r\n\n\n\nПочта ivan@mail.ru, тел. +7 (912) 345-67-89")
    assert "  " not in t and "\n\n\n" not in t
    s = scrub_pii(t)
    assert "<email>" in s and "<phone>" in s and "ivan@" not in s


def test_quality_filter():
    assert quality_ok(TEXT)
    assert not quality_ok("1 2 3 4 5 " * 100)
    assert not quality_ok("коротко")


def test_dedup_hash_ignores_case_and_punctuation():
    assert doc_hash("Привет, мир!") == doc_hash("привет мир")


def test_tokenizer_roundtrip_and_prepare(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    docs = [TEXT + f" Документ номер {i}." for i in range(30)] + [TEXT + " Документ номер 0."]
    (raw / "a.jsonl").write_text("\n".join(json.dumps({"text": d}, ensure_ascii=False) for d in docs), encoding="utf-8")
    tok_path = tmp_path / "tok.json"
    train_tokenizer(docs, 400, tok_path)
    tok = Tok(tok_path)
    s = "Облигации спокойнее акций. Ёлка 😀"
    assert tok.decode(tok.encode(s)) == s  # byte-level: nothing is ever lost
    meta = prepare(str(raw), str(tok_path), str(tmp_path / "proc"), val_fraction=0.1, min_chars=50)
    assert meta["dropped_duplicate"] == 1
    assert meta["tokens"]["train"] > 0 and meta["tokens"]["val"] > 0
    ts = TokenStream(tmp_path / "proc", "train")
    x, y = ts.batch(2, 16, "cpu")
    assert x.shape == (2, 16) and np.array_equal(x[:, 1:].numpy(), y[:, :-1].numpy())


def test_chat_mask_only_on_assistant(tmp_path):
    tok_path = tmp_path / "tok.json"
    train_tokenizer([TEXT] * 5, 300, tok_path)
    tok = Tok(tok_path)
    ids, mask = encode_dialog(tok, [{"role": "user", "content": "Привет"}, {"role": "assistant", "content": "Здравствуйте"}])
    assert len(ids) == len(mask)
    learned = [i for i, m in zip(ids, mask) if m]
    assert tok.decode(learned[:-1]) == "Здравствуйте" and learned[-1] == tok.id("<|end|>")


def test_lr_schedule():
    assert lr_at(0, 1.0, 0.1, 10, 100) == 0.1
    assert abs(lr_at(9, 1.0, 0.1, 10, 100) - 1.0) < 1e-9
    assert abs(lr_at(100, 1.0, 0.1, 10, 100) - 0.1) < 1e-9
    assert lr_at(50, 1.0, 0.1, 10, 100) < 1.0


def test_configs_load():
    for name in ["tiny", "small", "base", "xl"]:
        cfg = load_config(f"configs/{name}.yaml")
        assert cfg.model.d_model % cfg.model.n_heads == 0
