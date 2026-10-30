"""Text cleaning and quality filters. Data quality matters more than model size."""
from __future__ import annotations

import hashlib
import re
import unicodedata

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES = re.compile(r"[ \t ]+")
_NEWLINES = re.compile(r"\n{3,}")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?:\+7|8)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}")
_CARD = re.compile(r"\b(?:\d[ -]?){15,18}\d\b")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _CTRL.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    return _NEWLINES.sub("\n\n", text).strip()


def scrub_pii(text: str) -> str:
    """Replace e-mails, phone numbers and card-like numbers so the model never learns them."""
    text = _EMAIL.sub("<email>", text)
    text = _PHONE.sub("<phone>", text)
    return _CARD.sub("<card>", text)


def quality_ok(text: str, min_chars: int = 200) -> bool:
    """Cheap heuristics in the spirit of open web-corpus filters."""
    if len(text) < min_chars:
        return False
    letters = sum(ch.isalpha() for ch in text)
    if letters / len(text) < 0.6:                     # tables, code dumps, number lists
        return False
    words = text.split()
    if not words:
        return False
    mean_len = sum(len(w) for w in words) / len(words)
    if not 3 <= mean_len <= 12:                       # gibberish or glued text
        return False
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if lines and len(set(lines)) / len(lines) < 0.7:  # repeated boilerplate lines
        return False
    return True


def doc_hash(text: str) -> str:
    key = re.sub(r"\W+", " ", text.lower()).strip()
    return hashlib.sha1(key.encode("utf-8")).hexdigest()
