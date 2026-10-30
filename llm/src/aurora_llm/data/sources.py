"""Reading raw documents: plain .txt files and .jsonl files with a "text" field."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def iter_documents(raw_dir: str | Path) -> Iterator[str]:
    raw_dir = Path(raw_dir)
    files = sorted(p for p in raw_dir.rglob("*") if p.suffix in {".txt", ".jsonl"})
    if not files:
        raise FileNotFoundError(f"No .txt or .jsonl files in {raw_dir}. Run scripts/download_sample.py first.")
    for path in files:
        if path.suffix == ".txt":
            # Two blank lines separate documents inside one .txt file.
            for doc in path.read_text(encoding="utf-8", errors="replace").split("\n\n\n"):
                if doc.strip():
                    yield doc
        else:
            with path.open(encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        text = json.loads(line).get("text", "")
                    except json.JSONDecodeError:
                        continue
                    if text:
                        yield text
