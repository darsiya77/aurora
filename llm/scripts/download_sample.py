"""Download a sample of Russian Wikipedia into data/raw/ (no extra libraries needed).

  python scripts/download_sample.py --docs 3000
For real training use full dumps or open web corpora; see docs/DATA.md.
"""
import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://datasets-server.huggingface.co/rows"


def fetch(offset: int, length: int, dataset: str, config: str) -> list[dict]:
    q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": "train", "offset": offset, "length": length})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{API}?{q}", timeout=60) as r:
                return [row["row"] for row in json.load(r)["rows"]]
        except Exception as e:  # network hiccups: back off and retry
            wait = 2 ** attempt
            print(f"offset {offset}: {e}; retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Could not download rows at offset {offset}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=3000)
    ap.add_argument("--dataset", default="wikimedia/wikipedia")
    ap.add_argument("--config", default="20231101.ru")
    ap.add_argument("--out", default="data/raw/wiki_ru.jsonl")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for offset in range(0, a.docs, 100):
            for row in fetch(offset, min(100, a.docs - offset), a.dataset, a.config):
                f.write(json.dumps({"text": row["text"], "source": row.get("url", "")}, ensure_ascii=False) + "\n")
                n += 1
            print(f"{n}/{a.docs}", end="\r", flush=True)
    print(f"\nsaved {n} documents to {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
