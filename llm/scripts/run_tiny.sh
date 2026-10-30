#!/usr/bin/env bash
# Whole pipeline on a CPU in ~15 minutes: data -> tokenizer -> pretraining -> fine-tuning -> answer.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=${PYTHON:-python}

[ -s data/raw/wiki_ru.jsonl ] || $PY scripts/download_sample.py --docs 3000
$PY -m aurora_llm.tokenizer --raw data/raw --vocab 8000 --max-docs 3000
$PY -m aurora_llm.data.prepare --raw data/raw --out data/processed --val-fraction 0.02
$PY -m aurora_llm.train --config configs/tiny.yaml
$PY -m aurora_llm.eval.perplexity --ckpt runs/tiny/best.pt
$PY -m aurora_llm.generate --ckpt runs/tiny/best.pt --prompt "Россия — это" --max-new-tokens 60
$PY scripts/make_sft_sample.py
$PY -m aurora_llm.sft.train_sft --config configs/tiny.yaml
echo "Fine-tuned checkpoint: runs/tiny-sft/best.pt"
