#!/usr/bin/env bash
# Full run on a rented GPU server (1 x H100 80 GB for Moonlight-16B-A3B-Instruct).
#   bash scripts/assistant_gpu.sh                      # Moonlight (default)
#   CONFIG=configs/assistant/kimi-linear.yaml bash scripts/assistant_gpu.sh   # 2 x H100
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
CONFIG=${CONFIG:-configs/assistant/moonlight.yaml}
BASE=$(python -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['base_model'])")
OUT=$(python -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['out_dir'])")

pip install -r requirements.txt -r requirements-assistant.txt
python -m aurora_assistant.data --out data/assistant --train 4000 --eval 300
python -m aurora_assistant.evaluate --base "$BASE" --trust-remote-code --limit 300 --out "$OUT-eval-before.json"
python -m aurora_assistant.train --config "$CONFIG"
python -m aurora_assistant.evaluate --base "$BASE" --trust-remote-code --adapter "$OUT/adapter" --limit 300 --out "$OUT-eval-after.json"
python -m aurora_assistant.merge --base "$BASE" --trust-remote-code --adapter "$OUT/adapter" --out "$OUT/merged"
echo "Done. Chat: python -m aurora_assistant.serve --base $OUT/merged --trust-remote-code --host 0.0.0.0"
