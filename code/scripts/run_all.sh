#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/wqshen/mm-test/code"
TRAIN_DIR="$ROOT_DIR/train"
OUT_DIR="$ROOT_DIR/outputs"

mkdir -p "$OUT_DIR"

echo "[1/3] 训练 Daily_stress..."
python3 "$TRAIN_DIR/train.py" \
  --data-path "$ROOT_DIR/processed/mmash_user_level_features.csv" \
  --output-dir "$OUT_DIR" \
  --target label_daily_stress \
  --model-name rf\
  --no-missing-indicator

echo "[2/3] 训练 STAI1..."
python3 "$TRAIN_DIR/train.py" \
  --data-path "$ROOT_DIR/processed/mmash_user_level_features.csv" \
  --output-dir "$OUT_DIR" \
  --target label_stai1 \
  --model-name rf\
  --no-missing-indicator

echo "[3/3] 汇总结果..."
python3 - <<'PY'
import json
from pathlib import Path
import pandas as pd

out_dir = Path('/home/wqshen/mm-test/code/outputs')
files = [
    out_dir / 'summary_label_daily_stress_ridge.json',
    out_dir / 'summary_label_stai1_ridge.json',
]

rows = []
for f in files:
    if not f.exists():
        continue
    obj = json.loads(f.read_text(encoding='utf-8'))
    cv = obj.get('cv', {})
    rows.append({
        'target': obj.get('target'),
        'model': obj.get('model'),
        'n_samples': obj.get('n_samples'),
        'n_users': obj.get('n_users'),
        'n_features': obj.get('n_features'),
        'mae': cv.get('mae'),
        'rmse': cv.get('rmse'),
        'spearman': cv.get('spearman'),
    })

summary = pd.DataFrame(rows)
summary_path = out_dir / 'summary_all_tasks.csv'
summary.to_csv(summary_path, index=False)
print('[OK] 汇总已保存:', summary_path)
print(summary)
PY

echo "[DONE] 全部任务完成。"
