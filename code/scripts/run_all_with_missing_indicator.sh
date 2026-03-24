#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/wqshen/mm-test/code"
TRAIN_DIR="$ROOT_DIR/train"
OUT_DIR="$ROOT_DIR/outputs"
MODEL_NAME="rf"  # 可改为 ridge 或 rf

mkdir -p "$OUT_DIR"

echo "[1/3] 训练 Daily_stress (with missing indicator, model=$MODEL_NAME)..."
python3 "$TRAIN_DIR/train.py" \
  --data-path "$ROOT_DIR/processed/mmash_user_level_features.csv" \
  --output-dir "$OUT_DIR" \
  --target label_daily_stress \
  --model-name "$MODEL_NAME"

echo "[2/3] 训练 STAI1 (with missing indicator, model=$MODEL_NAME)..."
python3 "$TRAIN_DIR/train.py" \
  --data-path "$ROOT_DIR/processed/mmash_user_level_features.csv" \
  --output-dir "$OUT_DIR" \
  --target label_stai1 \
  --model-name "$MODEL_NAME"

echo "[3/3] 汇总结果..."
python3 - <<'PY'
import json
from pathlib import Path
import pandas as pd

out_dir = Path('/home/wqshen/mm-test/code/outputs')
model = 'rf'  # 与脚本 MODEL_NAME 保持一致
files = [
    out_dir / f'summary_label_daily_stress_{model}.json',
    out_dir / f'summary_label_stai1_{model}.json',
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
        'setting': 'with_missing_indicator',
        'n_samples': obj.get('n_samples'),
        'n_users': obj.get('n_users'),
        'n_features_initial': obj.get('n_features_initial'),
        'n_features_final': obj.get('n_features_final'),
        'mae': cv.get('mae'),
        'rmse': cv.get('rmse'),
        'spearman': cv.get('spearman'),
    })

summary = pd.DataFrame(rows)
summary_path = out_dir / f'summary_all_tasks_{model}_with_mi.csv'
summary.to_csv(summary_path, index=False)
print('[OK] 汇总已保存:', summary_path)
print(summary)
PY

echo "[DONE] 全部任务完成。"
