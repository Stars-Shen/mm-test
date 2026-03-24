#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/wqshen/mm-test/code"
BASE2_DIR="$ROOT_DIR/baseline2"

--n-splits 5 即采用5折交叉验证，并且是在day level进行划分。

python3 "$BASE2_DIR/train.py" --target label_daily_stress --epochs 120 --batch-size 8 --lr 1e-3 --n-splits 5
python3 "$BASE2_DIR/train.py" --target label_stai1 --epochs 120 --batch-size 8 --lr 1e-3 --n-splits 5

echo "[DONE] baseline2 两个任务完成。输出目录: /home/wqshen/mm-test/code/outputs_baseline2"
