#!/usr/bin/env bash
set -euo pipefail

# =========================
# 用户配置区（直接改这里）
# =========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 训练模式: event 或 structured或hybrid
DATA_FORMAT="hybrid"

# GPU/DDP
GPU_IDS="1,2"
NPROC_PER_NODE=2

# 是否先重建 event patch 数据（1=是，0=否）
RUN_EVENT_PREPROCESS=0
WINDOW_MIN=5

# 目标任务（event 用 label_*；structured 用 y_*）
TARGETS=(
  "y_daily_stress"
  "y_stai1"
  "y_stai2"
)

# 训练参数
EPOCHS=150
BATCH_SIZE=2
LR=1e-4
NUM_WORKERS=0
K_FOLDS=5
SEEDS=(42 123 2024)

# event 参数
EVENT_N_HEADS=8
EVENT_N_LAYERS=3
EVENT_DROPOUT=0.2
EVENT_MAX_TOKENS=1024
FUSION_TEMPORAL_LAYERS=1
# hybrid 双分支学习式对齐参数
HYBRID_ALIGN_HEADS=4
HYBRID_ALIGN_DROPOUT=0.1

# 输出
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# =========================
# 可选预处理
# =========================
if [[ "$DATA_FORMAT" == "event" && "$RUN_EVENT_PREPROCESS" -eq 1 ]]; then
  echo "[INFO] rebuild event patches... window_min=$WINDOW_MIN"
  python preprocess_multimodal_events.py --window_min "$WINDOW_MIN"
fi

# =========================
# 逐seed逐任务训练
# =========================
for SEED in "${SEEDS[@]}"; do
  for TARGET in "${TARGETS[@]}"; do
    TS="$(date +%Y%m%d_%H%M%S)"
    LOG_FILE="$LOG_DIR/train_${DATA_FORMAT}_${TARGET}_seed${SEED}_${TS}.log"
    echo "[INFO] start seed=$SEED target=$TARGET, log=$LOG_FILE"

    CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" train_regression.py \
      --data_format "$DATA_FORMAT" \
      --target "$TARGET" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --lr "$LR" \
      --num_workers "$NUM_WORKERS" \
      --k_folds "$K_FOLDS" \
      --seed "$SEED" \
      --event_n_heads "$EVENT_N_HEADS" \
      --event_n_layers "$EVENT_N_LAYERS" \
      --event_dropout "$EVENT_DROPOUT" \
      --event_max_tokens "$EVENT_MAX_TOKENS" \
      --fusion_temporal_layers "$FUSION_TEMPORAL_LAYERS" \
      --hybrid_align_heads "$HYBRID_ALIGN_HEADS" \
      --hybrid_align_dropout "$HYBRID_ALIGN_DROPOUT" \
      2>&1 | tee "$LOG_FILE"

    echo "[INFO] done seed=$SEED target=$TARGET"
    echo
  done
done

echo "[DONE] all targets finished. logs -> $LOG_DIR"
