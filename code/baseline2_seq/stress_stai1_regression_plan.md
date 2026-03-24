# Stress / STAI1 回归拟合计划（baseline2_seq）

## 1. 目标

基于当前 `train_regression.py`，完成以下两个回归任务：

- `stress`（对应目标字段：`y_daily_stress` 或 `label_daily_stress`）
- `stai1`（对应目标字段：`y_stai1` 或 `label_stai1`）

并输出可复现、可比较的验证结果（MAE / MSE）。

---

## 2. 当前脚本能力概览

`train_regression.py` 已具备：

- 单目标回归训练（通过 `--target` 切换目标）
- `structured` / `event` 两种输入格式
- mask 回归损失（仅在有效天、有效标签上计算）
- 早停（early stopping）
- 学习率调度（`ReduceLROnPlateau`）
    --默认情况：如果你用的是常见的 StepLR / MultiStepLR，放 epoch 结束
    --如果是 warmup / cosine / one-cycle 这类细粒度调度：放 每个 training step 结束
    --如果是 ReduceLROnPlateau：放 验证结束后
- best checkpoint 保存

说明：脚本内部已处理命名映射：

- `structured` 下优先使用 `y_*`
- `event` 下优先使用 `label_*`

---

## 3. 实验步骤

### Step 1：先跑通单任务基线

建议先使用 `structured` 格式完成两个独立模型：

#### 3.1 训练 stress

```bash
python train_regression.py \
  --data_format structured \
  --target y_daily_stress \
  --epochs 100 \
  --batch_size 4 \
  --lr 1e-4 \
  --seed 42
```

#### 3.2 训练 stai1

```bash
python train_regression.py \
  --data_format structured \
  --target y_stai1 \
  --epochs 100 \
  --batch_size 4 \
  --lr 1e-4 \
  --seed 42
```

训练完成后会在 `--save_dir` 下生成 best 模型，例如：

- `best_structured_y_daily_stress.pt`
- `best_structured_y_stai1.pt`

---

### Step 2：提升结果稳定性（多随机种子）

避免只看一次随机划分，建议至少跑 3 个 seed：

- `42`
- `123`
- `2024`

对每个任务统计：

- 验证集 MAE（越低越好）
- 验证集 MSE/val_loss（越低越好）

最后报告均值 ± 标准差。

---

### Step 3：对比输入格式（可选）

在 `structured` 跑稳后，可用同样配置对比 `event`：

#### stress（event）

```bash
python train_regression.py \
  --data_format event \
  --target label_daily_stress \
  --epochs 100 \
  --batch_size 4 \
  --lr 1e-4 \
  --seed 42
```

#### stai1（event）

```bash
python train_regression.py \
  --data_format event \
  --target label_stai1 \
  --epochs 100 \
  --batch_size 4 \
  --lr 1e-4 \
  --seed 42
```

---

## 4. 调参优先级（建议顺序）

优先从影响最大的参数开始：

1. 学习率：`1e-3` / `5e-4` / `1e-4`
2. 隐层维度：`hidden_dim=128` / `256`
3. batch size：`4` / `8` / `16`（受显存约束）
4. 早停耐心：`20~30`

建议一次只改一个关键参数，便于分析收益来源。

---

## 5. 结果记录模板

可按下表整理：

| data_format | target | seed |

| structured | y_daily_stress | 42 |：
fold1: best_val=76.0503, best_val_mae=6.9014
fold2: best_val=393.5393, best_val_mae=15.7298
fold3: best_val=159.9219, best_val_mae=10.1886
fold4: best_val=29.3756, best_val_mae=4.1074
fold5: best_val=497.8279, best_val_mae=18.0846
mean best_val=231.3430 ± 182.9368
mean best_val_mae=11.0024 ± 5.2446

| structured | y_daily_stress | 123 |  |  |  |
fold1: best_val=402.7230, best_val_mae=16.8133
fold2: best_val=250.6100, best_val_mae=13.6738
fold3: best_val=124.2226, best_val_mae=9.7964
fold4: best_val=138.2469, best_val_mae=9.9660
fold5: best_val=551.9777, best_val_mae=15.5840
mean best_val=293.5560 ± 163.3024
mean best_val_mae=13.1667 ± 2.8636


| structured | y_daily_stress | 2024 |  |  |  |
fold1: best_val=402.7230, best_val_mae=16.8133
fold2: best_val=250.6100, best_val_mae=13.6738
fold3: best_val=124.2226, best_val_mae=9.7964
fold4: best_val=138.2469, best_val_mae=9.9660
fold5: best_val=551.9777, best_val_mae=15.5840
mean best_val=293.5560 ± 163.3024
mean best_val_mae=13.1667 ± 2.8636


| structured | y_stai1 | 42 |  |  |  |
fold1: best_val=402.7230, best_val_mae=16.8133
fold2: best_val=250.6100, best_val_mae=13.6738
fold3: best_val=124.2226, best_val_mae=9.7964
fold4: best_val=138.2469, best_val_mae=9.9660
fold5: best_val=551.9777, best_val_mae=15.5840
mean best_val=293.5560 ± 163.3024
mean best_val_mae=13.1667 ± 2.8636


| structured | y_stai1 | 123 |  |  |  |
fold1: best_val=78.4365, best_val_mae=6.6323
fold2: best_val=119.7038, best_val_mae=8.5659
fold3: best_val=123.7915, best_val_mae=9.0212
fold4: best_val=262.8040, best_val_mae=13.1766
fold5: best_val=106.1660, best_val_mae=8.7631
mean best_val=138.1803 ± 64.3020
mean best_val_mae=9.2318 ± 2.1460


| structured | y_stai1 | 2024 |  |  |  |
fold1: best_val=111.1832, best_val_mae=6.7437
fold2: best_val=92.1100, best_val_mae=6.8394
fold3: best_val=182.8915, best_val_mae=11.2319
fold4: best_val=106.3009, best_val_mae=8.2493
fold5: best_val=80.5802, best_val_mae=7.5067
mean best_val=114.6132 ± 35.7978
mean best_val_mae=8.1142 ± 1.6499


| event | label_daily_stress | 42 |  |  |  |
fold1: best_val=139.1832, best_val_mae=6.7437
fold2: best_val=92.1100, best_val_mae=6.8394
fold3: best_val=182.8915, best_val_mae=11.2319
fold4: best_val=106.3009, best_val_mae=8.2493
fold5: best_val=80.5802, best_val_mae=7.5067
mean best_val=114.6132 ± 35.7978
mean best_val_mae=8.1142 ± 1.6499


| event | label_daily_stress | 123 |  |  |  |
fold1: best_val=432.8883, best_val_mae=18.2808
fold2: best_val=154.3623, best_val_mae=10.5617
fold3: best_val=71.9049, best_val_mae=6.7147
fold4: best_val=32.3759, best_val_mae=4.0934
fold5: best_val=251.2963, best_val_mae=12.1147
mean best_val=188.5656 ± 143.3901
mean best_val_mae=10.3530 ± 4.8666

| event | label_daily_stress | 2024 |  |  |  |


| event | label_stai1 | 42 |  |  |  |



| event | label_stai1 | 123 |  |  |  |


| event | label_stai1 | 2024 |  |  |  |

最终补充两行汇总：

- stress: MAE mean ± std, MSE mean ± std
- stai1: MAE mean ± std, MSE mean ± std

---

## 6. 已落地的日内建模改造（2026-03）

### 6.1 Structured 模态（小时桶）

已从“hour 维直接均值池化”升级为：

- `Linear(F->E)`
- `hour-of-day embedding`
- `1D CNN`（沿 hour 维）
- `masked attention pooling` 得到 day embedding

并新增/支持字段：

- `rr_hour_mask` / `act_hour_mask` / `sleep_hour_mask`，形状 `[B,D,H]`
- 若 `npz` 中暂无 hour 级 mask，则自动退化为：`day mask` 扩展到整小时（向后兼容）

对应文件：

- `dataset_structured.py`
- `model_multimodal_regression.py`
- `test_dataloader_shapes.py`

### 6.2 Event 模态（事件序列）

已从“step masked mean”升级为：

- `Linear(F->E)`
- 事件时间特征投影后相加（当前实现：`[delta_time, since_day_start]`）
- `TransformerEncoder`
- `masked attention pooling` 得到 day embedding

并新增字段：

- `rr_time_feat` / `act_time_feat` / `sleep_time_feat`，形状 `[B,D,T,2]`

对应文件：

- `dataset_event.py`
- `model_multimodal_regression.py`
- `test_dataloader_shapes.py`

### 6.3 兼容性说明

- 训练入口 `train_regression.py` 无需新增参数即可运行。
- 当 batch 中不存在新增字段时，模型内部有默认回退逻辑（零时间特征或 day mask 扩展）。
- 旧 checkpoint 因编码器结构变更通常不能直接加载到新模型参数（建议重新训练）。

## 7. 下一步（进阶）

如果单任务结果稳定，可考虑多任务联合训练（一个模型同时预测 stress + stai1）：

- 模型输出两个头：`pred_stress`、`pred_stai1`
- 总损失：`L = λ1 * L_stress + λ2 * L_stai1`
- 先试 `λ1=λ2=1`，再根据验证集表现调权重

预期收益：利用两个心理标签的相关性，有机会提升泛化能力。

---

## 7. 验收标准（建议）

最小可用版本：

1. 两个任务各自产出 best checkpoint
2. 每个任务至少 3 个 seed 结果
3. 输出一份对比结论（哪个 data_format 更优、当前最佳参数组合）

达到以上标准后，再进入多任务联合模型阶段。
## 8. 高频 RR 与跨模态时序优化（新增codex）

目标：避免 `event_max_tokens` 等间隔抽样造成关键变化丢失，同时显式建模跨模态（RR-Activity-Sleep）时序关系。

### 8.1 RR 从“点采样”改为“Patch 序列”

问题：RR 高频点数很大，直接下采样到固定 token 容易漏掉短时波动峰值与恢复过程。  
方案：按时间切片形成 patch（例如 30s / 60s / 120s），每个 patch 提取统计与动态特征，再以 patch 为 token。

每个 RR patch 建议特征：

- `rr_count, ibi_mean, ibi_std, ibi_min, ibi_max`
- `rmssd, sdnn, pnn50, hr_mean`
- `slope_mean`（patch 内斜率）
- `delta_to_prev_patch`（与上一 patch 的差分）
- `zscore_to_personal_baseline`（相对个体基线偏离）

收益：

- token 数大幅下降，训练稳定
- 保留局部生理变化语义（比等间隔抽点更鲁棒）
- 支持更长上下文（更容易看见“反应-恢复”链条）

### 8.2 模态内“变化层”（Delta Features）

在 RR / Activity / Sleep 三个分支都增加变化特征层：

- 一阶差分：`x_t - x_{t-1}`
- 滑窗偏离：`x_t - mean(x_{t-k:t-1})`
- 短期/长期比值：`short_window / long_window`

解释：压力/焦虑更常体现为“偏离与变化”，不是绝对值本身。

### 8.3 跨模态关系建模（显式 + 学习式）

显式关系特征（建议先做，样本小更稳）：

- 活动后恢复：`HRV_recovery_after_activity`
- 睡前激活：`pre_sleep_hr_or_hrv`
- 日内耦合：`corr(activity_intensity, hr_inst)`（分时段）

学习式关系（再做）：

- Cross-Attention：`RR <- Activity`、`RR <- Sleep`
- 门控融合：以 `Activity` 强度作为门控，抑制运动引起的“假压力高心率”
- 双层时序：模态内 encoder + 跨模态时序 encoder（day-level GRU/Transformer）

### 8.4 可执行实验路线（按优先级）

1. `Baseline-A`：当前 event + 等间隔抽样  
2. `Baseline-B`：RR patch token + 其余模态保持不变  
3. `Model-C`：B + Delta Features  
4. `Model-D`：C + 显式跨模态关系特征  
5. `Model-E`：D + Cross-Attention 融合

评估要求：

- 按 `user_id` 分组划分（避免泄漏）
- 统一汇报 `MAE/MSE`（stress / stai1 / stai2）
- 做消融：去掉 delta、去掉跨模态关系，观察性能变化

### 8.5 与相关方法的结合点

- TS2Vec 思路：可先做无监督预训练（模态内表征），再微调回归头
- PatchTST 思路：用 patch token 替代点级 token，是本项目最直接可落地的升级
- PhysioFormer 类思路：利用 cross-attention 对齐活动与心率变化，减弱运动混淆
- TCN 思路：在小样本场景下可作为更稳健 backbone 对照组

### 8.6 实施建议（本仓库）

- 在 `preprocess_multimodal_events.py` 增加 RR patch 化导出（新文件或开关）
- 在 `dataset_event.py` 增加 `rr_patch_time_feat` 支持
- 在 `model_multimodal_regression.py` 保持现有接口，新增可切换 encoder：
  - `EventPointEncoder`（现有）
  - `EventPatchEncoder`（新增）
- 在 `train_regression.py` 增加开关参数：
  - `--rr_use_patch`
  - `--rr_patch_sec`
  - `--use_delta_features`
  - `--use_cross_modal_attn`
