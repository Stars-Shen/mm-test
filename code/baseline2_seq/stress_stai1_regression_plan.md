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

### 8.7 当前已落地实现（代码状态）

已完成的改造：

- `preprocess_multimodal_events.py` 已升级为统一窗口 patch 预处理（默认 `--window_min 5`）
- RR / Activity / Sleep 都输出 patch 序列（变长）
- 增加 patch 内动态特征（slope）与 patch 间动态特征（delta）
- `dataset_event.py` 的时间特征改为优先使用真实时间列（`patch_center_min`）构建相对时间编码

字段含义（关键）：

- `slope`：patch 内趋势（段内变化），如 `ibi_slope` / `hr_slope`
  - 含义：该 patch 内信号是上升还是下降，以及变化快慢
- `delta`：相邻 patch 的差分（段间变化），如 `delta_ibi_mean` / `delta_hr_mean`
  - 含义：当前 patch 相对上一 patch 的跳变幅度
- `time_feat=[delta_time, since_day_start]`
  - `delta_time`：相邻 token 的时间间隔（归一化）
  - `since_day_start`：token 在当天中的相对位置（归一化）

当前 patch 特征维度（window=5min）：

- RR token dim = 17
- ACT token dim = 19
- SLEEP token dim = 12

建议运行流程：

1. 重建 patch 数据
   - `python preprocess_multimodal_events.py --window_min 5`
2. 训练 event 模型
   - `torchrun --standalone --nproc_per_node=2 train_regression.py --data_format event --target label_daily_stress --num_workers 0 ...`

备注：
- 这版改造已将 RR 序列从“几万点级 token”降为“百级 patch token”，并保留关键前后变化语义。



3.25代码改动
1、新增双分支数据组织与弱对齐（按 day 对齐，模态内的融合，structured保稳定全局，event保细节）
dataset_hybrid.py
2、新增两级策略模型
级别1：弱对齐（dataset 统一 day 槽位）-》时间对齐
级别2：学习式对齐（每模态 cross-attention + 门控融合）-》语义对齐，该关注谁，和谁相关，门控控制应该相信谁多一点
model_multimodal_regression.py (line 272)
DualBranchAlignBlock + MultiModalHybridRegressor
3、训练入口支持 hybrid，且保留原 structured/event
train_regression.py (line 363)
新增 --data_format hybrid
新增 --hybrid_align_heads、--hybrid_align_dropout
修复 event npz 的 numpy 版本兼容问题（你之前遇到过）
dataset_event.py (line 41)
feat_names 读取失败会自动回填默认特征名，不再因为 numpy._core 崩溃
训练脚本模板也加了 hybrid 参数

不足：当前不能区分模态到底是真的缺失还是说一直在检测只是没有，比如user11的sleep模态，可能根本就没睡，而不是没检测-》 obs_mask：该模态当天是否可观测（是否有记录能力） event_flag：该窗口是否发生了睡眠/活动（0也有语义，不等于缺失）

    跨模态对齐配合模态内对齐

1. 模态表征（先做）

用“个体基线偏离”替代绝对值：x_t - personal_baseline，尤其对 RR/HRV 很有效。
保留 patch 的动态特征：delta + slope + rolling std，再加“恢复速度”特征（活动后 10/20/30 分钟 HRV 回升幅度）。
给每个模态加“缺失/可信度”指示器（sleep 稀疏时很重要），作为模型输入而不只是 mask。
对 RR 做轻量自监督预训练（mask 重建或下一 patch 预测），再微调回归头。
2. 融合方式（第二步）

你现在 hybrid 很对，下一步可加“时延融合”：让 RR<-Activity 的 attention 允许 ±K 个 patch 偏移（处理生理反应滞后）。
用门控融合加先验约束：活动强时降低“压力相关心率升高”的权重，减少运动混淆。
加跨分支一致性约束：structured 分支和 event 分支同一天 embedding 不要差太远（L2 consistency）。
3. Loss 设计（马上能见效）

主损失从纯 MSE 改为 Huber 或 0.7*MSE + 0.3*MAE（抗异常值更稳）。
多任务联合（stress + stai1 + stai2）并用不确定性加权，比单任务通常更稳。
对目标做 fold 内标准化训练、预测后反标准化，优化会更容易。
可加排序损失（pairwise rank），强化“高压日 > 低压日”的相对关系。
4. 训练策略（别忽略）

严格按 user_id 分组验证（你现在在做，保持）。
小样本时模型容量再降一点（embed_dim/heads），常比加深网络更好。
报告 MAE 为主、MSE 为辅；MSE 几百在量纲上并不奇怪。




