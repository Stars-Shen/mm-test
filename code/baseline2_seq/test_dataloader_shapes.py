from __future__ import annotations

"""
快速测试两种数据组织方式的 DataLoader batch 形状。

运行：
python /home/wqshen/mm-test/code/baseline2_seq/test_dataloader_shapes.py
"""

from pathlib import Path

from torch.utils.data import DataLoader

from dataset_structured import StructuredDataset, structured_collate_fn
from dataset_event import EventDataset, event_collate_fn


BASE = Path('/home/wqshen/mm-test/code/baseline2_seq/processed')


def test_structured_loader(batch_size: int = 4) -> None:
    npz_path = BASE / 'multimodal_4d_tensors.npz'
    ds = StructuredDataset(str(npz_path))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=structured_collate_fn)

    print('\n[STRUCTURED] one epoch...')
    #day mask代表用户在该天是否有任何模态数据 [B,D]
    #rr mask表示在某天该模态是否有数据[B,D]
    for step, batch in enumerate(dl):
        print(f'-- step={step}')
        print('rr shape              =', batch['rr'].shape)
        print('act shape             =', batch['act'].shape)
        print('sleep shape           =', batch['sleep'].shape)
        print('day_mask shape        =', batch['day_mask'].shape)
        print('rr_mask shape         =', batch['rr_mask'].shape)
        print('act_mask shape        =', batch['act_mask'].shape)
        print('sleep_mask shape      =', batch['sleep_mask'].shape)
        print('rr_hour_mask shape    =', batch['rr_hour_mask'].shape)
        print('act_hour_mask shape   =', batch['act_hour_mask'].shape)
        print('sleep_hour_mask shape =', batch['sleep_hour_mask'].shape)
        print('y_stress shape        =', batch['y_daily_stress'].shape)
        print('y_stai1 shape         =', batch['y_stai1'].shape)
        print('user_id len           =', len(batch['user_id']))


def test_event_loader(batch_size: int = 4) -> None:
    npz_path = BASE / 'multimodal_event_sequences.npz'
    idx_path = BASE / 'multimodal_event_index.csv'
    ds = EventDataset(str(npz_path), str(idx_path))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=event_collate_fn)

    print('\n[EVENT] one epoch... (expect [B, D, T, F])')

# step 级 mask（天内时间步），形状 [B, D, T]
# rr_step_mask
# act_step_mask
# sleep_step_mask
# 含义：某天内第 t 个事件/时间步是否真实存在（不是 padding）。

# day 级 mask（按天），形状 [B, D]
# rr_day_mask：这天是否有 RR
# act_day_mask：这天是否有 ACT
# sleep_day_mask：这天是否有 SLEEP
# day_mask：这天是否至少有一种模态（rr_has or act_has or sleep_has）

# feat时间特征当前实现为： rr_time_feat / act_time_feat / sleep_time_feat，形状 [D,T,2]
# 第 0 维：delta_time（归一化后相邻差分）
# 第 1 维：since_day_start（0~1 线性位置）
# collate 后输出 [B,D,T,2]。
    for step, batch in enumerate(dl):
        print(f'-- step={step}')
        print('rr shape              =', batch['rr'].shape)
        print('act shape             =', batch['act'].shape)
        print('sleep shape           =', batch['sleep'].shape)
        print('rr_step_mask shape    =', batch['rr_step_mask'].shape)
        print('act_step_mask shape   =', batch['act_step_mask'].shape)
        print('sleep_step_mask shape =', batch['sleep_step_mask'].shape)
        print('rr_time_feat shape    =', batch['rr_time_feat'].shape)
        print('act_time_feat shape   =', batch['act_time_feat'].shape)
        print('sleep_time_feat shape =', batch['sleep_time_feat'].shape)
        print('day_mask shape        =', batch['day_mask'].shape)
        print('rr_day_mask shape     =', batch['rr_day_mask'].shape)
        print('act_day_mask shape    =', batch['act_day_mask'].shape)
        print('sleep_day_mask shape  =', batch['sleep_day_mask'].shape)
        print('days shape            =', batch['days'].shape)
        print('y_stress shape        =', batch['label_daily_stress'].shape)
        print('y_stai1 shape         =', batch['label_stai1'].shape)
        print('user_id len           =', len(batch['user_id']))


if __name__ == '__main__':
    test_structured_loader(batch_size=4)
    test_event_loader(batch_size=4)
