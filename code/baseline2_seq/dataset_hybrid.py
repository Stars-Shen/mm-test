from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from dataset_event import EventDataset
from dataset_structured import StructuredDataset


@dataclass
class HybridSample:
    user_id: str
    # structured branch
    struct_rr: np.ndarray
    struct_act: np.ndarray
    struct_sleep: np.ndarray
    struct_day_mask: np.ndarray
    struct_rr_mask: np.ndarray
    struct_act_mask: np.ndarray
    struct_sleep_mask: np.ndarray
    struct_rr_hour_mask: np.ndarray
    struct_act_hour_mask: np.ndarray
    struct_sleep_hour_mask: np.ndarray
    # event branch (day-aligned to structured day slots)
    event_rr: np.ndarray
    event_act: np.ndarray
    event_sleep: np.ndarray
    event_rr_step_mask: np.ndarray
    event_act_step_mask: np.ndarray
    event_sleep_step_mask: np.ndarray
    event_rr_time_feat: np.ndarray
    event_act_time_feat: np.ndarray
    event_sleep_time_feat: np.ndarray
    event_rr_day_mask: np.ndarray
    event_act_day_mask: np.ndarray
    event_sleep_day_mask: np.ndarray
    # labels
    day_mask: np.ndarray
    y_daily_stress: np.ndarray
    y_stai1: np.ndarray
    y_stai2: np.ndarray
    label_daily_stress: np.ndarray
    label_stai1: np.ndarray
    label_stai2: np.ndarray


def _pad_batch_4d(x_list: List[np.ndarray], m_list: List[np.ndarray], feat_dim: int) -> tuple[np.ndarray, np.ndarray]:
    b = len(x_list)
    if b == 0:
        return np.zeros((0, 0, 0, feat_dim), dtype=np.float32), np.zeros((0, 0, 0), dtype=np.float32)
    d_max = max(x.shape[0] for x in x_list)
    t_max = max(x.shape[1] for x in x_list) if d_max > 0 else 0
    out = np.zeros((b, d_max, t_max, feat_dim), dtype=np.float32)
    mask = np.zeros((b, d_max, t_max), dtype=np.float32)
    for i, (x, m) in enumerate(zip(x_list, m_list)):
        d_i, t_i = x.shape[0], x.shape[1]
        if d_i > 0 and t_i > 0:
            out[i, :d_i, :t_i, :] = x
            mask[i, :d_i, :t_i] = m
    return out, mask


class HybridDataset:
    """
    双分支数据集：
    - structured 分支：规则网格序列（稳健）
    - event 分支：patch/事件序列（细粒度动态）
    """

    def __init__(self, structured_npz: str, event_npz: str, event_index_csv: str):
        self.structured = StructuredDataset(structured_npz)
        self.event = EventDataset(event_npz, event_index_csv)

        s_user_to_idx = {str(u): i for i, u in enumerate(self.structured.users)}
        e_user_to_idx = {str(g.loc[0, 'user_id']): i for i, g in enumerate(self.event.user_groups)}

        self.user_ids = [u for u in self.structured.users if str(u) in e_user_to_idx]
        self.index_pairs = [(s_user_to_idx[str(u)], e_user_to_idx[str(u)]) for u in self.user_ids]

    def __len__(self) -> int:
        return len(self.index_pairs)

    @staticmethod
    def _align_event_to_struct_days(event_arr: np.ndarray, event_days: np.ndarray, target_days: int) -> np.ndarray:
        if event_arr.ndim != 3:
            return np.zeros((target_days, 0, 0), dtype=np.float32)
        _, t, f = event_arr.shape
        out = np.zeros((target_days, t, f), dtype=np.float32)
        for src_i, day_val in enumerate(event_days.tolist()):
            d = int(day_val)
            if 1 <= d <= target_days:
                out[d - 1] = event_arr[src_i]
        return out

    @staticmethod
    def _align_event_day_vec(vec: np.ndarray, event_days: np.ndarray, target_days: int) -> np.ndarray:
        out = np.zeros((target_days,), dtype=np.float32)
        for src_i, day_val in enumerate(event_days.tolist()):
            d = int(day_val)
            if 1 <= d <= target_days:
                out[d - 1] = float(vec[src_i])
        return out

    def __getitem__(self, idx: int) -> HybridSample:
        s_idx, e_idx = self.index_pairs[idx]
        s = self.structured[s_idx]
        e = self.event[e_idx]

        d = int(s.rr.shape[0])
        days = e.days.astype(np.int64)

        event_rr = self._align_event_to_struct_days(e.rr, days, d)
        event_act = self._align_event_to_struct_days(e.act, days, d)
        event_sleep = self._align_event_to_struct_days(e.sleep, days, d)
        event_rr_step_mask = self._align_event_to_struct_days(e.rr_step_mask[..., None], days, d)[..., 0]
        event_act_step_mask = self._align_event_to_struct_days(e.act_step_mask[..., None], days, d)[..., 0]
        event_sleep_step_mask = self._align_event_to_struct_days(e.sleep_step_mask[..., None], days, d)[..., 0]
        event_rr_time_feat = self._align_event_to_struct_days(e.rr_time_feat, days, d)
        event_act_time_feat = self._align_event_to_struct_days(e.act_time_feat, days, d)
        event_sleep_time_feat = self._align_event_to_struct_days(e.sleep_time_feat, days, d)
        event_rr_day_mask = self._align_event_day_vec(e.rr_day_mask, days, d)
        event_act_day_mask = self._align_event_day_vec(e.act_day_mask, days, d)
        event_sleep_day_mask = self._align_event_day_vec(e.sleep_day_mask, days, d)

        # 弱对齐层：统一day槽位并形成联合可用mask
        union_day_mask = np.maximum(
            s.day_mask.astype(np.float32),
            np.maximum(event_rr_day_mask, np.maximum(event_act_day_mask, event_sleep_day_mask)),
        )

        return HybridSample(
            user_id=str(s.user_id),
            struct_rr=s.rr.astype(np.float32),
            struct_act=s.act.astype(np.float32),
            struct_sleep=s.sleep.astype(np.float32),
            struct_day_mask=s.day_mask.astype(np.float32),
            struct_rr_mask=s.rr_mask.astype(np.float32),
            struct_act_mask=s.act_mask.astype(np.float32),
            struct_sleep_mask=s.sleep_mask.astype(np.float32),
            struct_rr_hour_mask=s.rr_hour_mask.astype(np.float32),
            struct_act_hour_mask=s.act_hour_mask.astype(np.float32),
            struct_sleep_hour_mask=s.sleep_hour_mask.astype(np.float32),
            event_rr=event_rr.astype(np.float32),
            event_act=event_act.astype(np.float32),
            event_sleep=event_sleep.astype(np.float32),
            event_rr_step_mask=event_rr_step_mask.astype(np.float32),
            event_act_step_mask=event_act_step_mask.astype(np.float32),
            event_sleep_step_mask=event_sleep_step_mask.astype(np.float32),
            event_rr_time_feat=event_rr_time_feat.astype(np.float32),
            event_act_time_feat=event_act_time_feat.astype(np.float32),
            event_sleep_time_feat=event_sleep_time_feat.astype(np.float32),
            event_rr_day_mask=event_rr_day_mask.astype(np.float32),
            event_act_day_mask=event_act_day_mask.astype(np.float32),
            event_sleep_day_mask=event_sleep_day_mask.astype(np.float32),
            day_mask=union_day_mask.astype(np.float32),
            y_daily_stress=s.y_daily_stress.astype(np.float32),
            y_stai1=s.y_stai1.astype(np.float32),
            y_stai2=s.y_stai2.astype(np.float32),
            label_daily_stress=s.y_daily_stress.astype(np.float32),
            label_stai1=s.y_stai1.astype(np.float32),
            label_stai2=s.y_stai2.astype(np.float32),
        )


def hybrid_collate_fn(batch: List[HybridSample]) -> Dict[str, Any]:
    event_rr, event_rr_step_mask = _pad_batch_4d(
        [b.event_rr for b in batch], [b.event_rr_step_mask for b in batch], batch[0].event_rr.shape[-1]
    )
    event_act, event_act_step_mask = _pad_batch_4d(
        [b.event_act for b in batch], [b.event_act_step_mask for b in batch], batch[0].event_act.shape[-1]
    )
    event_sleep, event_sleep_step_mask = _pad_batch_4d(
        [b.event_sleep for b in batch], [b.event_sleep_step_mask for b in batch], batch[0].event_sleep.shape[-1]
    )
    event_rr_time_feat, _ = _pad_batch_4d(
        [b.event_rr_time_feat for b in batch], [b.event_rr_step_mask for b in batch], 2
    )
    event_act_time_feat, _ = _pad_batch_4d(
        [b.event_act_time_feat for b in batch], [b.event_act_step_mask for b in batch], 2
    )
    event_sleep_time_feat, _ = _pad_batch_4d(
        [b.event_sleep_time_feat for b in batch], [b.event_sleep_step_mask for b in batch], 2
    )

    return {
        'user_id': [b.user_id for b in batch],
        # structured branch
        'struct_rr': np.stack([b.struct_rr for b in batch], axis=0),
        'struct_act': np.stack([b.struct_act for b in batch], axis=0),
        'struct_sleep': np.stack([b.struct_sleep for b in batch], axis=0),
        'struct_day_mask': np.stack([b.struct_day_mask for b in batch], axis=0),
        'struct_rr_mask': np.stack([b.struct_rr_mask for b in batch], axis=0),
        'struct_act_mask': np.stack([b.struct_act_mask for b in batch], axis=0),
        'struct_sleep_mask': np.stack([b.struct_sleep_mask for b in batch], axis=0),
        'struct_rr_hour_mask': np.stack([b.struct_rr_hour_mask for b in batch], axis=0),
        'struct_act_hour_mask': np.stack([b.struct_act_hour_mask for b in batch], axis=0),
        'struct_sleep_hour_mask': np.stack([b.struct_sleep_hour_mask for b in batch], axis=0),
        # event branch
        'event_rr': event_rr,
        'event_act': event_act,
        'event_sleep': event_sleep,
        'event_rr_step_mask': event_rr_step_mask,
        'event_act_step_mask': event_act_step_mask,
        'event_sleep_step_mask': event_sleep_step_mask,
        'event_rr_time_feat': event_rr_time_feat,
        'event_act_time_feat': event_act_time_feat,
        'event_sleep_time_feat': event_sleep_time_feat,
        'event_rr_day_mask': np.stack([b.event_rr_day_mask for b in batch], axis=0),
        'event_act_day_mask': np.stack([b.event_act_day_mask for b in batch], axis=0),
        'event_sleep_day_mask': np.stack([b.event_sleep_day_mask for b in batch], axis=0),
        # labels/masks
        'day_mask': np.stack([b.day_mask for b in batch], axis=0),
        'y_daily_stress': np.stack([b.y_daily_stress for b in batch], axis=0),
        'y_stai1': np.stack([b.y_stai1 for b in batch], axis=0),
        'y_stai2': np.stack([b.y_stai2 for b in batch], axis=0),
        'label_daily_stress': np.stack([b.label_daily_stress for b in batch], axis=0),
        'label_stai1': np.stack([b.label_stai1 for b in batch], axis=0),
        'label_stai2': np.stack([b.label_stai2 for b in batch], axis=0),
    }
