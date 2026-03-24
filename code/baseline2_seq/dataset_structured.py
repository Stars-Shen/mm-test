from __future__ import annotations

"""
Structured（小时桶）数据加载。

输入：
- multimodal_4d_tensors.npz

单个样本（按user）输出：
- rr/act/sleep: [day, hour, feat]
- day_mask: [day]
- rr_mask/act_mask/sleep_mask: [day]
- y_daily_stress/y_stai1/y_stai2: [day]
- user_id
"""

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class StructuredSample:
    rr: np.ndarray
    act: np.ndarray
    sleep: np.ndarray
    day_mask: np.ndarray
    rr_mask: np.ndarray
    act_mask: np.ndarray
    sleep_mask: np.ndarray
    rr_hour_mask: np.ndarray
    act_hour_mask: np.ndarray
    sleep_hour_mask: np.ndarray
    y_daily_stress: np.ndarray
    y_stai1: np.ndarray
    y_stai2: np.ndarray
    user_id: str


class StructuredDataset:
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        self.rr = data['rr_tensor']
        self.act = data['act_tensor']
        self.sleep = data['sleep_tensor']

        self.day_mask = data['day_mask']
        self.rr_mask = data['rr_day_mask']
        self.act_mask = data['act_day_mask']
        self.sleep_mask = data['sleep_day_mask']

        self.y_daily_stress = data['y_daily_stress']
        self.y_stai1 = data['y_stai1']
        self.y_stai2 = data['y_stai2'] if 'y_stai2' in data.files else np.full_like(self.y_stai1, np.nan, dtype=np.float32)
        self.users = data['users'].tolist()

        # 如果npz中没有小时级mask，则退化为“该模态该天有效 -> 全小时有效”。
        self.rr_hour_mask = data['rr_hour_mask'] if 'rr_hour_mask' in data.files else np.repeat(self.rr_mask[..., None], self.rr.shape[2], axis=2)
        self.act_hour_mask = data['act_hour_mask'] if 'act_hour_mask' in data.files else np.repeat(self.act_mask[..., None], self.act.shape[2], axis=2)
        self.sleep_hour_mask = data['sleep_hour_mask'] if 'sleep_hour_mask' in data.files else np.repeat(self.sleep_mask[..., None], self.sleep.shape[2], axis=2)

    def __len__(self) -> int:
        return self.rr.shape[0]

    def __getitem__(self, idx: int) -> StructuredSample:
        return StructuredSample(
            rr=self.rr[idx],
            act=self.act[idx],
            sleep=self.sleep[idx],
            day_mask=self.day_mask[idx],
            rr_mask=self.rr_mask[idx],
            act_mask=self.act_mask[idx],
            sleep_mask=self.sleep_mask[idx],
            rr_hour_mask=self.rr_hour_mask[idx],
            act_hour_mask=self.act_hour_mask[idx],
            sleep_hour_mask=self.sleep_hour_mask[idx],
            y_daily_stress=self.y_daily_stress[idx],
            y_stai1=self.y_stai1[idx],
            y_stai2=self.y_stai2[idx],
            user_id=str(self.users[idx]),
        )


def structured_collate_fn(batch: list[StructuredSample]) -> Dict[str, Any]:
    """输出批次维度：[B, day, hour, feat]。"""
    return {
        'rr': np.stack([b.rr for b in batch], axis=0),
        'act': np.stack([b.act for b in batch], axis=0),
        'sleep': np.stack([b.sleep for b in batch], axis=0),
        'day_mask': np.stack([b.day_mask for b in batch], axis=0),
        'rr_mask': np.stack([b.rr_mask for b in batch], axis=0),
        'act_mask': np.stack([b.act_mask for b in batch], axis=0),
        'sleep_mask': np.stack([b.sleep_mask for b in batch], axis=0),
        'rr_hour_mask': np.stack([b.rr_hour_mask for b in batch], axis=0),
        'act_hour_mask': np.stack([b.act_hour_mask for b in batch], axis=0),
        'sleep_hour_mask': np.stack([b.sleep_hour_mask for b in batch], axis=0),
        'y_daily_stress': np.stack([b.y_daily_stress for b in batch], axis=0),
        'y_stai1': np.stack([b.y_stai1 for b in batch], axis=0),
        'y_stai2': np.stack([b.y_stai2 for b in batch], axis=0),
        'user_id': [b.user_id for b in batch],
    }
