from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd


@dataclass
class UserEventSample:
    user_id: str
    days: np.ndarray
    rr: np.ndarray
    act: np.ndarray
    sleep: np.ndarray
    rr_step_mask: np.ndarray
    act_step_mask: np.ndarray
    sleep_step_mask: np.ndarray
    rr_time_feat: np.ndarray
    act_time_feat: np.ndarray
    sleep_time_feat: np.ndarray
    rr_day_mask: np.ndarray
    act_day_mask: np.ndarray
    sleep_day_mask: np.ndarray
    day_mask: np.ndarray
    label_daily_stress: np.ndarray
    label_stai1: np.ndarray
    label_stai2: np.ndarray


class UserEventDataset:
    def __init__(self, npz_path: str, index_csv: str):
        # DDP + DataLoader(num_workers>0) 下，直接持有 NpzFile 句柄容易触发并发解压错误。
        # 这里把 npz 全量读入内存并关闭文件句柄，避免 zlib 并发读失败。
        with np.load(npz_path, allow_pickle=True) as arr:
            self.arr = {}
            for k in arr.files:
                try:
                    self.arr[k] = arr[k]
                except ModuleNotFoundError:
                    # 兼容不同numpy版本之间 object pickle 路径变化（常见于 *_feat_names）
                    self.arr[k] = None
        self.index_df = pd.read_csv(index_csv)

        self._fill_missing_feat_names()

        self.rr_feat_dim = len(self.arr['rr_feat_names'])
        self.act_feat_dim = len(self.arr['act_feat_names'])
        self.sleep_feat_dim = len(self.arr['sleep_feat_names'])
        self.rr_feat_names = [str(x) for x in self.arr['rr_feat_names'].tolist()]
        self.act_feat_names = [str(x) for x in self.arr['act_feat_names'].tolist()]
        self.sleep_feat_names = [str(x) for x in self.arr['sleep_feat_names'].tolist()]

        self.has_label_daily_stress = 'label_daily_stress' in self.index_df.columns
        self.has_label_stai1 = 'label_stai1' in self.index_df.columns
        self.has_label_stai2 = 'label_stai2' in self.index_df.columns

        self.user_groups: List[pd.DataFrame] = []
        for _, g in self.index_df.groupby('user_id', sort=True):
            gg = g.copy()
            gg['day'] = gg['day'].astype(int)
            gg = gg.sort_values('day')
            self.user_groups.append(gg.reset_index(drop=True))

    def _infer_dim_by_prefix(self, prefix: str) -> int:
        for k, v in self.arr.items():
            if k.startswith(prefix) and isinstance(v, np.ndarray) and v.ndim == 2:
                return int(v.shape[1])
        return 0

    def _default_feat_names(self, modal: str, dim: int) -> np.ndarray:
        if modal == 'rr':
            if dim == 17:
                names = [
                    'patch_start_min', 'patch_end_min', 'patch_center_min',
                    'rr_count', 'ibi_mean', 'ibi_std', 'ibi_min', 'ibi_max',
                    'hr_mean', 'hr_std', 'rmssd', 'sdnn', 'pnn50',
                    'ibi_slope', 'hr_slope', 'delta_ibi_mean', 'delta_hr_mean',
                ]
                return np.array(names, dtype='<U32')
            if dim == 5:
                return np.array(['hour', 'minute_of_hour', 'second_of_minute', 'ibi_s', 'hr_inst'], dtype='<U32')
        if modal == 'act':
            if dim == 19:
                names = [
                    'patch_start_min', 'patch_end_min', 'patch_center_min',
                    'active_minutes', 'event_count', 'intensity_mean',
                    'delta_active_minutes', 'slope_active_minutes',
                ] + [f'code_{i}_minutes' for i in range(11)]
                return np.array(names, dtype='<U32')
            if dim == 4:
                return np.array(['start_hour', 'end_hour', 'duration_min', 'activity_code'], dtype='<U32')
        if modal == 'sleep':
            if dim == 12:
                names = [
                    'patch_start_min', 'patch_end_min', 'patch_center_min',
                    'sleep_minutes', 'segment_count',
                    'delta_sleep_minutes', 'slope_sleep_minutes',
                    'efficiency', 'latency', 'tst', 'waso', 'n_awakenings',
                ]
                return np.array(names, dtype='<U32')
            if dim == 10:
                names = [
                    'start_hour', 'end_hour', 'duration_min', 'cross_day', 'cross_hour',
                    'efficiency', 'latency', 'tst', 'waso', 'n_awakenings',
                ]
                return np.array(names, dtype='<U32')
        return np.array([f'feat_{i}' for i in range(dim)], dtype='<U32')

    def _fill_missing_feat_names(self) -> None:
        for modal, key, prefix in [
            ('rr', 'rr_feat_names', 'rr::'),
            ('act', 'act_feat_names', 'act::'),
            ('sleep', 'sleep_feat_names', 'sleep::'),
        ]:
            if key not in self.arr or self.arr[key] is None:
                dim = self._infer_dim_by_prefix(prefix)
                self.arr[key] = self._default_feat_names(modal, dim)

    def __len__(self) -> int:
        return len(self.user_groups)

    def _get_modal_arr(self, prefix: str, sample_id: str) -> np.ndarray:
        key = f'{prefix}::{sample_id}'
        if key in self.arr:
            return self.arr[key]
        d = self.rr_feat_dim if prefix == 'rr' else self.act_feat_dim if prefix == 'act' else self.sleep_feat_dim
        return np.zeros((0, d), dtype=np.float32)

    @staticmethod
    def _build_time_feat_from_sequence(seq: np.ndarray, feat_names: List[str]) -> np.ndarray:
        """
        构造事件时间特征: [delta_time, since_day_start]（归一化到[0,1]）。
        优先从token里的 patch_center_min（或等价时间列）提取真实时间；失败再回退到等间隔序号。
        """
        length = int(seq.shape[0])
        if length <= 0:
            return np.zeros((0, 2), dtype=np.float32)

        time_col = -1
        for name in ['patch_center_min', 'center_min', 'time_min', 'start_min', 'patch_start_min', 'hour']:
            if name in feat_names:
                time_col = feat_names.index(name)
                break

        if time_col >= 0:
            t = seq[:, time_col].astype(np.float32)
            t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            # hour列需要映射到分钟尺度
            if feat_names[time_col] == 'hour':
                t = t * 60.0
            t = np.maximum.accumulate(t)
            delta = np.zeros((length,), dtype=np.float32)
            if length > 1:
                delta[1:] = t[1:] - t[:-1]
            delta = np.clip(delta / 1440.0, 0.0, 1.0)
            since = np.clip(t / 1440.0, 0.0, 1.0)
            return np.stack([delta, since], axis=-1).astype(np.float32)

        if length == 1:
            return np.array([[0.0, 0.0]], dtype=np.float32)
        since = np.linspace(0.0, 1.0, num=length, dtype=np.float32)
        delta = np.zeros((length,), dtype=np.float32)
        delta[1:] = since[1:] - since[:-1]
        return np.stack([delta, since], axis=-1).astype(np.float32)

    @staticmethod
    def _pad_day_sequences(seq_list: List[np.ndarray], feat_dim: int) -> tuple[np.ndarray, np.ndarray]:
        d = len(seq_list)
        if d == 0:
            return np.zeros((0, 0, feat_dim), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)
        t_max = max(s.shape[0] for s in seq_list)
        padded = np.zeros((d, t_max, feat_dim), dtype=np.float32)
        mask = np.zeros((d, t_max), dtype=np.float32)
        for i, s in enumerate(seq_list):
            t = s.shape[0]
            if t > 0:
                padded[i, :t, :] = s
                mask[i, :t] = 1.0
        return padded, mask

    def __getitem__(self, idx: int) -> UserEventSample:
        g = self.user_groups[idx]
        user_id = str(g.loc[0, 'user_id'])
        days = g['day'].astype(int).to_numpy()

        rr_list: List[np.ndarray] = []
        act_list: List[np.ndarray] = []
        sleep_list: List[np.ndarray] = []
        rr_day_mask, act_day_mask, sleep_day_mask, day_mask = [], [], [], []
        rr_time_list: List[np.ndarray] = []
        act_time_list: List[np.ndarray] = []
        sleep_time_list: List[np.ndarray] = []
        y_stress, y_stai1, y_stai2 = [], [], []

        for _, r in g.iterrows():
            sample_id = str(r['sample_id'])
            rr = self._get_modal_arr('rr', sample_id)
            act = self._get_modal_arr('act', sample_id)
            sleep = self._get_modal_arr('sleep', sample_id)

            rr_list.append(rr)
            act_list.append(act)
            sleep_list.append(sleep)

            rr_time_list.append(self._build_time_feat_from_sequence(rr, self.rr_feat_names))
            act_time_list.append(self._build_time_feat_from_sequence(act, self.act_feat_names))
            sleep_time_list.append(self._build_time_feat_from_sequence(sleep, self.sleep_feat_names))

            rr_has = int(r['rr_has'])
            act_has = int(r['act_has'])
            sleep_has = int(r['sleep_has'])

            rr_day_mask.append(float(rr_has))
            act_day_mask.append(float(act_has))
            sleep_day_mask.append(float(sleep_has))
            day_mask.append(float(rr_has or act_has or sleep_has))

            y_stress.append(float(r['label_daily_stress']) if self.has_label_daily_stress and not pd.isna(r.get('label_daily_stress', np.nan)) else np.nan)
            y_stai1.append(float(r['label_stai1']) if self.has_label_stai1 and not pd.isna(r.get('label_stai1', np.nan)) else np.nan)
            y_stai2.append(float(r['label_stai2']) if self.has_label_stai2 and not pd.isna(r.get('label_stai2', np.nan)) else np.nan)

        rr_pad, rr_step_mask = self._pad_day_sequences(rr_list, self.rr_feat_dim)
        act_pad, act_step_mask = self._pad_day_sequences(act_list, self.act_feat_dim)
        sleep_pad, sleep_step_mask = self._pad_day_sequences(sleep_list, self.sleep_feat_dim)

        rr_time_feat, _ = self._pad_day_sequences(rr_time_list, 2)
        act_time_feat, _ = self._pad_day_sequences(act_time_list, 2)
        sleep_time_feat, _ = self._pad_day_sequences(sleep_time_list, 2)

        return UserEventSample(
            user_id=user_id,
            days=days.astype(np.int64),
            rr=rr_pad,
            act=act_pad,
            sleep=sleep_pad,
            rr_step_mask=rr_step_mask,
            act_step_mask=act_step_mask,
            sleep_step_mask=sleep_step_mask,
            rr_time_feat=rr_time_feat,
            act_time_feat=act_time_feat,
            sleep_time_feat=sleep_time_feat,
            rr_day_mask=np.array(rr_day_mask, dtype=np.float32),
            act_day_mask=np.array(act_day_mask, dtype=np.float32),
            sleep_day_mask=np.array(sleep_day_mask, dtype=np.float32),
            day_mask=np.array(day_mask, dtype=np.float32),
            label_daily_stress=np.array(y_stress, dtype=np.float32),
            label_stai1=np.array(y_stai1, dtype=np.float32),
            label_stai2=np.array(y_stai2, dtype=np.float32),
        )


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


def _pad_batch_2d(v_list: List[np.ndarray], pad_value: float = 0.0) -> np.ndarray:
    b = len(v_list)
    if b == 0:
        return np.zeros((0, 0), dtype=np.float32)
    d_max = max(v.shape[0] for v in v_list)
    out = np.full((b, d_max), pad_value, dtype=np.float32)
    for i, v in enumerate(v_list):
        out[i, :v.shape[0]] = v
    return out


def user_event_collate_fn(batch: List[UserEventSample]) -> Dict[str, Any]:
    rr, rr_step_mask = _pad_batch_4d([b.rr for b in batch], [b.rr_step_mask for b in batch], batch[0].rr.shape[-1])
    act, act_step_mask = _pad_batch_4d([b.act for b in batch], [b.act_step_mask for b in batch], batch[0].act.shape[-1])
    sleep, sleep_step_mask = _pad_batch_4d([b.sleep for b in batch], [b.sleep_step_mask for b in batch], batch[0].sleep.shape[-1])

    rr_time_feat, _ = _pad_batch_4d([b.rr_time_feat for b in batch], [b.rr_step_mask for b in batch], 2)
    act_time_feat, _ = _pad_batch_4d([b.act_time_feat for b in batch], [b.act_step_mask for b in batch], 2)
    sleep_time_feat, _ = _pad_batch_4d([b.sleep_time_feat for b in batch], [b.sleep_step_mask for b in batch], 2)

    return {
        'user_id': [b.user_id for b in batch],
        'days': _pad_batch_2d([b.days.astype(np.float32) for b in batch]).astype(np.int64),
        'rr': rr,
        'act': act,
        'sleep': sleep,
        'rr_step_mask': rr_step_mask,
        'act_step_mask': act_step_mask,
        'sleep_step_mask': sleep_step_mask,
        'rr_time_feat': rr_time_feat,
        'act_time_feat': act_time_feat,
        'sleep_time_feat': sleep_time_feat,
        'day_mask': _pad_batch_2d([b.day_mask for b in batch]),   #用于表示当前day是不是至少有1个模态有数据
        'rr_day_mask': _pad_batch_2d([b.rr_day_mask for b in batch]),
        'act_day_mask': _pad_batch_2d([b.act_day_mask for b in batch]),
        'sleep_day_mask': _pad_batch_2d([b.sleep_day_mask for b in batch]),
        'label_daily_stress': _pad_batch_2d([b.label_daily_stress for b in batch], pad_value=np.nan),
        'label_stai1': _pad_batch_2d([b.label_stai1 for b in batch], pad_value=np.nan),
        'label_stai2': _pad_batch_2d([b.label_stai2 for b in batch], pad_value=np.nan),
    }


# 兼容旧名字
EventDataset = UserEventDataset
event_collate_fn = user_event_collate_fn
