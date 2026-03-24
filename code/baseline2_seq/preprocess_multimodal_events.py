from __future__ import annotations

"""
多模态事件序列预处理脚本（RR / Activity / Sleep）
=================================================

本脚本做的事：
1) 把每个用户每天（user_id + day）组织成一个 sample_id。
2) 为每个 sample_id 构建三种变长事件序列：
   - RR events    : [T_rr, D_rr]
   - ACT events   : [T_act, D_act]
   - SLEEP events : [T_sleep, D_sleep]
3) 输出：
   - multimodal_event_sequences.npz：保存所有 sample 的事件序列
   - multimodal_event_index.csv：保存 sample 索引、长度、标签、是否缺失

说明：
- 这是“事件序列”方案，不是小时桶统计方案。
- 序列长度 T 是变长的，训练时需要 padding + mask。
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =========================
# 路径配置
# =========================
DATA_ROOT = Path('/home/wqshen/mm-test/dataset/physionet.org/files/mmash/1.0.0/DataPaper')
LABEL_PATH = Path('/home/wqshen/mm-test/code/processed/mmash_day_level_features.csv')
OUT_DIR = Path('/home/wqshen/mm-test/code/baseline2_seq/processed')
OUT_NPZ = OUT_DIR / 'multimodal_event_sequences.npz'
OUT_INDEX = OUT_DIR / 'multimodal_event_index.csv'


# =========================
# 事件特征定义（每个token的维度）
# =========================
RR_EVENT_FEAT = ['hour', 'minute_of_hour', 'second_of_minute', 'ibi_s', 'hr_inst']
ACT_EVENT_FEAT = ['start_hour', 'end_hour', 'duration_min', 'activity_code']
SLEEP_EVENT_FEAT = [
    'start_hour', 'end_hour', 'duration_min', 'cross_day', 'cross_hour',
    'efficiency', 'latency', 'tst', 'waso', 'n_awakenings'
]


# =========================
# 通用工具
# =========================
def _safe_float(x) -> float:
    """安全转float，失败返回NaN。"""
    try:
        return float(x)
    except Exception:
        return np.nan


def _to_minute(hhmm: str) -> int:
    """'HH:MM' -> 分钟数。"""
    h, m = hhmm.strip().split(':')
    return int(h) * 60 + int(m)


def _parse_hms(s: str) -> Tuple[int, int, int]:
    """解析 'HH:MM' 或 'HH:MM:SS'。"""
    p = str(s).strip().split(':')
    if len(p) == 3:
        return int(p[0]), int(p[1]), int(p[2])
    if len(p) == 2:
        return int(p[0]), int(p[1]), 0
    return 0, 0, 0


def _read_csv_maybe_skip_comment(path: Path, required_col: str) -> pd.DataFrame:
    """
    MMASH原始CSV有时第一行是注释：
    - 先普通读取
    - 若不存在 required_col，自动 skiprows=1 重读
    """
    df = pd.read_csv(path)
    if required_col not in df.columns:
        df = pd.read_csv(path, skiprows=1)
    if len(df.columns) > 0 and df.columns[0] == '':
        # 清理空首列（索引残留）
        df = df.drop(columns=[df.columns[0]])
    return df


def _normalize_day(day_val: int) -> int:
    """
    统一day到{1,2}：
    - day=1 -> 1
    - day=2 -> 2
    - 其他异常值（如-29）-> 2
    """
    if day_val == 1:
        return 1
    if day_val == 2:
        return 2
    return 2


# =========================
# 标签加载
# =========================
def load_labels() -> pd.DataFrame:
    """读取并去重day-level标签。"""
    if not LABEL_PATH.exists():
        return pd.DataFrame(columns=['user_id', 'day'])

    df = pd.read_csv(LABEL_PATH)
    keep = [c for c in ['user_id', 'day', 'label_daily_stress', 'label_stai1', 'label_stai2'] if c in df.columns]
    out = df[keep].drop_duplicates(['user_id', 'day']).copy()
    out['day'] = out['day'].astype(int).apply(_normalize_day)
    return out


# =========================
# RR 事件构建
# =========================
def build_rr_events(rr_csv: Path) -> Dict[int, np.ndarray]:
    """
    返回 {day: rr_events}，其中 rr_events 形状为 [T_rr, 4]。

    token = [hour, minute_of_hour, second_of_minute, ibi_s, hr_inst]
    """
    if not rr_csv.exists():
        return {}

    df = _read_csv_maybe_skip_comment(rr_csv, 'ibi_s')
    if not {'ibi_s', 'day', 'time'}.issubset(df.columns):
        return {}

    df['ibi_s'] = pd.to_numeric(df['ibi_s'], errors='coerce')
    df['day'] = pd.to_numeric(df['day'], errors='coerce')
    df = df.dropna(subset=['ibi_s', 'day', 'time']).copy()

    # 去除明显伪迹
    df = df[(df['ibi_s'] >= 0.2) & (df['ibi_s'] <= 2.5)]
    if df.empty:
        return {}

    events_by_day: Dict[int, List[List[float]]] = {}
    for _, r in df.iterrows():
        d = _normalize_day(int(r['day']))
        h, m, s = _parse_hms(str(r['time']))
        ibi = float(r['ibi_s'])
        hr_inst = 60.0 / ibi if ibi > 1e-6 else 0.0
        events_by_day.setdefault(d, []).append([float(h), float(m), float(s), ibi, hr_inst])

    out = {}
    for d, arr in events_by_day.items():
        x = np.array(arr, dtype=np.float32)
        # 按 (hour, minute, second) 排序
        idx = np.lexsort((x[:, 2], x[:, 1], x[:, 0]))
        out[d] = x[idx]

    return out


# =========================
# Activity 事件构建
# =========================
def build_activity_events(act_csv: Path) -> Dict[int, np.ndarray]:
    """
    返回 {day: act_events}，其中 act_events 形状为 [T_act, 4]。

    token = [start_hour, end_hour, duration_min, activity_code]
    """
    if not act_csv.exists():
        return {}

    df = _read_csv_maybe_skip_comment(act_csv, 'Activity')
    if not {'Activity', 'Start', 'Day'}.issubset(df.columns):
        return {}

    df['Activity'] = pd.to_numeric(df['Activity'], errors='coerce')
    df['Day'] = pd.to_numeric(df['Day'], errors='coerce')
    df = df.dropna(subset=['Activity', 'Day', 'Start']).copy()

    events_by_day: Dict[int, List[List[float]]] = {}
    for _, r in df.iterrows():
        code = int(r['Activity'])
        if code < 0 or code > 10:
            continue

        d = _normalize_day(int(r['Day']))

        st = str(r['Start'])
        ed = str(r['End']) if 'End' in df.columns and not pd.isna(r.get('End', np.nan)) else ''

        try:
            st_m = _to_minute(st)
        except Exception:
            continue

        # 缺失结束时间时，用 +30min 作为兜底
        if ed and ':' in ed:
            try:
                ed_m = _to_minute(ed)
            except Exception:
                ed_m = st_m + 30
        else:
            ed_m = st_m + 30

        # 跨午夜修正
        if ed_m <= st_m:
            ed_m += 1440

        dur = float(ed_m - st_m)
        st_hour = float((st_m // 60) % 24)
        ed_hour = float((ed_m // 60) % 24)

        events_by_day.setdefault(d, []).append([st_hour, ed_hour, dur, float(code)])

    out = {}
    for d, arr in events_by_day.items():
        x = np.array(arr, dtype=np.float32)
        idx = np.argsort(x[:, 0])
        out[d] = x[idx]

    return out


# =========================
# Sleep 事件构建
# =========================
def build_sleep_events(sleep_csv: Path) -> Dict[int, np.ndarray]:
    """
    返回 {day: sleep_events}，其中 sleep_events 形状为 [T_sleep, 10]。

    token = [
      start_hour, end_hour, duration_min, cross_day, cross_hour,
      efficiency, latency, tst, waso, n_awakenings
    ]
    """
    if not sleep_csv.exists():
        return {}

    df = _read_csv_maybe_skip_comment(sleep_csv, 'In Bed Date')
    if not {'In Bed Date', 'In Bed Time', 'Out Bed Date', 'Out Bed Time'}.issubset(df.columns):
        return {}

    events_by_day: Dict[int, List[List[float]]] = {}
    for _, r in df.iterrows():
        in_day = _safe_float(r['In Bed Date'])
        out_day = _safe_float(r['Out Bed Date'])
        if np.isnan(in_day) or np.isnan(out_day):
            continue

        try:
            st_m = _to_minute(str(r['In Bed Time']))
            ed_m = _to_minute(str(r['Out Bed Time']))
        except Exception:
            continue

        d = _normalize_day(int(in_day))

        start_abs = d * 1440 + st_m
        end_abs = int(out_day) * 1440 + ed_m
        if end_abs <= start_abs:
            end_abs += 1440

        dur = float(end_abs - start_abs)
        if dur <= 0:
            continue

        st_hour = float((st_m // 60) % 24)
        ed_hour = float((ed_m // 60) % 24)
        cross_day = 1.0 if int(out_day) != int(in_day) else 0.0
        cross_hour = 1.0 if (st_m // 60) != ((ed_m - 1) // 60) else 0.0

        eff = _safe_float(r.get('Efficiency', np.nan))
        lat = _safe_float(r.get('Latency', np.nan))
        tst = _safe_float(r.get('Total Sleep Time (TST)', np.nan))
        waso = _safe_float(r.get('Wake After Sleep Onset (WASO)', np.nan))
        awak = _safe_float(r.get('Number of Awakenings', np.nan))

        events_by_day.setdefault(d, []).append([
            st_hour, ed_hour, dur, cross_day, cross_hour, eff, lat, tst, waso, awak
        ])

    out = {}
    for d, arr in events_by_day.items():
        x = np.array(arr, dtype=np.float32)
        # 数值缺失统一填0；训练时可配合缺失指示器
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        idx = np.argsort(x[:, 0])
        out[d] = x[idx]

    return out


# =========================
# 主流程
# =========================
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_labels()

    # 存储各sample的三模态事件序列
    rr_store: Dict[str, np.ndarray] = {}
    act_store: Dict[str, np.ndarray] = {}
    sleep_store: Dict[str, np.ndarray] = {}

    # 索引表：记录每个sample长度、是否缺失、标签等
    index_rows = []

    user_dirs = sorted([p for p in DATA_ROOT.glob('user_*') if p.is_dir()])
    for udir in user_dirs:
        user_id = udir.name

        rr_by_day = build_rr_events(udir / 'RR.csv')
        act_by_day = build_activity_events(udir / 'Activity.csv')
        sleep_by_day = build_sleep_events(udir / 'sleep.csv')

        # 一个样本日，只要任一模态或标签出现，就纳入
        day_set = set(rr_by_day.keys()) | set(act_by_day.keys()) | set(sleep_by_day.keys())
        if not labels.empty:
            day_set |= set(labels[labels['user_id'] == user_id]['day'].astype(int).tolist())

        for d in sorted(day_set):
            sample_id = f'{user_id}_day{d}'

            rr_arr = rr_by_day.get(d, np.zeros((0, len(RR_EVENT_FEAT)), dtype=np.float32))
            act_arr = act_by_day.get(d, np.zeros((0, len(ACT_EVENT_FEAT)), dtype=np.float32))
            sleep_arr = sleep_by_day.get(d, np.zeros((0, len(SLEEP_EVENT_FEAT)), dtype=np.float32))

            rr_store[sample_id] = rr_arr
            act_store[sample_id] = act_arr
            sleep_store[sample_id] = sleep_arr

            row = {
                'sample_id': sample_id,
                'user_id': user_id,
                'day': int(d),
                'rr_len': int(rr_arr.shape[0]),
                'act_len': int(act_arr.shape[0]),
                'sleep_len': int(sleep_arr.shape[0]),
                'rr_has': int(rr_arr.shape[0] > 0),
                'act_has': int(act_arr.shape[0] > 0),
                'sleep_has': int(sleep_arr.shape[0] > 0),
            }

            # 关联标签
            if not labels.empty:
                m = labels[(labels['user_id'] == user_id) & (labels['day'].astype(int) == int(d))]
                if len(m) > 0:
                    for t in ['label_daily_stress', 'label_stai1', 'label_stai2']:
                        if t in m.columns:
                            row[t] = _safe_float(m[t].iloc[0])

            index_rows.append(row)

    # npz保存：用前缀区分模态，便于后续读取过滤
    save_dict = {}
    for sid, arr in rr_store.items():
        save_dict[f'rr::{sid}'] = arr
    for sid, arr in act_store.items():
        save_dict[f'act::{sid}'] = arr
    for sid, arr in sleep_store.items():
        save_dict[f'sleep::{sid}'] = arr

    # 附带保存特征名
    save_dict['rr_feat_names'] = np.array(RR_EVENT_FEAT, dtype=object)
    save_dict['act_feat_names'] = np.array(ACT_EVENT_FEAT, dtype=object)
    save_dict['sleep_feat_names'] = np.array(SLEEP_EVENT_FEAT, dtype=object)

    # 保存文件
    index_df = pd.DataFrame(index_rows)
    np.savez_compressed(OUT_NPZ, **save_dict)
    index_df.to_csv(OUT_INDEX, index=False)

    # 打印形状摘要（变长序列，用长度范围描述）
    rr_dim = len(RR_EVENT_FEAT)
    act_dim = len(ACT_EVENT_FEAT)
    sleep_dim = len(SLEEP_EVENT_FEAT)

    rr_min = int(index_df['rr_len'].min()) if len(index_df) else 0
    rr_max = int(index_df['rr_len'].max()) if len(index_df) else 0
    act_min = int(index_df['act_len'].min()) if len(index_df) else 0
    act_max = int(index_df['act_len'].max()) if len(index_df) else 0
    sleep_min = int(index_df['sleep_len'].min()) if len(index_df) else 0
    sleep_max = int(index_df['sleep_len'].max()) if len(index_df) else 0

    print(f'[OK] saved event npz: {OUT_NPZ}')
    print(f'[OK] saved index csv: {OUT_INDEX}')
    print(f'[OK] n_samples={len(index_df)}')
    print('[SHAPE] token维度:')
    print(f'  RR token dim    = {rr_dim}')
    print(f'  ACT token dim   = {act_dim}')
    print(f'  SLEEP token dim = {sleep_dim}')
    print('[SHAPE] 序列长度范围 (T_min ~ T_max):')
    print(f'  RR    : {rr_min} ~ {rr_max}  (单样本形状: [T_rr, {rr_dim}])')
    print(f'  ACT   : {act_min} ~ {act_max}  (单样本形状: [T_act, {act_dim}])')
    print(f'  SLEEP : {sleep_min} ~ {sleep_max}  (单样本形状: [T_sleep, {sleep_dim}])')


if __name__ == '__main__':
    main()
