from __future__ import annotations

"""
多模态事件序列预处理（Patch版本：RR / Activity / Sleep）
=======================================================

目标：
1) 统一按时间窗口（patch）构建序列，避免RR等间隔抽点丢失局部变化。
2) 保留模态内动态：patch内统计 + patch间delta/slope。
3) 为后续跨模态时序融合提供可对齐的时间语义（patch_center_min）。

输出：
- multimodal_event_sequences.npz
- multimodal_event_index.csv
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DATA_ROOT = Path('/home/wqshen/mm-test/dataset/physionet.org/files/mmash/1.0.0/DataPaper')
LABEL_PATH = Path('/home/wqshen/mm-test/code/processed/mmash_day_level_features.csv')
OUT_DIR = Path('/home/wqshen/mm-test/code/baseline2_seq/processed')

RR_PATCH_FEAT = [
    'patch_start_min', 'patch_end_min', 'patch_center_min',
    'rr_count', 'ibi_mean', 'ibi_std', 'ibi_min', 'ibi_max',
    'hr_mean', 'hr_std', 'rmssd', 'sdnn', 'pnn50',
    'ibi_slope', 'hr_slope', 'delta_ibi_mean', 'delta_hr_mean',
]

ACT_PATCH_FEAT = [
    'patch_start_min', 'patch_end_min', 'patch_center_min',
    'active_minutes', 'event_count', 'intensity_mean',
    'delta_active_minutes', 'slope_active_minutes',
] + [f'code_{i}_minutes' for i in range(11)]

SLEEP_PATCH_FEAT = [
    'patch_start_min', 'patch_end_min', 'patch_center_min',
    'sleep_minutes', 'segment_count',
    'delta_sleep_minutes', 'slope_sleep_minutes',
    'efficiency', 'latency', 'tst', 'waso', 'n_awakenings',
]


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def _normalize_day(day_val: int) -> int:
    if day_val == 1:
        return 1
    if day_val == 2:
        return 2
    return 2


def _read_csv_maybe_skip_comment(path: Path, required_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if required_col not in df.columns:
        df = pd.read_csv(path, skiprows=1)
    if len(df.columns) > 0 and df.columns[0] == '':
        df = df.drop(columns=[df.columns[0]])
    return df


def _parse_hms_to_second(s: str) -> int:
    p = str(s).strip().split(':')
    if len(p) == 3:
        h, m, sec = int(p[0]), int(p[1]), int(p[2])
    elif len(p) == 2:
        h, m, sec = int(p[0]), int(p[1]), 0
    else:
        return 0
    return h * 3600 + m * 60 + sec


def _to_minute(hhmm: str) -> int:
    h, m = hhmm.strip().split(':')
    return int(h) * 60 + int(m)


def _window_bounds(window_min: int) -> np.ndarray:
    n = int(np.ceil(1440 / window_min))
    starts = np.arange(n, dtype=np.int32) * window_min
    ends = np.minimum(starts + window_min, 1440)
    return np.stack([starts, ends], axis=1)


def _rolling_slope(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.allclose(x, x[0]):
        return 0.0
    xm = x.mean()
    ym = y.mean()
    den = np.sum((x - xm) ** 2)
    if den <= 1e-8:
        return 0.0
    num = np.sum((x - xm) * (y - ym))
    return float(num / den)


def load_labels() -> pd.DataFrame:
    if not LABEL_PATH.exists():
        return pd.DataFrame(columns=['user_id', 'day'])
    df = pd.read_csv(LABEL_PATH)
    keep = [c for c in ['user_id', 'day', 'label_daily_stress', 'label_stai1', 'label_stai2'] if c in df.columns]
    out = df[keep].drop_duplicates(['user_id', 'day']).copy()
    out['day'] = out['day'].astype(int).apply(_normalize_day)
    return out


def build_rr_patches(rr_csv: Path, window_min: int) -> Dict[int, np.ndarray]:
    if not rr_csv.exists():
        return {}
    df = _read_csv_maybe_skip_comment(rr_csv, 'ibi_s')
    if not {'ibi_s', 'day', 'time'}.issubset(df.columns):
        return {}

    df['ibi_s'] = pd.to_numeric(df['ibi_s'], errors='coerce')
    df['day'] = pd.to_numeric(df['day'], errors='coerce')
    df = df.dropna(subset=['ibi_s', 'day', 'time']).copy()
    df = df[(df['ibi_s'] >= 0.2) & (df['ibi_s'] <= 2.5)]
    if df.empty:
        return {}

    df['day'] = df['day'].astype(int).apply(_normalize_day)
    df['sec'] = df['time'].astype(str).apply(_parse_hms_to_second)
    df['hr'] = 60.0 / df['ibi_s']

    by_day: Dict[int, np.ndarray] = {}
    bounds = _window_bounds(window_min)

    for d, g in df.groupby('day'):
        g = g.sort_values('sec')
        rows: List[List[float]] = []

        prev_ibi_mean = np.nan
        prev_hr_mean = np.nan
        prev_center = np.nan
        prev_ibi_for_slope = np.nan
        prev_hr_for_slope = np.nan

        for st_m, ed_m in bounds:
            st_s = int(st_m) * 60
            ed_s = int(ed_m) * 60
            gg = g[(g['sec'] >= st_s) & (g['sec'] < ed_s)]
            if gg.empty:
                continue

            ibi = gg['ibi_s'].to_numpy(dtype=np.float32)
            hr = gg['hr'].to_numpy(dtype=np.float32)
            sec_rel = gg['sec'].to_numpy(dtype=np.float32) - float(st_s)

            rmssd = float(np.sqrt(np.mean(np.diff(ibi) ** 2))) if ibi.size >= 2 else 0.0
            sdnn = float(np.std(ibi)) if ibi.size >= 2 else 0.0
            pnn50 = float(np.mean(np.abs(np.diff(ibi)) > 0.05)) if ibi.size >= 2 else 0.0

            ibi_mean = float(np.mean(ibi))
            hr_mean = float(np.mean(hr))
            center = float((st_m + ed_m) / 2.0)

            ibi_slope_in_patch = _rolling_slope(sec_rel, ibi)
            hr_slope_in_patch = _rolling_slope(sec_rel, hr)

            delta_ibi_mean = 0.0 if np.isnan(prev_ibi_mean) else float(ibi_mean - prev_ibi_mean)
            delta_hr_mean = 0.0 if np.isnan(prev_hr_mean) else float(hr_mean - prev_hr_mean)

            if not np.isnan(prev_center):
                dt = max(center - prev_center, 1.0)
                if not np.isnan(prev_ibi_for_slope):
                    ibi_slope_in_patch += float((ibi_mean - prev_ibi_for_slope) / dt)
                if not np.isnan(prev_hr_for_slope):
                    hr_slope_in_patch += float((hr_mean - prev_hr_for_slope) / dt)

            rows.append([
                float(st_m), float(ed_m), center,
                float(ibi.size), ibi_mean, float(np.std(ibi)), float(np.min(ibi)), float(np.max(ibi)),
                hr_mean, float(np.std(hr)), rmssd, sdnn, pnn50,
                float(ibi_slope_in_patch), float(hr_slope_in_patch), delta_ibi_mean, delta_hr_mean,
            ])

            prev_ibi_mean = ibi_mean
            prev_hr_mean = hr_mean
            prev_center = center
            prev_ibi_for_slope = ibi_mean
            prev_hr_for_slope = hr_mean

        by_day[int(d)] = np.array(rows, dtype=np.float32) if rows else np.zeros((0, len(RR_PATCH_FEAT)), dtype=np.float32)

    return by_day


def build_activity_patches(act_csv: Path, window_min: int) -> Dict[int, np.ndarray]:
    if not act_csv.exists():
        return {}
    df = _read_csv_maybe_skip_comment(act_csv, 'Activity')
    if not {'Activity', 'Start', 'Day'}.issubset(df.columns):
        return {}

    df['Activity'] = pd.to_numeric(df['Activity'], errors='coerce')
    df['Day'] = pd.to_numeric(df['Day'], errors='coerce')
    df = df.dropna(subset=['Activity', 'Day', 'Start']).copy()

    intensity_map = {0: 0.0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0, 6: 0.5, 7: 0.2, 8: 0.2, 9: 0.1, 10: 0.1}

    bounds = _window_bounds(window_min)
    by_day_rows: Dict[int, List[List[float]]] = {}

    events_by_day: Dict[int, List[Tuple[int, int, int]]] = {}
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

        if ed and ':' in ed:
            try:
                ed_m = _to_minute(ed)
            except Exception:
                ed_m = st_m + 30
        else:
            ed_m = st_m + 30

        if ed_m <= st_m:
            ed_m += 1440

        events_by_day.setdefault(d, []).append((st_m, ed_m, code))

    for d, events in events_by_day.items():
        rows: List[List[float]] = []
        prev_active = np.nan
        prev_center = np.nan

        for st_m, ed_m in bounds:
            active_minutes = 0.0
            event_count = 0.0
            weighted_intensity = 0.0
            code_minutes = np.zeros((11,), dtype=np.float32)

            for ev_st, ev_ed, code in events:
                ov_st = max(st_m, ev_st)
                ov_ed = min(ed_m, ev_ed)
                ov = float(max(0, ov_ed - ov_st))
                if ov <= 0:
                    continue
                active_minutes += ov
                event_count += 1.0
                code_minutes[code] += ov
                weighted_intensity += ov * float(intensity_map.get(code, 0.0))

            if active_minutes <= 0:
                continue

            center = float((st_m + ed_m) / 2.0)
            intensity_mean = float(weighted_intensity / max(active_minutes, 1e-6))
            delta_active = 0.0 if np.isnan(prev_active) else float(active_minutes - prev_active)
            slope_active = 0.0
            if not np.isnan(prev_center):
                dt = max(center - prev_center, 1.0)
                slope_active = float((active_minutes - prev_active) / dt)

            row = [
                float(st_m), float(ed_m), center,
                float(active_minutes), float(event_count), intensity_mean,
                delta_active, slope_active,
            ] + code_minutes.tolist()
            rows.append(row)

            prev_active = active_minutes
            prev_center = center

        by_day_rows[int(d)] = np.array(rows, dtype=np.float32) if rows else np.zeros((0, len(ACT_PATCH_FEAT)), dtype=np.float32)

    return by_day_rows


def build_sleep_patches(sleep_csv: Path, window_min: int) -> Dict[int, np.ndarray]:
    if not sleep_csv.exists():
        return {}
    df = _read_csv_maybe_skip_comment(sleep_csv, 'In Bed Date')
    if not {'In Bed Date', 'In Bed Time', 'Out Bed Date', 'Out Bed Time'}.issubset(df.columns):
        return {}

    bounds = _window_bounds(window_min)
    by_day_rows: Dict[int, List[List[float]]] = {}

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

        st_abs = d * 1440 + st_m
        ed_abs = int(out_day) * 1440 + ed_m
        if ed_abs <= st_abs:
            ed_abs += 1440

        eff = _safe_float(r.get('Efficiency', np.nan))
        lat = _safe_float(r.get('Latency', np.nan))
        tst = _safe_float(r.get('Total Sleep Time (TST)', np.nan))
        waso = _safe_float(r.get('Wake After Sleep Onset (WASO)', np.nan))
        awak = _safe_float(r.get('Number of Awakenings', np.nan))

        rows: List[List[float]] = []
        prev_sleep = np.nan
        prev_center = np.nan

        for b_st, b_ed in bounds:
            b_st_abs = d * 1440 + int(b_st)
            b_ed_abs = d * 1440 + int(b_ed)
            ov = float(max(0, min(ed_abs, b_ed_abs) - max(st_abs, b_st_abs)))
            if ov <= 0:
                continue

            center = float((b_st + b_ed) / 2.0)
            delta_sleep = 0.0 if np.isnan(prev_sleep) else float(ov - prev_sleep)
            slope_sleep = 0.0
            if not np.isnan(prev_center):
                dt = max(center - prev_center, 1.0)
                slope_sleep = float((ov - prev_sleep) / dt)

            rows.append([
                float(b_st), float(b_ed), center,
                ov, 1.0,
                delta_sleep, slope_sleep,
                0.0 if np.isnan(eff) else float(eff),
                0.0 if np.isnan(lat) else float(lat),
                0.0 if np.isnan(tst) else float(tst),
                0.0 if np.isnan(waso) else float(waso),
                0.0 if np.isnan(awak) else float(awak),
            ])

            prev_sleep = ov
            prev_center = center

        by_day_rows[d] = np.array(rows, dtype=np.float32) if rows else np.zeros((0, len(SLEEP_PATCH_FEAT)), dtype=np.float32)

    return by_day_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_min', type=int, default=5)
    parser.add_argument('--out_npz', type=str, default=str(OUT_DIR / 'multimodal_event_sequences.npz'))
    parser.add_argument('--out_index', type=str, default=str(OUT_DIR / 'multimodal_event_index.csv'))
    args = parser.parse_args()

    out_npz = Path(args.out_npz)
    out_index = Path(args.out_index)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    labels = load_labels()

    rr_store: Dict[str, np.ndarray] = {}
    act_store: Dict[str, np.ndarray] = {}
    sleep_store: Dict[str, np.ndarray] = {}
    index_rows = []

    user_dirs = sorted([p for p in DATA_ROOT.glob('user_*') if p.is_dir()])
    for udir in user_dirs:
        user_id = udir.name

        rr_by_day = build_rr_patches(udir / 'RR.csv', args.window_min)
        act_by_day = build_activity_patches(udir / 'Activity.csv', args.window_min)
        sleep_by_day = build_sleep_patches(udir / 'sleep.csv', args.window_min)

        day_set = set(rr_by_day.keys()) | set(act_by_day.keys()) | set(sleep_by_day.keys())
        if not labels.empty:
            day_set |= set(labels[labels['user_id'] == user_id]['day'].astype(int).tolist())

        for d in sorted(day_set):
            sid = f'{user_id}_day{d}'

            rr_arr = rr_by_day.get(d, np.zeros((0, len(RR_PATCH_FEAT)), dtype=np.float32))
            act_arr = act_by_day.get(d, np.zeros((0, len(ACT_PATCH_FEAT)), dtype=np.float32))
            sleep_arr = sleep_by_day.get(d, np.zeros((0, len(SLEEP_PATCH_FEAT)), dtype=np.float32))

            rr_store[sid] = rr_arr
            act_store[sid] = act_arr
            sleep_store[sid] = sleep_arr

            row = {
                'sample_id': sid,
                'user_id': user_id,
                'day': int(d),
                'rr_len': int(rr_arr.shape[0]),
                'act_len': int(act_arr.shape[0]),
                'sleep_len': int(sleep_arr.shape[0]),
                'rr_has': int(rr_arr.shape[0] > 0),
                'act_has': int(act_arr.shape[0] > 0),
                'sleep_has': int(sleep_arr.shape[0] > 0),
            }

            if not labels.empty:
                m = labels[(labels['user_id'] == user_id) & (labels['day'].astype(int) == int(d))]
                if len(m) > 0:
                    for t in ['label_daily_stress', 'label_stai1', 'label_stai2']:
                        if t in m.columns:
                            row[t] = _safe_float(m[t].iloc[0])

            index_rows.append(row)

    save_dict = {}
    for sid, arr in rr_store.items():
        save_dict[f'rr::{sid}'] = arr
    for sid, arr in act_store.items():
        save_dict[f'act::{sid}'] = arr
    for sid, arr in sleep_store.items():
        save_dict[f'sleep::{sid}'] = arr

    save_dict['rr_feat_names'] = np.array(RR_PATCH_FEAT, dtype=object)
    save_dict['act_feat_names'] = np.array(ACT_PATCH_FEAT, dtype=object)
    save_dict['sleep_feat_names'] = np.array(SLEEP_PATCH_FEAT, dtype=object)

    index_df = pd.DataFrame(index_rows)
    np.savez_compressed(out_npz, **save_dict)
    index_df.to_csv(out_index, index=False)

    rr_dim = len(RR_PATCH_FEAT)
    act_dim = len(ACT_PATCH_FEAT)
    sleep_dim = len(SLEEP_PATCH_FEAT)

    rr_min = int(index_df['rr_len'].min()) if len(index_df) else 0
    rr_max = int(index_df['rr_len'].max()) if len(index_df) else 0
    act_min = int(index_df['act_len'].min()) if len(index_df) else 0
    act_max = int(index_df['act_len'].max()) if len(index_df) else 0
    sleep_min = int(index_df['sleep_len'].min()) if len(index_df) else 0
    sleep_max = int(index_df['sleep_len'].max()) if len(index_df) else 0

    print(f'[OK] saved event npz: {out_npz}')
    print(f'[OK] saved index csv: {out_index}')
    print(f'[OK] window_min={args.window_min}, n_samples={len(index_df)}')
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
