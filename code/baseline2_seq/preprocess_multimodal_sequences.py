from __future__ import annotations

"""
多模态4D序列预处理脚本（结构化小时桶版本）
================================================

目标
----
把 MMASH 原始模态文件（RR / Activity / Sleep）统一整理成可直接喂给 encoder 的 4D 张量：

- rr_tensor:    [user, day_slot, hour, rr_feat]
- act_tensor:   [user, day_slot, hour, act_feat]
- sleep_tensor: [user, day_slot, hour, sleep_feat]

同时输出：
- day_mask: [user, day_slot]                        -> 该用户该 day_slot 是否有效
- rr_day_mask / act_day_mask / sleep_day_mask       -> 该 day_slot 某模态是否存在
- y_daily_stress / y_stai1 / y_stai2               -> 与 day_slot 对齐的标签
- meta.csv                                          -> 便于人工检查的映射信息

说明
----
1) 这里的第一维 user 是“用户维度”，不是训练时 mini-batch。
2) day_slot 是“每个用户内部按天排序后的槽位索引”，用于张量对齐。
3) 本脚本采用“层级统计+连续片段增强”的数据组织，不是事件 token 序列方案。
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
OUT_NPZ = OUT_DIR / 'multimodal_4d_tensors.npz'
OUT_META = OUT_DIR / 'multimodal_4d_meta.csv'


# =========================
# 模态特征定义
# =========================
# RR 每小时统计特征
RR_FEAT_NAMES = [
    'rr_count',      # 该小时 RR 点数
    'rr_ibi_mean',   # IBI均值
    'rr_ibi_std',    # IBI标准差
    'rr_ibi_min',    # IBI最小值
    'rr_ibi_max',    # IBI最大值
]

# Activity 每小时特征：11种活动类型分钟数 + 总分钟 + 事件数
ACT_FEAT_NAMES = [f'act_code_{i}_minutes' for i in range(11)] + ['act_total_minutes', 'act_n_events']

# Sleep 每小时特征：分钟数、片段数、连续性、最长run
SLEEP_FEAT_NAMES = [
    'sleep_minutes',
    'sleep_segment_count',
    'sleep_cont_from_prev',
    'sleep_cont_to_next',
    'sleep_longest_run_len',
]

# 固定day槽位语义：索引0永远对应day1，索引1永远对应day2
TARGET_DAYS = [1, 2]


# =========================
# 通用工具函数
# =========================
def _to_minute(hhmm: str) -> int:
    """'HH:MM' -> 从00:00开始的分钟数。"""
    h, m = hhmm.strip().split(':')
    return int(h) * 60 + int(m)


def _safe_float(x) -> float:
    """安全转 float，失败返回 np.nan。"""
    try:
        return float(x)
    except Exception:
        return np.nan


def _hour_from_hhmm(s: str) -> int:
    """从 'HH:MM' / 'HH:MM:SS' 提取小时。"""
    return int(str(s).strip().split(':')[0]) % 24


def _normalize_day_for_mmash(day_val: int) -> int:
    """
    统一 day 到 {1, 2}。
    - day==1 保持1
    - day==2 保持2
    - 其他异常值（如 -29、0、3...）统一映射为2

    用于在预处理阶段修正跨午夜等异常编码，不改动原始CSV。
    """
    if day_val == 1:
        return 1
    if day_val == 2:
        return 2
    return 2


def _read_csv_maybe_skip_comment(path: Path, required_col: str) -> pd.DataFrame:
    """
    MMASH 文件常见：第一行是注释，第二行才是表头。
    若直接读取后不存在 required_col，则自动 skiprows=1 重读。
    """
    df = pd.read_csv(path)
    if required_col not in df.columns:
        df = pd.read_csv(path, skiprows=1)
    if len(df.columns) > 0 and df.columns[0] == '':
        # 去掉空首列（常见索引残留列）
        df = df.drop(columns=[df.columns[0]])
    return df


# =========================
# RR 模态预处理
# =========================
def build_rr_day_hour(rr_csv: Path) -> Dict[Tuple[int, int], np.ndarray]:
    """
    将 RR.csv 聚合成 {(day, hour): rr_feat_vector}。
    """
    if not rr_csv.exists():
        return {}

    df = _read_csv_maybe_skip_comment(rr_csv, 'ibi_s')
    if not {'ibi_s', 'day', 'time'}.issubset(df.columns):
        return {}

    # 数值化与基础过滤
    df['ibi_s'] = pd.to_numeric(df['ibi_s'], errors='coerce')
    df['day'] = pd.to_numeric(df['day'], errors='coerce')
    df = df.dropna(subset=['ibi_s', 'day', 'time']).copy()

    # 简单生理范围过滤（去明显伪迹）
    df = df[(df['ibi_s'] >= 0.2) & (df['ibi_s'] <= 2.5)]
    if df.empty:
        return {}

    # day 归一化 + 提取小时
    df['day'] = df['day'].astype(int).apply(_normalize_day_for_mmash)
    df['hour'] = df['time'].astype(str).apply(_hour_from_hhmm)

    out: Dict[Tuple[int, int], np.ndarray] = {}
    for (d, h), g in df.groupby(['day', 'hour']):
        v = g['ibi_s'].to_numpy(dtype=float)
        out[(int(d), int(h))] = np.array(
            [len(v), np.mean(v), np.std(v), np.min(v), np.max(v)],
            dtype=np.float32,
        )

    return out


# =========================
# Activity 模态预处理
# =========================
def _alloc_event_minutes(st_m: int, ed_m: int) -> Dict[int, float]:
    """
    把一个事件 [st_m, ed_m) 按重叠分钟分配到小时桶。
    返回 {hour: minutes}。
    """
    res: Dict[int, float] = {}
    cur = st_m
    while cur < ed_m:
        h_start = (cur // 60) * 60
        h_end = h_start + 60
        overlap_end = min(h_end, ed_m)
        mins = max(0, overlap_end - cur)

        hh = (h_start // 60) % 24
        if mins > 0:
            res[hh] = res.get(hh, 0.0) + float(mins)

        cur = overlap_end

    return res


def build_activity_day_hour(act_csv: Path) -> Dict[Tuple[int, int], np.ndarray]:
    """
    将 Activity.csv 聚合成 {(day, hour): act_feat_vector}。
    特征含义：
    - 各 activity code 的分钟数（0~10）
    - 该小时总活动分钟数
    - 该小时事件条数
    """
    if not act_csv.exists():
        return {}

    df = _read_csv_maybe_skip_comment(act_csv, 'Activity')
    if not {'Activity', 'Start', 'Day'}.issubset(df.columns):
        return {}

    df['Activity'] = pd.to_numeric(df['Activity'], errors='coerce')
    df['Day'] = pd.to_numeric(df['Day'], errors='coerce')
    df = df.dropna(subset=['Activity', 'Day', 'Start']).copy()

    out: Dict[Tuple[int, int], np.ndarray] = {}

    for _, r in df.iterrows():
        code = int(r['Activity'])
        if code < 0 or code > 10:
            continue

        day = _normalize_day_for_mmash(int(r['Day']))
        st = str(r['Start'])
        ed = str(r['End']) if 'End' in df.columns and not pd.isna(r.get('End', np.nan)) else ''

        # 起始时间解析
        try:
            st_m = _to_minute(st)
        except Exception:
            continue

        # 结束时间解析：缺失时默认 +30 分钟
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

        # 分配到小时桶
        alloc = _alloc_event_minutes(st_m, ed_m)
        for hh, mins in alloc.items():
            key = (day, hh)
            if key not in out:
                out[key] = np.zeros(len(ACT_FEAT_NAMES), dtype=np.float32)

            out[key][code] += mins   # 对应 activity code 分钟
            out[key][11] += mins     # act_total_minutes
            out[key][12] += 1.0      # act_n_events

    return out


# =========================
# Sleep 模态预处理
# =========================
def build_sleep_day_hour(sleep_csv: Path) -> Dict[Tuple[int, int], np.ndarray]:
    """
    将 sleep.csv 聚合成 {(day, hour): sleep_feat_vector}。
    含连续性增强特征：
    - cont_from_prev / cont_to_next
    - longest_run_len
    """
    if not sleep_csv.exists():
        return {}

    df = _read_csv_maybe_skip_comment(sleep_csv, 'In Bed Date')
    if not {'In Bed Date', 'In Bed Time', 'Out Bed Date', 'Out Bed Time'}.issubset(df.columns):
        return {}

    out: Dict[Tuple[int, int], np.ndarray] = {}

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

        day = _normalize_day_for_mmash(int(in_day))

        # 绝对分钟轴（用于跨天片段）
        st_abs = day * 1440 + st_m
        ed_abs = int(out_day) * 1440 + ed_m
        if ed_abs <= st_abs:
            ed_abs += 1440

        dur = float(ed_abs - st_abs)
        if dur <= 0:
            continue

        # 1) minutes / segment_count / longest_run_len
        cur = st_abs
        covered_hours_abs: List[int] = []
        while cur < ed_abs:
            h_start = (cur // 60) * 60
            h_end = h_start + 60
            overlap_end = min(h_end, ed_abs)
            mins = max(0, overlap_end - cur)

            hh = (h_start // 60) % 24
            key = (day, hh)
            if key not in out:
                out[key] = np.zeros(len(SLEEP_FEAT_NAMES), dtype=np.float32)

            if mins > 0:
                out[key][0] += mins                 # sleep_minutes
                out[key][1] += 1.0                  # sleep_segment_count
                out[key][4] = max(out[key][4], dur) # sleep_longest_run_len
                covered_hours_abs.append(h_start // 60)

            cur = overlap_end

        # 2) cont_from_prev / cont_to_next
        covered_hours_abs = sorted(set(covered_hours_abs))
        for h_abs in covered_hours_abs[:-1]:
            key_cur = (day, h_abs % 24)
            key_nxt = (day, (h_abs + 1) % 24)

            if key_cur not in out:
                out[key_cur] = np.zeros(len(SLEEP_FEAT_NAMES), dtype=np.float32)
            if key_nxt not in out:
                out[key_nxt] = np.zeros(len(SLEEP_FEAT_NAMES), dtype=np.float32)

            out[key_cur][3] = 1.0  # sleep_cont_to_next
            out[key_nxt][2] = 1.0  # sleep_cont_from_prev

    return out


# =========================
# 标签加载
# =========================
def load_labels() -> pd.DataFrame:
    """读取 day-level 标签并去重。"""
    if not LABEL_PATH.exists():
        return pd.DataFrame(columns=['user_id', 'day', 'label_daily_stress', 'label_stai1', 'label_stai2'])

    df = pd.read_csv(LABEL_PATH)
    keep = [c for c in ['user_id', 'day', 'label_daily_stress', 'label_stai1', 'label_stai2'] if c in df.columns]
    return df[keep].drop_duplicates(['user_id', 'day']).copy()


# =========================
# 主流程
# =========================
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    label_df = load_labels()

    # 用户目录
    user_dirs = sorted([p for p in DATA_ROOT.glob('user_*') if p.is_dir()])
    user_ids = [p.name for p in user_dirs]

    # 每用户的模态 day-hour 映射
    rr_maps: Dict[str, Dict[Tuple[int, int], np.ndarray]] = {}
    act_maps: Dict[str, Dict[Tuple[int, int], np.ndarray]] = {}
    sl_maps: Dict[str, Dict[Tuple[int, int], np.ndarray]] = {}

    # 每用户有效day集合（已归一化到1/2）
    user_days: Dict[str, List[int]] = {u: [] for u in user_ids}

    for udir in user_dirs:
        u = udir.name

        rr_maps[u] = build_rr_day_hour(udir / 'RR.csv')
        act_maps[u] = build_activity_day_hour(udir / 'Activity.csv')
        sl_maps[u] = build_sleep_day_hour(udir / 'sleep.csv')

        # 任一模态出现day就纳入
        dset = set([d for d, _ in rr_maps[u].keys()] + [d for d, _ in act_maps[u].keys()] + [d for d, _ in sl_maps[u].keys()])

        # 标签里的day也纳入（并做day归一化）
        if not label_df.empty:
            dset.update(
                label_df[label_df['user_id'] == u]['day'].dropna().astype(int).apply(_normalize_day_for_mmash).tolist()
            )

        user_days[u] = sorted(list(dset))

    # 只保留至少有1天数据的用户
    users = [u for u in user_ids if len(user_days[u]) > 0]
    n_user = len(users)
    n_day = len(TARGET_DAYS)  # 固定为2个槽位：day1/day2

    # 4D 张量初始化: [user, day_slot(固定day1/day2), hour, feat]
    rr = np.zeros((n_user, n_day, 24, len(RR_FEAT_NAMES)), dtype=np.float32)
    act = np.zeros((n_user, n_day, 24, len(ACT_FEAT_NAMES)), dtype=np.float32)
    slp = np.zeros((n_user, n_day, 24, len(SLEEP_FEAT_NAMES)), dtype=np.float32)

    # day / modality mask
    day_mask = np.zeros((n_user, n_day), dtype=np.float32)
    rr_mask = np.zeros((n_user, n_day), dtype=np.float32)
    act_mask = np.zeros((n_user, n_day), dtype=np.float32)
    slp_mask = np.zeros((n_user, n_day), dtype=np.float32)

    # 标签矩阵（按 day_slot 对齐）
    y_stress = np.full((n_user, n_day), np.nan, dtype=np.float32)
    y_stai1 = np.full((n_user, n_day), np.nan, dtype=np.float32)
    y_stai2 = np.full((n_user, n_day), np.nan, dtype=np.float32)

    # 元信息（用于检查映射）
    meta_rows = []

    for i, u in enumerate(users):
        ulab = label_df[label_df['user_id'] == u] if not label_df.empty else pd.DataFrame()

        # 固定槽位：j=0->day1, j=1->day2
        for j, d in enumerate(TARGET_DAYS):
            has_rr = has_act = has_sl = False
            for h in range(24):
                if (d, h) in rr_maps[u]:
                    rr[i, j, h, :] = rr_maps[u][(d, h)]
                    has_rr = True

                if (d, h) in act_maps[u]:
                    act[i, j, h, :] = act_maps[u][(d, h)]
                    has_act = True

                if (d, h) in sl_maps[u]:
                    slp[i, j, h, :] = sl_maps[u][(d, h)]
                    has_sl = True

            rr_mask[i, j] = 1.0 if has_rr else 0.0
            act_mask[i, j] = 1.0 if has_act else 0.0
            slp_mask[i, j] = 1.0 if has_sl else 0.0

            # day_mask 表示该day槽位是否至少有一个模态可用
            day_mask[i, j] = 1.0 if (has_rr or has_act or has_sl) else 0.0

            # day标签对齐
            if not ulab.empty:
                m = ulab[ulab['day'].astype(int).apply(_normalize_day_for_mmash) == int(d)]
                if len(m) > 0:
                    if 'label_daily_stress' in m.columns:
                        y_stress[i, j] = _safe_float(m['label_daily_stress'].iloc[0])
                    if 'label_stai1' in m.columns:
                        y_stai1[i, j] = _safe_float(m['label_stai1'].iloc[0])
                    if 'label_stai2' in m.columns:
                        y_stai2[i, j] = _safe_float(m['label_stai2'].iloc[0])

            meta_rows.append(
                {
                    'user_index': i,
                    'user_id': u,
                    'day': d,
                    'rr_has': int(rr_mask[i, j]),
                    'act_has': int(act_mask[i, j]),
                    'sleep_has': int(slp_mask[i, j]),
                    'day_has_any_modality': int(day_mask[i, j]),
                    'label_daily_stress': y_stress[i, j],
                    'label_stai1': y_stai1[i, j],
                    'label_stai2': y_stai2[i, j],
                }
            )

    # 保存张量与特征名
    np.savez_compressed(
        OUT_NPZ,
        rr_tensor=rr,
        act_tensor=act,
        sleep_tensor=slp,
        day_mask=day_mask,
        rr_day_mask=rr_mask,
        act_day_mask=act_mask,
        sleep_day_mask=slp_mask,
        y_daily_stress=y_stress,
        y_stai1=y_stai1,
        y_stai2=y_stai2,
        users=np.array(users, dtype=object),
        rr_feat_names=np.array(RR_FEAT_NAMES, dtype=object),
        act_feat_names=np.array(ACT_FEAT_NAMES, dtype=object),
        sleep_feat_names=np.array(SLEEP_FEAT_NAMES, dtype=object),
    )

    pd.DataFrame(meta_rows).to_csv(OUT_META, index=False)

    print(f'[OK] saved tensors: {OUT_NPZ}')
    print(f'[OK] saved meta: {OUT_META}')
    print(f'[OK] rr_tensor shape={rr.shape}')
    print(f'[OK] act_tensor shape={act.shape}')
    print(f'[OK] sleep_tensor shape={slp.shape}')
    print(f'[OK] rr_mask tensor shape={rr_mask.shape}')   
    print(f'[OK] act_mask tensor shape={act_mask.shape}')
    print(f'[OK] sleep_mask tensor shape={slp_mask.shape}')
#     [OK] rr_tensor shape=(22, 2, 24, 5)
# [OK] act_tensor shape=(22, 2, 24, 13)
# [OK] sleep_tensor shape=(22, 2, 24, 5)
# [OK] rr_mask tensor shape=(22, 2)   需要先把mask的维度和tensor对齐，然后才能自动广播
# [OK] act_mask tensor shape=(22, 2)
# [OK] sleep_mask tensor shape=(22, 2)
# [OK] 维度解释: [batch(user), day, hour, feature]
    print('[OK] 维度解释: [batch(user), day, hour, feature]')


if __name__ == '__main__':
    main()
