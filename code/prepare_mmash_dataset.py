#!/usr/bin/env python3
"""
MMASH 数据预处理脚本（稳健版）。

核心改进：
1) 列名标准化（小写、去空格、空格转下划线），降低字段匹配脆弱性；
2) 同时导出 day-level 和 user-level 两套数据；
3) 支持特征开关：PANAS / STAI2 是否作为输入特征；
4) 增强 summary，便于检查缺失与样本覆盖。

说明：
- questionnaire 标签（daily_stress/stai1/stai2）通常是用户级，
  在 day-level 表里会复制到该用户的每一天样本。
- 严格评估时请使用按 user_id 分组的划分（GroupKFold/LOSO）。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


ACTIVITY_MAP = {
    1: "sleeping",
    2: "laying_down",
    3: "sitting",
    4: "light_movement",
    5: "medium_movement",
    6: "heavy_movement",
    7: "eating",
    8: "small_screen",
    9: "large_screen",
    10: "caffeine",
    11: "smoking",
    12: "alcohol",
}


@dataclass
class Config:
    data_root: Path
    out_dir: Path
    include_stai2_as_feature: bool
    include_panas_as_feature: bool
    ibi_min: float = 0.3
    ibi_max: float = 2.0


def normalize_col(c: str) -> str:
    return str(c).strip().lower().replace(" ", "_")


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    df.columns = [str(c).strip() for c in df.columns]

    drop_cols = []
    for c in df.columns:
        lc = normalize_col(c)
        if lc.startswith("unnamed") or lc in {"", "cl"}:
            drop_cols.append(c)
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df.columns = [normalize_col(c) for c in df.columns]
    return df


def find_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_minutes(start: object, end: object) -> float:
    if pd.isna(start) or pd.isna(end):
        return 0.0
    try:
        st = pd.to_datetime(str(start), format="%H:%M")
        ed = pd.to_datetime(str(end), format="%H:%M")
        if ed < st:
            ed += pd.Timedelta(days=1)
        return float((ed - st).total_seconds() / 60.0)
    except Exception:
        return 0.0


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def extract_rr_features(rr_df: pd.DataFrame, ibi_min: float, ibi_max: float) -> pd.DataFrame:
    if rr_df.empty:
        return pd.DataFrame(columns=["day"])

    ibi_col = find_first_existing(rr_df, ["ibi_s", "ibi", "rr", "rr_interval"])
    day_col = find_first_existing(rr_df, ["day"])
    if ibi_col is None or day_col is None:
        return pd.DataFrame(columns=["day"])

    rr = rr_df.copy()
    rr[ibi_col] = to_num(rr[ibi_col])
    rr[day_col] = to_num(rr[day_col])
    rr = rr.dropna(subset=[ibi_col, day_col])
    rr = rr[(rr[ibi_col] >= ibi_min) & (rr[ibi_col] <= ibi_max)]

    rows: List[Dict[str, float]] = []
    for day, g in rr.groupby(day_col):
        ibi = g[ibi_col].to_numpy(dtype=float)
        if ibi.size < 5:
            continue

        hr = 60.0 / ibi
        diff_ibi = np.diff(ibi)

        sdnn = float(np.std(ibi, ddof=1)) if ibi.size > 1 else 0.0
        row = {
            "day": int(day),
            "rr_count": float(ibi.size),
            "rr_ibi_mean": float(np.mean(ibi)),
            "rr_ibi_std": sdnn,
            "ibi_min": float(np.min(ibi)),
            "ibi_max": float(np.max(ibi)),
            "hr_mean": float(np.mean(hr)),
            "hr_std": float(np.std(hr, ddof=1)) if hr.size > 1 else 0.0,
            "hr_min": float(np.min(hr)),
            "hr_max": float(np.max(hr)),
            "sdnn": sdnn,
            "rmssd": float(np.sqrt(np.mean(diff_ibi**2))) if diff_ibi.size > 0 else 0.0,
            "pnn50": float(np.mean(np.abs(diff_ibi) > 0.05)) if diff_ibi.size > 0 else 0.0,
            "cvnn": float(sdnn / np.mean(ibi)) if np.mean(ibi) > 0 else 0.0,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def extract_activity_features(activity_df: pd.DataFrame) -> pd.DataFrame:
    if activity_df.empty:
        return pd.DataFrame(columns=["day"])

    day_col = find_first_existing(activity_df, ["day"])
    act_col = find_first_existing(activity_df, ["activity", "activity_id"])
    start_col = find_first_existing(activity_df, ["start", "start_time"])
    end_col = find_first_existing(activity_df, ["end", "end_time"])
    if day_col is None or act_col is None:
        return pd.DataFrame(columns=["day"])

    ac = activity_df.copy()
    ac[day_col] = to_num(ac[day_col])
    ac[act_col] = to_num(ac[act_col])
    ac = ac.dropna(subset=[day_col, act_col])

    if start_col and end_col:
        ac["duration_min"] = [_safe_minutes(s, e) for s, e in zip(ac[start_col], ac[end_col])]
    else:
        ac["duration_min"] = 0.0

    rows: List[Dict[str, float]] = []
    for day, g in ac.groupby(day_col):
        row: Dict[str, float] = {
            "day": int(day),
            "activity_total_records": float(g.shape[0]),
            "activity_total_minutes": float(g["duration_min"].sum()),
        }
        for act_id, act_name in ACTIVITY_MAP.items():
            sel = g[g[act_col] == act_id]
            row[f"act_{act_name}_minutes"] = float(sel["duration_min"].sum())
            row[f"act_{act_name}_count"] = float(sel.shape[0])

        row["act_sedentary_minutes"] = row["act_laying_down_minutes"] + row["act_sitting_minutes"]
        row["act_mvpa_minutes"] = row["act_medium_movement_minutes"] + row["act_heavy_movement_minutes"]
        row["act_screen_minutes"] = row["act_small_screen_minutes"] + row["act_large_screen_minutes"]
        row["act_substance_events"] = row["act_caffeine_count"] + row["act_smoking_count"] + row["act_alcohol_count"]

        rows.append(row)

    return pd.DataFrame(rows)


def extract_sleep_features(sleep_df: pd.DataFrame) -> pd.DataFrame:
    if sleep_df.empty:
        return pd.DataFrame(columns=["day"])

    day_col = find_first_existing(sleep_df, ["in_bed_date", "day"])
    if day_col is None:
        return pd.DataFrame(columns=["day"])

    col_map = {
        "latency": "sleep_latency",
        "efficiency": "sleep_efficiency",
        "total_minutes_in_bed": "sleep_total_minutes_in_bed",
        "total_sleep_time_(tst)": "sleep_tst",
        "wake_after_sleep_onset_(waso)": "sleep_waso",
        "number_of_awakenings": "sleep_awakenings_n",
        "average_awakening_length": "sleep_awake_len_avg",
        "movement_index": "sleep_movement_index",
        "fragmentation_index": "sleep_fragmentation_index",
        "sleep_fragmentation_index": "sleep_fragmentation_composite",
    }

    sl = sleep_df.copy()
    sl[day_col] = to_num(sl[day_col])
    sl = sl.dropna(subset=[day_col])

    for raw_col in list(col_map.keys()):
        if raw_col in sl.columns:
            sl[raw_col] = to_num(sl[raw_col])

    sum_cols = [c for c in ["total_minutes_in_bed", "total_sleep_time_(tst)", "wake_after_sleep_onset_(waso)", "number_of_awakenings"] if c in sl.columns]
    mean_cols = [c for c in ["latency", "efficiency", "average_awakening_length", "movement_index", "fragmentation_index", "sleep_fragmentation_index"] if c in sl.columns]

    agg_sum = sl.groupby(day_col, as_index=False)[sum_cols].sum(numeric_only=True) if sum_cols else pd.DataFrame({day_col: sl[day_col].drop_duplicates()})
    agg_mean = sl.groupby(day_col, as_index=False)[mean_cols].mean(numeric_only=True) if mean_cols else pd.DataFrame({day_col: sl[day_col].drop_duplicates()})

    agg = agg_sum.merge(agg_mean, on=day_col, how="outer")
    agg = agg.rename(columns={day_col: "day"})

    rename_pairs = {k: v for k, v in col_map.items() if k in agg.columns}
    return agg.rename(columns=rename_pairs)


def _num(v: object) -> Optional[float]:
    try:
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def extract_static_features(q_df: pd.DataFrame, ui_df: pd.DataFrame, include_stai2_as_feature: bool, include_panas_as_feature: bool) -> Dict[str, Optional[float]]:
    q = q_df.iloc[0].to_dict() if not q_df.empty else {}
    ui = ui_df.iloc[0].to_dict() if not ui_df.empty else {}

    def get_q(*names: str) -> Optional[float]:
        for n in names:
            if n in q:
                return _num(q.get(n))
        return None

    def get_ui(*names: str) -> Optional[float]:
        for n in names:
            if n in ui:
                return _num(ui.get(n))
        return None

    features: Dict[str, Optional[float]] = {
        "age": get_ui("age"),
        "weight": get_ui("weight"),
        "height": get_ui("height"),
        "bmi": None,
        "meq": get_q("meq"),
        "pittsburgh": get_q("pittsburgh"),
        "bis": get_q("bisbas_bis", "bis_bis", "bis"),
        "bas_reward": get_q("bisbas_reward", "bas_reward", "reward"),
        "bas_drive": get_q("bisbas_drive", "bas_drive", "drive"),
        "bas_fun": get_q("bisbas_fun", "bas_fun", "fun"),
    }

    if features["height"] is not None and features["weight"] is not None and features["height"] > 0:
        h_m = features["height"] / 100.0
        features["bmi"] = float(features["weight"] / (h_m**2))

    gender_raw = None
    for gc in ["gender", "sex"]:
        if gc in ui:
            gender_raw = str(ui.get(gc)).strip().upper()
            break
    features["gender_male"] = 1.0 if gender_raw == "M" else 0.0 if gender_raw == "F" else None

    if include_stai2_as_feature:
        features["stai2_trait"] = get_q("stai2")

    if include_panas_as_feature:
        panas_pos_cols = [c for c in q.keys() if str(c).startswith("panas_pos_")]
        panas_neg_cols = [c for c in q.keys() if str(c).startswith("panas_neg_")]
        pos_vals = [_num(q.get(c)) for c in panas_pos_cols]
        neg_vals = [_num(q.get(c)) for c in panas_neg_cols]
        pos_vals = [v for v in pos_vals if v is not None]
        neg_vals = [v for v in neg_vals if v is not None]
        features["panas_pos_mean"] = float(np.mean(pos_vals)) if pos_vals else None
        features["panas_neg_mean"] = float(np.mean(neg_vals)) if neg_vals else None

    return features


def extract_labels(q_df: pd.DataFrame) -> Dict[str, Optional[float]]:
    if q_df.empty:
        return {"label_daily_stress": None, "label_stai1": None, "label_stai2": None}

    row = q_df.iloc[0]

    def get(*names: str) -> Optional[float]:
        for n in names:
            if n in row.index:
                return _num(row[n])
        return None

    return {
        "label_daily_stress": get("daily_stress"),
        "label_stai1": get("stai1"),
        "label_stai2": get("stai2"),
    }


def user_aggregate(day_df: pd.DataFrame) -> pd.DataFrame:
    if day_df.empty:
        return pd.DataFrame()

    numeric_cols = day_df.select_dtypes(include=[np.number]).columns.tolist()
    for c in ["day", "label_daily_stress", "label_stai1", "label_stai2"]:
        if c in numeric_cols:
            numeric_cols.remove(c)

    out: Dict[str, float] = {
        "n_days": float(day_df["day"].nunique()) if "day" in day_df else float(day_df.shape[0]),
    }

    for c in numeric_cols:
        vals = pd.to_numeric(day_df[c], errors="coerce")
        out[f"{c}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
        out[f"{c}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else np.nan
        out[f"{c}_min"] = float(vals.min()) if vals.notna().any() else np.nan
        out[f"{c}_max"] = float(vals.max()) if vals.notna().any() else np.nan

    return pd.DataFrame([out])


def build_datasets(cfg: Config) -> Dict[str, pd.DataFrame]:
    user_dirs = sorted([p for p in cfg.data_root.glob("user_*") if p.is_dir()])

    day_rows: List[pd.DataFrame] = []
    user_rows: List[pd.DataFrame] = []

    for udir in user_dirs:
        uid = udir.name
        required = {
            "rr": udir / "RR.csv",
            "activity": udir / "Activity.csv",
            "sleep": udir / "sleep.csv",
            "questionnaire": udir / "questionnaire.csv",
            "user_info": udir / "user_info.csv",
        }
        if not all(p.exists() for p in required.values()):
            continue

        rr_df = _read_csv(required["rr"])
        ac_df = _read_csv(required["activity"])
        sl_df = _read_csv(required["sleep"])
        q_df = _read_csv(required["questionnaire"])
        ui_df = _read_csv(required["user_info"])

        rr_feat = extract_rr_features(rr_df, cfg.ibi_min, cfg.ibi_max)
        ac_feat = extract_activity_features(ac_df)
        sl_feat = extract_sleep_features(sl_df)

        day_df = rr_feat.merge(ac_feat, how="outer", on="day").merge(sl_feat, how="outer", on="day")
        if day_df.empty:
            continue

        static_feats = extract_static_features(
            q_df=q_df,
            ui_df=ui_df,
            include_stai2_as_feature=cfg.include_stai2_as_feature,
            include_panas_as_feature=cfg.include_panas_as_feature,
        )
        labels = extract_labels(q_df)

        for k, v in static_feats.items():
            day_df[k] = v
        for k, v in labels.items():
            day_df[k] = v
        day_df["user_id"] = uid
        day_rows.append(day_df)

        uagg = user_aggregate(day_df)
        if not uagg.empty:
            uagg["user_id"] = uid
            for k, v in static_feats.items():
                if k not in uagg.columns:
                    uagg[k] = v
            for k, v in labels.items():
                uagg[k] = v
            user_rows.append(uagg)

    day_level = pd.concat(day_rows, ignore_index=True) if day_rows else pd.DataFrame()
    user_level = pd.concat(user_rows, ignore_index=True) if user_rows else pd.DataFrame()

    if not day_level.empty:
        day_level = day_level.sort_values(["user_id", "day"]).reset_index(drop=True)
    if not user_level.empty:
        user_level = user_level.sort_values(["user_id"]).reset_index(drop=True)

    return {"day": day_level, "user": user_level}


def summarize(day_df: pd.DataFrame, user_df: pd.DataFrame) -> pd.DataFrame:
    stats: Dict[str, float] = {}

    stats["day_rows"] = int(day_df.shape[0]) if not day_df.empty else 0
    stats["user_rows"] = int(user_df.shape[0]) if not user_df.empty else 0
    stats["n_users_day"] = int(day_df["user_id"].nunique()) if (not day_df.empty and "user_id" in day_df) else 0
    stats["n_users_user"] = int(user_df["user_id"].nunique()) if (not user_df.empty and "user_id" in user_df) else 0
    stats["n_unique_days"] = int(day_df["day"].nunique()) if (not day_df.empty and "day" in day_df) else 0

    for pfx in ["rr_", "act_", "sleep_"]:
        cols = [c for c in day_df.columns if c.startswith(pfx)] if not day_df.empty else []
        if cols:
            row_not_all_nan = (~day_df[cols].isna().all(axis=1)).mean()
            stats[f"coverage_{pfx[:-1]}"] = float(row_not_all_nan)

    for lb in ["label_daily_stress", "label_stai1", "label_stai2"]:
        if not user_df.empty and lb in user_df.columns:
            s = pd.to_numeric(user_df[lb], errors="coerce")
            stats[f"{lb}_non_null"] = int(s.notna().sum())
            if s.notna().any():
                stats[f"{lb}_min"] = float(s.min())
                stats[f"{lb}_max"] = float(s.max())
                stats[f"{lb}_mean"] = float(s.mean())

    if not day_df.empty:
        miss = day_df.isna().mean().sort_values(ascending=False)
        for i, (col, ratio) in enumerate(miss.head(5).items(), start=1):
            stats[f"missing_top{i}_col"] = col
            stats[f"missing_top{i}_ratio"] = float(ratio)

        if "user_id" in day_df.columns and "day" in day_df.columns:
            n_days_per_user = day_df.groupby("user_id")["day"].nunique()
            stats["days_per_user_min"] = float(n_days_per_user.min())
            stats["days_per_user_max"] = float(n_days_per_user.max())
            stats["days_per_user_mean"] = float(n_days_per_user.mean())

    return pd.DataFrame({"stat": list(stats.keys()), "value": list(stats.values())})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MMASH day-level and user-level datasets.")
    parser.add_argument("--data-root", type=Path, default=Path("/home/wqshen/mm-test/dataset/physionet.org/files/mmash/1.0.0/DataPaper"), help="Root directory containing user_* folders.")
    parser.add_argument("--out-dir", type=Path, default=Path("/home/wqshen/mm-test/model/processed"), help="Output directory for processed files.")
    parser.add_argument("--include-stai2-as-feature", action="store_true", help="Include STAI2 as input feature (otherwise label only).")
    parser.add_argument("--include-panas-as-feature", action="store_true", help="Include PANAS aggregate features (otherwise excluded).")
    parser.add_argument("--ibi-min", type=float, default=0.3, help="Lower bound for valid IBI (seconds).")
    parser.add_argument("--ibi-max", type=float, default=2.0, help="Upper bound for valid IBI (seconds).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        data_root=args.data_root,
        out_dir=args.out_dir,
        include_stai2_as_feature=args.include_stai2_as_feature,
        include_panas_as_feature=args.include_panas_as_feature,
        ibi_min=args.ibi_min,
        ibi_max=args.ibi_max,
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    ds = build_datasets(cfg)
    day_df, user_df = ds["day"], ds["user"]

    day_path = cfg.out_dir / "mmash_day_level_features.csv"
    user_path = cfg.out_dir / "mmash_user_level_features.csv"
    summary_path = cfg.out_dir / "dataset_summary.csv"

    day_df.to_csv(day_path, index=False)
    user_df.to_csv(user_path, index=False)
    summary_df = summarize(day_df, user_df)
    summary_df.to_csv(summary_path, index=False)

    print(f"[OK] Saved day-level dataset: {day_path}")
    print(f"[OK] Saved user-level dataset: {user_path}")
    print(f"[OK] Saved summary: {summary_path}")
    print(summary_df)


if __name__ == "__main__":
    main()
