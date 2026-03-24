from __future__ import annotations

"""
questionnaire 静态模态预处理
============================

输出：
1) static_questionnaire.npz
   - static_x: [n_user, d_static]
   - users: 用户ID
   - feature_names: 特征名
2) static_questionnaire.csv（便于人工查看）

默认策略：
- 读取每个 user_x/questionnaire.csv 的首行问卷值
- 排除明显标签泄漏字段：STAI1, STAI2, Daily_stress
- 保留其余静态问卷字段作为 static modality
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

DATA_ROOT = Path('/home/wqshen/mm-test/dataset/physionet.org/files/mmash/1.0.0/DataPaper')
OUT_DIR = Path('/home/wqshen/mm-test/code/baseline2_seq/processed')
OUT_NPZ = OUT_DIR / 'static_questionnaire.npz'
OUT_CSV = OUT_DIR / 'static_questionnaire.csv'

# 默认排除：与目标同源、强泄漏风险字段
DEFAULT_EXCLUDE_COLS = {'STAI1', 'STAI2', 'Daily_stress'}


def _read_csv_maybe_skip_comment(path: Path, required_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if required_col not in df.columns:
        df = pd.read_csv(path, skiprows=1)
    if len(df.columns) > 0 and df.columns[0] == '':
        df = df.drop(columns=[df.columns[0]])
    return df


def build_static_table() -> pd.DataFrame:
    rows: List[Dict] = []

    user_dirs = sorted([p for p in DATA_ROOT.glob('user_*') if p.is_dir()])
    for udir in user_dirs:
        user_id = udir.name
        q_path = udir / 'questionnaire.csv'
        if not q_path.exists():
            continue

        dfq = _read_csv_maybe_skip_comment(q_path, 'MEQ')
        if dfq.empty:
            continue

        # 问卷通常只有一行，这里取首行
        r = dfq.iloc[0]

        row = {'user_id': user_id}
        for c in dfq.columns:
            if c in DEFAULT_EXCLUDE_COLS:
                continue
            # 仅保留数值列（可转float）
            try:
                row[c] = float(r[c])
            except Exception:
                row[c] = np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # 按 user_id 排序，保证输出稳定
    out = out.sort_values('user_id').reset_index(drop=True)

    # 缺失值：用列中位数填补
    feat_cols = [c for c in out.columns if c != 'user_id']
    for c in feat_cols:
        med = out[c].median(skipna=True)
        if np.isnan(med):
            med = 0.0
        out[c] = out[c].fillna(med)

    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    static_df = build_static_table()
    if static_df.empty:
        print('[WARN] no questionnaire data found.')
        return

    users = static_df['user_id'].to_numpy(dtype=object)
    feat_cols = [c for c in static_df.columns if c != 'user_id']

    static_x = static_df[feat_cols].to_numpy(dtype=np.float32)

    np.savez_compressed(
        OUT_NPZ,
        static_x=static_x,
        users=users,
        feature_names=np.array(feat_cols, dtype=object),
    )
    static_df.to_csv(OUT_CSV, index=False)

    print(f'[OK] saved npz: {OUT_NPZ}')
    print(f'[OK] saved csv: {OUT_CSV}')
    print(f'[OK] static shape = {static_x.shape} (n_user, d_static)')
    print(f'[OK] d_static = {len(feat_cols)}')
    print(f'[OK] features = {feat_cols}')


if __name__ == '__main__':
    main()
