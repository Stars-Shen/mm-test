from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from config import TrainConfig
from data import build_feature_columns, load_user_level_data, make_xy
from metrics import regression_metrics
from modeling import ModelFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=Path("/home/wqshen/mm-test/code/processed/mmash_user_level_features.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("/home/wqshen/mm-test/code/outputs"))
    parser.add_argument("--target", type=str, default="label_daily_stress", choices=["label_daily_stress", "label_stai1"])
    parser.add_argument("--model-name", type=str, default="ridge", choices=["ridge", "rf"])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--use-stai2-feature", action="store_true")
    parser.add_argument("--keep-panas", action="store_true")
    parser.add_argument("--max-missing-ratio", type=float, default=0.40)
    parser.add_argument("--no-missing-indicator", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        target=args.target,
        model_name=args.model_name,
        n_splits=args.n_splits,
        random_state=args.random_state,
        drop_panas=not args.keep_panas,
        use_stai2_feature=args.use_stai2_feature,
        max_missing_ratio=args.max_missing_ratio,
        add_missing_indicator=not args.no_missing_indicator,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_user_level_data(cfg.data_path)
    feature_cols = build_feature_columns(df, cfg.drop_panas, cfg.use_stai2_feature)
    X, y, groups = make_xy(
        df=df,
        feature_cols=feature_cols,
        target=cfg.target,
        max_missing_ratio=cfg.max_missing_ratio,
        add_missing_indicator=cfg.add_missing_indicator,
    )

    n_unique_groups = len(np.unique(groups))
    n_splits = min(cfg.n_splits, n_unique_groups)
    if n_splits < 2:
        raise ValueError("可用用户数不足，无法做 GroupKFold。")

    cv = GroupKFold(n_splits=n_splits)

    fold_rows = []
    oof_pred = np.zeros_like(y, dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
        model = ModelFactory(model_name=cfg.model_name, random_state=cfg.random_state).build()
        model.fit(X.iloc[tr_idx], y[tr_idx])
        pred = model.predict(X.iloc[va_idx])
        oof_pred[va_idx] = pred

        m = regression_metrics(y[va_idx], pred)
        m["fold"] = fold
        m["n_train"] = int(len(tr_idx))
        m["n_valid"] = int(len(va_idx))
        fold_rows.append(m)

    fold_df = pd.DataFrame(fold_rows)
    cv_metrics = regression_metrics(y, oof_pred)

    fold_path = cfg.output_dir / f"cv_folds_{cfg.target}_{cfg.model_name}.csv"
    pred_path = cfg.output_dir / f"oof_pred_{cfg.target}_{cfg.model_name}.csv"
    summary_path = cfg.output_dir / f"summary_{cfg.target}_{cfg.model_name}.json"

    fold_df.to_csv(fold_path, index=False)
    pd.DataFrame({"y_true": y, "y_pred": oof_pred, "group": groups}).to_csv(pred_path, index=False)

    summary = {
        "target": cfg.target,
        "model": cfg.model_name,
        "n_samples": int(len(y)),
        "n_users": int(n_unique_groups),
        "n_features_initial": int(len(feature_cols)),  #初始处理后的特征列数量
        "n_features_final": int(X.shape[1]),  #经过缺失指示器处理后的特征列数量
        "drop_panas": cfg.drop_panas,
        "use_stai2_feature": cfg.use_stai2_feature,
        "max_missing_ratio": cfg.max_missing_ratio,
        "add_missing_indicator": cfg.add_missing_indicator,
        "cv": cv_metrics,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[OK] 训练完成")
    print(f"[OK] fold结果: {fold_path}")
    print(f"[OK] oof预测: {pred_path}")
    print(f"[OK] summary: {summary_path}")
    print("CV Metrics:", cv_metrics)


if __name__ == "__main__":
    main()

