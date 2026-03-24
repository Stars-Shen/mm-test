from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelFactory:
    model_name: str = "ridge"  # ridge | rf
    random_state: int = 42

    def build(self):
        if self.model_name == "rf":
            return RandomForestRegressor(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=2,
                random_state=self.random_state,
            )

        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),   #使用中位数插补器，防止user数据中出现缺失值，某种程度上会引入偏差
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=1.0, random_state=self.random_state)),
            ]
        )
