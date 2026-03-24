from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    data_path: Path = Path("/home/wqshen/mm-test/code/processed/mmash_user_level_features.csv")
    output_dir: Path = Path("/home/wqshen/mm-test/code/outputs")
    target: str = "label_daily_stress"
    model_name: str = "ridge"  # ridge | rf
    n_splits: int = 5
    random_state: int = 42
    drop_panas: bool = True
    use_stai2_feature: bool = False
    max_missing_ratio: float = 0.40  # 缺失率高于该阈值的特征将被删除
    add_missing_indicator: bool = True  # 是否为缺失特征添加0/1指示器，就是额外添加一个missing 0/1列，用来指示哪一列的哪个参数是补值的
