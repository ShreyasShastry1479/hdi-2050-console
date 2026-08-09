from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Config:
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    DATA_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "data")
    OUTPUT_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "data" / "output")

    HIST_START: int = 1990
    HIST_END: int = 2024
    FORECAST_END: int = 2050

    RANDOM_STATE: int = 42
    N_COUNTRIES: int = 195
    CV_TEST_WINDOW: int = 1  # rolling window in years
    CV_MIN_TRAIN_YEARS: int = 25  # skip early folds to speed up

    BACKTEST_TRAIN_END: int = 2013
    BACKTEST_START: int = 2014
    BACKTEST_END: int = 2023

    ENSEMBLE_MODELS: list = field(default_factory=lambda: [
        "random_forest", "ridge", "xgboost", "lightgbm", "catboost"
    ])

    SCENARIOS: list = field(default_factory=lambda: [
        "baseline", "high_growth", "low_growth", "green_transition"
    ])

    def __post_init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


config = Config()
