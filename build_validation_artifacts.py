"""Build reproducible benchmark and residual-calibration tables for validation."""

from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path("data/output/backtest_predictions.csv")
OUTPUT_DIR = Path("data/output")


def metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    valid = actual.notna() & predicted.notna()
    errors = predicted[valid] - actual[valid]
    return {
        "Observations": int(valid.sum()),
        "MAE": float(errors.abs().mean()),
        "RMSE": float(np.sqrt(np.mean(np.square(errors)))),
        "Mean_Error": float(errors.mean()),
    }


def main() -> None:
    data = pd.read_csv(INPUT)
    actual = data["actual_hdi"]
    methods = [
        (
            "Component ensemble",
            "held_out_component_reconstruction",
            "same_year_non_target_covariates",
            data["pred_hdi"],
        ),
        (
            "Previous-year HDI",
            "lag_only_forecast_baseline",
            "information_available_at_t_minus_1",
            data["hdi_lag1"],
        ),
        (
            "Lagged five-year trend",
            "lag_only_forecast_baseline",
            "information_available_at_t_minus_1_and_t_minus_5",
            (data["hdi_lag1"] + (data["hdi_lag1"] - data["hdi_lag5"]) / 4).clip(0.1, 1.0),
        ),
    ]
    benchmark_rows = []
    for method, test_type, timing, predicted in methods:
        benchmark_rows.append({
            "Method": method,
            "Test_Type": test_type,
            "Information_Timing": timing,
            **metrics(actual, predicted),
        })
    pd.DataFrame(benchmark_rows).to_csv(
        OUTPUT_DIR / "backtest_benchmarks.csv", index=False)

    residuals = data["pred_hdi"] - actual
    low = float(residuals.quantile(0.05))
    high = float(residuals.quantile(0.95))
    calibration_rows = [{
        "Scope": "pooled_2014_2023",
        "Year": "all",
        "Residual_P05": low,
        "Residual_P95": high,
        "Coverage": float(residuals.between(low, high).mean()),
        "Calibration_Status": "same_sample_descriptive_not_independent",
    }]
    for year, group in data.groupby("year"):
        year_residuals = group["pred_hdi"] - group["actual_hdi"]
        calibration_rows.append({
            "Scope": "held_out_year",
            "Year": int(year),
            "Residual_P05": low,
            "Residual_P95": high,
            "Coverage": float(year_residuals.between(low, high).mean()),
            "Calibration_Status": "pooled_band_applied_by_year",
        })
    pd.DataFrame(calibration_rows).to_csv(
        OUTPUT_DIR / "backtest_residual_calibration.csv", index=False)

    print(f"Wrote {len(benchmark_rows)} benchmark rows and {len(calibration_rows)} calibration rows.")


if __name__ == "__main__":
    main()
