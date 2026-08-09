import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import config
from src.models.ensemble import HDIPredictionPipeline
from src.hdi_calculator import hdi_calc


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 0
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(mape(y_true, y_pred)),
        "max_error": float(np.max(np.abs(y_true - y_pred))),
        "median_error": float(np.median(np.abs(y_true - y_pred))),
    }


def time_series_cv(
    df: pd.DataFrame,
    feature_cols: list,
    target_cols: dict,
    min_train_years: int = 15,
    test_window: int = 1,
) -> list:
    years = sorted(df["year"].unique())
    results = []

    for split_idx in range(min_train_years, len(years) - test_window + 1):
        train_end = years[split_idx - 1]
        test_start = years[split_idx]
        test_end = years[min(split_idx + test_window - 1, len(years) - 1)]

        train_mask = (df["year"] <= train_end)
        test_mask = (df["year"] >= test_start) & (df["year"] <= test_end)

        X_train = df.loc[train_mask, feature_cols]
        X_test = df.loc[test_mask, feature_cols]
        y_train = {target_cols[col]: df.loc[train_mask, target_cols[col]] for col in target_cols}
        y_test = {target_cols[col]: df.loc[test_mask, target_cols[col]] for col in target_cols}

        pipeline = HDIPredictionPipeline(lightweight=True)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        fold_metrics = {}
        for comp_name, target_col in target_cols.items():
            metrics = evaluate_predictions(y_test[target_col].values, preds[comp_name])
            fold_metrics[comp_name] = metrics

        actual_hdi = hdi_calc.compute_hdi(
            y_test["life_exp"],
            y_test["expected_school"],
            y_test["mean_school"],
            y_test["gni_ppp"],
        )
        pred_hdi = hdi_calc.compute_hdi(
            pd.Series(preds["life_expectancy"]),
            pd.Series(preds["expected_years_schooling"]),
            pd.Series(preds["mean_years_schooling"]),
            pd.Series(preds["gni_per_capita_ppp"]),
        )
        fold_metrics["hdi"] = evaluate_predictions(actual_hdi.values, pred_hdi.values)

        results.append({
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "n_train": train_mask.sum(),
            "n_test": test_mask.sum(),
            "metrics": fold_metrics,
            "pipeline": pipeline,
        })

    return results


def summarize_cv_results(results: list) -> dict:
    summary = {}
    metric_names = ["mae", "rmse", "r2", "mape"]

    all_targets = list(results[0]["metrics"].keys())
    for target in all_targets:
        summary[target] = {}
        for mn in metric_names:
            vals = [r["metrics"][target][mn] for r in results if target in r["metrics"]]
            summary[target][mn] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
    return summary


def walk_forward_backtest(
    df: pd.DataFrame,
    feature_cols: list,
    target_cols: dict,
    train_end_year: int = 2013,
    test_start_year: int = 2014,
    test_end_year: int = 2023,
) -> dict:
    """Train on data up to train_end_year, predict test_start_year to test_end_year.

    Returns per-year metrics, per-country metrics, and aggregate summary.
    """
    train_mask = df["year"] <= train_end_year
    test_mask = (df["year"] >= test_start_year) & (df["year"] <= test_end_year)

    X_train = df.loc[train_mask, feature_cols]
    X_test = df.loc[test_mask, feature_cols]
    y_train = {target_cols[col]: df.loc[train_mask, target_cols[col]] for col in target_cols}
    y_test_df = df.loc[test_mask].copy()

    pipeline = HDIPredictionPipeline(lightweight=True)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    actual_hdi = hdi_calc.compute_hdi(
        y_test_df["life_exp"], y_test_df["expected_school"],
        y_test_df["mean_school"], y_test_df["gni_ppp"],
    )
    pred_hdi = hdi_calc.compute_hdi(
        pd.Series(preds["life_expectancy"], index=y_test_df.index),
        pd.Series(preds["expected_years_schooling"], index=y_test_df.index),
        pd.Series(preds["mean_years_schooling"], index=y_test_df.index),
        pd.Series(preds["gni_per_capita_ppp"], index=y_test_df.index),
    )

    y_test_df["pred_hdi"] = pred_hdi.values
    y_test_df["actual_hdi"] = actual_hdi.values
    y_test_df["hdi_error"] = y_test_df["pred_hdi"] - y_test_df["actual_hdi"]
    y_test_df["abs_error"] = y_test_df["hdi_error"].abs()

    by_year = {}
    for year in range(test_start_year, test_end_year + 1):
        yr_mask = y_test_df["year"] == year
        yr_df = y_test_df[yr_mask]
        by_year[int(year)] = {
            "mae": float(yr_df["abs_error"].mean()),
            "rmse": float(np.sqrt((yr_df["hdi_error"] ** 2).mean())),
            "mean_error": float(yr_df["hdi_error"].mean()),
            "n_countries": int(yr_mask.sum()),
        }

    by_country = {}
    for cid in y_test_df["country_id"].unique():
        cdf = y_test_df[y_test_df["country_id"] == cid]
        if len(cdf) > 0:
            by_country[cid] = {
                "mae": float(cdf["abs_error"].mean()),
                "rmse": float(np.sqrt((cdf["hdi_error"] ** 2).mean())),
                "mean_error": float(cdf["hdi_error"].mean()),
                "n_years": int(len(cdf)),
            }

    overall = {
        "train_period": f"1990-{train_end_year}",
        "test_period": f"{test_start_year}-{test_end_year}",
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "overall_mae": float(y_test_df["abs_error"].mean()),
        "overall_rmse": float(np.sqrt((y_test_df["hdi_error"] ** 2).mean())),
        "overall_r2": float(r2_score(y_test_df["actual_hdi"], y_test_df["pred_hdi"])),
        "overall_mape": float(mape(y_test_df["actual_hdi"].values, y_test_df["pred_hdi"].values)),
        "mean_bias": float(y_test_df["hdi_error"].mean()),
        "max_abs_error": float(y_test_df["abs_error"].max()),
        "pct_within_001": float((y_test_df["abs_error"] < 0.01).mean() * 100),
        "pct_within_005": float((y_test_df["abs_error"] < 0.05).mean() * 100),
        "pct_within_01": float((y_test_df["abs_error"] < 0.1).mean() * 100),
    }

    worst = y_test_df.groupby("country_id")["abs_error"].mean().sort_values(ascending=False).head(10)
    best = y_test_df.groupby("country_id")["abs_error"].mean().sort_values(ascending=True).head(10)

    component_metrics = {}
    comp_map = {
        "life_exp": "life_expectancy",
        "expected_school": "expected_years_schooling",
        "mean_school": "mean_years_schooling",
        "gni_ppp": "gni_per_capita_ppp",
    }
    for target_col, comp_name in comp_map.items():
        actual = y_test_df[target_col].values
        predicted = preds[comp_name]
        component_metrics[comp_name] = evaluate_predictions(actual, predicted)

    return {
        "overall": overall,
        "by_year": by_year,
        "by_country": by_country,
        "component_metrics": component_metrics,
        "worst_10_countries": worst.to_dict(),
        "best_10_countries": best.to_dict(),
        "pipeline": pipeline,
        "predictions_df": y_test_df,
    }
