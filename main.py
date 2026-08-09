import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import config
from data.future import FUTURE_VARS
from data.collection import load_dataset
from data.countries import COUNTRY_NAMES, classify_archetype, INCOME_GROUP_LABELS
from src.feature_engineering import build_feature_matrix, get_feature_columns, TARGET_VARS
from src.hdi_calculator import hdi_calc
from src.models.ensemble import HDIPredictionPipeline
from src.forecasting.realistic import forecast_all_countries
from src.evaluation import time_series_cv, summarize_cv_results, walk_forward_backtest
from src.scenarios import ScenarioEngine
from src.explainability import ModelExplainer
from src.visualization import generate_all_plots


def main():
    t0 = time.time()

    print("=" * 60)
    print("  GLOBAL HDI PROJECTIONS TO 2050")
    print("  ML Ensemble Pipeline (RF+Ridge+XGB+LGBM+CatBoost)")
    print("=" * 60)

    print("\n[1/9] Loading dataset (1990-2025)...")
    raw_df = load_dataset()
    print(f"  {len(raw_df)} rows | {raw_df['country_id'].nunique()} countries | "
          f"{raw_df['year'].min()}-{raw_df['year'].max()}")

    print("\n[2/9] Building feature matrix...")
    df = build_feature_matrix(raw_df)
    feature_cols = get_feature_columns(df)
    print(f"  {len(feature_cols)} features engineered")

    print("\n[3/9] Computing official HDI from components...")
    df["hdi"] = hdi_calc.compute_hdi(
        df["life_exp"], df["expected_school"],
        df["mean_school"], df["gni_ppp"],
    )
    df = build_feature_matrix(df)
    feature_cols = get_feature_columns(df)
    print(f"  {len(feature_cols)} features (including HDI-derived)")
    print(f"  HDI range: {df['hdi'].min():.4f} - {df['hdi'].max():.4f}")

    print("\n[4/9] Walk-forward backtest (train 1990-2013, test 2014-2023)...")
    bt = walk_forward_backtest(
        df, feature_cols, TARGET_VARS,
        train_end_year=2013, test_start_year=2014, test_end_year=2023,
    )
    o = bt["overall"]
    print(f"\n  === HELD-OUT TEST RESULTS (2014-2023) ===")
    print(f"  HDI MAE:       {o['overall_mae']:.4f}")
    print(f"  HDI RMSE:      {o['overall_rmse']:.4f}")
    print(f"  HDI R2:        {o['overall_r2']:.4f}")
    print(f"  HDI MAPE:      {o['overall_mape']:.2f}%")
    print(f"  Mean bias:     {o['mean_bias']:+.4f}")
    print(f"  Max abs error: {o['max_abs_error']:.4f}")
    print(f"  Within 0.01:   {o['pct_within_001']:.1f}%")
    print(f"  Within 0.05:   {o['pct_within_005']:.1f}%")
    print(f"  Within 0.10:   {o['pct_within_01']:.1f}%")

    print(f"\n  Component-level metrics:")
    for comp, metrics in bt["component_metrics"].items():
        print(f"    {comp}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")

    print(f"\n  Year-by-year HDI MAE:")
    for yr, m in sorted(bt["by_year"].items()):
        print(f"    {yr}: MAE={m['mae']:.4f} (n={m['n_countries']})")

    print(f"\n  Worst 10 countries (by MAE):")
    for cid, mae in list(bt["worst_10_countries"].items())[:10]:
        name = COUNTRY_NAMES.get(cid, cid)
        print(f"    {cid} ({name}): MAE={mae:.4f}")

    print(f"\n  Best 10 countries (by MAE):")
    for cid, mae in list(bt["best_10_countries"].items())[:10]:
        name = COUNTRY_NAMES.get(cid, cid)
        print(f"    {cid} ({name}): MAE={mae:.4f}")

    print("\n[5/9] Running rolling cross-validation...")
    cv_results = time_series_cv(
        df, feature_cols, TARGET_VARS,
        min_train_years=config.CV_MIN_TRAIN_YEARS, test_window=1,
    )
    cv_summary = summarize_cv_results(cv_results)
    print("\n  Cross-Validation Results:")
    for target, metrics in cv_summary.items():
        print(f"  {target}:")
        for mn, vals in metrics.items():
            print(f"    {mn}: {vals['mean']:.6f} (+/- {vals['std']:.6f})")

    print("\n[6/9] Training final ensemble on full dataset...")
    train_mask = df["year"] <= config.HIST_END
    X_train = df.loc[train_mask, feature_cols]
    y_train = {col: df.loc[train_mask, col] for col in TARGET_VARS.values()}
    pipeline = HDIPredictionPipeline()
    pipeline.fit(X_train, y_train)

    print("\n[7/9] Forecasting independent variables to 2050...")
    forecast_years = np.arange(config.HIST_END + 1, config.FORECAST_END + 1)

    fc_vars = ["life_exp", "expected_school", "mean_school", "gni_ppp",
               "internet", "fertility", "urbanization", "gov_effectiveness",
               "corruption", "trade_openness", "co2_per_capita",
               "renewable_share", "eci", "physicians", "health_exp",
               "population",
               "gini", "infant_mortality", "rule_of_law", "political_stability",
               "rd_expenditure", "dependency_ratio", "broadband", "climate_risk",
               *FUTURE_VARS]

    from data.stability import apply_state_capacity_adjustments
    base_forecasts = forecast_all_countries(df, forecast_years, fc_vars)
    print(f"  {len(base_forecasts)} forecast rows generated")

    print("  Applying state capacity adjustments...")
    base_forecasts = apply_state_capacity_adjustments(base_forecasts)

    print("\n[8/9] Applying scenario engine...")
    scenario_engine = ScenarioEngine()
    all_scenarios = scenario_engine.generate_all_scenarios(base_forecasts)

    print("\n  Predicting HDI for all scenarios (with confidence intervals)...")
    for scenario_name, sdf in all_scenarios.items():
        sdf_fe = build_feature_matrix(sdf)
        missing = set(feature_cols) - set(sdf_fe.columns)
        for col in missing:
            sdf_fe[col] = 0
        X_fore = sdf_fe[feature_cols]
        preds = pipeline.predict(X_fore, with_intervals=True)
        sdf["predicted_hdi"] = hdi_calc.compute_hdi(
            pd.Series(preds["life_expectancy"]),
            pd.Series(preds["expected_years_schooling"]),
            pd.Series(preds["mean_years_schooling"]),
            pd.Series(preds["gni_per_capita_ppp"]),
        ).values
        sdf["hdi_lower"] = hdi_calc.compute_hdi(
            pd.Series(preds["life_expectancy_lower"]),
            pd.Series(preds["expected_years_schooling_lower"]),
            pd.Series(preds["mean_years_schooling_lower"]),
            pd.Series(preds["gni_per_capita_ppp_lower"]),
        ).values
        sdf["hdi_upper"] = hdi_calc.compute_hdi(
            pd.Series(preds["life_expectancy_upper"]),
            pd.Series(preds["expected_years_schooling_upper"]),
            pd.Series(preds["mean_years_schooling_upper"]),
            pd.Series(preds["gni_per_capita_ppp_upper"]),
        ).values
        all_scenarios[scenario_name] = sdf

    for name, sdf in all_scenarios.items():
        yr2050 = sdf[sdf["year"] == config.FORECAST_END]
        mean_hdi = yr2050["predicted_hdi"].mean()
        mean_lo = yr2050["hdi_lower"].mean()
        mean_hi = yr2050["hdi_upper"].mean()
        print(f"    {name}: Mean HDI 2050 = {mean_hdi:.4f} [{mean_lo:.4f} - {mean_hi:.4f}]")

    print("\n[9/9] Running explainability analysis...")
    explainer = ModelExplainer(pipeline, feature_cols)
    sample_idx = np.random.default_rng(42).choice(len(X_train), min(300, len(X_train)), replace=False)
    explainer.compute_shap_values(X_train.iloc[sample_idx], sample_size=300)
    global_summary = explainer.get_global_summary()

    print("\n  Top features per HDI component:")
    for comp_name, feat_df in global_summary.items():
        print(f"\n  {comp_name}:")
        for _, row in feat_df.head(5).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")

    print("\n  Generating visualizations...")
    baseline_df = all_scenarios["baseline"]
    generate_all_plots(
        scenario_results={k: v for k, v in all_scenarios.items()},
        cv_summary=cv_summary,
        importances=pipeline.get_feature_importances(),
        results_df=baseline_df,
    )

    print("\n  Saving results...")
    for name, sdf in all_scenarios.items():
        sdf.to_parquet(config.OUTPUT_DIR / f"projections_{name}.parquet", index=False)

    cv_df = pd.DataFrame([
        {
            "train_end": r["train_end"],
            "test_year": r["test_start"],
            **{f"{t}_{m}": r["metrics"][t][m]
               for t in r["metrics"]
               for m in ["mae", "rmse", "r2"]},
        }
        for r in cv_results
    ])
    cv_df.to_csv(config.OUTPUT_DIR / "cv_results.csv", index=False)

    bt["predictions_df"].to_csv(config.OUTPUT_DIR / "backtest_predictions.csv", index=False)

    bt_summary = pd.DataFrame([bt["overall"]])
    bt_summary.to_csv(config.OUTPUT_DIR / "backtest_summary.csv", index=False)

    bt_years = pd.DataFrame([
        {"year": yr, **m} for yr, m in sorted(bt["by_year"].items())
    ])
    bt_years.to_csv(config.OUTPUT_DIR / "backtest_by_year.csv", index=False)

    bt_countries = pd.DataFrame([
        {"country_id": cid, **m} for cid, m in sorted(bt["by_country"].items())
    ])
    bt_countries.to_csv(config.OUTPUT_DIR / "backtest_by_country.csv", index=False)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  COMPLETE ({elapsed:.1f}s)")
    print(f"  Output: {config.OUTPUT_DIR}")
    print(f"  Files: projections_*.parquet, backtest_*.csv, cv_results.csv,")
    print(f"         scenario_trajectories.png, cv_results.png,")
    print(f"         feature_importance.png, hdi_heatmap.png")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
