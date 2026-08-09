import numpy as np
import pandas as pd

from data.future import FUTURE_VARS


CORE_VARS = [
    "life_exp", "expected_school", "mean_school", "gni_ppp",
    "internet", "fertility", "urbanization", "gov_effectiveness",
    "corruption", "trade_openness", "co2_per_capita", "renewable_share",
    "eci", "physicians", "health_exp", "population",
    "gini", "infant_mortality", "rule_of_law", "political_stability",
    "rd_expenditure", "dependency_ratio", "broadband", "climate_risk",
] + FUTURE_VARS

TARGET_VARS = {
    "life_expectancy": "life_exp",
    "expected_years_schooling": "expected_school",
    "mean_years_schooling": "mean_school",
    "gni_per_capita_ppp": "gni_ppp",
}

TARGET_COMPONENTS = set(TARGET_VARS.values())

LEAKAGE_PRONE_FEATURES = {
    "hdi",
    "hdi_lag1",
    "hdi_lag3",
    "hdi_lag5",
    "hdi_diff1",
    "hdi_diff5",
    "hdi_mean5",
    "life_exp_diff1",
    "life_exp_diff5",
    "life_exp_mean5",
    "expected_school_diff1",
    "expected_school_diff5",
    "expected_school_mean5",
    "mean_school_diff1",
    "mean_school_diff5",
    "mean_school_mean5",
    "gni_ppp_diff1",
    "gni_ppp_diff5",
    "gni_ppp_mean5",
    "life_exp_gap_frontier",
    "education_gap_frontier",
    "income_gap_frontier",
    "human_capital_index",
    "gni_per_life_year",
    "gni_per_school_year",
    "health_system_quality",
    "inequality_adjusted_income",
    # Broad composites that are mostly re-statements of income, education,
    # digital access, or governance. Keeping these alongside their raw inputs
    # makes tree ensembles double-count correlated development level.
    "innovation_index",
    "digitalization",
    "digital_governance",
    "institutional_quality",
    "future_readiness_index",
    "ai_cloud_stack",
    "advanced_manufacturing_stack",
    "green_mobility_stack",
    "innovation_finance_stack",
}

STRUCTURAL_BREAK_COUNTRIES = {
    "UKR": 2014, "SYR": 2011, "YEM": 2014, "AFG": 2021,
    "SSD": 2013, "SDN": 2023, "MMR": 2021, "ETH": 2020,
    "LBY": 2014, "VEN": 2014, "LBN": 2019, "HTI": 2010,
    "IRQ": 2014, "SOM": 2006, "NGA": 2015, "MML": 2021,
}

LAG_VARS = [
    "life_exp", "expected_school", "mean_school", "gni_ppp",
    "internet", "gov_effectiveness", "corruption",
    "infant_mortality", "rule_of_law", "rd_expenditure",
    "ai_adoption_index", "robot_density", "green_energy_investment",
    "startup_ecosystem", "cloud_adoption",
]


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["country_id", "year"]).copy()
    df["years_since_1990"] = df["year"] - 1990

    existing_lag_cols = {c for c in df.columns if "_lag1" in c}
    vars_to_compute = [v for v in LAG_VARS if f"{v}_lag1" not in existing_lag_cols and v in df.columns]

    for var in vars_to_compute:
        grp = df.groupby("country_id")[var]
        df[f"{var}_lag1"] = grp.shift(1)
        df[f"{var}_lag5"] = grp.shift(5)
        df[f"{var}_diff1"] = df[var] - df[f"{var}_lag1"]
        df[f"{var}_diff5"] = df[var] - df[f"{var}_lag5"]
        df[f"{var}_mean5"] = grp.transform(lambda x: x.rolling(5, min_periods=1).mean())

    if "archetype" in df.columns:
        regional = df.groupby(["archetype", "year"])[
            ["life_exp", "mean_school", "gni_ppp", "internet"]
        ].transform("mean")
        for col in regional.columns:
            df[f"{col}_regional_avg"] = regional[col]

    df["life_exp_gap_frontier"] = df.groupby("year")["life_exp"].transform(lambda x: x.max() - x)
    df["education_gap_frontier"] = df.groupby("year")["mean_school"].transform(lambda x: x.max() - x)
    df["income_gap_frontier"] = df.groupby("year")["gni_ppp"].transform(lambda x: x.max() - x)

    df["human_capital_index"] = (
        0.5 * (df["life_exp"] / 85) +
        0.25 * (df["mean_school"] / 15) +
        0.25 * (df["expected_school"] / 18)
    )
    df["governance_index"] = (df["gov_effectiveness"] + df["corruption"]) / 2
    df["digitalization"] = df["internet"] * df["urbanization"]

    if "gni_ppp" in df.columns and "life_exp" in df.columns:
        df["gni_per_life_year"] = df["gni_ppp"] / df["life_exp"].clip(lower=30)
    if "gni_ppp" in df.columns and "mean_school" in df.columns:
        df["gni_per_school_year"] = df["gni_ppp"] / df["mean_school"].clip(lower=1)
    if "internet" in df.columns and "gov_effectiveness" in df.columns:
        df["digital_governance"] = df["internet"] * (df["gov_effectiveness"] + 2) / 4

    if "rule_of_law" in df.columns and "political_stability" in df.columns:
        df["institutional_quality"] = (df["rule_of_law"] + df["political_stability"] + df["gov_effectiveness"]) / 3
    if "infant_mortality" in df.columns and "life_exp" in df.columns:
        df["health_system_quality"] = df["life_exp"] / (df["infant_mortality"].clip(lower=1) + 1)
    if "gini" in df.columns and "gni_ppp" in df.columns:
        df["inequality_adjusted_income"] = df["gni_ppp"] * (1 - df["gini"])
    if "rd_expenditure" in df.columns and "broadband" in df.columns:
        df["innovation_index"] = (df["rd_expenditure"] / 5.0 + df["broadband"] / 50.0) / 2
    if "dependency_ratio" in df.columns and "fertility" in df.columns:
        df["demographic_pressure"] = df["dependency_ratio"] * df["fertility"] / 3.0
    if all(col in df.columns for col in FUTURE_VARS):
        df["future_readiness_index"] = df[FUTURE_VARS].mean(axis=1)
        df["ai_cloud_stack"] = df["ai_adoption_index"] * df["cloud_adoption"]
        df["advanced_manufacturing_stack"] = (
            df["robot_density"] + df["semiconductor_production"] + df["battery_capacity"]
        ) / 3.0
        df["green_mobility_stack"] = (
            df["green_energy_investment"] + df["ev_adoption"] + df["high_speed_rail"]
        ) / 3.0
        df["innovation_finance_stack"] = (
            df["startup_ecosystem"] + df["venture_capital"] + df["rd_expenditure"].clip(0, 5) / 5.0
        ) / 3.0

    if "conflict_flag" not in df.columns:
        df["conflict_flag"] = 0
        df["post_conflict_flag"] = 0
        for cid, break_year in STRUCTURAL_BREAK_COUNTRIES.items():
            mask = df["country_id"] == cid
            if mask.any():
                df.loc[mask, "conflict_flag"] = (df.loc[mask, "year"] >= break_year).astype(int)
                df.loc[mask, "post_conflict_flag"] = (
                    (df.loc[mask, "year"] >= break_year + 3) &
                    (df.loc[mask, "year"] <= break_year + 10)
                ).astype(int)

    if "hdi" in df.columns and "hdi_lag1" not in df.columns:
        hdi_grp = df.groupby("country_id")["hdi"]
        df["hdi_lag1"] = hdi_grp.shift(1)
        df["hdi_lag3"] = hdi_grp.shift(3)
        df["hdi_lag5"] = hdi_grp.shift(5)
        df["hdi_diff1"] = df["hdi"] - df["hdi_lag1"]
        df["hdi_diff5"] = df["hdi"] - df["hdi_lag5"]
        df["hdi_mean5"] = hdi_grp.transform(lambda x: x.rolling(5, min_periods=1).mean())

    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    return df


def is_leakage_prone_feature(column: str) -> bool:
    """Return True for features that expose same-year HDI target information."""
    if column in LEAKAGE_PRONE_FEATURES:
        return True
    if column.startswith("hdi_"):
        return True
    return False


def get_feature_columns(df: pd.DataFrame, allow_target_leakage: bool = False) -> list:
    exclude = {"country_id", "archetype", "year", "iso3", "country_name"}
    exclude.update(TARGET_VARS.values())
    if not allow_target_leakage:
        exclude.update(c for c in df.columns if is_leakage_prone_feature(c))
    return [c for c in df.columns if c not in exclude and df[c].dtype in ("float64", "float32", "int64")]
